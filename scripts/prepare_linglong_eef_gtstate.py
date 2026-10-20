"""Add recorded /end_pos measured EEF state to existing action-EEF buckets.

No kinematics are used. Preserve every original column, especially action_eef.
Use the original driver timestamp, with the original camera-aligned /end_pos
record only when the exact driver timestamp is absent. Never use *_exp poses.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


STATE_KEY = "observation.state_eef"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def recorded_state(source, table, parser):
    csv_path = Path(source["source_csv"])
    if sha256(csv_path) != source["source_sha256"]:
        raise ValueError(f"source CSV changed: {csv_path}")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    alignment = [json.loads(x) for x in (csv_path.parent / "alignment.jsonl").read_text().splitlines()]
    assert len(rows) == len(alignment)
    ts = np.array([int(r["timestamp_ns"]) for r in rows], dtype=np.int64)
    assert np.all(np.diff(ts) > 0)
    fps = source["fps"]
    grid = ts[0] + np.rint(np.arange(int((ts[-1] - ts[0]) * fps / 1e9) + 1) * 1e9 / fps).astype(np.int64)
    right = np.minimum(np.searchsorted(ts, grid), len(ts) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(abs(ts[left] - grid) <= abs(ts[right] - grid), left, right)
    joint = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
    q_csv = np.array([[float(rows[i][f"q_{j}"]) for j in range(24)] for i in selected], dtype=np.float32)
    np.testing.assert_array_equal(joint, q_csv)
    assert len(table) == source["length"]
    driver_ts = np.array([alignment[i]["source_timestamps"]["driver"] for i in selected], dtype=np.int64)
    camera_gt_ts = np.array([alignment[i]["source_timestamps"]["end_pos"] for i in selected], dtype=np.int64)
    bag = csv_path.parents[2] / "raw" / source["source_episode"] / "bag/bag_0.db3"
    if not bag.is_file():
        # 0923 held-out bags live in the earlier sample extraction. Resolve
        # from the decoder's inventory, not by guessing an episode name.
        inventory = json.loads((csv_path.parents[2] / "inventory.json").read_text())
        entry = next(r for r in inventory if str(r["episode"]) == source["source_episode"])
        bag = Path(entry["archive_member"]).parent / "bag/bag_0.db3"
        if not bag.is_file() or sha256(bag) != entry["db_sha256"]:
            raise ValueError(f"original inventory bag unavailable or changed: {bag}")
    with sqlite3.connect(bag.as_uri() + "?mode=ro", uri=True) as con:
        topic_id = con.execute("SELECT id FROM topics WHERE name='/end_pos'").fetchone()[0]
        poses = {}
        for (payload,) in con.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)):
            pose = parser.decode_end_pos(payload)
            stamp = pose["timestamp_ns"]
            if stamp in poses and poses[stamp] != pose:
                raise ValueError(f"conflicting /end_pos timestamps: {bag}")
            poses[stamp] = pose
    exact = np.array([int(t) in poses for t in driver_ts])
    used_ts = np.where(exact, driver_ts, camera_gt_ts)
    if not all(int(t) in poses for t in used_ts):
        raise ValueError(f"missing recorded EEF state: {bag}")
    if np.max(abs(used_ts - driver_ts)) > 80_000_000 or np.max(abs(used_ts - grid)) > 80_000_000:
        raise ValueError(f"EEF timestamp difference exceeds 80 ms: {bag}")
    state = np.empty((len(table), 16), dtype=np.float32)
    for offset, field in ((0, "ee_pose_l"), (8, "ee_pose_r")):
        raw = np.array([poses[int(t)][field] for t in used_ts], dtype=np.float64)
        quat = Rotation.from_euler("xyz", raw[:, 3:6], degrees=False).as_quat()[:, [3, 0, 1, 2]]
        # Start each episode with w >= 0, then choose the nearest quaternion sign.
        if quat[0, 0] < 0:
            quat[0] *= -1
        for i in range(1, len(quat)):
            if np.dot(quat[i - 1], quat[i]) < 0:
                quat[i] *= -1
        state[:, offset:offset + 3] = raw[:, :3]
        state[:, offset + 3:offset + 7] = quat
        state[:, offset + 7] = joint[:, offset + 7]
        np.testing.assert_array_equal(state[:, offset:offset + 3], raw[:, :3].astype(np.float32))
        np.testing.assert_allclose(np.linalg.norm(state[:, offset + 3:offset + 7], axis=1), 1, atol=1e-7)
        recovered = Rotation.from_quat(state[:, offset + 3:offset + 7][:, [1, 2, 3, 0]])
        np.testing.assert_allclose(recovered.as_matrix(), Rotation.from_euler("xyz", raw[:, 3:6]).as_matrix(), atol=2e-7)
    assert np.isfinite(state).all()
    np.testing.assert_array_equal(state[:, [7, 15]], joint[:, [7, 15]])
    timestamps = {"raw_bag": str(bag), "camera_timestamp_ns": ts[selected].tolist(), "driver_timestamp_ns": driver_ts.tolist(),
                  "eef_timestamp_ns": used_ts.tolist(), "grid_timestamp_ns": grid.tolist(),
                  "camera_aligned_fallback_frames": np.flatnonzero(~exact).tolist()}
    return state, timestamps


def convert(source, target, train_target, parser, parser_path, resume_empty=False):
    source, target = Path(source).resolve(), Path(target).absolute()
    existing_report = target / "eef_gtstate_conversion.json"
    resume = (resume_empty and existing_report.is_file()
              and json.loads(existing_report.read_text()) == {"status": "incomplete", "source": str(source)}
              and not list((target / "data").rglob("*.parquet")))
    if (target.exists() and not resume) or target.is_symlink() or source in target.parents:
        raise ValueError(f"choose a fresh output outside source: {target}")
    manifest = json.loads((source / "manifest.json").read_text())
    split = manifest["split"]
    assert split in ("train", "test") and manifest["action_eef"]["shape"] == [16]
    mapping = json.loads((source / "source_mapping.json").read_text())
    paths = sorted((source / "data").rglob("*.parquet"))
    assert len(paths) == len(mapping) == manifest["total_episodes"]
    by_episode = {r["episode_index"]: r for r in mapping}
    if not resume:
        shutil.copytree(source, target, symlinks=True,
                        ignore=lambda directory, names: ["data"] if Path(directory) == source else [])
        for link in source.rglob("*"):
            if link.is_symlink():
                copied, destination = target / link.relative_to(source), link.resolve(strict=True)
                copied.unlink()
                copied.symlink_to(os.path.relpath(destination, copied.parent), target_is_directory=destination.is_dir())
    write_json(target / "eef_gtstate_conversion.json", {"status": "incomplete", "source": str(source)})
    records, arrays, image_refs = [], [], set()
    for number, path in enumerate(paths):
        original = pq.read_table(path)
        assert STATE_KEY not in original.column_names
        index = int(path.stem.split("_")[-1])
        digest = sha256(path)
        state, times = recorded_state(by_episode[index], original, parser)
        converted = original.append_column(STATE_KEY, pa.FixedSizeListArray.from_arrays(pa.array(state.ravel()), 16))
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            pq.write_table(converted, stream, compression="zstd")
        saved = pq.read_table(destination)
        assert saved.select(original.column_names).equals(original, check_metadata=False)
        np.testing.assert_array_equal(np.array(saved[STATE_KEY].to_pylist(), dtype=np.float32), state)
        assert sha256(path) == digest
        for camera in ("head", "left", "right"):
            image_refs.update(original[f"observation.images.{camera}"].to_pylist())
        arrays.append(state)
        records.append({"parquet": relative.as_posix(), "source_episode": by_episode[index]["source_episode"],
                        "frames": len(state), "source_sha256": digest, "output_sha256": sha256(destination), **times})
        if (number + 1) % 20 == 0 or number + 1 == len(paths):
            print(f"{split}: verified {number + 1}/{len(paths)} episodes", flush=True)
    for ref in image_refs:
        if not (target / ref).is_file():
            raise FileNotFoundError(target / ref)
    values = np.concatenate(arrays).astype(np.float64)
    windows = sum(max(len(x) - 32, 0) for x in arrays)
    assert len(values) == manifest["total_frames"] and windows == manifest["total_windows"]
    if split == "train":
        norm = {"global_mean": values.mean(0).tolist(), "global_std": values.std(0, ddof=1).tolist(),
                "global_min": values.min(0).tolist(), "global_max": values.max(0).tolist(),
                "global_q01": np.quantile(values, .01, axis=0).tolist(),
                "global_q99": np.quantile(values, .99, axis=0).tolist()}
        write_json(target / "meta/state_eef_stats.json", {
            "source_column": STATE_KEY, "statistics_source": "train frames once, no window weighting",
            "num_episodes": len(paths), "num_transition": len(values), "std_ddof": 1,
            "state": {"default": norm}})
    else:
        # Both state and action transforms for held-out evaluation come from train.
        for name in ("state_eef_stats.json", "action_eef_stats.json"):
            shutil.copyfile(train_target / "meta" / name, target / "meta" / name)
    contract = {"column": STATE_KEY, "shape": [16], "dtype": "float32",
                "source": "recorded /end_pos.ee_pose_l and ee_pose_r (actual, not expected)",
                "reference": "base_link", "position_unit": "metre", "quaternion_order": "wxyz",
                "source_orientation": "Euler xyz radians", "kinematics_used": False,
                "layout": "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper",
                "gripper_source": "measured observation.state indices 7,15",
                "alignment": "exact driver timestamp; original camera-aligned end_pos if exact absent",
                "quaternion_sign": "continuous per hand per episode; initial w>=0",
                "normalization_stats": "meta/state_eef_stats.json", "normalization_split": "train"}
    info = json.loads((target / "meta/info.json").read_text())
    info["features"][STATE_KEY] = {"dtype": "float32", "shape": [16]}
    write_json(target / "meta/info.json", info)
    manifest.update(source_root=str(target.resolve()), eef_gtstate_source_bucket=str(source),
                    observation_state_eef=contract, normalization_dataset_dir=str(train_target.resolve()))
    write_json(target / "manifest.json", manifest)
    report = {"status": "complete", "source": str(source), "output": str(target.resolve()),
              "split": split, "episodes": len(paths), "frames": len(values), "windows": windows,
              "kinematics_used": False, "original_columns_preserved": True, "source_files_unchanged": True,
              "parser_sha256": sha256(parser_path), "image_references_checked": len(image_refs),
              "camera_aligned_fallback_frames": sum(len(r["camera_aligned_fallback_frames"]) for r in records),
              "max_eef_driver_delta_ms": max(max(abs(a-b) for a,b in zip(r["eef_timestamp_ns"], r["driver_timestamp_ns"])) for r in records) / 1e6,
              "observation_state_eef": contract, "files": records}
    write_json(target / "eef_gtstate_conversion.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=Path("/iag_ad_01/ad/xiejinxia/code/test/video-prediction-policy-2/data"))
    ap.add_argument("--output-root", type=Path, default=Path("data"))
    ap.add_argument("--parser", default="/iag_ad_01/ad/dushuyue/linglong_rawdata_spec/linglong_parser.py")
    ap.add_argument("--split", choices=("all", "train", "test"), default="all")
    ap.add_argument("--resume-empty-incomplete", action="store_true", help="only resume a matching failed bucket with no parquet files")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("linglong_raw_parser", args.parser)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    train = args.output_root / "linglong_0923_1010_eef_gtstate_train_15hz"
    test = args.output_root / "linglong_0923_1010_eef_gtstate_test_15hz"
    selected = [(s, p) for s, p in (("train", train), ("test", test)) if args.split in ("all", s)]
    if not args.resume_empty_incomplete and any(p.exists() or p.is_symlink() for _, p in selected):
        ap.error("both output buckets must be fresh")
    source_mappings = [json.loads((args.source_root / f"linglong_0923_1010_eef_{s}_15hz/source_mapping.json").read_text()) for s in ("train", "test")]
    identities = [{(r["collection"], r["source_episode"]) for r in rows} for rows in source_mappings]
    assert not identities[0] & identities[1], "train/test episode overlap"
    for split, target in selected:
        convert(args.source_root / f"linglong_0923_1010_eef_{split}_15hz", target, train, parser, args.parser,
                resume_empty=args.resume_empty_incomplete)


if __name__ == "__main__":
    main()
