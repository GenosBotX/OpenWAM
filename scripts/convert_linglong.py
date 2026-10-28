"""Convert raw Linglong zip/folder recordings into portable OpenWAM buckets.

One entry point decodes ROS2 SQLite/CDR, aligns to 15 Hz, generates joint and
EEF labels, copies sampled JPEGs, and writes train-only normalization. All
robot-specific code and the default FK model live in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

if __package__:
    from .linglong_eef import DEFAULT_MODEL, LinglongKinematics, make_quaternion_continuous
    from .linglong_raw import iter_bags, read_bag
else:
    from linglong_eef import DEFAULT_MODEL, LinglongKinematics, make_quaternion_continuous
    from linglong_raw import iter_bags, read_bag


FPS = 15
STATE_DIM = ACTION_DIM = 24
CAMERAS = ("head", "left", "right")
VECTOR_COLUMNS = {"observation.state": 24, "action": 24, "action_eef": 16, "observation.state_eef": 16}
NUM_FRAMES = 33


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_list(values, width):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"expected finite (*,{width}) array, got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


def _nearest(timestamps, targets):
    right = np.minimum(np.searchsorted(timestamps, targets), len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(timestamps[left] - targets) <= abs(timestamps[right] - targets), left, right)


def sample_episode(streams, fk, head_grid_tolerance_ns=20_000_000):
    """Use the historical head-master alignment, then nearest 15Hz grid.

    State EEF matches the selected driver's exact /end_pos timestamp when
    available, falling back to the camera-aligned /end_pos sample only.
    Raw q and q_exp, including both grippers, are preserved without heuristics.
    Invalid/gapped sources fail as a whole; never join across dropped frames.
    """
    times = {key: np.array([r["timestamp_ns"] for r in rows], dtype=np.int64)
             for key, rows in streams.items()}
    head = times["head"]
    if len(head) < 2 or np.any(np.diff(head) > 50_000_000):
        raise ValueError("head stream has fewer than two frames or a gap >50ms")
    aligned = {key: _nearest(ts, head) for key, ts in times.items()}
    for key, indices in aligned.items():
        if np.any(abs(times[key][indices] - head) > 80_000_000):
            raise ValueError(f"{key}: original head alignment exceeds 80ms")
    grid = head[0] + np.rint(np.arange(int((head[-1] - head[0]) * FPS / 1e9) + 1) * 1e9 / FPS).astype(np.int64)
    selected_head = _nearest(head, grid)
    if np.any(abs(head[selected_head] - grid) > head_grid_tolerance_ns):
        limit_ms = head_grid_tolerance_ns / 1_000_000
        raise ValueError(f"15Hz head alignment exceeds {limit_ms:g}ms")
    if len(grid) < NUM_FRAMES:
        raise ValueError(f"episode has {len(grid)} frames; OpenWAM requires at least {NUM_FRAMES}")
    selected = {key: indices[selected_head] for key, indices in aligned.items()}
    for key, indices in selected.items():
        if np.any(abs(times[key][indices] - grid) > 80_000_000):
            raise ValueError(f"{key}: 15Hz sensor alignment exceeds 80ms")
    q_raw = np.asarray([r["q"] for r in streams["driver"]], dtype=np.float32)
    a_raw = np.asarray([r["q_exp"] for r in streams["driver"]], dtype=np.float32)
    if not np.isfinite(q_raw).all() or not np.isfinite(a_raw).all():
        raise ValueError("non-finite raw joint state/action")
    joint, action = q_raw[selected["driver"]], a_raw[selected["driver"]]
    driver_ts = times["driver"][selected["driver"]]
    end_ts = times["end_pos"]
    by_stamp = {int(t): i for i, t in enumerate(end_ts)}
    exact = np.array([int(t) in by_stamp for t in driver_ts])
    end_indices = np.array([by_stamp.get(int(t), int(fallback))
                            for t, fallback in zip(driver_ts, selected["end_pos"])])
    used_ts = end_ts[end_indices]
    if np.any(abs(used_ts - driver_ts) > 80_000_000) or np.any(abs(used_ts - grid) > 80_000_000):
        raise ValueError("recorded EEF alignment exceeds 80ms")
    state_eef = np.empty((len(grid), 16), dtype=np.float32)
    for offset, field in ((0, "ee_pose_l"), (8, "ee_pose_r")):
        raw = np.array([streams["end_pos"][i][field] for i in end_indices], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("non-finite recorded EEF state")
        quat = Rotation.from_euler("xyz", raw[:, 3:6]).as_quat()[:, [3, 0, 1, 2]]
        if quat[0, 0] < 0:
            quat *= -1
        state_eef[:, offset:offset + 3] = raw[:, :3]
        state_eef[:, offset + 3:offset + 7] = quat
        state_eef[:, offset + 7] = joint[:, offset + 7]
    state_eef = make_quaternion_continuous(state_eef).astype(np.float32)
    action_eef = make_quaternion_continuous(fk.action_to_eef(action)).astype(np.float32)
    for values in (action_eef, state_eef):
        if not np.isfinite(values).all():
            raise ValueError("non-finite EEF values")
        for start in (3, 11):
            np.testing.assert_allclose(np.linalg.norm(values[:, start:start + 4], axis=1), 1, atol=1e-7)
    return {
        "observation.state": joint, "action": action,
        "action_eef": action_eef, "observation.state_eef": state_eef,
        "selected_images": {cam: selected[cam] for cam in CAMERAS},
        "timestamps": {"grid_timestamp_ns": grid.tolist(), "camera_timestamp_ns": head[selected_head].tolist(),
                       "driver_timestamp_ns": driver_ts.tolist(), "eef_timestamp_ns": used_ts.tolist(),
                       "camera_aligned_fallback_frames": np.flatnonzero(~exact).tolist()},
    }


def _save_images(streams, payload, stage, image_id):
    image_paths = {}
    for camera in CAMERAS:
        paths = {}
        for index in np.unique(payload["selected_images"][camera]):
            record = streams[camera][int(index)]
            # Preserve JPEG bytes exactly. Convert other Pillow-readable formats to RGB JPEG.
            data = record["data"]
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                is_jpeg = img.format == "JPEG"
                if not is_jpeg:
                    buffer = io.BytesIO()
                    img.convert("RGB").save(buffer, format="JPEG", quality=95)
                    data = buffer.getvalue()
            relative = Path("images") / image_id / camera / f"{int(index):06d}.jpg"
            dest = stage / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            paths[int(index)] = relative.as_posix()
        image_paths[camera] = [paths[int(i)] for i in payload["selected_images"][camera]]
    payload["image_paths"] = image_paths


def _write_episode(root, index, offset, task_index, payload):
    n = len(payload["action"])
    values = {key: _fixed_list(payload[key], width) for key, width in VECTOR_COLUMNS.items()}
    values.update({
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, index, dtype=np.int64)),
        "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
        "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        **{f"observation.images.{cam}": pa.array(payload["image_paths"][cam]) for cam in CAMERAS},
    })
    path = root / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(values), path, compression="zstd")


def _stats(arrays, ddof=0):
    values = np.concatenate(arrays).astype(np.float64)
    return {"count": len(values), "min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0, ddof=ddof).tolist(),
            "q01": np.quantile(values, .01, axis=0).tolist(), "q99": np.quantile(values, .99, axis=0).tolist()}


def _contracts(fk):
    common = {"shape": [16], "dtype": "float32", "reference": "base_link", "position_unit": "metre",
              "quaternion_order": "wxyz", "layout": "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper",
              "normalization_split": "train", "quaternion_sign": "continuous per hand per episode"}
    return {
        "action_eef": {**common, "column": "action_eef", "source": "local FK of raw absolute q_exp",
                       "wrist_local_tcp_m": fk.wrist_to_tcp.tolist(), "normalization_stats": "meta/action_eef_stats.json"},
        "observation_state_eef": {**common, "column": "observation.state_eef",
                                  "source": "recorded /end_pos.ee_pose_l and ee_pose_r (actual)",
                                  "source_orientation": "Euler xyz radians", "kinematics_used": False,
                                  "gripper_source": "measured q indices 7,15; no feedback repair",
                                  "alignment": "exact driver timestamp; camera-aligned /end_pos fallback",
                                  "normalization_stats": "meta/state_eef_stats.json"},
    }


def _write_split(root, split, records, tasks, train_stats, stage, contracts, provenance):
    root.mkdir()
    episodes, mapping, offset = [], [], 0
    for index, record in enumerate(records):
        payload = record["payload"]
        n = len(payload["action"])
        task_index = tasks.index(record["prompt"])
        _write_episode(root, index, offset, task_index, payload)
        image_id = record["image_id"]
        (root / "images").mkdir(exist_ok=True)
        shutil.move(str(stage / "images" / image_id), str(root / "images" / image_id))
        for paths in payload["image_paths"].values():
            if not all((root / ref).is_file() for ref in paths):
                raise FileNotFoundError(f"missing output image for {record['source_id']}")
        episodes.append({"episode_index": index, "episode_id": record["source_id"], "length": n,
                         "tasks": [record["prompt"]], "task_index": task_index})
        mapping.append({key: value for key, value in record.items() if key not in ("payload", "image_id")}
                       | {"episode_index": index, "length": n, "fps": FPS, **payload["timestamps"]})
        offset += n
    _write_jsonl(root / "meta/episodes.jsonl", episodes)
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
    _write_jsonl(root / "meta/episodes_stats.jsonl", [{"episode_index": e["episode_index"], "length": e["length"]} for e in episodes])
    features = {key: {"dtype": "float32", "shape": [width]} for key, width in VECTOR_COLUMNS.items()}
    features.update({f"observation.images.{cam}": {"dtype": "image"} for cam in CAMERAS})
    features.update({key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1]}
                     for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
    _write_json(root / "meta/info.json", {
        "codebase_version": "openwam_linglong_raw_v2", "robot_type": "linglong_h", "total_episodes": len(records),
        "total_frames": offset, "total_tasks": len(tasks), "fps": FPS, "total_videos": 0,
        "total_chunks": (len(records) + 999) // 1000, "chunks_size": 1000,
        "splits": {split: f"0:{len(records)}"}, "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": None,
    })
    stats = {key: train_stats[key] for key in ("state", "action")}
    _write_json(root / "meta/normalization_stats.json", stats)
    np.save(root / "meta/normalization_stats.npy", {
        label: {key: np.asarray(value, dtype=np.float32) for key, value in stats[stream].items() if key != "count"}
        for label, stream in (("joint", "action"), ("joint_state", "state"))}, allow_pickle=True)
    for name, stream, column in (("action_eef", "action", "action_eef"), ("state_eef", "state", "observation.state_eef")):
        _write_json(root / f"meta/{name}_stats.json", train_stats[name])
    _write_json(root / "source_mapping.json", mapping)
    _write_json(root / f"{split}_episodes.json", list(range(len(records))))
    _write_json(root / "manifest.json", {
        "format": "openwam_linglong_raw_v2", "status": "complete", "split": split, "source_root": ".",
        "prompt": tasks[0], "fps": FPS, "state_dim": 24, "action_dim": 24,
        "action_semantics": "absolute q_exp", "state_semantics": "measured q, no feedback repair",
        "layout": "left7,left_gripper,right7,right_gripper,waist4,head2,base2",
        "num_frames": NUM_FRAMES, "video_stride": 4, "num_video_frames": 9,
        "total_episodes": len(records), "total_frames": offset,
        "total_windows": sum(len(r["payload"]["action"]) - NUM_FRAMES + 1 for r in records),
        "image_storage": "JPEG files inside this bucket; paths relative to bucket",
        "normalization": "train frames once; test reuses train statistics", **contracts, **provenance,
    })
    return {"episodes": len(records), "frames": offset}


def convert_raw(input_path, output_root, *, prompt=None, episode_config=None, test_fraction=.1, seed=0,
                model=DEFAULT_MODEL, scratch=None, head_grid_tolerance_ns=20_000_000):
    input_path = Path(input_path).expanduser().resolve(strict=True)
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"choose a fresh output directory: {output_root}")
    if input_path.is_dir() and input_path in output_root.parents:
        raise ValueError("output must be outside the raw input directory")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0,1)")
    if prompt is not None and not prompt.strip():
        raise ValueError("prompt must not be blank")
    config = {} if episode_config is None else episode_config
    if not isinstance(config, dict):
        raise ValueError("episode config must be an object keyed by source ID or metadata episode ID")
    model = Path(model).expanduser().resolve(strict=True)
    fk = LinglongKinematics(model)
    output_root.mkdir(parents=True)
    stage = output_root / ".staging"
    stage.mkdir()
    report = {"status": "incomplete", "input": str(input_path), "model_sha256": _sha256(model),
              "test_fraction": test_fraction, "seed": seed,
              "head_grid_tolerance_ns": int(head_grid_tolerance_ns), "duplicates": []}
    _write_json(output_root / "conversion.json", report)
    records, seen, config_used = [], {}, set()
    for source_id, bag, metadata in iter_bags(input_path, scratch):
        digest = _sha256(bag)
        if digest in seen:
            report["duplicates"].append({"source_id": source_id, "same_as": seen[digest], "db_sha256": digest})
            continue
        seen[digest] = source_id
        episode_id = str(metadata.get("episode", source_id))
        key = source_id if source_id in config else episode_id
        entry = config.get(key, {})
        if key in config:
            config_used.add(key)
        if not isinstance(entry, dict):
            raise ValueError(f"episode config entry must be an object: {key}")
        task = entry.get("prompt", prompt)
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"{source_id} (episode={episode_id}): pass --prompt or an episode-config prompt")
        split = entry.get("split")
        if split not in (None, "train", "test"):
            raise ValueError(f"{source_id}: split must be train/test")
        streams = read_bag(bag)
        try:
            payload = sample_episode(streams, fk, head_grid_tolerance_ns)
        except ValueError as exc:
            raise ValueError(f"{source_id}: {exc}") from exc
        image_id = f"episode_{len(records):06d}"
        _save_images(streams, payload, stage, image_id)
        records.append({"source_id": source_id, "source_episode": episode_id, "metadata": metadata,
                        "db_sha256": digest, "prompt": task, "split": split, "payload": payload, "image_id": image_id})
        print(f"decoded {len(records)}: {source_id} ({len(payload['action'])} frames)", flush=True)
    if not records:
        raise ValueError("no raw .db3 recordings found in input")
    if set(config) - config_used:
        raise ValueError(f"unmatched episode-config keys: {sorted(set(config) - config_used)}")
    # A metadata ID is only a convenient config key when it identifies one bag.
    for key in config_used:
        matching = [r for r in records if key == r["source_id"] or key == r["source_episode"]]
        if len(matching) != 1:
            raise ValueError(f"ambiguous episode-config key {key}; use full source_id")
    unassigned = sorted((r for r in records if r["split"] is None), key=lambda r: r["db_sha256"])
    count = min(len(unassigned) - 1, max(1, round(len(unassigned) * test_fraction))) if test_fraction and len(unassigned) > 1 else 0
    order = np.random.default_rng(seed).permutation(len(unassigned))
    test_hashes = {unassigned[int(i)]["db_sha256"] for i in order[:count]}
    for record in unassigned:
        record["split"] = "test" if record["db_sha256"] in test_hashes else "train"
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    if not train:
        raise ValueError("at least one training episode is required for normalization")
    stats = {key: _stats([r["payload"][column] for r in train])
             for key, column in (("state", "observation.state"), ("action", "action"))}
    for name, stream, column in (("action_eef", "action", "action_eef"), ("state_eef", "state", "observation.state_eef")):
        values = _stats([r["payload"][column] for r in train], ddof=1)
        stats[name] = {"source_column": column, "statistics_source": "train frames once, no window weighting",
                       "num_episodes": len(train), "num_transition": values["count"], "std_ddof": 1,
                       stream: {"default": {f"global_{k}": v for k, v in values.items() if k != "count"}}}
    tasks = sorted({r["prompt"] for r in records})
    provenance = {"raw_input": str(input_path), "model_sha256": report["model_sha256"],
                  "split_method": "explicit episode config; otherwise seeded shuffle of bag hashes",
                  "split_seed": seed, "test_fraction": test_fraction}
    results = {}
    for split, rows in (("train", train), ("test", test)):
        if rows:
            root = output_root / f"linglong_openwam_{split}_15hz"
            results[split] = _write_split(root, split, rows, tasks, stats, stage, _contracts(fk), provenance)
    # Only remove this run's now-empty staging directories.
    (stage / "images").rmdir()
    stage.rmdir()
    report.update(status="complete", splits=results,
                  camera_aligned_fallback_frames=sum(len(r["payload"]["timestamps"]["camera_aligned_fallback_frames"]) for r in records))
    _write_json(output_root / "conversion.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw zip, folder, per-episode tar(.gz), or .db3")
    parser.add_argument("--output-root", type=Path, required=True, help="fresh directory; output includes sampled images")
    parser.add_argument("--prompt", help="task text for all episodes unless overridden by episode-config")
    parser.add_argument("--episode-config", type=Path, help='JSON {"source_id or metadata episode": {"prompt": "...", "split": "train|test"}}')
    parser.add_argument("--test-fraction", type=float, default=.1, help="held-out fraction of unassigned episodes; 0 for train only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="optional FK XML or linglong.zip; defaults to repository asset")
    parser.add_argument("--scratch-dir", type=Path, help="temporary archive extraction directory (default: system temp)")
    parser.add_argument("--head-grid-tolerance-ms", type=float, default=20,
                        help="maximum head-camera error against the 15Hz grid; head gaps above 50ms still fail")
    args = parser.parse_args()
    if args.prompt is None and args.episode_config is None:
        parser.error("pass --prompt or --episode-config; task meaning is not inferred from filenames")
    config = json.loads(args.episode_config.read_text()) if args.episode_config else None
    if not np.isfinite(args.head_grid_tolerance_ms) or args.head_grid_tolerance_ms <= 0:
        parser.error("--head-grid-tolerance-ms must be positive")
    convert_raw(args.input, args.output_root, prompt=args.prompt, episode_config=config,
                test_fraction=args.test_fraction, seed=args.seed, model=args.model, scratch=args.scratch_dir,
                head_grid_tolerance_ns=int(round(args.head_grid_tolerance_ms * 1_000_000)))


if __name__ == "__main__":
    main()
