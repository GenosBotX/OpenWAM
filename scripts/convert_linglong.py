"""Convert Linglong decoded episodes to the OpenWAM single-bucket format.

The converter intentionally keeps Linglong's native 24-D joint-space contract:
``state=q`` and ``action=q_exp``.  It creates 15 Hz episode parquet files with
relative JPEG references, OpenWAM metadata, train statistics, and a manifest.
No source files are modified or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROMPT = "Use the right arm to pick up the blue block from the table and place it in the box."
STATE_DIM = 24
ACTION_DIM = 24
FPS = 15


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fixed_list(values: np.ndarray, width: int) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"expected (*,{width}) array, got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), width)


def _sample_episode(decoded_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    csv_path = decoded_dir / "aligned.csv"
    alignment_path = decoded_dir / "alignment.jsonl"
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    with alignment_path.open() as stream:
        alignments = [json.loads(line) for line in stream]
    if len(rows) != len(alignments) or not rows:
        raise ValueError(f"{decoded_dir}: aligned.csv/alignment.jsonl length mismatch")

    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{decoded_dir}: non-monotonic timestamps")
    q = np.asarray([[float(row[f"q_{i}"]) for i in range(STATE_DIM)] for row in rows], dtype=np.float32)
    q_exp = np.asarray([[float(row[f"q_exp_{i}"]) for i in range(ACTION_DIM)] for row in rows], dtype=np.float32)
    if not np.isfinite(q).all() or not np.isfinite(q_exp).all():
        raise ValueError(f"{decoded_dir}: non-finite state/action")

    # Match the existing 15 Hz LeRobot conversion: nearest decoded row on a
    # uniform grid, with the head timestamp as the reference clock.
    grid = timestamps[0] + np.rint(
        np.arange(int((timestamps[-1] - timestamps[0]) * FPS / 1e9) + 1) * 1e9 / FPS
    ).astype(np.int64)
    right = np.minimum(np.searchsorted(timestamps, grid, side="left"), len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(timestamps[left] - grid) <= np.abs(timestamps[right] - grid)
    indices = np.where(choose_left, left, right)

    for tick, index in zip(grid, indices):
        alignment = alignments[int(index)]
        if abs(int(timestamps[index]) - int(tick)) > 20_000_000:
            raise ValueError(f"{decoded_dir}: head alignment exceeds 20 ms")
        for source_ts in alignment.get("source_timestamps", {}).values():
            if source_ts is None or abs(int(source_ts) - int(tick)) > 80_000_000:
                raise ValueError(f"{decoded_dir}: sensor alignment exceeds 80 ms")

    state = q[indices]
    action = q_exp[indices]
    source_root = decoded_dir.parents[1]  # processed/{train|test}
    image_paths = {}
    for camera in ("head", "left", "right"):
        rels = []
        for index in indices:
            image_rel = alignments[int(index)]["images"][camera]
            rel = Path("decoded") / decoded_dir.name / image_rel
            if not (source_root / rel).is_file():
                raise FileNotFoundError(source_root / rel)
            rels.append(rel.as_posix())
        image_paths[camera] = rels

    payload = {
        "state": state,
        "action": action,
        "timestamp": np.arange(len(indices), dtype=np.float32) / FPS,
        "image_paths": image_paths,
        "source_episode": decoded_dir.name,
        "source_sha256": _sha256(csv_path),
    }
    return payload, state, action


def _write_episode(out_root: Path, episode_index: int, payload: dict) -> None:
    n = len(payload["state"])
    chunk = episode_index // 1000
    path = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.state": _fixed_list(payload["state"], STATE_DIM),
            "action": _fixed_list(payload["action"], ACTION_DIM),
            "timestamp": pa.array(payload["timestamp"], type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "observation.images.head": pa.array(payload["image_paths"]["head"]),
            "observation.images.left": pa.array(payload["image_paths"]["left"]),
            "observation.images.right": pa.array(payload["image_paths"]["right"]),
        }
    )
    pq.write_table(table, path, compression="zstd")


def _stats(arrays: list[np.ndarray]) -> dict:
    x = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "count": int(len(x)),
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def convert_split(source_root: Path, output_root: Path, split: str, train_stats: dict | None = None) -> dict:
    inventory = json.loads((source_root / "inventory.json").read_text())
    decoded_root = source_root / "decoded"
    output_root.mkdir(parents=True, exist_ok=False)
    episodes = []
    mapping = []
    state_arrays, action_arrays = [], []
    total_frames = 0
    for episode_index, record in enumerate(inventory):
        source_id = str(record["episode"])
        payload, state, action = _sample_episode(decoded_root / source_id)
        _write_episode(output_root, episode_index, payload)
        n = len(state)
        episodes.append({"episode_index": episode_index, "episode_id": source_id, "length": n, "tasks": [PROMPT]})
        mapping.append({
            "episode_index": episode_index,
            "source_episode": source_id,
            "source_csv": str((decoded_root / source_id / "aligned.csv").resolve()),
            "source_sha256": payload["source_sha256"],
            "length": n,
            "fps": FPS,
            "action_semantics": "absolute q_exp",
        })
        state_arrays.append(state)
        action_arrays.append(action)
        total_frames += n

    stats = train_stats or {"state": _stats(state_arrays), "action": _stats(action_arrays)}
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": PROMPT}])
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", [
        {"episode_index": e["episode_index"], "length": e["length"]} for e in episodes
    ])
    _write_json(output_root / "meta" / "info.json", {
        "codebase_version": "openwam_linglong_v1",
        "robot_type": "linglong_h",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "observation.images.head": {"dtype": "image"},
            "observation.images.left": {"dtype": "image"},
            "observation.images.right": {"dtype": "image"},
        },
    })
    _write_json(output_root / "meta" / "normalization_stats.json", stats)
    np.save(
        output_root / "meta" / "normalization_stats.npy",
        {
            "joint": {k: np.asarray(v, dtype=np.float32) for k, v in stats["action"].items() if k != "count"},
            "joint_state": {k: np.asarray(v, dtype=np.float32) for k, v in stats["state"].items() if k != "count"},
        },
        allow_pickle=True,
    )
    _write_json(output_root / "source_mapping.json", mapping)
    _write_json(output_root / f"{split}_episodes.json", [e["episode_index"] for e in episodes])
    _write_json(output_root / "manifest.json", {
        "format": "openwam_linglong_v1",
        "status": "complete",
        "split": split,
        "source_root": str(source_root.resolve()),
        "prompt": PROMPT,
        "fps": FPS,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_semantics": "absolute q_exp",
        "state_semantics": "measured q",
        "layout": "left7,left_gripper,right7,right_gripper,waist4,head2,base2",
        "num_frames": 33,
        "video_stride": 4,
        "num_video_frames": 9,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "image_storage": "relative paths resolved against source_root",
        "normalization": "train split statistics; test reuses train statistics",
    })
    return {"episodes": len(episodes), "frames": total_frames, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_0923_processed"))
    parser.add_argument("--output-root", type=Path, default=Path("/iag_ad_01/ad/dushuyue/openpi/data"))
    parser.add_argument("--train-name", default="linglong_0923_openwam_train_15hz")
    parser.add_argument("--test-name", default="linglong_0923_openwam_test_15hz")
    args = parser.parse_args()
    train_out = args.output_root / args.train_name
    test_out = args.output_root / args.test_name
    if train_out.exists() or test_out.exists():
        raise FileExistsError("output already exists; choose another name or move it aside")
    train_result = convert_split(args.processed_root / "train", train_out, "train")
    convert_split(args.processed_root / "test", test_out, "test", train_stats=train_result["stats"])
    print(json.dumps({"train": train_result, "test": {"output": str(test_out)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
