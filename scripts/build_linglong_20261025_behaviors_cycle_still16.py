"""Copy the 20261025 behaviors bucket with one prompt and 16 still frames.

The source bucket repeats the last frame 64 times. This copy drops the final
48 of those copies and replaces every prompt. Images stay as a symlink.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE = Path("/vePFS/data/linglong/linglong_20261025_behaviors_openwam/linglong_openwam_train_15hz")
OUTPUT = Path("/vePFS/data/linglong/linglong_20261025_behaviors_cycle_still16_openwam/linglong_openwam_train_15hz")
PROMPT = (
    "Pick up the sheet workpiece from the box, move to the front of the workpiece storage rack, "
    "place the sheet workpiece into the upper shelf, and return to the front of the workpiece pickup box."
)
SOURCE_STILL = 64
STILL_FRAMES = 16
DROP = SOURCE_STILL - STILL_FRAMES
NUMERIC = ("observation.state", "action", "observation.state_eef", "action_eef")
WIDTH = {"observation.state": 24, "action": 24, "observation.state_eef": 16, "action_eef": 16}
CAMERAS = ("head", "left", "right")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def stats(values, ddof):
    values = np.concatenate(values).astype(np.float64)
    return {
        "count": int(len(values)),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0, ddof=ddof).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def fixed(values, width):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"expected finite (*,{width}), got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


def trim_episode(table, length):
    if table.num_rows != length or length < 33 + SOURCE_STILL:
        raise ValueError(f"episode length {table.num_rows} is not {length} with {SOURCE_STILL} still frames")
    columns = {key: np.asarray(table[key].to_pylist(), dtype=np.float64) for key in NUMERIC}
    images = {camera: table[f"observation.images.{camera}"].to_pylist() for camera in CAMERAS}
    for key, values in columns.items():
        if values.shape != (length, WIDTH[key]):
            raise ValueError(f"{key} shape {values.shape}")
        if np.max(np.ptp(values[-SOURCE_STILL:], axis=0)) > 0:
            raise ValueError(f"{key} tail is not {SOURCE_STILL} identical frames")
        if not np.allclose(values[-SOURCE_STILL], values[-SOURCE_STILL - 1]):
            raise ValueError(f"{key} still frames are not a copy of the last motion frame")
    for camera, paths in images.items():
        if len(paths) != length or len(set(paths[-SOURCE_STILL:])) != 1 or paths[-SOURCE_STILL] != paths[-SOURCE_STILL - 1]:
            raise ValueError(f"{camera} still frames are not one repeated image")
    kept = length - DROP
    columns = {key: values[:kept] for key, values in columns.items()}
    images = {camera: paths[:kept] for camera, paths in images.items()}
    for key, values in columns.items():
        if np.max(np.ptp(values[-STILL_FRAMES:], axis=0)) > 0:
            raise ValueError(f"{key} no longer ends with {STILL_FRAMES} still frames")
    return columns, images, kept


def write_episode(index, offset, columns, images):
    count = len(columns["action"])
    out = {
        key: fixed(columns[key], WIDTH[key])
        for key in NUMERIC
    }
    out.update({
        "timestamp": pa.array(np.arange(count, dtype=np.float32) / 15),
        "frame_index": pa.array(np.arange(count, dtype=np.int64)),
        "episode_index": pa.array(np.full(count, index, dtype=np.int64)),
        "index": pa.array(np.arange(offset, offset + count, dtype=np.int64)),
        "task_index": pa.array(np.zeros(count, dtype=np.int64)),
    })
    for camera in CAMERAS:
        out[f"observation.images.{camera}"] = pa.array(images[camera])
    path = OUTPUT / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), path, compression="zstd")
    return count


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_manifest = read_json(SOURCE / "manifest.json")
    if source_manifest.get("status") != "complete" or source_manifest.get("still_frames") != SOURCE_STILL:
        raise ValueError("source bucket is not the complete 64-still behaviors train set")
    episodes = [json.loads(line) for line in (SOURCE / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    mappings = read_json(SOURCE / "source_mapping.json")
    if len(episodes) != len(mappings):
        raise ValueError("episodes and source_mapping differ in length")
    OUTPUT.mkdir(parents=True)
    images = OUTPUT / "images"
    images.symlink_to(Path("..") / ".." / "linglong_20261025_behaviors_openwam" / "linglong_openwam_train_15hz" / "images")
    arrays = {key: [] for key in NUMERIC}
    written_episodes, written_mappings = [], []
    offset = 0
    for index, (episode, mapping) in enumerate(zip(episodes, mappings)):
        if episode["episode_index"] != index or mapping["episode_index"] != index:
            raise ValueError(f"episode {index} is out of order")
        if mapping.get("still_frames") != SOURCE_STILL or episode["length"] != mapping["length"]:
            raise ValueError(f"episode {index} still-frame record does not match its length")
        path = SOURCE / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
        columns, image_paths, kept = trim_episode(pq.read_table(path), int(episode["length"]))
        count = write_episode(index, offset, columns, image_paths)
        for key in NUMERIC:
            arrays[key].append(columns[key])
        written_episodes.append({
            "episode_index": index,
            "episode_id": episode["episode_id"],
            "length": count,
            "tasks": [PROMPT],
            "task_index": 0,
        })
        written_mappings.append({
            **mapping,
            "source_prompt": mapping["prompt"],
            "prompt": PROMPT,
            "source_still_frames": SOURCE_STILL,
            "still_frames": STILL_FRAMES,
            "source_length": mapping["length"],
            "length": count,
        })
        offset += count
        if (index + 1) % 50 == 0:
            print(json.dumps({"episodes": index + 1, "frames": offset}), flush=True)
    joint = {
        "state": stats(arrays["observation.state"], 0),
        "action": stats(arrays["action"], 0),
    }
    eef = {
        "action_eef": stats(arrays["action_eef"], 1),
        "observation.state_eef": stats(arrays["observation.state_eef"], 1),
    }
    write_jsonl(OUTPUT / "meta/episodes.jsonl", written_episodes)
    write_jsonl(OUTPUT / "meta/tasks.jsonl", [{"task_index": 0, "task": PROMPT}])
    write_jsonl(OUTPUT / "meta/episodes_stats.jsonl", [
        {"episode_index": episode["episode_index"], "length": episode["length"]} for episode in written_episodes
    ])
    info = read_json(SOURCE / "meta/info.json")
    info["total_episodes"] = len(written_episodes)
    info["total_frames"] = offset
    info["total_tasks"] = 1
    info["codebase_version"] = "openwam_linglong_behaviors_20261025_cycle_still16"
    write_json(OUTPUT / "meta/info.json", info)
    write_json(OUTPUT / "meta/normalization_stats.json", joint)
    np.save(OUTPUT / "meta/normalization_stats.npy", {
        label: {key: np.asarray(value, dtype=np.float32) for key, value in joint[stream].items() if key != "count"}
        for label, stream in (("joint", "action"), ("joint_state", "state"))
    }, allow_pickle=True)
    for name, column, stream in (
        ("action_eef", "action_eef", "action"),
        ("state_eef", "observation.state_eef", "state"),
    ):
        payload = eef[column]
        write_json(OUTPUT / f"meta/{name}_stats.json", {
            "source_column": column,
            "statistics_source": "copied behaviors train with the last 48 of 64 still frames removed",
            "num_episodes": len(written_episodes),
            "num_transition": payload["count"],
            "std_ddof": 1,
            stream: {"default": {f"global_{key}": value for key, value in payload.items() if key != "count"}},
        })
    write_json(OUTPUT / "source_mapping.json", written_mappings)
    write_json(OUTPUT / "train_episodes.json", list(range(len(written_episodes))))
    write_json(OUTPUT / "processing_report.json", {
        "source": str(SOURCE),
        "prompt": PROMPT,
        "episodes": len(written_episodes),
        "frames": offset,
        "windows": sum(episode["length"] - 32 for episode in written_episodes),
        "source_still_frames": SOURCE_STILL,
        "still_frames": STILL_FRAMES,
        "dropped_still_frames_per_episode": DROP,
    })
    manifest = dict(source_manifest)
    manifest.update({
        "format": "openwam_linglong_behaviors_20261025_cycle_still16",
        "prompt": PROMPT,
        "prompts": [PROMPT],
        "prompt_source": "one cycle prompt replacing the four behavior prompts",
        "total_episodes": len(written_episodes),
        "total_frames": offset,
        "total_windows": sum(episode["length"] - 32 for episode in written_episodes),
        "image_storage": "symlink to the 20261025 behaviors bucket images",
        "normalization": "recomputed after keeping 16 of the original 64 still frames",
        "still_frames": STILL_FRAMES,
        "source_bucket": str(SOURCE),
        "task_counts": {PROMPT: len(written_episodes)},
    })
    write_json(OUTPUT / "manifest.json", manifest)
    return manifest["total_episodes"], manifest["total_frames"], manifest["total_windows"]


if __name__ == "__main__":
    episodes, frames, windows = build()
    print(json.dumps({"episodes": episodes, "frames": frames, "windows": windows, "output": str(OUTPUT)}), flush=True)
