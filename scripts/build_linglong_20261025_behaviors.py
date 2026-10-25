"""Merge the sheet train bucket with the 2026-10-24 train bucket.

The sheet bucket is the published 1019+1020+1023 train prefix. This pass keeps
four prompts, cuts pickup episodes that lower both arms after the lift, holds
the unused expert stream still, then repeats the last frame 64 times.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.tos_cache import BoundedTosCache, DatasetStorage, TosBackend, join_tos


SHEET = "tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261024_sheet_20261019_20261020_20261023_train_v1"
NEW = Path("/vePFS/data/linglong/linglong_20261024_openwam/linglong_openwam_train_15hz")
OUTPUT = Path("/vePFS/data/linglong/linglong_20261025_behaviors_openwam/linglong_openwam_train_15hz")
PICKUP = "Pick up the sheet workpiece from the box."
MOVE_RACK = "Move to the front of the workpiece storage rack."
MOVE_BOX = "Move to the front of the workpiece pickup box."
PLACE = "Place the sheet workpiece into the upper shelf."
TASKS = (PICKUP, MOVE_RACK, MOVE_BOX, PLACE)
MOVES = {MOVE_RACK, MOVE_BOX}
CAMERAS = ("head", "left", "right")
NUMERIC = ("observation.state", "action", "observation.state_eef", "action_eef")
WIDTH = {"observation.state": 24, "action": 24, "observation.state_eef": 16, "action_eef": 16}
STILL_FRAMES = 64
BASE = slice(22, 24)


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


def read_jsonl_text(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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


def pickup_cut(state_eef):
    """Drop the tail where both arms descend after a high lift.

    Returns the number of leading frames to keep, and whether a tail was removed.
    A pickup that stays up, or never rises above 0.90 m, is unchanged.
    """
    left = np.asarray(state_eef[:, 2], dtype=np.float64)
    right = np.asarray(state_eef[:, 10], dtype=np.float64)
    if min(left.max(), right.max()) < 0.90 or max(left[-1], right[-1]) > 0.88:
        return len(left), False
    if left.max() - left[-1] < 0.06 or right.max() - right[-1] < 0.06:
        return len(left), False
    height = np.minimum(left, right)
    peak_i = int(height.argmax())
    near = np.flatnonzero((np.arange(len(height)) >= peak_i) & (height >= height[peak_i] - 0.02))
    if len(near) == 0:
        raise ValueError("pickup lift has no frame within 2cm of the peak")
    cut = int(near[-1]) + 1
    if cut < 33 or cut >= len(height):
        raise ValueError(f"pickup cut {cut} is outside 33..{len(height) - 1}")
    return cut, True


def hold_row(values, row=0):
    held = np.repeat(values[row:row + 1], len(values), axis=0)
    return held


def repeat_last(values, copies):
    return np.concatenate([values, np.repeat(values[-1:], copies, axis=0)], axis=0)


def repeat_last_paths(paths, copies):
    return [*paths, *([paths[-1]] * copies)]


def _fixed(values, width):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"expected finite (*,{width}), got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


def load_episodes(storage):
    manifest = json.loads(storage.text("manifest.json"))
    if manifest.get("status") != "complete" or manifest.get("source_root") != ".":
        raise ValueError(f"{storage.root} is not a complete portable bucket")
    episodes = read_jsonl_text(storage.text("meta/episodes.jsonl"))
    mapping = json.loads(storage.text("source_mapping.json"))
    if len(mapping) != len(episodes):
        raise ValueError(f"{storage.root} source_mapping length mismatch")
    return manifest, episodes, mapping


def process_episode(prompt, columns, images):
    state = columns["observation.state"]
    action = columns["action"]
    state_eef = columns["observation.state_eef"]
    action_eef = columns["action_eef"]
    cut = len(state)
    trimmed = False
    if prompt == PICKUP:
        cut, trimmed = pickup_cut(state_eef)
        state, action = state[:cut], action[:cut]
        state_eef, action_eef = state_eef[:cut], action_eef[:cut]
        images = {camera: paths[:cut] for camera, paths in images.items()}
    if prompt in (PICKUP, PLACE):
        state[:, BASE] = hold_row(state)[:, BASE]
        action[:, BASE] = hold_row(action)[:, BASE]
    elif prompt in MOVES:
        state_eef = hold_row(state_eef)
        action_eef = hold_row(action_eef)
    else:
        raise ValueError(f"unexpected prompt {prompt}")
    state = repeat_last(state, STILL_FRAMES)
    action = repeat_last(action, STILL_FRAMES)
    state_eef = repeat_last(state_eef, STILL_FRAMES)
    action_eef = repeat_last(action_eef, STILL_FRAMES)
    images = {camera: repeat_last_paths(paths, STILL_FRAMES) for camera, paths in images.items()}
    for values in (state_eef, action_eef):
        for start in (3, 11):
            norm = np.linalg.norm(values[:, start:start + 4], axis=1)
            if np.max(np.abs(norm - 1.0)) > 1e-3:
                raise ValueError("EEF quaternion left the unit sphere")
    if prompt in (PICKUP, PLACE) and np.max(np.ptp(state[:, BASE], axis=0)) > 0:
        raise ValueError("pickup/place base still moves")
    if prompt in MOVES and np.max(np.ptp(state_eef, axis=0)) > 0:
        raise ValueError("move EEF still moves")
    return {
        "observation.state": state,
        "action": action,
        "observation.state_eef": state_eef,
        "action_eef": action_eef,
        "images": images,
        "source_frames": cut,
        "trimmed": trimmed,
    }


def write_episode(output, index, offset, task_index, processed):
    count = len(processed["action"])
    columns = {
        key: _fixed(processed[key], WIDTH[key])
        for key in NUMERIC
    }
    columns.update({
        "timestamp": pa.array(np.arange(count, dtype=np.float32) / 15),
        "frame_index": pa.array(np.arange(count, dtype=np.int64)),
        "episode_index": pa.array(np.full(count, index, dtype=np.int64)),
        "index": pa.array(np.arange(offset, offset + count, dtype=np.int64)),
        "task_index": pa.array(np.full(count, task_index, dtype=np.int64)),
    })
    for camera in CAMERAS:
        columns[f"observation.images.{camera}"] = pa.array(processed["images"][camera])
    path = output / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd")
    return count


def build(sheet_storage, new_storage, output):
    resuming = output.exists()
    if resuming:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("status") != "incomplete":
            raise FileExistsError(output)
    else:
        output.mkdir(parents=True)
        write_json(output / "manifest.json", {"status": "incomplete", "split": "train"})
    backend = TosBackend()
    arrays = {key: [] for key in NUMERIC}
    episodes, mappings = [], []
    counts = {task: 0 for task in TASKS}
    trimmed = []
    offset = 0
    index = 0
    sources = (
        (sheet_storage, "sheet", ""),
        (new_storage, "20261024", "20261024/"),
    )
    for storage, origin, prefix in sources:
        manifest, source_episodes, source_mapping = load_episodes(storage)
        for episode, source in zip(source_episodes, source_mapping):
            prompt = episode["tasks"][0]
            if prompt not in TASKS:
                raise ValueError(f"{origin} episode {episode['episode_index']} prompt {prompt!r}")
            rel = f"data/chunk-{episode['episode_index'] // 1000:03d}/episode_{episode['episode_index']:06d}.parquet"
            done = output / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
            if done.is_file():
                table = pq.read_table(done)
                processed = {key: np.asarray(table[key].to_pylist(), dtype=np.float64) for key in NUMERIC}
                count = table.num_rows
                if count < 33 + STILL_FRAMES:
                    raise ValueError(f"{done} does not contain the source window plus {STILL_FRAMES} still frames")
                kept = count - STILL_FRAMES
                trimmed_episode = prompt == PICKUP and kept < episode["length"]
                for key in NUMERIC:
                    arrays[key].append(processed[key])
                episodes.append({
                    "episode_index": index,
                    "episode_id": episode["episode_id"],
                    "length": count,
                    "tasks": [prompt],
                    "task_index": TASKS.index(prompt),
                })
                mappings.append({
                    "episode_index": index,
                    "episode_id": episode["episode_id"],
                    "input_origin": origin,
                    "input_root": storage.root,
                    "input_episode_index": episode["episode_index"],
                    "prompt": prompt,
                    "split": "train",
                    "source_frames": episode["length"],
                    "kept_before_still": kept,
                    "pickup_lowered_arms_removed": trimmed_episode,
                    "still_frames": STILL_FRAMES,
                    "length": count,
                })
                if trimmed_episode:
                    trimmed.append({
                        "episode_index": index,
                        "episode_id": episode["episode_id"],
                        "source_frames": episode["length"],
                        "kept_frames": kept,
                    })
                counts[prompt] += 1
                offset += count
                index += 1
                continue
            with storage.open(rel) as handle:
                table = pq.read_table(handle)
            if table.num_rows != episode["length"]:
                raise ValueError(f"{rel} row count mismatch")
            columns = {key: np.asarray(table[key].to_pylist(), dtype=np.float64) for key in NUMERIC}
            images = {}
            for camera in CAMERAS:
                paths = table[f"observation.images.{camera}"].to_pylist()
                rewritten = []
                for path in paths:
                    if not isinstance(path, str) or not path.startswith("images/"):
                        raise ValueError(f"unexpected image path {path}")
                    rewritten.append(f"images/{prefix}{path[len('images/'):]}")
                images[camera] = rewritten
            processed = process_episode(prompt, columns, images)
            referenced = {path for camera in CAMERAS for path in processed["images"][camera]}
            original_by_rewritten = {}
            for camera in CAMERAS:
                for original, rewritten_path in zip(table[f"observation.images.{camera}"].to_pylist(), images[camera]):
                    original_by_rewritten[rewritten_path] = original

            def fetch(rewritten_path, source_rel=original_by_rewritten, storage=storage, output=output, backend=backend):
                dest = output / rewritten_path
                if dest.is_file() and dest.stat().st_size > 0:
                    return
                dest.parent.mkdir(parents=True, exist_ok=True)
                source_path = source_rel[rewritten_path]
                if not storage.remote:
                    dest.symlink_to((Path(storage.root) / source_path).resolve(strict=True))
                    return
                uri = join_tos(storage.root, source_path)
                temporary = dest.with_suffix(dest.suffix + ".part")
                size, etag = backend.head(uri)
                backend.download(uri, temporary, etag)
                if temporary.stat().st_size != size:
                    raise IOError(f"incomplete download {uri}")
                temporary.replace(dest)

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(fetch, sorted(referenced)))
            count = write_episode(output, index, offset, TASKS.index(prompt), processed)
            if processed["trimmed"]:
                trimmed.append({
                    "episode_index": index,
                    "episode_id": episode["episode_id"],
                    "source_frames": episode["length"],
                    "kept_frames": processed["source_frames"],
                })
            for key in NUMERIC:
                arrays[key].append(processed[key])
            episodes.append({
                "episode_index": index,
                "episode_id": episode["episode_id"],
                "length": count,
                "tasks": [prompt],
                "task_index": TASKS.index(prompt),
            })
            mappings.append({
                **{key: value for key, value in source.items() if key not in ("grid_timestamp_ns", "camera_timestamp_ns", "driver_timestamp_ns", "eef_timestamp_ns", "camera_aligned_fallback_frames")},
                "episode_index": index,
                "input_origin": origin,
                "input_root": storage.root,
                "input_episode_index": episode["episode_index"],
                "prompt": prompt,
                "split": "train",
                "source_frames": episode["length"],
                "kept_before_still": processed["source_frames"],
                "pickup_lowered_arms_removed": processed["trimmed"],
                "still_frames": STILL_FRAMES,
                "length": count,
            })
            counts[prompt] += 1
            offset += count
            index += 1
            if index % 20 == 0:
                print(json.dumps({"episodes": index, "frames": offset, "trimmed": len(trimmed)}), flush=True)
        del manifest
    train_stats = {
        "episodes": len(episodes),
        "joint": {
            "state": stats(arrays["observation.state"], 0),
            "action": stats(arrays["action"], 0),
        },
        "eef": {
            "action_eef": stats(arrays["action_eef"], 1),
            "observation.state_eef": stats(arrays["observation.state_eef"], 1),
        },
    }
    write_jsonl(output / "meta/episodes.jsonl", episodes)
    write_jsonl(output / "meta/tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(TASKS)])
    write_jsonl(output / "meta/episodes_stats.jsonl", [
        {"episode_index": episode["episode_index"], "length": episode["length"]} for episode in episodes
    ])
    features = {key: {"dtype": "float32", "shape": [width]} for key, width in WIDTH.items()}
    features.update({f"observation.images.{camera}": {"dtype": "image"} for camera in CAMERAS})
    features.update({
        key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1]}
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    })
    write_json(output / "meta/info.json", {
        "codebase_version": "openwam_linglong_behaviors_20261025",
        "robot_type": "linglong_h",
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(TASKS),
        "fps": 15,
        "total_videos": 0,
        "total_chunks": (len(episodes) + 999) // 1000,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
    })
    joint = train_stats["joint"]
    write_json(output / "meta/normalization_stats.json", joint)
    np.save(output / "meta/normalization_stats.npy", {
        label: {key: np.asarray(value, dtype=np.float32) for key, value in joint[stream].items() if key != "count"}
        for label, stream in (("joint", "action"), ("joint_state", "state"))
    }, allow_pickle=True)
    for name, column, stream in (
        ("action_eef", "action_eef", "action"),
        ("state_eef", "observation.state_eef", "state"),
    ):
        payload = train_stats["eef"][column]
        write_json(output / f"meta/{name}_stats.json", {
            "source_column": column,
            "statistics_source": "processed train frames once, including the repeated still frames",
            "num_episodes": len(episodes),
            "num_transition": payload["count"],
            "std_ddof": 1,
            stream: {"default": {f"global_{key}": value for key, value in payload.items() if key != "count"}},
        })
    write_json(output / "source_mapping.json", mappings)
    write_json(output / "train_episodes.json", list(range(len(episodes))))
    write_json(output / "processing_report.json", {
        "trimmed_pickups": trimmed,
        "trimmed_count": len(trimmed),
        "task_counts": counts,
        "still_frames": STILL_FRAMES,
    })
    sheet_manifest = json.loads(sheet_storage.text("manifest.json"))
    write_json(output / "manifest.json", {
        "format": "openwam_linglong_behaviors_20261025",
        "status": "complete",
        "split": "train",
        "source_root": ".",
        "prompt": PICKUP,
        "prompts": list(TASKS),
        "prompt_source": "sheet pickup/place plus 20261024 rack move, box move, and place",
        "fps": 15,
        "state_dim": 24,
        "action_dim": 24,
        "action_semantics": "absolute q_exp; base held at frame 0 for pickup and place",
        "state_semantics": "measured q; base held at frame 0 for pickup and place",
        "layout": "left7,left_gripper,right7,right_gripper,waist4,head2,base2",
        "num_frames": 33,
        "video_stride": 4,
        "num_video_frames": 9,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_windows": sum(episode["length"] - 32 for episode in episodes),
        "image_storage": "JPEG files inside this bucket; 20261024 images may be symlinks until publication",
        "normalization": "recomputed on this processed train, including 64 repeated final frames",
        "normalization_dataset_dir": ".",
        "still_frames": STILL_FRAMES,
        "pickup_lowered_arms_removed": len(trimmed),
        "move_eef": "action_eef and observation.state_eef held at frame 0",
        "pickup_place_base": "action and observation.state dims 22:24 held at frame 0",
        "action_eef": {
            **sheet_manifest["action_eef"],
            "normalization_split": "train",
            "normalization_stats": "meta/action_eef_stats.json",
            "move_episodes": "held at the first frame",
        },
        "observation_state_eef": {
            **sheet_manifest["observation_state_eef"],
            "normalization_split": "train",
            "normalization_stats": "meta/state_eef_stats.json",
            "move_episodes": "held at the first frame",
        },
        "input_buckets": [sheet_storage.root, str(new_storage.root)],
        "task_counts": counts,
    })
    return {"episodes": len(episodes), "frames": offset, "tasks": counts, "trimmed": len(trimmed)}


def _self_check():
    rising = np.linspace(0.7, 1.02, 80)
    falling = np.linspace(1.02, 0.84, 40)
    both = np.concatenate([rising, falling])
    state = np.zeros((len(both), 16))
    state[:, 2] = both
    state[:, 10] = both
    cut, trimmed = pickup_cut(state)
    if not trimmed or not 80 <= cut <= 90:
        raise AssertionError(f"expected the lift to be kept, cut={cut}")
    state[-1, 2] = state[-1, 10] = 1.01
    cut, trimmed = pickup_cut(state)
    if trimmed or cut != len(state):
        raise AssertionError("an episode that stays up must not be cut")
    low = np.zeros((100, 16))
    low[:, 2] = low[:, 10] = 0.85
    cut, trimmed = pickup_cut(low)
    if trimmed:
        raise AssertionError("a low pickup must not be cut")


def main():
    _self_check()
    cache = BoundedTosCache("/tmp/openwam-tos-behaviors-20261025", 8 * 1024**3, 512 * 1024**2)
    sheet = DatasetStorage(SHEET, cache)
    new = DatasetStorage(str(NEW.resolve()))
    if not NEW.is_dir():
        raise FileNotFoundError(NEW)
    result = build(sheet, new, OUTPUT)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
