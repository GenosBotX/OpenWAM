"""Merge the 2026-10-23 sheet pickup bucket with the 1019+1020 sheet bucket.

Source buckets stay unchanged. Images are symlinked. Episode indices and the
global frame index are rewritten. Normalization is recomputed on the merged
train frames; the test bucket reuses those statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SHEET = Path("/vePFS/data/linglong/linglong_20261019_20261020_openwam")
NEW = Path("/vePFS/data/linglong/linglong_1023_openwam")
OUTPUT = Path("/vePFS/data/linglong/linglong_20261019_20261020_20261023_openwam")
PICKUP = "Pick up the sheet workpiece from the box."
PLACE = "Place the sheet workpiece into the upper shelf."
TASKS = (PICKUP, PLACE)
CAMERAS = ("head", "left", "right")
NUMERIC = ("observation.state", "action", "observation.state_eef", "action_eef")
WIDTH = {"observation.state": 24, "action": 24, "observation.state_eef": 16, "action_eef": 16}


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


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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


def load_bucket(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"{root} is not complete")
    if manifest.get("source_root") != ".":
        raise ValueError(f"{root} source_root must be '.'")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    mapping = read_json(root / "source_mapping.json")
    if not isinstance(mapping, list) or len(mapping) != len(episodes):
        raise ValueError(f"{root} source_mapping does not match episodes")
    for index, episode in enumerate(episodes):
        if int(episode["episode_index"]) != index or int(episode["length"]) < 33:
            raise ValueError(f"{root} episode {index} is unordered or shorter than 33 frames")
        if episode["tasks"] != [mapping[index]["prompt"]] or episode["tasks"][0] not in TASKS:
            raise ValueError(f"{root} episode {index} prompt is not the sheet pickup/place text")
    return {"root": root, "episodes": episodes, "mapping": mapping, "manifest": manifest}


def check_frame(table, label):
    for key, width in WIDTH.items():
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f"{label} {key} is not finite (*,{width})")
    for key in ("observation.state_eef", "action_eef"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        for start in (3, 11):
            norm = np.linalg.norm(values[:, start:start + 4], axis=1)
            if np.max(np.abs(norm - 1.0)) > 1e-3:
                raise ValueError(f"{label} {key} quaternion is not unit")


def rewrite(table, episode_index, index_start, task_index, image_prefix):
    count = table.num_rows
    columns = []
    names = []
    for name in table.column_names:
        names.append(name)
        if name == "episode_index":
            columns.append(pa.array(np.full(count, episode_index, dtype=np.int64), type=pa.int64()))
        elif name == "index":
            columns.append(pa.array(np.arange(index_start, index_start + count, dtype=np.int64), type=pa.int64()))
        elif name == "task_index":
            columns.append(pa.array(np.full(count, task_index, dtype=np.int64), type=pa.int64()))
        elif name.startswith("observation.images."):
            paths = table[name].to_pylist()
            rewritten = []
            for path in paths:
                if not isinstance(path, str) or not path.startswith("images/"):
                    raise ValueError(f"unexpected image path {path}")
                rewritten.append(f"images/{image_prefix}/{path[len('images/'):]}")
            columns.append(pa.array(rewritten, type=pa.string()))
        else:
            columns.append(table[name])
    return pa.Table.from_arrays(columns, names=names)


def write_split(buckets, output, split, train_stats):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"status": "incomplete", "split": split})
    images = output / "images"
    images.mkdir()
    for bucket, prefix in buckets:
        link = images / prefix
        link.symlink_to((bucket["root"] / "images").resolve(strict=True))

    episodes, mappings = [], []
    arrays = {key: [] for key in NUMERIC}
    task_counts = {task: 0 for task in TASKS}
    offset = 0
    index = 0
    for bucket, prefix in buckets:
        for episode, source in zip(bucket["episodes"], bucket["mapping"]):
            prompt = episode["tasks"][0]
            task_index = TASKS.index(prompt)
            path = bucket["root"] / "data" / f"chunk-{episode['episode_index'] // 1000:03d}" / f"episode_{episode['episode_index']:06d}.parquet"
            table = pq.read_table(path)
            if table.num_rows != episode["length"]:
                raise ValueError(f"{path} row count {table.num_rows} != {episode['length']}")
            check_frame(table, path)
            rewritten = rewrite(table, index, offset, task_index, prefix)
            for camera in CAMERAS:
                sample = rewritten[f"observation.images.{camera}"][0].as_py()
                if not (output / sample).is_file():
                    raise FileNotFoundError(f"missing image {output / sample}")
            dest = output / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(rewritten, dest, compression="zstd")
            for key in NUMERIC:
                arrays[key].append(np.asarray(table[key].to_pylist(), dtype=np.float64))
            episodes.append({
                "episode_index": index,
                "episode_id": episode["episode_id"],
                "length": episode["length"],
                "tasks": [prompt],
                "task_index": task_index,
            })
            mappings.append({
                **source,
                "episode_index": index,
                "input_bucket": str(bucket["root"]),
                "input_episode_index": episode["episode_index"],
                "image_prefix": prefix,
                "prompt": prompt,
                "split": split,
                "length": episode["length"],
            })
            task_counts[prompt] += 1
            offset += episode["length"]
            index += 1

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
        "codebase_version": "openwam_linglong_merged_20261019_20261020_20261023",
        "robot_type": "linglong_h",
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(TASKS),
        "fps": 15,
        "total_videos": 0,
        "total_chunks": (len(episodes) + 999) // 1000,
        "chunks_size": 1000,
        "splits": {split: f"0:{len(episodes)}"},
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
    })
    if "eef" not in train_stats:
        train_stats["episodes"] = len(episodes)
        train_stats["joint"] = {
            "state": stats(arrays["observation.state"], ddof=0),
            "action": stats(arrays["action"], ddof=0),
        }
        train_stats["eef"] = {
            "action_eef": stats(arrays["action_eef"], ddof=1),
            "observation.state_eef": stats(arrays["observation.state_eef"], ddof=1),
        }
    write_split_stats(output, train_stats)
    write_json(output / "source_mapping.json", mappings)
    write_json(output / f"{split}_episodes.json", list(range(len(episodes))))
    state_contract = buckets[0][0]["manifest"]["observation_state_eef"]
    action_contract = buckets[0][0]["manifest"]["action_eef"]
    windows = sum(episode["length"] - 32 for episode in episodes)
    write_json(output / "manifest.json", {
        "format": "openwam_linglong_merged_20261019_20261020_20261023",
        "status": "complete",
        "split": split,
        "source_root": ".",
        "prompt": PICKUP,
        "prompts": list(TASKS),
        "prompt_source": "1019/1020 sheet prompts plus 20261023 pickup; arm wording is absent",
        "fps": 15,
        "state_dim": 24,
        "action_dim": 24,
        "action_semantics": "absolute q_exp",
        "state_semantics": "measured q, no feedback repair",
        "layout": "left7,left_gripper,right7,right_gripper,waist4,head2,base2",
        "num_frames": 33,
        "video_stride": 4,
        "num_video_frames": 9,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_windows": windows,
        "image_storage": "symlinks to the source bucket image directories",
        "normalization": "recomputed on the merged train; test reuses those statistics",
        "normalization_dataset_dir": str(OUTPUT / "linglong_openwam_train_15hz"),
        "action_eef": {**action_contract, "normalization_split": "train", "normalization_stats": "meta/action_eef_stats.json"},
        "observation_state_eef": {
            **state_contract,
            "source": "mixed: FK(measured q) for 20261019; recorded /end_pos for 20261020 and 20261023",
            "normalization_split": "train",
            "normalization_stats": "meta/state_eef_stats.json",
        },
        "input_buckets": [str(bucket["root"]) for bucket, _prefix in buckets],
        "task_counts": task_counts,
    })
    return {"episodes": len(episodes), "frames": offset, "windows": windows, "tasks": task_counts, "arrays": arrays}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train_buckets = [
        (load_bucket(SHEET / "linglong_openwam_train_15hz"), "sheet"),
        (load_bucket(NEW / "linglong_openwam_train_15hz"), "1023"),
    ]
    test_buckets = [
        (load_bucket(SHEET / "linglong_openwam_test_15hz"), "sheet"),
        (load_bucket(NEW / "linglong_openwam_test_15hz"), "1023"),
    ]
    for bucket, _prefix in (train_buckets[1], test_buckets[1]):
        prompts = {episode["tasks"][0] for episode in bucket["episodes"]}
        if prompts != {PICKUP}:
            raise ValueError(f"20261023 prompts must be pickup only, got {prompts}")
    OUTPUT.mkdir()
    computed = {}
    train = write_split(train_buckets, OUTPUT / "linglong_openwam_train_15hz", "train", computed)
    test = write_split(test_buckets, OUTPUT / "linglong_openwam_test_15hz", "test", computed)
    write_json(OUTPUT / "merge.json", {
        "status": "complete",
        "prompts": list(TASKS),
        "train": {key: train[key] for key in ("episodes", "frames", "windows", "tasks")},
        "test": {key: test[key] for key in ("episodes", "frames", "windows", "tasks")},
    })
    print(json.dumps(read_json(OUTPUT / "merge.json"), ensure_ascii=False, indent=2))


def write_split_stats(output, train_stats):
    joint = {key: train_stats["joint"][key] for key in ("state", "action")}
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
            "statistics_source": "merged train frames once; test is not included",
            "num_episodes": train_stats["episodes"],
            "num_transition": payload["count"],
            "std_ddof": 1,
            stream: {"default": {f"global_{key}": value for key, value in payload.items() if key != "count"}},
        })


if __name__ == "__main__":
    main()
