"""Publish 2026-10-26 no-target Piper-X episodes as virtual static-hold windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openwam.dataloader.tos_cache import TosBackend, parse_tos_uri
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d
from openwam.dataloader.utils.normalization import pin_rot6d_identity, ROT6D_DIMS_EEF20
from scripts.piper.convert_piper_x import _contracts, _fixed_list, _sha256, _stats, build_episode
from scripts.piper.kinematics import PiperXKinematics

BUCKET = "ml-platform-auto-created-required-2109640594-cn-beijing"
BUNDLE_PREFIX = "piperx/processed/bundles"
ARCHIVES = [
    "cleaned_20261026_135701_897423",
    "cleaned_20261026_150114_673305",
    "cleaned_20261026_195908_777293",
    "cleaned_20261026_skew36_recovery",
]
OLD_TRAIN = [
    "tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261022_eef_v1/train",
    "tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261023_eef_v1/train",
    "tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261024_eef_v1/train",
]
DESTINATION = "tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261026_static_eef_v1"
PROMPT = "Use the right hand to continuously place the objects on the table into the container on the table."
SPLIT_SEED = 0
TEST_FRACTION = 0.1
NUM_FRAMES = 33
ACTION_STEPS = NUM_FRAMES - 1
CAMERAS = (("front", "head"), ("left_wrist", "left"), ("right_wrist", "right"))
STATE_COLUMN = "observation.state_eef"
ACTION_COLUMN = "action_eef"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def retry(function):
    for attempt in range(6):
        try:
            return function()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(12, 2**attempt))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_eef20(values):
    values = np.asarray(values, dtype=np.float64).copy()
    result = []
    for start in (0, 8):
        q = values[:, start + 3:start + 7]
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        result.extend((values[:, start:start + 3], quat_wxyz_to_rot6d(q), values[:, start + 7:start + 8]))
    return np.concatenate(result, axis=-1).astype(np.float32)


def vector_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"invalid EEF20 normalization input {values.shape}")
    stats = {
        "mean": values.mean(0), "std": values.std(0, ddof=1),
        "min": values.min(0), "max": values.max(0),
        "q01": np.quantile(values, 0.01, axis=0), "q99": np.quantile(values, 0.99, axis=0),
    }
    for key in stats:
        stats[key] = stats[key].astype(np.float32)
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    return {key: value.tolist() for key, value in stats.items()}


def safe_relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or "\\" in str(value) or not path.parts:
        raise ValueError(f"unsafe archive path: {value!r}")
    return path


class StaticPublisher:
    def __init__(self, destination, work, workers):
        self.client = TosBackend().client
        self.bucket, self.prefix = parse_tos_uri(destination)
        self.destination = destination.rstrip("/")
        self.work = Path(work).expanduser().resolve()
        if not str(self.work).startswith("/tmp/"):
            raise ValueError("raw archives and conversion staging must stay under /tmp")
        self.work.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.work / "archives"
        self.archive_dir.mkdir(exist_ok=True)
        self.workers = int(workers)
        self.fk = PiperXKinematics()

    def get_at(self, bucket, key):
        try:
            return self.client.get_object(bucket, key).read()
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def get(self, relative):
        return self.get_at(self.bucket, self.prefix + "/" + relative)

    def put(self, relative, data):
        retry(lambda: self.client.put_object(self.bucket, self.prefix + "/" + relative, content=data))

    def upload(self, pair):
        relative, path = pair
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        result = retry(lambda: self.client.put_object_from_file(self.bucket, self.prefix + "/" + relative, str(path)))
        if result.etag.strip('"') != digest:
            raise IOError(f"TOS ETag mismatch for {relative}")
        return {"key": relative, "bytes": path.stat().st_size, "md5": digest}

    def download_archive(self, key, expected_bytes):
        source_key = f"{BUNDLE_PREFIX}/{key}.tar"
        target = self.archive_dir / f"{key}.tar"
        head = self.client.head_object(BUCKET, source_key)
        if int(head.content_length) != int(expected_bytes):
            raise ValueError(f"TAR size mismatch for {key}")
        if target.exists() and target.stat().st_size == expected_bytes:
            return target
        free = shutil.disk_usage(self.work).free
        if free < expected_bytes + 1024**3:
            raise RuntimeError(f"insufficient /tmp space to download {key}: free={free}, required={expected_bytes + 1024**3}")
        partial = target.with_suffix(".partial")
        def once():
            response = self.client.get_object(BUCKET, source_key, if_match=head.etag)
            with partial.open("wb") as out:
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
            if partial.stat().st_size != expected_bytes:
                raise IOError(f"incomplete TAR download for {key}")
        retry(once)
        partial.replace(target)
        return target

    def inspect_archive(self, key, sidecar, archive_path):
        if sha256_file(archive_path) != sidecar["sha256"]:
            raise ValueError(f"TAR SHA256 mismatch: {key}")
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            files = [item for item in members if item.isfile()]
            if len(files) != int(sidecar["files"]) + 1:
                raise ValueError(f"unexpected TAR member count for {key}: {len(files)} files, sidecar={sidecar['files']}")
            by_name = {item.name: item for item in files}
            if len(by_name) != len(files):
                raise ValueError(f"duplicate TAR member names in {key}")
            bundle_name = f"{key}/bundle_manifest.json"
            bundle_member = by_name.get(bundle_name)
            if bundle_member is None:
                raise ValueError(f"missing bundle manifest in {key}")
            bundle_bytes = archive.extractfile(bundle_member).read()
            if hashlib.sha256(bundle_bytes).hexdigest() != sidecar["bundle_manifest_sha256"]:
                raise ValueError(f"bundle manifest SHA256 mismatch in {key}")
            bundle = json.loads(bundle_bytes)
            episodes = bundle.get("episodes")
            if not isinstance(episodes, dict) or len(episodes) != int(sidecar["episodes"]):
                raise ValueError(f"unexpected episode count in {key}")
            report = []
            total_frames = total_images = 0
            for episode_name, episode_meta in sorted(episodes.items()):
                folder = safe_relative(episode_meta["folder"])
                if folder.parts[0] != "episodes":
                    raise ValueError(f"invalid episode folder in {key}: {folder}")
                prefix = f"{key}/{folder.as_posix()}/"
                manifest_member = by_name.get(prefix + "manifest.json")
                samples_member = by_name.get(prefix + "samples.jsonl")
                if manifest_member is None or samples_member is None:
                    raise ValueError(f"missing episode metadata in {key}/{episode_name}")
                manifest = json.loads(archive.extractfile(manifest_member).read())
                sample_bytes = archive.extractfile(samples_member).read()
                rows = [json.loads(line) for line in sample_bytes.splitlines() if line.strip()]
                if manifest.get("fps") != 15.0 or len(rows) != int(manifest.get("samples", -1)):
                    raise ValueError(f"invalid fps/sample count for {key}/{episode_name}")
                if not rows or len({str(row["sample_id"]) for row in rows}) != len(rows):
                    raise ValueError(f"empty episode or duplicate sample ids for {key}/{episode_name}")
                image_names = set()
                for row in rows:
                    for camera, _ in CAMERAS:
                        rel = safe_relative(row["images"][camera])
                        member_name = prefix + rel.as_posix()
                        if member_name not in by_name:
                            raise FileNotFoundError(f"missing image in TAR: {member_name}")
                        image_names.add(member_name)
                if len(image_names) != len(rows) * len(CAMERAS):
                    raise ValueError(f"image paths are reused within {key}/{episode_name}")
                report.append({
                    "archive": key + ".tar", "folder": folder.as_posix(), "episode": str(episode_name),
                    "frames": len(rows), "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
                })
                total_frames += len(rows)
                total_images += len(image_names)
            jpg_count = sum(1 for name in by_name if name.lower().endswith((".jpg", ".jpeg")) and "/episodes/" in name)
            if jpg_count != total_images:
                raise ValueError(f"JPEG count mismatch in {key}: tar={jpg_count}, manifests={total_images}")
            return report, len(files), total_frames, total_images

    def make_inventory_and_split(self):
        inventories = []
        for key in ARCHIVES:
            sidecar = json.loads(self.get_at(BUCKET, f"{BUNDLE_PREFIX}/{key}.json"))
            if sidecar.get("schema") != "dual_arm_clean_bundle_upload.v1" or sidecar.get("date") != "2026-10-26":
                raise ValueError(f"invalid upload sidecar for {key}")
            if sidecar.get("archive") != key + ".tar":
                raise ValueError(f"sidecar archive name mismatch for {key}")
            archive_path = self.download_archive(key, sidecar["bytes"])
            entries, files, frames, images = self.inspect_archive(key, sidecar, archive_path)
            if len(entries) != sidecar["episodes"] or files != sidecar["files"] + 1:
                raise ValueError(f"archive inventory mismatch for {key}")
            inventories.extend(entries)
            print(json.dumps({"archive": key, "episodes": len(entries), "frames": frames, "jpegs": images}), flush=True)
        if len(inventories) != 38:
            raise ValueError(f"expected 38 new episodes, got {len(inventories)}")
        ids = [(x["archive"], x["folder"]) for x in inventories]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate archive episode identities")
        order = np.random.default_rng(SPLIT_SEED).permutation(len(inventories))
        n_test = max(1, round(len(inventories) * TEST_FRACTION))
        test = {int(i) for i in order[:n_test]}
        entries = []
        counters = {"train": 0, "test": 0}
        for i, item in enumerate(inventories):
            split = "test" if i in test else "train"
            index = counters[split]
            counters[split] += 1
            entries.append({**item, "split": split, "episode_index": index})
        plan = {
            "schema": "piper_x_static_hold_split.v1", "date": "2026-10-26", "seed": SPLIT_SEED,
            "test_fraction": TEST_FRACTION, "split_method": "seeded episode permutation across all four archives; no label stratification",
            "prompt": PROMPT, "archives": inventories, "entries": entries,
            "counts": {split: sum(x["split"] == split for x in entries) for split in ("train", "test")},
            "frames": {split: sum(x["frames"] for x in entries if x["split"] == split) for split in ("train", "test")},
        }
        if plan["counts"] != {"train": 34, "test": 4}:
            raise AssertionError(plan["counts"])
        return plan

    def extract_episode(self, archive, entry, stage):
        shutil.rmtree(stage, ignore_errors=True)
        folder = safe_relative(entry["folder"])
        root = stage / folder.name
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"{Path(archive).stem}/{folder.as_posix()}/"
        with tarfile.open(self.archive_dir / archive, "r:*") as tar:
            for member in tar.getmembers():
                name = PurePosixPath(member.name)
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                rel = safe_relative(name.relative_to(PurePosixPath(prefix)))
                destination = root.joinpath(*rel.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        return root

    @staticmethod
    def clamp_gripper_encoder_noise(segment):
        path = segment / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        changed = 0
        for row in rows:
            for side in ("left", "right"):
                for field in (row["arms"][side]["gripper"], row["actions"][side]):
                    key = "width" if "width" in field else "position"
                    value = float(field[key] if key == "width" else field[key][6])
                    if value < -0.001 or value > 0.101:
                        raise ValueError(f"gripper range violation: {side}={value}")
                    if value < 0:
                        if key == "width": field[key] = 0.0
                        else: field[key][6] = 0.0
                        changed += 1
        if changed:
            path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        return rows, changed

    def process_episode(self, entry, plan):
        split, index = entry["split"], entry["episode_index"]
        marker = f"episodes/{split}_{index:06d}.json"
        old = self.get(marker)
        if old is not None:
            summary = json.loads(old)
            if summary["samples_sha256"] != entry["samples_sha256"] or summary["frames"] != entry["frames"]:
                raise ValueError(f"existing progress marker conflicts for {marker}")
            print(json.dumps({"resume": marker}), flush=True)
            return summary
        stage = self.work / "episode"
        segment = self.extract_episode(entry["archive"], entry, stage)
        if _sha256(segment / "samples.jsonl") != entry["samples_sha256"]:
            raise ValueError(f"sample hash mismatch after extracting {entry['episode']}")
        rows, gripper_clamps = self.clamp_gripper_encoder_noise(segment)
        record = build_episode(segment, self.fk, PROMPT, {"prompt": PROMPT, "split": split}, fk_mismatch_policy="record")
        # No-target data is a stationary policy: targets are the measured EEF state, not recorded motion commands.
        record["action"] = record["state"].copy()
        n = len(rows)
        image_paths = {}
        jobs = []
        for source_camera, output_camera in CAMERAS:
            paths = []
            sources = record["image_sources"][output_camera]
            for sample_id, source in zip(record["sample_ids"], sources):
                with Image.open(source) as image:
                    if image.mode != "RGB":
                        raise ValueError(f"non-RGB image: {source}")
                    image.verify()
                relative = f"images/episode_{index:06d}/{output_camera}/{sample_id}.jpg"
                paths.append(relative)
                jobs.append((f"{split}/{relative}", source))
            if len(paths) != n or len(set(paths)) != n:
                raise ValueError(f"invalid image list for {entry['episode']}/{output_camera}")
            image_paths[output_camera] = paths
        prior_frames = sum(x["frames"] for x in plan["entries"] if x["split"] == split and x["episode_index"] < index)
        values = {
            STATE_COLUMN: _fixed_list(record["state"], 16),
            ACTION_COLUMN: _fixed_list(record["action"], 16),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / 15.0),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, index, dtype=np.int64)),
            "index": pa.array(np.arange(prior_frames, prior_frames + n, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            **{f"observation.images.{camera}": pa.array(paths) for camera, paths in image_paths.items()},
        }
        parquet = stage / "episode.parquet"
        pq.write_table(pa.table(values), parquet, compression="zstd")
        jobs.append((f"{split}/data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet", parquet))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            objects = list(pool.map(self.upload, jobs))
        summary = {
            "episode": entry["episode"], "archive": entry["archive"], "folder": entry["folder"],
            "split": split, "episode_index": index, "samples_sha256": entry["samples_sha256"],
            "frames": n, "static_windows": n, "fk_report": record["fk_report"],
            "gripper_encoder_noise_clamps": gripper_clamps,
            "action_policy": "each row action_eef equals measured observation.state_eef; reader repeats it for 32 predicted steps",
            "video_policy": "reader repeats each row's three camera images across all 9 video positions",
            "objects": len(objects), "bytes": sum(item["bytes"] for item in objects),
        }
        self.put(f"objects/{split}_{index:06d}.json", json_bytes(objects))
        self.put(marker, json_bytes(summary))
        print(json.dumps({"episode": entry["episode"], "split": split, "frames": n, "bytes": summary["bytes"]}), flush=True)
        shutil.rmtree(stage, ignore_errors=True)
        return summary

    def old_train_arrays(self):
        def get_episode(root, index, expected_length):
            bucket, prefix = parse_tos_uri(root)
            key = f"{prefix}/data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
            data = retry(lambda: self.client.get_object(bucket, key).read())
            table = pq.read_table(pa.BufferReader(data), columns=[STATE_COLUMN, ACTION_COLUMN])
            if len(table) != expected_length:
                raise ValueError(f"old train parquet length mismatch: {key}")
            state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
            action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
            return root, index, state, action

        states, actions = [], []
        counts = {}
        for root in OLD_TRAIN:
            bucket, prefix = parse_tos_uri(root)
            payload = self.client.get_object(bucket, prefix + "/meta/episodes.jsonl").read().decode()
            episodes = [json.loads(line) for line in payload.splitlines() if line.strip()]
            with ThreadPoolExecutor(max_workers=min(24, self.workers)) as pool:
                futures = [pool.submit(get_episode, root, i, int(item["length"])) for i, item in enumerate(episodes)]
                records = sorted((future.result() for future in futures), key=lambda x: x[1])
            states.extend(item[2] for item in records)
            actions.extend(item[3] for item in records)
            counts[root] = {"episodes": len(episodes), "frames": sum(int(x["length"]) for x in episodes)}
            print(json.dumps({"normalization_source": root, **counts[root]}), flush=True)
        return states, actions, counts

    @staticmethod
    def stats16_payload(states, actions, source_text, num_episodes, num_unique_frames, action_targets):
        payload = {"normalization_stats.json": json_bytes({
            "action": _stats(actions), "state": _stats(states),
            "source": source_text, "test_excluded": True,
        })}
        for stream, column, arrays in (("action", ACTION_COLUMN, actions), ("state", STATE_COLUMN, states)):
            stats = _stats(arrays, ddof=1)
            payload[f"{stream}_eef_stats.json"] = json_bytes({
                "source_column": column,
                "statistics_source": source_text,
                "num_episodes": num_episodes,
                "num_transition": action_targets if stream == "action" else num_unique_frames,
                "std_ddof": 1,
                stream: {"default": {f"global_{key}": value for key, value in stats.items() if key != "count"}},
            })
        return payload

    def publish_metadata(self, plan, summaries):
        summaries_by_key = {(x["archive"], x["folder"]): x for x in summaries}
        old_states, old_actions, old_counts = self.old_train_arrays()
        train_entries = [x for x in plan["entries"] if x["split"] == "train"]
        new_state, new_action = [], []
        for entry in train_entries:
            key = f"train/data/chunk-{entry['episode_index'] // 1000:03d}/episode_{entry['episode_index']:06d}.parquet"
            data = self.get(key)
            if data is None:
                raise FileNotFoundError(key)
            table = pq.read_table(pa.BufferReader(data), columns=[STATE_COLUMN, ACTION_COLUMN])
            if len(table) != entry["frames"]:
                raise ValueError(f"new train parquet length mismatch: {key}")
            state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
            action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
            if not np.array_equal(state, action):
                raise ValueError(f"static row action differs from state: {key}")
            new_state.append(state)
            new_action.append(action)
        state_arrays = old_states + new_state
        # Keep the existing source-row statistics convention for every dataset.
        # Repeating a target within a window does not duplicate its source row.
        action_arrays = old_actions + new_action
        unique_frames = sum(map(len, old_states)) + sum(map(len, new_state))
        action_targets = sum(map(len, action_arrays))
        num_episodes = sum(v["episodes"] for v in old_counts.values()) + len(train_entries)
        old_train_uris = list(OLD_TRAIN)
        source_text = "20261022/23/24 Piper-X train rows plus 20261026 static train rows; each source row counted once; all test episodes excluded"
        stats16 = self.stats16_payload(state_arrays, action_arrays, source_text, num_episodes, unique_frames, action_targets)
        packed_action = pack_eef20(np.concatenate(action_arrays, axis=0))
        packed_state = pack_eef20(np.concatenate(state_arrays, axis=0))
        payload20 = {
            "format": "piper_x_eef20_normalization_v1",
            "sources": old_train_uris + [self.destination + "/train"],
            "num_episodes": num_episodes,
            "num_frames": unique_frames,
            "num_action_targets": action_targets,
            "rotation_normalization": "identity",
            "test_excluded": True,
            "statistics_weighting": {"old_train_action": "one per source row", "static_train_action": "one per source row", "state": "one per source row"},
            "action": vector_stats(packed_action), "state": vector_stats(packed_state),
        }
        norm_prefix = "normalization/meta/"
        self.put(norm_prefix + "eef20_normalization_stats.json", json_bytes(payload20))
        for name, data in stats16.items():
            self.put(norm_prefix + name, data)
        normalization_report = {
            "sources": old_counts | {self.destination + "/train": {"episodes": len(train_entries), "frames": sum(x["frames"] for x in train_entries)}},
            "unique_train_frames": unique_frames, "action_target_rows": action_targets,
            "train_episodes": num_episodes, "test_excluded": True, "weighting": payload20["statistics_weighting"],
        }
        self.put("normalization/publication.json", json_bytes(normalization_report))

        for split in ("train", "test"):
            entries = sorted((x for x in plan["entries"] if x["split"] == split), key=lambda x: x["episode_index"])
            if [x["episode_index"] for x in entries] != list(range(len(entries))):
                raise ValueError(f"non-contiguous {split} episode indices")
            episodes = [{"episode_index": x["episode_index"], "episode_id": x["episode"], "length": x["frames"], "tasks": [PROMPT], "task_index": 0} for x in entries]
            frames = sum(x["frames"] for x in entries)
            windows = frames
            for name, data in stats16.items():
                self.put(f"{split}/meta/{name}", data)
            self.put(f"{split}/meta/eef20_normalization_stats.json", json_bytes(payload20))
            self.put(f"{split}/meta/episodes.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in episodes).encode())
            self.put(f"{split}/meta/tasks.jsonl", (json.dumps({"task_index": 0, "task": PROMPT}, ensure_ascii=False) + "\n").encode())
            self.put(f"{split}/meta/episodes_stats.jsonl", "".join(json.dumps({"episode_index": x["episode_index"], "length": x["length"]}) + "\n" for x in episodes).encode())
            selected = [summaries_by_key[(x["archive"], x["folder"])] for x in entries]
            self.put(f"{split}/source_mapping.json", json_bytes(selected))
            features = {STATE_COLUMN: {"dtype": "float32", "shape": [16]}, ACTION_COLUMN: {"dtype": "float32", "shape": [16]}}
            features.update({f"observation.images.{camera}": {"dtype": "image"} for _, camera in CAMERAS})
            self.put(f"{split}/meta/info.json", json_bytes({
                "codebase_version": "openwam_piper_x_eef_v1", "robot_type": "piper_x",
                "total_episodes": len(entries), "total_frames": frames, "total_tasks": 1, "fps": 15,
                "splits": {split: f"0:{len(entries)}"}, "features": features,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }))
            manifest = {
                "format": "openwam_piper_x_eef_v1", "status": "complete", "split": split,
                "source_root": ".", "prompt": PROMPT, "fps": 15, "num_frames": NUM_FRAMES,
                "video_stride": 4, "total_episodes": len(entries), "total_frames": frames,
                "total_windows": windows, "static_hold": True,
                "static_hold_policy": "one training window per source frame; repeat that frame's three images across 9 video slots and measured EEF state across 32 action targets",
                "image_storage": "independent JPEG objects in this bucket; each source image stored once",
                "normalization": "20261022/23/24 train rows plus 20261026 static train rows; tests excluded",
                "normalization_source": self.destination + "/normalization",
                "eef20_stats": "meta/eef20_normalization_stats.json",
                "active_raw_action_dims": list(range(16)), "active_raw_state_dims": list(range(16)),
                "single_episode_reused_for_test": False, "split_seed": SPLIT_SEED,
                "test_fraction": TEST_FRACTION,
                "split_method": "seeded episode permutation across all 2026-10-26 archives; no label stratification",
                "fk_mismatch_policy": "record", "quality_filter": False, "task_stratification": False,
                "urdf_sha256": _sha256(self.fk.urdf_path), "tos_immutable": True,
                **_contracts(self.fk),
            }
            manifest["action_eef"]["source"] = "measured observation state copied as a stationary action target"
            manifest["action_eef"]["kinematics_used"] = False
            manifest["observation_state_eef"]["source"] = "recorded pose; gripper width mapped by the URDF upper limit"
            self.put(f"{split}/manifest.json", json_bytes(manifest))
        report = {
            "status": "complete", "destination": self.destination,
            "splits": {split: {"episodes": plan["counts"][split], "frames": plan["frames"][split], "static_windows": plan["frames"][split]} for split in ("train", "test")},
            "normalization": normalization_report,
            "fk_exceptions": [
                {"episode": item["episode"], "split": item["split"], "side": side, **fk_report}
                for item in summaries for side, fk_report in item["fk_report"].items()
                if fk_report["position_error_mm_max"] > 12 or fk_report["orientation_error_deg_max"] > 4
            ],
            "gripper_encoder_noise_clamps": sum(x["gripper_encoder_noise_clamps"] for x in summaries),
            "objects": sum(x["objects"] for x in summaries), "bytes": sum(x["bytes"] for x in summaries),
        }
        self.put("publication.json", json_bytes(report))
        (self.work / "publication.json").write_bytes(json_bytes(report))
        print(json.dumps(report), flush=True)

    def run(self):
        if self.get("publication.json") is not None:
            raise ValueError("destination already has a complete publication")
        plan = self.make_inventory_and_split()
        encoded = json_bytes(plan)
        old_plan = self.get("split.json")
        if old_plan is not None and old_plan != encoded:
            raise ValueError("existing split plan differs; refusing to mix dataset versions")
        self.put("split.json", encoded)
        summaries = []
        for archive in sorted({x["archive"] for x in plan["entries"]}):
            for entry in [x for x in plan["entries"] if x["archive"] == archive]:
                summaries.append(self.process_episode(entry, plan))
            (self.archive_dir / archive).unlink(missing_ok=True)
        if len(summaries) != 38:
            raise ValueError(f"incomplete conversion: {len(summaries)} episodes")
        expected = {x["episode"] for x in plan["entries"]}
        if {x["episode"] for x in summaries} != expected:
            raise ValueError("published episode identities are incomplete")
        self.publish_metadata(plan, summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--work", default="/tmp/piperx-static-20261026")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    StaticPublisher(args.destination, args.work, args.workers).run()


if __name__ == "__main__":
    main()
