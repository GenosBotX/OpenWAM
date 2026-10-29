"""Compute Piper-X relative EEF20 action statistics from TOS train buckets.

Only parquet state/action columns are read. Test buckets are never included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.piper_x import _pack_eef20
from openwam.dataloader.tos_cache import TosBackend, parse_tos_uri
from openwam.dataloader.utils.relative_eef import (
    RELATIVE_EEF_CONTRACT, RELATIVE_SEMANTICS, RELATIVE_STATS_FORMAT,
    absolute_to_relative_eef20,
)


class Reservoir:
    def __init__(self, size=200_000, dimensions=20, seed=0):
        self.size, self.rng = int(size), np.random.default_rng(seed)
        self.values = np.empty((self.size, dimensions), dtype=np.float32)
        self.count = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1, self.values.shape[1])
        count = len(values)
        start = self.count
        initial = max(0, min(count, self.size - start))
        if initial:
            self.values[start:start + initial] = values[:initial]
        remaining = values[initial:]
        if len(remaining):
            seen = np.arange(start + initial + 1, start + count + 1, dtype=np.int64)
            slots = self.rng.integers(0, seen, size=len(remaining))
            selected = np.flatnonzero(slots < self.size)
            if len(selected):
                destinations = slots[selected]
                reverse = destinations[::-1]
                _, reverse_indices = np.unique(reverse, return_index=True)
                last = len(destinations) - 1 - reverse_indices
                self.values[destinations[last]] = remaining[selected[last]]
        self.count += count

    def sample(self):
        return self.values[:min(self.count, self.size)]


class Stats:
    def __init__(self):
        self.count = 0
        self.sum = np.zeros(20, dtype=np.float64)
        self.sum2 = np.zeros(20, dtype=np.float64)
        self.minimum = np.full(20, np.inf, dtype=np.float64)
        self.maximum = np.full(20, -np.inf, dtype=np.float64)
        self.reservoir = Reservoir()

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 20)
        if not len(values):
            return
        self.count += len(values)
        self.sum += values.sum(0)
        self.sum2 += np.square(values).sum(0)
        self.minimum = np.minimum(self.minimum, values.min(0))
        self.maximum = np.maximum(self.maximum, values.max(0))
        self.reservoir.update(values)

    def payload(self, pin_rot6d=True):
        if self.count < 2:
            raise ValueError("relative action statistics need at least two rows")
        mean = self.sum / self.count
        variance = np.maximum((self.sum2 - self.count * np.square(mean)) / (self.count - 1), 0.0)
        values = self.reservoir.sample()
        result = {
            "count": int(self.count),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "q01": np.quantile(values, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(values, 0.99, axis=0).astype(np.float32),
        }
        if pin_rot6d:
            for key, value in (("mean", 0), ("std", 1), ("min", -1), ("max", 1), ("q01", -1), ("q99", 1)):
                result[key][[3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]] = value
        return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}


def read(client, uri):
    bucket, key = parse_tos_uri(uri)
    return client.get_object(bucket, key).read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", action="append")
    parser.add_argument("--config-name", help="Training config providing the full list of train sources")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--upload", help="TOS URI for relative_eef20_normalization_stats.json")
    args = parser.parse_args()
    if args.config_name:
        if args.train_source:
            raise ValueError("Use --config-name or --train-source, not both")
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None):
            config = compose(config_name=args.config_name)
        args.train_source = list(config.dataloader.dataset_dir)
        if (int(config.dataloader.num_frames), int(config.dataloader.window_stride)) != (args.num_frames, args.window_stride):
            raise ValueError("Statistics window settings must match the selected training config")
    if not args.train_source:
        raise ValueError("Specify --config-name or --train-source")
    if args.num_frames != 33 or args.window_stride != 1:
        raise ValueError("Piper-X relative stats currently require 33-frame stride-1 windows")
    if len(set(args.train_source)) != len(args.train_source):
        raise ValueError("Duplicate training sources")
    client = TosBackend().client
    action_stats, state_stats = Stats(), Stats()
    source_meta = {}
    for source in args.train_source:
        manifest = json.loads(read(client, source + "/manifest.json"))
        if manifest.get("split") != "train" or manifest.get("status") != "complete":
            raise ValueError(f"relative stats source is not train: {source}")
        episodes = [json.loads(line) for line in read(client, source + "/meta/episodes.jsonl").splitlines() if line.strip()]
        source_frames = 0
        source_windows = 0
        source_digest = hashlib.sha256()
        for episode in episodes:
            index = int(episode["episode_index"])
            data = read(client, source + f"/data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet")
            source_digest.update(data)
            table = pq.read_table(pa.BufferReader(data), columns=["observation.state_eef", "action_eef"])
            state16 = np.asarray(table["observation.state_eef"].to_pylist(), dtype=np.float32)
            action16 = np.asarray(table["action_eef"].to_pylist(), dtype=np.float32)
            if (len(state16) != int(episode["length"]) or state16.shape != (len(state16), 16)
                    or action16.shape != state16.shape or not np.isfinite(state16).all()
                    or not np.isfinite(action16).all()):
                raise ValueError(f"Invalid state/action table: {source} episode {index}")
            state = _pack_eef20(state16)
            action = _pack_eef20(action16)
            state_stats.update(state)
            source_frames += len(state)
            static_hold = bool(manifest.get("static_hold", False))
            if static_hold and not np.array_equal(state16, action16):
                raise ValueError("Static preprocessing must store action identical to state")
            extent = 1 if static_hold else args.num_frames
            starts = np.arange(0, max(0, len(state) - extent + 1), args.window_stride)
            for begin in range(0, len(starts), 512):
                batch = starts[begin:begin + 512]
                base = state[batch, None, :]
                target = (np.repeat(base, args.num_frames - 1, axis=1) if static_hold
                          else action[batch[:, None] + np.arange(args.num_frames - 1)])
                action_stats.update(absolute_to_relative_eef20(target, base))
                source_windows += len(batch)
        source_meta[source] = {"episodes": len(episodes), "frames": source_frames,
                               "windows": source_windows, "parquet_sha256": source_digest.hexdigest()}
        print(json.dumps({"source": source, **source_meta[source]}), flush=True)
    payload = {
        "format": RELATIVE_STATS_FORMAT,
        "test_excluded": True,
        "contract": RELATIVE_EEF_CONTRACT,
        **RELATIVE_SEMANTICS,
        "num_frames": args.num_frames,
        "window_stride": args.window_stride,
        "sources": source_meta,
        "train_episodes": sum(item["episodes"] for item in source_meta.values()),
        "train_frames": sum(item["frames"] for item in source_meta.values()),
        "train_windows": sum(item["windows"] for item in source_meta.values()),
        "quantiles": "deterministic uniform reservoir, capacity 200000, seed 0; min/max exact",
        "action": action_stats.payload(),
        "state": state_stats.payload(pin_rot6d=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.upload:
        bucket, key = parse_tos_uri(args.upload)
        client.put_object(bucket, key, content=args.output.read_bytes())
        if read(client, args.upload) != args.output.read_bytes():
            raise RuntimeError("TOS statistics upload verification failed")
    print(json.dumps({"output": str(args.output), "upload": args.upload, "train_episodes": payload["train_episodes"], "train_frames": payload["train_frames"], "train_windows": payload["train_windows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
