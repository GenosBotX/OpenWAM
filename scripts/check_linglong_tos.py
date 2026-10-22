"""Compare actual TOS samples against the original local Linglong EEF bucket."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from openwam.dataloader.linglong import LinglongDataset


def check(local, remote, cache_dir, max_bytes, samples=8):
    shared = dict(state_key="observation.state_eef", action_key="action_eef", action_mode="eef_quat",
                  unify_action_map=["0-6", 9, "34-40", 43], unify_state_map=["0-6", 9, "34-40", 43], window_stride=4)
    a = LinglongDataset(local, **shared)
    b = LinglongDataset(remote, cache_dir=cache_dir, cache_max_bytes=max_bytes, **shared)
    indices = np.linspace(0, len(b) - 1, min(samples, len(b)), dtype=int).tolist()
    started = time.monotonic()
    expected_samples = [a[index] for index in indices]
    local_seconds = round(time.monotonic() - started, 3)
    timings = []
    for phase in ("cold", "repeat"):
        started = time.monotonic()
        for index, expected in zip(indices, expected_samples):
            got = b[index]
            # The TOS fixture preserves the first N complete episodes.
            for key in ("action", "proprio", "action_mask", "proprio_mask", "video_mask"):
                assert torch.equal(expected[key], got[key]), key
            for x, y in zip(expected["video"], got["video"]):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            assert expected["prompt"] == got["prompt"]
        timings.append({"phase": phase, "seconds": round(time.monotonic() - started, 3), "samples": len(indices)})
    # Exercise a real worker pool with the same cache object (fork on Linux).
    started = time.monotonic()
    loader = DataLoader(torch.utils.data.Subset(b, indices), batch_size=None, num_workers=2)
    assert sum(1 for _ in loader) == len(indices)
    result = {"equal": True, "samples": len(indices), "remote_windows": len(b), "local_seconds": local_seconds, "timings": timings,
              "worker_seconds": round(time.monotonic() - started, 3), "cache": b.tos_cache.stats()}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-bytes", type=int, default=8388608)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = check(args.local, args.remote, args.cache_dir, args.max_bytes, args.samples)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
