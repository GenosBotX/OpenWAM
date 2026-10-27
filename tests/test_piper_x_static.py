"""Static samples repeat one observation without leaking adjacent frames."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest
import torch

from openwam.dataloader.piper_x import PiperXDataset, STATE_COLUMN, ACTION_COLUMN


def make_bucket(root, static, lengths, bad_action=False):
    (root / "meta").mkdir(parents=True)
    manifest = {
        "format": "openwam_piper_x_eef_v1", "status": "complete", "split": "train",
        "prompt": "Hold position.", "static_hold": static,
        "observation_state_eef": {"column": STATE_COLUMN, "shape": [16], "quaternion_order": "wxyz", "reference": "base_link"},
        "action_eef": {"column": ACTION_COLUMN, "shape": [16], "quaternion_order": "wxyz", "reference": "base_link"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    episodes = []
    for episode, length in enumerate(lengths):
        state = np.zeros((length, 16), dtype=np.float32)
        state[:, (3, 11)] = 1
        state[:, 0] = np.arange(length) / 100 + episode
        action = state.copy()
        if bad_action:
            action[-1, 0] += .1
        values = {STATE_COLUMN: state.tolist(), ACTION_COLUMN: action.tolist()}
        for camera, offset in (("head", 0), ("left", 50), ("right", 100)):
            paths = []
            for frame in range(length):
                rel = f"images/{episode}_{camera}_{frame}.png"
                path = root / rel
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (12, 12), (frame + offset, episode, 0)).save(path)
                paths.append(rel)
            values[f"observation.images.{camera}"] = paths
        path = root / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(values), path)
        episodes.append({"episode_index": episode, "length": length})
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episodes))
    stats = {}
    for key, value in (("mean", 0), ("std", 1), ("min", -1), ("max", 1), ("q01", -1), ("q99", 1)):
        stats[key] = [value] * 20
    (root / "meta/eef20_normalization_stats.json").write_text(json.dumps({
        "format": "piper_x_eef20_normalization_v1", "test_excluded": True,
        "action": stats, "state": stats,
    }))
    return root


def reader(root):
    return PiperXDataset(str(root), action_mode="eef", height=32, width=32, normalize_mode=None)


def test_static_covers_every_frame_and_episode_boundary(tmp_path):
    dataset = reader(make_bucket(tmp_path / "static", True, [3, 2]))
    assert len(dataset) == 5
    assert dataset._windows == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    first, last, next_episode = dataset[0], dataset[2], dataset[3]
    assert last["start_frame"] == 2
    assert next_episode["episode_index"] == 1
    assert next_episode["action"][0, 0] == 1
    assert not torch.equal(first["action"], last["action"])
    for sample in (first, last, next_episode):
        assert sample["action"].shape == (32, 80)
        assert torch.equal(sample["action"], sample["proprio"].expand(32, -1))
        assert sample["action_mask"].sum().item() == 32 * 20
        assert len(sample["video"]) == 9
        assert all(np.array_equal(np.asarray(frame), np.asarray(sample["video"][0])) for frame in sample["video"])
    assert not np.array_equal(np.asarray(first["video"][0]), np.asarray(last["video"][0]))


def test_static_rejects_moving_target(tmp_path):
    dataset = reader(make_bucket(tmp_path / "bad", True, [2], bad_action=True))
    with pytest.raises(ValueError, match="identical"):
        dataset[1]


def test_normal_sequence_keeps_temporal_motion(tmp_path):
    dataset = reader(make_bucket(tmp_path / "normal", False, [34]))
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["start_frame"] == 1
    assert sample["action"][0, 0] != sample["action"][-1, 0]
    assert not np.array_equal(np.asarray(sample["video"][0]), np.asarray(sample["video"][-1]))
