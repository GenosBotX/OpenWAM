"""EEF proprio stays 16D while chassis action[:, 22:24] is supervised."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.linglong import LinglongDataset, append_base_action


EEF_SLOTS = [0, 1, 2, 3, 4, 5, 6, 9, 34, 35, 36, 37, 38, 39, 40, 43]
BASE_SLOTS = [74, 75]


def _eef_row():
    row = np.full(16, 0.5, dtype=np.float32)
    row[3:7] = (1, 0, 0, 0)
    row[11:15] = (1, 0, 0, 0)
    return row


def _stats(width, **overrides):
    values = {key: np.zeros(width, dtype=np.float32) for key in ("mean", "std", "min", "q01")}
    values["std"][:] = 1
    values["max"] = np.ones(width, dtype=np.float32)
    values["q99"] = np.ones(width, dtype=np.float32)
    for key, value in overrides.items():
        values[key] = np.asarray(value, dtype=np.float32)
    return values


def _write_bucket(root: Path):
    (root / "images").mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(root / "images" / "frame.jpg")
    eef = np.stack([_eef_row() for _ in range(33)])
    joint = np.zeros((33, 24), dtype=np.float32)
    joint[:, 22:24] = (0.25, 0.5)
    image = ["images/frame.jpg"] * 33
    table = pa.table({
        "observation.state_eef": pa.array(eef.tolist(), type=pa.list_(pa.float32())),
        "action_eef": pa.array(eef.tolist(), type=pa.list_(pa.float32())),
        "action": pa.array(joint.tolist(), type=pa.list_(pa.float32())),
        "observation.state": pa.array(joint.tolist(), type=pa.list_(pa.float32())),
        "observation.images.head": image,
        "observation.images.left": image,
        "observation.images.right": image,
    })
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(table, data / "episode_000000.parquet")
    (root / "meta").mkdir()
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 33, "tasks": ["move"]}) + "\n"
    )
    joint_stats = _stats(24)
    joint_stats["min"][23] = -1
    (root / "meta" / "normalization_stats.json").write_text(json.dumps({
        "state": {key: value.tolist() for key, value in _stats(24).items()},
        "action": {key: value.tolist() for key, value in joint_stats.items()},
    }))
    eef_stats = _stats(16)

    def side(name, column):
        return {
            "source_column": column,
            name: {"default": {f"global_{key}": value.tolist() for key, value in eef_stats.items()}},
        }

    (root / "meta" / "action_eef_stats.json").write_text(json.dumps(side("action", "action_eef")))
    (root / "meta" / "state_eef_stats.json").write_text(json.dumps(side("state", "observation.state_eef")))
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "split": "train",
        "source_root": ".",
        "prompt": "move",
        "observation_state_eef": {
            "column": "observation.state_eef",
            "shape": [16],
            "reference": "base_link",
            "quaternion_order": "wxyz",
        },
    }))


def _dataset(root, *, supervise, action_dims, monkeypatch):
    monkeypatch.delenv("OPENWAM_LINGLONG_DATASET_DIR", raising=False)
    monkeypatch.delenv("OPENWAM_LINGLONG_SOURCE_ROOT", raising=False)
    action_map = ["0-6", 9, "34-40", 43] + ([74, 75] if supervise else [])
    return LinglongDataset(
        str(root),
        action_key="action_eef",
        state_key="observation.state_eef",
        action_mode="eef_quat",
        unify_action_map=action_map,
        unify_state_map=["0-6", 9, "34-40", 43],
        active_raw_action_dims=action_dims,
        active_raw_state_dims=list(range(16)),
        supervise_base_action=supervise,
        height=32,
        width=32,
    )


def test_append_base_action_keeps_eef_prefix():
    eef = np.arange(32, dtype=np.float32).reshape(2, 16)
    joint = np.zeros((2, 24), dtype=np.float32)
    joint[:, 22:24] = [[1, 2], [3, 4]]
    out = append_base_action(eef, joint)
    assert out.shape == (2, 18)
    assert np.array_equal(out[:, :16], eef)
    assert np.array_equal(out[:, 16:], joint[:, 22:24])


def test_chassis_action_is_supervised_without_chassis_proprio(tmp_path, monkeypatch):
    _write_bucket(tmp_path)
    dataset = _dataset(tmp_path, supervise=True, action_dims=list(range(18)), monkeypatch=monkeypatch)
    assert dataset.raw_action_dim == 18
    assert dataset.raw_state_dim == 16
    assert dataset._action_stats["min"].shape == (18,)
    assert dataset._state_stats["min"].shape == (16,)
    assert list(dataset._action_dst_index[-2:]) == BASE_SLOTS
    assert set(BASE_SLOTS).isdisjoint(dataset._state_dst_index.tolist())

    sample = dataset[0]
    action = sample["action"].numpy()
    proprio = sample["proprio"].numpy()
    assert action.shape == (32, 80)
    assert proprio.shape == (1, 80)
    assert np.allclose(action[:, 74], -0.5)
    assert np.allclose(action[:, 75], 0.5)
    assert np.all(proprio[:, BASE_SLOTS] == 0)
    assert sample["action_mask"].numpy()[:, BASE_SLOTS].all()
    assert not sample["proprio_mask"].numpy()[:, BASE_SLOTS].any()
    assert sample["proprio_mask"].numpy()[:, EEF_SLOTS].all()
    assert sample["action_mask"].numpy()[:, EEF_SLOTS].all()


def test_default_eef_training_does_not_supervise_chassis(tmp_path, monkeypatch):
    _write_bucket(tmp_path)
    dataset = _dataset(tmp_path, supervise=False, action_dims=list(range(16)), monkeypatch=monkeypatch)
    sample = dataset[0]
    assert dataset.raw_action_dim == 16
    assert np.all(sample["action"].numpy()[:, BASE_SLOTS] == 0)
    assert not sample["action_mask"].numpy()[:, BASE_SLOTS].any()


def test_chassis_supervision_rejects_a_masked_base(tmp_path, monkeypatch):
    _write_bucket(tmp_path)
    with pytest.raises(ValueError, match="dims 16 and 17"):
        _dataset(tmp_path, supervise=True, action_dims=list(range(16)), monkeypatch=monkeypatch)
