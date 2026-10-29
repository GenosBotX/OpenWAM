"""Relative targets, directional statistics and the real deploy engine boundary."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.dataloader.piper_x import PiperXDataset
from openwam.dataloader.utils.relative_eef import RELATIVE_SEMANTICS, RELATIVE_STATS_FORMAT
from openwam.deploy.model_loader import _build_normalizer, repr_contract_from_cfg
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.policy import WAMPolicy
from openwam.deploy.executors import ExecutionConfig
from openwam.deploy.relative_actions import capture_action_reference, restore_action_chunk
from tests.test_piper_x_static import make_bucket
from tests.test_relative_eef import pose


def stats_file(root):
    payload = json.loads((root / "meta/eef20_normalization_stats.json").read_text())
    payload.update(format=RELATIVE_STATS_FORMAT, **RELATIVE_SEMANTICS,
                   num_frames=33, window_stride=1, sources={str(root): {}})
    (root / "meta/relative_eef20_normalization_stats.json").write_text(json.dumps(payload))
    return payload


def config(relative=True):
    return OmegaConf.create({
        "dataloader": {"type": "piper_x", "action_mode": "eef", "relative_to_state": relative,
            "normalize_mode": "min-max", "num_frames": 33, "window_stride": 1,
            "unify_action": True, "unify_action_map": ["0-9", "34-43"]},
        "inference": {"denoise_steps": 1, "num_frames": 33, "action_response_mode": "chunk"},
        "optimization": {"decode_video": False},
    })


def test_reader_relative_window_uses_first_state_not_previous_action(tmp_path):
    root = make_bucket(tmp_path / "normal", False, [35]); stats_file(root)
    relative = PiperXDataset(str(root), action_mode="eef", relative_to_state=True, height=32, width=32)
    absolute = PiperXDataset(str(root), action_mode="eef", height=32, width=32)
    sample = relative[1]
    np.testing.assert_allclose(sample["action"][:, 0], np.arange(32) / 100, atol=1e-6)
    np.testing.assert_array_equal(sample["proprio"], absolute[1]["proprio"])
    assert len(relative) == len(absolute) == 3
    np.testing.assert_array_equal(sample["action_mask"], absolute[1]["action_mask"])


def test_static_relative_gt_has_zero_position_identity_rotation(tmp_path):
    root = make_bucket(tmp_path / "static", True, [3, 2]); stats_file(root)
    ds = PiperXDataset(str(root), action_mode="eef", relative_to_state=True, height=32, width=32)
    sample = ds[4]
    np.testing.assert_allclose(sample["action"][:, [0, 1, 2, 34, 35, 36]], 0, atol=1e-6)
    np.testing.assert_allclose(sample["action"][:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (32, 1)), atol=1e-6)
    assert sample["proprio"][0, 0] != 0
    assert all(np.array_equal(sample["video"][0], frame) for frame in sample["video"])


def test_reader_rejects_absolute_or_stale_statistics(tmp_path):
    root = make_bucket(tmp_path / "normal", False, [34])
    with pytest.raises(FileNotFoundError):
        PiperXDataset(str(root), action_mode="eef", relative_to_state=True)
    payload = stats_file(root); payload["rotation_delta"] = "wrong_order"
    (root / "meta/relative_eef20_normalization_stats.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rotation_delta"):
        PiperXDataset(str(root), action_mode="eef", relative_to_state=True)


def test_checkpoint_normalizer_requires_matching_semantics(tmp_path):
    root = make_bucket(tmp_path / "normal", False, [34]); stats_file(root)
    ds = PiperXDataset(str(root), action_mode="eef", relative_to_state=True)
    np.save(tmp_path / "normalization_stats.npy", ds.normalization_stats_payload)
    normalizer = _build_normalizer(config(), str(tmp_path))
    sample = ds[1]
    raw = normalizer.unnormalize(sample["action"].numpy())
    np.testing.assert_allclose(raw[:, 0], np.arange(32) / 100, atol=1e-6)
    with pytest.raises(ValueError, match="Absolute"):
        _build_normalizer(config(False), str(tmp_path))
    payload = dict(ds.normalization_stats_payload); payload.pop("piper_x_action_contract")
    np.save(tmp_path / "normalization_stats.npy", payload)
    with pytest.raises(ValueError, match="metadata"):
        _build_normalizer(config(), str(tmp_path))


def make_engine(monkeypatch):
    import openwam.deploy.engine as module
    monkeypatch.setattr(module, "make_schedule", lambda *args, **kwargs: object())
    cfg = config()
    relative = np.tile(pose(.01, 0, 0, .2), (32, 1))
    relative[:, 10] = -.02
    arch = SimpleNamespace(repr_contract=repr_contract_from_cfg(cfg),
        video_scheduler=None, action_scheduler=None,
        normalize_deploy_proprio=lambda x: np.asarray(x) * 17,
        generate=lambda **kw: {"actions": relative.copy(), "video": None})
    return JointInferenceEngine(cfg, architecture=arch), relative


def test_engine_decodes_once_before_sync_buffer(monkeypatch):
    engine, relative = make_engine(monkeypatch)
    policy = WAMPolicy(engine, config(), ExecutionConfig(mode="sync", inference_horizon=2))
    state = pose(.4, .1, .2, -.3)
    first = policy.predict_action({"state": state})
    changed = pose(.9, .1, .2, -.3)
    buffered = policy.predict_action({"state": changed})
    assert abs(first[0] - .41) < 1e-6
    np.testing.assert_array_equal(first, buffered)
    third = policy.predict_action({"state": changed})
    assert abs(third[0] - .91) < 1e-6
    chunk = policy.predict_action_chunk({"state": state})
    assert chunk.shape == (32, 20)
    np.testing.assert_allclose(chunk[0], first, atol=1e-6)


def test_request_reference_is_a_copy_and_missing_state_fails():
    contract = repr_contract_from_cfg(config()); state = pose(.4, .1, .2, -.3)
    reference = capture_action_reference(contract, state)
    state[0] = 100
    assert reference[0] < 1
    with pytest.raises(ValueError, match="requires"):
        capture_action_reference(contract, None)
    with pytest.raises(ValueError, match="20D"):
        capture_action_reference(contract, np.zeros(80))
    original = {"actions": np.zeros((32, 20))}
    assert restore_action_chunk(original, None) is original


def test_resume_cannot_ignore_relative_semantics(tmp_path):
    from openwam.train.utils.checkpointing import verify_resume_normalization_stats
    root = make_bucket(tmp_path / "normal", False, [34]); stats_file(root)
    ds = PiperXDataset(str(root), action_mode="eef", relative_to_state=True)
    np.save(tmp_path / "normalization_stats.npy", ds.normalization_stats_payload)
    verify_resume_normalization_stats(str(tmp_path), ds)
    changed = dict(ds.normalization_stats_payload)
    changed.pop("piper_x_action_contract")
    np.save(tmp_path / "normalization_stats.npy", changed)
    with pytest.raises(ValueError):
        verify_resume_normalization_stats(str(tmp_path), ds)


def test_async_prefetch_uses_generation_state_when_adopted(monkeypatch):
    engine, _ = make_engine(monkeypatch)
    policy = WAMPolicy(engine, config(), ExecutionConfig(mode="async", inference_horizon=2, inference_delay_steps=1))
    try:
        first = policy.predict_action({"state": pose(.4, .1, .2, -.3)})
        second = policy.predict_action({"state": pose(.6, .1, .2, -.3)})
        third = policy.predict_action({"state": pose(.9, .1, .2, -.3)})
        np.testing.assert_array_equal(first, second)
        assert abs(third[0] - .61) < 1e-6
    finally:
        policy.shutdown()


def test_websocket_predict_and_handshake_return_absolute_actions(monkeypatch):
    from PIL import Image
    from openwam.deploy.server import PolicyServer
    engine, _ = make_engine(monkeypatch)
    server = PolicyServer(engine, config())
    observation = {"state": pose(.4, .1, .2, -.3).tolist(), "prompt": "pick",
        "images": {"head_camera": Image.new("RGB", (32, 32))}}
    response = server.predict(observation)
    assert np.asarray(response["action"]).shape == (32, 20)
    assert abs(response["action"][0][0] - .41) < 1e-6
    pong = server._pong_payload()
    assert pong["action_target"] == "absolute_pose"
    assert pong["model_action_target"] == "relative_pose"
    assert pong["model_action_contract_id"] == "piper-x-relative-eef20-base-v1"
    assert pong["relative_actions_decoded_by"] == "server"
