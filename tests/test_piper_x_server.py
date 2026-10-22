"""Piper-X 边界校验和服务类注入，不加载权重或使用 GPU。"""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.deploy import server
from openwam.deploy.obs_preprocess import ObsValidationError
from openwam.deploy.piper_x import PiperXPolicyServer, validate_state


@pytest.fixture
def setup_server():
    cfg = OmegaConf.create({
        "dataloader": {"type": "robodojo", "variant": "real", "embodiment": "piper_x",
                       "action_mode": "eef", "unify_action": True, "normalize_mode": "min-max"},
        "model": {"architecture": {"state_dim": 80, "action_dim": 80}},
        "inference": {"num_frames": 33, "action_response_mode": "chunk", "inference_mode": "sync"},
    })
    normalizer = SimpleNamespace(_action_dst_index=np.r_[0:10, 34:44],
                                 _state_dst_index=np.r_[0:10, 34:44], stats={"mean": np.zeros(20)})
    engine = SimpleNamespace(architecture=SimpleNamespace(normalizer=normalizer))
    return engine, cfg


def state():
    return [0.25, 0, 0.25, 1, 0, 0, 0, 1, 0, 0.5] * 2


def test_valid_state_is_not_modified():
    original = state()
    validate_state(original)
    assert original == state()


@pytest.mark.parametrize("bad", [None, [0] * 16, [0] * 80, [state()], [float("nan")] * 20,
                                  [float("inf")] * 20, ["invalid"] * 20])
def test_reject_invalid_state(bad):
    with pytest.raises(ObsValidationError):
        validate_state(bad)


@pytest.mark.parametrize(("index", "value"), [(9, -0.1), (19, 1.1), (3, 0), (17, 0)])
def test_reject_gripper_or_degenerate_rotation(index, value):
    values = state()
    values[index] = value
    with pytest.raises(ObsValidationError):
        validate_state(values)


@pytest.mark.parametrize(("field", "value"), [("dataloader.embodiment", "other"),
                           ("dataloader.normalize_mode", "z-score"), ("model.architecture.state_dim", 20)])
def test_reject_checkpoint_mismatch(setup_server, field, value):
    engine, cfg = setup_server
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError, match="Piper-X requires"):
        PiperXPolicyServer(engine, cfg)


@pytest.mark.parametrize("attr", ["_action_dst_index", "_state_dst_index", "stats"])
def test_reject_normalizer_mismatch(setup_server, attr):
    engine, cfg = setup_server
    setattr(engine.architecture.normalizer, attr, {"mean": np.zeros(80)} if attr == "stats" else np.arange(20))
    with pytest.raises(ValueError):
        PiperXPolicyServer(engine, cfg)


def test_contract(setup_server):
    engine, cfg = setup_server
    pong = PiperXPolicyServer(engine, cfg, "owner")._pong_payload()
    assert pong["state_dim"] == pong["action_dim"] == 20
    assert pong["model_state_dim"] == pong["model_action_dim"] == 80
    assert pong["num_actions"] == 32
    assert pong["readiness_token"] == "owner"
    assert pong["action_response_mode"] == "chunk"


@pytest.mark.parametrize("shape", [(20,), (31, 20), (32, 80), (32, 20)])
def test_output_validation(setup_server, monkeypatch, shape):
    engine, cfg = setup_server
    actions = np.zeros(shape)
    monkeypatch.setattr(server.PolicyServer, "predict", lambda self, obs: {"action": actions.tolist()})
    policy = PiperXPolicyServer(engine, cfg)
    if shape == (32, 20):
        assert policy.predict({"state": state()})["action"] == actions.tolist()
        actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite physical EEF20"):
        policy.predict({"state": state()})


def test_main_injects_server_without_replacing_global(monkeypatch):
    cfg = OmegaConf.create({"inference": {"denoise_steps": 10}})
    received = {}
    native_class = server.PolicyServer

    def build(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(cfg=cfg, run=lambda **kwargs: received.update(run=kwargs))

    monkeypatch.setattr(server, "resolve_deploy_args", lambda args: (cfg, "/checkpoint"))
    monkeypatch.setattr(server, "_log_attention_backends", lambda logger: None)
    monkeypatch.setattr(server, "build_server_from_config", build)
    server.main(["--host", "127.0.0.1", "--port", "8849"], server_class=PiperXPolicyServer)
    assert received["server_class"] is PiperXPolicyServer
    assert received["run"] == {"host": "127.0.0.1", "port": 8849}
    assert server.PolicyServer is native_class


def test_builder_constructs_requested_class(setup_server, monkeypatch):
    from openwam import deploy
    from openwam.deploy import model_loader

    engine, cfg = setup_server
    monkeypatch.setattr(model_loader, "load_from_checkpoint_dir",
                        lambda *args, **kwargs: (cfg, engine.architecture))
    monkeypatch.setattr(deploy, "JointInferenceEngine", lambda **kwargs: engine)
    policy = server.build_server_from_config(None, "/unused", server_class=PiperXPolicyServer)
    assert type(policy) is PiperXPolicyServer
    assert policy.engine is engine
    assert type(server.build_server_from_config(None, "/unused")) is server.PolicyServer
