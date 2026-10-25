"""Representation contract must advertise checkpoint training truth."""

from types import SimpleNamespace

from omegaconf import OmegaConf

from openwam.deploy import server as server_module
from openwam.deploy.model_loader import repr_contract_from_cfg
from openwam.deploy.server import PolicyServer

TRAINING_DL = {
    "action_mode": "robocasa365",
}


def test_repr_contract_from_cfg_reads_compact_values():
    contract = repr_contract_from_cfg(OmegaConf.create({"dataloader": TRAINING_DL}))
    assert contract == {"representation": "robocasa365"}


def test_repr_contract_defaults():
    contract = repr_contract_from_cfg(OmegaConf.create({"dataloader": {}}))
    assert contract == {"representation": "joint"}


def test_piper_x_eef20_contract_and_runtime_chunk_length():
    cfg = OmegaConf.create({
        "dataloader": {"type": "piper_x", "action_mode": "eef", "normalize_mode": "min-max"},
        "model": {"architecture": {"state_dim": 80, "action_dim": 80}},
        "inference": {"action_response_mode": "chunk", "num_frames": 33},
    })
    contract = repr_contract_from_cfg(cfg)
    assert contract["contract_id"] == "piper-x-absolute-eef20-v1"
    assert contract["image_color_order"] == "rgb"
    assert contract["prompt_template"] == "raw"
    assert contract["gripper_open_width_m"] == 0.1
    assert contract["endpoint"] == "tcp_link"
    engine = SimpleNamespace(architecture=SimpleNamespace(repr_contract=contract))
    assert PolicyServer(engine=engine, cfg=cfg)._pong_payload()["num_actions"] == 32
    cfg.dataloader.action_mode = "eef_quat"
    assert "contract_id" not in repr_contract_from_cfg(cfg)
    cfg.dataloader.action_mode = "eef"
    cfg.dataloader.relative_to_state = True
    assert "contract_id" not in repr_contract_from_cfg(cfg)


def test_repr_contract_preserves_optional_legacy_markers():
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "action_mode": "legacy",
                "binary_action_dims": [3],
                "gripper_convention": "minus1_closed_plus1_open",
            }
        }
    )
    assert repr_contract_from_cfg(cfg) == {
        "representation": "legacy",
        "binary_action_dims": [3],
        "gripper_convention": "minus1_closed_plus1_open",
    }


def test_pong_contract_uses_architecture_not_deploy_override():
    training_cfg = OmegaConf.create({"dataloader": TRAINING_DL})
    contract = repr_contract_from_cfg(training_cfg)
    engine = SimpleNamespace(architecture=SimpleNamespace(repr_contract=contract))
    merged = OmegaConf.merge(
        training_cfg,
        OmegaConf.create({"dataloader": {"action_mode": "joint"}}),
    )
    pong = PolicyServer(engine=engine, cfg=merged)._ckpt_contract()
    assert pong == contract


def test_pong_contract_empty_without_architecture():
    server = PolicyServer(engine=SimpleNamespace(architecture=None), cfg=OmegaConf.create({}))
    assert server._ckpt_contract() == {}


def test_pong_contract_can_prove_process_ownership():
    server = PolicyServer(
        engine=SimpleNamespace(architecture=None),
        cfg=OmegaConf.create({}),
        readiness_token="owner",
    )
    assert server._ckpt_contract() == {}
    assert server._pong_payload() == {"readiness_token": "owner"}


def test_main_forwards_readiness_token_to_server_builder(monkeypatch):
    cfg = OmegaConf.create({"inference": {"denoise_steps": 20, "denoise_mode": "sync"}})
    built = {}

    class Server:
        def __init__(self):
            self.cfg = cfg

        def run(self, *, host, port):
            built["run"] = (host, port)

    def build_server_from_config(**kwargs):
        built.update(kwargs)
        return Server()

    monkeypatch.setattr(server_module, "resolve_deploy_args", lambda args: (cfg, "/checkpoint"))
    monkeypatch.setattr(server_module, "build_server_from_config", build_server_from_config)
    monkeypatch.setattr(server_module, "_log_attention_backends", lambda logger: None)
    server_module.main(["--readiness-token", "owner"])

    assert built["readiness_token"] == "owner"
    assert built["run"] == ("0.0.0.0", 8848)
