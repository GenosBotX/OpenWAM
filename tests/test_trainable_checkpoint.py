"""Trainable checkpoint round trips, integrity checks, and publisher integration."""
import copy
import io
import json
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.train.utils import trainable_checkpoint as overlay
from openwam.train.utils.async_checkpoint import CheckpointPublisher
from openwam.train.utils.checkpointing import find_latest_weights, write_weights_atomic


class TinyModel(torch.nn.Module):
    save_checkpoint = BaseWAMArchitecture.save_checkpoint
    load_checkpoint = BaseWAMArchitecture.load_checkpoint

    def __init__(self, device="cpu"):
        super().__init__()
        self.fixed = torch.nn.Linear(3, 3, device=device)
        self.fixed.requires_grad_(False)
        self.head = torch.nn.Linear(3, 2, device=device)
        self.register_buffer("counter", torch.tensor(0, device=device))
        self.fixed.register_buffer("running", torch.zeros(3, device=device))
        self.register_buffer("scratch", torch.ones(3, device=device), persistent=False)

    def forward(self, value):
        return self.head(self.fixed(value) + self.fixed.running)


class Accelerator:
    is_main_process = is_local_main_process = True
    process_index = 0
    num_processes = 1

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass

    def get_state_dict(self, model):
        raise AssertionError("Overlay must not gather the full frozen model")

    def register_save_state_pre_hook(self, hook):
        class Handle:
            def remove(self):
                pass
        weights = [object()]
        hook([], weights, None)
        assert not weights
        return Handle()

    def save_state(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "optimizer.bin").write_bytes(b"optimizer-state")


def setup(tmp_path, model=None):
    model = model or TinyModel()
    base = tmp_path / "base.safetensors"
    model.save_checkpoint(str(base))
    model.load_checkpoint(str(base))
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.yaml").write_text("model: tiny\n")
    (output / "normalization_stats.npy").write_bytes(b"normalization")
    manifest = overlay.prepare_overlay(model, output_path=str(output))
    with torch.no_grad():
        model.head.weight.add_(0.5)
        model.fixed.running.add_(2)
        model.counter.add_(7)
    return model, base, output, manifest


def publish(model, output, manifest, step=1):
    return write_weights_atomic(
        model, str(output), step, overlay.snapshot_trainable(model, manifest),
        trainable_only=True, overlay_config=manifest,
    )


def rewrite(path, change):
    state = load_file(path)
    manifest = overlay.read_manifest(path)
    change(state, manifest)
    save_file(state, path, metadata={
        "format": overlay.FORMAT,
        overlay.MANIFEST_KEY: json.dumps(manifest),
    })


def test_exact_roundtrip_and_meta_materialization(tmp_path):
    model, _, output, manifest = setup(tmp_path)
    path = publish(model, output, manifest)
    assert set(load_file(path)) == {"head.weight", "head.bias", "counter", "fixed.running"}
    assert "scratch" not in load_file(path)
    assert not list(output.glob("*.manifest.json"))  # one atomic deploy artifact
    restored = TinyModel()
    restored.load_checkpoint(path)
    meta = TinyModel(device="meta")
    meta.load_checkpoint(path)
    for key, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], expected)
        assert torch.equal(meta.state_dict()[key], expected)
    value = torch.randn(5, 3)
    assert torch.equal(restored(value), model(value))
    full = output / "full.safetensors"
    model.save_checkpoint(str(full))
    legacy = TinyModel()
    legacy.load_checkpoint(str(full))
    assert torch.equal(legacy(value), restored(value))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("include_full_state", [False, True])
def test_publisher_snapshots_before_training_continues(tmp_path, enabled, include_full_state):
    model, _, output, manifest = setup(tmp_path)
    expected = copy.deepcopy(model.state_dict())
    gate = threading.Event()
    publisher = CheckpointPublisher(
        Accelerator(), enabled=enabled, overlay_config=manifest, copy_gate=gate,
        staging_root=str(tmp_path / "stage"), min_free_bytes=1,
    )
    try:
        publisher.publish(
            model, str(output), 10, 1, 0, resume_context=None, final=False,
            include_full_state=include_full_state, prune="rolling", keep_last_k=1,
        )
        if enabled:
            assert not (output / "checkpoint_step_10.safetensors").exists()
        with torch.no_grad():
            model.head.weight.add_(100)
            model.counter.add_(10)
            model.fixed.running.add_(9)
        gate.set()
        publisher.wait()
        restored = overlay.load_checkpoint_state(find_latest_weights(str(output)))
        for key in expected:
            assert torch.equal(restored[key], expected[key])
        state_dir = output / "accel_state_step_10"
        if include_full_state:
            assert (state_dir / "trainer_state.json").is_file()
            assert (state_dir / "optimizer.bin").read_bytes() == b"optimizer-state"
        else:
            assert not state_dir.exists()
    finally:
        gate.set()
        publisher.close()


def test_wrong_base_rejected(tmp_path, monkeypatch):
    model, base, output, manifest = setup(tmp_path)
    path = publish(model, output, manifest)
    wrong = tmp_path / "wrong.safetensors"
    TinyModel().save_checkpoint(str(wrong))
    monkeypatch.setenv("OPENWAM_BASE_CHECKPOINT", str(wrong))
    with pytest.raises(ValueError, match="SHA-256"):
        overlay.load_checkpoint_state(path)


@pytest.mark.parametrize("artifact", ["config.yaml", "normalization_stats.npy"])
def test_changed_or_missing_deployment_assets_rejected(tmp_path, artifact):
    model, _, output, manifest = setup(tmp_path)
    path = publish(model, output, manifest)
    (output / artifact).write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact"):
        overlay.load_checkpoint_state(path)
    (output / artifact).unlink()
    with pytest.raises(ValueError, match="artifact"):
        overlay.load_checkpoint_state(path)


@pytest.mark.parametrize("change", [
    lambda state, manifest: state.pop("head.weight"),
    lambda state, manifest: state.update(extra=torch.ones(1)),
    lambda state, manifest: state.update({"head.weight": torch.ones(1)}),
    lambda state, manifest: manifest["frozen_parameters"].append("not_in_base"),
])
def test_invalid_overlay_rejected(tmp_path, change):
    model, _, output, manifest = setup(tmp_path)
    path = publish(model, output, manifest)
    rewrite(path, change)
    with pytest.raises(ValueError):
        overlay.load_checkpoint_state(path)


def test_frozen_parameter_must_match_base(tmp_path):
    model, base, _, _ = setup(tmp_path)
    with torch.no_grad():
        model.fixed.weight.add_(1)
    with pytest.raises(ValueError, match="differs from base"):
        overlay.prepare_overlay(model, base_path=str(base))


def test_trainable_aliases_and_changed_layout(tmp_path):
    model = TinyModel()
    model.alias = model.head
    # Base save_file cannot serialize shared storage; save disjoint copies.
    base = tmp_path / "base.safetensors"
    save_file({k: v.clone() for k, v in model.state_dict().items()}, str(base))
    manifest = overlay.prepare_overlay(model, base_path=str(base))
    snapshot = overlay.snapshot_trainable(model, manifest)
    assert torch.equal(snapshot["alias.weight"], snapshot["head.weight"])
    assert snapshot["alias.weight"].data_ptr() != snapshot["head.weight"].data_ptr()
    model.head.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable|layout changed"):
        overlay.snapshot_trainable(model, manifest)


def test_overlay_finetune_flattens_to_original_base(tmp_path):
    model, base, output, manifest = setup(tmp_path)
    path = publish(model, output, manifest)
    restored = TinyModel()
    restored.load_checkpoint(path)
    next_manifest = overlay.prepare_overlay(restored)
    assert next_manifest["base"]["path"] == str(base)
    assert next_manifest["base"]["sha256"] == manifest["base"]["sha256"]


def test_public_download_uses_shared_cache_and_checks_hash(tmp_path, monkeypatch):
    _, base, _, _ = setup(tmp_path)
    expected = base.read_bytes()
    spec = {
        "path": str(tmp_path / "absent"),
        "sha256": overlay.sha256_file(base),
        "url": "https://example.invalid/base.safetensors",
    }
    monkeypatch.setenv("OPENWAM_CHECKPOINT_CACHE", str(tmp_path / "cache"))
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(expected)

    monkeypatch.setattr(overlay, "urlopen", download)
    cached = overlay.resolve_base(spec)
    assert overlay.sha256_file(cached) == spec["sha256"]
    assert overlay.resolve_base(spec) == cached
    assert len(calls) == 1
    assert not list((tmp_path / "cache").glob("*.partial"))
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        overlay.resolve_base(spec)
    assert not (tmp_path / "cache" / ("0" * 64 + ".safetensors")).exists()
    assert not list((tmp_path / "cache").glob("*.partial"))


def test_failed_write_is_not_discoverable_and_retention_works(tmp_path, monkeypatch):
    model, _, output, manifest = setup(tmp_path)
    from openwam.train.utils.checkpointing import manage_checkpoints
    publish(model, output, manifest, step=1)
    publish(model, output, manifest, step=2)
    manage_checkpoints(str(output), 1)
    assert not (output / "checkpoint_step_1.safetensors").exists()
    assert find_latest_weights(str(output)).endswith("checkpoint_step_2.safetensors")

    def failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(model, "save_checkpoint", failure)
    with pytest.raises(OSError, match="disk full"):
        publish(model, output, manifest, step=3)
    assert find_latest_weights(str(output)).endswith("checkpoint_step_2.safetensors")


def test_trainer_prepares_manifest_and_rejects_sharded_parameters(tmp_path):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from openwam.train.openwam_trainer import OpenWAMTrainer

    model, _, output, _ = setup(tmp_path)
    trainer = object.__new__(OpenWAMTrainer)
    trainer.architecture = model
    trainer.accelerator = Accelerator()
    trainer.cfg = OmegaConf.create({"training": {"save_trainable_only": True, "zero_stage": 2}})
    trainer.configure_checkpoint_overlay(str(output))
    assert trainer._checkpoint_overlay["trainable_parameters"] == ["head.bias", "head.weight"]
    trainer.accelerator.state = SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3))
    with pytest.raises(ValueError, match="ZeRO-3"):
        trainer.configure_checkpoint_overlay(str(output))


def test_unmaterialized_and_unsupported_format_rejected(tmp_path):
    model, _, output, manifest = setup(tmp_path)
    with pytest.raises(ValueError, match="unmaterialized"):
        overlay.snapshot_trainable(TinyModel(device="meta"), manifest)
    path = output / "unsupported.safetensors"
    save_file({"weight": torch.ones(1)}, str(path), metadata={"format": "openwam_trainable_overlay_v999"})
    with pytest.raises(ValueError, match="Unsupported"):
        overlay.load_checkpoint_state(str(path))
