"""Real Accelerate round trips and exact step/parameter contracts for compact resume."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

from openwam.train.utils.async_checkpoint import CheckpointPublisher
from openwam.train.utils.checkpointing import find_latest_accel_state, load_full_state
from openwam.train.utils.compact_resume import _load_deepspeed_module, paired_overlay
from openwam.train.utils.trainable_checkpoint import read_manifest
from tests.test_trainable_checkpoint import TinyModel, setup


@pytest.fixture(autouse=True)
def isolated_accelerator_state():
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def rng_sample():
    return random.random(), np.random.rand(3), torch.rand(3)


def step(acc, model, optimizer, scheduler):
    value = torch.randn(4, 3)
    optimizer.zero_grad()
    loss = model(value).square().mean()
    acc.backward(loss)
    optimizer.step()
    scheduler.step()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_cpu_resume_next_update_and_rng(tmp_path, asynchronous):
    model, _, output, manifest = setup(tmp_path)
    acc = Accelerator(cpu=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1 / (n + 1))
    model, optimizer, scheduler = acc.prepare(model, optimizer, scheduler)
    step(acc, model, optimizer, scheduler)
    publisher = CheckpointPublisher(
        acc, enabled=asynchronous, overlay_config=manifest,
        staging_root=str(tmp_path / "stage"), min_free_bytes=1,
    )
    publisher.publish(model, str(output), 4, 1, 0, resume_context={"at_optimizer_boundary": True},
                      final=False, include_full_state=True, prune="rolling", keep_last_k=1)
    publisher.close()
    saved = copy.deepcopy(model.state_dict())
    optimizer_saved = copy.deepcopy(optimizer.state_dict())
    scheduler_saved = copy.deepcopy(scheduler.state_dict())
    rng_expected = rng_sample()
    step(acc, model, optimizer, scheduler)
    expected = copy.deepcopy(model.state_dict())
    optimizer_expected = copy.deepcopy(optimizer.state_dict())
    state_dir = output / "accel_state_step_4"
    assert not list(state_dir.glob("*safetensors"))
    assert not list(state_dir.glob("pytorch_model*"))
    assert "fixed.weight" not in load_file(str(output / "checkpoint_step_4.safetensors"))
    # Loading a separately constructed model must also restore frozen weights.
    restored_acc = Accelerator(cpu=True)
    restored = TinyModel()
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda n: 1 / (n + 1))
    restored, restored_optimizer, restored_scheduler = restored_acc.prepare(restored, restored_optimizer, restored_scheduler)
    identities = [id(p) for p in restored.parameters()]
    meta = load_full_state(restored_acc, str(state_dir), architecture=restored)
    assert meta["global_step"] == 4
    assert identities == [id(p) for p in restored.parameters()]
    assert_tree_equal(restored.state_dict(), saved)
    assert_tree_equal(restored_optimizer.state_dict(), optimizer_saved)
    assert_tree_equal(restored_scheduler.state_dict(), scheduler_saved)
    rng_actual = rng_sample()
    assert rng_actual[0] == rng_expected[0]
    assert np.array_equal(rng_actual[1], rng_expected[1])
    assert torch.equal(rng_actual[2], rng_expected[2])
    step(restored_acc, restored, restored_optimizer, restored_scheduler)
    assert_tree_equal(restored.state_dict(), expected)
    assert_tree_equal(restored_optimizer.state_dict(), optimizer_expected)
    assert find_latest_accel_state(str(output)) == str(state_dir)


def make_state(tmp_path):
    from tests.test_trainable_checkpoint import publish
    from openwam.train.utils.compact_resume import resume_descriptor
    from openwam.train.utils.checkpointing import write_trainer_state

    model, _, output, manifest = setup(tmp_path)
    path = publish(model, output, manifest, step=4)
    directory = output / "accel_state_step_4"
    write_trainer_state(str(directory), global_step=4, opt_step=1, epoch=0,
                        model_checkpoint=resume_descriptor(4, manifest))
    meta = json.loads((directory / "trainer_state.json").read_text())
    return model, output, path, directory, meta


@pytest.mark.parametrize("problem", ["missing", "wrong_step", "wrong_base", "wrong_filename", "wrong_format"])
def test_reject_unpaired_resume(tmp_path, problem):
    model, output, path, directory, meta = make_state(tmp_path)
    if problem == "missing":
        Path(path).unlink()
    elif problem == "wrong_step":
        meta["global_step"] = 3
    elif problem == "wrong_base":
        meta["model_checkpoint"]["base_sha256"] = "0" * 64
    elif problem == "wrong_filename":
        meta["model_checkpoint"]["weights"] = "../checkpoint_step_4.safetensors"
    else:
        meta["model_checkpoint"]["format"] = "unknown"
    (directory / "trainer_state.json").write_text(json.dumps(meta))
    with pytest.raises((FileNotFoundError, ValueError)):
        paired_overlay(str(directory), meta)
    assert find_latest_accel_state(str(output)) is None


def test_reject_changed_training_layout_before_load(tmp_path):
    model, _, _, directory, _ = make_state(tmp_path)
    model.fixed.requires_grad_(True)
    with pytest.raises(ValueError, match="layout mismatch"):
        load_full_state(Accelerator(cpu=True), str(directory), architecture=model)


def test_deepspeed_loader_allows_only_frozen_missing_and_identical_overlay(tmp_path):
    model, _, path, _, _ = make_state(tmp_path)
    manifest = read_manifest(path)
    module = load_file(path)
    callback = _load_deepspeed_module(manifest)
    callback(src=module, dst=model)
    incomplete = dict(module)
    incomplete.pop("head.weight")
    with pytest.raises(ValueError, match="keys"):
        callback(src=incomplete, dst=model)
    wrong = {**module, "head.weight": module["head.weight"] + 1}
    with pytest.raises(ValueError, match="differs from paired"):
        callback(src=wrong, dst=model)
