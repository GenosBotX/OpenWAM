"""Optimizer checkpoints paired with a trainable overlay, without frozen weights."""
from __future__ import annotations

import os

RESUME_FORMAT = "openwam_trainable_resume_v1"


def _distributed_type(accelerator):
    return str(getattr(accelerator, "distributed_type", "NO")).split(".")[-1]


def save_accelerator_state(accelerator, directory, *, trainable_only=False):
    if not trainable_only:
        accelerator.save_state(directory)
        return
    kind = _distributed_type(accelerator)
    if kind == "DEEPSPEED":
        accelerator.save_state(directory, exclude_frozen_parameters=True)
    elif kind in ("FSDP", "MEGATRON_LM"):
        raise ValueError(f"Compact resume does not support {kind}")
    else:
        # Accelerate still writes optimizer/scheduler/scaler/RNG/custom state.
        handle = accelerator.register_save_state_pre_hook(lambda models, weights, output_dir: weights.clear())
        try:
            accelerator.save_state(directory)
        finally:
            handle.remove()


def resume_descriptor(global_step, manifest):
    if manifest is None:
        return None
    return {
        "format": RESUME_FORMAT,
        "weights": f"checkpoint_step_{int(global_step)}.safetensors",
        "base_sha256": manifest["base"]["sha256"],
    }


def paired_overlay(state_dir, meta):
    """Validate the exact matching step; never select latest weights independently."""
    from openwam.train.utils.trainable_checkpoint import read_manifest

    spec = meta["model_checkpoint"]
    if spec.get("format") != RESUME_FORMAT:
        raise ValueError("Unsupported resume checkpoint format")
    step = int(meta["global_step"])
    expected = f"checkpoint_step_{step}.safetensors"
    if spec.get("weights") != expected or os.path.basename(os.path.normpath(state_dir)) != f"accel_state_step_{step}":
        raise ValueError("Resume state and overlay step mismatch")
    path = os.path.join(os.path.dirname(os.path.abspath(state_dir)), expected)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume requires its matching trainable checkpoint: {path}")
    manifest = read_manifest(path)
    if manifest is None or manifest.get("global_step") != step:
        raise ValueError("Resume requires a trainable overlay from the same step")
    if manifest["base"]["sha256"] != spec.get("base_sha256"):
        raise ValueError("Resume base checkpoint SHA-256 mismatch")
    return path, manifest


def _validate_layout(model, manifest):
    from openwam.train.utils.trainable_checkpoint import _layout

    _, trainable, buffers, frozen = _layout(model)
    for field, actual in (("trainable_parameters", trainable), ("persistent_buffers", buffers),
                          ("frozen_parameters", frozen)):
        if actual != manifest[field]:
            raise ValueError(f"Resume parameter layout mismatch: {field}")


def _load_deepspeed_module(manifest):
    def load(*, src, dst):
        import torch

        expected = set(manifest["trainable_parameters"]) | set(manifest["persistent_buffers"])
        if set(src) != expected:
            raise ValueError("DeepSpeed resume module keys differ from trainable manifest")
        current = dst.state_dict()
        for name, tensor in src.items():
            spec = manifest["tensors"][name]
            if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != spec["shape"]:
                raise ValueError(f"DeepSpeed resume tensor shape mismatch: {name}")
            if current[name].shape != tensor.shape or current[name].dtype != tensor.dtype:
                raise ValueError(f"DeepSpeed resume tensor dtype mismatch: {name}")
            # Both files represent the same step. Reject accidentally mixed files.
            if not torch.equal(current[name], tensor.to(current[name].device)):
                raise ValueError(f"DeepSpeed resume module differs from paired overlay: {name}")
        missing, unexpected = dst.load_state_dict(src, strict=False)
        if set(missing) != set(manifest["frozen_parameters"]) or unexpected:
            raise ValueError("DeepSpeed resume may omit only the declared frozen parameters")
    return load


def load_compact_state(accelerator, state_dir, meta, architecture):
    from openwam.train.utils.trainable_checkpoint import load_checkpoint_state

    path, manifest = paired_overlay(state_dir, meta)
    model = accelerator.unwrap_model(architecture)
    _validate_layout(model, manifest)
    if any(t.is_meta for t in model.state_dict().values()):
        raise ValueError("Resume must materialize the model before creating the optimizer")
    # Copy in place: optimizer and DeepSpeed must retain their Parameter objects.
    model.load_state_dict(load_checkpoint_state(path), strict=True)
    model._loaded_checkpoint_path = os.path.abspath(path)
    kind = _distributed_type(accelerator)
    if kind == "DEEPSPEED":
        accelerator.load_state(state_dir, custom_load_fn=_load_deepspeed_module(manifest))
    elif kind in ("FSDP", "MEGATRON_LM"):
        raise ValueError(f"Compact resume does not support {kind}")
    else:
        handle = accelerator.register_load_state_pre_hook(lambda models, input_dir: models.clear())
        try:
            accelerator.load_state(state_dir)
        finally:
            handle.remove()
