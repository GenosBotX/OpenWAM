"""Trainable weight overlays backed by a content-addressed full checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from safetensors import safe_open
from safetensors.torch import load_file

FORMAT = "openwam_trainable_overlay_v1"
MANIFEST_KEY = "openwam_manifest"


@lru_cache(maxsize=16)
def _hash_file(path, identity):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    path = os.path.realpath(path)
    stat = os.stat(path)
    return _hash_file(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def read_manifest(path):
    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    fmt = metadata.get("format", "")
    if fmt != FORMAT:
        if MANIFEST_KEY in metadata or fmt.startswith("openwam_trainable_"):
            raise ValueError(f"Unsupported checkpoint format: {fmt!r}")
        return None
    if MANIFEST_KEY not in metadata:
        raise ValueError("Trainable checkpoint is missing its embedded manifest")
    manifest = json.loads(metadata[MANIFEST_KEY])
    if manifest.get("format") != FORMAT:
        raise ValueError("Checkpoint manifest format mismatch")
    return manifest


def _check_digest(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("A complete lowercase SHA-256 is required for the base checkpoint")
    if sha256_file(path) != expected:
        raise ValueError(f"Base checkpoint SHA-256 mismatch: {path}")
    return os.fspath(path)


def _validate_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("checkpoint_base_url must be a public HTTPS URL without credentials or query tokens")


def resolve_base(base, checkpoint_dir=None):
    """Prefer a verified local copy; download once into a separate shared cache."""
    override = os.environ.get("OPENWAM_BASE_CHECKPOINT")
    if override:
        return _check_digest(os.path.expanduser(override), base["sha256"])
    local = base.get("path")
    if local:
        local = os.path.expanduser(local)
        if not os.path.isabs(local) and checkpoint_dir:
            local = os.path.join(checkpoint_dir, local)
        if os.path.isfile(local):
            return _check_digest(local, base["sha256"])
    digest = base.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid base checkpoint SHA-256")
    root = Path(os.environ.get("OPENWAM_CHECKPOINT_CACHE", "~/.cache/openwam/checkpoints")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cached = root / f"{digest}.safetensors"
    # Serialize downloads across local inference/training processes.
    import fcntl
    with open(root / f"{digest}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cached.is_file():
            return _check_digest(cached, digest)
        url = base.get("url")
        if not url:
            raise FileNotFoundError("Base checkpoint unavailable; set OPENWAM_BASE_CHECKPOINT to the matching full weights")
        _validate_url(url)
        temporary = None
        try:
            with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                dir=root, suffix=".partial", delete=False
            ) as output:
                temporary = output.name
                for chunk in iter(lambda: response.read(16 * 1024 * 1024), b""):
                    output.write(chunk)
            _check_digest(temporary, digest)
            os.replace(temporary, cached)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return str(cached)


def _layout(model):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    state = model.state_dict()
    if any(name.startswith("vlm_backbone.") for name in state):
        raise ValueError("Trainable overlays currently require a model without separately saved VLM weights")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("Trainable overlays require tensor-only state_dict entries")
    trainable = sorted(name for name, param in parameters.items() if param.requires_grad and name in state)
    # Persistent buffers are all included: a frozen module can still update one.
    persistent = sorted(set(buffers) & set(state))
    unknown = set(state) - set(parameters) - set(persistent)
    if unknown:
        raise ValueError(f"Unsupported custom state_dict entries: {sorted(unknown)}")
    frozen = sorted(set(state) - set(trainable) - set(persistent))
    if not trainable:
        raise ValueError("No trainable parameters to checkpoint")
    return state, trainable, persistent, frozen


def prepare_overlay(model, *, base_path=None, base_url=None, base_sha256=None, output_path=None):
    """Validate the frozen base before training and return an immutable manifest."""
    from openwam.train.utils.checkpointing import find_latest_weights

    source = base_path or getattr(model, "_loaded_checkpoint_path", None)
    if not source:
        raise ValueError("Trainable checkpoints require finetune_ckpt_path or checkpoint_base_path")
    source = os.path.abspath(os.path.expanduser(os.fspath(source)))
    if os.path.isdir(source):
        source = find_latest_weights(source)
    previous = read_manifest(source)
    if previous:
        base = dict(previous["base"])
        source = resolve_base(base, os.path.dirname(source))
    else:
        base = {"path": source, "sha256": sha256_file(source)}
    if base_sha256 and base_sha256 != base["sha256"]:
        raise ValueError("Configured base checkpoint SHA-256 mismatch")
    if base_url:
        _validate_url(base_url)
        base["url"] = base_url
    if read_manifest(source) is not None:
        raise ValueError("Base checkpoint must contain full weights")
    state, trainable, buffers, frozen = _layout(model)
    with safe_open(source, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for name in frozen:
            if name not in available:
                raise ValueError(f"Frozen parameter absent from base checkpoint: {name}")
            actual = state[name].detach()
            expected = handle.get_tensor(name)
            if actual.is_meta or actual.shape != expected.shape:
                raise ValueError(f"Frozen parameter shape/materialization mismatch: {name}")
            if not torch.equal(actual.cpu(), expected.to(dtype=actual.dtype)):
                raise ValueError(f"Frozen parameter differs from base checkpoint: {name}; use a matching full base")
    artifacts = {}
    if output_path:
        for name in ("config.yaml", "normalization_stats.npy"):
            path = os.path.join(output_path, name)
            if os.path.isfile(path):
                artifacts[name] = sha256_file(path)
    return {
        "format": FORMAT,
        "base": base,
        "trainable_parameters": trainable,
        "persistent_buffers": buffers,
        "frozen_parameters": frozen,
        "artifacts_sha256": artifacts,
    }


def snapshot_trainable(model, manifest):
    """Clone selected tensors synchronously; a writer never reads live weights."""
    state, trainable, buffers, frozen = _layout(model)
    for field, values in (
        ("trainable_parameters", trainable),
        ("persistent_buffers", buffers),
        ("frozen_parameters", frozen),
    ):
        if manifest.get(field) != values:
            raise ValueError(f"Checkpoint parameter layout changed during training: {field}")
    result = {}
    for name in trainable + buffers:
        tensor = state[name]
        if tensor.is_meta:
            raise ValueError(f"Cannot snapshot unmaterialized tensor: {name}")
        result[name] = tensor.detach().to(device="cpu", copy=True).contiguous()
    return result


def checkpoint_metadata(manifest, state, step):
    manifest = dict(manifest, global_step=int(step))
    expected = set(manifest["trainable_parameters"]) | set(manifest["persistent_buffers"])
    if set(state) != expected:
        raise ValueError("Checkpoint snapshot keys do not match trainable manifest")
    manifest["tensors"] = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
        for name, tensor in state.items()
    }
    return {"format": FORMAT, MANIFEST_KEY: json.dumps(manifest, sort_keys=True)}


def load_checkpoint_state(path):
    """Load legacy full weights or verified base plus overlay."""
    path = os.fspath(path)
    manifest = read_manifest(path)
    if manifest is None:
        return load_file(path)
    directory = os.path.dirname(os.path.abspath(path))
    for name, digest in manifest.get("artifacts_sha256", {}).items():
        if name not in ("config.yaml", "normalization_stats.npy"):
            raise ValueError(f"Unsupported checkpoint artifact: {name}")
        artifact = os.path.join(directory, name)
        if not os.path.isfile(artifact) or sha256_file(artifact) != digest:
            raise ValueError(f"Checkpoint artifact missing or SHA-256 mismatch: {name}")
    overlay = load_file(path)
    expected = set(manifest["trainable_parameters"]) | set(manifest["persistent_buffers"])
    frozen = set(manifest["frozen_parameters"])
    if set(overlay) != expected or set(manifest.get("tensors", {})) != expected or expected & frozen:
        raise ValueError("Checkpoint overlay keys do not match manifest")
    for name, tensor in overlay.items():
        spec = manifest["tensors"][name]
        if list(tensor.shape) != spec["shape"] or str(tensor.dtype).replace("torch.", "") != spec["dtype"]:
            raise ValueError(f"Checkpoint tensor metadata mismatch: {name}")
    base_path = resolve_base(manifest["base"], directory)
    if read_manifest(base_path) is not None:
        raise ValueError("Base checkpoint must contain full weights")
    with safe_open(base_path, framework="pt", device="cpu") as handle:
        missing = frozen - set(handle.keys())
        if missing:
            raise ValueError(f"Base checkpoint is missing frozen parameters: {sorted(missing)}")
        result = {name: handle.get_tensor(name) for name in sorted(frozen)}
    result.update(overlay)
    return result
