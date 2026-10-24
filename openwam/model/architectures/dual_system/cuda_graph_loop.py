"""Fixed-shape CUDA Graph runner for the DualSystem MoT inference loop."""

from __future__ import annotations

import copy
import weakref
from dataclasses import fields, is_dataclass
from typing import Any

import torch


def _map_tree(value: Any, *, clone_tensors: bool, _memo=None) -> Any:
    if _memo is None:
        _memo = {}
    if isinstance(value, torch.Tensor):
        if not clone_tensors:
            return value
        if id(value) in _memo:
            return _memo[id(value)]
        result = torch.empty_strided(value.shape, value.stride(), device=value.device, dtype=value.dtype)
        _copy_tensor(value, result)
        _memo[id(value)] = result
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in fields(value):
            setattr(result, field.name, _map_tree(getattr(value, field.name), clone_tensors=clone_tensors, _memo=_memo))
        return result
    if isinstance(value, dict):
        return {key: _map_tree(item, clone_tensors=clone_tensors, _memo=_memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tree(item, clone_tensors=clone_tensors, _memo=_memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tree(item, clone_tensors=clone_tensors, _memo=_memo) for item in value)
    return value


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device, tensor.requires_grad)


def _signature(value: Any, _memo=None) -> Any:
    if _memo is None:
        _memo = {}
    if isinstance(value, torch.Tensor):
        if id(value) in _memo:
            return ("tensor_alias", _memo[id(value)])
        _memo[id(value)] = len(_memo)
        return ("tensor", *_tensor_signature(value))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple((field.name, _signature(getattr(value, field.name), _memo)) for field in fields(value)),
        )
    if isinstance(value, dict):
        return (dict, tuple((key, _signature(item, _memo)) for key, item in value.items()))
    if isinstance(value, list):
        return (list, tuple(_signature(item, _memo) for item in value))
    if isinstance(value, tuple):
        return (tuple, tuple(_signature(item, _memo) for item in value))
    if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device)):
        return (type(value), value)
    return (type(value), id(value))


def _copy_tensor(source: torch.Tensor, target: torch.Tensor) -> None:
    # Preserve expanded/strided input layouts so compilation sees the same
    # strides as the baseline. Copy only one element along broadcast axes.
    unique = tuple(
        slice(0, 1) if stride == 0 and size > 1 else slice(None) for size, stride in zip(target.shape, target.stride())
    )
    target[unique].copy_(source[unique])


def _copy_inputs(source: Any, target: Any, *, skip_context_kv: bool = False) -> None:
    if isinstance(source, torch.Tensor):
        if not isinstance(target, torch.Tensor) or (
            tuple(source.shape),
            source.dtype,
            source.device,
            source.requires_grad,
        ) != (tuple(target.shape), target.dtype, target.device, target.requires_grad):
            raise RuntimeError("CUDA Graph MoT input tensor signature changed after capture")
        _copy_tensor(source, target)
        return
    if is_dataclass(source) and not isinstance(source, type):
        if type(source) is not type(target):
            raise RuntimeError("CUDA Graph MoT state type changed after capture")
        for field in fields(source):
            if skip_context_kv and field.name == "context_kv":
                continue
            _copy_inputs(
                getattr(source, field.name),
                getattr(target, field.name),
                skip_context_kv=skip_context_kv,
            )
        return
    if isinstance(source, dict):
        if not isinstance(target, dict) or source.keys() != target.keys():
            raise RuntimeError("CUDA Graph MoT state mapping changed after capture")
        for key in source:
            if skip_context_kv and key == "context_kv":
                continue
            _copy_inputs(source[key], target[key], skip_context_kv=skip_context_kv)
        return
    if isinstance(source, (list, tuple)):
        if type(source) is not type(target) or len(source) != len(target):
            raise RuntimeError("CUDA Graph MoT state sequence changed after capture")
        for src_item, dst_item in zip(source, target):
            _copy_inputs(src_item, dst_item, skip_context_kv=skip_context_kv)


def _tensor_leaves(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensor_leaves(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def _context_kv_tensors(vstate: Any, astate: Any) -> tuple[torch.Tensor, ...]:
    """Return request-scoped cross-attention K/V leaves in stable traversal order."""
    video_extras = getattr(vstate, "extras", None)
    video_kv = video_extras.get("context_kv") if isinstance(video_extras, dict) else None
    action_payload = getattr(astate, "payload", None)
    action_kv = getattr(action_payload, "context_kv", None)
    return tuple((*_tensor_leaves(video_kv), *_tensor_leaves(action_kv)))


def _weak_tensor_refs(tensors: tuple[torch.Tensor, ...]) -> tuple[weakref.ReferenceType, ...]:
    return tuple(weakref.ref(tensor) for tensor in tensors)


def _same_tensor_refs(refs: tuple[weakref.ReferenceType, ...], tensors: tuple[torch.Tensor, ...]) -> bool:
    return len(refs) == len(tensors) and all(ref() is tensor for ref, tensor in zip(refs, tensors))


class CudaGraphMoTLoop:
    """Cache one CUDA Graph per exact state signature and refresh tensor inputs.

    Request K/V tensors must remain immutable; a new request supplies new
    tensor objects. Returned states reference reusable graph output buffers
    and must be consumed before the next replay. Calls must be serialized.
    """

    def __init__(self, loop):
        self.loop = loop
        self._entries: dict[tuple, tuple] = {}

    def __call__(self, vstate, astate):
        if torch.is_grad_enabled():
            raise RuntimeError("CUDA Graph MoT loop only supports inference with gradients disabled")
        signature = _signature((vstate, astate))
        entry = self._entries.get(signature)
        if entry is None:
            entry = self._capture(vstate, astate)
            self._entries[signature] = entry
        static_v_in, static_a_in, graph, output_states, kv_refs, kv_initialized = entry
        current_kv = _context_kv_tensors(vstate, astate)
        same_kv = _same_tensor_refs(kv_refs, current_kv)
        # The first call refreshes the complete request. Subsequent denoising
        # steps reuse request-scoped K/V buffers when their tensor identities
        # are unchanged, while latent/timestep/modulation state is refreshed.
        _copy_inputs(vstate, static_v_in, skip_context_kv=kv_initialized and same_kv)
        _copy_inputs(astate, static_a_in, skip_context_kv=kv_initialized and same_kv)
        if not same_kv:
            kv_refs = _weak_tensor_refs(current_kv)
        self._entries[signature] = (
            static_v_in,
            static_a_in,
            graph,
            output_states,
            kv_refs,
            True,
        )
        graph.replay()
        return output_states

    def _capture(self, vstate, astate):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph MoT loop requested without CUDA")
        static_v_in, static_a_in = _map_tree((vstate, astate), clone_tensors=True)
        tensors = [*_tensor_leaves(static_v_in), *_tensor_leaves(static_a_in)]
        if not tensors or any(not tensor.is_cuda for tensor in tensors):
            raise RuntimeError("CUDA Graph MoT loop requires every state tensor to be on CUDA")
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise RuntimeError(f"CUDA Graph MoT loop requires one CUDA device, got {devices}")

        device = tensors[0].device
        warmup_v = _map_tree(static_v_in, clone_tensors=False)
        warmup_a = _map_tree(static_a_in, clone_tensors=False)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.no_grad():
            self.loop(warmup_v, warmup_a)
        torch.cuda.current_stream(device).wait_stream(stream)

        capture_v = _map_tree(static_v_in, clone_tensors=False)
        capture_a = _map_tree(static_a_in, clone_tensors=False)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph), torch.no_grad():
                output_states = self.loop(capture_v, capture_a)
        except Exception as exc:
            raise RuntimeError("CUDA Graph MoT loop capture failed") from exc
        return (
            static_v_in,
            static_a_in,
            graph,
            output_states,
            _weak_tensor_refs(_context_kv_tensors(vstate, astate)),
            False,
        )


__all__ = ["CudaGraphMoTLoop"]
