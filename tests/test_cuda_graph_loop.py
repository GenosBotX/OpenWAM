from dataclasses import dataclass

import pytest
import torch

from openwam.model.architectures.dual_system.cuda_graph_loop import (
    CudaGraphMoTLoop,
    _copy_inputs,
    _map_tree,
    _signature,
)


@dataclass
class _State:
    hidden: torch.Tensor
    nested: dict
    marker: object


@dataclass
class _Payload:
    context_kv: tuple
    latent: torch.Tensor


@dataclass
class _GraphState:
    extras: dict
    payload: _Payload


def test_graph_input_clone_refreshes_nested_tensors_and_keeps_opaque_objects():
    marker = object()
    source = _State(torch.arange(6).reshape(2, 3), {"context": (torch.ones(2),)}, marker)
    static = _map_tree(source, clone_tensors=True)
    source.hidden.fill_(7)
    source.nested["context"][0].fill_(9)

    _copy_inputs(source, static)

    assert torch.equal(static.hidden, torch.full((2, 3), 7))
    assert torch.equal(static.nested["context"][0], torch.full((2,), 9.0))
    assert static.hidden.data_ptr() != source.hidden.data_ptr()
    assert static.marker is marker


def test_graph_signature_separates_tensor_layouts():
    contiguous = torch.empty((2, 3))
    transposed = torch.empty((3, 2)).t()

    assert _signature(contiguous) != _signature(transposed)


def test_graph_static_buffers_preserve_sliced_and_broadcast_strides():
    for tensor in [torch.randn(6, 10)[:, ::2], torch.randn(1, 3).expand(4, 3)]:
        static = _map_tree(tensor, clone_tensors=True)
        assert static.stride() == tensor.stride()
        assert torch.equal(static, tensor)
        updated = tensor * 2
        _copy_inputs(updated, static)
        assert torch.equal(static, updated)


def test_graph_static_buffers_preserve_shared_tensor_references_across_states():
    mask = torch.ones(2, 3, dtype=torch.bool)
    a, b = _map_tree(({"mask": mask}, {"mask": mask}), clone_tensors=True)
    assert a["mask"] is b["mask"]
    assert a["mask"] is not mask
    assert _signature((a, b)) != _signature((a, {"mask": mask.clone()}))


def test_graph_input_refresh_rejects_changed_shapes():
    source = torch.empty((2, 3))
    static = _map_tree(source, clone_tensors=True)

    with pytest.raises(RuntimeError, match="signature changed"):
        _copy_inputs(torch.empty((3, 2)), static)


def test_context_kv_copy_can_be_skipped_without_skipping_changing_state():
    video_kv = torch.tensor([1.0, 2.0])
    action_kv = torch.tensor([3.0, 4.0])
    video = _GraphState({"context_kv": (video_kv,), "latent": torch.tensor([5.0])}, _Payload((), torch.empty(0)))
    action = _GraphState({}, _Payload((action_kv,), torch.tensor([6.0])))
    static_video, static_action = _map_tree((video, action), clone_tensors=True)

    video_kv.add_(10)
    action_kv.add_(10)
    video.extras["latent"].add_(10)
    action.payload.latent.add_(10)
    _copy_inputs(video, static_video, skip_context_kv=True)
    _copy_inputs(action, static_action, skip_context_kv=True)

    assert torch.equal(static_video.extras["context_kv"][0], torch.tensor([1.0, 2.0]))
    assert torch.equal(static_action.payload.context_kv[0], torch.tensor([3.0, 4.0]))
    assert torch.equal(static_video.extras["latent"], torch.tensor([15.0]))
    assert torch.equal(static_action.payload.latent, torch.tensor([16.0]))


def test_context_kv_identity_detects_new_request():
    from openwam.model.architectures.dual_system.cuda_graph_loop import (
        _context_kv_tensors,
        _same_tensor_refs,
        _weak_tensor_refs,
    )

    video = _GraphState({"context_kv": (torch.ones(2),)}, _Payload((), torch.empty(0)))
    action = _GraphState({}, _Payload((torch.zeros(2),), torch.empty(0)))
    refs = _weak_tensor_refs(_context_kv_tensors(video, action))
    assert _same_tensor_refs(refs, _context_kv_tensors(video, action))
    video.extras["context_kv"] = (torch.full((2,), 2.0),)
    assert not _same_tensor_refs(refs, _context_kv_tensors(video, action))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph test requires a GPU")
def test_cuda_graph_refreshes_inputs_and_separates_grid_metadata():
    def loop(video, action):
        video.hidden = video.hidden * video.nested["scale"]
        action.hidden = action.hidden + video.hidden.sum()
        return video, action

    marker = object()
    runner = CudaGraphMoTLoop(loop)
    with torch.inference_mode():
        video = _State(
            torch.arange(6, device="cuda").reshape(2, 3),
            {"scale": torch.tensor(2, device="cuda"), "grid": (2, 3)},
            marker,
        )
        action = _State(torch.zeros(2, device="cuda"), {}, marker)
        result_v, result_a = runner(video, action)
        assert torch.equal(result_v.hidden, video.hidden * 2)
        assert torch.equal(result_a.hidden, torch.full((2,), 30.0, device="cuda"))

        video.hidden.fill_(4)
        video.nested["scale"].fill_(3)
        action.hidden.fill_(5)
        result_v, result_a = runner(video, action)
        assert torch.equal(result_v.hidden, torch.full((2, 3), 12, device="cuda"))
        assert torch.equal(result_a.hidden, torch.full((2,), 77.0, device="cuda"))
        assert len(runner._entries) == 1

        video.nested["grid"] = (3, 2)
        runner(video, action)
        assert len(runner._entries) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph test requires a GPU")
def test_cuda_graph_reuses_request_kv_and_refreshes_new_request(monkeypatch):
    from openwam.model.architectures.dual_system import cuda_graph_loop

    def loop(video, action):
        video.extras["latent"] = video.extras["latent"] + video.extras["context_kv"][0]
        action.payload.latent = action.payload.latent + action.payload.context_kv[0]
        return video, action

    with torch.inference_mode():
        video_kv = torch.tensor([2.0], device="cuda")
        action_kv = torch.tensor([3.0], device="cuda")
        video = _GraphState(
            {"context_kv": (video_kv,), "latent": torch.tensor([5.0], device="cuda")},
            _Payload((), torch.empty(0, device="cuda")),
        )
        action = _GraphState({}, _Payload((action_kv,), torch.tensor([7.0], device="cuda")))
        runner = CudaGraphMoTLoop(loop)
        result_v, result_a = runner(video, action)
        assert result_v.extras["latent"].item() == 7
        assert result_a.payload.latent.item() == 10

        copied_kv = []
        original_copy = cuda_graph_loop._copy_tensor

        def track_copy(source, target):
            if source is video_kv or source is action_kv:
                copied_kv.append(source)
            original_copy(source, target)

        monkeypatch.setattr(cuda_graph_loop, "_copy_tensor", track_copy)
        video.extras["latent"].fill_(11)
        action.payload.latent.fill_(13)
        result_v, result_a = runner(video, action)
        assert result_v.extras["latent"].item() == 13
        assert result_a.payload.latent.item() == 16
        assert not copied_kv

        video_kv = torch.tensor([17.0], device="cuda")
        action_kv = torch.tensor([19.0], device="cuda")
        video.extras["context_kv"] = (video_kv,)
        action.payload.context_kv = (action_kv,)
        result_v, result_a = runner(video, action)
        assert result_v.extras["latent"].item() == 28
        assert result_a.payload.latent.item() == 32
        assert len(copied_kv) == 2
        assert len(runner._entries) == 1
