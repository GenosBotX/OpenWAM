import pytest
import torch

from openwam.model.architectures.dual_system.cuda_graph_loop import CudaGraphDenoiseStep


def test_full_step_rejects_grad_enabled_calls():
    with pytest.raises(RuntimeError, match="gradients disabled"):
        CudaGraphDenoiseStep(lambda x, c: x)({}, {})


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_step_refreshes_raw_inputs_context_and_keeps_outputs_static(monkeypatch):
    from openwam.model.architectures.dual_system import cuda_graph_loop

    # Includes input/timestep transforms and both prediction heads.
    def step(x, context):
        video = x["latents"] * 2 + x["timestep"]
        action = x["noisy_actions"] + x["action_timestep"] * 3
        hidden = video + action + context["context_kv"][0] + context["state"]
        return hidden * 4, hidden + 7

    calls = []
    original_copy = cuda_graph_loop._copy_tensor
    with torch.inference_mode():
        inputs = {k: torch.full((2,), float(i), device="cuda") for i, k in enumerate(
            ("latents", "timestep", "noisy_actions", "action_timestep"))}
        context = {"context_kv": (torch.ones(2, device="cuda"),),
                   "state": torch.full((2,), 5., device="cuda")}
        runner = CudaGraphDenoiseStep(step)
        video, action = runner(inputs, context)
        saved_video, saved_action = video.clone(), action.clone()
        ptrs = (video.data_ptr(), action.data_ptr())

        def track(source, target):
            calls.append(source)
            original_copy(source, target)
        monkeypatch.setattr(cuda_graph_loop, "_copy_tensor", track)

        for x in inputs.values():
            x.add_(2)
        expected = step(inputs, context)
        video, action = runner(inputs, context)
        assert torch.equal(video, expected[0])
        assert torch.equal(action, expected[1])
        assert (video.data_ptr(), action.data_ptr()) == ptrs
        assert all(t is not context["context_kv"][0] and t is not context["state"] for t in calls)
        assert not torch.equal(saved_video, video)
        assert not torch.equal(saved_action, action)

        # Same shape, new request: refresh state and K/V without recapture.
        context = {"context_kv": (torch.full((2,), 9., device="cuda"),),
                   "state": torch.full((2,), 13., device="cuda")}
        video, action = runner(inputs, context)
        expected = step(inputs, context)
        assert torch.equal(video, expected[0])
        assert torch.equal(action, expected[1])
        assert len(runner._entries) == 1

        # A new shape gets its own graph rather than stale buffers.
        inputs = {k: torch.ones(3, device="cuda") for k in inputs}
        context = {"context_kv": (torch.ones(3, device="cuda"),),
                   "state": torch.ones(3, device="cuda")}
        video, action = runner(inputs, context)
        assert torch.equal(video, step(inputs, context)[0])
        assert len(runner._entries) == 2
