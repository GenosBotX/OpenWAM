from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.deploy.video_fp8 import FP8Linear, quantize_video_blocks


def _architecture(device="cpu"):
    return SimpleNamespace(
        action_backbone=nn.Linear(32, 32, device=device, dtype=torch.bfloat16),
        video_backbone=SimpleNamespace(
            dit=SimpleNamespace(blocks=nn.ModuleList([
                nn.Sequential(nn.Linear(32, 32, device=device, dtype=torch.bfloat16)),
            ])),
        ),
    )


def test_fp8_rejects_cpu_without_changing_model():
    architecture = _architecture()
    source = architecture.video_backbone.dit.blocks[0][0]
    with pytest.raises(ValueError, match="BF16 CUDA"):
        quantize_video_blocks(architecture)
    assert architecture.video_backbone.dit.blocks[0][0] is source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native FP8 requires CUDA")
def test_native_fp8_padding_compile_and_action_isolation():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native FP8 requires Ada or newer")
    torch.manual_seed(42)
    architecture = _architecture("cuda")
    original = architecture.video_backbone.dit.blocks[0][0]
    action = architecture.action_backbone.weight.detach().clone()
    info = quantize_video_blocks(architecture)
    assert quantize_video_blocks(architecture) is info
    layer = architecture.video_backbone.dit.blocks[0][0]
    assert isinstance(layer, FP8Linear)
    assert layer.weight_fp8.dtype == torch.float8_e4m3fn
    assert torch.equal(action, architecture.action_backbone.weight)
    assert architecture.action_backbone.weight.dtype == torch.bfloat16
    x = torch.randn(1, 17, 32, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(x)
        actual = layer(x)
        assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
        assert (actual - expected).float().square().mean().sqrt() < 0.06
        compiled = torch.compile(layer, dynamic=False, fullgraph=True)
        result = compiled(x)
        assert torch.isfinite(result).all()
        # Compile fuses scale/rounding arithmetic, so FP8 bin boundaries can
        # differ from eager. Check quantization error against BF16 and replay
        # stability rather than requiring exact eager FP8 rounding.
        assert (result - expected).float().square().mean().sqrt() < 0.06
        torch.testing.assert_close(compiled(x), result, atol=0, rtol=0)


def test_engine_quantizes_before_compile(monkeypatch):
    from omegaconf import OmegaConf

    import openwam.deploy.video_fp8 as fp8
    from openwam.deploy.engine import JointInferenceEngine

    events = []
    monkeypatch.setattr(fp8, "quantize_video_blocks", lambda arch: events.append("fp8"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    engine = JointInferenceEngine.__new__(JointInferenceEngine)
    engine.cfg = OmegaConf.create({"optimization": {
        "video_fp8": {"enabled": True}, "compile": {"enabled": True, "required": True},
    }})
    engine.architecture = SimpleNamespace(apply_compile_optimizations=lambda cfg: events.append("compile"))
    engine._init_optimizations()
    assert events == ["fp8", "compile"]


def test_required_compile_failure_is_not_silently_ignored():
    from omegaconf import OmegaConf

    from openwam.deploy.engine import JointInferenceEngine

    def fail(cfg):
        raise RuntimeError("compile failed")

    engine = JointInferenceEngine.__new__(JointInferenceEngine)
    engine.cfg = OmegaConf.create({"optimization": {"compile": {"enabled": True, "required": True}}})
    engine.architecture = SimpleNamespace(apply_compile_optimizations=fail)
    with pytest.raises(RuntimeError, match="Required inference compile"):
        engine._init_optimizations()
