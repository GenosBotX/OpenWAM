"""Opt-in native E4M3FN GEMMs for Wan video blocks; action stays BF16.

Apply to a fully restored BF16 CUDA model before compilation or graph capture.
Quantization is approximate and is disabled unless explicitly configured.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class FP8Linear(nn.Module):
    """Static per-tensor weight scale and dynamic per-tensor activation scale."""

    def __init__(self, source):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        weight = source.weight.detach().float()
        scale = weight.abs().amax().clamp_min(1e-12) / 448.0
        self.register_buffer("weight_fp8", (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn))
        self.register_buffer("weight_scale", scale.float())
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().clone())

    def forward(self, x):
        shape = x.shape
        matrix = x.reshape(-1, self.in_features)
        rows = matrix.shape[0]
        scale = matrix.float().abs().amax().clamp_min(1e-12) / 448.0
        quantized = (matrix.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        padding = (-rows) % 16
        if padding:
            quantized = F.pad(quantized, (0, 0, 0, padding))
        result = torch._scaled_mm(
            quantized, self.weight_fp8.t(), scale_a=scale, scale_b=self.weight_scale,
            out_dtype=torch.bfloat16, use_fast_accum=False,
        )[:rows]
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*shape[:-1], self.out_features)


def quantize_video_blocks(architecture):
    """Quantize Wan block Linear layers once, after loading base plus overlay.

    Validate the complete selection before replacing anything. Unsupported
    hardware/dtypes fail explicitly instead of silently selecting BF16.
    """
    previous = getattr(architecture, "_video_fp8_info", None)
    if previous is not None:
        return previous
    video = getattr(architecture, "video_backbone", None)
    blocks = getattr(getattr(video, "dit", None), "blocks", None)
    action = getattr(architecture, "action_backbone", None)
    if blocks is None or action is None:
        raise ValueError("Video FP8 requires Wan DiT blocks and a BF16 action backbone")
    action_before = {name: (p.data_ptr(), p.dtype) for name, p in action.named_parameters()}
    if not action_before or any(dtype != torch.bfloat16 for _, dtype in action_before.values()):
        raise ValueError("Video FP8 requires action parameters to remain BF16")
    selected = []
    for index, block in enumerate(blocks):
        for name, module in block.named_modules():
            if isinstance(module, FP8Linear):
                raise ValueError("Video blocks are partially quantized; reload the checkpoint first")
            if isinstance(module, nn.Linear):
                if not module.weight.is_cuda or module.weight.dtype != torch.bfloat16:
                    raise ValueError("Video FP8 requires fully restored BF16 CUDA weights")
                capability = torch.cuda.get_device_capability(module.weight.device)
                if capability < (8, 9):
                    raise RuntimeError("Native FP8 requires compute capability >= 8.9 (Ada/Hopper or newer)")
                if module.in_features % 16 or module.out_features % 16:
                    raise ValueError(f"FP8 Linear dimensions must be multiples of 16: {name}")
                if module.bias is not None and module.bias.dtype != torch.bfloat16:
                    raise ValueError("Video FP8 requires BF16 biases")
                selected.append((index, block, name, module))
    if not selected:
        raise ValueError("No Wan block Linear layers found for FP8")
    names = []
    for index, block, name, module in selected:
        parent_path, _, leaf = name.rpartition(".")
        parent = block.get_submodule(parent_path) if parent_path else block
        setattr(parent, leaf, FP8Linear(module))
        names.append(f"video_backbone.dit.blocks.{index}.{name}")
    action_after = {name: (p.data_ptr(), p.dtype) for name, p in action.named_parameters()}
    if action_before != action_after:
        raise RuntimeError("Action parameters changed during video quantization")
    architecture._video_fp8_info = {
        "format": "E4M3FN", "scaling": "per-tensor dynamic activation; static weight",
        "modules": names, "action_dtype": "bfloat16", "accumulator": "float32", "use_fast_accum": False,
    }
    logger.info("Enabled video E4M3FN for %d Wan block Linear layers; action remains BF16", len(names))
    return architecture._video_fp8_info
