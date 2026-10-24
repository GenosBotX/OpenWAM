"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`DualSystemMoTDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.cuda_graph_loop import CudaGraphMoTLoop
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from openwam.model.compile_options import (
    as_bool,
    cfg_get,
    compile_enabled,
    section_enabled,
    self_attn_compile_cfg,
    torch_compile_kwargs,
)

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: MoT-style mixed attention at every layer.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mot_driver: DualSystemMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object]]] = None
        self._compiled_request_context_kv = None
        self._cuda_graph_mot_run_joint_loop: Optional[CudaGraphMoTLoop] = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at DualSystemMoTDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )

        # MoT driver is built once both backbones are available. The video
        # backbone is normally constructed in ``BaseWAMArchitecture._init_video_backbone``
        # (already done by ``super().__init__``), so we can build the driver
        # right here. Tests that swap in a mock video backbone after init call
        # :meth:`build_mot_driver` directly.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(cfg.get("attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "first_frame_causal")),
        }
        if self.video_backbone is not None:
            self.build_mot_driver()

    def build_mot_driver(self) -> DualSystemMoTDriver:
        """Construct the :class:`DualSystemMoTDriver` from the current backbones.

        Re-callable; raises if either backbone is missing. Tests that swap in
        a mock video backbone after ``__init__`` should call this method to
        wire up the driver afterwards.
        """
        if self.video_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: video_backbone is not "
                "set. Construct the architecture with a video_backbone config or attach "
                "one before calling this method."
            )
        if self.action_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: action_backbone is not "
                "set. Architecture must be built from a non-None cfg."
            )

        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    @property
    def mot_driver(self) -> DualSystemMoTDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time MoT joint loop when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_mot_run_joint_loop = None
        self._compiled_request_context_kv = None
        self._cuda_graph_mot_run_joint_loop = None
        section = self_attn_compile_cfg(compile_cfg)
        cuda_graph_cfg = cfg_get(section, "cuda_graph", None)
        if as_bool(cfg_get(cuda_graph_cfg, "enabled", False), default=False):
            if as_bool(cfg_get(section, "dynamic", False), default=False):
                raise ValueError("CUDA Graph MoT requires self_attn.dynamic=false")
            if not section_enabled(section, default=True):
                raise ValueError("CUDA Graph MoT requires self_attn.enabled=true")
            if self.video_backbone is None or self.action_backbone is None:
                raise RuntimeError("CUDA Graph MoT requested before backbones are ready")
            driver = self._mot_driver or self.build_mot_driver()
            self._cuda_graph_mot_run_joint_loop = CudaGraphMoTLoop(driver.run_joint_loop)
            logger.info("Enabled fixed-shape CUDA Graph replay for the dual self-attn MoT loop")
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        if not section_enabled(section, default=True):
            logger.info("dual self-attn MoT compile disabled by config; running eager.")
            return
        if self.video_backbone is None or self.action_backbone is None:
            logger.warning("dual self-attn MoT compile requested before backbones are ready; running eager.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_joint_loop(vstate, astate):
            return driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        loop_kwargs = dict(kwargs)
        if self._cuda_graph_mot_run_joint_loop is not None:
            # Explicit capture owns the graph boundary. Retain the compiler's
            # other mode options while disabling its nested automatic graphs.
            mode = loop_kwargs.get("mode", "default")
            options = dict(torch._inductor.list_mode_options(mode))
            if options.get("triton.cudagraphs", False):
                loop_kwargs.pop("mode", None)
                options["triton.cudagraphs"] = False
                loop_kwargs["options"] = options
        try:
            self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **loop_kwargs)
            self._compiled_request_context_kv = torch.compile(self._prepare_request_context_kv, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_mot_run_joint_loop = None
            self._compiled_request_context_kv = None
            if self._cuda_graph_mot_run_joint_loop is not None:
                raise RuntimeError("CUDA Graph MoT torch.compile setup failed") from exc
            logger.warning("dual self-attn MoT torch.compile setup failed; running eager: %s", exc)
            return
        if self._cuda_graph_mot_run_joint_loop is not None:
            self._cuda_graph_mot_run_joint_loop = CudaGraphMoTLoop(self._compiled_mot_run_joint_loop)
        logger.info("Enabled dual self-attn MoT compile with torch.compile kwargs=%s", kwargs)

    def _prepare_request_context_kv(self, video_context, action_context):
        # Keep RMSNorm in the same compiler execution mode as the MoT loop.
        # Computing it eagerly changes bf16 rounding relative to compiled MoT.
        return (
            tuple(block.cross_attn.prepare_context_kv(video_context) for block in self.video_backbone.dit.blocks),
            tuple(block.cross_attn.prepare_context_kv(action_context) for block in self.action_backbone.blocks),
        )

    def prepare_inference_request_context(self, inputs_shared: dict, *, proprio=None, cfg_scale: float = 1.0) -> dict:
        """Prepare fixed Wan/action text context and cross-attention K/V once per inference request."""
        inputs_shared = dict(inputs_shared)
        if self._compiled_mot_run_joint_loop is not None and self._compiled_request_context_kv is None:
            return inputs_shared
        # CFG has distinct conditional and unconditional contexts. Keep its
        # existing per-forward path until both branches have independent caches.
        if self.training or cfg_scale != 1.0 or inputs_shared.get("clip_feature") is not None:
            return inputs_shared
        vb = self.video_backbone
        ab = self.action_backbone
        # This cache is deliberately scoped to Wan's text cross-attention
        # contract; other video backbones may use a different context layout.
        from openwam.model.video_backbone.wan_backbone import WanBase

        dit = getattr(vb, "dit", None)
        if (
            not isinstance(vb, WanBase)
            or dit is None
            or not hasattr(dit, "text_embedding")
            or not hasattr(ab, "prepare_joint_self_attn_request_context")
            or ab.variant != "joint_self_attn"
        ):
            return inputs_shared
        if any(
            getattr(block.cross_attn, "has_image_input", False) or not hasattr(block.cross_attn, "prepare_context_kv")
            for block in dit.blocks
        ):
            return inputs_shared

        pipeline_inputs = self._append_proprio_context_token(inputs_shared, proprio)
        raw_context = pipeline_inputs["context"]
        context_mask = pipeline_inputs.get("context_mask")
        if context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=raw_context.device)
            positions = torch.arange(raw_context.shape[1], device=raw_context.device)
            context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        if context_mask is not None:
            context_mask = context_mask.to(device=raw_context.device, dtype=torch.bool)

        video_weight = dit.text_embedding[0].weight
        video_context = dit.text_embedding(raw_context.to(device=video_weight.device, dtype=video_weight.dtype))
        action_context, action_context_mask, _ = ab.prepare_joint_self_attn_request_context(
            raw_context, context_mask, prepare_kv=False
        )
        prepare_kv = self._compiled_request_context_kv or self._prepare_request_context_kv
        try:
            video_context_kv, action_context_kv = prepare_kv(video_context, action_context)
        except Exception:
            if self._compiled_request_context_kv is None or self._cuda_graph_mot_run_joint_loop is not None:
                raise
            # Preserve the original compiled numerical path if preparation
            # cannot be compiled; never silently substitute eager K/V.
            logger.exception("Compiled request K/V preparation failed; using per-step context projection.")
            self._compiled_request_context_kv = None
            return inputs_shared
        pipeline_inputs["context"] = video_context
        pipeline_inputs["context_mask"] = context_mask
        pipeline_inputs["context_is_embedded"] = True
        pipeline_inputs["context_kv"] = video_context_kv
        pipeline_inputs["_request_context_prepared"] = True
        pipeline_inputs["_request_action_context"] = action_context
        pipeline_inputs["_request_action_context_mask"] = action_context_mask
        pipeline_inputs["_request_action_context_kv"] = action_context_kv
        return pipeline_inputs

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        pipeline_inputs = dict(pipeline_inputs)
        request_context_prepared = bool(pipeline_inputs.pop("_request_context_prepared", False))
        request_action_context = pipeline_inputs.pop("_request_action_context", None)
        request_action_context_mask = pipeline_inputs.pop("_request_action_context_mask", None)
        request_action_context_kv = pipeline_inputs.pop("_request_action_context_kv", None)
        if not request_context_prepared:
            pipeline_inputs = self._append_proprio_context_token(pipeline_inputs, proprio)
        action_context = request_action_context if request_context_prepared else pipeline_inputs.get("context")
        action_context_mask = (
            request_action_context_mask if request_context_prepared else pipeline_inputs.get("context_mask")
        )
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
        # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
        # path; TI2V itself fires that path first so these kwargs are inert for
        # it). VACE and I2V do NOT emit ``first_frame_latents`` (VACE routes
        # its first-frame condition through ``vace_context``; I2V uses the
        # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
        # structurally inert — kept on only so the joint MoT driver gets the
        # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
        # pass False.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            context_is_embedded=request_context_prepared,
            context_kv=request_action_context_kv,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
        cuda_graph_loop = getattr(self, "_cuda_graph_mot_run_joint_loop", None)
        if cuda_graph_loop is not None and (
            self.training or use_gradient_checkpointing or use_gradient_checkpointing_offload
        ):
            raise RuntimeError("CUDA Graph MoT requires eval mode without gradient checkpointing")
        if cuda_graph_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            try:
                vstate, astate = cuda_graph_loop(vstate, astate)
            except Exception as exc:
                raise RuntimeError("dual self-attn CUDA Graph MoT loop failed") from exc
        elif compiled_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            try:
                vstate, astate = compiled_loop(vstate, astate)
            except Exception as exc:
                self._compiled_mot_run_joint_loop = None
                logger.warning("dual self-attn compiled MoT loop failed; falling back to eager: %s", exc)
                vstate, astate = driver.run_joint_loop(
                    vstate,
                    astate,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        else:
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vb.finalize(vstate), ab.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
