"""DiT Velocity Caching for accelerated inference.

DreamZero-inspired optimization: between consecutive denoising steps,
if the velocity (noise) prediction changes minimally (measured by cosine
similarity), reuse the cached prediction instead of running another
expensive DiT forward pass.

This is complementary to TeaCache (which caches based on timestep
embedding similarity within a single step). DiT velocity caching
operates across steps in the denoising loop.
"""

from numbers import Integral
from typing import Optional, Sequence

import torch
from torch import Tensor


class DiTVelocityCache:
    """Cache DiT velocity predictions and skip recomputation when similar.

    At each denoising step, after computing the velocity prediction,
    call :meth:`update` to store it. At the next step, call
    :meth:`should_recompute` to check whether the new prediction is
    expected to differ significantly. If not, :meth:`get_cached` returns
    the previous prediction directly.

    The decision is based on cosine similarity between consecutive velocity
    predictions. Joint video/action callers additionally require the cached
    action prediction to be stable against the previous action prediction.
    When all required similarities exceed the threshold, the cached value is
    reused with optional linear interpolation.

    Args:
        cosine_threshold: Minimum cosine similarity to trigger cache reuse.
            Higher values = more conservative caching (fewer skips).
            Typical range: 0.95-0.995. Default 0.99.
        max_consecutive_skips: Maximum number of consecutive steps to skip
            before forcing a recomputation. Prevents drift accumulation.
        interpolation_weight: When reusing, optionally blend cached velocity
            toward the expected direction. 0.0 = pure cache, 1.0 = full recompute.
        mode: "adaptive" uses similarity and max_consecutive_skips (default).
            "fixed" recomputes only at recompute_steps, with a cold-cache
            fallback when a required prediction is missing.
        recompute_steps: Strictly increasing, one-based steps starting with 1.
            Fixed mode ignores cosine_threshold and max_consecutive_skips.
    """

    def __init__(
        self,
        cosine_threshold: float = 0.99,
        max_consecutive_skips: int = 3,
        interpolation_weight: float = 0.0,
        *,
        mode: str = "adaptive",
        recompute_steps: Sequence[int] = (1, 3),
    ):
        if mode not in ("adaptive", "fixed"):
            raise ValueError("dit_cache.mode must be 'adaptive' or 'fixed'")
        try:
            steps = tuple(recompute_steps)
        except TypeError as exc:
            raise ValueError("dit_cache.recompute_steps must be a sequence of integers") from exc
        if (not steps or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 1 for s in steps)
                or steps[0] != 1 or tuple(sorted(set(steps))) != steps):
            raise ValueError("dit_cache.recompute_steps must be strictly increasing positive integers starting with 1")
        self.mode = mode
        self.recompute_steps = steps
        self.cosine_threshold = cosine_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.interpolation_weight = interpolation_weight

        self._cached_velocity: Optional[Tensor] = None
        self._cached_action_velocity: Optional[Tensor] = None
        self._cached_sigma: Optional[float] = None
        self._prev_velocity: Optional[Tensor] = None
        self._prev_action_velocity: Optional[Tensor] = None
        self._consecutive_skips: int = 0
        self._total_skips: int = 0
        self._total_steps: int = 0

    def validate_schedule(self, num_steps: int, denoise_mode: str):
        """Reject unsupported fixed schedules before any denoising work.

        Fixed step numbers refer to joint synchronous updates, not client
        execution ticks. Async schedules can contain single-stream plateaus.
        """
        if self.mode == "fixed":
            if denoise_mode != "sync":
                raise ValueError("Fixed DiT cache requires inference.denoise_mode=sync")
            if self.recompute_steps[-1] > num_steps:
                raise ValueError("dit_cache.recompute_steps exceeds inference.denoise_steps")

    def should_recompute(self, current_sigma: float, *, require_action: bool = False,
                         action_mask: Optional[Tensor] = None) -> bool:
        """Decide whether to run the DiT forward pass or reuse cache.

        Args:
            current_sigma: Current noise level (sigma = t / 1000).
            require_action: Whether the caller also needs a cached action
                prediction. Joint video/action denoising can only skip a full
                forward when both streams have cached predictions.
            action_mask: Optional boolean mask broadcastable to the action
                velocity. RTC compares only the unfrozen, represented suffix;
                clean prefix and unused unified-action channels cannot vote
                for a cache hit. An empty comparison always recomputes.

        Returns:
            True if the DiT should be run, False if cache can be reused.
        """
        self._total_steps += 1

        if self.mode == "fixed":
            if (self._total_steps in self.recompute_steps
                    or self._cached_velocity is None
                    or (require_action and self._cached_action_velocity is None)):
                return True
            self._consecutive_skips += 1
            self._total_skips += 1
            return False

        if self._cached_velocity is None or self._prev_velocity is None:
            return True

        if require_action and (self._cached_action_velocity is None or self._prev_action_velocity is None):
            return True

        if self._consecutive_skips >= self.max_consecutive_skips:
            return True

        # Cosine similarity between last two velocity predictions
        v1 = self._prev_velocity.flatten().float()
        v2 = self._cached_velocity.flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        if cos_sim.item() < self.cosine_threshold:
            return True

        if require_action:
            a1 = self._prev_action_velocity
            a2 = self._cached_action_velocity
            if action_mask is not None:
                if action_mask.dtype != torch.bool:
                    raise ValueError("action_mask must be boolean")
                mask = torch.broadcast_to(action_mask, a2.shape)
                a1, a2 = a1.masked_select(mask), a2.masked_select(mask)
                if a2.numel() == 0:
                    return True
            a1, a2 = a1.flatten().float(), a2.flatten().float()
            action_cos_sim = torch.nn.functional.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0))
            if not bool(torch.isfinite(action_cos_sim).all()) or action_cos_sim.item() < self.cosine_threshold:
                return True

        self._consecutive_skips += 1
        self._total_skips += 1
        return False

    def get_cached(self) -> Tensor:
        """Return the cached velocity prediction.

        Call this when :meth:`should_recompute` returns False.
        """
        if self._cached_velocity is None:
            raise RuntimeError("No cached velocity available")
        return self._cached_velocity

    def get_cached_action(self) -> Optional[Tensor]:
        """Return the cached action prediction, when the last update had one."""
        return self._cached_action_velocity

    def update(self, velocity: Tensor, sigma: float, action_velocity: Optional[Tensor] = None):
        """Store a new velocity prediction in the cache.

        Call this after each DiT forward pass.

        Args:
            velocity: (B, C, T, H, W) or (B, T, D) velocity prediction.
            sigma: Current noise level.
            action_velocity: Optional action noise prediction from the same
                joint forward. When present, joint denoising can skip the whole
                forward instead of re-running it just to recover action output.
        """
        self._prev_velocity = self._cached_velocity
        self._prev_action_velocity = self._cached_action_velocity
        self._cached_velocity = velocity.detach().clone()
        self._cached_action_velocity = action_velocity.detach().clone() if action_velocity is not None else None
        self._cached_sigma = sigma
        self._consecutive_skips = 0

    def reset(self):
        """Clear all cached state. Call between inference runs."""
        self._cached_velocity = None
        self._cached_action_velocity = None
        self._cached_sigma = None
        self._prev_velocity = None
        self._prev_action_velocity = None
        self._consecutive_skips = 0
        self._total_skips = 0
        self._total_steps = 0

    @property
    def skip_rate(self) -> float:
        """Fraction of steps that were skipped via caching."""
        if self._total_steps == 0:
            return 0.0
        return self._total_skips / self._total_steps

    @property
    def stats(self) -> dict:
        """Return caching statistics."""
        return {
            "total_steps": self._total_steps,
            "total_skips": self._total_skips,
            "skip_rate": self.skip_rate,
            "cosine_threshold": self.cosine_threshold,
            "mode": self.mode,
            "recompute_steps": list(self.recompute_steps) if self.mode == "fixed" else None,
        }
