"""Piper RTC wire adapter: absolute EEF20 -> request-relative normalized prefix."""
import numpy as np

from openwam.dataloader.utils.relative_eef import absolute_to_relative_eef20
from openwam.deploy.obs_preprocess import ObsValidationError


def prepare_rtc_prefix(architecture, prefix, reference_state):
    lengths = getattr(architecture, "rtc_prefix_lengths", ())
    if not lengths:
        raise ObsValidationError("RTC prefix requires an RTC-trained checkpoint")
    if reference_state is None:
        raise ObsValidationError("RTC wire adapter currently requires relative EEF20 and an absolute input state")
    if hasattr(prefix, "detach"):
        prefix = prefix.detach().float().cpu().numpy()
    try:
        raw = np.asarray(prefix, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ObsValidationError("action_prefix must be a numeric absolute EEF20 matrix [d,20]") from exc
    if raw.shape == (0,):
        raw = raw.reshape(0, 20)
    if raw.ndim != 2 or raw.shape[1] != 20 or not np.isfinite(raw).all():
        raise ObsValidationError("action_prefix must be a finite absolute EEF20 matrix [d,20]")
    if len(raw) not in lengths:
        raise ObsValidationError(f"Unsupported RTC prefix length {len(raw)}; trained lengths: {lengths}")
    try:
        relative = absolute_to_relative_eef20(raw, reference_state)
    except ValueError as exc:
        raise ObsValidationError(f"Invalid RTC EEF pose: {exc}") from exc
    normalizer = getattr(architecture, "normalizer", None)
    normalized = relative if normalizer is None else normalizer.normalize_actions(relative)
    if normalized.shape != (len(raw), architecture.action_dim):
        raise ValueError("RTC prefix normalization does not match model action dimension")
    return np.array(normalized, copy=True), np.array(raw, copy=True)
