"""Clean-prefix RTC conditioning; lengths count action ticks, not video frames."""
from numbers import Integral

import torch

DEFAULT_PREFIX_LENGTHS = (0, 6, 7, 8, 9, 10, 11, 12)


def validate_prefix_lengths(values, horizon=None):
    values = tuple(values)
    if not values or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 0 for d in values):
        raise ValueError("RTC prefix_lengths must be nonnegative integers")
    if len(set(values)) != len(values) or 0 not in values:
        raise ValueError("RTC prefix_lengths must be unique and include 0 for cold starts")
    if horizon is not None and max(values) >= horizon:
        raise ValueError("RTC prefix lengths must be smaller than the action horizon")
    return tuple(int(d) for d in values)


def sample_prefix_mask(batch_size, horizon, lengths, *, device):
    """Independently sample one of the configured lengths per example, uniformly."""
    lengths = validate_prefix_lengths(lengths, horizon)
    choices = torch.tensor(lengths, device=device, dtype=torch.long)
    delays = choices[torch.randint(len(lengths), (batch_size,), device=device)]
    return torch.arange(horizon, device=device)[None, :] < delays[:, None]


def suffix_padding_mask(actions, action_is_pad, prefix):
    """Combine clean-prefix exclusion with the existing time/dimension padding.

    The current Piper reader emits complete windows. Reject truncated examples
    with missing prefix ticks or no supervised suffix rather than changing the
    requested sampling distribution or silently returning zero loss.
    """
    if action_is_pad is None:
        padding = torch.zeros_like(actions, dtype=torch.bool)
    else:
        padding = action_is_pad.to(device=actions.device, dtype=torch.bool)
        if padding.shape == actions.shape[:2]:
            padding = padding.unsqueeze(-1).expand_as(actions)
        if padding.shape != actions.shape:
            raise ValueError("RTC action_is_pad must have shape [B,H] or [B,H,D]")
    if bool((prefix & padding.all(dim=-1)).any()):
        raise ValueError("RTC prefix includes padded action ticks; use complete action windows")
    suffix_padding = padding | prefix.unsqueeze(-1)
    if bool(suffix_padding.flatten(1).all(dim=1).any()):
        raise ValueError("RTC requires at least one valid supervised suffix element per example")
    return suffix_padding
