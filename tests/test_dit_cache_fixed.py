"""Fixed joint velocity reuse, configuration routing and invalid schedules."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from openwam.deploy.optimizations.dit_cache import DiTVelocityCache


def test_fixed_reuses_latest_scheduled_prediction_without_similarity_or_skip_limit():
    cache = DiTVelocityCache(mode="fixed", recompute_steps=[1, 3],
                             cosine_threshold=1.1, max_consecutive_skips=0)
    observed = []
    for step in range(1, 6):
        if cache.should_recompute(1 - step / 6, require_action=True):
            # Opposite directions deliberately cannot pass a cosine test.
            value = 1. if step == 1 else -3.
            cache.update(torch.full((1, 4), value), 1 - step / 6,
                         torch.full((1, 32, 20), value * 10))
        observed.append((cache.get_cached()[0, 0].item(), cache.get_cached_action()[0, 0, 0].item()))
    assert observed == [(1., 10.), (1., 10.), (-3., -30.), (-3., -30.), (-3., -30.)]
    assert cache.stats["total_steps"] == 5
    assert cache.stats["total_skips"] == 3
    assert cache.stats["mode"] == "fixed"
    assert cache.stats["recompute_steps"] == [1, 3]
    cache.reset()
    assert cache.stats["total_steps"] == 0
    assert cache.should_recompute(.9, require_action=True)


def test_fixed_missing_joint_action_prediction_forces_forward():
    cache = DiTVelocityCache(mode="fixed")
    assert cache.should_recompute(.9)
    cache.update(torch.ones(1, 4), .9)
    assert cache.should_recompute(.8, require_action=True)
    assert cache.stats["total_skips"] == 0


@pytest.mark.parametrize("steps", [[], [2, 3], [0, 1], [1, 1], [1, 3, 2],
                                        [True, 3], [1, 3.0], "1,3", None])
def test_invalid_fixed_steps_rejected(steps):
    with pytest.raises(ValueError, match="recompute_steps"):
        DiTVelocityCache(mode="fixed", recompute_steps=steps)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        DiTVelocityCache(mode="typo")


def _engine(overrides=()):
    from openwam.deploy.engine import JointInferenceEngine
    cfg = OmegaConf.create({
        "inference": {"denoise_steps": 5, "denoise_mode": "sync", "num_frames": 33},
        "optimization": {"dit_cache": {"enabled": True}},
    })
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return JointInferenceEngine(cfg, architecture=SimpleNamespace())


def test_deploy_dotlist_switches_fixed_adaptive_and_disabled():
    fixed = _engine(["optimization.dit_cache.mode=fixed", "optimization.dit_cache.recompute_steps=[1,3]"])
    assert fixed._dit_cache.mode == "fixed"
    assert fixed._dit_cache.recompute_steps == (1, 3)
    adaptive = _engine()
    assert adaptive._dit_cache.mode == "adaptive"
    disabled = _engine(["optimization.dit_cache.enabled=false"])
    assert disabled._dit_cache is None


@pytest.mark.parametrize("override", ["inference.denoise_steps=2", "inference.denoise_mode=async"])
def test_deploy_rejects_invalid_fixed_schedule_at_startup(override):
    with pytest.raises(ValueError, match="requires|exceeds"):
        _engine(["optimization.dit_cache.mode=fixed", override])


@pytest.mark.parametrize("conditions", [{"denoise_steps": 2}, {"denoise_mode": "async"}])
def test_request_override_cannot_bypass_fixed_schedule_validation(conditions):
    engine = _engine(["optimization.dit_cache.mode=fixed"])
    with pytest.raises(ValueError, match="requires|exceeds"):
        engine.generate(conditions)
