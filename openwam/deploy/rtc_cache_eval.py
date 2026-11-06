"""Offline paired cache ablation; never connects to the robot or its service."""
from copy import deepcopy
import time

import numpy as np


def _summary(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                max=float(values.max()))


def _rotations(values):
    a, b = values[..., :3], values[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if (n < 1e-8).any():
        raise ValueError('degenerate rotation first column')
    a = a / n
    b = b - (a*b).sum(-1, keepdims=True)*a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    if (n < 1e-8).any():
        raise ValueError('degenerate rotation second column')
    b = b / n
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def compare_actions(reference, candidate, d):
    """Physical suffix deviations, not task accuracy or a ground-truth loss."""
    reference, candidate = (np.asarray(x, dtype=np.float64) for x in (reference, candidate))
    if (reference.shape != (32, 20) or candidate.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()
            or type(d) is not int or not 0 <= d < 32):
        raise ValueError('expected finite physical EEF20 chunks [32,20] and valid d')
    result = {}
    for arm, offset in (('left', 0), ('right', 10)):
        a, b = reference[d:, offset:offset+10], candidate[d:, offset:offset+10]
        position = np.linalg.norm(a[:, :3]-b[:, :3], axis=-1)*1000
        ra, rb = _rotations(a[:, 3:9]), _rotations(b[:, 3:9])
        trace = np.einsum('...ij,...ij->...', ra, rb)
        angle = np.degrees(np.arccos(np.clip((trace-1)/2, -1, 1)))
        grip = np.abs(a[:, 9]-b[:, 9])
        step_error = np.linalg.norm(np.diff(a[:, :3], axis=0)-np.diff(b[:, :3], axis=0), axis=-1)*1000
        result[arm] = dict(position_mm=_summary(position), rotation_deg=_summary(angle),
            gripper_abs=_summary(grip),
            gripper_threshold_disagreement_fraction=float(np.mean((a[:, 9] >= .5) != (b[:, 9] >= .5))),
            first_suffix_position_mm=float(position[0]), first_suffix_rotation_deg=float(angle[0]),
            step_delta_error_mm=None if not len(step_error) else _summary(step_error))
    return result


def paired_run(engine, conditions, cache, *, synchronize=lambda: None, cached_first=False):
    """Same conditions/seed/weights; off/off measures numerical repeatability.

    The caller warms both paths before measuring. Conditions are copied because
    preprocessing/generation can attach arrays to a request dictionary.
    """
    original = engine._dit_cache
    records = {}
    order = ('on', 'off', 'off_repeat') if cached_first else ('off', 'off_repeat', 'on')
    try:
        for name in order:
            engine._dit_cache = cache if name == 'on' else None
            synchronize()
            started = time.perf_counter()
            result = engine.generate(deepcopy(conditions))
            synchronize()
            elapsed = time.perf_counter()-started
            records[name] = dict(seconds=elapsed, actions=np.array(result['actions'], copy=True),
                                 cache_stats=dict(cache.stats) if name == 'on' else None)
    finally:
        engine._dit_cache = original
    if records['on']['cache_stats'].get('total_steps', 0) == 0:
        raise ValueError('cache-on path was not exercised; cannot report a cache ablation')
    d = len(conditions.get('action_prefix', []))
    prefix = np.asarray(conditions.get('action_prefix', []), dtype=np.float32).reshape(d, 20)
    for record in records.values():
        if not np.array_equal(record['actions'][:d].astype(np.float32), prefix):
            raise ValueError('physical prefix was not preserved exactly')
    report = dict(d=d, seed=conditions.get('seed', 42),
        cache_error=compare_actions(records['off']['actions'], records['on']['actions'], d),
        baseline_repeatability=compare_actions(records['off']['actions'], records['off_repeat']['actions'], d),
        latency_s={key: record['seconds'] for key, record in records.items()},
        paired_speedup=records['off']['seconds']/records['on']['seconds'],
        cache_stats=records['on']['cache_stats'])
    return report, {key: record['actions'] for key, record in records.items()}
