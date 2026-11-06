"""Check physical error units and paired replay isolation without a GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from openwam.deploy.rtc_cache_eval import compare_actions, paired_run


def chunk():
    result = np.zeros((32, 20), dtype=np.float32)
    for offset in (0, 10):
        result[:, offset+3:offset+9] = [1, 0, 0, 0, 1, 0]
    return result


def test_physical_units_and_prefix_exclusion():
    reference = chunk()
    candidate = reference.copy()
    candidate[:6, :3] = 100  # Frozen prefix is checked separately by paired_run.
    candidate[6:, 0] = .001
    candidate[6:, 3:9] = [0, 1, 0, -1, 0, 0]  # +90 degrees about Z.
    candidate[6:, 9] = .75
    metrics = compare_actions(reference, candidate, 6)
    assert metrics['left']['position_mm']['mean'] == pytest.approx(1)
    assert metrics['left']['rotation_deg']['max'] == pytest.approx(90)
    assert metrics['left']['gripper_threshold_disagreement_fraction'] == 1
    assert metrics['left']['step_delta_error_mm']['max'] == 0
    assert metrics['right']['position_mm']['max'] == 0


@pytest.mark.parametrize('invalid', ['shape', 'nan', 'rotation'])
def test_invalid_output_cannot_produce_accuracy_report(invalid):
    candidate = chunk()
    if invalid == 'shape':
        candidate = candidate[:-1]
    elif invalid == 'nan':
        candidate[8, 0] = np.nan
    else:
        candidate[8, 3:9] = 0
    with pytest.raises(ValueError):
        compare_actions(chunk(), candidate, 6)


class Engine:
    def __init__(self):
        self._dit_cache = object()
        self.fail = False
        self.corrupt_prefix = False

    def generate(self, conditions):
        if self.fail:
            raise RuntimeError('failed inference')
        assert conditions['seed'] == 19
        actions = chunk()
        d = len(conditions['action_prefix'])
        actions[:d] = conditions['action_prefix']
        conditions['action_prefix'][:] = 0  # Mutations must not leak to next run.
        if self._dit_cache is not None:
            actions[d:, 0] += .001
        if self.corrupt_prefix:
            actions[0, 0] += 1
        return {'actions': actions}


@pytest.mark.parametrize('cached_first', [False, True])
def test_paired_replay_preserves_context_and_reports_repeatability(cached_first):
    engine = Engine()
    original = engine._dit_cache
    prefix = chunk()[:6]
    prefix[:, 0] = .25
    conditions = {'seed': 19, 'action_prefix': prefix}
    report, outputs = paired_run(engine, conditions, SimpleNamespace(stats={'total_steps': 5, 'total_skips': 3}),
                                 cached_first=cached_first)
    assert engine._dit_cache is original
    assert (conditions['action_prefix'][:, 0] == .25).all()
    assert report['cache_error']['left']['position_mm']['max'] == pytest.approx(1)
    assert report['baseline_repeatability']['left']['position_mm']['max'] == 0
    np.testing.assert_array_equal(outputs['on'][:6], prefix)


def test_paired_replay_restores_cache_on_failure_and_rejects_corrupt_prefix():
    engine = Engine()
    original = engine._dit_cache
    conditions = {'seed': 19, 'action_prefix': chunk()[:6]}
    engine.fail = True
    with pytest.raises(RuntimeError, match='failed inference'):
        paired_run(engine, conditions, SimpleNamespace(stats={'total_steps': 5}))
    assert engine._dit_cache is original
    engine.fail = False
    engine.corrupt_prefix = True
    with pytest.raises(ValueError, match='prefix was not preserved'):
        paired_run(engine, conditions, SimpleNamespace(stats={'total_steps': 5}))
    assert engine._dit_cache is original


def test_paired_replay_refuses_a_silently_disabled_cache():
    with pytest.raises(ValueError, match='was not exercised'):
        paired_run(Engine(), {'seed': 19, 'action_prefix': chunk()[:6]},
                   SimpleNamespace(stats={'total_steps': 0}))
