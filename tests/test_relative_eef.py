import numpy as np
import pytest

from openwam.dataloader.utils.relative_eef import (
    absolute_to_relative_eef20, relative_to_absolute_eef20,
)


def pose(x, y, z, angle):
    c, s = np.cos(angle), np.sin(angle)
    value = np.zeros(20, dtype=np.float32)
    value[0:3] = [x, y, z]
    value[3:9] = [c, s, 0, -s, c, 0]
    value[9] = .4
    value[10:13] = [x + .1, y, z]
    value[13:19] = [c, s, 0, -s, c, 0]
    value[19] = .7
    return value


def test_relative_round_trip_and_gripper_is_absolute():
    state = pose(.1, -.2, .3, .3)
    target = pose(.2, -.1, .35, -.4)
    target[9], target[19] = .2, .8
    relative = absolute_to_relative_eef20(target, state)
    restored = relative_to_absolute_eef20(relative, state)
    np.testing.assert_allclose(restored, target, atol=2e-6)
    np.testing.assert_allclose(relative[[9, 19]], target[[9, 19]], atol=0)
    np.testing.assert_allclose(relative[[0, 1, 2]], target[[0, 1, 2]] - state[[0, 1, 2]], atol=2e-6)


def test_batch_reference_broadcast_and_identity_static():
    state = np.stack([pose(.1, 0, .2, 0), pose(.2, 0, .2, .2)])
    relative = absolute_to_relative_eef20(state, state)
    np.testing.assert_allclose(relative[:, :3], 0, atol=1e-7)
    np.testing.assert_allclose(relative[:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (2, 1)), atol=1e-7)
    restored = relative_to_absolute_eef20(relative, state)
    np.testing.assert_allclose(restored, state, atol=2e-6)


def test_rejects_bad_width():
    with pytest.raises(ValueError, match="20D"):
        absolute_to_relative_eef20(np.zeros(16), np.zeros(16))


def test_rotation_composition_order_and_base_axis_translation():
    from scipy.spatial.transform import Rotation
    state, target = pose(.1, .2, .3, 0), pose(.15, .3, .4, 0)
    expected = []
    for side, offset in enumerate((0, 10)):
        base = Rotation.from_euler("xyz", [.7, -.3, .5 + side]).as_matrix()
        delta = Rotation.from_euler("xyz", [-.2, .6, -.4]).as_matrix()
        goal = delta @ base
        state[offset + 3:offset + 9] = np.r_[base[:, 0], base[:, 1]]
        target[offset + 3:offset + 9] = np.r_[goal[:, 0], goal[:, 1]]
        expected.append(np.r_[delta[:, 0], delta[:, 1]])
    encoded = absolute_to_relative_eef20(target, state)
    for side, offset in enumerate((0, 10)):
        np.testing.assert_allclose(encoded[offset + 3:offset + 9], expected[side], atol=1e-6)
        np.testing.assert_allclose(encoded[offset:offset + 3], [.05, .1, .1], atol=1e-6)
    np.testing.assert_allclose(relative_to_absolute_eef20(encoded, state), target, atol=1e-6)


def test_chunk_broadcast_no_cross_arm_or_cross_reference_mix():
    references = np.stack([pose(.1, .2, .3, .7), pose(.5, .6, .7, -.4)])
    goals = np.stack([pose(.2, .3, .4, -.2), pose(.8, .5, .3, .1)])
    goals = np.repeat(goals[:, None], 32, axis=1)
    encoded = absolute_to_relative_eef20(goals, references[:, None])
    restored = relative_to_absolute_eef20(encoded, references[:, None])
    np.testing.assert_allclose(restored, goals, atol=1e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_and_degenerate_rotations_rejected(bad):
    state = pose(0, 0, 0, 0)
    invalid = state.copy(); invalid[3] = bad
    with pytest.raises(ValueError):
        relative_to_absolute_eef20(invalid, state)
    invalid = state.copy(); invalid[3:9] = 0
    with pytest.raises(ValueError, match="degenerate"):
        relative_to_absolute_eef20(invalid, state)


def test_statistics_reservoir_is_bounded_and_reproducible():
    from scripts.piper.compute_relative_eef_stats import Reservoir
    values = np.arange(40_000, dtype=np.float32).reshape(2_000, 20)
    first, second = Reservoir(size=37), Reservoir(size=37)
    for chunk in np.array_split(values, 7):
        first.update(chunk)
        second.update(chunk)
    assert first.count == len(values)
    assert first.sample().shape == (37, 20)
    np.testing.assert_array_equal(first.sample(), second.sample())
    assert first.sample().min() >= values.min()
    assert first.sample().max() <= values.max()
