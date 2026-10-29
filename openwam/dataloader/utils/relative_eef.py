"""Relative-to-observation-state transforms for Piper-X EEF20."""

from __future__ import annotations

import numpy as np

RELATIVE_EEF_CONTRACT = "piper-x-relative-eef20-base-v1"
RELATIVE_STATS_FORMAT = "piper_x_relative_eef20_normalization_v1"
RELATIVE_SEMANTICS = {
    "action_contract": RELATIVE_EEF_CONTRACT,
    "reference": "input_state",
    "translation_frame": "per_arm_robot_base",
    "rotation_delta": "R_action_times_R_state_transpose",
    "gripper_target": "absolute",
    "state_target": "absolute_pose",
}


def validate_relative_metadata(metadata, *, num_frames=None, window_stride=None):
    for key, value in RELATIVE_SEMANTICS.items():
        if metadata.get(key) != value:
            raise ValueError(f"Relative EEF metadata {key} must be {value!r}")
    if num_frames is not None and metadata.get("num_frames") != num_frames:
        raise ValueError("Relative EEF statistics num_frames differs from training windows")
    if window_stride is not None and metadata.get("window_stride") != window_stride:
        raise ValueError("Relative EEF statistics window_stride differs from training windows")


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError("Piper-X rot6d must contain six finite values")
    first, second = value[..., :3], value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Piper-X rot6d first axis is degenerate")
    x_axis = first / first_norm
    second = second - np.sum(x_axis * second, axis=-1, keepdims=True) * x_axis
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Piper-X rot6d second axis is degenerate")
    y_axis = second / second_norm
    return np.stack((x_axis, y_axis, np.cross(x_axis, y_axis)), axis=-1)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def _validate_pair(values, state):
    values = np.asarray(values, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if values.ndim == 0 or state.ndim == 0 or values.shape[-1] != 20 or state.shape[-1] != 20:
        raise ValueError("Piper-X relative EEF requires 20D xyz+rot6d+gripper")
    if not np.isfinite(values).all() or not np.isfinite(state).all():
        raise ValueError("Piper-X EEF contains non-finite values")
    try:
        np.broadcast_shapes(values.shape[:-1], state.shape[:-1])
    except ValueError as exc:
        raise ValueError("Piper-X action and reference state shapes do not broadcast") from exc
    return np.broadcast_arrays(values, state)


def absolute_to_relative_eef20(absolute, reference_state) -> np.ndarray:
    """Encode action pose relative to one absolute state; gripper stays absolute.

    Translation differences use each arm's base axes. Rotation is
    ``R_action @ R_state.T`` in those same axes.
    """
    absolute, state = _validate_pair(absolute, reference_state)
    result = np.array(absolute, copy=True)
    for offset in (0, 10):
        ref_rotation = _rot6d_to_matrix(state[..., offset + 3:offset + 9])
        action_rotation = _rot6d_to_matrix(absolute[..., offset + 3:offset + 9])
        result[..., offset:offset + 3] = absolute[..., offset:offset + 3] - state[..., offset:offset + 3]
        relative_rotation = action_rotation @ np.swapaxes(ref_rotation, -1, -2)
        result[..., offset + 3:offset + 9] = _matrix_to_rot6d(relative_rotation)
    return result.astype(np.float32)


def relative_to_absolute_eef20(relative, reference_state) -> np.ndarray:
    """Decode relative EEF20 against the exact state used to generate it."""
    relative, state = _validate_pair(relative, reference_state)
    result = np.array(relative, copy=True)
    for offset in (0, 10):
        ref_rotation = _rot6d_to_matrix(state[..., offset + 3:offset + 9])
        relative_rotation = _rot6d_to_matrix(relative[..., offset + 3:offset + 9])
        result[..., offset:offset + 3] = relative[..., offset:offset + 3] + state[..., offset:offset + 3]
        absolute_rotation = relative_rotation @ ref_rotation
        result[..., offset + 3:offset + 9] = _matrix_to_rot6d(absolute_rotation)
    return result.astype(np.float32)
