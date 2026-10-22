"""Piper-X EEF20 协议校验；模型加载、图像预处理和归一化复用原生服务。"""

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.utils.poses import rot6d_to_quat_wxyz
from openwam.deploy.obs_preprocess import ObsValidationError
from openwam.deploy.server import PolicyServer


EEF_DIM = 20
MODEL_DIM = 80
UNIFIED_INDICES = np.r_[0:10, 34:44]


def validate_state(state):
    """只校验原始 EEF20，不修改、归一化或补偿输入位姿。"""
    try:
        values = np.asarray(state, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ObsValidationError("Piper-X state must be a numeric EEF20 list") from exc
    if values.shape != (EEF_DIM,) or not np.isfinite(values).all():
        raise ObsValidationError(
            f"Piper-X requires finite raw EEF20 state shape (20,), got {values.shape}; "
            "use xyz + rot6d + gripper per arm, left then right"
        )
    if np.any((values[[9, 19]] < 0) | (values[[9, 19]] > 1)):
        raise ObsValidationError("Piper-X grippers at indices 9,19 must be in [0,1] (0 closed,1 open)")
    try:
        for start in (3, 13):
            rot6d_to_quat_wxyz(values[start:start + 6])
    except ValueError as exc:
        raise ObsValidationError(f"Invalid Piper-X state rotation: {exc}") from exc


class PiperXPolicyServer(PolicyServer):
    """限定 Piper-X 原生 EEF20 checkpoint，补充握手与输入输出校验。"""

    def __init__(self, engine, cfg, readiness_token=None):
        super().__init__(engine, cfg, readiness_token)
        expected = {
            "dataloader.type": "robodojo",
            "dataloader.variant": "real",
            "dataloader.embodiment": "piper_x",
            "dataloader.action_mode": "eef",
            "dataloader.unify_action": True,
            "dataloader.normalize_mode": "min-max",
            "model.architecture.state_dim": MODEL_DIM,
            "model.architecture.action_dim": MODEL_DIM,
        }
        for field, value in expected.items():
            if OmegaConf.select(cfg, field) != value:
                raise ValueError(f"Piper-X requires {field}={value!r}")
        normalizer = engine.architecture.normalizer
        for attr in ("_action_dst_index", "_state_dst_index"):
            if not np.array_equal(getattr(normalizer, attr, None), UNIFIED_INDICES):
                raise ValueError(f"Piper-X EEF20 requires {attr}=[0..9,34..43]")
        if np.asarray(normalizer.stats["mean"]).shape != (EEF_DIM,):
            raise ValueError("Piper-X requires its own 20D normalization statistics")

    def _ckpt_contract(self):
        return {
            **super()._ckpt_contract(),
            "embodiment": "piper_x",
            "contract_id": "robodojo-real-native-eef20-v1",
            "state_dim": EEF_DIM,
            "action_dim": EEF_DIM,
            "model_state_dim": MODEL_DIM,
            "model_action_dim": MODEL_DIM,
            "num_actions": int(OmegaConf.select(self.cfg, "inference.num_frames")) - 1,
            "eef_layout": [
                "left.xyz", "left.rot6d", "left.gripper",
                "right.xyz", "right.rot6d", "right.gripper",
            ],
            "rotation_representation": "rot6d_first_two_columns_concatenated",
            "position_units": "meters",
            "target_frame": "per_arm_robot_base",
            "endpoint": "recorded_ee_pose",
            "action_target": "absolute_pose",
            "gripper_convention": "zero_closed_one_open",
            "normalization": "server_side_min_max",
        }

    def predict(self, obs):
        validate_state(obs.get("state"))
        result = super().predict(obs)
        actions = np.asarray(result["action"])
        expected_shape = (EEF_DIM,)
        if self.action_response_mode == "chunk":
            expected_shape = (int(OmegaConf.select(self.cfg, "inference.num_frames")) - 1, EEF_DIM)
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise ValueError(f"Piper-X requires finite physical EEF20 actions with shape {expected_shape}")
        return result
