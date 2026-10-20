"""Linglong joint-vector to EEF conversion used by dataset preparation."""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_INDICES = {
    **{f"left_{name}_joint": index for index, name in enumerate(ARM_PARTS)},
    **{f"right_{name}_joint": index + 8 for index, name in enumerate(ARM_PARTS)},
}
JOINT_INDICES = {
    **ARM_INDICES,
    "waist_1_joint": 16,
    "waist_2_joint": 17,
    "waist_3_joint": 18,
    "waist_yaw_joint": 19,
    "head_yaw_joint": 20,
    "head_pitch_joint": 21,
}
WRIST_TO_TCP = np.array([0.0813, 0.0, 0.0081], dtype=np.float64)
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "assets/linglong/kinematics.xml"


def _body_transform(body, batch):
    transform = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
    transform[:, :3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
    wxyz = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
    transform[:, :3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    return transform


class LinglongKinematics:
    """Forward kinematics for both Linglong arms and their TCP offsets."""

    def __init__(self, model_zip=DEFAULT_MODEL, wrist_to_tcp=WRIST_TO_TCP):
        model_path = Path(model_zip)
        if model_path.suffix.lower() == ".xml":
            root = ET.parse(model_path).getroot()
        else:
            with zipfile.ZipFile(model_path) as archive:
                root = ET.fromstring(archive.read("linglong/linglong-h-stack-parts.xml"))
        compiler = root.find("compiler")
        if compiler is None or compiler.get("angle") != "radian":
            raise ValueError("this converter expects MJCF angles in radians")
        self.parents, self.bodies = {}, {}
        for parent in root.iter():
            for child in parent:
                self.parents[child] = parent
                if child.tag == "body":
                    self.bodies[child.attrib["name"]] = child
        self.wrist_to_tcp = np.asarray(wrist_to_tcp, dtype=np.float64)
        if self.wrist_to_tcp.shape != (3,):
            raise ValueError("wrist_to_tcp must have shape (3,)")

    @staticmethod
    def _q24(q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim not in (1, 2) or not np.isfinite(q).all():
            raise ValueError("joint values must be finite, with shape (D,) or (N,D)")
        if q.shape[-1] == 22:
            padded = np.zeros(q.shape[:-1] + (24,), dtype=q.dtype)
            padded[..., :22] = q
            return padded
        if q.shape[-1] == 24:
            return q
        raise ValueError(f"expected q shape (*,22) or (*,24), got {q.shape}")

    def _fk_body(self, q, body_name):
        q = self._q24(q)
        batch = len(q)
        body = self.bodies[body_name]
        chain = []
        while body.tag == "body":
            chain.append(body)
            body = self.parents[body]
        transform = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
        for body in reversed(chain):
            transform = transform @ _body_transform(body, batch)
            for joint in body.findall("joint"):
                name = joint.attrib["name"]
                if name not in JOINT_INDICES:
                    raise ValueError(f"unmapped joint in EEF chain: {name}")
                if joint.attrib.get("type", "hinge") != "hinge":
                    raise ValueError(f"unsupported non-hinge joint {name}")
                if float(joint.get("ref", "0")) != 0 or not np.allclose(
                    np.fromstring(joint.get("pos", "0 0 0"), sep=" "), 0
                ):
                    raise ValueError(f"nonzero joint pivot/reference is not supported: {name}")
                axis = np.fromstring(joint.attrib["axis"], sep=" ")
                axis = axis / np.linalg.norm(axis)
                rotation = Rotation.from_rotvec(q[:, JOINT_INDICES[name], None] * axis).as_matrix()
                joint_transform = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
                joint_transform[:, :3, :3] = rotation
                transform = transform @ joint_transform
        return transform

    def _eef8(self, q, side):
        wrist = self._fk_body(q, f"{side}_wrist_yaw_link")
        position = wrist[:, :3, 3] + wrist[:, :3, :3] @ self.wrist_to_tcp
        xyzw = Rotation.from_matrix(wrist[:, :3, :3]).as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        gripper_index = 7 if side == "left" else 15
        q24 = self._q24(q)
        return np.concatenate((position, wxyz, q24[:, gripper_index, None]), axis=-1)

    def action_to_eef(self, action):
        action = np.asarray(action)
        one = action.ndim == 1
        q = action[None] if one else action
        q24 = self._q24(q)
        result = np.concatenate((self._eef8(q24, "left"), self._eef8(q24, "right")), axis=-1)
        return result[0] if one else result


def make_quaternion_continuous(values):
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[-1] != 16:
        raise ValueError("expected shape (frames,16)")
    for hand_start in (0, 8):
        for index in range(1, len(values)):
            if np.dot(values[index - 1, hand_start + 3:hand_start + 7],
                      values[index, hand_start + 3:hand_start + 7]) < 0:
                values[index, hand_start + 3:hand_start + 7] *= -1
    return values
