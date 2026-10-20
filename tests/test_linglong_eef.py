import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.linglong_eef import LinglongKinematics, WRIST_TO_TCP


class LinglongKinematicsTest(unittest.TestCase):
    def test_rotates_tcp_offset_and_preserves_grippers(self):
        xml = '''<mujoco><compiler angle="radian"/><worldbody><body name="base_link">
          <body name="left_wrist_yaw_link" pos="0.3 0.2 0.5">
            <joint name="left_shoulder_pitch_joint" axis="0 0 1"/>
          </body>
          <body name="right_wrist_yaw_link" pos="0.3 -0.2 0.5">
            <joint name="right_shoulder_pitch_joint" axis="0 0 1"/>
          </body>
        </body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.zip"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("linglong/linglong-h-stack-parts.xml", xml)
            fk = LinglongKinematics(model_path)
            action = np.zeros((2, 24))
            action[:, 0] = np.pi / 2
            action[:, 7] = [0.25, 0.75]
            action[:, 15] = [0.75, 0.25]
            before = action.copy()
            result = fk.action_to_eef(action)

        expected_left = np.array([0.3, 0.2 + WRIST_TO_TCP[0], 0.5 + WRIST_TO_TCP[2]])
        np.testing.assert_allclose(result[:, :3], np.tile(expected_left, (2, 1)))
        expected_right = np.array([0.3, -0.2, 0.5]) + WRIST_TO_TCP
        np.testing.assert_allclose(result[:, 8:11], np.tile(expected_right, (2, 1)))
        np.testing.assert_allclose(result[:, 3:7], np.tile([2 ** -0.5, 0, 0, 2 ** -0.5], (2, 1)))
        np.testing.assert_array_equal(result[:, [7, 15]], action[:, [7, 15]])
        np.testing.assert_array_equal(action, before)


if __name__ == "__main__":
    unittest.main()
