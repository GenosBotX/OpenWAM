import struct
import unittest

from scripts.linglong_end_pos import decode_end_pos


class LinglongEndPosTest(unittest.TestCase):
    def test_decodes_timestamp_and_pose_fields(self):
        payload = bytearray(b"\x00\x01\x00\x00")
        payload.extend(struct.pack("<iI", 123, 456))
        frame_id = b"base_link\0"
        payload.extend(struct.pack("<I", len(frame_id)))
        payload.extend(frame_id)
        payload.extend(b"\0" * (-len(payload) % 4))
        expected = {}
        for name in ("ee_pose_l", "ee_pose_l_exp", "ee_pose_r", "ee_pose_r_exp",
                     "ee_pose_waist", "ee_pose_waist_exp"):
            values = [float(len(expected) * 10 + index) for index in range(7)]
            payload.extend(struct.pack("<7f", *values))
            expected[name] = values

        decoded = decode_end_pos(bytes(payload))

        self.assertEqual(decoded["timestamp_ns"], 123_000_000_456)
        self.assertEqual(decoded["frame_id"], "base_link")
        for name, values in expected.items():
            self.assertEqual(decoded[name], values)

    def test_rejects_wrong_encapsulation(self):
        with self.assertRaisesRegex(ValueError, "unsupported CDR encapsulation"):
            decode_end_pos(b"\x00\x02\x00\x00")


if __name__ == "__main__":
    unittest.main()
