"""Decode the recorded end_pos CDR message fields needed by OpenWAM."""

import struct


class _CDRReader:
    def __init__(self, payload):
        if len(payload) < 4 or payload[:2] != b"\x00\x01":
            raise ValueError("unsupported CDR encapsulation (expected little-endian CDR)")
        self.data = payload
        self.pos = 4

    def _take(self, size):
        if self.pos + size > len(self.data):
            raise ValueError(f"truncated CDR payload at offset {self.pos} (need {size} bytes)")
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def _align(self, size):
        self.pos = (self.pos + size - 1) // size * size

    def i32(self):
        self._align(4)
        return struct.unpack("<i", self._take(4))[0]

    def u32(self):
        self._align(4)
        return struct.unpack("<I", self._take(4))[0]

    def f32(self):
        self._align(4)
        return struct.unpack("<f", self._take(4))[0]

    def string(self):
        size = self.u32()
        raw = self._take(size)
        if size and raw.endswith(b"\0"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def finish(self):
        tail = self.data[self.pos:]
        if any(tail):
            raise ValueError(f"unexpected trailing CDR data ({len(tail)} bytes)")


def decode_end_pos(payload):
    """Decode timestamps and measured/expected left and right end-effector poses."""
    reader = _CDRReader(payload)
    seconds, nanoseconds = reader.i32(), reader.u32()
    frame_id = reader.string()
    names = ("ee_pose_l", "ee_pose_l_exp", "ee_pose_r", "ee_pose_r_exp",
             "ee_pose_waist", "ee_pose_waist_exp")
    poses = {name: [reader.f32() for _ in range(7)] for name in names}
    reader.finish()
    return {"timestamp_ns": seconds * 1_000_000_000 + nanoseconds,
            "frame_id": frame_id, **poses}
