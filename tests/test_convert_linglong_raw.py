import io
import json
import sqlite3
import struct
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from scripts.convert_linglong import convert_raw
from scripts.linglong_raw import iter_bags


def _align(buffer, size):
    buffer.extend(b"\0" * (-len(buffer) % size))


def _i32(buffer, value):
    _align(buffer, 4)
    buffer.extend(struct.pack("<i", value))


def _u32(buffer, value):
    _align(buffer, 4)
    buffer.extend(struct.pack("<I", value))


def _string(buffer, value):
    raw = value.encode() + b"\0"
    _u32(buffer, len(raw))
    buffer.extend(raw)


def _header(buffer, stamp):
    _i32(buffer, stamp // 1_000_000_000)
    _u32(buffer, stamp % 1_000_000_000)
    _string(buffer, "base_link")


def _driver(stamp, value):
    data = bytearray(b"\0\1\0\0")
    _header(data, stamp)
    _u32(data, 6)
    for count in (7, 7, 4, 2, 2, 2):
        _u32(data, count)
        for _ in range(count):
            data.extend(struct.pack("<6f", value, value + .1, 0, 0, 0, 0))
    return bytes(data)


def _end_pos(stamp):
    data = bytearray(b"\0\1\0\0")
    _header(data, stamp)
    for pose in range(6):
        values = [0.2 + pose * .01, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0]
        for value in values:
            _align(data, 4)
            data.extend(struct.pack("<f", value))
    return bytes(data)


def _image(stamp, jpeg):
    data = bytearray(b"\0\1\0\0")
    _header(data, stamp)
    _string(data, "jpeg")
    _u32(data, len(jpeg))
    data.extend(jpeg)
    return bytes(data)


def _make_bag(path, offset=0):
    image = Image.new("RGB", (8, 8), (100, 20, 30))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    jpeg = encoded.getvalue()
    topics = [
        ("/driver_pvt", "driver_pvt/msg/DriverPVT"),
        ("/end_pos", "end_pos/msg/EndPos"),
        *((f"/camera/{cam}/image_raw/compressed", "sensor_msgs/msg/CompressedImage") for cam in ("head", "left", "right")),
    ]
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)")
        con.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
        for index, (name, typ) in enumerate(topics, 1):
            con.execute("INSERT INTO topics VALUES(?,?,?,?,?)", (index, name, typ, "cdr", ""))
        for frame in range(70):
            stamp = 10_000_000_000 + frame * 33_333_333
            for topic_id, (name, _) in enumerate(topics, 1):
                if topic_id == 1:
                    payload = _driver(stamp, frame / 100 + offset)
                elif topic_id == 2:
                    payload = _end_pos(stamp)
                else:
                    payload = _image(stamp, jpeg)
                con.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES(?,?,?)",
                            (topic_id, stamp, payload))


class ConvertLinglongRawTest(unittest.TestCase):
    def test_nested_zip_to_portable_joint_and_eef_buckets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "bag_0.db3"
            _make_bag(bag)
            episode = root / "2026-01-01T0000.00"
            episode.mkdir()
            metadata = {"episode": "7", "taskName": "generic"}
            (episode / "__loongdata_metadata.json").write_text(json.dumps(metadata))
            tar_path = root / "episode.tar"
            with tarfile.open(tar_path, "w") as archive:
                archive.add(bag, arcname="bag/bag_0.db3")
                archive.add(episode / "__loongdata_metadata.json", arcname="__loongdata_metadata.json")
            raw_zip = root / "raw.zip"
            with zipfile.ZipFile(raw_zip, "w") as archive:
                archive.write(tar_path, "2026-01-01T0000.00/episode.tar")
            output = root / "output"

            report = convert_raw(raw_zip, output, prompt="Pick the block", test_fraction=0)

            self.assertEqual(report["splits"]["train"]["episodes"], 1)
            bucket = output / "linglong_openwam_train_15hz"
            table = pq.read_table(bucket / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(table.num_rows, 35)
            for column, width in (("observation.state", 24), ("action", 24),
                                  ("action_eef", 16), ("observation.state_eef", 16)):
                self.assertEqual(len(table[column][0].as_py()), width)
            manifest = json.loads((bucket / "manifest.json").read_text())
            self.assertEqual(manifest["split"], "train")
            self.assertEqual(manifest["observation_state_eef"]["kinematics_used"], False)
            for camera in ("head", "left", "right"):
                reference = table[f"observation.images.{camera}"][0].as_py()
                self.assertTrue((bucket / reference).is_file())
            self.assertTrue((bucket / "meta/action_eef_stats.json").is_file())
            self.assertTrue((bucket / "meta/state_eef_stats.json").is_file())

    def test_folder_config_split_and_duplicate_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            for episode_id, offset in (("7", 0), ("8", 10)):
                episode = raw / episode_id
                (episode / "bag").mkdir(parents=True)
                _make_bag(episode / "bag/bag_0.db3", offset=offset)
                (episode / "__loongdata_metadata.json").write_text(json.dumps({"episode": episode_id}))
            # Include an already-extracted episode and its archive together.
            with tarfile.open(raw / "duplicate.tar", "w") as archive:
                archive.add(raw / "7", arcname="7")
            config = {"7": {"prompt": "Train task", "split": "train"},
                      "8/bag/bag_0.db3": {"prompt": "Test task", "split": "test"}}
            output = root / "output"
            report = convert_raw(raw, output, episode_config=config)
            self.assertEqual(len(report["duplicates"]), 1)
            self.assertEqual(report["splits"], {"train": {"episodes": 1, "frames": 35},
                                                "test": {"episodes": 1, "frames": 35}})
            train = output / "linglong_openwam_train_15hz"
            test = output / "linglong_openwam_test_15hz"
            for filename in ("normalization_stats.json", "action_eef_stats.json", "state_eef_stats.json"):
                self.assertEqual((train / "meta" / filename).read_bytes(), (test / "meta" / filename).read_bytes())
            stats = json.loads((test / "meta/normalization_stats.json").read_text())
            self.assertLess(stats["action"]["max"][0], 1)  # Test values >10 must not enter train stats.
            table = pq.read_table(test / "data/chunk-000/episode_000000.parquet")
            self.assertGreater(table["action"][0].as_py()[0], 10)
            episodes = [json.loads(row) for row in (test / "meta/episodes.jsonl").read_text().splitlines()]
            self.assertEqual(episodes[0]["tasks"], ["Test task"])

    def test_rejects_ambiguous_metadata_episode_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            for name, offset in (("a", 0), ("b", 1)):
                episode = raw / name
                (episode / "bag").mkdir(parents=True)
                _make_bag(episode / "bag/bag_0.db3", offset=offset)
                (episode / "__loongdata_metadata.json").write_text('{"episode": "7"}')
            with self.assertRaisesRegex(ValueError, "ambiguous episode-config key 7"):
                convert_raw(raw, root / "out", episode_config={"7": {"prompt": "Task"}})

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../bag_0.db3", b"invalid")
            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                list(iter_bags(path))


if __name__ == "__main__":
    unittest.main()
