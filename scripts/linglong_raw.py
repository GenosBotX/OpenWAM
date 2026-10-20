"""Local ROS2 CDR/SQLite and archive readers for Linglong raw recordings.

DriverPVT: Header, sequence<Limb>; Limb: sequence<Joint>;
Joint: float32 q, q_exp, qd, qd_exp, tau, tau_exp.
CompressedImage: Header, string format, sequence<uint8> data.
EndPos is decoded by the sibling module. No ROS installation is required.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import struct
import tarfile
import tempfile
import zipfile

if __package__:
    from .linglong_end_pos import _CDRReader, decode_end_pos
else:
    from linglong_end_pos import _CDRReader, decode_end_pos


TOPICS = {
    "/driver_pvt": ("driver", "driver_pvt/msg/DriverPVT"),
    "/end_pos": ("end_pos", "end_pos/msg/EndPos"),
    **{f"/camera/{cam}/image_raw/compressed": (cam, "sensor_msgs/msg/CompressedImage")
       for cam in ("head", "left", "right")},
}


def decode_driver(payload):
    reader = _CDRReader(payload)
    stamp = reader.i32() * 1_000_000_000 + reader.u32()
    frame_id = reader.string()
    limbs = []
    for _ in range(reader.u32()):
        joints = []
        for _ in range(reader.u32()):
            joints.append(struct.unpack("<6f", reader._take(24)))
        limbs.append(joints)
    reader.finish()
    if [len(limb) for limb in limbs] != [7, 7, 4, 2, 2, 2]:
        raise ValueError("expected DriverPVT limb lengths [7,7,4,2,2,2]")
    ordered = [*limbs[0], limbs[4][0], *limbs[1], limbs[4][1],
               *limbs[2], *limbs[3], *limbs[5]]
    return {"timestamp_ns": stamp, "frame_id": frame_id,
            "q": [joint[0] for joint in ordered], "q_exp": [joint[1] for joint in ordered]}


def decode_image(payload):
    reader = _CDRReader(payload)
    stamp = reader.i32() * 1_000_000_000 + reader.u32()
    frame_id = reader.string()
    fmt = reader.string()
    data = reader._take(reader.u32())
    reader.finish()
    return {"timestamp_ns": stamp, "frame_id": frame_id, "format": fmt, "data": data}


def read_bag(path):
    path = Path(path)
    streams = {kind: [] for kind, _ in TOPICS.values()}
    # mode=ro is essential: a typo must never create an empty SQLite file.
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        topics = {}
        for tid, name, typ, fmt in con.execute("SELECT id,name,type,serialization_format FROM topics"):
            if name not in TOPICS:
                continue
            kind, expected = TOPICS[name]
            if typ != expected or fmt != "cdr":
                raise ValueError(f"{path}: unsupported topic schema {name}: {typ}/{fmt}")
            if kind in topics.values():
                raise ValueError(f"{path}: repeated topic {name}")
            topics[tid] = kind
        for tid, data in con.execute("SELECT topic_id,data FROM messages ORDER BY timestamp,id"):
            kind = topics.get(tid)
            if kind is None:
                continue
            decode = decode_driver if kind == "driver" else decode_end_pos if kind == "end_pos" else decode_image
            record = decode(data)
            previous = streams[kind]
            if previous and record["timestamp_ns"] <= previous[-1]["timestamp_ns"]:
                raise ValueError(f"{path}: non-monotonic or duplicate {kind} header timestamps")
            previous.append(record)
    for kind, records in streams.items():
        if not records:
            raise ValueError(f"{path}: missing/empty {kind} stream")
    return streams


def _archive(path):
    return path.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"unsafe archive path: {name}")
    return path


def _metadata(bag):
    path = bag.parent.parent / "__loongdata_metadata.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def iter_bags(source, scratch=None, *, identity=None, depth=0):
    """Yield (logical source ID, bag path, metadata) while temp files exist.

    Support directories, db3, tar(.gz) and zip with nested per-episode tars.
    Extract one nested archive at a time; ignore unrelated archive files.
    Never follow archive links or use original absolute paths from metadata.
    """
    source = Path(source)
    if depth > 6:
        raise ValueError("raw archive nesting exceeds six levels")
    if source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.is_file() and (p.suffix == ".db3" or _archive(p)))
        for path in paths:
            rel = path.relative_to(source).as_posix()
            yield from iter_bags(path, scratch, identity=rel, depth=depth + 1)
        return
    label = identity or source.name
    if source.suffix == ".db3":
        yield label, source, _metadata(source)
        return
    if not _archive(source):
        raise ValueError(f"expected raw directory, zip, tar(.gz), or .db3: {source}")
    with tempfile.TemporaryDirectory(prefix="openwam-linglong-", dir=scratch) as directory:
        root = Path(directory)
        if source.suffix.lower() == ".zip":
            archive = zipfile.ZipFile(source)
            entries = [(m.filename, m.is_dir(), stat.S_ISLNK(m.external_attr >> 16), m)
                       for m in archive.infolist()]
            opener = archive.open
        else:
            archive = tarfile.open(source, "r:*")
            entries = [(m.name, m.isdir(), not (m.isfile() or m.isdir()), m) for m in archive.getmembers()]
            opener = archive.extractfile
        try:
            seen = set()
            for name, is_dir, is_link, entry in entries:
                rel = _safe_member(name)
                if is_link:
                    raise ValueError(f"archive links/special files are unsupported: {name}")
                if is_dir:
                    continue
                if rel.as_posix() in seen:
                    raise ValueError(f"repeated archive member: {name}")
                seen.add(rel.as_posix())
                if rel.suffix != ".db3" and rel.name != "__loongdata_metadata.json":
                    continue
                dest = root.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with opener(entry) as inp, dest.open("xb") as out:
                    shutil.copyfileobj(inp, out)
            for bag in sorted(root.rglob("*.db3")):
                yield f"{label}!{bag.relative_to(root).as_posix()}", bag, _metadata(bag)
            for name, is_dir, _, entry in entries:
                rel = _safe_member(name)
                if is_dir or not _archive(rel):
                    continue
                # One temp file per nested archive, removed after its bags are consumed.
                with tempfile.TemporaryDirectory(prefix="nested-", dir=root) as nested:
                    path = Path(nested) / rel.name
                    with opener(entry) as inp, path.open("xb") as out:
                        shutil.copyfileobj(inp, out)
                    yield from iter_bags(path, scratch, identity=f"{label}!{rel.as_posix()}", depth=depth + 1)
        finally:
            archive.close()
