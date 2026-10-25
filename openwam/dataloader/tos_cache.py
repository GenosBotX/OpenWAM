"""Process-safe, byte-bounded TOS file cache (Linux local disk only).

Dataset prefixes must be immutable. All ranks on a node use the same cache
directory and limits. File leases protect readers from concurrent eviction;
reservations include downloads in progress. Credentials never enter the index.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import time
import threading
from urllib.parse import urlsplit


def parse_tos_uri(uri):
    parsed = urlsplit(str(uri))
    if parsed.scheme != "tos" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Expected tos://bucket/key, got {uri!r}")
    key = parsed.path.lstrip("/").rstrip("/")
    if any(p in (".", "..") for p in key.split("/")):
        raise ValueError("TOS paths cannot contain dot components")
    return parsed.netloc, key


def join_tos(root, relative):
    rel = PurePosixPath(str(relative))
    if rel.is_absolute() or ".." in rel.parts or ":" in str(relative) or "\\" in str(relative):
        raise ValueError(f"Expected a safe relative object path: {relative!r}")
    return str(root).rstrip("/") + "/" + str(rel)


class TosBackend:
    """Lazily create one SDK client per process, including forked workers."""

    def __init__(self):
        self._pid = None
        self._client = None

    def __getstate__(self):
        return {"_pid": None, "_client": None}

    @property
    def client(self):
        if self._pid != os.getpid():
            import tos

            logging.getLogger("tos").setLevel(logging.WARNING)
            self._client = tos.TosClientV2(
                os.environ["TOS_ACCESS_KEY"], os.environ["TOS_SECRET_KEY"],
                os.environ.get("TOS_ENDPOINT", "tos-cn-beijing.volces.com"),
                os.environ.get("TOS_REGION", "cn-beijing"),
                security_token=os.environ.get("TOS_SECURITY_TOKEN"),
            )
            self._pid = os.getpid()
        return self._client

    def head(self, uri):
        bucket, key = parse_tos_uri(uri)
        result = self.client.head_object(bucket, key)
        return int(result.content_length), result.etag

    def download(self, uri, destination, etag):
        bucket, key = parse_tos_uri(uri)
        result = self.client.get_object(bucket, key, if_match=etag)
        try:
            with open(destination, "wb") as out:
                while True:
                    chunk = result.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        finally:
            stream = result.content
            # SDK CRC adapters wrap Response in .data; Response wraps requests
            # in .resp. Close it on both success and partial-download failure.
            for _ in range(8):
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
                    break
                nested = getattr(stream, "data", getattr(stream, "resp", None))
                if nested is None:
                    break
                stream = nested


class CacheFullError(RuntimeError):
    pass


class BoundedTosCache:
    def __init__(self, root, max_bytes, min_free_bytes=1024**3, backend=None):
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = int(max_bytes)
        self.min_free_bytes = int(min_free_bytes)
        if self.max_bytes <= 0 or self.min_free_bytes < 0:
            raise ValueError("Cache max_bytes must be positive, min_free_bytes nonnegative")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "locks").mkdir(exist_ok=True)
        self.backend = backend or TosBackend()
        self._db = None
        self._db_owner = None
        # A single cross-process lock protects the index and quota reservations.
        with self._index() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (value TEXT)")
            config = json.dumps([self.max_bytes, self.min_free_bytes])
            row = db.execute("SELECT value FROM config").fetchone()
            if row and row[0] != config:
                raise ValueError("All processes sharing a cache must use identical limits; use another directory")
            if not row:
                db.execute("INSERT INTO config VALUES (?)", (config,))
            db.execute("CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, size INTEGER, atime REAL, ready INTEGER)")
            db.execute("CREATE INDEX IF NOT EXISTS entries_lru ON entries(atime,id)")
            db.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER)")
            # Keep quota accounting in the same transaction as every entry change.
            # Scanning all cached images on each miss serializes every worker.
            db.execute("CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY CHECK(id=1), total INTEGER NOT NULL, inflight INTEGER NOT NULL, objects INTEGER NOT NULL)")
            if db.execute("SELECT 1 FROM usage WHERE id=1").fetchone() is None:
                db.execute("INSERT INTO usage SELECT 1, COALESCE(SUM(size),0), COALESCE(SUM(CASE WHEN ready=0 THEN size ELSE 0 END),0), COUNT(*) FROM entries")
            db.execute("""CREATE TRIGGER IF NOT EXISTS usage_insert AFTER INSERT ON entries BEGIN
                UPDATE usage SET total=total+NEW.size,
                    inflight=inflight+CASE WHEN NEW.ready=0 THEN NEW.size ELSE 0 END,
                    objects=objects+1 WHERE id=1;
                END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS usage_delete AFTER DELETE ON entries BEGIN
                UPDATE usage SET total=total-OLD.size,
                    inflight=inflight-CASE WHEN OLD.ready=0 THEN OLD.size ELSE 0 END,
                    objects=objects-1 WHERE id=1;
                END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS usage_update AFTER UPDATE OF size,ready ON entries BEGIN
                UPDATE usage SET total=total+NEW.size-OLD.size,
                    inflight=inflight+CASE WHEN NEW.ready=0 THEN NEW.size ELSE 0 END
                        -CASE WHEN OLD.ready=0 THEN OLD.size ELSE 0 END WHERE id=1;
                END""")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_db"] = None
        state["_db_owner"] = None
        return state

    @contextlib.contextmanager
    def _index(self):
        with open(self.root / "index.lock", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owner = (os.getpid(), threading.get_ident())
            if self._db_owner != owner:
                # Never use a SQLite connection inherited from a parent worker.
                if self._db is not None:
                    self._db.close()
                self._db = None
                self._db_owner = None
                db = sqlite3.connect(self.root / "index.sqlite3", timeout=60, check_same_thread=False)
                try:
                    db.execute("PRAGMA journal_mode=WAL")
                    # Keep crash durability; no NORMAL/OFF durability shortcut.
                    db.execute("PRAGMA synchronous=FULL")
                except BaseException:
                    db.close()
                    raise
                self._db, self._db_owner = db, owner
            with self._db:
                yield self._db

    @staticmethod
    def _count(db, name, amount=1):
        db.execute("INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (name, amount))

    def _reserve(self, db, identity, size):
        if size > self.max_bytes:
            raise CacheFullError(f"Object size {size} exceeds cache limit {self.max_bytes}")
        used, inflight = db.execute("SELECT total,inflight FROM usage WHERE id=1").fetchone()
        def enough():
            # Remaining in-flight bytes also consume future physical space.
            return used + size <= self.max_bytes and shutil.disk_usage(self.root).free - inflight - size >= self.min_free_bytes
        last_time, last_id = -1, ""
        while not enough():
            victims = db.execute("SELECT id,size,ready,atime FROM entries WHERE (atime,id) > (?,?) ORDER BY atime,id LIMIT 128", (last_time, last_id)).fetchall()
            if not victims:
                break
            for victim, victim_size, ready, accessed in victims:
                last_time, last_id = accessed, victim
                if enough():
                    break
                if victim == identity:
                    continue
                with open(self._lock_path(victim), "a+b") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    for suffix in (".data", ".part"):
                        (self.root / (victim + suffix)).unlink(missing_ok=True)
                    db.execute("DELETE FROM entries WHERE id=?", (victim,))
                    used -= victim_size
                    if not ready:
                        inflight -= victim_size
                    self._count(db, "evictions")
        if not enough():
            raise CacheFullError("Cache quota/free-space guard reached and remaining files are leased; increase limit or reduce concurrency")
        db.execute("INSERT INTO entries VALUES (?,?,?,0)", (identity, size, time.time()))

    def _lock_path(self, identity):
        # A fixed stripe set avoids one permanent inode per image ever seen.
        return self.root / "locks" / (identity[:3] + ".lock")

    @contextlib.contextmanager
    def open(self, uri):
        parse_tos_uri(uri)
        identity = hashlib.sha256(str(uri).encode()).hexdigest()
        data = self.root / (identity + ".data")
        partial = self.root / (identity + ".part")
        with open(self._lock_path(identity), "a+b") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            with self._index() as db:
                row = db.execute("SELECT size,ready FROM entries WHERE id=?", (identity,)).fetchone()
                hit = row and row[1] and data.is_file() and data.stat().st_size == row[0]
                if hit:
                    db.execute("UPDATE entries SET atime=? WHERE id=?", (time.time(), identity))
                    self._count(db, "hits")
                else:
                    db.execute("DELETE FROM entries WHERE id=?", (identity,))
                    data.unlink(missing_ok=True)
                    partial.unlink(missing_ok=True)
            if not hit:
                size, etag = self.backend.head(uri)
                with self._index() as db:
                    self._reserve(db, identity, size)
                    self._count(db, "misses")
                began = time.monotonic()
                try:
                    self.backend.download(uri, partial, etag)
                    if partial.stat().st_size != size:
                        raise IOError("Incomplete TOS download")
                    os.replace(partial, data)
                    with self._index() as db:
                        db.execute("UPDATE entries SET ready=1,atime=? WHERE id=?", (time.time(), identity))
                        self._count(db, "download_bytes", size)
                        self._count(db, "download_ms", int(1000 * (time.monotonic() - began)))
                except BaseException:
                    partial.unlink(missing_ok=True)
                    data.unlink(missing_ok=True)
                    with self._index() as db:
                        db.execute("DELETE FROM entries WHERE id=?", (identity,))
                        self._count(db, "errors")
                    raise
            # Shared leases allow concurrent hits, while blocking eviction.
            fcntl.flock(lease, fcntl.LOCK_SH)
            with open(data, "rb") as handle:
                yield handle

    def stats(self):
        with self._index() as db:
            result = dict(db.execute("SELECT name,value FROM counters"))
            result["reserved_bytes"], result["inflight_bytes"], result["objects"] = db.execute(
                "SELECT total,inflight,objects FROM usage WHERE id=1"
            ).fetchone()
            result["max_bytes"] = self.max_bytes
            return result


class DatasetStorage:
    """Local paths and TOS objects share the same scoped file interface."""

    def __init__(self, root, cache=None):
        self.root = str(root)
        self.remote = self.root.startswith("tos://")
        self.cache = cache
        if self.remote:
            parse_tos_uri(self.root)
            if cache is None:
                raise ValueError("TOS storage requires a bounded cache")

    @contextlib.contextmanager
    def open(self, relative):
        if self.remote:
            with self.cache.open(join_tos(self.root, relative)) as handle:
                yield handle
        else:
            with open(Path(self.root) / relative, "rb") as handle:
                yield handle

    def text(self, relative):
        with self.open(relative) as handle:
            return handle.read().decode("utf-8")
