import multiprocessing as mp
from pathlib import Path
import time

import pytest

from openwam.dataloader.tos_cache import BoundedTosCache, CacheFullError, join_tos


class FilesBackend:
    def __init__(self, source, broken=False):
        self.source = Path(source)
        self.broken = broken

    def head(self, uri):
        return (self.source / uri.rsplit("/", 1)[1]).stat().st_size, "immutable"

    def download(self, uri, destination, etag):
        with open(self.source / "downloads", "ab") as counter:
            counter.write(b"1")
        time.sleep(0.02)
        data = (self.source / uri.rsplit("/", 1)[1]).read_bytes()
        Path(destination).write_bytes(data[:1] if self.broken else data)


def worker(root, source, queue):
    try:
        cache = BoundedTosCache(root, 16, 0, FilesBackend(source))
        for _ in range(5):
            with cache.open("tos://bucket/a") as f:
                assert f.read() == b"abcd"
        queue.put(None)
    except BaseException as exc:
        queue.put(repr(exc))


def test_multiprocess_deduplicates_downloads(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_bytes(b"abcd")
    ctx = mp.get_context("fork")
    queue = ctx.SimpleQueue()
    processes = [ctx.Process(target=worker, args=(tmp_path / "cache", source, queue)) for _ in range(4)]
    for process in processes:
        process.start()
    results = [queue.get() for _ in processes]
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert results == [None] * 4
    assert (source / "downloads").read_bytes() == b"1"
    stats = BoundedTosCache(tmp_path / "cache", 16, 0, FilesBackend(source)).stats()
    assert stats["hits"] == 19 and stats["misses"] == 1


def test_eviction_quota_and_reader_lease(tmp_path):
    for name in "abc":
        (tmp_path / name).write_bytes(name.encode() * 4)
    cache = BoundedTosCache(tmp_path / "cache", 4, 0, FilesBackend(tmp_path))
    with cache.open("tos://bucket/a") as pinned:
        with pytest.raises(CacheFullError):
            with cache.open("tos://bucket/b"):
                pass
        assert pinned.read() == b"aaaa"
    for name in "bca":
        with cache.open("tos://bucket/" + name) as f:
            assert f.read() == name.encode() * 4
        assert sum(p.stat().st_size for p in cache.root.glob("*.data")) <= 4
    assert cache.stats()["evictions"] == 3


def test_failed_download_cleans_reservations(tmp_path):
    (tmp_path / "a").write_bytes(b"abcd")
    cache = BoundedTosCache(tmp_path / "cache", 8, 0, FilesBackend(tmp_path, broken=True))
    with pytest.raises(IOError):
        with cache.open("tos://bucket/a"):
            pass
    assert cache.stats()["reserved_bytes"] == 0
    assert not list(cache.root.glob("*.part"))
    cache.backend.broken = False
    with cache.open("tos://bucket/a") as f:
        assert f.read() == b"abcd"


def test_oversize_and_mismatched_limits(tmp_path):
    (tmp_path / "a").write_bytes(b"a" * 9)
    cache = BoundedTosCache(tmp_path / "cache", 8, 0, FilesBackend(tmp_path))
    with pytest.raises(CacheFullError):
        with cache.open("tos://bucket/a"):
            pass
    with pytest.raises(ValueError, match="identical limits"):
        BoundedTosCache(cache.root, 16, 0)


@pytest.mark.parametrize("path", ["../secret", "/absolute", "a/../../x", "tos://other/file", "a\\b"])
def test_remote_path_traversal_rejected(path):
    with pytest.raises(ValueError):
        join_tos("tos://bucket/version", path)


def assert_usage_matches_entries(cache):
    with cache._index() as db:
        tracked = db.execute("SELECT total,inflight,objects FROM usage WHERE id=1").fetchone()
        actual = db.execute("SELECT COALESCE(SUM(size),0),COALESCE(SUM(CASE WHEN ready=0 THEN size ELSE 0 END),0),COUNT(*) FROM entries").fetchone()
        assert tracked == actual


def test_usage_survives_failure_eviction_and_reopen(tmp_path):
    for name in "ab":
        (tmp_path / name).write_bytes(name.encode() * 4)
    backend = FilesBackend(tmp_path, broken=True)
    cache = BoundedTosCache(tmp_path / "cache", 4, 0, backend)
    with pytest.raises(IOError):
        with cache.open("tos://bucket/a"):
            pass
    assert_usage_matches_entries(cache)
    backend.broken = False
    for name in "aba":
        with cache.open("tos://bucket/" + name) as f:
            assert f.read() == name.encode() * 4
        assert_usage_matches_entries(cache)
    reopened = BoundedTosCache(cache.root, 4, 0, backend)
    assert_usage_matches_entries(reopened)
    assert reopened.stats()["reserved_bytes"] == 4
    assert reopened.stats()["inflight_bytes"] == 0


def test_migrate_legacy_index_and_account_inflight(tmp_path):
    import sqlite3
    root = tmp_path / "cache"
    root.mkdir()
    with sqlite3.connect(root / "index.sqlite3") as db:
        db.execute("CREATE TABLE entries (id TEXT PRIMARY KEY,size INTEGER,atime REAL,ready INTEGER)")
        db.executemany("INSERT INTO entries VALUES (?,?,?,?)", [("old",4,0,1),("pending",3,0,0)])
    cache = BoundedTosCache(root, 16, 0, FilesBackend(tmp_path))
    assert_usage_matches_entries(cache)
    assert cache.stats()["reserved_bytes"] == 7
    assert cache.stats()["inflight_bytes"] == 3
    with cache._index() as db:
        db.execute("UPDATE entries SET ready=1 WHERE id='pending'")
        db.execute("DELETE FROM entries WHERE id='old'")
    assert_usage_matches_entries(cache)
    assert cache.stats()["reserved_bytes"] == 3
    assert cache.stats()["inflight_bytes"] == 0


def use_inherited_cache(cache, queue):
    try:
        with cache.open("tos://bucket/a") as f:
            assert f.read() == b"abcd"
        queue.put((None, cache.stats()["reserved_bytes"]))
    except BaseException as exc:
        queue.put((repr(exc), None))


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_live_connection_is_not_reused_by_child(tmp_path, method):
    (tmp_path / "a").write_bytes(b"abcd")
    cache = BoundedTosCache(tmp_path / "cache", 8, 0, FilesBackend(tmp_path))
    with cache.open("tos://bucket/a") as f:
        assert f.read() == b"abcd"
    ctx = mp.get_context(method)
    queue = ctx.Queue()
    child = ctx.Process(target=use_inherited_cache, args=(cache, queue))
    child.start()
    child.join(15)
    if child.is_alive():
        child.terminate()
        child.join()
        pytest.fail("child cache access hung")
    assert child.exitcode == 0
    assert queue.get(timeout=5) == (None, 4)
    assert_usage_matches_entries(cache)
    assert cache.stats()["hits"] == 1
