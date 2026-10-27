import multiprocessing as mp
import errno
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openwam.dataloader.tos_cache import BoundedTosCache, CacheFullError, TosBackend, join_tos


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


def streaming_result(chunks, size=4):
    # Match the SDK's nested CRC adapter -> Response -> requests.Response.
    response = Mock()
    return SimpleNamespace(
        content_length=size, read=Mock(side_effect=chunks),
        content=SimpleNamespace(data=SimpleNamespace(resp=response)),
    ), response


def stub_tos_backend(monkeypatch, outcomes):
    backend = TosBackend()
    backend._pid = os.getpid()
    backend._client = Mock()
    backend._client.get_object.side_effect = outcomes
    backend._client.head_object.return_value = SimpleNamespace(content_length=4, etag="immutable")
    pause = Mock()
    monkeypatch.setattr("openwam.dataloader.tos_cache.time.sleep", pause)
    return backend, pause


@pytest.mark.parametrize("failure", ["timeout", "short_body"])
def test_stream_retry_restarts_object_and_keeps_etag(tmp_path, monkeypatch, failure):
    from requests.exceptions import ConnectionError

    broken, broken_response = streaming_result([
        b"xyz", ConnectionError("read timed out") if failure == "timeout" else b"",
    ])
    good, good_response = streaming_result([b"ab", b"cd", b""])
    backend, pause = stub_tos_backend(monkeypatch, [broken, good])
    cache = BoundedTosCache(tmp_path / "cache", 8, 0, backend)
    with cache.open("tos://bucket/a") as handle:
        assert handle.read() == b"abcd"
    assert backend.client.get_object.call_count == 2
    for call in backend.client.get_object.call_args_list:
        assert call.args == ("bucket", "a")
        assert call.kwargs == {"if_match": "immutable"}
    broken_response.close.assert_called_once()
    good_response.close.assert_called_once()
    pause.assert_called_once()
    assert cache.stats()["reserved_bytes"] == 4
    assert cache.stats()["inflight_bytes"] == 0
    assert not list(cache.root.glob("*.part"))


def test_stream_retry_exhaustion_cleans_cache(tmp_path, monkeypatch):
    from requests.exceptions import ConnectionError

    results = [streaming_result([b"xy", ConnectionError("read timed out")]) for _ in range(4)]
    backend, pause = stub_tos_backend(monkeypatch, [item[0] for item in results])
    cache = BoundedTosCache(tmp_path / "cache", 8, 0, backend)
    with pytest.raises(ConnectionError):
        with cache.open("tos://bucket/a"):
            pass
    assert backend.client.get_object.call_count == 4
    assert pause.call_count == 3
    for _, response in results:
        response.close.assert_called_once()
    assert cache.stats()["reserved_bytes"] == 0
    assert cache.stats()["inflight_bytes"] == 0
    assert cache.stats()["errors"] == 1
    assert not list(cache.root.glob("*.part"))
    assert not list(cache.root.glob("*.data"))


@pytest.mark.parametrize("status", [403, 404, 412, 429, 503])
def test_http_errors_retry_only_transient_status(tmp_path, monkeypatch, status):
    from tos.exceptions import TosServerError

    error = TosServerError(
        SimpleNamespace(request_id="test", headers={}, status=status),
        "test error", "test", "test", "test",
    )
    good, response = streaming_result([b"abcd", b""])
    backend, pause = stub_tos_backend(monkeypatch, [error, good])
    destination = tmp_path / "object.part"
    if status in (429, 503):
        backend.download("tos://bucket/a", destination, "immutable")
        assert destination.read_bytes() == b"abcd"
        assert backend.client.get_object.call_count == 2
        pause.assert_called_once()
        response.close.assert_called_once()
    else:
        with pytest.raises(TosServerError):
            backend.download("tos://bucket/a", destination, "immutable")
        assert backend.client.get_object.call_count == 1
        pause.assert_not_called()


def test_local_disk_full_does_not_retry(tmp_path, monkeypatch):
    import builtins

    good, response = streaming_result([b"abcd", b""])
    backend, pause = stub_tos_backend(monkeypatch, [good])
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", Mock(side_effect=OSError(errno.ENOSPC, "disk full")))
        with pytest.raises(OSError) as caught:
            backend.download("tos://bucket/a", tmp_path / "object.part", "immutable")
    assert caught.value.errno == errno.ENOSPC
    assert backend.client.get_object.call_count == 1
    pause.assert_not_called()
    response.close.assert_called_once()
