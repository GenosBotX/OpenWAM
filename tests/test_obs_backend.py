import io
import os
import pickle
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from openwam.dataloader.obs_backend import ObsBackend, ObsReadError
from openwam.dataloader.tos_cache import BoundedTosCache, DatasetStorage, parse_object_uri


def backend(client):
    result = ObsBackend()
    result._pid = os.getpid()
    result._created = time.monotonic()
    result._client = client
    return result


def response(data, length=None):
    stream = io.BytesIO(data)
    return NS(status=200, body=NS(response=stream, contentLength=len(data) if length is None else length)), stream


def test_conditional_stream_and_close(tmp_path):
    result, stream = response(b'payload')
    client = Mock()
    client.getObjectMetadata.return_value = NS(status=200, body=NS(contentLength=7, etag='"abc"'))
    client.getObject.return_value = result
    reader = backend(client)
    assert reader.head('obs://bucket/key') == (7, '"abc"')
    target = tmp_path / 'object'
    reader.download('obs://bucket/key', target, '"abc"')
    assert target.read_bytes() == b'payload' and stream.closed
    args, kwargs = client.getObject.call_args
    assert args == ('bucket', 'key') and kwargs['headers'].if_match == 'abc'
    assert kwargs['loadStreamInMemory'] is False


def test_partial_body_retries_from_start(tmp_path, monkeypatch):
    monkeypatch.setattr('openwam.dataloader.obs_backend.time.sleep', lambda _: None)
    first, a = response(b'ab', 4)
    second, b = response(b'abcd')
    client = Mock()
    client.getObject.side_effect = [first, second]
    target = tmp_path / 'object'
    backend(client).download('obs://bucket/key', target, 'etag')
    assert target.read_bytes() == b'abcd' and a.closed and b.closed
    assert client.getObject.call_count == 2


@pytest.mark.parametrize('status,exception', [(403, ObsReadError), (404, FileNotFoundError), (412, ObsReadError)])
def test_permanent_errors_do_not_retry(tmp_path, status, exception):
    client = Mock()
    client.getObject.return_value = NS(status=status, errorCode='Permanent')
    target = tmp_path / 'partial'
    target.write_bytes(b'old')
    with pytest.raises(exception):
        backend(client).download('obs://bucket/key', target, 'etag')
    assert client.getObject.call_count == 1 and not target.exists()


def test_backend_pickle_excludes_client():
    reader = backend(Mock())
    restored = pickle.loads(pickle.dumps(reader))
    assert restored._pid is None and restored._client is None


def test_obs_storage_uses_same_bounded_cache(tmp_path):
    client = Mock()
    client.getObjectMetadata.return_value = NS(status=200, body=NS(contentLength=4, etag='abc'))
    client.getObject.side_effect = lambda *a, **kw: response(b'data')[0]
    cache = BoundedTosCache(tmp_path / 'cache', 4, 0, backend(client))
    storage = DatasetStorage('obs://bucket/prefix', cache)
    assert storage.text('file') == storage.text('file') == 'data'
    assert cache.stats()['hits'] == 1 and cache.stats()['reserved_bytes'] == 4
    assert client.getObject.call_count == 1


@pytest.mark.parametrize('uri', ['obs://b/../x', 'obs://b/key?q=1', 'obs://user@b/x', 'https://b/x'])
def test_invalid_remote_paths(uri):
    with pytest.raises(ValueError):
        parse_object_uri(uri)


def test_disk_full_closes_stream_without_retry(tmp_path, monkeypatch):
    import errno
    client = Mock()
    result, stream = response(b'data')
    client.getObject.return_value = result
    with monkeypatch.context() as patch:
        patch.setattr('builtins.open', Mock(side_effect=OSError(errno.ENOSPC, 'full')))
        with pytest.raises(OSError) as caught:
            backend(client).download('obs://bucket/key', tmp_path / 'partial', 'etag')
    assert caught.value.errno == errno.ENOSPC
    assert stream.closed and client.getObject.call_count == 1


def test_transient_status_retries_and_closes_success(tmp_path, monkeypatch):
    monkeypatch.setattr('openwam.dataloader.obs_backend.time.sleep', lambda _: None)
    client = Mock()
    result, stream = response(b'data')
    client.getObject.side_effect = [NS(status=503, errorCode='SlowDown'), result]
    monkeypatch.setattr(ObsBackend, 'client', property(lambda self: client))
    target = tmp_path / 'object'
    ObsBackend().download('obs://bucket/key', target, 'etag')
    assert target.read_bytes() == b'data' and stream.closed
    assert client.getObject.call_count == 2


def test_client_recreated_after_fork_or_refresh(monkeypatch):
    import sys
    old, new = Mock(), Mock()
    factory = Mock(return_value=new)
    auth = Mock(return_value=NS(AK='test', SK='test', TOKEN='test'))
    # Stub auth module without loading the notebook runtime in a unit test.
    monkeypatch.setitem(sys.modules, 'moxing.framework.cloud_utils', NS(get_auth=auth))
    monkeypatch.setattr('obs.ObsClient', factory)
    reader = backend(old)
    assert reader.client is old
    reader._pid = -1
    assert reader.client is new
    old.close.assert_not_called()
    reader._created = time.monotonic() - 301
    assert reader.client is new and factory.call_count == 2
    new.close.assert_called_once()
