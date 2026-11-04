"""Streaming OBS reads using the existing ModelArts identity, never saved keys."""
from __future__ import annotations

import contextlib
import errno
import http.client
import logging
import os
from pathlib import Path
import random
import time

from openwam.dataloader.tos_cache import parse_object_uri


class ObsReadError(IOError):
    def __init__(self, result):
        self.status = result.status
        self.code = result.errorCode
        super().__init__(f"OBS read failed: HTTP {self.status}, {self.code}")


class ObsBackend:
    DOWNLOAD_ATTEMPTS = 4

    def __init__(self):
        self._pid = None
        self._client = None
        self._created = 0

    def __getstate__(self):
        # Neither credentials nor sockets are serialized to spawned workers.
        return {"_pid": None, "_client": None, "_created": 0}

    @property
    def client(self):
        if self._pid != os.getpid() or time.monotonic() - self._created >= 300:
            # get_auth re-reads the credential file maintained by ModelArts.
            # This is the same credential provider used by installed MoXing.
            from moxing.framework.cloud_utils import get_auth
            from obs import ObsClient

            auth = get_auth()
            endpoint = os.environ.get(
                "OPENWAM_OBS_ENDPOINT",
                os.environ.get("OBS_ENDPOINT", "https://obs.cn-southwest-2.myhuaweicloud.com"),
            )
            if self._pid == os.getpid() and self._client is not None:
                self._client.close()
            self._client = ObsClient(
                access_key_id=auth.AK, secret_access_key=auth.SK,
                security_token=auth.TOKEN, server=endpoint,
                is_secure=True, ssl_verify=True, timeout=60,
                long_conn_mode=True, use_http_conns=True,
            )
            self._pid, self._created = os.getpid(), time.monotonic()
        return self._client

    @staticmethod
    def _parts(uri):
        scheme, bucket, key = parse_object_uri(uri)
        if scheme != "obs" or not key:
            raise ValueError("Expected obs://bucket/object-key")
        return bucket, key

    @staticmethod
    def _check(result):
        if not 200 <= result.status < 300:
            if result.status == 404:
                raise FileNotFoundError("OBS object not found")
            raise ObsReadError(result)

    def _retry(self, exc, attempt):
        if isinstance(exc, ObsReadError):
            retryable = exc.status in (408, 429, 500, 502, 503, 504) or exc.code in (
                "ExpiredToken", "InvalidToken", "TokenRefreshRequired",
            )
            if retryable:
                self._created = 0
        else:
            retryable = isinstance(exc, (TimeoutError, ConnectionError, http.client.HTTPException, OSError))
            if isinstance(exc, (FileNotFoundError, PermissionError)) or getattr(exc, "errno", None) in (errno.ENOSPC, errno.EDQUOT, errno.EROFS):
                retryable = False
        if attempt + 1 == self.DOWNLOAD_ATTEMPTS or not retryable:
            return False
        logging.getLogger(__name__).warning(
            "OBS read interrupted (%s); retry %d/%d", type(exc).__name__,
            attempt + 2, self.DOWNLOAD_ATTEMPTS,
        )
        time.sleep(2**attempt + random.uniform(0, .5))
        return True

    def head(self, uri):
        bucket, key = self._parts(uri)
        for attempt in range(self.DOWNLOAD_ATTEMPTS):
            try:
                result = self.client.getObjectMetadata(bucket, key)
                self._check(result)
                return int(result.body.contentLength), result.body.etag
            except Exception as exc:
                if not self._retry(exc, attempt):
                    raise

    def download(self, uri, destination, etag):
        from obs import GetObjectHeader

        bucket, key = self._parts(uri)
        for attempt in range(self.DOWNLOAD_ATTEMPTS):
            stream = None
            try:
                try:
                    result = self.client.getObject(
                        bucket, key,
                        # OBS SDK quotes this header itself; quoted HEAD ETags
                        # otherwise produce HTTP 412 (verified on ModelArts).
                        headers=GetObjectHeader(if_match=str(etag).strip('"')),
                        loadStreamInMemory=False,
                    )
                    self._check(result)
                    stream = result.body.response
                    received = 0
                    with open(destination, "wb") as out:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            received += len(chunk)
                    if received != int(result.body.contentLength):
                        raise IOError("Incomplete OBS response body")
                finally:
                    if stream is not None:
                        with contextlib.suppress(Exception):
                            stream.close()
                return
            except Exception as exc:
                Path(destination).unlink(missing_ok=True)
                if not self._retry(exc, attempt):
                    raise
