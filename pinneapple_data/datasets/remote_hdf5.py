"""Read an HDF5 file over HTTP without downloading it, or from disk, behind one small interface.

The datasets this matters for are 1-13 GB each (PDEBench) or 2-20 TB each (The Well): a 5 GB download to read one
sample is not a workflow, and a benchmark that needs it never runs in CI. HDF5 is built for this -- it keeps its
metadata (superblock, B-tree, chunk index) in a few kilobytes and reads only the chunks a slice touches -- so the
only thing missing is a file object that can seek over HTTP. That is :class:`HTTPRangeFile`: a seekable
``io.RawIOBase`` that answers each ``read`` with one HTTP ``Range`` request.

    >>> from pinneapple_data.datasets.remote_hdf5 import open_hdf5
    >>> with open_hdf5("https://.../Sod6.hdf5") as f:      # or a local path
    ...     print(f["density"].shape)

Two behaviours worth knowing:

* a file at or below ``max_cache_bytes`` (64 MiB by default) is fetched once, streamed into memory, and served
  from there -- for a 5 MB case that is one request instead of a few hundred;
* a bigger file is read by range only. Metadata reads are cheap; reading a *slice* is cheap; iterating every
  element of a chunked dataset over HTTP is not, and the docstrings of the adapters say so.

The generic-URL handling (resolve the redirect of a DOI landing endpoint, then range-read the S3 target) is what
makes DaRUS -- where PDEBench lives -- work; Hugging Face's ``resolve`` endpoints behave the same way.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

__all__ = ["HTTPRangeFile", "open_hdf5", "is_url", "HttpRangeError"]

_TIMEOUT = 120
_CHUNK = 1 << 20


class HttpRangeError(RuntimeError):
    """The server did not answer a range request the way HDF5 needs it to."""


class HTTPRangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP(S) that only transfers the bytes asked for.

    ``bytes_read`` counts what actually crossed the network, which is what makes the cost of a read visible in a
    benchmark log instead of guessed at."""

    def __init__(self, url: str, *, timeout: float = _TIMEOUT, max_cache_bytes: int = 64 << 20,
                 user_agent: str = "pinneapple-data/1 (+https://github.com/PINNeAPPle-Labs/PINNeAPPle)") -> None:
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.bytes_read = 0
        self._cache: io.BytesIO | None = None
        self._pos = 0
        self._requests = 0
        head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(head, timeout=timeout) as response:
                self._size = int(response.headers.get("Content-Length") or 0)
                self._final_url = response.geturl()
                self._accept_ranges = (response.headers.get("Accept-Ranges") or "").lower() == "bytes"
        except urllib.error.HTTPError as exc:                       # some hosts refuse HEAD but serve GET
            if exc.code >= 500:
                raise HttpRangeError(f"HEAD {url} failed: HTTP {exc.code}") from exc
            self._size, self._final_url, self._accept_ranges = 0, url, False
        self._user_agent = user_agent
        if max_cache_bytes and 0 < self._size <= max_cache_bytes:
            self._cache = io.BytesIO(self._fetch_all())
            self._size = self._cache.getbuffer().nbytes

    # ── io.RawIOBase ───────────────────────────────────────────────────────────────────────────────────────
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            size = max(0, self._size - self._pos)
        if size == 0 or self._pos >= self._size:
            return b""
        if self._cache is not None:
            self._cache.seek(self._pos)
            data = self._cache.read(size)
            self._pos += len(data)
            return data
        end = min(self._pos + size, self._size) - 1
        data = self._fetch_range(self._pos, end)
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:                                   # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    @property
    def size(self) -> int:
        return self._size

    @property
    def requests(self) -> int:
        return self._requests

    def describe(self) -> dict[str, Any]:
        return {"url": self._final_url, "size": self._size, "bytes_read": self.bytes_read,
                "requests": self._requests, "cached": self._cache is not None}

    # ── transport ──────────────────────────────────────────────────────────────────────────────────────────
    def _fetch_range(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self._final_url, headers={
            "Range": f"bytes={start}-{end}", "User-Agent": self._user_agent})
        self._requests += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            if response.status != 206:
                raise HttpRangeError(f"{self.url} answered HTTP {response.status} to a range request; "
                                     "this server cannot serve HDF5 remotely (download the file instead)")
            data = response.read()
        self.bytes_read += len(data)
        return data

    def _fetch_all(self) -> bytes:
        self._requests += 1
        chunks: list[bytes] = []
        with urllib.request.urlopen(urllib.request.Request(self._final_url, headers={
                "User-Agent": self._user_agent}), timeout=self.timeout) as response:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                chunks.append(chunk)
        data = b"".join(chunks)
        self.bytes_read += len(data)
        return data


def is_url(source: str | Path) -> bool:
    """True for an ``http(s)://`` source (a local path never is, however many slashes it has)."""
    text = str(source)
    parsed = urllib.parse.urlparse(text)
    return parsed.scheme in ("http", "https")


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:                                          # pragma: no cover - optional dependency
        raise ImportError("reading HDF5 needs h5py (pip install h5py)") from exc
    return h5py


@contextmanager
def open_hdf5(source: str | Path, *, timeout: float = _TIMEOUT, max_cache_bytes: int = 64 << 20,
              comm=None) -> Iterator[IO]:
    """Open an HDF5 file at a local path or an http(s) URL; yields an ``h5py.File``.

    The remote handle is handed to h5py with ``driver="fileobj"``, so no temporary download is made and closing
    the context releases both."""
    h5py = _import_h5py()
    if is_url(str(source)):
        handle = HTTPRangeFile(str(source), timeout=timeout, max_cache_bytes=max_cache_bytes)
        f = h5py.File(handle, "r", driver="fileobj")
        try:
            yield f
        finally:
            f.close()
            handle.close()
        return
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"no such HDF5 file: {path}")
    f = h5py.File(path, "r")
    try:
        yield f
    finally:
        f.close()
