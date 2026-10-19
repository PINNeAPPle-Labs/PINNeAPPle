"""Open uploaded zip archives safely: no member outside the destination (zip-slip), no symbolic links, and limits on
the number of members, the total uncompressed size, the size of one member and the compression ratio (zip bombs).

Sizes are counted while decompressing, not taken from the archive's own header, which an attacker writes.
"""
from __future__ import annotations

import io
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO

__all__ = ["UnsafeArchive", "ArchiveLimits", "check_member_name", "check_zip", "safe_extract_zip", "read_members",
           "gunzip"]

_CHUNK = 1 << 20


class UnsafeArchive(ValueError):
    """The archive breaks a safety rule or a limit."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 20_000
    max_total_bytes: int = 1 << 30            # 1 GiB uncompressed
    max_member_bytes: int = 1 << 30
    max_ratio: float | None = 200.0           # uncompressed / compressed, over the whole archive (None: no ratio
                                              # check; e.g. CFD cases with uniform fields compress very well)


DEFAULT_LIMITS = ArchiveLimits()


def check_member_name(name: str) -> str:
    """The member name as a safe relative POSIX path, or :class:`UnsafeArchive`."""
    if not name or "\x00" in name:
        raise UnsafeArchive("empty or invalid member name")
    norm = name.replace("\\", "/")
    p = PurePosixPath(norm)
    if p.is_absolute() or norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        raise UnsafeArchive(f"absolute path in the archive: {name}")
    if any(part == ".." for part in p.parts):
        raise UnsafeArchive(f"'..' in an archive path: {name}")
    return str(p)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def check_zip(z: zipfile.ZipFile, limits: ArchiveLimits = DEFAULT_LIMITS) -> list[zipfile.ZipInfo]:
    """Check names, link entries and declared sizes; return the file members (directories left out)."""
    infos = z.infolist()
    if len(infos) > limits.max_members:
        raise UnsafeArchive(f"the archive has more than {limits.max_members} members")
    total = packed = 0
    for i in infos:
        check_member_name(i.filename)
        if _is_symlink(i):
            raise UnsafeArchive(f"symbolic link in the archive: {i.filename}")
        if i.file_size > limits.max_member_bytes:
            raise UnsafeArchive(f"{i.filename} is larger than {limits.max_member_bytes} bytes uncompressed")
        total += i.file_size
        packed += i.compress_size
    if total > limits.max_total_bytes:
        raise UnsafeArchive(f"the archive expands to more than {limits.max_total_bytes} bytes")
    if limits.max_ratio and packed and total / packed > limits.max_ratio and total > 10 * _CHUNK:
        raise UnsafeArchive(f"compression ratio {total / packed:.0f} is above {limits.max_ratio:g} (zip bomb?)")
    return [i for i in infos if not i.is_dir()]


class _Budget:
    def __init__(self, limits: ArchiveLimits):
        self.limits, self.total = limits, 0

    def copy(self, src: IO[bytes], dst: IO[bytes], name: str) -> None:
        n = 0
        while chunk := src.read(_CHUNK):
            n += len(chunk)
            self.total += len(chunk)
            if n > self.limits.max_member_bytes:
                raise UnsafeArchive(f"{name} expands past {self.limits.max_member_bytes} bytes")
            if self.total > self.limits.max_total_bytes:
                raise UnsafeArchive(f"the archive expands past {self.limits.max_total_bytes} bytes")
            dst.write(chunk)


def _open(source: str | os.PathLike | bytes | IO[bytes] | zipfile.ZipFile) -> zipfile.ZipFile:
    if isinstance(source, zipfile.ZipFile):
        return source
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        return zipfile.ZipFile(source)
    except zipfile.BadZipFile as e:
        raise UnsafeArchive(f"not a zip archive: {e}") from None


def safe_extract_zip(source, dest: str | os.PathLike, limits: ArchiveLimits = DEFAULT_LIMITS) -> list[Path]:
    """Extract ``source`` (path, bytes, file object or ZipFile) under ``dest``; return the written files."""
    z = _open(source)
    root = Path(dest).resolve()
    root.mkdir(parents=True, exist_ok=True)
    budget, out = _Budget(limits), []
    for info in check_zip(z, limits):
        target = (root / check_member_name(info.filename)).resolve()
        if not target.is_relative_to(root):
            raise UnsafeArchive(f"member escapes the destination: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(target, "wb") as dst:
            budget.copy(src, dst, info.filename)
        out.append(target)
    return out


def read_members(source, limits: ArchiveLimits = DEFAULT_LIMITS, skip=None) -> dict[str, bytes]:
    """All file members in memory, ``{safe name: bytes}``, under the same checks and counted limits. ``skip``: a
    predicate on the member name for members to leave out (not read, not counted)."""
    z = _open(source)
    budget, out = _Budget(limits), {}
    for info in check_zip(z, limits):
        if skip is not None and skip(info.filename.replace("\\", "/")):
            continue
        buf = io.BytesIO()
        with z.open(info) as src:
            budget.copy(src, buf, info.filename)
        out[check_member_name(info.filename)] = buf.getvalue()
    return out


def gunzip(data: bytes, max_bytes: int = 1 << 30) -> bytes:
    """Decompress gzip ``data``, refusing output above ``max_bytes`` (a gzip bomb) without producing it first."""
    import zlib
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = bytearray()
    buf = data
    while buf:
        out += d.decompress(buf, max_bytes + 1 - len(out))
        if len(out) > max_bytes:
            raise UnsafeArchive(f"gzip data expands past {max_bytes} bytes")
        buf = d.unconsumed_tail
        if d.eof:
            rest = d.unused_data
            if rest and rest[:2] == b"\x1f\x8b":            # concatenated gzip members
                d, buf = zlib.decompressobj(16 + zlib.MAX_WBITS), rest
            else:
                break
    return bytes(out)
