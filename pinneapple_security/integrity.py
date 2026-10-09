"""Integrity: content digests of files, arrays and records, and Merkle manifests of datasets and model folders.

A manifest lists every file of a folder with its SHA-256 and size, and a Merkle root over the sorted entries, so one
64-character string identifies the whole dataset and :meth:`Manifest.verify` says exactly which files were added,
removed or changed. Arrays are hashed with their dtype and shape (two arrays with the same bytes but a different shape
are different data). Records (dicts) are hashed in canonical JSON (sorted keys, no whitespace).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["sha256_bytes", "sha256_file", "digest_array", "digest_record", "canonical_json", "merkle_root",
           "Manifest", "build_manifest"]

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON bytes (sorted keys, compact separators, UTF-8); non-JSON values go through ``str``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def digest_record(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def digest_array(a: Any) -> str:
    """SHA-256 of an array's dtype, shape and C-ordered bytes (NumPy arrays and PyTorch tensors)."""
    import numpy as np

    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def merkle_root(leaves: Sequence[str]) -> str:
    """Merkle root (SHA-256) of hex leaf digests; an odd node is paired with itself. Empty input → hash of b''."""
    level = [bytes.fromhex(x) for x in leaves]
    if not level:
        return sha256_bytes(b"")
    level = [hashlib.sha256(b"\x00" + x).digest() for x in level]          # leaf / node domain separation
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


@dataclass
class Manifest:
    """Files of a folder with their digests. ``files``: relative path (POSIX) → {"sha256", "size"}."""

    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    algorithm: str = "sha256"
    root: str = ""

    def __post_init__(self):
        if not self.root:
            self.root = self.merkle_root()

    def merkle_root(self) -> str:
        return merkle_root([digest_record([p, self.files[p]["sha256"], self.files[p]["size"]]) for p in sorted(self.files)])

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "merkle_root": self.root, "files": self.files}

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        m = cls(files=dict(d["files"]), algorithm=d.get("algorithm", "sha256"))
        if d.get("merkle_root") and d["merkle_root"] != m.root:
            raise ValueError("manifest merkle_root does not match its file list (the manifest itself was edited)")
        return m

    @classmethod
    def load(cls, path: str) -> Manifest:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def diff(self, other: Manifest) -> dict[str, list[str]]:
        a, b = set(self.files), set(other.files)
        changed = sorted(p for p in a & b if self.files[p]["sha256"] != other.files[p]["sha256"])
        return {"added": sorted(b - a), "removed": sorted(a - b), "changed": changed}

    def verify(self, folder: str, **kw: Any) -> dict[str, Any]:
        """Rebuild the manifest of ``folder`` (same include/exclude rules) and compare. ``ok`` only if nothing differs."""
        now = build_manifest(folder, **kw)
        d = self.diff(now)
        return {"ok": not any(d.values()), **d, "expected_root": self.root, "actual_root": now.root}


def _walk(folder: str, include: Iterable[str], exclude: Iterable[str]):
    include, exclude = list(include), list(exclude)
    for dirpath, dirnames, names in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not any(fnmatch.fnmatch(d, e) for e in exclude))
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(dirpath, n), folder).replace(os.sep, "/")
            if any(fnmatch.fnmatch(rel, e) or fnmatch.fnmatch(n, e) for e in exclude):
                continue
            if include and not any(fnmatch.fnmatch(rel, i) or fnmatch.fnmatch(n, i) for i in include):
                continue
            yield rel


def build_manifest(folder: str, include: Iterable[str] = (), exclude: Iterable[str] = (".git", "__pycache__", "*.sig", "*.sig.json",
                   "manifest.json", "attestation.json")) -> Manifest:
    """Manifest of every file under ``folder`` (glob ``include`` filters; ``exclude`` drops names, folders, paths)."""
    files = {}
    for rel in _walk(folder, include, exclude):
        p = os.path.join(folder, rel)
        files[rel] = {"sha256": sha256_file(p), "size": os.path.getsize(p)}
    return Manifest(files=files)
