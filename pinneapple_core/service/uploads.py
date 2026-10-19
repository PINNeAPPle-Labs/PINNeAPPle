"""Uploaded files with a size limit, safe names and a SHA-256, stored under one directory.

    store = UploadStore("/data/uploads", max_bytes=50 * 2**20, allowed_suffixes={".csv", ".parquet", ".json"})
    rec = store.save("cases.csv", fileobj, owner="ip:203.0.113.7")   # streams; UploadTooLarge past the limit
    store.path(rec["id"])                                            # Path to read it back

Each upload lives in ``<root>/<id>/<safe name>`` with ``<root>/<id>/meta.json``.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, BinaryIO

_ID = re.compile(r"^[0-9a-f]{16}$")


class UploadTooLarge(ValueError):
    pass


class UploadRejected(ValueError):
    pass


def safe_name(name: str) -> str:
    """Base name only, with characters outside ``[A-Za-z0-9._-]`` replaced; never empty or hidden."""
    base = Path(str(name).replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip(".")
    return base[:120] or "upload"


class UploadStore:
    def __init__(self, root: str | Path, max_bytes: int = 50 * 2**20, allowed_suffixes: set[str] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.allowed = {s.lower() for s in allowed_suffixes} if allowed_suffixes else None

    def save(self, filename: str, data: BinaryIO | bytes, owner: str = "", chunk: int = 1 << 20) -> dict[str, Any]:
        name = safe_name(filename)
        suffix = Path(name).suffix.lower()
        if self.allowed is not None and suffix not in self.allowed:
            raise UploadRejected(f"file type {suffix or '(none)'} is not accepted; allowed: {sorted(self.allowed)}")
        upload_id = secrets.token_hex(8)
        d = self.root / upload_id
        d.mkdir()
        target = d / name
        h, size = hashlib.sha256(), 0
        try:
            with open(target, "wb") as out:
                if isinstance(data, (bytes, bytearray)):
                    blocks = [bytes(data)]
                else:
                    blocks = iter(lambda: data.read(chunk), b"")
                for block in blocks:
                    size += len(block)
                    if size > self.max_bytes:
                        raise UploadTooLarge(f"upload larger than the limit of {self.max_bytes / 2**20:g} MB")
                    h.update(block)
                    out.write(block)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        meta = {"id": upload_id, "name": name, "size": size, "sha256": h.hexdigest(), "owner": owner,
                "created": time.time()}
        (d / "meta.json").write_text(json.dumps(meta))
        return meta

    def get(self, upload_id: str) -> dict[str, Any]:
        if not _ID.match(str(upload_id)):
            raise KeyError(f"unknown upload {upload_id!r}")
        meta = self.root / upload_id / "meta.json"
        if not meta.is_file():
            raise KeyError(f"unknown upload {upload_id!r}")
        return json.loads(meta.read_text())

    def path(self, upload_id: str) -> Path:
        return self.root / upload_id / self.get(upload_id)["name"]

    def list(self, owner: str | None = None) -> list[dict[str, Any]]:
        out = []
        for meta in self.root.glob("*/meta.json"):
            m = json.loads(meta.read_text())
            if owner is None or m.get("owner") == owner:
                out.append(m)
        return sorted(out, key=lambda m: m["created"], reverse=True)

    def delete(self, upload_id: str) -> None:
        self.get(upload_id)
        shutil.rmtree(self.root / upload_id, ignore_errors=True)

    def prune(self, max_age_days: float) -> int:
        """Delete uploads older than ``max_age_days``; returns how many."""
        cutoff, n = time.time() - max_age_days * 86400, 0
        for m in self.list():
            if m["created"] < cutoff:
                self.delete(m["id"])
                n += 1
        return n
