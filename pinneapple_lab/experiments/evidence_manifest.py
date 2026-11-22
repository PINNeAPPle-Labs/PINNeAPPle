"""Build and verify content-addressed run_results manifests without executing solvers.

A manifest is an integrity receipt, not evidence of physical correctness.
Only regular files beneath the requested root are included; symlinks are denied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAX_FILES = 10000
MAX_FILE_BYTES = 256 * 1024 * 1024
MANIFEST_NAME = "evidence-manifest.json"


def _files(root: Path):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("root must be a real directory")
    paths = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    if len(paths) > MAX_FILES:
        raise ValueError("too many evidence files")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("symlinks and special files are not allowed")
        if path.name == MANIFEST_NAME and path.parent == root:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("evidence file exceeds per-file limit")
        yield path


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_manifest(root: str | Path) -> dict:
    base = Path(root).resolve(strict=True)
    entries = [
        {"path": path.relative_to(base).as_posix(),
         "size_bytes": path.stat().st_size,
         "sha256": _digest(path)}
        for path in _files(base)
    ]
    payload = {"schema": "pinneapple.run-results-evidence/1",
               "algorithm": "sha256", "files": entries,
               "scientific_status": "integrity_only_not_physics_validated"}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_manifest(root: str | Path, manifest: dict) -> dict:
    actual = build_manifest(root)
    expected = manifest.get("manifest_sha256")
    return {"valid": (manifest == actual and isinstance(expected, str)),
            "expected_sha256": expected,
            "actual_sha256": actual["manifest_sha256"],
            "file_count": len(actual["files"]),
            "scientific_status": "integrity_only_not_physics_validated"}


def write_manifest(root: str | Path) -> dict:
    base = Path(root).resolve(strict=True)
    manifest = build_manifest(base)
    path = base / MANIFEST_NAME
    if path.is_symlink():
        raise ValueError("manifest target must not be a symlink")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return manifest
