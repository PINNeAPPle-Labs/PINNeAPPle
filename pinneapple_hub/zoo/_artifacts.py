"""Artifact store of the zoo: manifest, SHA-256 checks, local cache and lazy download from the Hugging Face Hub.

Every file a model or dataset needs is listed in ``manifest.json`` with its size, SHA-256 and the Hub repo/revision
that hosts it. A file is resolved in this order:

1. bundled with the package (``pinneapple_hub/zoo/assets/...``; only small files, < ~2 MB),
2. the local cache (``$PINNEAPPLE_ZOO_CACHE`` or ``~/.cache/pinneapple/zoo``),
3. downloaded from the Hub (``huggingface_hub`` if installed, else plain HTTPS), checked, then cached.

A checksum that does not match raises :class:`ChecksumMismatchError` (a corrupt cache entry is deleted and fetched
again when the network is allowed). Offline (``PINNEAPPLE_OFFLINE=1``, ``HF_HUB_OFFLINE=1`` or ``offline=True``)
nothing is downloaded. A file the Hub does not have yet raises :class:`ArtifactNotPublishedError`, which says how to
publish it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
MANIFEST = HERE / "manifest.json"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


class ZooError(RuntimeError):
    """Base class of the zoo's errors."""


class ArtifactNotPublishedError(ZooError):
    """The artifact is in the manifest but not (yet) on the Hugging Face Hub."""


class ArtifactUnavailableError(ZooError):
    """The artifact is neither bundled nor cached and cannot be downloaded (offline or network error)."""


class ChecksumMismatchError(ZooError):
    """A file's SHA-256 differs from the manifest: corrupt, truncated or tampered with."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _manifest_cached(path: str, mtime: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def manifest() -> dict[str, Any]:
    """The parsed ``manifest.json`` (re-read when the file changes)."""
    return _manifest_cached(str(MANIFEST), MANIFEST.stat().st_mtime)


def cache_dir() -> Path:
    return Path(os.environ.get("PINNEAPPLE_ZOO_CACHE") or Path.home() / ".cache" / "pinneapple" / "zoo")


def is_offline(offline: bool | None = None) -> bool:
    if offline is not None:
        return bool(offline)
    return any(os.environ.get(k, "").strip().lower() in ("1", "true", "yes", "on")
               for k in ("PINNEAPPLE_OFFLINE", "HF_HUB_OFFLINE"))


def entry(kind: str, name: str, version: str | None = None) -> dict[str, Any]:
    """The manifest entry of a model (``kind="models"``) or dataset (``kind="datasets"``); with ``version``, the entry
    of that release -- the current one or one listed under ``superseded`` (its files are on the Hub, under its
    own tag; the reason it was superseded is in ``superseded_reason``)."""
    items = manifest()[kind]
    if name not in items:
        raise KeyError(f"no {kind[:-1]} named {name!r} in the zoo; available: {sorted(items)}")
    cur = items[name]
    if version is None or version == cur["version"]:
        return cur
    for old in cur.get("superseded", []):
        if old["version"] == version:
            base = {k: v for k, v in cur.items() if k not in ("superseded", "files", "version", "revision")}
            return {**base, **old}
    raise KeyError(f"{name!r} has no version {version!r}; versions: "
                   f"{[cur['version']] + [o['version'] for o in cur.get('superseded', [])]}")


def versions(kind: str, name: str) -> list[dict[str, str]]:
    """Every release of an item, newest first: version, revision and, for older ones, why they were superseded."""
    cur = entry(kind, name)
    out = [{"version": cur["version"], "revision": cur["revision"], "status": "current"}]
    for old in cur.get("superseded", []):
        out.append({"version": old["version"], "revision": old["revision"], "status": "superseded",
                    "superseded_by": old.get("superseded_by", ""), "reason": old.get("superseded_reason", "")})
    return out


def _check(path: Path, spec: dict[str, Any], what: str) -> None:
    got = sha256_file(path)
    if got != spec["sha256"]:
        raise ChecksumMismatchError(f"{what}: SHA-256 {got} does not match the manifest ({spec['sha256']}); "
                                    f"the file at {path} is corrupt or was modified")


def _hf_url(repo: str, repo_type: str, revision: str, filename: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{HF_ENDPOINT}/{prefix}{repo}/resolve/{revision}/{filename}"


def _not_published(kind: str, name: str, ent: dict[str, Any], filename: str, detail: str) -> ArtifactNotPublishedError:
    return ArtifactNotPublishedError(
        f"{kind[:-1]} {name!r} v{ent['version']} ({filename}) is not published on the Hugging Face Hub yet "
        f"(repo {ent['hf_repo']}, revision {ent['revision']}; {detail}). Maintainers publish the zoo with the "
        "'Publish zoo to the Hugging Face Hub' workflow once the HF_TOKEN secret is set, or locally with "
        "`python scripts/zoo/publish_to_hub.py`.")


def revisions(ent: dict[str, Any]) -> list[str]:
    """Hub revisions holding an entry's files, in the order they are tried: its ``revision`` and, when that is a
    republication of the same files (``v1.0.0.post1``: new card, Parquet or safetensors copies), the tag of the
    version itself (``v1.0.0``). Every file is checked against the manifest's SHA-256 whichever tag it came from."""
    base = f"v{ent['version']}"
    return [ent["revision"]] + ([base] if ent["revision"].startswith(base + ".post") else [])


def _download(kind: str, name: str, ent: dict[str, Any], filename: str, dest: Path) -> None:
    revs = revisions(ent)
    for i, rev in enumerate(revs):
        try:
            return _download_rev(kind, name, {**ent, "revision": rev}, filename, dest)
        except ArtifactNotPublishedError:
            if i == len(revs) - 1:
                raise


def _download_rev(kind: str, name: str, ent: dict[str, Any], filename: str, dest: Path) -> None:
    repo_type = "dataset" if kind == "datasets" else "model"
    tmpdir = Path(tempfile.mkdtemp(prefix="zoo_dl_", dir=dest.parent))
    try:
        try:
            import huggingface_hub as hub
        except ImportError:
            hub = None
        if hub is not None:
            from huggingface_hub.utils import (
                EntryNotFoundError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            )
            try:
                got = hub.hf_hub_download(ent["hf_repo"], filename, revision=ent["revision"], repo_type=repo_type,
                                          local_dir=str(tmpdir))
            except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as e:
                raise _not_published(kind, name, ent, filename, type(e).__name__) from e
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403, 404):
                    raise _not_published(kind, name, ent, filename, f"HTTP {status}") from e
                raise ArtifactUnavailableError(f"could not download {filename} of {name!r}: {e}") from e
            shutil.move(got, tmpdir / "file")
        else:
            url = _hf_url(ent["hf_repo"], repo_type, ent["revision"], filename)
            try:
                with urllib.request.urlopen(url, timeout=60) as r, open(tmpdir / "file", "wb") as f:
                    shutil.copyfileobj(r, f)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    raise _not_published(kind, name, ent, filename, f"HTTP {e.code}") from e
                raise ArtifactUnavailableError(f"could not download {url}: {e}") from e
            except (urllib.error.URLError, OSError) as e:
                raise ArtifactUnavailableError(f"could not download {url}: {e}") from e
        _check(tmpdir / "file", ent["files"][filename], f"downloaded {filename} of {name!r}")
        os.replace(tmpdir / "file", dest)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _bundled(spec: dict[str, Any]) -> Path | None:
    """Path of the bundled copy, if there is one (``PINNEAPPLE_ZOO_IGNORE_BUNDLED=1`` hides it, e.g. to test the
    download path or to force the Hub copy)."""
    if not spec.get("bundled") or os.environ.get("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "") == "1":
        return None
    p = ASSETS / spec["bundled"]
    return p if p.exists() else None


def fetch(kind: str, name: str, filename: str, *, offline: bool | None = None, version: str | None = None) -> Path:
    """Local path of one file of a model or dataset (``version``: a superseded release instead of the current one),
    verified against the manifest's SHA-256."""
    ent = entry(kind, name, version)
    if filename not in ent["files"]:
        raise KeyError(f"{name!r} has no file {filename!r}; files: {sorted(ent['files'])}")
    spec = ent["files"][filename]
    p = _bundled(spec)
    if p is not None:
        _check(p, spec, f"bundled {filename} of {name!r}")
        return p
    dest = cache_dir() / kind / name / ent["version"] / filename
    if dest.exists():
        try:
            _check(dest, spec, f"cached {filename} of {name!r}")
            return dest
        except ChecksumMismatchError:
            if is_offline(offline):
                raise
            dest.unlink()                                          # corrupt cache entry: fetch it again
    if is_offline(offline):
        raise ArtifactUnavailableError(
            f"{filename} of {kind[:-1]} {name!r} v{ent['version']} is not bundled or cached and the zoo is offline "
            f"(PINNEAPPLE_OFFLINE / HF_HUB_OFFLINE); cache it once online or copy it to {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download(kind, name, ent, filename, dest)
    return dest


def status(kind: str, name: str) -> dict[str, str]:
    """Where each file of an item currently resolves from without downloading: bundled | cached | remote."""
    ent = entry(kind, name)
    out = {}
    for fn, spec in ent["files"].items():
        if _bundled(spec) is not None:
            out[fn] = "bundled"
        elif (cache_dir() / kind / name / ent["version"] / fn).exists():
            out[fn] = "cached"
        else:
            out[fn] = "remote"
    return out
