"""Disk caches of 3-D result views (a JSON description next to its glTF binary).

A web app that shows a solver or model result in 3-D builds the view once (it can take seconds: reading the mesh,
evaluating a model, decimating) and serves it again from disk. Two shapes are used:

- :func:`cached_view`: one view per **key** (a job, a step and its hash, a held-out sample). The first call builds
  it, later calls with the same key read it back;
- :func:`save_view` / :func:`prune_views`: views of one-off results kept under a random id, oldest dropped first.

::

    meta, glb = cached_view(job_dir / "views", f"test_{k}", lambda: prediction_view(...))
    return {**meta, "glb": url}, FileResponse(glb)
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pinneapple_core.jsonable import to_jsonable

__all__ = ["cached_view", "view_paths", "save_view", "prune_views"]

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def view_paths(directory: str | Path, key: str) -> tuple[Path, Path]:
    """The ``(json, glb)`` paths of ``key`` in ``directory`` (characters outside ``[A-Za-z0-9_.-]`` become ``_``,
    so a key cannot leave the directory)."""
    safe = _UNSAFE.sub("_", str(key))
    if not safe.strip("."):
        raise ValueError(f"view key {key!r} is empty after sanitising")
    d = Path(directory)
    return d / f"{safe}.json", d / f"{safe}.glb"


def cached_view(directory: str | Path, key: str, build: Callable[[], tuple[Any, dict]]) -> tuple[dict, Path]:
    """The cached view ``key`` in ``directory``: ``(description, glb_path)``.

    When both files exist they are read back; otherwise ``build()`` is called. It returns ``(view, payload)``:
    ``view`` has a ``glb()`` method (a :class:`~pinneapple_tools.visualization.studio.FieldView`) and ``payload`` is
    the description (made JSON-safe with :func:`~pinneapple_core.jsonable.to_jsonable`). The glTF is written first and
    the JSON last, so an interrupted build is rebuilt on the next call. Exceptions of ``build`` propagate and leave
    no description behind.
    """
    js, glb = view_paths(directory, key)
    if not (js.is_file() and glb.is_file()):
        view, payload = build()
        js.parent.mkdir(parents=True, exist_ok=True)
        glb.write_bytes(view.glb())
        js.write_text(json.dumps(to_jsonable(payload)))
    return json.loads(js.read_text()), glb


def save_view(directory: str | Path, view: Any) -> str:
    """Write ``view.glb()`` under a new random id (32 hex characters) in ``directory``; returns the id."""
    vid = uuid.uuid4().hex
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}.glb").write_bytes(view.glb())
    return vid


def prune_views(directory: str | Path, keep: int, pattern: str = "*.glb") -> int:
    """Keep the ``keep`` newest files matching ``pattern`` in ``directory`` (by modification time), delete the
    rest; returns how many were deleted."""
    old = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[max(int(keep), 0):]
    for p in old:
        p.unlink(missing_ok=True)
    return len(old)
