"""Where an OpenFOAM case starts: the folder that holds ``system/controlDict``.

One implementation for the tools that receive a case as a folder or a zip (an uploaded base case, a run artifact, a
result folder)::

    from pinneapple_simulation.external_solvers.openfoam import extract_case, find_case_root, is_case
    root = extract_case("upload.zip", workdir / "base")       # safe extraction, then the case root inside it
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["is_case", "find_case_root", "extract_case"]


def is_case(path: str | os.PathLike) -> bool:
    """Whether ``path`` is an OpenFOAM case directory (it holds ``system/controlDict``)."""
    return (Path(path) / "system" / "controlDict").is_file()


def find_case_root(directory: str | os.PathLike) -> Path:
    """The case directory at or below ``directory``: ``directory`` itself if it is a case, else the first
    sub-directory that is one, in sorted path order (so a zip holding ``mycase/system/controlDict`` gives
    ``mycase``). Raises ``ValueError`` when there is none."""
    d = Path(directory)
    for c in [d, *sorted(p for p in d.rglob("*") if p.is_dir())]:
        if is_case(c):
            return c
    raise ValueError("the zip holds no OpenFOAM case (system/controlDict)")


def extract_case(zip_path: str | os.PathLike, dest: str | os.PathLike, limits=None) -> Path:
    """Extract ``zip_path`` into ``dest`` with :func:`pinneapple_security.archives.safe_extract_zip` (no path
    traversal, no links; ``limits``: an :class:`~pinneapple_security.archives.ArchiveLimits`, default its defaults)
    and return the case root inside it (:func:`find_case_root`). An unsafe or oversized archive raises
    ``ValueError`` with the reason."""
    from pinneapple_security.archives import DEFAULT_LIMITS, UnsafeArchive, safe_extract_zip
    try:
        safe_extract_zip(zip_path, dest, limits or DEFAULT_LIMITS)
    except UnsafeArchive as exc:
        raise ValueError(str(exc)) from None
    return find_case_root(dest)
