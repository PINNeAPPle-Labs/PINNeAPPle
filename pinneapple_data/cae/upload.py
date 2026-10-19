"""Expand an upload (loose files and/or zip archives) into {relative path: bytes}, with limits.

Unlike the metadata extractor's reader, nothing useful is skipped: meshes (.vtk, .stl, .msh), results (.frd, time
directories) and gzipped OpenFOAM files are kept. Junk (__MACOSX, ._*, .git) and parallel processor* folders are
dropped; a single common top folder is stripped.
"""
from __future__ import annotations

import re
from typing import Dict, Sequence, Tuple

MAX_FILES = 20000
MAX_UNZIPPED = 1024 * 1024 * 1024


def _skip(p: str) -> bool:
    return "__MACOSX" in p or p.split("/")[-1].startswith("._") or bool(re.search(r"(^|/)(processor\d+|\.git)(/|$)", p))


def expand(files: Sequence[Tuple[str, bytes]], max_unzipped: int = MAX_UNZIPPED) -> Dict[str, bytes]:
    fs: Dict[str, bytes] = {}
    total = 0
    for name, data in files:
        name = name.replace("\\", "/").lstrip("/")
        if name.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
            from pinneapple_security.archives import ArchiveLimits, UnsafeArchive, read_members
            try:                                   # names checked, links refused, sizes counted while inflating
                members = read_members(data, ArchiveLimits(max_members=MAX_FILES, max_total_bytes=max_unzipped - total,
                                                           max_member_bytes=max_unzipped, max_ratio=None), skip=_skip)
            except UnsafeArchive as e:
                raise ValueError(f"The archive was refused: {e}") from None
            for p, b in members.items():
                total += len(b)
                fs[p] = b
        else:
            total += len(data)
            fs[name] = data
    keys = list(fs)
    if keys and all("/" in k for k in keys):
        first = {k.split("/", 1)[0] for k in keys}
        if len(first) == 1:
            pre = next(iter(first)) + "/"
            fs = {k[len(pre):]: v for k, v in fs.items()}
    return fs
