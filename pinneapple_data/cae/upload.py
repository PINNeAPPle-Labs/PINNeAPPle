"""Expand an upload (loose files and/or zip archives) into {relative path: bytes}, with limits.

Unlike the metadata extractor's reader, nothing useful is skipped: meshes (.vtk, .stl, .msh), results (.frd, time
directories) and gzipped OpenFOAM files are kept. Junk (__MACOSX, ._*, .git) and parallel processor* folders are
dropped; a single common top folder is stripped.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, Sequence, Tuple

MAX_FILES = 20000
MAX_UNZIPPED = 1024 * 1024 * 1024


def expand(files: Sequence[Tuple[str, bytes]], max_unzipped: int = MAX_UNZIPPED) -> Dict[str, bytes]:
    fs: Dict[str, bytes] = {}
    total = 0
    for name, data in files:
        name = name.replace("\\", "/").lstrip("/")
        if name.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                infos = [i for i in z.infolist() if not i.is_dir()]
                if len(infos) > MAX_FILES:
                    raise ValueError(f"The archive has {len(infos)} files (limit {MAX_FILES}).")
                for i in infos:
                    p = i.filename.replace("\\", "/")
                    if "__MACOSX" in p or p.split("/")[-1].startswith("._") or re.search(r"(^|/)(processor\d+|\.git)(/|$)", p):
                        continue
                    total += i.file_size
                    if total > max_unzipped:
                        raise ValueError(f"The archive expands to more than {max_unzipped // 2**20} MB.")
                    fs[p] = z.read(i)
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
