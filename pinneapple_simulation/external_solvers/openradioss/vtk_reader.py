"""Readers for OpenRadioss results converted by ``anim_to_vtk`` (legacy ASCII VTK).

``read_vtk_mesh`` parses one full frame (points, cell connectivity, node/part ids).
``read_case_displacements`` stacks the per-frame displacement blocks written by
:func:`~pinneapple_simulation.external_solvers.openradioss.runner.run_openradioss_case`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

__all__ = ["RadiossMesh", "read_vtk_mesh", "read_case_displacements"]


@dataclass
class RadiossMesh:
    points: np.ndarray        # (N, 3) coordinates of the frame (initial frame -> undeformed)
    node_ids: np.ndarray      # (N,) OpenRadioss node ids
    cells: list               # list of int arrays (connectivity, 0-based point indices)
    cell_types: np.ndarray    # (E,)
    part_ids: np.ndarray      # (E,) OpenRadioss part id per element
    time: float


def _read_block(lines, start, n, ncol, dtype=float):
    vals = []
    i = start
    while len(vals) < n * ncol:
        vals.extend(lines[i].split())
        i += 1
    return np.array(vals[: n * ncol], dtype=dtype).reshape(n, ncol), i


def read_vtk_mesh(path: str | Path) -> RadiossMesh:
    lines = Path(path).read_text().splitlines()
    idx: Dict[str, int] = {}
    for i, ln in enumerate(lines):
        key = ln.split(" ")[0]
        if key in ("POINTS", "CELLS", "CELL_TYPES", "POINT_DATA", "CELL_DATA", "TIME"):
            idx.setdefault(key, i)
        if ln.startswith("SCALARS NODE_ID"):
            idx["NODE_ID"] = i
        if ln.startswith("SCALARS PART_ID"):
            idx["PART_ID"] = i
    t = float(lines[idx["TIME"] + 1]) if "TIME" in idx else float("nan")
    npt = int(lines[idx["POINTS"]].split()[1])
    pts, _ = _read_block(lines, idx["POINTS"] + 1, npt, 3)
    ncell = int(lines[idx["CELLS"]].split()[1])
    cells = []
    for ln in lines[idx["CELLS"] + 1: idx["CELLS"] + 1 + ncell]:
        v = np.array(ln.split(), dtype=np.int64)
        cells.append(v[1: 1 + v[0]])
    ctypes = np.array(lines[idx["CELL_TYPES"] + 1: idx["CELL_TYPES"] + 1 + ncell], dtype=np.int64) \
        if "CELL_TYPES" in idx else np.zeros(ncell, dtype=np.int64)
    nid, _ = _read_block(lines, idx["NODE_ID"] + 2, npt, 1, dtype=np.int64)  # +2 skips LOOKUP_TABLE
    pid, _ = _read_block(lines, idx["PART_ID"] + 2, ncell, 1, dtype=np.int64)
    return RadiossMesh(pts.astype(np.float32), nid[:, 0], cells, ctypes, pid[:, 0], t)


def read_case_displacements(frames_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(times[T], disp[T, N, 3])`` from ``disp_XXX.txt``/``time_XXX.txt`` files."""
    d = Path(frames_dir)
    files = sorted(d.glob("disp_*.txt"))
    if not files:
        raise FileNotFoundError(f"no disp_*.txt in {d}")
    disp = np.stack([np.loadtxt(f, dtype=np.float32).reshape(-1, 3) for f in files])
    times = np.array([float((d / f.name.replace("disp_", "time_")).read_text().split()[0])
                      for f in files], dtype=np.float32)
    return times, disp
