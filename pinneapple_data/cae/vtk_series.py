"""Legacy-VTK time series: the frames of a structural run as ``Mesh`` objects, one file per instant.

This is the format an explicit-FEA campaign leaves behind when the vendor's ANIM output is converted with
``anim_to_vtk`` (OpenRadioss), and it is the family the post-processors of LS-DYNA, CalculiX and the rest write. The
reader is written from the published VTK file format and the subset ``anim_to_vtk`` produces — points, cells, cell
types, ``POINT_DATA``/``CELL_DATA`` scalars, vectors and fields, and a ``TIME`` value — with no dependency on
``meshio``, so a campaign on a machine without the optional extras still reads its own results.

Three rules are worth stating, because a structural run is easy to get wrong here:

* **Identifiers are not fields.** ``anim_to_vtk`` writes ``SCALARS NODE_ID`` and ``SCALARS PART_ID`` beside the
  physical quantities. They come back as integer arrays on the :class:`VtkFrame` (``node_ids``/``part_ids``) instead
  of polluting ``point_data``/``cell_data``: a surrogate trained on a column called ``PART_ID`` is a bug waiting for
  the second part.
* **Cells are grouped by type and the per-cell arrays follow them.** ``Mesh`` stores one ``CellBlock`` per element
  type, so the cell arrays are permuted with the same index. A ``CELL_DATA`` array that silently stops matching its
  cells is the worst failure this reader could have, so the permutation is built once and applied to every array.
* **What is not modelled is said, not guessed.** Binary legacy VTK, the XML family (``.vtu``/``.vtp``) and polygons
  with more than four sides raise a message that names the route instead of returning a plausible wrong answer.

The writer round-trips through the same subset at a chosen precision: single (``%.9e``, what ``anim_to_vtk`` writes,
so the OpenRadioss path can be compared at the precision it stores) or double (lossless).

    from pinneapple_data.cae.vtk_series import read_vtk_series, write_vtk_series
    frames = read_vtk_series("case/frames")              # sorted by the number in the file name
    write_vtk_series(meshes, "out", times=times)         # frame_000.vtk, frame_001.vtk, ...
"""
from __future__ import annotations

import io
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import VTK_ID
from .model import CellBlock, Mesh

__all__ = ["VtkFormatError", "VtkFrame", "frame_number", "read_vtk_frame", "read_vtk_series", "write_vtk_frame",
           "write_vtk_series"]

TYPE_OF_ID = {v: k for k, v in VTK_ID.items()}
_VTK_TYPES = frozenset({"bit", "unsigned_char", "char", "unsigned_short", "short", "unsigned_int", "int",
                        "unsigned_long", "long", "vtkIdType", "float", "double", "long_long", "unsigned_long_long",
                        "__int64"})
_DATASETS = ("UNSTRUCTURED_GRID", "POLYDATA")


class VtkFormatError(ValueError):
    """A file outside the subset this reader covers, or one whose own counts do not agree with its content."""


@dataclass
class VtkFrame:
    """One instant of a run: its mesh and fields, the file it came from, and the time the file declares."""
    mesh: Mesh
    time: float | None = None
    node_ids: np.ndarray | None = None
    part_ids: np.ndarray | None = None
    path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        return self.mesh.points

    def field(self, name: str) -> np.ndarray | None:
        """A point or cell array by name, whichever it is."""
        if name in self.mesh.point_data:
            return self.mesh.point_data[name]
        return self.mesh.cell_data.get(name)


# ------------------------------------------------------------------------------------------------------- reading
class _Tokens:
    """A cursor over the whitespace-separated tokens of an ASCII legacy-VTK file (values wrap across lines)."""

    def __init__(self, tokens: list[str]):
        self.t = tokens
        self.i = 0

    def next(self) -> str:
        if self.i >= len(self.t):
            raise VtkFormatError("the file ends in the middle of a block")
        v = self.t[self.i]
        self.i += 1
        return v

    def peek(self) -> str:
        return self.t[self.i] if self.i < len(self.t) else ""

    def int(self) -> int:
        return int(self.next())

    def floats(self, n: int) -> np.ndarray:
        if self.i + n > len(self.t):
            raise VtkFormatError(f"expected {n} values and found {len(self.t) - self.i}")
        out = np.array(self.t[self.i: self.i + n], dtype=np.float64)
        self.i += n
        return out

    def ints(self, n: int) -> np.ndarray:
        if self.i + n > len(self.t):
            raise VtkFormatError(f"expected {n} integers and found {len(self.t) - self.i}")
        out = np.array(self.t[self.i: self.i + n], dtype=np.int64)
        self.i += n
        return out


def _length_prefixed(flat: np.ndarray, n: int) -> list[np.ndarray]:
    """The ``n`` length-prefixed cells of a ``CELLS``/``POLYGONS`` list, and nothing left over."""
    out: list[np.ndarray] = []
    i = 0
    for _ in range(n):
        if i >= len(flat):
            raise VtkFormatError(f"the cell list ends after {len(out)} of {n} cells")
        k = int(flat[i])
        out.append(flat[i + 1: i + 1 + k])
        i += 1 + k
    if i != len(flat):
        raise VtkFormatError(f"the cell list declares {n} cells and {len(flat)} integers, {i} were consumed")
    return out


def _polygon_cell(cell: np.ndarray) -> str:
    if len(cell) == 3:
        return "triangle"
    if len(cell) == 4:
        return "quad"
    raise VtkFormatError(f"a polygon with {len(cell)} nodes is not modelled; a crash run writes hexahedra, "
                         "tetrahedra, wedges, pyramids or three- and four-node faces")


def _group_cells(cells: list[np.ndarray], types: list[str]) -> tuple[list[CellBlock], np.ndarray]:
    """One block per element type, plus the index each grouped cell had in the file."""
    blocks: list[CellBlock] = []
    order: list[int] = []
    for name in dict.fromkeys(types):
        idx = [j for j, t in enumerate(types) if t == name]
        conn = np.array([cells[j] for j in idx], dtype=np.int64)
        if conn.ndim != 2:
            raise VtkFormatError(f"the {name} cells do not all have the same number of nodes")
        blocks.append(CellBlock(name, conn))
        order.extend(idx)
    return blocks, np.array(order, dtype=np.int64)


def _read_arrays(tk: _Tokens, n: int, into: dict[str, np.ndarray]) -> None:
    """The arrays of one ``POINT_DATA``/``CELL_DATA`` section; stops at the next section keyword."""
    while tk.i < len(tk.t):
        key = tk.peek().upper()
        if key in ("POINT_DATA", "CELL_DATA", "DATASET", "POINTS", "CELLS"):
            return
        if key == "SCALARS":
            tk.next()
            name = tk.next()
            if tk.peek() in _VTK_TYPES:
                tk.next()
            ncomp = int(tk.next()) if tk.peek().isdigit() else 1
            if tk.peek().upper() == "LOOKUP_TABLE":
                tk.next()
                tk.next()
            vals = tk.floats(n * ncomp)
            into[name] = vals if ncomp == 1 else vals.reshape(n, ncomp).copy()
        elif key in ("VECTORS", "NORMALS"):
            tk.next()
            name = tk.next()
            if tk.peek() in _VTK_TYPES:
                tk.next()
            into[name] = tk.floats(n * 3).reshape(n, 3).copy()
        elif key == "TEXTURE_COORDINATES":
            tk.next()
            name = tk.next()
            ndim = tk.int()
            if tk.peek() in _VTK_TYPES:
                tk.next()
            v = tk.floats(n * ndim)
            into[name] = v if ndim == 1 else v.reshape(n, ndim).copy()
        elif key == "FIELD":
            tk.next()
            tk.next()                                   # the name of the field data
            for _ in range(tk.int()):
                fname = tk.next()
                ncomp, ntup = tk.int(), tk.int()
                if tk.peek() in _VTK_TYPES:
                    tk.next()
                v = tk.floats(ncomp * ntup)
                if ntup == n:
                    v = v.reshape(ncomp, ntup).T.copy()
                    into[fname] = v[:, 0].copy() if ncomp == 1 else v
        else:
            return                                      # a section this reader does not model


def read_vtk_frame(path: str | Path) -> VtkFrame:
    """Read one legacy-VTK ASCII file (``UNSTRUCTURED_GRID`` or ``POLYDATA``) into a :class:`VtkFrame`."""
    p = Path(path)
    text = p.read_text(encoding="latin-1")
    first = text.split("\n", 1)[0]
    if not first.startswith("# vtk DataFile"):
        raise VtkFormatError(f"{p.name}: not a legacy VTK file")
    m = re.search(r"^DATASET\s+(\w+)", text, re.M | re.I)
    if m is None:
        raise VtkFormatError(f"{p.name}: no DATASET line")
    body = text[m.start():]
    kind = m.group(1).upper()
    if kind not in _DATASETS:
        raise VtkFormatError(f"{p.name}: DATASET {kind} is not a frame of a structural run "
                             f"(expected {' or '.join(_DATASETS)})")
    if re.search(r"^BINARY\s*$", text[: m.start()], re.M | re.I):
        raise VtkFormatError(f"{p.name}: BINARY legacy VTK is not read here — write the frame as ASCII the way "
                             "`anim_to_vtk` does, or convert it with meshio")

    tk = _Tokens(body.split())
    time: float | None = None
    points = np.zeros((0, 3))
    cells: list[np.ndarray] = []
    types: list[str] = []
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}
    while tk.i < len(tk.t):
        key = tk.next().upper()
        if key == "TIME":
            try:
                time = float(tk.peek())            # either `TIME value` or `TIME` and the value on the next line
            except ValueError:
                pass
            else:
                tk.next()
        elif key == "POINTS":
            n = tk.int()
            tk.next()                              # the storage type
            points = tk.floats(3 * n).reshape(n, 3)
        elif key == "CELLS":
            n, size = tk.int(), tk.int()
            cells = _length_prefixed(tk.ints(size), n)
        elif key == "CELL_TYPES":
            n = tk.int()
            if n != len(cells):
                raise VtkFormatError(f"{p.name}: CELL_TYPES lists {n} types and the file has {len(cells)} cells")
            ids = tk.ints(n)
            types = []
            for tid in ids.tolist():
                name = "polygon" if tid == 7 else TYPE_OF_ID.get(tid)
                if name is None:
                    raise VtkFormatError(f"{p.name}: VTK cell type {tid} is not an element this library models")
                types.append(name)
        elif key in ("VERTICES", "LINES", "POLYGONS", "TRIANGLE_STRIPS"):
            n = tk.int()
            got = _length_prefixed(tk.ints(tk.int()), n)
            cells = got
            name = {"VERTICES": "vertex", "LINES": "line", "POLYGONS": None, "TRIANGLE_STRIPS": "triangle"}[key]
            types = [_polygon_cell(c) if name is None else name for c in got]
        elif key == "POINT_DATA":
            n = tk.int()
            if n != len(points):
                raise VtkFormatError(f"{p.name}: POINT_DATA declares {n} points, the file has {len(points)}")
            _read_arrays(tk, n, point_data)
        elif key == "CELL_DATA":
            n = tk.int()
            if n != len(cells):
                raise VtkFormatError(f"{p.name}: CELL_DATA declares {n} cells, the file has {len(cells)}")
            _read_arrays(tk, n, cell_data)
    if cells and not types:
        raise VtkFormatError(f"{p.name}: the file has {len(cells)} cells and no CELL_TYPES, so no element type can be "
                             "assigned to them")
    blocks, order = _group_cells(cells, types) if cells else ([], np.zeros(0, dtype=np.int64))
    reordered = {k: (np.asarray(v, float)[order] if np.ndim(v) and len(v) == len(order) else np.asarray(v, float))
                 for k, v in cell_data.items()}
    node_ids = point_data.pop("NODE_ID", None)
    part_ids = reordered.pop("PART_ID", None)
    if part_ids is None and "PART_ID" in point_data:      # some writers put the part id on the nodes
        part_ids = point_data.pop("PART_ID")
    point_data.pop("PART_ID", None)
    reordered.pop("NODE_ID", None)
    mesh = Mesh(points=points, blocks=blocks,
                point_data={k: np.asarray(v, float) for k, v in point_data.items()},
                cell_data={k: np.asarray(v, float) for k, v in reordered.items()},
                source={"format": "VTK (legacy ASCII)", "file": p.name, "dataset": kind, "time": time})
    return VtkFrame(mesh=mesh, time=time, path=str(p),
                    node_ids=None if node_ids is None else np.asarray(node_ids, dtype=np.int64).ravel(),
                    part_ids=None if part_ids is None else np.asarray(part_ids, dtype=np.int64).ravel())


def frame_number(path: str | Path) -> tuple[int, str]:
    """Sort key for a frame file: the last run of digits in its name, then the name (``frame_012`` -> 12)."""
    name = Path(path).name
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else -1), name


def read_vtk_series(paths: Iterable[str | Path] | str | Path) -> list[VtkFrame]:
    """Read a time series: the files themselves, or the directory that holds them, sorted by frame number."""
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        files = [p for p in root.iterdir() if p.suffix.lower() in (".vtk", ".vtu", ".vtp")] if root.is_dir() \
            else [root]
    else:
        files = [Path(p) for p in paths]
    if not files:
        raise FileNotFoundError(f"no VTK frames found in {paths}")
    return [read_vtk_frame(p) for p in sorted(files, key=frame_number)]


# ------------------------------------------------------------------------------------------------------- writing
def write_vtk_frame(path: str | Path, mesh: Mesh, *, time: float | None = None, precision: str = "single",
                    title: str = "PINNeAPPle structural frame", part_ids: np.ndarray | None = None) -> Path:
    """Write one frame as legacy-VTK ASCII, in the subset ``anim_to_vtk`` produces.

    ``precision="single"`` (the default) writes nine significant digits — what a single-precision field survives and
    what the converter stores, so the two paths are compared at the precision they share. ``precision="double"``
    writes seventeen digits and is lossless. ``part_ids`` writes the ``PART_ID`` cell array ``anim_to_vtk`` writes,
    which is how a part survives the trip through the file.
    """
    if precision not in ("single", "double"):
        raise ValueError("precision must be 'single' or 'double'")
    fmt = "%.9e" if precision == "single" else "%.17e"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET UNSTRUCTURED_GRID"]
    if time is not None:
        out.extend(["TIME", repr(float(time))])
    pts = np.asarray(mesh.points, dtype=np.float64)
    out.append(f"POINTS {len(pts)} double")
    out.extend(_fmt(pts, fmt))
    conns = [np.asarray(b.conn, dtype=np.int64) for b in mesh.blocks]
    ncell = sum(len(c) for c in conns)
    out.append(f"CELLS {ncell} {sum(c.shape[0] * (1 + c.shape[1]) for c in conns)}")
    for c in conns:
        for row in c:
            out.append(" ".join([str(len(row))] + [str(int(v)) for v in row]))
    out.append(f"CELL_TYPES {ncell}")
    for b in mesh.blocks:
        tid = VTK_ID.get(b.type)
        if tid is None or tid in (42, 7):
            raise VtkFormatError(f"element type {b.type!r} has no fixed legacy-VTK cell type")
        out.extend([str(tid)] * len(b.conn))
    cells_out = dict(mesh.cell_data)
    if part_ids is not None:
        cells_out = {"PART_ID": np.asarray(part_ids, dtype=np.float64)} | cells_out
    for n, data, keyword in ((len(pts), mesh.point_data, "POINT_DATA"), (ncell, cells_out, "CELL_DATA")):
        if not data:
            continue
        out.append(f"{keyword} {n}")
        for name, vals in data.items():
            a = np.asarray(vals, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] == 3:
                out.append(f"VECTORS {name} double")
                out.extend(_fmt(a, fmt))
            else:
                out.append(f"SCALARS {name} double 1")
                out.append("LOOKUP_TABLE default")
                out.extend(_fmt(a.reshape(len(a), -1), fmt))
    p.write_text("\n".join(out) + "\n", encoding="latin-1")
    return p


def _fmt(a: np.ndarray, fmt: str) -> list[str]:
    """The rows of an array as text at the chosen precision, one string per row."""
    buf = io.StringIO()
    np.savetxt(buf, np.asarray(a, dtype=np.float64).reshape(len(a), -1), fmt=fmt)
    return buf.getvalue().splitlines()


def write_vtk_series(meshes: Sequence[Mesh], out_dir: str | Path, *, times: Sequence[float] | None = None,
                     precision: str = "single", prefix: str = "frame",
                     part_ids: np.ndarray | None = None) -> list[Path]:
    """One file per frame, named ``<prefix>_000.vtk``, ``<prefix>_001.vtk``, ... — the campaign's layout."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    return [write_vtk_frame(root / f"{prefix}_{k:03d}.vtk", m, time=None if times is None else float(times[k]),
                            precision=precision, part_ids=part_ids) for k, m in enumerate(meshes)]
