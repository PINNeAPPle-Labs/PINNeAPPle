"""The passage: an axisymmetric duct described span by span, and its hexahedral sector mesh.

A "passage" is what a nozzle, an orifice, a venturi, a pipe reducer and a globe valve all have in
common: a bore that runs along one axis and changes radius, with the odd solid in the way. This
module describes such a passage as a list of :class:`Span` objects — an interval along the axis,
the radial cell bands it is meshed with, and its axial cell size — and turns that list into a
:class:`GlobeValve <...>`-free OpenFOAM ``polyMesh``. The valve is then a particular span list
(:func:`pinneapple_simulation.pipelines.compressible_internal.studies.globe_valve`), not a special
case in the mesher.

Conformity, and why the spec is checked
---------------------------------------
Blocks must match face to face: no ``mergePatchPairs``, no snapping, no hanging nodes. Two
neighbouring spans share their face only where their rings overlap, so the radii of the narrower
span must be a **subset** of the other's. Where they are not, the extra faces are not shared and
become walls — which is exactly what a shoulder in the bore is. :meth:`PassageSpec.problems`
refuses a spec that breaks that, before any meshing happens.

The axis is a small core (:attr:`PassageSpec.r_core`, 0.5 mm by default) carrying a ``symmetry``
patch rather than a degenerate line of points: it keeps every cell a hexahedron, and at 0.25 % of a
20 mm bore it costs nothing measurable.

The mesh is a sector (:attr:`PassageSpec.sector_deg`, 90 deg by default). The passage is a solid of
revolution, so the sector with two ``symmetryPlane`` patches is the 360 deg solution at a quarter of
the cells; integral quantities are scaled to the whole passage by ``360/sector_deg``, and that
scaling is stated wherever it is used (:mod:`.post`, :mod:`.checks`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Ring", "Span", "PassageSpec", "MeshSummary", "build_mesh", "PATCH_TYPES"]

PATCH_TYPES = {"inlet": "patch", "outlet": "patch", "wall": "wall",
               "axis": "symmetry", "sym0": "symmetryPlane", "sym1": "symmetryPlane"}


@dataclass(frozen=True)
class Ring:
    """A radial band of cells between ``r0`` and ``r1`` mm, with ``n`` cells across it.

    ``r0_end`` and ``r1_end`` give the radii at the far end of the span; leaving them out makes the
    band a cylinder, setting them makes it a cone. A tapered passage — a nozzle, a diffuser, a
    reducer — is then one span instead of a staircase of them.
    """

    r0: float
    r1: float
    n: int
    r0_end: float | None = None
    r1_end: float | None = None

    def radii(self, s: float = 0.0) -> tuple[float, float]:
        """The band's two radii at fraction ``s`` of the span (0 = start, 1 = end)."""
        a = self.r0 + ((self.r0_end if self.r0_end is not None else self.r0) - self.r0) * s
        b = self.r1 + ((self.r1_end if self.r1_end is not None else self.r1) - self.r1) * s
        return a, b

    def check(self) -> list[str]:
        bad = []
        for s in (0.0, 1.0):
            a, b = self.radii(s)
            if not b > a:
                bad.append(f"ring {a:g}..{b:g} mm at {s * 100:.0f} % of the span is not a band")
        if self.n < 1:
            bad.append(f"ring {self.r0:g}..{self.r1:g} mm has no cells")
        return bad


@dataclass(frozen=True)
class Span:
    """An interval along the axis, meshed with ``rings`` and an axial cell size ``h_axial``."""

    name: str
    x0: float
    x1: float
    rings: tuple[Ring, ...]
    h_axial: float

    def radii_at(self, s: float = 0.0) -> tuple[float, ...]:
        """The radii of every ring boundary at fraction ``s`` of the span."""
        return (self.rings[0].radii(s)[0],) + tuple(r.radii(s)[1] for r in self.rings)

    @property
    def radii(self) -> tuple[float, ...]:
        return self.radii_at(0.0)

    @property
    def n_axial(self) -> int:
        return max(2, int(math.ceil(abs(self.x1 - self.x0) / self.h_axial)))

    def blocks(self) -> list[dict[str, Any]]:
        out = []
        for k, ring in enumerate(self.rings):
            r0, r1 = ring.radii(0.0)
            r0e, r1e = ring.radii(1.0)
            out.append({"name": f"{self.name}_{k}", "x0": self.x0, "x1": self.x1,
                        "r0": r0, "r1": r1, "r0_end": r0e, "r1_end": r1e,
                        "nx": self.n_axial, "nr": ring.n})
        return out


@dataclass
class PassageSpec:
    """An axisymmetric passage: spans along x, a sector, and a core standing in for the axis."""

    spans: tuple[Span, ...]
    sector_deg: float = 90.0
    n_theta: int = 24
    r_core: float = 0.5
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ validation
    def problems(self) -> list[str]:
        bad: list[str] = []
        if not self.spans:
            return ["the passage has no spans"]
        if not (0.0 < self.sector_deg <= 360.0):
            bad.append("sector_deg must be in (0, 360]")
        if self.n_theta < 1:
            bad.append("n_theta must be at least 1")
        for s in self.spans:
            for ring in s.rings:
                bad += [f"{s.name}: {m}" for m in ring.check()]
            if s.x1 <= s.x0:
                bad.append(f"{s.name}: x1 must be greater than x0")
            if s.rings[0].r0 < self.r_core - 1e-9:
                bad.append(f"{s.name}: the innermost ring starts below the core (r = {s.rings[0].r0} mm)")
        for a, b in zip(self.spans, self.spans[1:], strict=False):
            if abs(a.x1 - b.x0) > 1e-9:
                bad.append(f"{a.name} ends at x = {a.x1} but {b.name} starts at x = {b.x0}")
            else:
                left, right = set(a.radii_at(1.0)), set(b.radii_at(0.0))
                if not (left <= right or right <= left):
                    bad.append(f"{a.name} and {b.name} do not share their radii at x = {a.x1:g} "
                               f"({sorted(left)} vs {sorted(right)}): the blocks would not match")
        return bad

    # ------------------------------------------------------------------ derived
    @property
    def x_min(self) -> float:
        return self.spans[0].x0

    @property
    def x_max(self) -> float:
        return self.spans[-1].x1

    @property
    def area_scale(self) -> float:
        return 360.0 / self.sector_deg

    def blocks(self) -> list[dict[str, Any]]:
        bad = self.problems()
        if bad:
            raise ValueError("; ".join(bad))
        out: list[dict[str, Any]] = []
        for span in self.spans:
            out += span.blocks()
        return out

    def summary(self) -> dict[str, Any]:
        return {"title": self.title, "spans": len(self.spans), "sector_deg": self.sector_deg,
                "n_theta": int(self.n_theta), "r_core_mm": self.r_core,
                "x_min_mm": self.x_min, "x_max_mm": self.x_max, **self.meta}


@dataclass
class MeshSummary:
    n_points: int
    n_cells: int
    n_faces: int
    n_internal: int
    patch_faces: dict[str, int]
    cell_size_mm: float
    sector_deg: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _newell(pts: np.ndarray) -> np.ndarray:
    """Face normal (magnitude = 2 x area) by Newell's method, valid for a non-planar quad."""
    nxt = np.roll(pts, -1, axis=0)
    return np.array([(pts[:, 1] * nxt[:, 2] - pts[:, 2] * nxt[:, 1]).sum(),
                     (pts[:, 2] * nxt[:, 0] - pts[:, 0] * nxt[:, 2]).sum(),
                     (pts[:, 0] * nxt[:, 1] - pts[:, 1] * nxt[:, 0]).sum()])


def build_mesh(passage: PassageSpec, case_dir: str, *, write: bool = True) -> MeshSummary:
    """Build the sector mesh of ``passage`` and (by default) write it into ``case_dir``."""
    blocks = passage.blocks()
    n_t = int(passage.n_theta)
    dtheta = math.radians(passage.sector_deg) / n_t
    millimetres = 1.0 / 1000.0

    # ---------------------------------------------------------------- nodes, shared across blocks
    # A point belongs to the mesh, not to a block: two blocks that meet on a face must name the same
    # point there, or every block becomes its own disconnected region (checkMesh then reports
    # "Number of regions: N" and the solver has no path between them). The key is the position.
    points: list[tuple[float, float, float]] = []
    lookup: dict[tuple[float, float, float], int] = {}

    def node(x_mm: float, r_mm: float, k: int) -> int:
        t = k * dtheta
        y_mm, z_mm = r_mm * math.cos(t), r_mm * math.sin(t)
        key = (round(x_mm, 6), round(y_mm, 6), round(z_mm, 6))
        idx = lookup.get(key)
        if idx is None:
            idx = len(points)
            points.append((x_mm * millimetres, y_mm * millimetres, z_mm * millimetres))
            lookup[key] = idx
        return idx

    face_of_key: dict[tuple[int, ...], int] = {}
    face_verts: list[list[int]] = []
    face_cells: list[list[int]] = []
    face_tag: list[str] = []
    cell_count = 0
    cell_size_mm = 0.0

    for blk in blocks:
        xs = np.linspace(blk["x0"], blk["x1"], blk["nx"] + 1)
        nx, nr = int(blk["nx"]), int(blk["nr"])
        r0e = float(blk.get("r0_end", blk["r0"]))
        r1e = float(blk.get("r1_end", blk["r1"]))

        # the band can taper, so the radii are evaluated per station along x; the loop variables are
        # bound as defaults so the closure cannot see a later block's numbers
        r_lo, r_hi = float(blk["r0"]), float(blk["r1"])

        def radius(i: int, j: int, _nx: int = nx, _nr: int = nr,
                   _a: float = r_lo, _ae: float = r0e, _b: float = r_hi, _be: float = r1e) -> float:
            t = i / _nx
            lo = _a + (_ae - _a) * t
            hi = _b + (_be - _b) * t
            return float(lo + (hi - lo) * j / _nr)

        cell_size_mm = max(cell_size_mm, float(xs[1] - xs[0]), abs(radius(0, 1) - radius(0, 0)))
        vid = {}
        for i in range(nx + 1):
            for j in range(nr + 1):
                for k in range(n_t + 1):
                    vid[(i, j, k)] = node(float(xs[i]), radius(i, j), k)

        for i in range(nx):
            for j in range(nr):
                for k in range(n_t):
                    a, bb = vid[(i, j, k)], vid[(i + 1, j, k)]
                    c, d = vid[(i + 1, j + 1, k)], vid[(i, j + 1, k)]
                    e, f = vid[(i, j, k + 1)], vid[(i + 1, j, k + 1)]
                    g, h = vid[(i + 1, j + 1, k + 1)], vid[(i, j + 1, k + 1)]
                    centre = 0.125 * np.sum(np.array([points[t] for t in (a, bb, c, d, e, f, g, h)]), axis=0)
                    for side, verts in (("-x", [a, d, h, e]), ("+x", [bb, c, g, f]),
                                        ("-r", [a, bb, f, e]), ("+r", [d, c, g, h]),
                                        ("-t", [a, bb, c, d]), ("+t", [e, f, g, h])):
                        pts = np.array([points[t] for t in verts])
                        if float(_newell(pts) @ (pts.mean(axis=0) - centre)) < 0.0:
                            verts = verts[::-1]
                        key = tuple(sorted(verts))
                        fid = face_of_key.get(key)
                        if fid is not None:
                            face_cells[fid].append(cell_count)
                            continue
                        first = blk["name"].startswith(passage.spans[0].name)
                        last = blk["name"].startswith(passage.spans[-1].name)
                        if side == "-x" and first:
                            tag = "inlet"
                        elif side == "+x" and last:
                            tag = "outlet"
                        elif side == "-t":
                            tag = "sym0"
                        elif side == "+t":
                            tag = "sym1"
                        elif side == "-r" and abs(blk["r0"] - passage.r_core) < 1e-9 \
                                and abs(r0e - passage.r_core) < 1e-9:
                            tag = "axis"
                        else:
                            tag = "wall"
                        face_of_key[key] = len(face_verts)
                        face_verts.append(verts)
                        face_cells.append([cell_count])
                        face_tag.append(tag)
                    cell_count += 1

    # ---------------------------------------------------------------- owner / neighbour split
    internal = [fid for fid, cells in enumerate(face_cells) if len(cells) == 2]
    internal.sort(key=lambda fid: (face_cells[fid][0], face_cells[fid][1]))
    boundary: dict[str, list[int]] = {name: [] for name in PATCH_TYPES}
    for fid, cells in enumerate(face_cells):
        if len(cells) == 1:
            tag = face_tag[fid]
            if tag not in boundary:
                raise RuntimeError(f"face {fid} has no patch")
            boundary[tag].append(fid)

    order = internal + [fid for name in PATCH_TYPES for fid in boundary[name]]
    owner = [face_cells[fid][0] for fid in order]
    neighbour = [face_cells[fid][1] for fid in internal]

    summary = MeshSummary(n_points=len(points), n_cells=cell_count, n_faces=len(order),
                          n_internal=len(internal),
                          patch_faces={name: len(ids) for name, ids in boundary.items()},
                          cell_size_mm=round(cell_size_mm, 3), sector_deg=passage.sector_deg)
    for name, ids in boundary.items():
        if not ids:
            raise RuntimeError(f"patch {name} is empty: the passage has no {name} face")
    if not write:
        return summary
    _write(case_dir, points, order, face_verts, owner, neighbour, boundary)
    return summary


def _write(case_dir: str, points, order, face_verts, owner, neighbour, boundary) -> None:
    mesh_dir = os.path.join(case_dir, "constant", "polyMesh")
    os.makedirs(mesh_dir, exist_ok=True)

    body = f"{len(points)}\n(\n" + "".join(f"({x:.9g} {y:.9g} {z:.9g})\n" for x, y, z in points) + ")\n"
    _foam_file(os.path.join(mesh_dir, "points"), "vectorField", "points", body)

    lines = [f"{len(v)}(" + " ".join(str(x) for x in v) + ")" for v in (face_verts[fid] for fid in order)]
    _foam_file(os.path.join(mesh_dir, "faces"), "faceList", "faces",
               f"{len(order)}\n(\n" + "\n".join(lines) + "\n)\n")

    _foam_file(os.path.join(mesh_dir, "owner"), "labelList", "owner",
               f"{len(owner)}\n(\n" + "\n".join(str(v) for v in owner) + "\n)\n")
    _foam_file(os.path.join(mesh_dir, "neighbour"), "labelList", "neighbour",
               f"{len(neighbour)}\n(\n" + "\n".join(str(v) for v in neighbour) + "\n)\n")

    start = len(owner) - sum(len(ids) for ids in boundary.values())
    entries = []
    for name, kind in PATCH_TYPES.items():
        n = len(boundary[name])
        entries.append(f"    {name}\n    {{\n        type            {kind};\n"
                       f"        nFaces          {n};\n        startFace       {start};\n    }}")
        start += n
    _foam_file(os.path.join(mesh_dir, "boundary"), "polyBoundaryMesh", "boundary",
               f"{len(PATCH_TYPES)}\n(\n" + "\n".join(entries) + "\n)\n")


def _foam_file(path: str, cls: str, obj: str, body: str) -> None:
    header = ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
              "| =========                 |                                                 |\n"
              "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
              "|  \\\\    /   O peration     | Version:  v2606                                 |\n"
              "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
              "|    \\\\/     M anipulation  |                                                 |\n"
              "\\*---------------------------------------------------------------------------*/\n")
    text = (header + "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n    location    \"constant/polyMesh\";\n"
            f"    object      {obj};\n}}\n\n" + body + "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
