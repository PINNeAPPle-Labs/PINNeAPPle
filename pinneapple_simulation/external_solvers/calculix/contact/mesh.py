"""Structured hexahedral blocks and CalculiX's brick face numbering: the geometry every contact case is built from.

A case is a handful of blocks (a base plate, a slider, a rigid indenter) meshed as grids of C3D8 bricks. Nothing here
knows about contact: a block is nodes, elements and the faces/ node sets its case asks for, selected by *geometry* —
the plane a face lies in, and which side of that plane the element sits on — instead of by a face number typed into
the deck by hand. That matters: picking the wrong local face loads or contacts the wrong side of a body, and the
solver reports a converged, wrong answer. Every surface a case uses is therefore derived from coordinates.

The face table is CalculiX's brick convention (``*SURFACE, TYPE=ELEMENT``), with local nodes numbered the way
:class:`Grid` builds them: 1-4 around the bottom face, 5-8 around the top, both counter-clockwise seen from +z.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

#: CalculiX brick faces: face number -> the four local node indices (0-based) around it, in the manual's order.
FACES: dict[int, tuple[int, int, int, int]] = {
    1: (0, 1, 2, 3),      # z = zmin
    2: (4, 7, 6, 5),      # z = zmax
    3: (0, 4, 5, 1),      # y = ymin
    4: (1, 2, 6, 5),      # x = xmax
    5: (2, 3, 7, 6),      # y = ymax
    6: (3, 0, 4, 7),      # x = xmin
}

AXES = {"x": 0, "y": 1, "z": 2}
TOL = 1e-9


def axis(x0: float, x1: float, n: int, grading: float = 1.0, cluster: str = "low") -> list[float]:
    """``n + 1`` coordinates from ``x0`` to ``x1``. ``grading`` > 1 clusters the cells towards one end — the end the
    contact patch is at, which is what makes a curved-on-flat case affordable: fine where the pressure is, coarse
    where the body only carries it away."""
    if n < 1:
        raise ValueError("a block needs at least one cell per direction")
    if grading <= 0:
        raise ValueError("grading must be positive")
    t = [(i / n) ** grading for i in range(n + 1)]
    if cluster == "high":
        t = [1 - v for v in reversed(t)]
    return [x0 + (x1 - x0) * v for v in t]


def mirror(values: Sequence[float]) -> list[float]:
    """A one-sided axis turned into the symmetric one around 0 (``[0, .5, 1]`` -> ``[-1, -.5, 0, .5, 1]``)."""
    return sorted({-v for v in values} | set(values))


@dataclass
class Grid:
    """A rectilinear block of C3D8 bricks.

    ``shift`` moves every node as a function of its x coordinate — the parabolic crown of a cylinder, or the tilted
    plane of an incline — so a case can keep a rectangular grid while presenting a curved or inclined surface.

    ``polar`` reads the first two axes as radius and angle (radians) and places the nodes in Cartesian coordinates
    ``(r cos t, r sin t)``: a ring, a shaft or a quarter of either, still meshed as bricks. The faces then have to be
    selected by radius (:meth:`faces_where`), not by a plane — which is what a cylindrical surface needs.

    ``warp`` moves a node as a function of its position on the surface — the sphere cap of an indenter, whose height
    over a grid cell depends on both in-plane coordinates, which ``shift`` (a function of ``x`` alone) cannot express.
    It is applied last, so a block stays a block of bricks with a curved face, and the curved face is still selected
    by :meth:`faces` with a ``curved`` test.
    """
    xs: Sequence[float]
    ys: Sequence[float]
    zs: Sequence[float]
    first_node: int = 1
    first_elem: int = 1
    shift: Callable[[float], tuple[float, float, float]] | None = None
    polar: bool = False
    warp: Callable[[tuple[float, float, float]], tuple[float, float, float]] | None = None
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict, init=False)
    cells: list[list[int]] = field(default_factory=list, init=False)
    elem_ids: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        nx, ny, nz = len(self.xs) - 1, len(self.ys) - 1, len(self.zs) - 1
        if min(nx, ny, nz) < 1:
            raise ValueError("a block needs two coordinates on every axis")
        ids: dict[tuple[int, int, int], int] = {}
        nid = self.first_node
        for k, z in enumerate(self.zs):
            for j, y in enumerate(self.ys):
                for i, x in enumerate(self.xs):
                    dx, dy, dz = self.shift(x) if self.shift else (0.0, 0.0, 0.0)
                    px, py = x, y
                    if self.polar:                       # never rebind the loop variables: the next pass needs them
                        px, py = x * math.cos(y), x * math.sin(y)
                    ids[(i, j, k)] = nid
                    point = (px + dx, py + dy, z + dz)
                    self.nodes[nid] = self.warp(point) if self.warp else point
                    nid += 1
        eid = self.first_elem
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    self.cells.append([ids[(i, j, k)], ids[(i + 1, j, k)], ids[(i + 1, j + 1, k)], ids[(i, j + 1, k)],
                                       ids[(i, j, k + 1)], ids[(i + 1, j, k + 1)], ids[(i + 1, j + 1, k + 1)],
                                       ids[(i, j + 1, k + 1)]])
                    self.elem_ids.append(eid)
                    eid += 1
        self._next_node, self._next_elem = nid, eid

    # ----------------------------------------------------------------------------------------------- selection
    @property
    def node_ids(self) -> list[int]:
        return sorted(self.nodes)

    @property
    def next_node(self) -> int:
        """The first node id this block did not use: where the next one starts."""
        return self._next_node

    @property
    def next_elem(self) -> int:
        return self._next_elem

    def _in_plane(self, ids: Iterable[int], at: int, value: float, tol: float) -> bool:
        return all(abs(self.nodes[n][at] - value) <= tol for n in ids)

    def faces(self, plane: str, value: float, side: int | None = None, tol: float = TOL,
              curved: Callable[[tuple[float, float, float]], float] | None = None) -> list[tuple[int, int]]:
        """``(element id, face number)`` for every face lying in ``plane = value``.

        ``side`` keeps only the faces whose element is on that side of the plane (``-1`` below, ``+1`` above), which
        is how one plane yields both the top of the block underneath and the bottom of the block on top. ``curved``
        replaces the plane test with ``|plane coordinate - curve(point)| <= tol``: a face of a body whose surface is
        not flat, such as the crown of an indenter.
        """
        at = AXES[plane]
        out: list[tuple[int, int]] = []
        for eid, cell in zip(self.elem_ids, self.cells, strict=True):
            for face, local in FACES.items():
                pts = [self.nodes[cell[k]] for k in local]
                if curved is not None:
                    ok = all(abs(p[at] - curved(p)) <= tol for p in pts)
                else:
                    ok = all(abs(p[at] - value) <= tol for p in pts)
                if not ok:
                    continue
                if side is not None:
                    # the element's own side of the surface: the curve where there is one, the plane otherwise
                    centre = tuple(sum(self.nodes[n][a] for n in cell) / 8 for a in range(3))
                    level = curved(centre) if curved is not None else value
                    if (centre[at] - level) * side < 0:
                        continue
                out.append((eid, face))
        return out

    def faces_where(self, test: Callable[[tuple[float, float, float]], bool],
                    tol: float = TOL) -> list[tuple[int, int]]:
        """``(element id, face number)`` for every face whose four nodes satisfy ``test`` — the way a cylindrical
        surface is named (by its radius), since it lies on no plane."""
        out: list[tuple[int, int]] = []
        for eid, cell in zip(self.elem_ids, self.cells, strict=True):
            for face, local in FACES.items():
                if all(test(self.nodes[cell[k]]) for k in local):
                    out.append((eid, face))
        return out

    def nodes_at_radius(self, radius: float, tol: float = 1e-9) -> list[int]:
        """Node ids on a cylinder of that radius (distance from the z axis)."""
        return [n for n, p in self.nodes.items() if abs(math.hypot(p[0], p[1]) - radius) <= tol]

    def nodes_on(self, plane: str, value: float, tol: float = TOL) -> list[int]:
        """Node ids whose ``plane`` coordinate is ``value`` (within ``tol``)."""
        at = AXES[plane]
        return [n for n, p in self.nodes.items() if abs(p[at] - value) <= tol]

    def nodes_where(self, test: Callable[[tuple[float, float, float]], bool]) -> list[int]:
        return [n for n, p in self.nodes.items() if test(p)]

    def area(self, plane: str, value: float, tol: float = TOL) -> float:
        """The exact area of that plane of the block, for the load a pressure produces."""
        at = AXES[plane]
        others = [i for i in (0, 1, 2) if i != at]
        spans = []
        for i in others:
            coords = [p[i] for p in self.nodes.values() if abs(p[at] - value) <= tol]
            spans.append((max(coords) - min(coords)) if coords else 0.0)
        return spans[0] * spans[1]


def merge(*grids: Grid) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, list[int]]]]:
    """The node table and the ``(element id, its eight nodes)`` list of a set of blocks, in deck order."""
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, list[int]]] = []
    for g in grids:
        nodes.update(g.nodes)
        elements += list(zip(g.elem_ids, g.cells, strict=True))
    return nodes, elements


def nset(ids: Sequence[int], per_line: int = 16) -> list[str]:
    """An ``*NSET`` body. CalculiX refuses more than 16 entries on a line, and a hand-written deck that breaks the
    rule stops with a message about the *next* keyword, which is why this is a helper and not a detail of each case."""
    ids = sorted(ids)
    return [", ".join(str(i) for i in ids[i:i + per_line]) for i in range(0, len(ids), per_line)]
