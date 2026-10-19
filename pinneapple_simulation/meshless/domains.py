"""2-D domains and node sets for the meshless study.

A :class:`Domain2D` has a signed distance (negative inside, from ``pinneapple_design.geometry.csg``) and its boundary
as exact pieces (straight segments and circular arcs), so boundary nodes lie on the true boundary and the corners of a
polygon are nodes. ``nodes(h)`` returns a quasi-uniform node set: a Cartesian lattice of spacing ``h`` kept at least
``0.4 h`` inside, plus boundary nodes at spacing ``h`` along every piece (optionally jittered, for robustness checks).

An uploaded point cloud (``point_cloud_domain``) has no signed distance: it has one node set (the points given) and
the methods that need the geometry itself (embedded finite differences, refinement) are not available on it.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Domain2D", "NodeSet", "domain", "polygon_domain", "point_cloud_domain", "DOMAINS"]


@dataclass
class NodeSet:
    X: np.ndarray                 # (N, 2) nodes, interior first then boundary
    boundary: np.ndarray          # (N,) bool
    h: float

    @property
    def n(self) -> int:
        return int(len(self.X))

    @property
    def interior(self) -> np.ndarray:
        return ~self.boundary


@dataclass
class Domain2D:
    name: str
    sdf: Callable[[np.ndarray], np.ndarray] | None
    pieces: list[tuple] = field(default_factory=list)   # ("seg", p0, p1) | ("arc", c, r, a0, a1)
    lo: tuple[float, float] = (0.0, 0.0)
    hi: tuple[float, float] = (1.0, 1.0)
    description: str = ""
    cloud: NodeSet | None = None                     # point-cloud domains: the given nodes
    eval_points: np.ndarray | None = None            # point-cloud domains: held-out points

    @property
    def has_geometry(self) -> bool:
        return self.sdf is not None

    @property
    def diameter(self) -> float:
        return float(np.hypot(self.hi[0] - self.lo[0], self.hi[1] - self.lo[1]))

    def contains(self, P: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return self.sdf(np.asarray(P, float)) < -margin

    def area(self, n: int = 400) -> float:
        if self.cloud is not None:
            from scipy.spatial import ConvexHull
            return float(ConvexHull(self.cloud.X).volume)
        g = np.stack(np.meshgrid(np.linspace(self.lo[0], self.hi[0], n), np.linspace(self.lo[1], self.hi[1], n)),
                     -1).reshape(-1, 2)
        return float(self.contains(g).mean() * (self.hi[0] - self.lo[0]) * (self.hi[1] - self.lo[1]))

    def boundary_nodes(self, h: float) -> np.ndarray:
        out = []
        for p in self.pieces:
            if p[0] == "seg":
                a, b = np.asarray(p[1], float), np.asarray(p[2], float)
                k = max(1, int(math.ceil(np.linalg.norm(b - a) / h - 1e-9)))
                s = np.arange(k) / k                                   # the end point starts the next piece
                out.append(a + s[:, None] * (b - a))
            else:
                c, r, a0, a1 = np.asarray(p[1], float), float(p[2]), float(p[3]), float(p[4])
                k = max(3, int(math.ceil(abs(a1 - a0) * r / h - 1e-9)))
                th = a0 + (a1 - a0) * np.arange(k) / k
                out.append(c + r * np.c_[np.cos(th), np.sin(th)])
        B = np.concatenate(out)
        keep = np.ones(len(B), bool)                                  # drop duplicates where pieces meet
        for i in range(len(B)):
            if keep[i]:
                d = np.hypot(*(B[i + 1:] - B[i]).T)
                keep[i + 1:][d < 1e-9] = False
        return B[keep]

    def nodes(self, h: float, *, jitter: float = 0.0, seed: int = 0) -> NodeSet:
        """Lattice nodes at spacing ``h`` at least ``0.4 h`` inside, boundary nodes at spacing ``h``.
        ``jitter``: random displacement of the interior nodes, as a fraction of ``h``."""
        if self.cloud is not None:
            return self.cloud
        x = np.arange(self.lo[0], self.hi[0] + 0.5 * h, h)
        y = np.arange(self.lo[1], self.hi[1] + 0.5 * h, h)
        G = np.stack(np.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
        if jitter:
            G = G + np.random.default_rng(seed).uniform(-jitter * h, jitter * h, G.shape)
        G = G[self.contains(G, 0.4 * h)]
        B = self.boundary_nodes(h)
        return NodeSet(np.vstack([G, B]), np.r_[np.zeros(len(G), bool), np.ones(len(B), bool)], float(h))

    def sample_eval(self, n: int = 2000, seed: int = 12345) -> np.ndarray:
        """Evaluation points strictly inside, independent of every node set (seeded)."""
        if self.eval_points is not None:
            return self.eval_points
        rng = np.random.default_rng(seed)
        out = []
        while sum(len(o) for o in out) < n:
            P = rng.uniform(self.lo, self.hi, (4 * n, 2))
            out.append(P[self.contains(P, 1e-9)])
        return np.concatenate(out)[:n]

    def outline(self, h: float | None = None) -> list[list[list[float]]]:
        """Boundary as closed polylines (for drawing)."""
        h = h or self.diameter / 200
        if self.cloud is not None:
            return []
        lines, cur, last = [], [], None
        for p in self.pieces:
            pts = Domain2D("", None, [p]).boundary_nodes(h)
            end = np.asarray(p[2], float) if p[0] == "seg" else \
                np.asarray(p[1]) + p[2] * np.array([math.cos(p[4]), math.sin(p[4])])
            if last is not None and np.hypot(*(pts[0] - last)) > 1e-9:
                lines.append(cur)
                cur = []
            cur += pts.tolist() + [end.tolist()]
            last = end
        if cur:
            lines.append(cur)
        return lines

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description, "bbox": [list(self.lo), list(self.hi)],
                "has_geometry": self.has_geometry, "outline": self.outline()}


def _poly_pieces(V) -> list[tuple]:
    V = np.asarray(V, float)
    return [("seg", V[i], V[(i + 1) % len(V)]) for i in range(len(V))]


def _circle(c, r, ccw=True) -> tuple:
    return ("arc", np.asarray(c, float), float(r), 0.0, 2 * math.pi) if ccw else \
        ("arc", np.asarray(c, float), float(r), 2 * math.pi, 0.0)


def polygon_domain(vertices: Sequence[Sequence[float]], name: str = "polygon",
                   holes: Sequence[tuple[tuple[float, float], float]] = ()) -> Domain2D:
    """A simple polygon (vertices in order), optionally with circular holes ``((cx, cy), r)``."""
    from pinneapple_design.geometry.csg import CSGCircle, CSGPolygon
    V = np.asarray(vertices, float)
    if V.ndim != 2 or V.shape[1] != 2 or len(V) < 3:
        raise ValueError("a polygon needs at least three (x, y) vertices")
    if np.allclose(V[0], V[-1]):
        V = V[:-1]
    if 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]) < 0:
        V = V[::-1]                                                    # counter-clockwise
    shape = CSGPolygon(V)
    pieces = _poly_pieces(V)
    for c, r in holes:
        shape = shape - CSGCircle(c[0], c[1], r)
        pieces.append(_circle(c, r, ccw=False))
    return Domain2D(name, shape.sdf, pieces, tuple(V.min(0)), tuple(V.max(0)),
                    f"polygon with {len(V)} vertices" + (f" and {len(holes)} hole(s)" if holes else ""))


def _disk() -> Domain2D:
    from pinneapple_design.geometry.csg import CSGCircle
    return Domain2D("disk", CSGCircle(0.0, 0.0, 1.0).sdf, [_circle((0, 0), 1.0)], (-1.0, -1.0), (1.0, 1.0),
                    "unit disk")


def _annulus() -> Domain2D:
    from pinneapple_design.geometry.csg import annulus
    return Domain2D("annulus", annulus(0.0, 0.0, 0.4, 1.0).sdf, [_circle((0, 0), 1.0), _circle((0, 0), 0.4, False)],
                    (-1.0, -1.0), (1.0, 1.0), "annulus 0.4 < r < 1")


DOMAINS: dict[str, Callable[[], Domain2D]] = {
    "square": lambda: polygon_domain([(0, 0), (1, 0), (1, 1), (0, 1)], "square"),
    "lshape": lambda: polygon_domain([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)], "lshape"),
    "disk": _disk,
    "annulus": _annulus,
    "plate_hole": lambda: polygon_domain([(0, 0), (1, 0), (1, 1), (0, 1)], "plate_hole", holes=[((0.5, 0.5), 0.2)]),
}
_DESC = {"square": "unit square", "lshape": "L-shape (unit square minus its upper-right quarter; re-entrant corner)",
         "plate_hole": "unit square with a central hole of radius 0.2"}


def domain(name: str) -> Domain2D:
    """A preset: ``square``, ``lshape``, ``disk``, ``annulus``, ``plate_hole``."""
    if name not in DOMAINS:
        raise ValueError(f"unknown domain {name!r}; presets: {sorted(DOMAINS)}")
    d = DOMAINS[name]()
    d.description = _DESC.get(name, d.description)
    return d


def point_cloud_domain(X: np.ndarray, boundary: np.ndarray, *, eval_fraction: float = 0.1, seed: int = 0,
                       name: str = "point cloud") -> Domain2D:
    """An uploaded point cloud: nodes ``X`` (N, 2) with a boundary flag. ``eval_fraction`` of the interior points
    are held out as evaluation points (not used by any method)."""
    X = np.asarray(X, float)[:, :2]
    b = np.asarray(boundary, bool).ravel()
    if len(X) != len(b) or b.sum() < 3 or (~b).sum() < 5:
        raise ValueError("a point cloud needs x, y and a boundary flag, with at least 3 boundary and 5 interior points")
    rng = np.random.default_rng(seed)
    inner = np.flatnonzero(~b)
    k = int(round(eval_fraction * len(inner)))
    held = rng.choice(inner, size=k, replace=False) if k else np.array([], int)
    keep = np.setdiff1d(np.arange(len(X)), held)
    keep = np.r_[keep[~b[keep]], keep[b[keep]]]
    from scipy.spatial import cKDTree
    h = float(np.median(cKDTree(X[keep]).query(X[keep], k=2)[0][:, 1]))
    ns = NodeSet(X[keep], b[keep], h)
    return Domain2D(name, None, [], tuple(X.min(0)), tuple(X.max(0)), f"{len(X)} uploaded points", ns,
                    X[held] if k else None)
