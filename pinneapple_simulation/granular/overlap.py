"""Exact overlap geometry for convex bodies: the separating axis, and the volume two bodies share.

This is the part of the granular solver that makes it a *volume-based* contact model instead of a penetration-depth
one.  For two convex bodies the intersection is again convex, so the shared volume can be computed exactly by
clipping one body's face loops against the other's face planes and integrating the divergence theorem over what is
left.  The force magnitude is then a function of that volume (see
:mod:`pinneapple_simulation.granular.contacts`), which is what keeps a polyhedral particle from sinking into its
neighbour differently depending on which corner meets which face.

Three quantities come out of one call and are all used downstream:

* ``volume``  -- the force magnitude,
* ``centroid`` -- the contact point, so torque and the wear patch land where the material actually is,
* ``normal``/``depth`` -- the minimum-translation axis from the separating axis theorem, so the normal force pushes
  along the direction that separates the pair fastest.

    >>> from pinneapple_simulation.granular import box
    >>> from pinneapple_simulation.granular.overlap import overlap_volume
    >>> a = box(1.0, 1.0, 1.0).transformed(translation=[0.0, 0.0, 0.0])
    >>> b = box(1.0, 1.0, 1.0).transformed(translation=[0.5, 0.0, 0.0])
    >>> round(overlap_volume(a, b)[0], 12)
    0.5

All routines work in world coordinates: a body is passed already rotated and translated (``ConvexPolyhedron`` keeps
its own body frame, the caller does the rigid motion).
"""
from __future__ import annotations

import math

import numpy as np

from .shapes import ConvexPolyhedron

__all__ = [
    "SeparatingAxis",
    "separating_axis",
    "intersection_loops",
    "loops_volume_centroid",
    "overlap_volume",
    "clipped_by_plane",
    "polytope_from_halfspaces",
    "sphere_plane_cap_volume",
    "sphere_sphere_lens_volume",
]

_EPS = 1.0e-12


class SeparatingAxis:
    """Result of :func:`separating_axis`: ``overlap`` is ``False`` when a gap exists along ``normal``."""

    __slots__ = ("overlap", "normal", "depth")

    def __init__(self, overlap: bool, normal: np.ndarray, depth: float) -> None:
        self.overlap = bool(overlap)
        self.normal = normal
        self.depth = float(depth)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SeparatingAxis(overlap={self.overlap}, depth={self.depth:.3e})"


# -------------------------------------------------------------------------------------------- separating axis
def _candidate_axes(a: ConvexPolyhedron, b: ConvexPolyhedron,
                    Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Face normals of both bodies plus every edge-edge cross product, normalised and de-duplicated.

    For convex polyhedra this set is complete: two disjoint convex polytopes always separate along a face normal of
    one of them or a cross product of one edge from each (the separating axis theorem for polytopes).
    """
    na = a.face_normals @ Ra.T
    nb = b.face_normals @ Rb.T
    ea = np.unique(np.round(a.vertices[a.edges[:, 1]] - a.vertices[a.edges[:, 0]], 12), axis=0)
    eb = np.unique(np.round(b.vertices[b.edges[:, 1]] - b.vertices[b.edges[:, 0]], 12), axis=0)
    ea = ea @ Ra.T
    eb = eb @ Rb.T
    axes = [na, nb]
    if len(ea) and len(eb):
        cross = np.cross(ea[:, None, :], eb[None, :, :]).reshape(-1, 3)
        lengths = np.linalg.norm(cross, axis=1)
        keep = lengths > 1.0e-8
        cross = cross[keep] / lengths[keep, None]
        if len(cross):
            axes.append(cross)
    allaxes = np.vstack(axes)
    # round to drop duplicates that only differ by floating-point noise, then keep unit axes
    _, unique = np.unique(np.round(allaxes, 9), axis=0, return_index=True)
    return allaxes[np.sort(unique)]


def separating_axis(a: ConvexPolyhedron, b: ConvexPolyhedron,
                    Ra: np.ndarray | None = None, Rb: np.ndarray | None = None,
                    ta: np.ndarray | None = None, tb: np.ndarray | None = None,
                    tol: float = 0.0) -> SeparatingAxis:
    """Minimum-translation axis of two convex polyhedra (``Ra``/``Rb`` rotations, ``ta``/``tb`` positions)."""
    Ra = np.eye(3) if Ra is None else np.asarray(Ra, dtype=float)
    Rb = np.eye(3) if Rb is None else np.asarray(Rb, dtype=float)
    ta = np.zeros(3) if ta is None else np.asarray(ta, dtype=float)
    tb = np.zeros(3) if tb is None else np.asarray(tb, dtype=float)
    va = a.vertices @ Ra.T + ta
    vb = b.vertices @ Rb.T + tb
    centre = tb - ta
    axes = _candidate_axes(a, b, Ra, Rb)
    pa = va @ axes.T
    pb = vb @ axes.T
    lo = np.maximum(pa.min(0), pb.min(0))
    hi = np.minimum(pa.max(0), pb.max(0))
    depth = hi - lo                                                 # > 0 means the projections overlap
    if np.any(depth <= tol):
        k = int(np.argmin(depth))                                   # the direction of fastest separation
        n = axes[k].copy()
        if n @ centre < 0.0:
            n = -n
        return SeparatingAxis(False, n, float(-depth[k]))
    k = int(np.argmin(depth))
    n = axes[k].copy()
    if n @ centre < 0.0:
        n = -n
    return SeparatingAxis(True, n, float(depth[k]))


# ----------------------------------------------------------------------------------------------- clipping
def _clip_polygon(loop: np.ndarray, n: np.ndarray, d: float,
                  tol: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sutherland-Hodgman: the part of a convex polygon loop with ``n . x <= d``, plus the cut points."""
    side = loop @ n - d
    out: list[np.ndarray] = []
    cut: list[np.ndarray] = []
    m = len(loop)
    for k in range(m):
        cur, s_cur = loop[k], side[k]
        nxt, s_nxt = loop[(k + 1) % m], side[(k + 1) % m]
        inside_cur, inside_nxt = s_cur <= tol, s_nxt <= tol
        if inside_cur:
            out.append(cur)
        if inside_cur != inside_nxt:
            denom = s_cur - s_nxt
            p = cur + (nxt - cur) * (s_cur / denom) if abs(denom) > _EPS else cur
            out.append(p)
            cut.append(p)
    if not out:
        return np.zeros((0, 3)), cut
    pts = np.array(out)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > tol
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        keep[-1] = False
    pts = pts[keep]
    return pts, cut


def _cap_loop(cut: list[np.ndarray], n: np.ndarray, tol: float) -> np.ndarray:
    """Order the cut points around the cap so the loop winds counter-clockwise seen from ``+n``."""
    if len(cut) < 3:
        return np.zeros((0, 3))
    pts = np.array(cut)
    centre = pts.mean(0)
    # de-duplicate
    rel = pts - centre
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        close = np.linalg.norm(rel - rel[i], axis=1) <= tol
        close[i] = False
        keep &= ~close
    pts, rel = pts[keep], rel[keep]
    if len(pts) < 3:
        return np.zeros((0, 3))
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, seed)
    n1 = np.linalg.norm(t1)
    if n1 < _EPS:  # pragma: no cover - defensive
        return np.zeros((0, 3))
    t1 = t1 / n1
    t2 = np.cross(n, t1)
    ang = np.arctan2(rel @ t2, rel @ t1)
    order = np.argsort(ang, kind="stable")
    pts = pts[order]
    # reject a degenerate (collinear) cap: its area is zero, the intersection has no face there
    area = 0.5 * np.linalg.norm(np.cross(pts - pts[0], np.roll(pts, -1, axis=0) - pts[0]).sum(0))
    if area <= tol * tol:
        return np.zeros((0, 3))
    return pts


def _plane_from_polyhedron(p: ConvexPolyhedron, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = p.face_normals @ R.T
    offsets = p.face_offsets + normals @ t
    return normals, offsets


def intersection_loops(a: ConvexPolyhedron, b: ConvexPolyhedron,
                       Ra: np.ndarray | None = None, Rb: np.ndarray | None = None,
                       ta: np.ndarray | None = None, tb: np.ndarray | None = None,
                       tol: float = 1.0e-12) -> list[np.ndarray]:
    """Face loops (world coordinates) of ``a`` intersected with ``b``; empty when they do not overlap."""
    Ra = np.eye(3) if Ra is None else np.asarray(Ra, dtype=float)
    Rb = np.eye(3) if Rb is None else np.asarray(Rb, dtype=float)
    ta = np.zeros(3) if ta is None else np.asarray(ta, dtype=float)
    tb = np.zeros(3) if tb is None else np.asarray(tb, dtype=float)
    vb = b.vertices @ Rb.T + tb
    loops = [vb[list(face)] for face in b.faces]
    normals, offsets = _plane_from_polyhedron(a, Ra, ta)
    for n, d in zip(normals, offsets, strict=True):
        side = np.concatenate([loop @ n for loop in loops]) - d
        if not np.any(side > tol):
            continue                                                # body already inside: no cut, no cap face
        new: list[np.ndarray] = []
        cut: list[np.ndarray] = []
        for loop in loops:
            out, pts = _clip_polygon(loop, n, d, tol)
            if len(out) >= 3:
                new.append(out)
            cut.extend(pts)
        cap = _cap_loop(cut, n, tol)
        if len(cap):
            new.append(cap)
        loops = new
        if not loops:
            return []
    return loops


def loops_volume_centroid(loops: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Volume and centroid of a closed, consistently wound set of face loops (divergence theorem)."""
    if not loops:
        return 0.0, np.zeros(3)
    ref = np.vstack(loops).mean(0)
    volume = 0.0
    moment = np.zeros(3)
    for loop in loops:
        p = loop - ref
        for k in range(1, len(p) - 1):
            a, b, c = p[0], p[k], p[k + 1]
            det = float(np.dot(a, np.cross(b, c))) / 6.0
            if abs(det) < _EPS:
                continue
            volume += det
            moment += det * (a + b + c) / 4.0
    if abs(volume) < _EPS:
        return 0.0, nsafe(ref)
    centroid = ref + moment / volume
    return abs(volume), centroid


def nsafe(x: np.ndarray) -> np.ndarray:  # pragma: no cover - tiny helper
    return np.asarray(x, dtype=float)


def clipped_by_plane(p: ConvexPolyhedron, R: np.ndarray, t: np.ndarray,
                     normal: np.ndarray, offset: float,
                     tol: float = 1.0e-12) -> tuple[float, np.ndarray, float]:
    """Volume, centroid and penetration depth of the part of a polyhedron beyond ``normal . x >= offset``.

    The depth is how far the deepest vertex reaches past the plane: with the volume it gives the contact area
    (``A ~ V / depth``) and therefore the contact stiffness ``dF/ddepth = k_v V / depth``, which is what the damping
    and the tangential spring have to be sized from.
    """
    n = np.asarray(normal, dtype=float)
    n = n / max(np.linalg.norm(n), _EPS)
    loops = [p.world_vertices(R, t)[list(face)] for face in p.faces]
    side = np.concatenate([loop @ n for loop in loops]) - offset
    if not np.any(side > tol):
        return 0.0, np.zeros(3), 0.0                                 # the body does not reach the wall
    new: list[np.ndarray] = []
    cut: list[np.ndarray] = []
    keep_n, keep_d = -n, -offset                                     # keep: n . x >= offset
    for loop in loops:
        out, pts = _clip_polygon(loop, keep_n, keep_d, tol)
        if len(out) >= 3:
            new.append(out)
        cut.extend(pts)
    cap = _cap_loop(cut, keep_n, tol)
    if len(cap):
        new.append(cap)
    volume, centroid = loops_volume_centroid(new)
    depth = float(side.max()) if len(side) else 0.0
    return volume, centroid, depth


def overlap_volume(a: ConvexPolyhedron, b: ConvexPolyhedron,
                   Ra: np.ndarray | None = None, Rb: np.ndarray | None = None,
                   ta: np.ndarray | None = None, tb: np.ndarray | None = None,
                   tol: float = 1.0e-12) -> tuple[float, np.ndarray, SeparatingAxis]:
    """Shared volume, its centroid, and the separating axis of two convex polyhedra.

    The separating axis test runs first: it is cheap, it is exact, and it rejects the overwhelming majority of
    neighbour pairs (which never touch) before any clipping happens.
    """
    axis = separating_axis(a, b, Ra, Rb, ta, tb)
    if not axis.overlap:
        return 0.0, np.zeros(3), axis
    loops = intersection_loops(a, b, Ra, Rb, ta, tb, tol=tol)
    volume, centroid = loops_volume_centroid(loops)
    return volume, centroid, axis


# ------------------------------------------------------------------------------------- convex polytope tools
def polytope_from_halfspaces(normals: np.ndarray, offsets: np.ndarray,
                             interior: np.ndarray | None = None,
                             tol: float = 1.0e-10) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Vertices and outward-wound faces of ``{x : normal_i . x <= offset_i}`` (bounded polytope).

    Used by the Wulff construction, where every crystal facet contributes one plane at distance = its surface
    energy.  Returns arrays ready for :class:`~pinneapple_simulation.granular.shapes.ConvexPolyhedron`.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    normals = np.asarray(normals, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.asarray(offsets, dtype=float)
    if interior is None:
        interior = np.zeros(3)
    half = np.c_[normals, -offsets]                                  # scipy wants [A | -b] for A x + b <= 0
    hs = HalfspaceIntersection(half, np.asarray(interior, dtype=float))
    points = np.unique(np.round(hs.intersections, 12), axis=0)
    if len(points) < 4:
        raise ValueError("half-space intersection is not a bounded 3-D polytope")
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    remap = {int(old): new for new, old in enumerate(hull.vertices)}
    faces = []
    for simplex, plane in zip(hull.simplices, hull.equations, strict=True):
        loop = [remap[int(i)] for i in simplex]
        n = np.cross(verts[loop[1]] - verts[loop[0]], verts[loop[2]] - verts[loop[0]])
        if np.dot(n, plane[:3]) < 0.0:
            loop.reverse()
        faces.append(tuple(loop))
    _ = tol
    return verts, tuple(faces)


# ---------------------------------------------------------------------------------------- analytic overlaps
def sphere_sphere_lens_volume(r1, r2, distance):
    """Volume shared by two spheres (the lens of the classical sphere-cap formulas); array-aware."""
    r1, r2, d = (np.asarray(x, dtype=float) for x in (r1, r2, distance))
    d = np.maximum(d, 1.0e-30)
    h1 = r1 + (d ** 2 + r1 ** 2 - r2 ** 2) / (2.0 * d) - d
    h2 = r2 + (d ** 2 + r2 ** 2 - r1 ** 2) / (2.0 * d) - d
    h1 = np.clip(h1, 0.0, 2.0 * r1)
    h2 = np.clip(h2, 0.0, 2.0 * r2)
    lens = math.pi * (h1 * h1 * (3.0 * r1 - h1) + h2 * h2 * (3.0 * r2 - h2)) / 3.0
    lens = np.where(d >= r1 + r2, 0.0, lens)
    buried = 4.0 / 3.0 * math.pi * np.minimum(r1, r2) ** 3
    lens = np.where(d <= np.abs(r1 - r2), buried, lens)
    return lens if lens.ndim else float(lens)


def sphere_plane_cap_volume(radius, penetration):
    """Volume of a spherical cap of depth ``penetration``; array-aware."""
    r = np.asarray(radius, dtype=float)
    d = np.clip(np.asarray(penetration, dtype=float), 0.0, 2.0 * r)
    cap = math.pi * d * d * (3.0 * r - d) / 3.0
    return cap if cap.ndim else float(cap)
