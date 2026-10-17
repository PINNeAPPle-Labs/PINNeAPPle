"""Conforming Delaunay meshes of machine sectors and their assembly into a rotating machine mesh.

:func:`mesh_sector` triangulates one annular sector (:class:`~.geometry.SectorGeometry`) so that every edge
of every polygon, both arcs and both radial sides are edges of the mesh: boundary points are placed at the
local size ``h``, interior points on a staggered polar lattice, and any boundary segment whose diametral
circle contains another point is split (or the interior point removed) until every segment is a Gabriel edge,
which a Delaunay triangulation always contains. Elements are then named by the polygon that contains their
centroid, so the regions are the polygons exactly.

:class:`MachineMesh` copies the rotor-pole sector and the stator-slot sector around the circle (identical
elements in every pole and slot, so cogging and ripple carry no mesh noise of their own) and joins them
through the air gap with three ring layers: rotor air, moving band, stator air. Turning the rotor moves the
rotor nodes rigidly and re-triangulates only the band (:func:`~..mesh.zipper`), so rotor and stator
elements keep their indices at every angle.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from ..mesh import Mesh2D, _orient, zipper
from .geometry import MachineGeometry, Polygon, SectorGeometry, polygon_area

__all__ = ["SectorMesh", "mesh_sector", "replicate_sector", "MachineMesh"]

TWO_PI = 2.0 * math.pi


@dataclass
class SectorMesh:
    points: np.ndarray       # (N, 2)
    tris: np.ndarray         # (M, 3), counter-clockwise
    region: list[str]        # per element: polygon name or the sector's fill
    polygon: np.ndarray      # (M,) index of the polygon of each element, -1 for fill
    arc_in: np.ndarray       # node indices on the inner arc, by increasing angle
    arc_out: np.ndarray
    side0: np.ndarray        # node indices on the side phi = -span/2, by increasing radius
    side1: np.ndarray        # same radii on phi = +span/2

    def min_angle(self) -> float:
        """Smallest interior angle of all triangles (degrees)."""
        p = self.points[self.tris]
        ang = []
        for k in range(3):
            a, b, c = p[:, k], p[:, (k + 1) % 3], p[:, (k + 2) % 3]
            u, v = b - a, c - a
            cosv = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
            ang.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
        return float(np.min(ang))


def _subdivide(p0, p1, h):
    n = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / h)))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return p0 + t * (p1 - p0)


def mesh_sector(geom: SectorGeometry, size: Callable[[np.ndarray], np.ndarray] | float, n_in: int, n_out: int,
                smooth: int = 3, mirror: bool = False, max_rounds: int = 60) -> SectorMesh:
    """Mesh ``geom`` with element size ``size(r)`` (metres, a function of radius, or a constant); the arcs get
    ``n_in`` and ``n_out`` equal intervals (plus the extra nodes of ``geom.arc_nodes_in/out``). ``smooth``
    rounds of Laplacian smoothing of the interior points improve the angles (boundaries stay fixed).

    ``mirror=True`` (geometry symmetric about the sector axis, ``n_in`` and ``n_out`` even) meshes the half
    ``phi < 0`` and reflects it, so the mesh itself is symmetric: a symmetric rotor in a symmetric stator then
    has exactly zero torque at the aligned positions, and the cogging curve no mean offset."""
    hf = (lambda r: np.full_like(np.asarray(r, float), float(size))) if np.isscalar(size) else size
    half = geom.span / 2.0
    if not mirror:
        P, tri = _triangulate(geom, hf, n_in, n_out, smooth, max_rounds)
    else:
        if n_in % 2 or n_out % 2:
            raise ValueError("mirror=True needs even n_in and n_out")
        q = half / 2.0  # the lower half [-half, 0], turned by +q to be centred on the x-axis
        polys = []
        for pg in geom.polygons:
            xy = _clip_lower(pg.xy)
            r = np.hypot(xy[:, 0], xy[:, 1])
            for rr in (geom.r_in, geom.r_out):  # a chord clipped on the axis next to an arc: snap onto the arc
                near = (np.abs(r - rr) < 1e-6 * rr) & (np.abs(xy[:, 1]) <= 1e-15)
                xy[near] = np.column_stack([np.full(near.sum(), rr), np.zeros(near.sum())])
            if len(xy) >= 3 and abs(polygon_area(xy)) > 0:
                polys.append(Polygon(pg.name, _turn(xy, q)))
        hg = SectorGeometry(geom.r_in, geom.r_out, half, geom.fill, polys,
                            [a + q for a in geom.arc_nodes_in if a < -1e-15],
                            [a + q for a in geom.arc_nodes_out if a < -1e-15])
        Ph, th = _triangulate(hg, hf, n_in // 2, n_out // 2, smooth, max_rounds)
        Ph = _turn(Ph, -q)
        on_axis = np.abs(Ph[:, 1]) < 1e-12 * geom.r_out
        n = len(Ph)
        mirror_of = np.arange(n)
        off = np.where(~on_axis)[0]
        mirror_of[off] = n + np.arange(len(off))
        P = np.vstack([Ph, Ph[off] * np.array([1.0, -1.0])])
        tri = np.vstack([th, mirror_of[th]])
        tri = _orient(P, tri)

    # nodes on arcs and sides
    def on_circle(r):
        return np.where(np.abs(np.hypot(P[:, 0], P[:, 1]) - r) < 1e-9 * r)[0]

    def on_side(a):
        n = np.array([math.cos(a + math.pi / 2), math.sin(a + math.pi / 2)])
        d = np.abs(P @ n)
        along = P @ np.array([math.cos(a), math.sin(a)])
        return np.where((d < 1e-12 + 1e-9 * geom.r_out) & (along > 0))[0]

    ai, ao = on_circle(geom.r_in), on_circle(geom.r_out)
    ai = ai[np.argsort(np.arctan2(P[ai, 1], P[ai, 0]))]
    ao = ao[np.argsort(np.arctan2(P[ao, 1], P[ao, 0]))]
    s0, s1 = on_side(-half), on_side(half)
    s0 = s0[np.argsort(np.hypot(P[s0, 0], P[s0, 1]))]
    s1 = s1[np.argsort(np.hypot(P[s1, 0], P[s1, 1]))]
    if len(s0) != len(s1) or not np.allclose(np.hypot(*P[s0].T), np.hypot(*P[s1].T), rtol=0, atol=1e-12):
        raise RuntimeError("mesh_sector: the two radial sides have different nodes")

    from matplotlib.path import Path

    poly = -np.ones(len(tri), np.int64)
    cc = P[tri].mean(1)
    for k, pg in enumerate(geom.polygons):
        poly[Path(pg.xy).contains_points(cc)] = k
    names = [geom.fill if k < 0 else geom.polygons[k].name for k in poly]
    return SectorMesh(P, tri, names, poly, ai, ao, s0, s1)


def _triangulate(geom: SectorGeometry, hf, n_in: int, n_out: int, smooth: int, max_rounds: int):
    """Points and triangles of a conforming Delaunay mesh of ``geom`` (see :func:`mesh_sector`)."""
    half = geom.span / 2.0
    pts: list[np.ndarray] = []
    segs: list[tuple[int, int, float]] = []  # (i, j, circle radius or 0 for straight)

    def add(p):
        pts.append(np.asarray(p, float))
        return len(pts) - 1
    def arc_nodes(r, n, extra):
        ang = list(-half + geom.span * np.arange(n + 1) / n)
        step = geom.span / n
        for a in extra:
            k = int(np.argmin(np.abs(np.asarray(ang) - a)))
            if abs(ang[k] - a) < 0.3 * step and 0 < k < n:
                ang[k] = a
            elif abs(ang[k] - a) > 1e-12:
                ang.append(a)
        ang = np.sort(np.asarray(ang))
        return [add((r * math.cos(a), r * math.sin(a))) for a in ang]

    arc_in = arc_nodes(geom.r_in, n_in, geom.arc_nodes_in)
    arc_out = arc_nodes(geom.r_out, n_out, geom.arc_nodes_out)
    for ring, r in ((arc_in, geom.r_in), (arc_out, geom.r_out)):
        segs += [(ring[k], ring[k + 1], r) for k in range(len(ring) - 1)]
    # radial sides: same radii on both
    radii = [geom.r_in]
    while radii[-1] < geom.r_out:
        radii.append(radii[-1] + float(hf(radii[-1])))
    radii = np.linspace(geom.r_in, geom.r_out, len(radii)) if len(radii) > 2 else np.array([geom.r_in, geom.r_out])

    # polygon vertices on a radial side become side nodes, on both sides (the sector is copied around)
    def side_radius(q):
        r = math.hypot(q[0], q[1])
        a = math.atan2(q[1], q[0])
        return r if abs(abs(a) - half) * r < 1e-9 * geom.r_out and geom.r_in < r < geom.r_out else None

    extra = sorted({round(rv, 15) for poly in geom.polygons for q in poly.xy if (rv := side_radius(q)) is not None})
    if extra:
        h_side = np.diff(radii).min() if len(radii) > 1 else geom.r_out - geom.r_in
        radii = np.array([r for r in radii if min(abs(r - e) for e in extra) > 0.3 * h_side] + extra)
        radii = np.sort(np.concatenate([radii, [geom.r_in, geom.r_out]]))
        radii = radii[np.concatenate([[True], np.diff(radii) > 1e-12 * geom.r_out])]
    for sg, ring_end in ((-1, 0), (1, -1)):
        a = sg * half
        idx = [arc_in[ring_end]] + [add((r * math.cos(a), r * math.sin(a))) for r in radii[1:-1]] + [arc_out[ring_end]]
        segs += [(idx[k], idx[k + 1], 0.0) for k in range(len(idx) - 1)]
    # polygon edges (corners shared between polygons become one node). Polygon points on the arcs or on the
    # radial sides snap to the boundary's nodes (or are left out): the boundary segments already bound the
    # polygon there.
    def on_arc(q):
        r = math.hypot(q[0], q[1])
        return (abs(r - geom.r_in) < 1e-9 * geom.r_in or abs(r - geom.r_out) < 1e-9 * geom.r_out
                or side_radius(q) is not None)

    for poly in geom.polygons:
        xy = poly.xy
        nodes = []
        for k in range(len(xy)):
            p0, p1 = xy[k], xy[(k + 1) % len(xy)]
            h = float(hf(np.hypot(*(0.5 * (p0 + p1)))))
            for q in _subdivide(p0, p1, h)[:-1]:
                if on_arc(q):
                    j = _find(pts, q, 1e-9 * geom.r_out)
                    if j is not None:
                        nodes.append((j, True))
                else:
                    nodes.append((_find_or_add(pts, q), False))
        for k in range(len(nodes)):
            (a, arc_a), (b, arc_b) = nodes[k], nodes[(k + 1) % len(nodes)]
            if not (arc_a and arc_b):
                segs.append((a, b, 0.0))
    segs = _dedupe_segments(segs)
    n_bnd = len(pts)

    # interior lattice
    r = geom.r_in
    row = 0
    interior = []
    while True:
        h = float(hf(r))
        r = r + 0.866 * h if row else r + 0.5 * h
        if r >= geom.r_out - 0.3 * float(hf(geom.r_out)):
            break
        h = float(hf(r))
        dphi = h / r
        n = max(1, int(math.floor(geom.span / dphi)))
        phis = -half + (np.arange(n) + (0.5 if row % 2 else 1.0)) * geom.span / (n + (0 if row % 2 else 1))
        phis = phis[np.abs(phis) < half - 0.3 * dphi]
        interior += [(r * math.cos(a), r * math.sin(a)) for a in phis]
        row += 1
    interior = np.asarray(interior).reshape(-1, 2)
    bnd = np.asarray(pts)
    if len(interior):
        d, _ = cKDTree(bnd).query(interior)
        hloc = hf(np.hypot(interior[:, 0], interior[:, 1]))
        interior = interior[d > 0.55 * hloc]

    P = np.vstack([bnd, interior])
    free = np.zeros(len(P), bool)
    free[n_bnd:] = True
    seg_list = list(segs)
    corner = ~free
    for sweep in range(smooth + 1):
        P, free, seg_list, corner = _recover(P, free, seg_list, max_rounds, corner)
        if sweep == smooth:
            break
        P = _smooth(P, free, Delaunay(P).simplices)
        r = np.hypot(P[free, 0], P[free, 1])
        ph = np.arctan2(P[free, 1], P[free, 0])
        h = hf(r)
        r = np.clip(r, geom.r_in + 0.3 * h, geom.r_out - 0.3 * h)
        lim = half - 0.3 * h / r
        ph = np.clip(ph, -lim, lim)
        P[free] = np.column_stack([r * np.cos(ph), r * np.sin(ph)])
    tri = Delaunay(P).simplices
    c = P[tri].mean(1)
    rc_ = np.hypot(c[:, 0], c[:, 1])
    ph = np.arctan2(c[:, 1], c[:, 0])
    ok = (rc_ > geom.r_in) & (rc_ < geom.r_out) & (np.abs(ph) < half)
    tri = _orient(P, tri[ok])
    edges = set(map(tuple, np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)))
    missing = [s for s in seg_list if tuple(sorted(s[:2])) not in edges]
    if missing:
        raise RuntimeError(f"mesh_sector: {len(missing)} boundary segments not recovered")
    return P, tri


def _turn(xy: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray(xy, float) @ np.array([[c, s], [-s, c]])


def _clip_lower(xy: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a closed polygon to the half-plane y <= 0."""
    out = []
    n = len(xy)
    for k in range(n):
        p, q = xy[k], xy[(k + 1) % n]
        pin, qin = p[1] <= 1e-15, q[1] <= 1e-15
        if pin:
            out.append((p[0], min(p[1], 0.0) if abs(p[1]) <= 1e-15 else p[1]))
        if pin != qin and abs(q[1] - p[1]) > 0:
            t = p[1] / (p[1] - q[1])
            if 0.0 < t < 1.0:
                out.append((p[0] + t * (q[0] - p[0]), 0.0))
    res = []
    for pnt in out:
        if not res or abs(pnt[0] - res[-1][0]) + abs(pnt[1] - res[-1][1]) > 1e-15:
            res.append(pnt)
    if len(res) > 1 and abs(res[0][0] - res[-1][0]) + abs(res[0][1] - res[-1][1]) <= 1e-15:
        res.pop()
    return np.array(res).reshape(-1, 2)


def _recover(P, free, seg_list, max_rounds, corner=None):
    """Make every segment a Gabriel edge: drop interior points inside a segment's diametral circle, or split
    the segment when a boundary point is inside (arc segments are split on their circle). ``corner`` marks the
    input boundary points; a straight segment from a corner to a split point is split on concentric shells
    around the corner (Ruppert / Shewchuk), which ends for small input angles."""
    if corner is None:
        corner = ~free
    for _ in range(max_rounds):
        tree = cKDTree(P)
        drop = np.zeros(len(P), bool)
        new_pts, new_segs, changed = [], [], False
        for (i, j, rc) in seg_list:
            m = 0.5 * (P[i] + P[j])
            rho = 0.5 * np.linalg.norm(P[i] - P[j])
            near = [k for k in tree.query_ball_point(m, rho * (1 - 1e-9)) if k != i and k != j and not drop[k]]
            if not near:
                new_segs.append((i, j, rc))
                continue
            changed = True
            if all(free[k] for k in near):
                drop[near] = True
                new_segs.append((i, j, rc))
            else:
                k = len(P) + len(new_pts)
                new_pts.append(m if rc == 0.0 else m * (rc / np.linalg.norm(m)))
                if rc == 0.0 and corner[i] != corner[j]:
                    # concentric shells: split at a power-of-two distance from the corner, so segments meeting
                    # at a small angle get their points at equal distances and stop encroaching on each other
                    v, w = (i, j) if corner[i] else (j, i)
                    L = 2.0 * rho
                    d = 2.0 ** round(math.log2(0.5 * L))
                    d = min(max(d, 0.25 * L), 0.75 * L)
                    new_pts[-1] = P[v] + (P[w] - P[v]) * (d / L)
                new_segs += [(i, k, rc), (k, j, rc)]
        if not changed:
            return P, free, seg_list, corner
        keep = ~drop
        if new_pts:
            P = np.vstack([P, np.asarray(new_pts)])
            free = np.concatenate([free, np.zeros(len(new_pts), bool)])
            corner = np.concatenate([corner, np.zeros(len(new_pts), bool)])
            keep = np.concatenate([keep, np.ones(len(new_pts), bool)])
        remap = -np.ones(len(P), np.int64)
        remap[keep] = np.arange(keep.sum())
        P, free, corner = P[keep], free[keep], corner[keep]
        seg_list = [(int(remap[i]), int(remap[j]), rc) for i, j, rc in new_segs]
    raise RuntimeError("mesh_sector: boundary recovery did not converge")


def _smooth(P, free, tri, iters=4):
    """Laplacian smoothing of the free points over the edge graph of ``tri``."""
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    P = P.copy()
    for _ in range(iters):
        s = np.zeros_like(P)
        n = np.zeros(len(P))
        np.add.at(s, e[:, 0], P[e[:, 1]])
        np.add.at(s, e[:, 1], P[e[:, 0]])
        np.add.at(n, e[:, 0], 1)
        np.add.at(n, e[:, 1], 1)
        P[free] = 0.5 * P[free] + 0.5 * s[free] / n[free, None]
    return P


def _find(pts, q, tol):
    for k in range(len(pts)):
        if abs(pts[k][0] - q[0]) <= tol and abs(pts[k][1] - q[1]) <= tol:
            return k
    return None


def _find_or_add(pts, q, tol=1e-12):
    for k in range(len(pts) - 1, -1, -1):
        if abs(pts[k][0] - q[0]) <= tol and abs(pts[k][1] - q[1]) <= tol:
            return k
    pts.append(np.asarray(q, float))
    return len(pts) - 1


def _dedupe_segments(segs):
    seen, out = set(), []
    for i, j, rc in segs:
        key = (min(i, j), max(i, j))
        if i != j and key not in seen:
            seen.add(key)
            out.append((i, j, rc))
    return out


def _replicate(sm: SectorMesh, copies: int, span: float):
    """Copy a sector mesh ``copies`` times around the circle, merging the shared radial sides.

    Returns points, tris, sector index per element, local element index, and the global node indices of the
    inner and outer arcs (by increasing angle, the closing node not repeated)."""
    nloc = len(sm.points)
    is_s0 = np.zeros(nloc, bool)
    is_s0[sm.side0] = True
    new_local = np.where(~is_s0)[0]
    gmap = np.empty((copies, nloc), np.int64)
    pts, count = [], 0
    for k in range(copies):
        a = k * span
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        if k == 0:
            gmap[0] = np.arange(nloc)
            pts.append(sm.points @ R.T)
            count = nloc
            continue
        gmap[k, new_local] = count + np.arange(len(new_local))
        gmap[k, sm.side0] = gmap[k - 1, sm.side1]
        pts.append((sm.points[new_local]) @ R.T)
        count += len(new_local)
    # close the circle: the last copy's side1 is the first copy's side0
    last = gmap[copies - 1, sm.side1].copy()
    keep = np.ones(count, bool)
    keep[last] = False
    remap_last = np.arange(count)
    remap_last[last] = gmap[0, sm.side0]
    points = np.vstack(pts)
    tris = np.vstack([remap_last[gmap[k][sm.tris]] for k in range(copies)])
    order = -np.ones(count, np.int64)
    order[keep] = np.arange(keep.sum())
    points = points[keep]
    tris = order[tris]
    gmap = order[remap_last[gmap]]
    sector = np.repeat(np.arange(copies), len(sm.tris))
    local = np.tile(np.arange(len(sm.tris)), copies)
    arc_in = np.concatenate([gmap[k, sm.arc_in[:-1]] for k in range(copies)])
    arc_out = np.concatenate([gmap[k, sm.arc_out[:-1]] for k in range(copies)])
    return points, tris, sector, local, arc_in, arc_out


def replicate_sector(sm: SectorMesh, copies: int, span: float) -> Mesh2D:
    """The full circle from ``copies`` rotated copies of a sector mesh. Region index per element: the polygon
    index (fill = -1 shifted to 0, polygons from 1); ``boundary``: the nodes of both arcs; extra attributes
    ``sector`` and ``local`` give each element's copy and its index in the sector."""
    P, T, sec, loc, ai, ao = _replicate(sm, copies, span)
    names = ["fill"] + [f"polygon{k}" for k in range(int(sm.polygon.max()) + 1)]
    mesh = Mesh2D(P, _orient(P, T), sm.polygon[loc] + 1, names, boundary=np.concatenate([ai, ao]))
    mesh.sector, mesh.local = sec, loc
    return mesh


def _symmetric_strip(ia, ang_a, ib, ang_b, sectors: int) -> np.ndarray:
    """Triangulate the strip between two closed rings that are both symmetric about the axes ``2 pi k / sectors``:
    each half sector by the open zipper, the other half as its mirror image (same input, mirrored angles)."""
    pitch = TWO_PI / sectors
    tol = 1e-9
    out = []
    for k in range(sectors):
        c = k * pitch
        for sg in (-1.0, 1.0):
            sel = []
            for idx, ang in ((ia, ang_a), (ib, ang_b)):
                d = sg * (np.mod(np.asarray(ang) - c + math.pi, TWO_PI) - math.pi)  # mirrored for sg = 1
                m = (d >= -pitch / 2 - tol) & (d <= tol)
                o = np.argsort(d[m])
                sel.append((np.asarray(idx)[m][o], d[m][o]))
            (na, da), (nb, db) = sel
            out.append(zipper(na, da, nb, db, closed=False))
    return np.vstack(out)


class MachineMesh:
    """Mesh of a complete radial-flux machine (:class:`~.geometry.MachineGeometry`).

    ``n_gap`` nodes on the air-gap circles (a multiple of poles and slots), element size ``h_gap`` =
    ``2 pi r_gap / n_gap`` at the gap growing by ``grading`` per metre away from it, at most ``h_max``.
    ``mirror`` meshes half a pole and half a slot pitch and reflects them (symmetric geometry assumed).
    The gap is split into three equal ring layers: rotor air (turns with the rotor), moving band, stator air.
    Element order: rotor, rotor air, stator, stator air, band (the band is re-triangulated at every angle).

    Attributes after construction: ``n_rotor_nodes`` (rotor nodes come first and turn with the rotor),
    ``region_names``, ``region`` (per element), ``pole``/``slot`` index per element (-1 elsewhere),
    ``magnet_angle`` (per element, rotor frame, including the pole's polarity; NaN outside magnets),
    ``annuli``: [(r1, r2, element indices)] of the two air layers for the Maxwell-stress torque,
    ``boundary``: nodes of the shaft and outer circles (A = 0).
    """

    def __init__(self, geometry: MachineGeometry, n_gap: int = 960, h_max: float | None = None,
                 grading: float = 0.2, smooth: int = 3, mirror: bool = True):
        rot, sta = geometry.rotor, geometry.stator
        P, S = rot.poles, sta.slots
        if n_gap % P or n_gap % S or (mirror and (n_gap // P % 2 or n_gap // S % 2)):
            raise ValueError("n_gap must be a multiple of the numbers of poles and of slots (of twice those with "
                             "mirror=True)")
        self.geometry = geometry
        self.n_gap = n_gap
        r_gap = 0.5 * (rot.r_out + sta.r_in)
        h_gap = TWO_PI * r_gap / n_gap
        h_max = h_max or 6.0 * h_gap
        self.size = lambda r: np.minimum(h_max, h_gap + grading * np.abs(np.asarray(r, float) - r_gap))
        hr = float(self.size(rot.r_in))
        n_rin = 2 * max(1, int(round(rot.pole_pitch * rot.r_in / (2 * hr))))
        hs = float(self.size(sta.r_out))
        n_sout = 2 * max(1, int(round(sta.slot_pitch * sta.r_out / (2 * hs))))
        self.rotor_sector = mesh_sector(rot.sector(h_gap), self.size, n_rin, n_gap // P, smooth=smooth, mirror=mirror)
        self.stator_sector = mesh_sector(sta.sector(h_gap), self.size, n_gap // S, n_sout, smooth=smooth,
                                         mirror=mirror)

        rp, rt, rsec, rloc, r_shaft, r_surf = _replicate(self.rotor_sector, P, rot.pole_pitch)
        sp, st, ssec, sloc, s_bore, s_outer = _replicate(self.stator_sector, S, sta.slot_pitch)
        g = sta.r_in - rot.r_out
        r1, r2 = rot.r_out + g / 3.0, rot.r_out + 2.0 * g / 3.0
        ang_surf = np.arctan2(rp[r_surf, 1], rp[r_surf, 0])
        o = np.argsort(np.mod(ang_surf, TWO_PI))
        r_surf = r_surf[o]
        base = np.mod(ang_surf[o][0], TWO_PI)
        ang1 = base + TWO_PI * np.arange(n_gap) / n_gap
        ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
        n_r = len(rp) + n_gap  # rotating nodes: rotor + ring r1
        ring2 = np.column_stack([r2 * np.cos(ang1 - base), r2 * np.sin(ang1 - base)])
        self.points0 = np.vstack([rp, ring1, ring2, sp])
        self.n_rotor_nodes = n_r
        i1 = len(rp) + np.arange(n_gap)
        i2 = n_r + np.arange(n_gap)
        s_off = n_r + n_gap
        s_bore = s_bore + s_off
        ab = np.arctan2(self.points0[s_bore, 1], self.points0[s_bore, 0])
        s_bore = s_bore[np.argsort(np.mod(ab, TWO_PI))]
        self._ring1, self._ring2 = i1, i2
        self._ang1, self._ang2 = ang1, ang1 - base

        t_rair = zipper(r_surf, np.mod(np.arctan2(rp[r_surf, 1], rp[r_surf, 0]), TWO_PI), i1, np.mod(ang1, TWO_PI))
        t_sair = _symmetric_strip(i2, self._ang2, s_bore, np.arctan2(self.points0[s_bore, 1],
                                                                     self.points0[s_bore, 0]), S)
        fixed = [rt, t_rair, st + s_off, t_sair]
        self._fixed = _orient(self.points0, np.vstack(fixed))
        n_band = 2 * n_gap
        n_fixed = len(self._fixed)
        self.n_elements = n_fixed + n_band

        # regions
        names = ["rotor_iron", "pocket", "gap", "stator_iron", "opening", "wedge"]
        names += [f"magnet{k}" for k in range(P)] + [f"slot{k}" for k in range(S)]
        idx = {n: i for i, n in enumerate(names)}
        region = np.empty(self.n_elements, np.int64)
        pole = -np.ones(self.n_elements, np.int64)
        slot = -np.ones(self.n_elements, np.int64)
        mag = np.full(self.n_elements, np.nan)
        rnames = np.array(self.rotor_sector.region, dtype=object)[rloc]
        polys = self.rotor_sector.polygon[rloc]
        geo_r = rot.sector(h_gap)
        for e, (nm, sec, pk) in enumerate(zip(rnames, rsec, polys, strict=True)):
            if nm.startswith("magnet"):
                region[e] = idx[f"magnet{sec}"]
                a = geo_r.polygons[pk].magnetization + sec * rot.pole_pitch
                mag[e] = a + (math.pi if sec % 2 else 0.0)
            elif nm.startswith("pocket"):
                region[e] = idx["pocket"]
            else:
                region[e] = idx["rotor_iron"]
            pole[e] = sec
        n0 = len(rt)
        region[n0:n0 + len(t_rair)] = idx["gap"]
        n0 += len(t_rair)
        snames = np.array(self.stator_sector.region, dtype=object)[sloc]
        for k, (nm, sec) in enumerate(zip(snames, ssec, strict=True)):
            if nm == "slot":
                region[n0 + k] = idx[f"slot{sec}"]
            else:
                region[n0 + k] = idx[nm]
            slot[n0 + k] = sec
        n0 += len(st)
        region[n0:] = idx["gap"]  # stator air and band
        self.region_names = names
        self.region = region
        self.pole = pole
        self.slot = slot
        self.magnet_angle = mag
        ra = np.arange(len(rt), len(rt) + len(t_rair))
        sa = np.arange(len(rt) + len(t_rair) + len(st), n_fixed)
        self.annuli = [(rot.r_out, r1, ra), (r2, sta.r_in, sa)]
        self.rotor_elements = np.arange(len(rt) + len(t_rair))
        self.boundary = np.concatenate([r_shaft, s_outer + s_off])
        self.band_step = TWO_PI / n_gap

    def mesh(self, theta: float = 0.0) -> Mesh2D:
        """The machine with the rotor turned by ``theta`` (counter-clockwise, radians)."""
        c, s = math.cos(theta), math.sin(theta)
        pts = self.points0.copy()
        n = self.n_rotor_nodes
        pts[:n] = self.points0[:n] @ np.array([[c, s], [-s, c]])
        band = zipper(self._ring1, np.mod(self._ang1 + theta, TWO_PI), self._ring2, np.mod(self._ang2, TWO_PI))
        tris = np.vstack([self._fixed, _orient(pts, band)])
        return Mesh2D(pts, tris, self.region, self.region_names, boundary=self.boundary)
