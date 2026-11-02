"""Tessellation of exact bodies (#1116): quality presets, a fast analytic mesher and the kernel mesher.

``tessellate(body, quality="normal", mesher="auto")`` returns a :class:`TessMesh` in metres with a face id per
triangle.

Quality presets set the chordal deviation (relative to the body's bounding-box diagonal) and the largest angle
between neighbouring normals on curved faces:

    draft   2e-3 D, 30 deg      normal  5e-4 D, 15 deg      fine  1e-4 D, 7.5 deg

Meshers
-------
* ``fast``: bodies made only of planes, cylinders, cones and spheres whose curved faces are simple: a band
  between coaxial full circles, or a whole sphere. Every edge is sampled **once** and shared by the faces on both
  sides, so the mesh is watertight by construction; planes are triangulated with earcut (holes included), bands as
  strips between their two rings. The chord error of a circle sampled with n segments is r (1 - cos(pi/n)), so
  n = ceil(pi / acos(1 - tol/r)) (and at least 360/angle) keeps every edge and band within the tolerance.
* ``kernel``: the exact kernel's tessellation of the whole body (any surface; watertight). Face ids come from the
  per-face meshes of the same tolerance (nearest triangle centroid).
* ``auto``: ``fast`` when every face qualifies, else ``kernel`` for the whole body (mixing the two would leave
  cracks where their edge samplings differ); the faces that forced the kernel are listed in ``fallback_faces``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .step import MM, CadBody

QUALITY = {"draft": (2e-3, 30.0), "normal": (5e-4, 15.0), "fine": (1e-4, 7.5)}
_FAST_KINDS = {"plane", "cylinder", "cone", "sphere"}


@dataclass
class TessMesh:
    """A triangle mesh of one body, in metres."""
    vertices: np.ndarray                 # (n, 3) [m]
    triangles: np.ndarray                # (m, 3) int, counter-clockwise seen from outside
    face_id: np.ndarray                  # (m,) index into body.faces
    mesher: str                          # "fast" | "kernel"
    quality: str
    chord_tolerance_m: float
    angle_deg: float
    seconds: float
    fallback_faces: list[int] = field(default_factory=list)
    reason: str = ""

    @property
    def n_triangles(self) -> int:
        return int(len(self.triangles))

    def area(self) -> float:
        """Total triangle area [m^2]."""
        a, b, c = (self.vertices[self.triangles[:, k]] for k in range(3))
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    def volume(self) -> float:
        """Enclosed volume [m^3] by the divergence theorem (meaningful for closed meshes)."""
        a, b, c = (self.vertices[self.triangles[:, k]] for k in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def edge_use(self) -> np.ndarray:
        """How many triangles use each undirected edge (2 everywhere on a closed manifold)."""
        t = self.triangles
        e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return counts

    def is_watertight(self) -> bool:
        return bool(np.all(self.edge_use() == 2))

    def euler_characteristic(self) -> int:
        """V - E + F of the mesh (2 - 2 genus for one closed surface)."""
        return int(len(np.unique(self.triangles)) - len(self.edge_use()) + len(self.triangles))

    def face_areas(self) -> np.ndarray:
        """Mesh area [m^2] per body face."""
        a, b, c = (self.vertices[self.triangles[:, k]] for k in range(3))
        ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return np.bincount(self.face_id, weights=ar, minlength=int(self.face_id.max()) + 1)

    def report(self) -> dict[str, Any]:
        return {"mesher": self.mesher, "quality": self.quality, "chord_tolerance_m": self.chord_tolerance_m,
                "angle_deg": self.angle_deg, "vertices": int(len(self.vertices)), "triangles": self.n_triangles,
                "watertight": self.is_watertight(), "euler": self.euler_characteristic(),
                "area_m2": self.area(), "volume_m3": self.volume(), "seconds": self.seconds,
                "fallback_faces": self.fallback_faces, "reason": self.reason}


def _tolerances(body: CadBody, quality: str) -> tuple[float, float]:
    if quality not in QUALITY:
        raise ValueError(f"quality must be one of {sorted(QUALITY)}")
    rel, ang = QUALITY[quality]
    diag_mm = float(np.linalg.norm(body.size_m)) / MM
    return rel * diag_mm, ang


def circle_segments(radius: float, sweep_rad: float, tol: float, angle_deg: float) -> int:
    """Segments for an arc of the given radius and sweep so the sagitta is <= tol and each segment turns <= angle."""
    if radius <= tol:
        n_tol = 3
    else:
        n_tol = math.ceil(sweep_rad / (2.0 * math.acos(1.0 - tol / radius)))
    n_ang = math.ceil(math.degrees(sweep_rad) / angle_deg)
    return max(n_tol, n_ang, 3 if sweep_rad > math.pi * 1.5 else 1)


# --------------------------------------------------------------------------------------------- fast mesher
def _key(p) -> tuple:
    return tuple(int(round(c * 1e6)) for c in p)


class _EdgeCache:
    """Each edge sampled once (in mm); faces on both sides reuse the same points."""

    def __init__(self, tol: float, angle: float):
        self.tol, self.angle = tol, angle
        self.store: dict[tuple, np.ndarray] = {}

    def polyline(self, e: Any) -> np.ndarray:
        a, m, b = (np.asarray(list(e.point_at(t)), float) for t in (0.0, 0.5, 1.0))
        ka, km, kb = _key(a), _key(m), _key(b)
        key = (min(ka, kb), km, max(ka, kb))
        if key in self.store:
            pts = self.store[key]
            return pts if _key(pts[0]) == ka else pts[::-1]
        kind = str(e.geom_type).lower()
        if kind == "line":
            pts = np.stack([a, b])
        elif kind == "circle":
            r = float(e.radius if not callable(e.radius) else e.radius())
            sweep = float(e.length) / r
            n = circle_segments(r, sweep, self.tol, self.angle)
            pts = np.array([list(e.point_at(t)) for t in np.linspace(0.0, 1.0, n + 1)], float)
            if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
                pts[-1] = pts[0]
        else:
            raise _NotFast(f"edge type {kind}")
        self.store[key] = pts
        return pts


class _NotFast(Exception):
    pass


def _chain(polys: list[np.ndarray]) -> np.ndarray:
    """Join polylines of one wire into a closed ring (no repeated end point)."""
    polys = [p for p in polys if len(p)]
    if len(polys) == 1 and np.linalg.norm(polys[0][0] - polys[0][-1]) < 1e-9:
        return polys[0][:-1]
    ring = list(polys.pop(0))
    while polys:
        end = _key(ring[-1])
        for i, p in enumerate(polys):
            if _key(p[0]) == end:
                ring.extend(p[1:])
                polys.pop(i)
                break
            if _key(p[-1]) == end:
                ring.extend(p[::-1][1:])
                polys.pop(i)
                break
        else:
            raise _NotFast("open wire")
    if _key(ring[0]) == _key(ring[-1]):
        ring.pop()
    return np.array(ring)


def _plane_face(f: Any, cache: _EdgeCache) -> tuple[np.ndarray, np.ndarray]:
    import mapbox_earcut as earcut
    pl = f.plane
    o = np.asarray(list(pl.origin), float)
    ex = np.asarray(list(pl.x_dir), float)
    ez = np.asarray(list(pl.z_dir), float)
    ey = np.cross(ez, ex)
    rings = [_chain([cache.polyline(e) for e in f.outer_wire().edges()])]
    rings += [_chain([cache.polyline(e) for e in w.edges()]) for w in f.inner_wires()]
    pts3 = np.vstack(rings)
    uv = np.stack([(pts3 - o) @ ex, (pts3 - o) @ ey], 1)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = _flip_degenerate(uv, np.asarray(earcut.triangulate_float64(uv, ends), np.int64).reshape(-1, 3))
    normal = np.asarray(list(f.normal_at(f.interior_point())), float)    # outward, as the face sits in the body
    return pts3, _orient(pts3, tri, normal)


def _min_angles(uv: np.ndarray, tri: np.ndarray) -> np.ndarray:
    P = uv[tri]
    out = np.full(len(tri), np.pi)
    for i in range(3):
        u, v = P[:, (i + 1) % 3] - P[:, i], P[:, (i + 2) % 3] - P[:, i]
        out = np.minimum(out, np.abs(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1))))
    return out


def _flip_degenerate(uv: np.ndarray, tri: np.ndarray, min_angle_deg: float = 1e-4) -> np.ndarray:
    """Remove the zero-area triangles earcut leaves where ring points are collinear (the tops of holes in a row).

    Lawson edge flips by the max-min angle rule, started at the triangles whose smallest angle is below
    ``min_angle_deg`` and spreading only to the quads a flip changes. The vertices and the boundary edges do not
    change, so the body mesh stays watertight; the other triangles are kept as earcut made them (thin ones remain
    where a large face has only its boundary points: `mesh_quality` reports them)."""
    if not len(tri):
        return tri
    flat = np.where(_min_angles(uv, tri) < np.radians(min_angle_deg))[0]
    if not len(flat):
        return tri
    tri = [list(map(int, t)) for t in tri]
    edges: dict[tuple[int, int], set[int]] = {}

    def key(a, b):
        return (a, b) if a < b else (b, a)

    for k, t in enumerate(tri):
        for i in range(3):
            edges.setdefault(key(t[i], t[(i + 1) % 3]), set()).add(k)

    def min_angle(t):
        return float(_min_angles(uv, np.asarray([t]))[0])

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    queue = [key(tri[k][i], tri[k][(i + 1) % 3]) for k in flat for i in range(3)]
    for _ in range(50 * len(tri)):
        if not queue:
            break
        a, b = queue.pop()
        ks = edges.get((a, b), set())
        if len(ks) != 2:
            continue
        k1, k2 = ks
        c = next(v for v in tri[k1] if v != a and v != b)
        d = next(v for v in tri[k2] if v != a and v != b)
        A, B, C, D = uv[a], uv[b], uv[c], uv[d]
        if not (cross(C, D, A) * cross(C, D, B) < 0 and cross(A, B, C) * cross(A, B, D) <= 0):
            continue                                             # the quad is not strictly convex across cd
        n1, n2 = [c, d, a], [d, c, b]
        if min(min_angle(n1), min_angle(n2)) <= min(min_angle(tri[k1]), min_angle(tri[k2])) + 1e-12:
            continue
        for k in (k1, k2):
            for i in range(3):
                edges[key(tri[k][i], tri[k][(i + 1) % 3])].discard(k)
        tri[k1], tri[k2] = n1, n2
        for k in (k1, k2):
            for i in range(3):
                edges.setdefault(key(tri[k][i], tri[k][(i + 1) % 3]), set()).add(k)
        queue += [key(a, c), key(c, b), key(b, d), key(d, a)]
    return np.asarray(tri, np.int64)


def _orient(pts: np.ndarray, tri: np.ndarray, normal: np.ndarray) -> np.ndarray:
    if not len(tri):
        return tri
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    n = np.cross(b - a, c - a)
    flip = (n @ normal) < 0
    tri = tri.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def _band_face(f: Any, cache: _EdgeCache, axis_o: np.ndarray, axis_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder or cone face between two coaxial full circles: a strip between the two rings."""
    rings = []
    for e in f.edges():
        if str(e.geom_type).lower() != "circle" or not (e.is_closed() if callable(e.is_closed) else e.is_closed):
            if bool(e.is_seam() if callable(e.is_seam) else e.is_seam):
                continue
            raise _NotFast("band face with a non-circular or partial boundary")
        rings.append(cache.polyline(e)[:-1])
    if len(rings) != 2:
        raise _NotFast(f"band face with {len(rings)} boundary circles")
    ex = np.cross(axis_d, [1.0, 0, 0] if abs(axis_d[0]) < 0.9 else [0, 1.0, 0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(axis_d, ex)

    def ang(r):
        d = r - axis_o
        return np.mod(np.arctan2(d @ ey, d @ ex), 2 * np.pi)

    ordered = []
    for r in rings:
        a = ang(r)
        o = np.argsort(a)
        ordered.append((r[o], a[o]))
    (A, aa), (B, ab) = ordered
    nA, nB = len(A), len(B)
    pts = np.vstack([A, B])
    tris = []
    i = j = 0
    # merge the two angle sequences around the circle (closing back to the start)
    while i < nA or j < nB:
        ta = aa[(i + 1) % nA] + (2 * np.pi if i + 1 >= nA else 0) if i < nA else np.inf
        tb = ab[(j + 1) % nB] + (2 * np.pi if j + 1 >= nB else 0) if j < nB else np.inf
        if ta <= tb:
            tris.append([i % nA, (i + 1) % nA, nA + j % nB])
            i += 1
        else:
            tris.append([i % nA, nA + (j + 1) % nB, nA + j % nB])
            j += 1
    tri = np.array(tris, np.int64)
    # outward normal of this face at a sample point decides the winding
    p0 = 0.5 * (A[0] + B[np.argmin(np.abs(ab - aa[0]))])
    nrm = np.asarray(list(f.normal_at(p0.tolist())), float)
    return pts, _orient_band(pts, tri, nrm, p0)


def _orient_band(pts, tri, nrm, p0):
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    k = int(np.argmin(np.linalg.norm((a + b + c) / 3 - p0, axis=1)))
    n = np.cross(b[k] - a[k], c[k] - a[k])
    return tri if n @ nrm >= 0 else tri[:, [0, 2, 1]]


def _sphere_face(f: Any, tol: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    if [e for e in f.edges() if not bool(e.is_seam() if callable(e.is_seam) else e.is_seam)
            and float(e.length) > 1e-9]:
        raise _NotFast("trimmed sphere")
    R = float(f.radius)
    ax = f.axis
    c = np.asarray(list(ax.origin), float)
    n_lon = circle_segments(R, 2 * math.pi, tol, angle)
    n_lat = max(2, math.ceil(n_lon / 2))
    th = np.linspace(0, math.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0, 2 * math.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    pts = c + R * np.vstack([[0, 0, 1], ring, [0, 0, -1]])
    tris = []
    idx = lambda i, j: 1 + i * n_lon + (j % n_lon)  # noqa: E731
    for j in range(n_lon):
        tris.append([0, idx(0, j), idx(0, j + 1)])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            tris += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    last = len(pts) - 1
    for j in range(n_lon):
        tris.append([last, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)])
    tri = np.array(tris, np.int64)
    a, b, cc = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    out = (np.cross(b - a, cc - a) * ((a + b + cc) / 3 - c)).sum(1)
    tri[out < 0] = tri[out < 0][:, [0, 2, 1]]
    return pts, tri


def _fast(body: CadBody, tol: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = _EdgeCache(tol, angle)
    V, T, F = [], [], []
    off = 0
    for cf in body.faces:
        f = cf.shape
        if cf.kind == "plane":
            pts, tri = _plane_face(f, cache)
        elif cf.kind in ("cylinder", "cone"):
            ax = f.axis
            pts, tri = _band_face(f, cache, np.asarray(list(ax.origin), float),
                                  np.asarray(list(ax.direction), float))
        elif cf.kind == "sphere":
            pts, tri = _sphere_face(f, tol, angle)
        else:
            raise _NotFast(f"face type {cf.kind}")
        V.append(pts)
        T.append(tri + off)
        F.append(np.full(len(tri), cf.index))
        off += len(pts)
    V, T = np.vstack(V), np.vstack(T)
    # weld the shared edge samples (identical points by construction)
    q = np.round(V * 1e6).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.ravel()[T], np.concatenate(F)


def _kernel(body: CadBody, tol: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The kernel's mesh of the whole body, and the body face of each triangle.

    A triangle's face is the one its three vertices share, found by matching them to the vertices of the per-face
    meshes at the same tolerance. Where that is not one face (a cap triangle with all three vertices on the rim it
    shares with a band, or vertices the per-face meshes place differently), the face nearest to the centroid by the
    kernel's exact distance among the faces of its vertices (or, failing those, of the nearest per-face triangles)
    decides. Centroids alone are not enough: a long thin triangle can have its centroid next to another face."""
    import smlibpy as sm
    from scipy.spatial import cKDTree
    m = sm.tessellate(body.shape, tolerance=tol, angle=angle)
    V, T = np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)
    fverts, flab, cents, clab = [], [], [], []
    for cf in body.faces:
        fm = sm.tessellate(cf.shape, tolerance=tol, angle=angle)
        fv, ft = np.asarray(fm.vertices, float), np.asarray(fm.triangles, np.int64)
        fverts.append(fv)
        flab.append(np.full(len(fv), cf.index))
        if len(ft):
            cents.append(fv[ft].mean(1))
            clab.append(np.full(len(ft), cf.index))
    fl = np.concatenate(flab)
    size = float(np.ptp(V, axis=0).max()) if len(V) else 1.0
    hits = cKDTree(np.vstack(fverts)).query_ball_point(V, 1e-7 * size)
    nf = len(body.faces)
    pv = np.repeat(np.arange(len(V)), [len(h) for h in hits])
    pl = fl[np.concatenate([np.asarray(h, np.int64) for h in hits])] if len(pv) else np.zeros(0, np.int64)
    known = np.unique(pv * nf + pl)                             # (vertex, face) incidences, encoded
    # candidates: the faces of each triangle's first vertex that its other two vertices also touch
    order = np.argsort(pv, kind="stable")
    pv, pl = pv[order], pl[order]
    start = np.searchsorted(pv, np.arange(len(V)))
    count = np.diff(np.r_[start, len(pv)])
    k = np.repeat(np.arange(len(T)), count[T[:, 0]])
    off = np.arange(len(k)) - np.repeat(np.cumsum(count[T[:, 0]]) - count[T[:, 0]], count[T[:, 0]])
    lab = pl[start[T[k, 0]] + off]
    ok = np.isin(T[k, 1] * nf + lab, known) & np.isin(T[k, 2] * nf + lab, known)
    n_common = np.bincount(k[ok], minlength=len(T))
    F = np.full(len(T), -1, np.int64)
    F[k[ok]] = lab[ok]
    C = V[T].mean(1)
    rest = np.where(n_common != 1)[0]
    if len(rest):
        vsets = [frozenset(fl[h].tolist()) for h in hits]
        _, near = cKDTree(np.vstack(cents)).query(C[rest], k=4)
        clab = np.concatenate(clab)
        for r, kk in enumerate(rest):
            a, b, c = T[kk]
            common = vsets[a] & vsets[b] & vsets[c]
            cand = common or (vsets[a] | vsets[b] | vsets[c]) or frozenset(clab[near[r]].tolist())
            F[kk] = min(cand, key=lambda f: body.faces[f].shape.distance_to(C[kk].tolist()))
    return V, T, F


def tessellate(body: CadBody, quality: str = "normal", mesher: str = "auto") -> TessMesh:
    """Triangle mesh of an exact body (see the module docstring for the presets and meshers).

    Parameters
    ----------
    body : CadBody
        From :func:`pinneapple_design.geometry.brep.read_step`.
    quality : {"draft", "normal", "fine"}
    mesher : {"auto", "fast", "kernel"}
    """
    tol, ang = _tolerances(body, quality)
    t0 = time.perf_counter()
    reason, fallback = "", []
    use = mesher
    if mesher in ("auto", "fast"):
        bad = [f.index for f in body.faces if f.kind not in _FAST_KINDS]
        try:
            if bad:
                raise _NotFast(f"faces of type {sorted({body.faces[i].kind for i in bad})}")
            V, T, F = _fast(body, tol, ang)
            use = "fast"
        except _NotFast as e:
            if mesher == "fast":
                raise ValueError(f"the fast mesher cannot mesh this body: {e}") from None
            reason, fallback, use = str(e), bad, "kernel"
    if use == "kernel":
        V, T, F = _kernel(body, tol, ang)
    return TessMesh(vertices=V * MM, triangles=T, face_id=F, mesher=use, quality=quality, chord_tolerance_m=tol * MM,
                    angle_deg=ang, seconds=time.perf_counter() - t0, fallback_faces=fallback, reason=reason)
