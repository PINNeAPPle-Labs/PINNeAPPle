"""Geometry for immersed analysis: anything that answers "how far inside or outside is this point".

Every geometry is a level set ``phi(X)`` (negative inside, positive outside, zero on the surface) with a bounding box
and a Lipschitz bound (1 for a signed distance). The solver never meshes it: the cut-cell quadrature only evaluates
``phi`` on a background grid and its subcells.

- :class:`SDFGeometry` wraps any vectorised signed distance (the primitives below, their CSG, the SDFs of
  ``pinneapple_design.geometry.gen.sdf_shapes`` or any callable).
- :class:`VoxelGeometry` turns a voxel mask (CT scan, topology optimisation, a rasterised part) into a signed distance
  by Euclidean distance transforms, interpolated trilinearly.
- :class:`STLGeometry` takes a triangle surface: the inside test is the generalised winding number (Jacobson et al.,
  SIGGRAPH 2013; robust to small holes and flipped facets), evaluated with a dipole tree (Barill et al., SIGGRAPH
  2018), and the distance is the exact point-triangle distance to the nearest candidates.
"""
from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


class Geometry:
    """A level set with a bounding box. Subclasses implement :meth:`phi`."""

    dim: int = 3
    lipschitz: float = 1.0

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def phi(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inside(self, X: np.ndarray) -> np.ndarray:
        return self.phi(X) < 0

    # CSG on level sets (min / max keep a valid distance bound, so the Lipschitz bound is the larger of the two)
    def __or__(self, other: Geometry) -> SDFGeometry:
        return union(self, other)

    def __and__(self, other: Geometry) -> SDFGeometry:
        return intersection(self, other)

    def __sub__(self, other: Geometry) -> SDFGeometry:
        return difference(self, other)


@dataclass
class SDFGeometry(Geometry):
    """Any vectorised level set ``fn(X) -> (n,)`` with its bounding box ``(lo, hi)``."""

    fn: Callable[[np.ndarray], np.ndarray]
    box: tuple[Sequence[float], Sequence[float]]
    lipschitz: float = 1.0
    name: str = "sdf"

    def __post_init__(self):
        self.dim = len(self.box[0])

    @property
    def bounds(self):
        return np.asarray(self.box[0], float), np.asarray(self.box[1], float)

    def phi(self, X):
        return np.asarray(self.fn(np.asarray(X, float)), float).reshape(-1)


def _box_union(a: Geometry, b: Geometry):
    return np.minimum(a.bounds[0], b.bounds[0]), np.maximum(a.bounds[1], b.bounds[1])


def union(a: Geometry, b: Geometry) -> SDFGeometry:
    return SDFGeometry(lambda X: np.minimum(a.phi(X), b.phi(X)), _box_union(a, b), max(a.lipschitz, b.lipschitz),
                       "union")


def intersection(a: Geometry, b: Geometry) -> SDFGeometry:
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    return SDFGeometry(lambda X: np.maximum(a.phi(X), b.phi(X)), (lo, hi), max(a.lipschitz, b.lipschitz),
                       "intersection")


def difference(a: Geometry, b: Geometry) -> SDFGeometry:
    return SDFGeometry(lambda X: np.maximum(a.phi(X), -b.phi(X)), a.bounds, max(a.lipschitz, b.lipschitz),
                       "difference")


# ---------------------------------------------------------------- exact signed distances of primitives

def ball(center: Sequence[float], radius: float) -> SDFGeometry:
    """Disk (2-D) or sphere (3-D)."""
    c = np.asarray(center, float)
    return SDFGeometry(lambda X: np.linalg.norm(X - c, axis=1) - radius, (c - radius, c + radius), name="ball")


def box(lo: Sequence[float], hi: Sequence[float]) -> SDFGeometry:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, e = (lo + hi) / 2, (hi - lo) / 2

    def fn(X):
        q = np.abs(X - c) - e
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    return SDFGeometry(fn, (lo, hi), name="box")


def cylinder(center: Sequence[float], radius: float, length: float, axis: int = 2) -> SDFGeometry:
    """Capped circular cylinder (3-D) along a coordinate axis, centred at ``center``."""
    c = np.asarray(center, float)
    others = [k for k in range(3) if k != axis]

    def fn(X):
        r = np.linalg.norm(X[:, others] - c[others], axis=1) - radius
        z = np.abs(X[:, axis] - c[axis]) - length / 2
        q = np.stack([r, z], axis=1)
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    half = np.full(3, radius)
    half[axis] = length / 2
    return SDFGeometry(fn, (c - half, c + half), name="cylinder")


def torus(center: Sequence[float], R: float, r: float) -> SDFGeometry:
    """Torus around the z axis: tube radius ``r`` at distance ``R`` from the axis."""
    c = np.asarray(center, float)

    def fn(X):
        Y = X - c
        q = np.stack([np.hypot(Y[:, 0], Y[:, 1]) - R, Y[:, 2]], axis=1)
        return np.linalg.norm(q, axis=1) - r
    half = np.array([R + r, R + r, r])
    return SDFGeometry(fn, (c - half, c + half), name="torus")


def annulus(center: Sequence[float], r_in: float, r_out: float) -> SDFGeometry:
    """Ring (2-D) or hollow sphere (3-D): exact signed distance ``|rho - r_m| - w``."""
    c = np.asarray(center, float)
    rm, w = (r_in + r_out) / 2, (r_out - r_in) / 2
    return SDFGeometry(lambda X: np.abs(np.linalg.norm(X - c, axis=1) - rm) - w, (c - r_out, c + r_out),
                       name="annulus")


# ---------------------------------------------------------------- voxels

@dataclass
class VoxelGeometry(Geometry):
    """A voxel mask (True = material) with the position of the corner of voxel 0 and the voxel size.

    The signed distance is the Euclidean distance transform of the material and of the void, measured between voxel
    centres and shifted by half a voxel so the zero level runs through the voxel faces; between centres it is
    interpolated (multi)linearly. The surface is therefore the staircase of the voxels smoothed over one voxel.
    """

    mask: np.ndarray
    origin: Sequence[float]
    spacing: float | Sequence[float]
    pad: int = 2

    def __post_init__(self):
        from scipy import ndimage
        m = np.asarray(self.mask, bool)
        self.dim = m.ndim
        sp = np.broadcast_to(np.asarray(self.spacing, float), (self.dim,)).copy()
        m = np.pad(m, self.pad, constant_values=False)
        d_out = ndimage.distance_transform_edt(~m, sampling=sp)        # outside: distance to the nearest material
        d_in = ndimage.distance_transform_edt(m, sampling=sp)          # inside: distance to the nearest void
        half = 0.5 * sp.min()
        self._phi = np.where(m, -(d_in - half), d_out - half)
        self._sp = sp
        self._origin = np.asarray(self.origin, float) - self.pad * sp   # corner of padded voxel 0
        self.lipschitz = 1.0

    @property
    def bounds(self):
        lo = np.asarray(self.origin, float)
        return lo, lo + np.asarray(self.mask.shape) * self._sp

    def phi(self, X):
        from scipy import ndimage
        idx = (np.asarray(X, float) - self._origin) / self._sp - 0.5      # voxel-centre index coordinates
        return ndimage.map_coordinates(self._phi, idx.T, order=1, mode="nearest")

    @classmethod
    def from_geometry(cls, geom: Geometry, spacing: float, margin: int = 1) -> VoxelGeometry:
        """Rasterise another geometry (voxel centres inside it): a test and demo source of voxel data."""
        lo, hi = geom.bounds
        lo = lo - margin * spacing
        n = np.ceil((hi + margin * spacing - lo) / spacing).astype(int)
        axes = [lo[k] + (np.arange(n[k]) + 0.5) * spacing for k in range(len(n))]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(n))
        return cls(geom.phi(P).reshape(tuple(n)) < 0, lo, spacing)


# ---------------------------------------------------------------- triangle surfaces (STL)

def read_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) and triangles (m, 3) of an ASCII or binary STL (coincident vertices merged)."""
    data = open(path, "rb").read()
    tris = None
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            tris = rec["v"].astype(float)
    if tris is None:
        words = data.decode("utf-8", "replace").split()
        vals = [float(words[i + 1 + k]) for i, w in enumerate(words) if w == "vertex" for k in range(3)]
        tris = np.asarray(vals, float).reshape(-1, 3, 3)
    pts = tris.reshape(-1, 3)
    scale = max(np.ptp(pts, axis=0).max(), 1e-300)
    key = np.round(pts / (scale * 1e-9)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return pts[first], inv.reshape(-1, 3)


def write_stl(path: str, V: np.ndarray, F: np.ndarray, header: str = "pinneapple immersed") -> str:
    """Binary STL of a triangle surface."""
    T = np.asarray(V, float)[np.asarray(F)]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    rec = np.zeros(len(T), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, T
    with open(path, "wb") as f:
        f.write(header.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(T)))
        f.write(rec.tobytes())
    return path


def icosphere(radius: float = 1.0, subdivisions: int = 3, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated sphere (subdivided icosahedron, outward-oriented): vertices and triangles, e.g. for an STL."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(subdivisions):
        edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        ue, inv = np.unique(edges, axis=0, return_inverse=True)
        mid = V[ue].mean(1)
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = len(V) + inv.ravel().reshape(3, -1).T                       # midpoints of edges 01, 12, 20
        V = np.vstack([V, mid])
        a, b, c = F.T
        F = np.concatenate([np.stack([a, m[:, 0], m[:, 2]], 1), np.stack([b, m[:, 1], m[:, 0]], 1),
                            np.stack([c, m[:, 2], m[:, 1]], 1), m])
    return np.asarray(center, float) + radius * V, F


def _solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed solid angle of triangles (a, b, c) seen from the origin (Van Oosterom & Strackee 1983)."""
    la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
    num = np.einsum("...i,...i", a, np.cross(b, c))
    den = la * lb * lc + np.einsum("...i,...i", a, b) * lc + np.einsum("...i,...i", a, c) * lb \
        + np.einsum("...i,...i", b, c) * la
    return 2.0 * np.arctan2(num, den)


def _closest_on_triangles(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest points of triangles (A, B, C) to points P, all (n, 3) (Ericson, Real-Time Collision Detection,
    5.1.5), vectorised over the Voronoi regions of the triangle. Returns the squared distances, the closest points
    and the feature they lie on (0 face, 1 edge AB, 2 edge AC, 3 edge BC, 4 vertex A, 5 vertex B, 6 vertex C)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(va + vb + vc == 0, 1.0, va + vb + vc)
    v, w = vb / den, vc / den
    Q = A + ab * v[:, None] + ac * w[:, None]                           # interior projection
    feat = np.zeros(len(P), np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.clip(np.where(d1 - d3 != 0, d1 / (d1 - d3), 0), 0, 1)
        t_ac = np.clip(np.where(d2 - d6 != 0, d2 / (d2 - d6), 0), 0, 1)
        t_bc = np.clip(np.where((d4 - d3) + (d5 - d6) != 0, (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0), 0, 1)
    for code, m, q in ((3, (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), B + (C - B) * t_bc[:, None]),
                       (2, (vb <= 0) & (d2 >= 0) & (d6 <= 0), A + ac * t_ac[:, None]),
                       (1, (vc <= 0) & (d1 >= 0) & (d3 <= 0), A + ab * t_ab[:, None]),
                       (6, (d6 >= 0) & (d5 <= d6), C), (5, (d3 >= 0) & (d4 <= d3), B), (4, (d1 <= 0) & (d2 <= 0), A)):
        Q = np.where(m[:, None], q, Q)
        feat[m] = code
    return np.einsum("ij,ij->i", P - Q, P - Q), Q, feat


@dataclass
class STLGeometry(Geometry):
    """A closed (or nearly closed) triangle surface. Inside = generalised winding number above 1/2.

    ``sample_spacing`` controls the surface samples that index the triangles for the distance query: the distance is
    exact within ``near`` of the surface (where the cut cells are) and a lower bound farther away, which is all the
    cut-cell classification needs. ``sign``: ``"winding"`` evaluates the winding number at every point;
    ``"auto"`` (default) does so far from the surface and for any surface that is not watertight and consistently
    oriented, and inside the near band of a watertight one takes the sign of the angle-weighted pseudonormal of the
    closest feature (Baerentzen & Aanaes, IEEE TVCG 2005), which is exact there and much cheaper.
    """

    V: np.ndarray
    F: np.ndarray
    sample_spacing: float | None = None
    leaf_size: int = 16
    beta: float = 2.0
    near: float | None = None
    sign: str = "auto"
    _stats: dict = field(default_factory=dict, repr=False)
    # the signed distance to a planar facet is linear, so the clipped surface interpolates it without root
    # refinement and a shallower octree only rounds the facet edges (heat sink at h = 1 mm, no refinement:
    # 8 s of quadrature at depth 1, 43 s at depth 2; each level-set value is a nearest-triangle query)
    default_depth = 2
    default_refine = 0

    def __post_init__(self):
        from scipy.spatial import cKDTree
        self.V = np.asarray(self.V, float)
        self.F = np.asarray(self.F, np.int64)
        self.dim, self.lipschitz = 3, 1.0
        T = self.V[self.F]
        self._T = T
        diag = float(np.linalg.norm(np.ptp(self.V, axis=0)))
        s = self.sample_spacing or diag / 200
        self.sample_spacing = s
        self.near = self.near or 8 * s
        # surface samples on a barycentric lattice per triangle (spacing <= s): KD-tree for nearest candidates
        L = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
        m = np.maximum(1, np.ceil(L / s)).astype(int)
        pts, ids = [], []
        for mm in np.unique(m):
            sel = np.nonzero(m == mm)[0]
            ij = np.array([(i, j) for i in range(mm + 1) for j in range(mm + 1 - i)], float) / mm
            lam = np.column_stack([1 - ij.sum(1), ij])
            pts.append(np.einsum("kj,tjd->tkd", lam, T[sel]).reshape(-1, 3))
            ids.append(np.repeat(sel, len(lam)))
        self._samples = np.concatenate(pts)
        self._sample_tri = np.concatenate(ids)
        self._kdt = cKDTree(self._samples)
        self._rmax = float(np.max(L / m))
        self._build_tree()
        self._pseudonormals()
        self._stats.update(triangles=len(self.F), samples=len(self._samples), watertight=self.watertight)

    @classmethod
    def from_file(cls, path: str, **kw) -> STLGeometry:
        V, F = read_stl(path)
        return cls(V, F, **kw)

    @property
    def bounds(self):
        return self.V.min(axis=0), self.V.max(axis=0)

    # ---------------------------------------------------------- winding number (dipole tree)
    def _build_tree(self):
        T = self._T
        cen = T.mean(axis=1)
        an = 0.5 * np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])        # area-weighted normals
        area = np.linalg.norm(an, axis=1)
        order = np.arange(len(T))
        nodes = []                                                       # (start, end, left, right)
        stack = [(0, len(T), -1, 0)]
        perm = order.copy()
        rec = []
        while stack:
            s, e, parent, side = stack.pop()
            k = len(rec)
            rec.append([s, e, -1, -1])
            if parent >= 0:
                rec[parent][2 + side] = k
            if e - s > self.leaf_size:
                idx = perm[s:e]
                c = cen[idx]
                ax = int(np.argmax(np.ptp(c, axis=0)))
                o = np.argsort(c[:, ax], kind="stable")
                perm[s:e] = idx[o]
                mid = (s + e) // 2
                stack.append((mid, e, k, 1))
                stack.append((s, mid, k, 0))
        rec = np.asarray(rec, np.int64)
        self._perm = perm
        nn = len(rec)
        self._ns, self._ne, self._left, self._right = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
        self._ctr = np.zeros((nn, 3))
        self._rad = np.zeros(nn)
        self._dip = np.zeros((nn, 3))
        for k in range(nn):
            idx = perm[self._ns[k]:self._ne[k]]
            a = area[idx]
            c = (a[:, None] * cen[idx]).sum(0) / max(a.sum(), 1e-300) if a.sum() > 0 else cen[idx].mean(0)
            self._ctr[k] = c
            self._rad[k] = np.linalg.norm(T[idx].reshape(-1, 3) - c, axis=1).max()
            self._dip[k] = an[idx].sum(0)
        nodes.clear()
        self._stats["tree_nodes"] = nn

    def winding_number(self, X: np.ndarray, chunk: int = 20000) -> np.ndarray:
        X = np.asarray(X, float)
        out = np.zeros(len(X))
        Tp = self._T[self._perm]
        for s in range(0, len(X), chunk):
            Q = X[s:s + chunk]
            w = np.zeros(len(Q))
            qi = np.arange(len(Q))
            ni = np.zeros(len(Q), np.int64)
            while len(qi):
                dv = self._ctr[ni] - Q[qi]
                dist = np.linalg.norm(dv, axis=1)
                far = dist > self.beta * self._rad[ni]
                if far.any():
                    f = np.einsum("ij,ij->i", dv[far], self._dip[ni[far]]) / dist[far] ** 3
                    np.add.at(w, qi[far], f)
                qn, nn = qi[~far], ni[~far]
                leaf = self._left[nn] < 0
                if leaf.any():
                    ql, nl = qn[leaf], nn[leaf]
                    cnt = self._ne[nl] - self._ns[nl]
                    qq = np.repeat(ql, cnt)
                    tt = np.repeat(self._ns[nl] - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())
                    tri = Tp[tt]
                    P = Q[qq]
                    om = _solid_angle(tri[:, 0] - P, tri[:, 1] - P, tri[:, 2] - P)
                    np.add.at(w, qq, om)
                inner = ~leaf
                qi = np.concatenate([qn[inner], qn[inner]])
                ni = np.concatenate([self._left[nn[inner]], self._right[nn[inner]]])
            out[s:s + chunk] = w / (4 * np.pi)
        return out

    def _pseudonormals(self):
        T, F = self._T, self.F
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nf = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        loc = [(0, 1), (0, 2), (1, 2)]                                   # local edges AB, AC, BC
        und = np.sort(np.stack([F[:, list(e)] for e in loc], 1), axis=2).reshape(-1, 2)
        ue, inv = np.unique(und, axis=0, return_inverse=True)
        en = np.zeros((len(ue), 3))
        np.add.at(en, inv.ravel(), np.repeat(nf, 3, axis=0))
        self._edge_normal = en[inv.ravel()].reshape(len(F), 3, 3)
        vn = np.zeros((len(self.V), 3))
        for i in range(3):
            a = T[:, (i + 1) % 3] - T[:, i]
            b = T[:, (i + 2) % 3] - T[:, i]
            cosang = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1),
                                                             1e-300)
            np.add.at(vn, F[:, i], np.arccos(np.clip(cosang, -1, 1))[:, None] * nf)
        self._vertex_normal = vn
        self._face_normal = nf
        directed = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)
        key = directed[:, 0] * len(self.V) + directed[:, 1]
        rkey = directed[:, 1] * len(self.V) + directed[:, 0]
        uniq = len(np.unique(key)) == len(key)
        self.watertight = bool(uniq and np.all(np.isin(rkey, key)))

    # ---------------------------------------------------------- distance
    def closest(self, X: np.ndarray, k: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unsigned distance (exact within ``near``, a lower bound beyond), and for the near points the closest
        triangle, point and feature (-1 where not computed)."""
        X = np.asarray(X, float)
        d1, _ = self._kdt.query(X, k=1)
        dist = np.maximum(d1 - self._rmax, 0.0)                         # a lower bound, exact enough far away
        tri_c = np.full(len(X), -1, np.int64)
        Qc = np.zeros((len(X), 3))
        feat = np.full(len(X), -1, np.int8)
        near = np.nonzero(d1 < self.near)[0]
        if len(near):
            kk = min(k, len(self._samples))
            _, idx = self._kdt.query(X[near], k=kk)
            tri = self._sample_tri[np.atleast_2d(idx).reshape(len(near), kk)]
            pair = np.unique(np.arange(len(near))[:, None] * len(self.F) + tri)      # distinct (point, triangle)
            qi, ti = pair // len(self.F), pair % len(self.F)
            T = self._T[ti]
            d2, Q, f = _closest_on_triangles(X[near][qi], T[:, 0], T[:, 1], T[:, 2])
            o = np.lexsort((d2, qi))                                    # per point, the nearest first
            first = o[np.r_[True, qi[o][1:] != qi[o][:-1]]]
            dist[near] = np.sqrt(d2[first])
            tri_c[near], Qc[near], feat[near] = ti[first], Q[first], f[first]
        return dist, tri_c, Qc, feat

    def unsigned_distance(self, X: np.ndarray) -> np.ndarray:
        return self.closest(X)[0]

    def _pseudo_sign(self, X, tri, Q, feat) -> np.ndarray:
        F = self.F[tri]
        N = self._face_normal[tri].copy()
        for code, e in ((1, 0), (2, 1), (3, 2)):
            m = feat == code
            N[m] = self._edge_normal[tri[m], e]
        for code, v in ((4, 0), (5, 1), (6, 2)):
            m = feat == code
            N[m] = self._vertex_normal[F[m, v]]
        return np.sign(np.einsum("ij,ij->i", X - Q, N))

    def phi(self, X):
        X = np.asarray(X, float)
        d, tri, Q, feat = self.closest(X)
        inside = np.zeros(len(X), bool)
        fast = (tri >= 0) & (d > 0) if (self.sign == "auto" and self.watertight) else np.zeros(len(X), bool)
        if fast.any():
            s = self._pseudo_sign(X[fast], tri[fast], Q[fast], feat[fast])
            inside[np.nonzero(fast)[0][s < 0]] = True
            fast[np.nonzero(fast)[0][s == 0]] = False                    # degenerate: ask the winding number
        rest = np.nonzero(~fast)[0]
        if len(rest):
            inside[rest] = self.winding_number(X[rest]) > 0.5
        return np.where(inside, -d, d)
