"""The regular grid the operator lives on, and the two maps between it and the finite-element mesh.

A neural operator needs every sample on the same lattice; a finite-element result lives on an unstructured tetrahedral
mesh that changes with every design. This module is the bridge, and it is the reason the whole pipeline can share one
discretisation from the solver to the dashboard:

``occupancy(nodes, elements, grid)``
    which grid cells are inside the part, by an exact barycentric test in the tetrahedra (a KD-tree over tet centroids
    pre-selects the candidates). No convex-hull shortcut: the part has a flare, and a hull test would fill it.
``rasterize(nodes, values, grid, inside)``
    nodal fields to the grid, inverse-distance weighted over the ``k`` nearest nodes, zeroed outside the part.
``sample_field(field, grid, coords)``
    the inverse map: a grid field to arbitrary coordinates, trilinearly. This is what lets a predicted grid answer a
    CalculiX deck's nodes (``virtual_ccx``) and a dashboard's probe points.

The grid is a fixed window on the world -- the same box for every design -- so a design that is shorter simply leaves
its top rows empty, which is what the occupancy channel is for. Coordinates are kept in metres; the operator sees the
box as a unit cube (the caller normalises), so cell aspect ratio is a sampling choice and not something the network
has to learn.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["GridSpec", "occupancy", "rasterize", "fill_outside", "signed_distance", "sample_field", "surface_faces"]


@dataclass
class GridSpec:
    """A box and a cell count: cell centres are ``origin + (i + 1/2) * spacing``."""
    origin: Sequence[float]           # (3,) lower corner, m
    size: Sequence[float]             # (3,) box edge lengths, m
    n: Sequence[int]                  # (3,) cells per axis

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.size = np.asarray(self.size, float)
        self.n = tuple(int(v) for v in self.n)
        if np.any(self.size <= 0) or min(self.n) < 2:
            raise ValueError(f"bad grid: size {self.size.tolist()}, n {self.n}")

    @property
    def spacing(self) -> np.ndarray:
        return self.size / np.asarray(self.n, float)

    def centers(self) -> np.ndarray:
        """(prod(n), 3) cell centres, in C order."""
        ax = [self.origin[j] + (np.arange(self.n[j]) + 0.5) * self.spacing[j] for j in range(3)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
        return g.reshape(-1, 3)

    def axes(self) -> list[np.ndarray]:
        return [self.origin[j] + (np.arange(self.n[j]) + 0.5) * self.spacing[j] for j in range(3)]

    def to_unit(self, points: np.ndarray) -> np.ndarray:
        """World coordinates to [0, 1]^3 over the box (the operator's domain)."""
        return (np.asarray(points, float) - self.origin) / self.size

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float)
        return np.all((p >= self.origin) & (p <= self.origin + self.size), axis=-1)

    def as_dict(self) -> dict[str, Any]:
        return {"origin": self.origin.tolist(), "size": self.size.tolist(), "n": list(self.n)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridSpec:
        return cls(d["origin"], d["size"], d["n"])

    @classmethod
    def around(cls, lo: Sequence[float], hi: Sequence[float], n: Sequence[int],
               margin: float = 0.0) -> GridSpec:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        pad = margin * float(np.max(hi - lo)) if margin < 1 else margin
        return cls(lo - pad, (hi + pad) - (lo - pad), n)


def _tet_corners(elements: np.ndarray, element_type: str) -> np.ndarray:
    """(m, 4) corner indices of each element, whatever the family (higher-order nodes are dropped)."""
    e = np.asarray(elements, int)
    t = str(element_type).upper()
    if t.startswith(("C3D4", "DC3D4")) or e.shape[1] == 4:
        return e[:, :4]
    if t.startswith(("C3D10", "DC3D10")) or e.shape[1] == 10:
        return e[:, :4]
    if e.shape[1] == 8:                                     # hexahedron -> six tetrahedra
        h = e[:, :8]
        split = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 5), (3, 4, 5, 7), (1, 3, 5, 6), (3, 5, 6, 7)]
        return np.concatenate([h[:, list(s)] for s in split], axis=0)
    raise ValueError(f"cannot turn {element_type!r} with {e.shape[1]} nodes into tetrahedra")


def _barycentric_inside(P: np.ndarray, T: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """(B, 4) points vs (B, 4, 3) tetrahedra -> whether each point is inside its tetrahedron.

    Cramer's rule on the three edge vectors, with cross products, instead of a batched 3x3 solve: the solve costs
    ~10x more and this test runs over every grid cell times every candidate tetrahedron.
    """
    a, b, c, d = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    v0, v1, v2 = b - a, c - a, d - a
    r = P - a
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v0)
    n3 = np.cross(v0, v1)
    det = np.einsum("ij,ij->i", v0, n1)
    ok = np.abs(det) > 1e-30
    det = np.where(ok, det, 1.0)
    l1 = np.einsum("ij,ij->i", r, n1) / det
    l2 = np.einsum("ij,ij->i", r, n2) / det
    l3 = np.einsum("ij,ij->i", r, n3) / det
    l0 = 1.0 - l1 - l2 - l3
    return ok & (l0 >= -tol) & (l1 >= -tol) & (l2 >= -tol) & (l3 >= -tol)


def occupancy(nodes: np.ndarray, elements: np.ndarray, grid: GridSpec, element_type: str = "C3D4",
              k: int = 12, workers: int = 1) -> np.ndarray:
    """(nx, ny, nz) boolean: the grid cells whose centre lies inside the part."""
    X = np.asarray(nodes, float)
    tets = _tet_corners(elements, element_type)
    T = X[tets]                                             # (m, 4, 3)
    cent = T.mean(1)
    tree = cKDTree(cent)
    P = grid.centers()
    k = min(k, len(tets))
    inside = np.zeros(len(P), bool)
    # a cell outside the mesh's bounding box is outside the mesh: skip the test for most of the empty box
    lo, hi = X.min(0), X.max(0)
    cand = np.all((P >= lo) & (P <= hi), axis=1)
    if not cand.any():
        return inside.reshape(grid.n)
    Q = P[cand]
    _, idx = tree.query(Q, k=k, workers=workers)
    idx = np.atleast_2d(idx)
    hit_all = np.zeros(len(Q), bool)
    chunk = max(1, 2_000_000 // max(1, k * 4))
    for s in range(0, len(Q), chunk):
        sl = slice(s, s + chunk)
        cand_t = idx[sl]                                     # (b, k) tet ids
        rep = np.repeat(Q[sl], k, axis=0)                    # (b*k, 3)
        TT = T[cand_t.reshape(-1)]                           # (b*k, 4, 3)
        hit_all[sl] = _barycentric_inside(rep, TT).reshape(-1, k).any(1)
    inside[cand] = hit_all
    return inside.reshape(grid.n)


def _barycentric_weights(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(B, 4) weights of each point in its tetrahedron (Cramer's rule, the same algebra as the inside test)."""
    a, b, c, d = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    v0, v1, v2 = b - a, c - a, d - a
    r = P - a
    n1, n2, n3 = np.cross(v1, v2), np.cross(v2, v0), np.cross(v0, v1)
    det = np.einsum("ij,ij->i", v0, n1)
    det = np.where(np.abs(det) > 1e-30, det, 1.0)
    l1 = np.einsum("ij,ij->i", r, n1) / det
    l2 = np.einsum("ij,ij->i", r, n2) / det
    l3 = np.einsum("ij,ij->i", r, n3) / det
    return np.stack([1.0 - l1 - l2 - l3, l1, l2, l3], axis=1)


def rasterize_barycentric(nodes: np.ndarray, values: np.ndarray, grid: GridSpec, elements: np.ndarray,
                          element_type: str = "C3D4", workers: int = 1, k: int = 12,
                          inside: np.ndarray | None = None) -> np.ndarray:
    """Nodal ``values`` sampled onto the grid by exact interpolation inside the element holding each cell.

    The inverse-distance sampler (:func:`rasterize`) reproduces smooth fields but not a stress concentration: on the
    nominal design the largest node-level error of von Mises is 27 % of the peak through IDW, and it sits at the
    flare -- exactly the number the surrogate is judged on. Linear interpolation inside the tetrahedron is exact for
    a field that is linear over an element, so the peak survives the trip to the grid and back.

    Cells that no element contains stay at zero; pair this with :func:`fill_outside`.
    """
    X = np.asarray(nodes, float)
    V = np.asarray(values, float)
    scalar = V.ndim == 1
    if scalar:
        V = V[:, None]
    tets = _tet_corners(elements, element_type)
    T = X[tets]
    tree = cKDTree(T.mean(1))
    P = grid.centers()
    out = np.zeros((len(P),) + V.shape[1:])
    found = np.zeros(len(P), bool)
    lo, hi = X.min(0), X.max(0)
    box_idx = np.flatnonzero(np.all((P >= lo) & (P <= hi), axis=1))
    if len(box_idx):
        Q = P[box_idx]
        kk = min(int(k), len(tets))
        _, idx = tree.query(Q, k=kk, workers=workers)
        idx = np.atleast_2d(idx).reshape(len(Q), kk)
        chunk = max(1, 1_500_000 // max(1, kk))
        for s0 in range(0, len(Q), chunk):
            sl = slice(s0, s0 + chunk)
            cand_t = idx[sl]                                     # (b, kk)
            rep = np.repeat(Q[sl], kk, axis=0)                   # (b*kk, 3)
            ok2 = _barycentric_inside(rep, T[cand_t.reshape(-1)], tol=1e-9).reshape(len(cand_t), kk)
            has = ok2.any(1)
            if not has.any():
                continue
            rows = np.arange(int(has.sum())) * kk
            first = ok2.argmax(1)[has]                           # first candidate that contains the cell
            elem = cand_t[has].reshape(-1)[rows + first]
            w = _barycentric_weights(Q[sl][has], T[elem])
            got = np.einsum("bk,bk...->b...", w, V[tets[elem]])
            target = box_idx[sl][has]
            out[target] = got
            found[target] = True
    if inside is not None:
        m = np.asarray(inside, bool).reshape(-1)
        out[~m] = 0.0
        found &= m
    if scalar:
        out = out[:, 0]
    return out.reshape(*grid.n, *out.shape[1:])


def rasterize(nodes: np.ndarray, values: np.ndarray, grid: GridSpec, inside: np.ndarray | None = None,
              k: int = 8, workers: int = 1, reach: float | None = None) -> np.ndarray:
    """Nodal ``values`` (n, ...) sampled onto the grid: inverse-distance weights over the ``k`` nearest nodes.

    ``inside`` masks the cells that are not the part. ``reach`` is off by default and should stay off: zeroing cells
    farther than ``reach`` from any node looks like a safe guard but is a source of error, because a mesh coarser
    than the grid has legitimate cells several cells away from the nearest node, and a zero there is read back as a
    zero field. Measured on a linear field on a coarse mesh, the cell-to-node round trip goes from 52 % of the range
    (wrong) with a reach cut to 4e-3 % with it off.
    """
    X = np.asarray(nodes, float)
    V = np.asarray(values, float)
    scalar = V.ndim == 1
    if scalar:
        V = V[:, None]
    P = grid.centers()
    k = min(k, len(X))
    d, idx = cKDTree(X).query(P, k=k, workers=workers)
    d = np.atleast_2d(d)
    idx = np.atleast_2d(idx)
    w = 1.0 / np.maximum(d, 1e-12)
    w /= w.sum(1, keepdims=True)
    out = np.einsum("pk,pk...->p...", w, V[idx])
    if reach is not None:
        out[(d[:, 0] > float(reach))] = 0.0
    if inside is not None:
        out[~np.asarray(inside, bool).reshape(-1)] = 0.0
    if scalar:
        out = out[:, 0]
    return out.reshape(*grid.n, *out.shape[1:])


def fill_outside(field: np.ndarray, inside: np.ndarray, grid: GridSpec, workers: int = 1) -> np.ndarray:
    """Give every cell outside the part the value of the nearest cell inside it.

    This is not cosmetic. The part's plate is 6 mm thick and the grid's cells are ~9 mm, so the plate barely fills
    one layer of cells and it is *outside* cells that touch it. Leaving zeros there and reading the field back with
    :func:`sample_field` pulls every node near the plate towards zero: measured on the nominal design, the
    node round-trip error is 21 % of the peak displacement with zeros and **0.4 %** with this fill (refining the
    grid changes neither number). The operator is trained on the filled field for the same reason -- it has one
    smooth field over the whole box instead of a cliff at the surface -- and ``occupancy`` still says which cells
    are the part.
    """
    F = np.asarray(field, float)
    lead = F.shape[:3]
    m = np.asarray(inside, bool).reshape(-1)
    if m.all() or not m.any():
        return F
    C = grid.centers()
    flat = F.reshape((len(m),) + F.shape[3:]).copy()
    _, j = cKDTree(C[m]).query(C[~m], workers=workers)
    flat[~m] = flat[m][j]
    return flat.reshape(lead + F.shape[3:])


def sample_field(field: np.ndarray, grid: GridSpec, coords: np.ndarray, *, edge: bool = True) -> np.ndarray:
    """Grid ``field`` (nx, ny, nz[, c]) evaluated at ``coords`` (..., 3): trilinear, zero far outside the box.

    ``edge`` extends the box by one cell and clamps to it. That is not cosmetic either: the grid's box begins at the
    part's clamped face, so the nodes *on* that face lie half a cell outside the first cell centre and a strict
    "zero outside" rule would report zero displacement there for any model -- a check that passes for free. Beyond
    one cell the value is zero, so a point genuinely off the part is still not silently given an interior value.
    """
    F = np.asarray(field, float)
    shape = np.asarray(coords).shape
    C = np.asarray(coords, float).reshape(-1, 3)
    n = np.asarray(grid.n)
    u = (C - grid.origin) / grid.spacing - 0.5                # cell-centre index space
    reach = 1.0 if edge else 0.0
    inside = np.all((u >= -reach) & (u <= n - 1.0 + reach), axis=1) if len(C) else np.zeros(0, bool)
    out = np.zeros((len(C),) + F.shape[3:])
    if np.any(inside):
        ui = np.clip(u[inside], 0.0, n - 1.0)
        i0 = np.floor(ui).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        f = ui - i0
        acc = np.zeros((int(inside.sum()),) + F.shape[3:])
        for corner in np.ndindex(2, 2, 2):
            w = np.ones(int(inside.sum()))
            ix = []
            for j, high in enumerate(corner):
                ix.append(i1[:, j] if high else i0[:, j])
                w = w * (f[:, j] if high else (1.0 - f[:, j]))
            acc += w.reshape((-1,) + (1,) * (F.ndim - 3)) * F[ix[0], ix[1], ix[2]]
        out[inside] = acc
    return out.reshape(shape[:-1] + F.shape[3:])


def signed_distance(inside: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Signed distance (m) of every cell to the part's surface: negative inside, positive outside.

    Exact on the lattice -- two Euclidean distance transforms of the occupancy mask -- so it costs nothing next to
    a triangle-by-triangle distance and is consistent with the mask the operator is scored against.
    """
    from scipy import ndimage
    m = np.asarray(inside, bool)
    sp = tuple(float(s) for s in grid.spacing)
    d_out = ndimage.distance_transform_edt(~m, sampling=sp)
    d_in = ndimage.distance_transform_edt(m, sampling=sp)
    return np.where(m, -d_in, d_out)


def surface_faces(nodes: np.ndarray, elements: np.ndarray, element_type: str = "C3D4") -> np.ndarray:
    """Corner-node boundary faces (t, 3) of a tetrahedral mesh, for drawing and for the SDF."""
    tets = _tet_corners(elements, element_type) if str(element_type).upper().startswith(("C3D10", "C3D4")) \
        else np.asarray(elements, int)[:, :4]
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]], axis=0)
    key = np.sort(faces, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inv] == 1]
