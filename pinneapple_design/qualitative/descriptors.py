"""Geometric descriptors of closed triangulated bodies and assemblies of named parts.

Everything the qualitative models need comes from the surface mesh: per-face areas, normals and centroids, the
volume and second moments (exact polyhedral integrals over signed tetrahedra), extents, projected (frontal) area,
sharp edges, and section properties along an axis from a voxelization. An ``Assembly`` keeps the parts separate, so
a change to one part (scaled, moved, removed, replaced) can be traced to its physical effect.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Assembly", "Descriptors", "as_mesh", "describe", "extrude", "section_profile"]


def as_mesh(geom: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` from a tuple, a trimesh-like object (``.vertices``/``.faces``) or an ``Assembly``."""
    if isinstance(geom, Assembly):
        return geom.mesh()
    if hasattr(geom, "vertices") and hasattr(geom, "faces"):
        return np.asarray(geom.vertices, float), np.asarray(geom.faces, np.int64)
    V, F = geom
    return np.asarray(V, float), np.asarray(F, np.int64)


def extrude(polygon: np.ndarray, depth: float = 1.0, axis: str = "y") -> tuple[np.ndarray, np.ndarray]:
    """Closed prism from a counter-clockwise 2-D polygon (n, 2) in the (x, z) plane, extruded along ``axis`` by
    ``depth`` (centred), so 2-D sections can be compared with the same models."""
    P = np.asarray(polygon, float)
    n = len(P)
    a = np.c_[P[:, 0], np.full(n, -depth / 2), P[:, 1]]
    b = np.c_[P[:, 0], np.full(n, depth / 2), P[:, 1]]
    V = np.vstack([a, b, a.mean(0), b.mean(0)])
    ca, cb = 2 * n, 2 * n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [ca, j, i], [cb, n + i, n + j]]
    V, F = np.array(V), np.array(F, np.int64)
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1]
    if axis == "z":
        V = V[:, [0, 2, 1]]
        F = F[:, ::-1]
    return V, F


def _signed_volume(V, F) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


@dataclass
class Descriptors:
    """Geometric descriptors of one body (SI units). ``direction`` is the flow / load reference direction."""
    volume: float
    wetted_area: float
    centroid: np.ndarray
    extents: np.ndarray                    # bounding box size (x, y, z)
    length: float                          # extent along ``direction``
    frontal_area: float                    # area projected on the plane normal to ``direction``
    equivalent_diameter: float             # sqrt(4 frontal_area / pi)
    slenderness: float                     # length / equivalent_diameter
    sphericity: float                      # pi^(1/3) (6 V)^(2/3) / S (1 for a sphere)
    characteristic_length: float           # V / S
    forward_flat_area: float               # faces facing the flow at < 30 deg from head-on
    rear_flat_area: float                  # faces facing downstream at < 30 deg (blunt base)
    sharp_edge_length: float               # edges whose faces meet at more than 60 deg
    second_moments: np.ndarray             # integral of (x - c)(x - c)^T dV
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def as_dict(self) -> dict[str, float]:
        out = {k: float(v) for k, v in self.__dict__.items() if np.isscalar(v)}
        out.update({f"extent_{a}": float(self.extents[i]) for i, a in enumerate("xyz")})
        return out


LABELS = {  # descriptor -> (pt, en)
    "volume": ("volume", "volume"), "wetted_area": ("área molhada", "wetted area"),
    "length": ("comprimento", "length"), "frontal_area": ("área frontal", "frontal area"),
    "slenderness": ("esbeltez L/D", "slenderness L/D"), "sphericity": ("esfericidade", "sphericity"),
    "characteristic_length": ("espessura característica V/S", "characteristic thickness V/S"),
    "forward_flat_area": ("área plana frontal", "forward flat area"),
    "rear_flat_area": ("base plana traseira", "blunt rear base"),
    "sharp_edge_length": ("arestas vivas", "sharp edges"),
}


def face_geometry(V: np.ndarray, F: np.ndarray):
    """Per-face area, unit normal and centroid."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(area2, 1e-300)[:, None]
    return 0.5 * area2, n, (a + b + c) / 3


def _mass_properties(V, F):
    """Volume, centroid and second moments about the centroid (Eberly's polyhedral integrals)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = det.sum() / 6
    first = (det[:, None] * (a + b + c)).sum(0) / 24
    cen = first / vol
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (det * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                              + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j] + a[:, j] * c[:, i]
                              + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum() / 120
    return float(vol), cen, S - vol * np.outer(cen, cen)


def _sharp_edges(V, F, normals, angle_deg=60.0) -> float:
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    i = np.nonzero(same)[0]
    cosang = np.einsum("ij,ij->i", normals[owner[i]], normals[owner[i + 1]])
    sharp = cosang < np.cos(np.radians(angle_deg))
    return float(np.linalg.norm(V[e[i[sharp], 0]] - V[e[i[sharp], 1]], axis=1).sum())


def describe(geom: Any, direction=(1.0, 0.0, 0.0)) -> Descriptors:
    """Descriptors of a closed, outward-oriented triangulated body. The projected area is
    ``0.5 * sum |n . d| A`` (exact for convex bodies, an upper bound otherwise)."""
    V, F = as_mesh(geom)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    A, n, _ = face_geometry(V, F)
    vol, cen, S2 = _mass_properties(V, F)
    vol = abs(vol)
    S = float(A.sum())
    nd = n @ d
    frontal = float(0.5 * np.sum(np.abs(nd) * A))
    proj = V @ d
    length = float(proj.max() - proj.min())
    deq = float(np.sqrt(4 * frontal / np.pi))
    c30 = np.cos(np.radians(30))
    return Descriptors(
        volume=vol, wetted_area=S, centroid=cen, extents=V.max(0) - V.min(0), length=length, frontal_area=frontal,
        equivalent_diameter=deq, slenderness=length / max(deq, 1e-300),
        sphericity=float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / S), characteristic_length=vol / S,
        forward_flat_area=float(A[nd < -c30].sum()), rear_flat_area=float(A[nd > c30].sum()),
        sharp_edge_length=_sharp_edges(V, F, n), second_moments=S2, direction=d)


def section_profile(geom: Any, axis: int = 0, n_slices: int = 40, n_grid: int | tuple[int, int] = 48):
    """Section properties along ``axis`` from a voxelization by ray parity: slice positions ``x`` (centres), area
    ``A(x)``, centroid of each section in the two other coordinates, second moments ``I(x)`` about the section's
    centroidal axes (``I_a``: about the first remaining axis, i.e. bending in the second; ``I_b`` the other way) and
    the extreme fibre distances ``c_a``, ``c_b``. Also returns the voxel occupancy ``(n_slices, n_a, n_b)`` and the
    cell sizes, e.g. for a voxel finite-element model. ``n_grid``: cells across the section (one number or a pair)."""
    V, F = as_mesh(geom)
    oth = [k for k in range(3) if k != axis]
    lo, hi = V.min(0), V.max(0)
    xs = np.linspace(lo[axis], hi[axis], n_slices + 1)
    xc = 0.5 * (xs[1:] + xs[:-1])
    na, nb = (n_grid, n_grid) if np.isscalar(n_grid) else (int(n_grid[0]), int(n_grid[1]))
    ga = np.linspace(lo[oth[0]], hi[oth[0]], na + 1)
    gb = np.linspace(lo[oth[1]], hi[oth[1]], nb + 1)
    ac, bc = 0.5 * (ga[1:] + ga[:-1]), 0.5 * (gb[1:] + gb[:-1])
    da, db = ga[1] - ga[0], gb[1] - gb[0]
    tri = V[F]                                               # (nf, 3, 3)
    occ = np.zeros((n_slices, na, nb), bool)
    # cast rays along a (the second remaining axis) for every (x, b) pair: intersect with triangles in the (x, b)
    # projection, sort the hits along a, fill between pairs
    P = tri[:, :, [axis, oth[1]]]
    for ib, b in enumerate(bc):
        for ix, x in enumerate(xc):
            p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
            den = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
            ok = np.abs(den) > 1e-18
            w0 = ((p1[:, 1] - p2[:, 1]) * (x - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (b - p2[:, 1])) / np.where(ok, den, 1)
            w1 = ((p2[:, 1] - p0[:, 1]) * (x - p2[:, 0]) + (p0[:, 0] - p2[:, 0]) * (b - p2[:, 1])) / np.where(ok, den, 1)
            w2 = 1 - w0 - w1
            hit = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not hit.any():
                continue
            av = np.sort(w0[hit] * tri[hit, 0, oth[0]] + w1[hit] * tri[hit, 1, oth[0]] + w2[hit] * tri[hit, 2, oth[0]])
            av = av[np.r_[True, np.diff(av) > 1e-12]]          # drop duplicate hits on shared edges
            for k in range(0, len(av) - 1, 2):
                occ[ix, (ac >= av[k]) & (ac <= av[k + 1]), ib] = True
    dA = da * db
    A = occ.sum((1, 2)) * dA
    Aa, Bb = np.meshgrid(ac, bc, indexing="ij")
    with np.errstate(invalid="ignore", divide="ignore"):
        ca = np.array([np.sum(Aa[o]) * dA / max(a_, 1e-300) for o, a_ in zip(occ, A, strict=True)])
        cb = np.array([np.sum(Bb[o]) * dA / max(a_, 1e-300) for o, a_ in zip(occ, A, strict=True)])
    I_a = np.array([np.sum((Bb[o] - cb[k]) ** 2) * dA + (o.sum() * dA * db ** 2 / 12) for k, o in enumerate(occ)])
    I_b = np.array([np.sum((Aa[o] - ca[k]) ** 2) * dA + (o.sum() * dA * da ** 2 / 12) for k, o in enumerate(occ)])
    c_a = np.array([np.max(np.abs(Bb[o] - cb[k])) + db / 2 if o.any() else 0.0 for k, o in enumerate(occ)])
    c_b = np.array([np.max(np.abs(Aa[o] - ca[k])) + da / 2 if o.any() else 0.0 for k, o in enumerate(occ)])
    return {"x": xc, "dx": xs[1] - xs[0], "A": A, "centroid_a": ca, "centroid_b": cb, "I_a": I_a, "I_b": I_b,
            "c_a": c_a, "c_b": c_b, "axes": (axis, oth[0], oth[1]), "occupancy": occ,
            "cell": (float(xs[1] - xs[0]), float(da), float(db)), "origin": (float(lo[axis]), float(ga[0]), float(gb[0]))}


class Assembly:
    """Named parts of one geometry. Parts are treated as touching, not overlapping (areas add)."""

    def __init__(self, parts: Mapping[str, Any]):
        self.parts = {k: as_mesh(v) for k, v in parts.items()}

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        Vs, Fs, off = [], [], 0
        for V, F in self.parts.values():
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        return np.vstack(Vs), np.vstack(Fs)

    def face_owner(self) -> np.ndarray:
        return np.concatenate([np.full(len(F), i) for i, (_, F) in enumerate(self.parts.values())])

    @property
    def names(self) -> list[str]:
        return list(self.parts)

    def _copy(self, parts) -> Assembly:
        a = Assembly({})
        a.parts = parts
        return a

    def scaled(self, part: str, factors=(1.0, 1.0, 1.0), about: str | tuple = "centroid") -> Assembly:
        """Scale one part by ``factors`` (x, y, z) about its centroid, its minimum corner (``"min"``: grows away from
        where it is attached at the low side) or a given point."""
        V, F = self.parts[part]
        if about == "centroid":
            p = V.mean(0)
        elif about == "min":
            p = V.min(0)
        elif about == "max":
            p = V.max(0)
        else:
            p = np.asarray(about, float)
        parts = dict(self.parts)
        parts[part] = (p + (V - p) * np.asarray(factors, float), F)
        return self._copy(parts)

    def moved(self, part: str, offset) -> Assembly:
        V, F = self.parts[part]
        parts = dict(self.parts)
        parts[part] = (V + np.asarray(offset, float), F)
        return self._copy(parts)

    def without(self, part: str) -> Assembly:
        return self._copy({k: v for k, v in self.parts.items() if k != part})

    def replaced(self, part: str, geom: Any) -> Assembly:
        parts = dict(self.parts)
        parts[part] = as_mesh(geom)
        return self._copy(parts)
