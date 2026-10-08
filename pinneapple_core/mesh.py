"""Mesh: simplex meshes (intervals, triangles, tetrahedra) with P1 calculus.

A :class:`Mesh` is ``points (N, d)`` plus ``cells (M, d+1)`` vertex indices.
Fields on a mesh are nodal (one value per point) and piecewise linear, so
``integrate`` is exact for P1 data and ``gradient`` is exact for linear data.

>>> from pinneapple_core import Mesh
>>> m = Mesh.structured([0, 0], [1, 1], (8, 8))
>>> m.n_cells
128
"""
from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree

_BARY_TOL = 1e-9


class Mesh:
    """Unstructured simplex mesh in R^d (d = 1, 2 or 3)."""

    def __init__(self, points: np.ndarray, cells: np.ndarray) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        cells = np.asarray(cells, dtype=np.int64)
        if pts.ndim != 2 or pts.shape[1] not in (1, 2, 3):
            raise ValueError("points must have shape (N, d) with d in {1, 2, 3}")
        d = pts.shape[1]
        if cells.ndim != 2 or cells.shape[1] != d + 1:
            raise ValueError(f"cells must have shape (M, {d + 1}) for a {d}D simplex mesh")
        if cells.size and (cells.min() < 0 or cells.max() >= pts.shape[0]):
            raise ValueError("cells reference vertices that do not exist")
        self.points = np.ascontiguousarray(pts)
        self.cells = np.ascontiguousarray(cells)
        self._cache: dict = {}
        if self.n_cells and np.any(self.cell_volumes <= 0):
            raise ValueError("mesh has degenerate (zero-volume) cells")

    # -- constructors -------------------------------------------------------
    @classmethod
    def interval(cls, a: float, b: float, n: int) -> "Mesh":
        """``n`` equal segments between ``a`` and ``b``."""
        return cls.structured([a], [b], (n,))

    @classmethod
    def structured(cls, lo: Sequence[float], hi: Sequence[float], shape: Sequence[int]) -> "Mesh":
        """Simplex mesh of a box: ``shape`` cells per axis, each cell cut into
        d! simplices along the same diagonal (conforming, Kuhn triangulation)."""
        lo = np.atleast_1d(np.asarray(lo, dtype=np.float64))
        hi = np.atleast_1d(np.asarray(hi, dtype=np.float64))
        shape = tuple(int(s) for s in shape)
        d = len(shape)
        if not (lo.shape == hi.shape == (d,)) or d not in (1, 2, 3) or min(shape) < 1:
            raise ValueError("lo, hi and shape must describe a 1D, 2D or 3D box with >= 1 cell per axis")
        axes = [np.linspace(lo[a], hi[a], shape[a] + 1) for a in range(d)]
        pts = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
        node_shape = tuple(s + 1 for s in shape)
        idx = np.arange(pts.shape[0]).reshape(node_shape)
        corners = np.stack(
            [g.ravel() for g in np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")], axis=1
        )
        cells = []
        for perm in itertools.permutations(range(d)):
            offset = np.zeros(d, dtype=np.int64)
            verts = [tuple(corners.T + offset[:, None])]
            for ax in perm:
                offset = offset.copy()
                offset[ax] += 1
                verts.append(tuple(corners.T + offset[:, None]))
            cells.append(np.stack([idx[v] for v in verts], axis=1))
        return cls(pts, np.concatenate(cells))

    @classmethod
    def from_points(cls, points: np.ndarray, keep=None) -> "Mesh":
        """Delaunay mesh of a point set (d >= 2). ``keep(centroids) -> bool mask``
        drops cells outside a non-convex region, e.g. ``domain.contains``."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError("from_points needs (N, 2) or (N, 3) points")
        cells = Delaunay(pts).simplices
        if keep is not None:
            cells = cells[np.asarray(keep(pts[cells].mean(axis=1)), dtype=bool)]
        cells = _drop_degenerate(pts, cells)
        used = np.unique(cells)
        remap = -np.ones(pts.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.size)
        return cls(pts[used], remap[cells])

    # -- geometry -----------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def cell_points(self) -> np.ndarray:
        """``(M, d+1, d)`` vertex coordinates of every cell."""
        return self.points[self.cells]

    @property
    def cell_volumes(self) -> np.ndarray:
        """Length / area / volume of each cell."""
        if "vol" not in self._cache:
            t = self._edges()
            self._cache["vol"] = np.abs(np.linalg.det(t)) / math.factorial(self.dim)
        return self._cache["vol"]

    @property
    def cell_centroids(self) -> np.ndarray:
        return self.cell_points.mean(axis=1)

    @property
    def volume(self) -> float:
        return float(self.cell_volumes.sum())

    def _edges(self) -> np.ndarray:
        cp = self.cell_points
        return cp[:, 1:, :] - cp[:, :1, :]  # (M, d, d), rows are edge vectors

    @property
    def grad_lambda(self) -> np.ndarray:
        """``(M, d+1, d)`` gradients of the barycentric coordinates of each cell."""
        if "gl" not in self._cache:
            inv = np.linalg.inv(self._edges())  # (M, d, d)
            g = np.transpose(inv, (0, 2, 1))  # row k = grad of lambda_{k+1}
            g0 = -g.sum(axis=1, keepdims=True)
            self._cache["gl"] = np.concatenate([g0, g], axis=1)
        return self._cache["gl"]

    def boundary_facets(self) -> np.ndarray:
        """``(K, d)`` vertex indices of the facets that belong to exactly one cell."""
        if "bf" not in self._cache:
            c = self.cells
            facets = np.concatenate([np.delete(c, i, axis=1) for i in range(c.shape[1])])
            key = np.sort(facets, axis=1)
            uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
            first = np.zeros(uniq.shape[0], dtype=np.int64)
            first[inv.ravel()[::-1]] = np.arange(facets.shape[0])[::-1]
            pick = first[counts == 1]
            self._cache["bf"] = facets[pick]
            # facet j came from cell pick % M with local vertex pick // M removed
            m = self.n_cells
            self._cache["bf_owner"] = (pick % m, self.cells[pick % m, pick // m])
        return self._cache["bf"]

    def boundary_geometry(self) -> dict:
        """Boundary facets with their centroid, outward unit normal and measure
        (length in 2D, area in 3D, 1 in 1D): keys ``facets, centroids, normals, measures``."""
        facets = self.boundary_facets()
        cell, opposite = self._cache["bf_owner"]
        fp = self.points[facets]  # (K, d, d)
        d = self.dim
        if d == 1:
            normal = np.ones((facets.shape[0], 1))
            measure = np.ones(facets.shape[0])
        elif d == 2:
            e = fp[:, 1, :] - fp[:, 0, :]
            measure = np.linalg.norm(e, axis=1)
            normal = np.stack([e[:, 1], -e[:, 0]], axis=1) / measure[:, None]
        else:
            c = np.cross(fp[:, 1, :] - fp[:, 0, :], fp[:, 2, :] - fp[:, 0, :])
            norm = np.linalg.norm(c, axis=1)
            measure = 0.5 * norm
            normal = c / norm[:, None]
        centroids = fp.mean(axis=1)
        away = centroids - self.points[opposite]
        flip = np.where((normal * away).sum(axis=1) < 0, -1.0, 1.0)
        return {"facets": facets, "centroids": centroids, "normals": normal * flip[:, None], "measures": measure}

    def boundary_nodes(self) -> np.ndarray:
        """Sorted indices of the points on the boundary."""
        return np.unique(self.boundary_facets())

    # -- P1 calculus on nodal values ---------------------------------------
    def cell_gradient(self, values: np.ndarray) -> np.ndarray:
        """Constant gradient of the P1 interpolant on each cell: ``(M, k, d)``."""
        v = _as_2d(values, self.n_points)
        return np.einsum("mik,mid->mkd", v[self.cells], self.grad_lambda)

    def gradient(self, values: np.ndarray) -> np.ndarray:
        """Nodal gradient ``(N, k, d)``: volume-weighted average of the cell
        gradients around each node. Exact for linear data."""
        cg = self.cell_gradient(values)
        w = self.cell_volumes
        n, k, d = self.n_points, cg.shape[1], self.dim
        out = np.zeros((n, k * d))
        flat = (cg * w[:, None, None]).reshape(self.n_cells, k * d)
        for i in range(self.dim + 1):
            np.add.at(out, self.cells[:, i], flat)
        wsum = np.zeros(n)
        for i in range(self.dim + 1):
            np.add.at(wsum, self.cells[:, i], w)
        return (out / wsum[:, None]).reshape(n, k, d)

    def integrate(self, values: np.ndarray) -> np.ndarray:
        """Integral of the P1 interpolant over the mesh, one value per component. Exact."""
        v = _as_2d(values, self.n_points)
        cell_mean = v[self.cells].mean(axis=1)  # (M, k)
        return (cell_mean * self.cell_volumes[:, None]).sum(axis=0)

    def locate(self, x: np.ndarray, k: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Containing cell and barycentric coordinates of each query point.

        Returns ``(cell, bary)`` with ``cell == -1`` and NaN coordinates for
        points outside the mesh."""
        x = _as_points(x, self.dim)
        q = x.shape[0]
        if "tree" not in self._cache:
            self._cache["tree"] = cKDTree(self.cell_centroids)
        tree = self._cache["tree"]
        cell = -np.ones(q, dtype=np.int64)
        bary = np.full((q, self.dim + 1), np.nan)
        todo = np.arange(q)
        for kk in (min(k, self.n_cells), min(256, self.n_cells)):
            if todo.size == 0:
                break
            _, cand = tree.query(x[todo], k=kk)
            cand = cand.reshape(todo.size, -1)
            origin = self.cell_points[cand, 0, :]  # (t, kk, d)
            lam = np.einsum("tkd,tkid->tki", x[todo][:, None, :] - origin, self.grad_lambda[cand])
            lam[:, :, 0] += 1.0  # lambda_0 = 1 - sum(lambda_i): gradient part already sums to zero
            ok = lam.min(axis=2) >= -_BARY_TOL
            hit = ok.any(axis=1)
            first = ok.argmax(axis=1)
            sel = todo[hit]
            cell[sel] = cand[hit, first[hit]]
            bary[sel] = lam[hit, first[hit]]
            todo = todo[~hit]
            if kk == self.n_cells:
                break
        return cell, bary

    def interpolate(self, values: np.ndarray, x: np.ndarray, fill: str = "nan") -> np.ndarray:
        """P1 interpolation of nodal values at ``x``: ``(Q, k)``.

        Points outside the mesh get NaN (``fill="nan"``) or the value of the
        nearest node (``fill="nearest"``)."""
        v = _as_2d(values, self.n_points)
        x = _as_points(x, self.dim)
        cell, bary = self.locate(x)
        out = np.full((x.shape[0], v.shape[1]), np.nan)
        inside = cell >= 0
        out[inside] = np.einsum("qi,qik->qk", bary[inside], v[self.cells[cell[inside]]])
        if fill == "nearest" and not inside.all():
            if "ptree" not in self._cache:
                self._cache["ptree"] = cKDTree(self.points)
            _, nn = self._cache["ptree"].query(x[~inside])
            out[~inside] = v[nn]
        elif fill not in ("nan", "nearest"):
            raise ValueError("fill must be 'nan' or 'nearest'")
        return out

    def __repr__(self) -> str:
        return f"Mesh(dim={self.dim}, points={self.n_points}, cells={self.n_cells})"


def _as_2d(values: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.shape[:1] != (n,):
        raise ValueError(f"expected {n} nodal values, got shape {v.shape}")
    return v.reshape(n, -1)


def _as_points(x: np.ndarray, d: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None] if d == 1 else x[None, :]
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(f"points must have shape (Q, {d}), got {x.shape}")
    return x


def _drop_degenerate(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    cp = points[cells]
    t = cp[:, 1:, :] - cp[:, :1, :]
    vol = np.abs(np.linalg.det(t))
    return cells[vol > 1e-12 * max(vol.max(), 1e-300)]
