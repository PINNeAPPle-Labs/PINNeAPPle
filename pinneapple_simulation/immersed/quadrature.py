"""Cut-cell quadrature by adaptive octree subdivision (quadtree in 2-D).

Every background cell is classified from the level set at its corners and centre (a box whose centre is farther from
the surface than its half-diagonal, times the Lipschitz bound, cannot be cut). Cut cells are split into 2^d children,
recursively, down to ``depth`` levels. Children that are inside get a tensor Gauss rule; the leaves still cut at the
last level are split into d! Kuhn simplices and each simplex is clipped by the level set (Parvizian, Duester & Rank,
Comput. Mech. 2007, octree integration; simplex clipping as in marching tetrahedra). The surface points on the
simplices are found by a few regula-falsi steps on the true level set, so the clipped surface interpolates the real
one: the geometric error is O((h / 2^depth)^2). The same clipping yields the boundary facets, with their normals, for
the surface quadrature of Nitsche, Neumann and Robin terms.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cache
from itertools import permutations, product
from math import factorial

import numpy as np

from . import basis
from .geometry import Geometry
from .grid import BackgroundGrid

OUTSIDE, CUT, INSIDE = 0, 1, 2


@cache
def simplex_rule(dim: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature on the unit simplex of dimension ``dim`` (0..3): barycentric coordinates (q, dim + 1) and weights
    summing to 1. ``n`` >= 1: collapsed (Duffy) Gauss with n points per direction; ``n`` = 0: the symmetric
    degree-2 rule (dim + 1 points, 2 on a segment); ``n`` < 0: the centroid."""
    if dim == 0:
        return np.ones((1, 1)), np.ones(1)
    if n < 0:                                    # centroid (degree 1)
        return np.full((1, dim + 1), 1 / (dim + 1)), np.ones(1)
    if n == 0:                                   # symmetric, positive, degree-2 rules (Stroud)
        if dim == 1:
            g = 0.5 / np.sqrt(3)
            return np.array([[0.5 + g, 0.5 - g], [0.5 - g, 0.5 + g]]), np.full(2, 0.5)
        a = {2: 1 / 6, 3: 0.1381966011250105}[dim]
        b = 1 - dim * a
        lam = np.full((dim + 1, dim + 1), a)
        np.fill_diagonal(lam, b)
        return lam, np.full(dim + 1, 1 / (dim + 1))
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    pts, wts = [], []
    for idx in product(range(n), repeat=dim):
        u = x[list(idx)]
        ww = np.prod(w[list(idx)])
        if dim == 1:
            p, J = [u[0]], 1.0
        elif dim == 2:
            p, J = [u[0], (1 - u[0]) * u[1]], (1 - u[0])
        else:
            p = [u[0], (1 - u[0]) * u[1], (1 - u[0]) * (1 - u[1]) * u[2]]
            J = (1 - u[0]) ** 2 * (1 - u[1])
        pts.append(p)
        wts.append(ww * J)
    P = np.array(pts)
    W = np.array(wts)
    return np.column_stack([1 - P.sum(1), P]), W / W.sum()


@cache
def _kuhn(d: int) -> np.ndarray:
    """(d!, d + 1) corner numbers (C order over the axes) of the Kuhn simplices of the unit box."""
    corners = {c: i for i, c in enumerate(product((0, 1), repeat=d))}
    out = []
    for perm in permutations(range(d)):
        v = [0] * d
        s = [corners[tuple(v)]]
        for ax in perm:
            v[ax] = 1
            s.append(corners[tuple(v)])
        out.append(s)
    return np.array(out)


# clipping templates: for k inside vertices, the pieces of the inside region and of the surface, as indices into
# [I_0 .. I_{k-1}, P_00, P_01, ...] where P_ij is the surface point on the edge from inside vertex i to outside j
def _p(k: int, d: int, i: int, j: int) -> int:
    return k + i * (d + 1 - k) + j


@cache
def _templates(d: int, k: int) -> tuple[list, list]:
    P = lambda i, j: _p(k, d, i, j)                                     # noqa: E731
    if d == 2:
        if k == 1:
            return [(0, P(0, 0), P(0, 1))], [(P(0, 0), P(0, 1))]
        if k == 2:
            return [(0, 1, P(1, 0)), (0, P(1, 0), P(0, 0))], [(P(0, 0), P(1, 0))]
    if d == 3:
        if k == 1:
            return [(0, P(0, 0), P(0, 1), P(0, 2))], [(P(0, 0), P(0, 1), P(0, 2))]
        def prism(v):
            return [(v[0], v[1], v[2], v[3]), (v[1], v[2], v[3], v[4]), (v[2], v[3], v[4], v[5])]
        if k == 2:
            v = (0, P(0, 0), P(0, 1), 1, P(1, 0), P(1, 1))
            return prism(v), [(P(0, 0), P(0, 1), P(1, 1)), (P(0, 0), P(1, 1), P(1, 0))]
        if k == 3:
            v = (0, 1, 2, P(0, 0), P(1, 0), P(2, 0))
            return prism(v), [(P(0, 0), P(1, 0), P(2, 0))]
    raise ValueError(f"no clipping template for d={d}, k={k}")


@dataclass
class Points:
    cell: np.ndarray             # (n,) background cell of each point
    xi: np.ndarray               # (n, d) unit-cell coordinates
    w: np.ndarray                # (n,) physical weights (volume or area)
    normal: np.ndarray | None = None   # (n, d) outward unit normals (surface points)

    def __len__(self):
        return len(self.w)

    def subset(self, m: np.ndarray) -> Points:
        return Points(self.cell[m], self.xi[m], self.w[m], None if self.normal is None else self.normal[m])

    @staticmethod
    def concat(parts: list[Points], d: int, surface: bool = False) -> Points:
        parts = [p for p in parts if len(p)]
        if not parts:
            return Points(np.zeros(0, np.int64), np.zeros((0, d)), np.zeros(0), np.zeros((0, d)) if surface else None)
        nrm = np.concatenate([p.normal for p in parts]) if surface else None
        return Points(np.concatenate([p.cell for p in parts]), np.concatenate([p.xi for p in parts]),
                      np.concatenate([p.w for p in parts]), nrm)


@dataclass
class CutQuadrature:
    """Volume quadrature of the cut cells, the list of the cells entirely inside, and the boundary quadrature."""

    grid: BackgroundGrid
    status: np.ndarray           # per cell: OUTSIDE, CUT, INSIDE
    volume: Points               # quadrature of the inside part of the cut cells
    surface: Points              # quadrature of the boundary (facets of the clipped simplices)
    n_gauss: int
    stats: dict = field(default_factory=dict)
    fine: Points | None = None   # the fine octree points (positive weights, all inside), when kept

    @property
    def inside_cells(self) -> np.ndarray:
        return np.nonzero(self.status == INSIDE)[0]

    @property
    def cut_cells(self) -> np.ndarray:
        return np.nonzero(self.status == CUT)[0]

    @property
    def active_cells(self) -> np.ndarray:
        return np.nonzero(self.status != OUTSIDE)[0]

    def cut_fraction(self) -> tuple[np.ndarray, np.ndarray]:
        """Cut cells and the fraction of each one that is inside the part."""
        cells = self.cut_cells
        vol = np.bincount(self.volume.cell, self.volume.w, minlength=self.grid.n_cells)[cells]
        return cells, vol / np.prod(self.grid.h)

    def inside_points(self, n_gauss: int | None = None) -> Points:
        """Tensor Gauss points of the cells entirely inside."""
        q, w = basis.gauss_cell(n_gauss or self.n_gauss, self.grid.d)
        cells = self.inside_cells
        vol = float(np.prod(self.grid.h))
        return Points(np.repeat(cells, len(w)), np.tile(q, (len(cells), 1)), np.tile(w * vol, len(cells)))

    def all_points(self, n_gauss: int | None = None, fine: bool = False) -> Points:
        """Every volume point: Gauss points of the inside cells and the cut-cell rule (the fine one if asked and
        kept)."""
        cut = self.fine if (fine and self.fine is not None) else self.volume
        return Points.concat([self.inside_points(n_gauss), cut], self.grid.d)

    def measure(self) -> float:
        return float(self.volume.w.sum() + len(self.inside_cells) * np.prod(self.grid.h))

    def area(self) -> float:
        return float(self.surface.w.sum())


class _LatticePhi:
    """The level set on the lattice of cell corners refined ``res`` times, evaluated once per distinct point (a
    cache shared by all octree levels: the corners of a box are corners of its children)."""

    def __init__(self, geom: Geometry, grid: BackgroundGrid, res: int):
        self.geom, self.grid, self.res = geom, grid, res
        self.shape = tuple(k * res + 1 for k in grid.n)
        self.calls = 0
        self._keys = np.zeros(0, np.int64)
        self._vals = np.zeros(0)

    def physical(self, q: np.ndarray) -> np.ndarray:
        return self.grid.lo + q * self.grid.h / self.res

    def __call__(self, q: np.ndarray) -> np.ndarray:
        flat = q.reshape(-1, q.shape[-1])
        key = np.ravel_multi_index(flat.T, self.shape)
        uk, inv = np.unique(key, return_inverse=True)
        pos = np.searchsorted(self._keys, uk)
        hit = pos < len(self._keys)
        hit[hit] = self._keys[pos[hit]] == uk[hit]
        vals = np.empty(len(uk))
        vals[hit] = self._vals[pos[hit]]
        new = uk[~hit]
        if len(new):
            pts = self.grid.lo + np.stack(np.unravel_index(new, self.shape), axis=1) * self.grid.h / self.res
            vals[~hit] = self.geom.phi(pts)
            self.calls += len(new)
            keys = np.concatenate([self._keys, new])
            o = np.argsort(keys, kind="stable")
            self._keys, self._vals = keys[o], np.concatenate([self._vals, vals[~hit]])[o]
        return vals[inv.ravel()].reshape(q.shape[:-1])


@cache
def _fit_basis(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss points of [0, 1] with the monomial coefficients of their Lagrange polynomials and of the
    antiderivatives (columns = polynomials, rows = powers 0..m)."""
    xg = (np.polynomial.legendre.leggauss(m)[0] + 1) / 2
    C = np.linalg.inv(np.vander(xg, m, increasing=True)).copy()        # l_a(x) = sum_j x^j C[j, a]
    A = np.vstack([np.zeros((1, m)), C / np.arange(1, m + 1)[:, None]])
    return xg, C, A


def _tensor_rows(tabs: list[np.ndarray]) -> np.ndarray:
    out = tabs[0]
    for t in tabs[1:]:
        out = (out[:, :, None] * t[:, None, :]).reshape(len(out), -1)
    return out


class MomentFitter:
    """Moment fitting: everything integrated over the inside part of a cut cell is compressed onto the m^d Gauss
    points of the whole cell, with the weights w_g = int l_g (l_g the Lagrange polynomials of the Gauss points).
    The compressed rule integrates every polynomial of degree m - 1 per axis exactly as the fine rule does
    (stiffness and mass of Q_p need m = 2p + 1); its points can lie outside the part and its weights can be
    negative (Mueller, Kummer & Oberlack, IJNME 2013)."""

    def __init__(self, grid: BackgroundGrid, cells: np.ndarray, m: int):
        self.grid, self.cells, self.m = grid, cells, m
        self.xg, self.C, self.A = _fit_basis(m)
        self.W = np.zeros((len(cells), m ** grid.d))

    def _add(self, cell: np.ndarray, rows: np.ndarray, w: np.ndarray | None = None):
        from scipy import sparse
        r = np.searchsorted(self.cells, cell)
        val = np.ones(len(cell)) if w is None else w
        S = sparse.csr_matrix((val, (r, np.arange(len(cell)))), shape=(len(self.cells), len(cell)))
        self.W += S @ rows

    def measures(self, pts: Points, weights: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Fitted weights of several measures carried by the same points (rows of ``weights``, e.g. dS and n_k dS on
        the boundary): (n_measures, n_cells, m^d)."""
        from scipy import sparse
        weights = np.atleast_2d(weights)
        out = np.zeros((len(weights), len(self.cells), self.m ** self.grid.d))
        for s in range(0, len(pts), chunk):
            sl = slice(s, s + chunk)
            V = _tensor_rows([np.vander(pts.xi[sl, k], self.m, increasing=True) @ self.C
                              for k in range(self.grid.d)])
            r = np.searchsorted(self.cells, pts.cell[sl])
            for i, w in enumerate(weights):
                S = sparse.csr_matrix((w[sl], (r, np.arange(len(r)))), shape=(len(self.cells), len(r)))
                out[i] += S @ V
        return out

    def points(self, pts: Points, chunk: int = 8192):
        for s in range(0, len(pts), chunk):
            xi = pts.xi[s:s + chunk]
            V = [np.vander(xi[:, k], self.m, increasing=True) @ self.C for k in range(self.grid.d)]
            self._add(pts.cell[s:s + chunk], _tensor_rows(V), pts.w[s:s + chunk])

    def boxes(self, cells: np.ndarray, lo: np.ndarray, size: float):
        """Exact weights of boxes entirely inside: products of 1-D integrals of the Lagrange polynomials."""
        tabs = []
        for k in range(self.grid.d):
            a = np.vander(lo[:, k] + size, self.m + 1, increasing=True) @ self.A
            b = np.vander(lo[:, k], self.m + 1, increasing=True) @ self.A
            tabs.append(a - b)
        self._add(cells, _tensor_rows(tabs) * np.prod(self.grid.h))

    def result(self) -> Points:
        d, nq = self.grid.d, self.m ** self.grid.d
        pts = np.array(list(product(self.xg, repeat=d)))
        return Points(np.repeat(self.cells, nq), np.tile(pts, (len(self.cells), 1)), self.W.ravel())


def build_quadrature(geom: Geometry, grid: BackgroundGrid, depth: int = 3, n_gauss: int | None = None,
                     n_simplex: int = 0, n_facet: int | None = None, refine: int = 3, fit: bool = True,
                     keep_fine: bool = False, chunk: int = 50000) -> CutQuadrature:
    """Octree cut-cell quadrature of ``geom`` on ``grid``.

    ``depth``: subdivision levels of cut cells (the leaves are h / 2^depth); ``n_gauss``: Gauss points per axis on
    the cells entirely inside (default order + 1); ``n_simplex``: rule on the clipped simplices and facets (0 = the
    symmetric degree-2 rules, n >= 1 = collapsed Gauss with n points per direction); ``n_facet``: the same for the
    boundary facets (default 0: degree 2; the centroid, -1, is cheaper but costs accuracy in Q2); ``refine``: regula-falsi steps
    that put the facet vertices on the true surface (0 = linear interpolation of the level set); ``fit``: compress
    the volume rule of each cut cell by moment fitting onto (2 order + 1)^d points (exact for the stiffness and mass
    integrands, see :class:`MomentFitter`); ``keep_fine``: also keep the fine octree points (positive weights, all inside
    the part: used for error norms).
    """
    t0 = time.perf_counter()
    d, p = grid.d, grid.order
    ng = int(n_gauss or p + 1)
    ns = int(n_simplex)
    nf = int(n_facet if n_facet is not None else 0)
    res = 2 ** (depth + 1)                       # lattice units per cell (corners and centres of the finest boxes)
    lat = _LatticePhi(geom, grid, res)
    off = np.array(list(product((0, 1), repeat=d)), np.int64)          # box corners, C order
    hnorm = float(np.linalg.norm(grid.h))
    lip = float(getattr(geom, "lipschitz", 1.0))
    vcell = float(np.prod(grid.h))

    cells = np.arange(grid.n_cells, dtype=np.int64)
    sub = np.zeros((grid.n_cells, d), np.int64)
    status = np.full(grid.n_cells, OUTSIDE, np.int8)
    cmul_all = grid.cell_multi(cells) * res
    gq, gw = basis.gauss_cell(ng, d)
    fine_parts, surf_parts = [], []
    fitter = None
    leaves = None
    n_boxes = 0
    for level in range(depth + 1):
        sz = res >> level
        if not len(cells):
            break
        n_boxes += len(cells)
        base = cmul_all[cells] + sub
        pc = lat(base[:, None, :] + off[None] * sz)                    # (m, 2^d) corner values
        pm = lat(base + sz // 2)                                       # centre values
        r = 0.5 * hnorm * sz / res
        allneg, allpos = (pc < 0).all(1), (pc > 0).all(1)
        inside = allneg & (pm < -lip * r)
        outside = allpos & (pm > lip * r)
        if level == depth:                       # leaves: same-sign corners decide, the rest is clipped
            inside |= allneg
            outside |= allpos
        amb = ~(inside | outside)
        if level == 0:
            status[cells[inside]] = INSIDE
            status[cells[amb]] = CUT
            if fit:
                fitter = MomentFitter(grid, np.nonzero(status == CUT)[0], 2 * p + 1)
        elif inside.any():
            ci, lo = cells[inside], sub[inside] / res
            s = sz / res
            if fit:
                fitter.boxes(ci, lo, s)
            if keep_fine or not fit:
                fine_parts.append(Points(np.repeat(ci, len(gw)), (lo[:, None, :] + s * gq[None]).reshape(-1, d),
                                         np.tile(gw * s ** d * vcell, len(ci))))
        if level == depth:
            leaves = (cells[amb], sub[amb], pc[amb], sz)
        else:
            c2, s2 = cells[amb], sub[amb]
            half = sz // 2
            cells = np.repeat(c2, 2 ** d)
            sub = (s2[:, None, :] + off[None] * half).reshape(-1, d)
    n_leaf = n_simp = 0
    if leaves is not None and len(leaves[0]):
        lc_all, ls_all, lpc_all, sz = leaves
        n_leaf = len(lc_all)
        K = _kuhn(d)                                                    # (d!, d + 1)
        for c0 in range(0, n_leaf, chunk):
            lc, ls, lpc = lc_all[c0:c0 + chunk], ls_all[c0:c0 + chunk], lpc_all[c0:c0 + chunk]
            sc = np.repeat(lc, len(K))
            sq = (ls[:, None, None, :] + off[K][None] * sz).reshape(-1, d + 1, d)    # local lattice coords
            sphi = lpc[:, K].reshape(-1, d + 1)
            k_in = (sphi < 0).sum(1)
            parts = []
            full = k_in == d + 1
            if full.any():
                parts.append(_simplex_points(grid, sc[full], sq[full] / res, ns))
            for k in range(1, d + 1):
                sel = np.nonzero(k_in == k)[0]
                if not len(sel):
                    continue
                n_simp += len(sel)
                order = np.argsort(~(sphi[sel] < 0), axis=1, kind="stable")          # inside vertices first
                q = np.take_along_axis(sq[sel], order[:, :, None], axis=1)
                f = np.take_along_axis(sphi[sel], order, axis=1)
                vol, srf = _clip(geom, grid, lat, sc[sel], cmul_all[sc[sel]], q, f, k, res, ns, nf, refine)
                parts.append(vol)
                surf_parts.append(srf)
            pts = Points.concat(parts, d)
            if fit:
                fitter.points(pts)
            if keep_fine or not fit:
                fine_parts.append(pts)
    surface = Points.concat(surf_parts, d, surface=True)
    fine = Points.concat(fine_parts, d) if (keep_fine or not fit) else None
    volume = fitter.result() if fit else fine
    # cut cells that ended with no inside volume are inactive; keep only the points of active cells
    vol_of = np.bincount(volume.cell, volume.w, minlength=grid.n_cells)
    status[(status == CUT) & (vol_of <= 1e-12 * vcell)] = OUTSIDE
    surface = surface.subset(status[surface.cell] != OUTSIDE)
    volume = volume.subset(status[volume.cell] != OUTSIDE)
    if fine is not None:
        fine = fine.subset((status[fine.cell] != OUTSIDE) & (fine.w > 0))
    stats = {"depth": depth, "boxes": n_boxes, "leaves": n_leaf, "clipped_simplices": n_simp,
             "phi_evaluations": lat.calls, "volume_points": len(volume), "surface_points": len(surface),
             "cells_inside": int((status == INSIDE).sum()), "cells_cut": int((status == CUT).sum()),
             "moment_fitted": bool(fit), "seconds": time.perf_counter() - t0}
    return CutQuadrature(grid, status, volume, surface, ng, stats, fine)


def _simplex_points(grid: BackgroundGrid, cells: np.ndarray, V: np.ndarray, n: int) -> Points:
    """Quadrature points of whole simplices with vertices ``V`` (m, d + 1, d) in unit-cell coordinates."""
    d = V.shape[-1]
    lam, w = simplex_rule(d, n)
    E = (V[:, 1:] - V[:, :1]) * grid.h
    vol = np.abs(np.linalg.det(E)) / factorial(d)
    xi = np.einsum("qk,mkd->mqd", lam, V).reshape(-1, d)
    return Points(np.repeat(cells, len(w)), xi, (vol[:, None] * w[None]).ravel())


def _roots(geom: Geometry, lat: _LatticePhi, qa: np.ndarray, qb: np.ndarray, fa: np.ndarray, fb: np.ndarray,
           refine: int) -> np.ndarray:
    """Position t in [0, 1] of the surface on lattice edges from qa (inside) to qb, by Illinois regula falsi on the
    true level set; edges are deduplicated first."""
    shape = lat.shape
    ka, kb = np.ravel_multi_index(qa.T, shape), np.ravel_multi_index(qb.T, shape)
    uk, first, inv = np.unique(np.stack([ka, kb], 1), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    fa, fb = fa[first].astype(float), fb[first].astype(float)
    xa, xb = lat.physical(qa[first]), lat.physical(qb[first])
    ta, tb = np.zeros(len(uk)), np.ones(len(uk))
    side = np.zeros(len(uk), np.int8)
    tol = 1e-6 * float(np.linalg.norm(lat.grid.h)) / lat.res          # converged: |phi| below 1e-6 of a leaf
    live = (fb != 0) & (fa != 0)
    for _ in range(refine):
        if not live.any():
            break
        L = np.nonzero(live)[0]
        t = ta[L] - fa[L] * (tb[L] - ta[L]) / np.where(fb[L] - fa[L] == 0, 1, fb[L] - fa[L])
        f = geom.phi(xa[L] + t[:, None] * (xb[L] - xa[L]))
        lat.calls += len(L)
        neg = f < 0
        # Illinois: halve the stale end when the same side is kept twice
        fa_new = np.where(neg, f, np.where(side[L] == 1, fa[L] / 2, fa[L]))
        fb_new = np.where(~neg, f, np.where(side[L] == -1, fb[L] / 2, fb[L]))
        ta[L], tb[L] = np.where(neg, t, ta[L]), np.where(~neg, t, tb[L])
        side[L] = np.where(neg, -1, 1)
        fa[L], fb[L] = fa_new, fb_new
        live[L] = np.abs(f) > tol
        done = L[np.abs(f) <= tol]
        ta[done] = tb[done] = t[np.abs(f) <= tol]                        # converged on the surface
        fa[done], fb[done] = -1.0, 1.0
    t = np.where(fb == 0, tb, ta - fa * (tb - ta) / np.where(fb - fa == 0, 1, fb - fa))
    return np.clip(t, 0, 1)[inv]


def _clip(geom, grid, lat, cells, cbase, q, f, k, res, n, nf, refine) -> tuple[Points, Points]:
    """Clip simplices with ``k`` inside vertices (listed first). q: (m, d + 1, d) local lattice coordinates."""
    m, nv, d = q.shape
    ii, jj = np.meshgrid(np.arange(k), np.arange(k, nv), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()                                   # inside-outside vertex pairs, P order
    qa = (cbase[:, None, :] + q[:, ii]).reshape(-1, d)
    qb = (cbase[:, None, :] + q[:, jj]).reshape(-1, d)
    t = _roots(geom, lat, qa, qb, f[:, ii].ravel(), f[:, jj].ravel(), refine).reshape(m, len(ii))
    X = q / res                                                        # unit-cell coordinates of the vertices
    P = X[:, ii] + t[:, :, None] * (X[:, jj] - X[:, ii])
    pts = np.concatenate([X[:, :k], P], axis=1)                        # labels as in the templates
    vt, st = _templates(d, k)
    vols = [_simplex_points(grid, cells, pts[:, list(s)], n) for s in vt]
    # facets: physical measure and the normal pointing to the first outside vertex
    lam, w = simplex_rule(d - 1, nf)
    out = X[:, k]
    srf = []
    for s in st:
        F = pts[:, list(s)]                                            # (m, d, d)
        E = (F[:, 1:] - F[:, :1]) * grid.h
        if d == 2:
            nrm = np.stack([E[:, 0, 1], -E[:, 0, 0]], 1)
            meas = np.linalg.norm(E[:, 0], axis=1)
        else:
            nrm = np.cross(E[:, 0], E[:, 1])
            meas = 0.5 * np.linalg.norm(nrm, axis=1)
        nl = np.linalg.norm(nrm, axis=1)
        nrm = nrm / np.where(nl == 0, 1, nl)[:, None]
        sgn = np.sign(np.einsum("ij,ij->i", nrm, (out - F[:, 0]) * grid.h))
        nrm = nrm * np.where(sgn == 0, 1, sgn)[:, None]
        xi = np.einsum("qk,mkd->mqd", lam, F).reshape(-1, d)
        srf.append(Points(np.repeat(cells, len(w)), xi, (meas[:, None] * w[None]).ravel(),
                          np.repeat(nrm, len(w), axis=0)))
    return Points.concat(vols, d), Points.concat(srf, d, surface=True)
