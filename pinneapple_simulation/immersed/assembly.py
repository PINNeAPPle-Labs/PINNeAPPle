"""Assembly on the immersed discretisation: element matrices of whole cells (one reference matrix, the grid is
uniform), of cut cells (moment-fitted rules: one matrix product for all of them), of boundary points (reduced per
cell), and the ghost-penalty face matrices; scatter into sparse matrices; condition numbers; linear solves."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy import sparse

from . import basis
from .geometry import Geometry
from .grid import BackgroundGrid, DofMap
from .quadrature import CUT, CutQuadrature, MomentFitter, Points, build_quadrature


@dataclass
class Discretization:
    """Background grid + cut-cell quadrature + unknowns for one geometry. Shared by the heat, elasticity and
    transient solvers; build it once and reuse it for several load cases or materials."""

    geometry: Geometry
    h: float
    order: int = 1
    depth: int | None = None
    pad: float = 1.0
    n_simplex: int = 0
    refine: int = 3
    keep_fine: bool = False
    jitter: bool = True
    timings: dict = field(default_factory=dict)

    def __post_init__(self):
        t0 = time.perf_counter()
        if self.depth is None:
            self.depth = 2 if self.order == 1 else 3
        self.grid = BackgroundGrid.around(self.geometry.bounds, self.h, self.order, self.pad, self.jitter)
        self.quad = build_quadrature(self.geometry, self.grid, depth=self.depth, n_simplex=self.n_simplex,
                                     refine=self.refine, keep_fine=self.keep_fine)
        self.dofs = DofMap.build(self.grid, self.quad.active_cells)
        self.timings["quadrature_s"] = time.perf_counter() - t0

    @property
    def d(self) -> int:
        return self.grid.d

    @property
    def n_dofs(self) -> int:
        return self.dofs.n_dofs

    def summary(self) -> dict:
        cells, frac = self.quad.cut_fraction()
        return {"order": self.order, "h": self.h, "dim": self.d, "depth": self.depth, "dofs": self.n_dofs,
                "cells_active": int(len(self.dofs.cells)), "cells_cut": int(len(cells)),
                "min_cut_fraction": float(frac.min()) if len(frac) else 1.0,
                "volume": self.quad.measure(), "area": self.quad.area(), **self.quad.stats}

    # ------------------------------------------------------------ field evaluation
    def evaluate(self, u: np.ndarray, X: np.ndarray, gradient: bool = False, ncomp: int = 1) -> np.ndarray:
        """Values (or gradients) of a discrete field at points; NaN outside the active cells."""
        X = np.atleast_2d(np.asarray(X, float))
        cell, xi, ok = self.grid.locate(X)
        row = self.dofs.index[cell]
        ok &= row >= 0
        U = np.asarray(u).reshape(-1, ncomp)
        shp = (len(X), ncomp, self.d) if gradient else (len(X), ncomp)
        out = np.full(shp, np.nan)
        if ok.any():
            dofs = self.dofs.cell_dofs[row[ok]]
            if gradient:
                G = basis.grad(self.order, xi[ok]) / self.grid.h
                out[ok] = np.einsum("nbk,nbc->nck", G, U[dofs])
            else:
                N = basis.shape(self.order, xi[ok])
                out[ok] = np.einsum("nb,nbc->nc", N, U[dofs])
        return out[:, 0] if ncomp == 1 else out

    def physical(self, pts: Points) -> np.ndarray:
        return self.grid.to_physical(pts.cell, pts.xi)

    @cached_property
    def fitted_reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Basis values and physical gradients at the moment-fitted points of one cut cell, with their unit-cell
        coordinates (the same points in every cut cell)."""
        cut = self.quad.cut_cells
        nq = (2 * self.order + 1) ** self.d
        xi = self.quad.volume.xi[:nq] if len(cut) else np.zeros((0, self.d))
        return basis.shape(self.order, xi), basis.grad(self.order, xi) / self.grid.h, xi

    def cut_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Cut cells and their (n_cut, nq) fitted weights."""
        cut = self.quad.cut_cells
        nq = (2 * self.order + 1) ** self.d
        return cut, self.quad.volume.w.reshape(len(cut), nq)

    def volume_points(self, fine: bool = False) -> Points:
        return self.quad.all_points(fine=fine)


class SurfaceMeasures:
    """Boundary measures (dS, n_k dS, data-weighted ones) of a set of boundary points, moment-fitted per cell onto
    the (2p + 1)^d Gauss points: every boundary term of a Q_p element becomes one small matrix product per measure,
    exact with respect to the fine facet rule (the integrands are in Q_2p)."""

    def __init__(self, disc: Discretization, pts: Points, weights: dict[str, np.ndarray]):
        self.cells = np.unique(pts.cell)
        names = list(weights)
        fit = MomentFitter(disc.grid, self.cells, 2 * disc.order + 1)
        W = fit.measures(pts, np.stack([pts.w * np.asarray(weights[k], float) for k in names]))
        self.W = dict(zip(names, W, strict=True))
        N, G, _ = disc.fitted_reference
        if not len(N):
            from itertools import product as _p
            xg = (np.polynomial.legendre.leggauss(2 * disc.order + 1)[0] + 1) / 2
            xi = np.array(list(_p(xg, repeat=disc.d)))
            N, G = basis.shape(disc.order, xi), basis.grad(disc.order, xi) / disc.grid.h
        self.N, self.G = N, G

    def outer(self, name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-cell matrices sum_q W[name][c, q] a_q b_q^T for point tables a, b (nq, nb)."""
        nq, nb = a.shape
        return (self.W[name] @ np.einsum("qi,qj->qij", a, b).reshape(nq, -1)).reshape(-1, nb, b.shape[1])

    def vector(self, name: str, a: np.ndarray) -> np.ndarray:
        return self.W[name] @ a


def reference_cell(grid: BackgroundGrid) -> dict[str, np.ndarray]:
    """Stiffness (grad . grad), mass and the gradient table of a whole cell, in physical units."""
    q, w = basis.gauss_cell(grid.order + 1, grid.d)
    vol = float(np.prod(grid.h))
    N = basis.shape(grid.order, q)
    G = basis.grad(grid.order, q) / grid.h
    return {"K": np.einsum("q,qik,qjk->ij", w * vol, G, G), "M": np.einsum("q,qi,qj->ij", w * vol, N, N),
            "G": G, "N": N, "w": w * vol, "xi": q}


def scatter(cell_dofs: np.ndarray, mats: np.ndarray, n: int, ncomp: int = 1) -> sparse.csr_matrix:
    """Sum element matrices (m, nb*ncomp, nb*ncomp) into an n*ncomp sparse matrix; dofs are interleaved by
    component (node-major)."""
    if not len(mats):
        return sparse.csr_matrix((n * ncomp, n * ncomp))
    idx = (cell_dofs[:, :, None] * ncomp + np.arange(ncomp)[None, None, :]).reshape(len(cell_dofs), -1)
    out = sparse.csr_matrix((n * ncomp, n * ncomp))
    step = max(1, 4_000_000 // (idx.shape[1] ** 2))
    for s in range(0, len(idx), step):
        rows = idx[s:s + step]
        r = np.repeat(rows, rows.shape[1], axis=1).ravel()
        c = np.tile(rows, (1, rows.shape[1])).ravel()
        out = out + sparse.csr_matrix((mats[s:s + step].ravel(), (r, c)), shape=out.shape)
    return out


def scatter_vector(cell_dofs: np.ndarray, vecs: np.ndarray, n: int, ncomp: int = 1) -> np.ndarray:
    idx = (cell_dofs[:, :, None] * ncomp + np.arange(ncomp)[None, None, :]).reshape(len(cell_dofs), -1)
    return np.bincount(idx.ravel(), vecs.ravel(), minlength=n * ncomp)


def reduce_by_cell(cell: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], chunk: int = 20000
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Group point contributions by cell: ``fn(index_array)`` returns per-point arrays (n, ...) that are summed per
    cell. Returns the distinct cells and the sums."""
    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    cells, starts = np.unique(cs, return_index=True)
    ends = np.append(starts[1:], len(cs))
    out = []
    i = 0
    while i < len(cells):
        j = i
        while j < len(cells) and ends[j] - starts[i] <= chunk:
            j += 1
        j = max(j, i + 1)
        sl = order[starts[i]:ends[j - 1]]
        vals = fn(sl)
        out.append(np.add.reduceat(vals, starts[i:j] - starts[i], axis=0))
        i = j
    return cells, np.concatenate(out) if out else np.zeros((0,))


def ghost_faces(quad: CutQuadrature, dofs: DofMap) -> list[np.ndarray]:
    """Per axis, the (m, 2) pairs of active cells sharing a face where at least one of the two is cut."""
    grid = quad.grid
    st = quad.status.reshape(grid.n)
    out = []
    for k in range(grid.d):
        a = [slice(None)] * grid.d
        b = [slice(None)] * grid.d
        a[k], b[k] = slice(0, -1), slice(1, None)
        sa, sb = st[tuple(a)], st[tuple(b)]
        m = (sa > 0) & (sb > 0) & ((sa == CUT) | (sb == CUT))
        ids = np.arange(grid.n_cells).reshape(grid.n)
        out.append(np.stack([ids[tuple(a)][m], ids[tuple(b)][m]], axis=1))
    return out


def ghost_reference(grid: BackgroundGrid, axis: int) -> np.ndarray:
    """Face matrix sum_j h^(2j-1) int_F [d^j_n u][d^j_n v] for the face between a cell (left) and its neighbour
    along ``axis`` (right), j = 1..order (Burman, C. R. Acad. Sci. Paris 2010): (2 nb, 2 nb)."""
    d, p = grid.d, grid.order
    q, w = basis.gauss_cell(p + 1, d - 1)
    area = float(np.prod(np.delete(grid.h, axis)))
    hn = float(grid.h[axis])
    nb = grid.n_local
    G = np.zeros((2 * nb, 2 * nb))
    for j in range(1, p + 1):
        xl = np.insert(q, axis, 1.0, axis=1)
        xr = np.insert(q, axis, 0.0, axis=1)
        a = np.concatenate([basis.normal_derivative(p, xl, axis, j), -basis.normal_derivative(p, xr, axis, j)],
                           axis=1) / hn ** j
        G += hn ** (2 * j - 1) * np.einsum("q,qi,qj->ij", w * area, a, a)
    return G


def ghost_matrix(disc: Discretization, ncomp: int = 1) -> sparse.csr_matrix:
    """Ghost penalty (unit coefficient) over all ghost faces, for ``ncomp`` components."""
    n = disc.n_dofs
    out = sparse.csr_matrix((n * ncomp, n * ncomp))
    for k, pairs in enumerate(ghost_faces(disc.quad, disc.dofs)):
        if not len(pairs):
            continue
        G = ghost_reference(disc.grid, k)
        if ncomp > 1:
            G = np.kron(G, np.eye(ncomp))
        cd = np.concatenate([disc.dofs.dofs_of(pairs[:, 0]), disc.dofs.dofs_of(pairs[:, 1])], axis=1)
        out = out + scatter(cd, np.broadcast_to(G, (len(pairs),) + G.shape), n, ncomp)
    return out


def condition_number(A: sparse.spmatrix, scaled: bool = True) -> dict:
    """Spectral condition number of a symmetric positive definite matrix (Lanczos for the largest eigenvalue,
    shift-invert for the smallest), raw and after symmetric diagonal (Jacobi) scaling."""
    from scipy.sparse.linalg import eigsh

    def kappa(M):
        M = sparse.csc_matrix(M)
        lmax = float(eigsh(M, k=1, which="LA", return_eigenvectors=False, tol=1e-6, maxiter=5000)[0])
        try:
            lmin = float(eigsh(M, k=1, sigma=0.0, which="LM", return_eigenvectors=False, tol=1e-8)[0])
        except Exception:                      # numerically singular: report it as such
            lmin = 0.0
        return lmax / lmin if lmin > 0 else float("inf"), lmin, lmax

    out = {}
    k, lmin, lmax = kappa(A)
    out.update(cond=k, lambda_min=lmin, lambda_max=lmax)
    if scaled:
        dg = A.diagonal()
        D = sparse.diags(1 / np.sqrt(np.abs(np.where(dg == 0, 1, dg))))
        out["cond_jacobi"] = kappa(D @ A @ D)[0]
    return out


def solve_linear(A: sparse.spmatrix, b: np.ndarray, solver: str = "auto", tol: float = 1e-10,
                 near_null: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Direct sparse solve (SuperLU) or conjugate gradients with smoothed-aggregation AMG (pyamg)."""
    t0 = time.perf_counter()
    A = sparse.csr_matrix(A)
    if solver == "auto":                       # sparse LU fills in badly on 3-D couplings: AMG-CG beyond small systems
        solver = "direct" if A.nnz <= 300_000 else "amg"
    info = {"solver": solver}
    if solver == "amg":
        try:
            import pyamg
        except ImportError:
            solver = info["solver"] = "direct"
    if solver == "direct":
        from scipy.sparse.linalg import spsolve
        x = spsolve(sparse.csc_matrix(A), b)
    else:
        import pyamg
        from scipy.sparse.linalg import cg
        ml = pyamg.smoothed_aggregation_solver(A, B=near_null, max_coarse=500)
        res = []
        x, flag = cg(A, b, rtol=tol, maxiter=5000, M=ml.aspreconditioner(),
                     callback=lambda xk: res.append(1))
        info.update(iterations=len(res), converged=flag == 0)
    info["residual"] = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    info["seconds"] = time.perf_counter() - t0
    return x, info
