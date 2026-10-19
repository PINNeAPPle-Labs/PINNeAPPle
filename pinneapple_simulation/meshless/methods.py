"""Five ways to solve the same 2-D problem on the same domain.

==========  ===================================================================================================
``kansa``   global RBF collocation (Kansa 1990): multiquadric φ = sqrt(1 + (ε r)²) centred at every node; dense
``rbf_fd``  RBF-generated finite differences: polyharmonic spline r³ + polynomials up to ``degree`` on the ``n``
            nearest nodes of each node (Flyer, Fornberg, Bayona, Barnett 2016); sparse
``fd``      5-point finite differences on a Cartesian grid embedded in the domain, Shortley-Weller formula where
            the grid meets the curved boundary (the boundary intersections are the boundary nodes); sparse
``fem``     linear (P1) finite elements on the Delaunay triangulation of the nodes, cells outside removed; sparse
``pinn``    a physics-informed network trained on the same nodes within a wall-clock budget
            (``pinneapple_analysis.cost.train_within_budget``)
==========  ===================================================================================================

``discretize(method, domain, nodes)`` builds the operator (setup cost measured), ``solve(problem, disc)`` solves the
steady problem or marches the heat equation (Crank-Nicolson) and returns the nodal values, the solve time, a 1-norm
condition-number estimate of the system matrix and the noise amplification (relative change of the solution per
relative perturbation of the right-hand side). ``disc.evaluate(u, P)`` gives the solution anywhere inside, with each
method's own interpolant (global RBF, local RBF-FD interpolation, P1, the network).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .domains import Domain2D, NodeSet
from .problems import Problem

__all__ = ["METHODS", "Discretization", "discretize", "solve", "rbf_fd_weights"]

METHODS = {
    "kansa": {"label": "Kansa (global RBF)", "family": "meshless", "matrix": "dense"},
    "rbf_fd": {"label": "RBF-FD", "family": "meshless", "matrix": "sparse"},
    "pinn": {"label": "PINN", "family": "meshless (neural)", "matrix": "none"},
    "fd": {"label": "Finite differences (embedded grid)", "family": "mesh-based", "matrix": "sparse"},
    "fem": {"label": "Finite elements (P1)", "family": "mesh-based", "matrix": "sparse"},
}
KANSA_MAX_NODES = 4000


@dataclass
class Discretization:
    method: str
    nodes: NodeSet
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray]
    setup_s: float
    L: Any = None                       # nodal Laplacian (rows of interior nodes)
    K: Any = None                       # FEM stiffness
    M: Any = None                       # FEM mass
    kansa: dict[str, np.ndarray] | None = None
    info: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def n_unknowns(self) -> int:
        return self.nodes.n


# ------------------------------------------------------------------------------------------------ Kansa
def _mq(r, eps):
    return np.sqrt(1.0 + (eps * r) ** 2)


def _mq_lap(r, eps):
    s = 1.0 + (eps * r) ** 2
    return eps ** 2 * (1.0 + s) / s ** 1.5           # 2-D Laplacian of sqrt(1 + (eps r)^2)


def _dist(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def default_eps(domain: Domain2D) -> float:
    """Shape parameter of the multiquadric: 3 / diameter (a fixed, not node-scaled, value: spectral convergence until
    the condition number stops it)."""
    return 3.0 / domain.diameter


def _kansa(domain, nodes, eps=None, **_):
    if nodes.n > KANSA_MAX_NODES:
        raise ValueError(f"Kansa uses dense {nodes.n}x{nodes.n} matrices; this server stops at {KANSA_MAX_NODES} nodes")
    eps = float(eps or default_eps(domain))
    X = nodes.X
    R = _dist(X, X)
    Phi, LPhi = _mq(R, eps), _mq_lap(R, eps)

    def evaluate(coef, P):
        out = np.empty(len(P))
        for i in range(0, len(P), 4000):
            out[i:i + 4000] = _mq(_dist(P[i:i + 4000], X), eps) @ coef
        return out
    return {"kansa": {"Phi": Phi, "LPhi": LPhi}, "evaluate": evaluate, "info": {"eps": eps, "rbf": "multiquadric",
                                                                                 "nnz": int(Phi.size)}}


# ------------------------------------------------------------------------------------------------ RBF-FD
def _monomials(deg: int):
    return [(a, d - a) for d in range(deg + 1) for a in range(d, -1, -1)]


def rbf_fd_weights(X: np.ndarray, Q: np.ndarray, idx: np.ndarray, op: str = "lap", degree: int = 4) -> np.ndarray:
    """RBF-FD weights (q, n) of the stencils ``idx`` (q, n) (rows of ``X``) for the operator ``op`` at the points
    ``Q``: ``lap`` (the Laplacian) or ``eval`` (interpolation). PHS r³ + all monomials up to ``degree``, in local
    coordinates centred at the point and scaled by the stencil radius."""
    mons = _monomials(degree)
    m = len(mons)
    q, n = idx.shape
    if n < m + 1:
        raise ValueError(f"stencils of {n} nodes are too small for degree {degree} ({m} monomials)")
    W = np.empty((q, n))
    for s in range(0, q, 2000):
        st = idx[s:s + 2000]
        Z = X[st] - Q[s:s + 2000, None, :]                              # (b, n, 2)
        sc = np.sqrt((Z ** 2).sum(-1)).max(1)[:, None, None] + 1e-300
        Z = Z / sc
        b = len(st)
        D = np.sqrt(((Z[:, :, None, :] - Z[:, None, :, :]) ** 2).sum(-1))
        A = np.zeros((b, n + m, n + m))
        A[:, :n, :n] = D ** 3
        P = np.stack([Z[..., 0] ** a * Z[..., 1] ** c for a, c in mons], -1)   # (b, n, m)
        A[:, :n, n:] = P
        A[:, n:, :n] = np.transpose(P, (0, 2, 1))
        rhs = np.zeros((b, n + m))
        r0 = np.sqrt((Z ** 2).sum(-1))                                 # distances to the evaluation point (origin)
        if op == "lap":
            rhs[:, :n] = 9.0 * r0                                      # Δ r³ = 9 r in 2-D
            for j, (a, c) in enumerate(mons):                          # Δ of x^a y^c at the origin
                if (a, c) in ((2, 0), (0, 2)):
                    rhs[:, n + j] = 2.0
        elif op == "eval":
            rhs[:, :n] = r0 ** 3
            rhs[:, n] = 1.0                                            # only the constant monomial is 1 at 0
        else:
            raise ValueError("op is lap or eval")
        w = np.linalg.solve(A, rhs[..., None])[..., 0][:, :n]
        W[s:s + 2000] = w / (sc[:, 0] ** 2 if op == "lap" else 1.0)
    return W


def _rbf_fd(domain, nodes, degree=4, stencil=None, **_):
    X = nodes.X
    m = len(_monomials(degree))
    n = int(stencil or 2 * m + 1)
    n = min(n, nodes.n)
    tree = cKDTree(X)
    inner = np.flatnonzero(nodes.interior)
    _, idx = tree.query(X[inner], k=n)
    W = rbf_fd_weights(X, X[inner], idx, "lap", degree)
    L = sps.csr_matrix((W.ravel(), (np.repeat(inner, n), idx.ravel())), shape=(nodes.n, nodes.n))

    def evaluate(u, P):
        _, st = tree.query(P, k=n)
        return (rbf_fd_weights(X, P, st, "eval", degree) * u[st]).sum(1)
    return {"L": L, "evaluate": evaluate, "info": {"degree": degree, "stencil": n, "rbf": "phs r^3",
                                                  "nnz": int(L.nnz)}}


# ------------------------------------------------------------------------------------------------ FD (Shortley-Weller)
def _crossing(sdf, P, e, hmax):
    """Distance from inside points P along direction e to the boundary (bisection on the signed distance)."""
    lo, hi = np.zeros(len(P)), np.full(len(P), hmax)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        inside = sdf(P + mid[:, None] * e) < 0
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    return 0.5 * (lo + hi)


def fd_nodes(domain: Domain2D, h: float):
    """Grid nodes inside the domain and the points where grid lines cross the boundary."""
    if not domain.has_geometry:
        raise ValueError("embedded finite differences need the geometry (a signed distance), not a point cloud")
    x = np.arange(domain.lo[0], domain.hi[0] + 0.5 * h, h)
    y = np.arange(domain.lo[1], domain.hi[1] + 0.5 * h, h)
    gi, gj = np.meshgrid(np.arange(len(x)), np.arange(len(y)), indexing="ij")
    G = np.c_[x[gi.ravel()], y[gj.ravel()]]
    ins = domain.sdf(G) < -1e-12 * h
    num = -np.ones(G.shape[0], int)
    num[ins] = np.arange(ins.sum())
    return G, ins, num, (len(x), len(y)), gi.ravel(), gj.ravel()


def _fd(domain, nodes, **_):
    h = nodes.h
    G, ins, num, (nx, ny), gi, gj = fd_nodes(domain, h)
    Pin = G[ins]
    n_in = len(Pin)
    rows, cols, vals = [], [], []
    bpts = []
    dist = {}
    for name, (dx, dy) in {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}.items():
        ii, jj = gi[ins] + dx, gj[ins] + dy
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        nb = np.full(n_in, -1)
        flat = ii[ok] * ny + jj[ok]
        nb[ok] = num[flat]
        d = np.full(n_in, h)
        out = nb < 0
        if out.any():
            e = np.array([dx, dy], float)
            d[out] = np.maximum(_crossing(domain.sdf, Pin[out], e, h), 1e-6 * h)
            bpts.append(Pin[out] + d[out][:, None] * e)
        dist[name] = (nb, d, out)
    B = np.concatenate(bpts) if bpts else np.zeros((0, 2))
    # merge boundary points that coincide (a grid node on both a horizontal and a vertical crossing)
    if len(B):
        key = np.round(B / (1e-9 * h + 1e-15)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        Bu = B[first]
    else:
        Bu, inv = B, np.zeros(0, int)
    cursor = 0
    bid = {}
    for name in ("E", "W", "N", "S"):
        nb, d, out = dist[name]
        k = int(out.sum())
        b = np.full(n_in, -1)
        b[out] = n_in + inv.ravel()[cursor:cursor + k]
        cursor += k
        bid[name] = np.where(out, b, nb)
    me = np.arange(n_in)
    for a, b_ in (("E", "W"), ("N", "S")):
        hp, hm = dist[a][1], dist[b_][1]
        cp = 2.0 / (hp * (hp + hm))
        cm = 2.0 / (hm * (hp + hm))
        rows += [me, me, me]
        cols += [bid[a], bid[b_], me]
        vals += [cp, cm, -(cp + cm)]
    N = n_in + len(Bu)
    L = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    X = np.vstack([Pin, Bu])
    nodes_fd = NodeSet(X, np.r_[np.zeros(n_in, bool), np.ones(len(Bu), bool)], h)
    mesh = _p1_mesh(domain, nodes_fd)

    def evaluate(u, P):
        return mesh["mesh"].interpolate(u[mesh["map"]], P, fill="nearest")[:, 0]
    return {"L": L, "nodes": nodes_fd, "evaluate": evaluate, "info": {"nnz": int(L.nnz), "grid_h": h}}


# ------------------------------------------------------------------------------------------------ FEM P1
def _p1_mesh(domain, nodes):
    from pinneapple_core import Mesh
    X = nodes.X
    m = Mesh.from_points(X, keep=(lambda c: domain.sdf(c) < 0) if domain.has_geometry else None)
    if not domain.has_geometry:                       # point cloud: drop the long cells spanning the convex hull
        e = np.stack([np.linalg.norm(m.points[m.cells[:, i]] - m.points[m.cells[:, (i + 1) % 3]], axis=1)
                      for i in range(3)], 1).max(1)
        m = Mesh(m.points, m.cells[e < 3 * nodes.h])
    _, mp = cKDTree(X).query(m.points)
    return {"mesh": m, "map": mp}


def _fem(domain, nodes, **_):
    pm = _p1_mesh(domain, nodes)
    m, mp = pm["mesh"], pm["map"]
    used = np.zeros(nodes.n, bool)
    used[mp] = True
    if not used.all():                                # nodes no triangle touches: drop them
        keep = np.flatnonzero(used)
        nodes = NodeSet(nodes.X[keep], nodes.boundary[keep], nodes.h)
        pm = _p1_mesh(domain, nodes)
        m, mp = pm["mesh"], pm["map"]
    g, vol, C = m.grad_lambda, m.cell_volumes, mp[m.cells]
    Ke = np.einsum("mid,mjd->mij", g, g) * vol[:, None, None]
    Me = (np.ones((3, 3)) + np.eye(3))[None] / 12.0 * vol[:, None, None]
    r = np.repeat(C, 3, axis=1).ravel()
    c = np.tile(C, (1, 3)).ravel()
    K = sps.csr_matrix((Ke.ravel(), (r, c)), shape=(nodes.n, nodes.n))
    M = sps.csr_matrix((Me.ravel(), (r, c)), shape=(nodes.n, nodes.n))

    def evaluate(u, P):
        return m.interpolate(u[mp], P, fill="nearest")[:, 0]
    q = _quality(m)
    return {"K": K, "M": M, "nodes": nodes, "evaluate": evaluate,
            "info": {"nnz": int(K.nnz), "cells": int(m.n_cells), "min_angle_deg": q}}


def _quality(m) -> float:
    P = m.points[m.cells]
    ang = []
    for i in range(3):
        a, b, c = P[:, i], P[:, (i + 1) % 3], P[:, (i + 2) % 3]
        u, v = b - a, c - a
        ang.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)),
                                                -1, 1))))
    return float(np.min(ang))


# ------------------------------------------------------------------------------------------------ dispatch
_BUILD = {"kansa": _kansa, "rbf_fd": _rbf_fd, "fd": _fd, "fem": _fem}


def discretize(method: str, domain: Domain2D, nodes: NodeSet, **options) -> Discretization:
    """Build the operator of ``method`` on ``nodes`` (FD builds its own grid nodes at the same spacing)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; methods: {list(METHODS)}")
    if method == "pinn":
        return Discretization("pinn", nodes, lambda u, P: u(P), 0.0, options=options)
    t0 = time.perf_counter()
    out = _BUILD[method](domain, nodes, **options)
    setup = time.perf_counter() - t0
    return Discretization(method, out.pop("nodes", nodes), out.pop("evaluate"), setup, L=out.get("L"),
                          K=out.get("K"), M=out.get("M"), kansa=out.get("kansa"), info=out.get("info", {}),
                          options=options)


class _Factor:
    """LU of a sparse or dense matrix, with transpose solves (for the condition estimate)."""

    def __init__(self, A):
        self.A = A
        self.sparse = sps.issparse(A)
        if self.sparse:
            self.lu = spla.splu(sps.csc_matrix(A))
        else:
            import scipy.linalg as sla
            self.lu = sla.lu_factor(A)

    def solve(self, b, trans=False):
        if self.sparse:
            return self.lu.solve(b, trans="T" if trans else "N")
        import scipy.linalg as sla
        return sla.lu_solve(self.lu, b, trans=1 if trans else 0)

    def cond1(self) -> float:
        n = self.A.shape[0]
        inv = spla.LinearOperator((n, n), matvec=lambda v: self.solve(np.asarray(v).ravel()),
                                  rmatvec=lambda v: self.solve(np.asarray(v).ravel(), trans=True), dtype=float)
        a = spla.onenormest(sps.csc_matrix(self.A)) if self.sparse else float(np.abs(self.A).sum(0).max())
        return float(a * spla.onenormest(inv))


def _rows(mask, A):
    D = sps.diags(mask.astype(float))
    return D @ A if sps.issparse(A) else mask[:, None] * A


def _noise(fac, rhs, u, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(len(rhs))
    d *= 1e-8 * np.linalg.norm(rhs) / np.linalg.norm(d)
    du = fac.solve(rhs + d) - u
    return float((np.linalg.norm(du) / max(np.linalg.norm(u), 1e-300)) / 1e-8)


def solve(problem: Problem, disc: Discretization, *, seed: int = 0) -> dict[str, Any]:
    """Solve on the discretization; returns ``u`` (nodal values, or the network for the PINN) and the costs."""
    if disc.method == "pinn":
        from .pinn import train_pinn
        return train_pinn(problem, disc, seed=seed, **disc.options)
    nodes = disc.nodes
    X, b, inner = nodes.X, nodes.boundary, nodes.interior
    N = nodes.n
    Id = sps.identity(N, format="csr")
    t0 = time.perf_counter()
    c = problem.c
    if not problem.time_dependent:
        if disc.kansa is not None:
            Phi, LPhi = disc.kansa["Phi"], disc.kansa["LPhi"]
            A = inner[:, None] * (-LPhi - c * Phi) + b[:, None] * Phi
            rhs = np.where(b, problem.g(X), problem.f(X))
            fac = _Factor(A)
            coef = fac.solve(rhs)
            u = Phi @ coef
        elif disc.K is not None:
            A = _rows(inner, disc.K - c * disc.M) + sps.diags(b.astype(float))
            rhs = np.where(b, problem.g(X), disc.M @ problem.f(X))
            fac = _Factor(A)
            u = fac.solve(rhs)
        else:
            A = _rows(inner, -disc.L - c * Id) + sps.diags(b.astype(float))
            rhs = np.where(b, problem.g(X), problem.f(X))
            fac = _Factor(A)
            u = fac.solve(rhs)
        out = {"u": u, "solve_s": time.perf_counter() - t0}
        out["evaluate"] = _evaluator(disc, u, coef if disc.kansa is not None else None)
        out["cond"] = fac.cond1()
        out["noise_amplification"] = _noise(fac, rhs, u, seed)
        return out
    # heat: Crank-Nicolson, u = g on the boundary at every step
    dt, a, th = problem.t_end / problem.n_steps, problem.alpha, 0.5
    if disc.kansa is not None:
        Phi = disc.kansa["Phi"]
        L = np.linalg.solve(Phi.T, disc.kansa["LPhi"].T).T              # nodal Laplacian L = LPhi Phi^-1
        Id = np.eye(N)
        A = inner[:, None] * (Id - th * dt * a * L) + np.diag(b.astype(float))
        Bm = Id + (1 - th) * dt * a * L
        Mf = None
    elif disc.K is not None:
        A = _rows(inner, disc.M + th * dt * a * disc.K) + sps.diags(b.astype(float))
        Bm = disc.M - (1 - th) * dt * a * disc.K
        Mf = disc.M
    else:
        A = _rows(inner, Id - th * dt * a * disc.L) + sps.diags(b.astype(float))
        Bm = Id + (1 - th) * dt * a * disc.L
        Mf = None
    fac = _Factor(A)
    u = problem.u0(X)
    f0 = problem.f(X, 0.0)
    rhs = None
    for k in range(problem.n_steps):
        t1 = (k + 1) * dt
        f1 = problem.f(X, t1)
        src = dt * (th * f1 + (1 - th) * f0)
        rhs = Bm @ u + (Mf @ src if Mf is not None else src)
        rhs = np.where(b, problem.g(X, t1), rhs)
        u = fac.solve(rhs)
        f0 = f1
    out = {"u": u, "solve_s": time.perf_counter() - t0, "cond": fac.cond1(),
           "noise_amplification": _noise(fac, rhs, fac.solve(rhs), seed)}
    out["evaluate"] = _evaluator(disc, u, np.linalg.solve(disc.kansa["Phi"], u) if disc.kansa is not None else None)
    return out


def _evaluator(disc, u, coef=None):
    """The solution at any points inside: the global RBF expansion (Kansa) or the method's interpolant of u."""
    if coef is not None:
        return lambda P: disc.evaluate(coef, P)
    return lambda P: disc.evaluate(u, P)

