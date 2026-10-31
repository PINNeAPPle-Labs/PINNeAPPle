"""Incompressible flow on an immersed discretisation: steady Stokes and Navier-Stokes around a body given as an SDF,
a polygon (2-D) or an STL surface (3-D), on a Cartesian grid that the body cuts. No body-fitted mesh.

Unknowns: Q2 velocity and Q1 pressure (Taylor-Hood on the background grid). The grid covers the flow box exactly,
so the box faces carry ordinary conditions (prescribed velocity, wall, slip, do-nothing outflow); the body surface
cuts the cells and its no-slip (or prescribed-velocity) condition is imposed by Nitsche's method. Weak form, with
``n`` the outward normal of the fluid on the body surface Gamma and ``g`` the body velocity:

    nu (grad u, grad v) + ((w . grad) u, v) - (p, div v) - (q, div u)
      - <nu du/dn - p n, v> - <nu dv/dn + q n, u - g> + <sigma (u - g), v> + s_u(u, v) - s_p(p, q) = (f, v)

``sigma = beta nu / h + U`` (Nitsche penalty with a convective part), ``s_u`` the ghost penalty on the velocity
derivatives across the faces of cut cells scaled by ``nu + U h`` and ``s_p`` the ghost penalty on the pressure
gradient scaled by ``h^2 / (nu + U h)`` (Burman & Hansbo 2014; Massing, Larson, Logg & Rognes, J. Sci. Comput. 61,
2014), so that the inf-sup constant and the condition number do not depend on how the body cuts the cells. ``U`` is a
reference velocity of the flow (0 for Stokes).

Navier-Stokes is solved by Newton's method from the Stokes solution. The force on the body is computed by the
variational (reaction) form: the momentum residual tested with ``e_k phi`` where ``phi`` = 1 near the body and 0 on
the box faces (Babuska & Miller 1984); the Nitsche surface flux is reported next to it.

    from pinneapple_simulation.immersed import flow
    body = flow.PolygonGeometry(flow.regular_polygon((0.2, 0.2), 0.05, 512))
    fd = flow.FlowDiscretization(body, ((0, 0), (2.2, 0.41)), h=0.1 / 16)
    sol = flow.solve_navier_stokes(fd, nu=1e-3, bcs={"x-": flow.Velocity(inflow), "y-": "wall", "y+": "wall",
                                                     "x+": "outflow"}, u_ref=0.2)
    sol.force(), sol.pressure(points), sol.velocity(points)
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from . import basis
from .assembly import Discretization, SurfaceMeasures, ghost_faces, ghost_reference
from .geometry import Geometry, SDFGeometry
from .grid import BackgroundGrid, DofMap

__all__ = ["Velocity", "FlowDiscretization", "FlowSolution", "PolygonGeometry", "regular_polygon",
           "solve_stokes", "solve_navier_stokes", "assemble_flow"]

FACES = ("x-", "x+", "y-", "y+", "z-", "z+")


@dataclass
class Velocity:
    """Prescribed velocity on a box face: a constant vector or ``fn(X) -> (n, d)``."""

    value: Any


# ------------------------------------------------------------------------------------------------ 2-D polygons
def regular_polygon(center: Sequence[float], radius: float, n: int) -> np.ndarray:
    """Vertices (n, 2), counter-clockwise, of the regular n-gon inscribed in a circle."""
    t = 2 * np.pi * np.arange(n) / n
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


class PolygonGeometry(Geometry):
    """A closed polyline (the 2-D counterpart of an STL surface): exact signed distance to its segments, negative
    inside (even-odd rule)."""

    default_depth = 2
    default_refine = 0         # the distance to a segment is piecewise linear along a cell edge: no root refinement

    def __init__(self, V: np.ndarray, chunk: int = 4096):
        self.V = np.asarray(V, float)
        if self.V.ndim != 2 or self.V.shape[1] != 2 or len(self.V) < 3:
            raise ValueError("a polygon needs (n >= 3, 2) vertices")
        self.A, self.B = self.V, np.roll(self.V, -1, axis=0)
        self.dim, self.lipschitz, self.chunk = 2, 1.0, int(chunk)

    @property
    def bounds(self):
        return self.V.min(axis=0), self.V.max(axis=0)

    def area(self) -> float:
        x, y = self.V.T
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def phi(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        out = np.empty(len(X))
        A, E = self.A, self.B - self.A
        ee = np.einsum("ij,ij->i", E, E)
        step = max(1, self.chunk * 256 // len(A))
        for s in range(0, len(X), step):
            P = X[s:s + step, None, :] - A[None]                                     # (m, ns, 2)
            t = np.clip(np.einsum("mij,ij->mi", P, E) / ee, 0.0, 1.0)
            dist = np.sqrt(np.min(np.sum((P - t[..., None] * E[None]) ** 2, axis=2), axis=1))
            y = X[s:s + step, 1][:, None]
            cross = (A[None, :, 1] > y) != (self.B[None, :, 1] > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = A[None, :, 0] + (y - A[None, :, 1]) * E[None, :, 0] / E[None, :, 1]
            inside = (np.count_nonzero(cross & (X[s:s + step, 0][:, None] < xint), axis=1) % 2) == 1
            out[s:s + step] = np.where(inside, -dist, dist)
        return out


# ------------------------------------------------------------------------------------------------ discretisation
class FlowDiscretization:
    """Background grid that fits the flow box exactly (per-axis cell size close to ``h``), cut by the body.

    ``body``: the solid (inside = solid) as any :class:`Geometry`; or ``fluid``: the fluid itself (inside = fluid),
    for flows bounded by immersed walls. ``depth`` / ``refine``: octree depth and root refinement of the cut-cell
    quadrature (the body's defaults when not given); ``n_simplex`` / ``n_facet``: Gauss points per direction on the
    clipped simplices and boundary facets (3: exact for the Q2 integrands on straight cuts; the degree-2 rules the
    heat solver uses by default leave a 1e-5 to 1e-7 error in Poiseuille flow between cut walls, 3 gives 1e-13).
    """

    def __init__(self, body: Geometry | None, box, h: float | Sequence[float], fluid: Geometry | None = None,
                 depth: int | None = None, refine: int | None = None, n_simplex: int = 3, n_facet: int = 3):
        t0 = time.perf_counter()
        lo, hi = (np.asarray(b, float) for b in box)
        d = len(lo)
        hv = np.broadcast_to(np.asarray(h, float), (d,))
        n = tuple(int(max(1, round(L))) for L in (hi - lo) / hv)
        hv = (hi - lo) / np.asarray(n)
        self.box, self.d, self.body = (lo, hi), d, body
        self.grid = BackgroundGrid(lo, hv, n, 2)
        self.grid_p = BackgroundGrid(lo, hv, n, 1)
        if fluid is None:
            if body is None:
                big = 1e3 * float(np.max(hi - lo))
                fluid = SDFGeometry(lambda X: np.full(len(X), -big), (lo, hi))
            else:
                fluid = SDFGeometry(lambda X, g=body: -g.phi(X), (lo, hi), lipschitz=getattr(body, "lipschitz", 1.0),
                                    name="fluid")
                for k in ("default_depth", "default_refine"):
                    if hasattr(body, k):
                        setattr(fluid, k, getattr(body, k))
        self.fluid = fluid
        self.disc = Discretization(fluid, h=float(hv.max()), order=2, depth=depth, refine=refine,
                                   n_simplex=n_simplex, grid=self.grid, n_facet=n_facet)
        self.pdofs = DofMap.build(self.grid_p, self.disc.quad.active_cells)
        self.timings = {"quadrature_s": time.perf_counter() - t0}

    @property
    def n_u(self) -> int:
        return self.disc.n_dofs

    @property
    def n_p(self) -> int:
        return self.pdofs.n_dofs

    @property
    def n_unknowns(self) -> int:
        return self.d * self.n_u + self.n_p

    @property
    def h(self) -> float:
        return float(np.max(self.grid.h))

    def summary(self) -> dict:
        return {**self.disc.summary(), "cells": self.grid.n, "h_axes": self.grid.h.tolist(),
                "velocity_dofs": self.d * self.n_u, "pressure_dofs": self.n_p, "unknowns": self.n_unknowns,
                "quadrature_s": self.timings["quadrature_s"]}

    # ---------------------------------------------------------------- evaluation
    def evaluate_velocity(self, x: np.ndarray, X: np.ndarray, gradient: bool = False) -> np.ndarray:
        return self.disc.evaluate(x[:self.d * self.n_u], X, gradient=gradient, ncomp=self.d)

    def evaluate_pressure(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        cell, xi, ok = self.grid_p.locate(X)
        row = self.pdofs.index[cell]
        ok &= row >= 0
        out = np.full(len(X), np.nan)
        if ok.any():
            P = x[self.d * self.n_u:]
            out[ok] = np.einsum("nb,nb->n", basis.shape(1, xi[ok]), P[self.pdofs.cell_dofs[row[ok]]])
        return out


# ------------------------------------------------------------------------------------------------ assembly helpers
def _kron_eye(M: np.ndarray, d: int) -> np.ndarray:
    """(m, nb, nb) scalar blocks -> (m, nb d, nb d) with the identity on the interleaved components."""
    m, a, b = M.shape
    out = np.zeros((m, a, d, b, d))
    for i in range(d):
        out[:, :, i, :, i] = M
    return out.reshape(m, a * d, b * d)


def _vdofs(cell_dofs: np.ndarray, d: int) -> np.ndarray:
    return (cell_dofs[:, :, None] * d + np.arange(d)).reshape(len(cell_dofs), -1)


class _COO:
    """Accumulates element blocks (rows (m, r), cols (m, c), values (m, r, c)) into one sparse matrix."""

    def __init__(self, n: int):
        self.n, self.r, self.c, self.v = n, [], [], []

    def add(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, scale: float = 1.0) -> None:
        if not len(vals):
            return
        R = np.broadcast_to(rows[:, :, None], vals.shape)
        C = np.broadcast_to(cols[:, None, :], vals.shape)
        self.r.append(R.ravel())
        self.c.append(C.ravel())
        self.v.append((scale * vals).ravel())

    def matrix(self) -> sparse.csr_matrix:
        if not self.v:
            return sparse.csr_matrix((self.n, self.n))
        return sparse.csr_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                                 shape=(self.n, self.n))


def _cell_sets(fd: FlowDiscretization):
    """The two volume rules: Gauss (4 points per axis) on the cells entirely inside, the moment-fitted rule on cut
    cells. Each: (cells, xi (nq, d), weights (m, nq), N (nq, nb), G (nq, nb, d), Np (nq, nbp), Gp (nq, nbp, d))."""
    disc, grid = fd.disc, fd.grid
    vol = float(np.prod(grid.h))
    out = []
    inside = disc.quad.inside_cells
    if len(inside):
        q, w = basis.gauss_cell(4, fd.d)
        out.append((inside, q, np.broadcast_to(w * vol, (len(inside), len(w)))))
    cut, W = disc.cut_weights()
    if len(cut):
        out.append((cut, disc.fitted_reference[2], W))
    sets = []
    for cells, xi, W in out:
        sets.append((cells, xi, W, basis.shape(2, xi), basis.grad(2, xi) / grid.h, basis.shape(1, xi),
                     basis.grad(1, xi) / grid.h))
    return sets


def _linear_blocks(fd: FlowDiscretization, sets, nu: float, coo: _COO, vol_only: _COO | None = None) -> None:
    """nu (grad u, grad v), -(p, div v), -(q, div u) over all active cells."""
    d, nu_off = fd.d, fd.d * fd.n_u
    for cells, _, W, N, G, Np, _Gp in sets:
        ud = _vdofs(fd.disc.dofs.dofs_of(cells), d)
        pd = fd.pdofs.dofs_of(cells) + nu_off
        nq, nb = N.shape
        K = (W @ np.einsum("qak,qbk->qab", G, G).reshape(nq, -1)).reshape(len(cells), nb, nb)
        Kv = _kron_eye(K, d)
        # (div v) q: rows (a, i) of velocity, cols pressure b: - sum_q W d_i N_a Np_b
        D = (W @ np.einsum("qai,qb->qaib", G, Np).reshape(nq, -1)).reshape(len(cells), nb * d, Np.shape[1])
        for target in (coo, vol_only):
            if target is None:
                continue
            target.add(ud, ud, Kv, nu)
            target.add(ud, pd, D, -1.0)
            target.add(pd, ud, D.transpose(0, 2, 1), -1.0)


def _convection(fd: FlowDiscretization, sets, x: np.ndarray, coo_c: _COO, coo_d: _COO | None, chunk: int = 4000
                ) -> None:
    """((w . grad) u, v) into ``coo_c`` and its Newton part ((u . grad) w, v) into ``coo_d``, w = the velocity of x."""
    d = fd.d
    U = x[:d * fd.n_u].reshape(-1, d)
    for cells, _, Wall, N, G, *_ in sets:
        for s in range(0, len(cells), chunk):
            cs, W = cells[s:s + chunk], Wall[s:s + chunk]
            cd = fd.disc.dofs.dofs_of(cs)
            Uc = U[cd]                                                        # (m, nb, d)
            w = np.einsum("qb,mbi->mqi", N, Uc)
            wg = np.einsum("mqk,qbk->mqb", w, G)                              # w . grad N_b
            S = np.einsum("mq,qa,mqb->mab", W, N, wg)
            ud = _vdofs(cd, d)
            coo_c.add(ud, ud, _kron_eye(S, d))
            if coo_d is not None:
                gw = np.einsum("qbj,mbi->mqij", G, Uc)                        # d_j w_i
                M = np.einsum("mq,qa,qb,mqij->maibj", W, N, N, gw).reshape(len(cs), N.shape[1] * d, -1)
                coo_d.add(ud, ud, M)


def _load(fd: FlowDiscretization, sets, f) -> np.ndarray:
    b = np.zeros(fd.n_unknowns)
    if f is None:
        return b
    d = fd.d
    for cells, xi, W, N, *_ in sets:
        X = fd.grid.to_physical(np.repeat(cells, len(xi)), np.tile(xi, (len(cells), 1)))
        F = np.asarray(f(X), float).reshape(len(cells), len(xi), d)
        vals = np.einsum("mq,qa,mqi->mai", W, N, F).reshape(len(cells), -1)
        np.add.at(b, _vdofs(fd.disc.dofs.dofs_of(cells), d).ravel(), vals.ravel())
    return b


def _value(v, X: np.ndarray, d: int) -> np.ndarray:
    if callable(v):
        return np.asarray(v(X), float).reshape(len(X), d)
    return np.broadcast_to(np.asarray(v, float), (len(X), d)).copy()


def assemble_flow(fd: FlowDiscretization, nu: float, u_ref: float = 0.0, body_velocity=None, force=None,
                  beta: float = 100.0, ghost_u: float = 0.1, ghost_p: float = 0.1) -> dict:
    """The linear part of the system (viscous, pressure coupling, Nitsche on the body, ghost penalties), the load
    and the pieces the force evaluation needs. Unknowns: velocity interleaved by node, then pressure.

    Ghost coefficients: 0.1 on the velocity and on the pressure. Measured on a channel whose wall cuts a cell at
    fluid fractions 0.5 to 1e-4 (h = 1/8, Q2/Q1): with 0.1 / 0.1 the condition number stays at 2.2e6-2.3e6; with
    the heat solver's Q2 value 0.01 it grows from 3.6e6 to 1.8e7 at 1e-4, and the velocity block alone (diagonally
    scaled) from 3e2 to 5e5 (tests/test_immersed_flow.py)."""
    t0 = time.perf_counter()
    d, disc, n = fd.d, fd.disc, fd.n_unknowns
    nu_off = d * fd.n_u
    h = fd.h
    sets = _cell_sets(fd)
    coo, vol = _COO(n), _COO(n)
    _linear_blocks(fd, sets, nu, coo, vol)
    b = _load(fd, sets, force)
    b_vol = b.copy()
    sigma = beta * nu / h + u_ref
    S = disc.quad.surface
    info = {"nitsche_sigma": sigma, "surface_points": len(S)}
    if len(S):
        Xs = disc.physical(S)
        g = np.zeros((len(S), d)) if body_velocity is None else _value(body_velocity, Xs, d)
        nrm = S.normal
        w = {"1": np.ones(len(S))}
        for a in range(d):
            w[f"n{a}"] = nrm[:, a]
            w[f"g{a}"] = g[:, a]
            for c in range(d):
                w[f"g{c}n{a}"] = g[:, c] * nrm[:, a]
        w["gn"] = np.einsum("ij,ij->i", g, nrm)
        sm = SurfaceMeasures(disc, S, w)
        Nf, Gf = sm.N, sm.G
        xi = disc.fitted_reference[2]
        Npf = basis.shape(1, xi)
        cells = sm.cells
        ud = _vdofs(disc.dofs.dofs_of(cells), d)
        pd = fd.pdofs.dofs_of(cells) + nu_off
        sym = sum(sm.outer(f"n{a}", Nf, Gf[:, :, a]) for a in range(d))          # <N_a, dN_b/dn>
        Ks = -nu * (sym + sym.transpose(0, 2, 1)) + sigma * sm.outer("1", Nf, Nf)
        coo.add(ud, ud, _kron_eye(Ks, d))
        # <p n, v> in the momentum rows and <q n, u> in the continuity rows
        nb, nbp = Nf.shape[1], Npf.shape[1]
        P = np.zeros((len(cells), nb, d, nbp))
        for a in range(d):
            P[:, :, a, :] = sm.outer(f"n{a}", Nf, Npf)
        P = P.reshape(len(cells), nb * d, nbp)
        coo.add(ud, pd, P)
        coo.add(pd, ud, P.transpose(0, 2, 1))
        if body_velocity is not None:
            rhs = np.zeros((len(cells), nb, d))
            for c in range(d):
                dn = sum(sm.vector(f"g{c}n{a}", Gf[:, :, a]) for a in range(d))
                rhs[:, :, c] = -nu * dn + sigma * sm.vector(f"g{c}", Nf)
            np.add.at(b, ud.ravel(), rhs.ravel())
            np.add.at(b, pd.ravel(), sm.vector("gn", Npf).ravel())
    # ghost penalties across the faces of cut cells
    su, sp = ghost_u * (nu + u_ref * h), ghost_p * h ** 2 / (nu + u_ref * h)
    n_faces = 0
    for k, pairs in enumerate(ghost_faces(disc.quad, disc.dofs)):
        if not len(pairs):
            continue
        n_faces += len(pairs)
        Gu = ghost_reference(fd.grid, k)
        cdu = np.concatenate([disc.dofs.dofs_of(pairs[:, 0]), disc.dofs.dofs_of(pairs[:, 1])], axis=1)
        vd = _vdofs(cdu, d)
        coo.add(vd, vd, np.broadcast_to(np.kron(Gu, np.eye(d)), (len(pairs), vd.shape[1], vd.shape[1])), su)
        Gp = ghost_reference(fd.grid_p, k)
        cdp = np.concatenate([fd.pdofs.dofs_of(pairs[:, 0]), fd.pdofs.dofs_of(pairs[:, 1])], axis=1) + nu_off
        coo.add(cdp, cdp, np.broadcast_to(Gp, (len(pairs),) + Gp.shape), -sp)
    info.update(ghost_faces=n_faces, ghost_u=su, ghost_p=sp, assembly_s=time.perf_counter() - t0)
    return {"L": coo.matrix(), "L_vol": vol.matrix(), "b": b, "b_vol": b_vol, "sets": sets, "info": info}


# ------------------------------------------------------------------------------------------------ box conditions
def _box_dirichlet(fd: FlowDiscretization, bcs: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Constrained unknowns and their values from the box-face conditions; whether the pressure is fixed by an
    outflow face."""
    d = fd.d
    X = fd.disc.dofs.coords()
    lo, hi = fd.box
    tol = 1e-9 * float(np.max(hi - lo))
    fixed = np.zeros((fd.n_u, d), bool)
    val = np.zeros((fd.n_u, d))
    slip = np.zeros((fd.n_u, d), bool)
    outflow = False
    bad = [k for k in bcs if k not in FACES[:2 * d]]
    if bad:
        raise ValueError(f"unknown faces {bad}")
    for face, bc in bcs.items():
        ax, side = FACES.index(face) // 2, FACES.index(face) % 2
        on = np.abs(X[:, ax] - (hi[ax] if side else lo[ax])) < tol
        if isinstance(bc, Velocity):
            fixed[on] = True
            val[on] = _value(bc.value, X[on], d)
        elif bc == "wall":
            fixed[on] = True
            val[on] = 0.0
        elif bc == "slip":
            slip[on, ax] = True
        elif bc == "outflow":
            outflow = True
        else:
            raise ValueError(f"face {face}: {bc!r} is not a Velocity, 'wall', 'slip' or 'outflow'")
    slip &= ~fixed
    fixed |= slip
    idx = np.nonzero(fixed.ravel())[0]
    return idx, val.ravel()[idx], outflow


# ------------------------------------------------------------------------------------------------ solution
@dataclass
class FlowSolution:
    fd: FlowDiscretization
    x: np.ndarray
    nu: float
    info: dict = field(default_factory=dict)
    _sys: dict | None = field(default=None, repr=False)

    def velocity(self, X: np.ndarray) -> np.ndarray:
        return self.fd.evaluate_velocity(self.x, X)

    def velocity_gradient(self, X: np.ndarray) -> np.ndarray:
        return self.fd.evaluate_velocity(self.x, X, gradient=True)

    def pressure(self, X: np.ndarray) -> np.ndarray:
        return self.fd.evaluate_pressure(self.x, X)

    def force(self) -> dict:
        """Force on the body (per unit depth in 2-D, density 1): variational (reaction) form and Nitsche surface flux.
        """
        fd, d = self.fd, self.fd.d
        sys = self._sys
        X = fd.disc.dofs.coords()
        phi = fd.fluid.phi(X)
        near = phi > -3.0 * fd.h                     # body, its cut cells and their neighbours
        lo, hi = fd.box
        tol = 1e-9 * float(np.max(hi - lo))
        on_box = np.any((np.abs(X - lo) < tol) | (np.abs(X - hi) < tol), axis=1)
        out = {}
        if np.any(near & on_box):              # the test function would touch the box faces: no reaction force
            out["variational"] = np.full(d, np.nan)
            out["note"] = "the body is within 3 h of the box faces: only the surface flux is available"
        r = sys["L_vol"] @ self.x - sys["b_vol"]
        if self.info.get("navier_stokes"):
            coo = _COO(fd.n_unknowns)
            _convection(fd, sys["sets"], self.x, coo, None)
            r = r + coo.matrix() @ self.x
        if "variational" not in out:
            F = np.zeros(d)
            for k in range(d):
                V = np.zeros(fd.n_unknowns)
                V[np.nonzero(near)[0] * d + k] = 1.0
                F[k] = -float(V @ r)
            out["variational"] = F
        S = fd.disc.quad.surface
        if len(S):
            Xs = fd.disc.physical(S)
            gu = self.velocity_gradient(Xs)                                   # (n, d, d): d_j u_i
            dun = np.einsum("nij,nj->ni", gu, S.normal)
            p = self.pressure(Xs)
            u = self.velocity(Xs)
            g = np.zeros_like(u) if sys.get("body_velocity") is None else _value(sys["body_velocity"], Xs, d)
            sigma = sys["info"]["nitsche_sigma"]
            t = -(self.nu * dun - p[:, None] * S.normal) + sigma * (u - g)
            out["surface"] = (S.w[:, None] * t).sum(0)
        return out

    def divergence_l2(self) -> float:
        """L2 norm of div u over the fluid (cut-cell rule)."""
        tot = 0.0
        for cells, xi, W, *_ in self._sys["sets"]:
            X = self.fd.grid.to_physical(np.repeat(cells, len(xi)), np.tile(xi, (len(cells), 1)))
            gu = self.velocity_gradient(X)
            div = np.trace(gu, axis1=1, axis2=2).reshape(len(cells), len(xi))
            tot += float(np.sum(W * div ** 2))
        return float(np.sqrt(tot))


def _solve_constrained(A: sparse.csr_matrix, b: np.ndarray, idx: np.ndarray, vals: np.ndarray,
                       solver: str = "auto") -> np.ndarray:
    from ..numerical_solvers.sparse_direct import solve
    n = A.shape[0]
    free = np.ones(n, bool)
    free[idx] = False
    x = np.zeros(n)
    x[idx] = vals
    Af = A[free][:, free]
    rhs = b[free] - A[free][:, idx] @ vals
    x[free] = solve(Af, rhs, solver)
    return x


def _constraints(fd: FlowDiscretization, bcs: dict) -> tuple[np.ndarray, np.ndarray]:
    idx, vals, outflow = _box_dirichlet(fd, bcs)
    if not outflow:                                  # enclosed flow: pin one pressure (the mean is removed after)
        cell = fd.disc.quad.inside_cells
        p0 = fd.pdofs.dofs_of(cell[:1])[0, 0] if len(cell) else 0
        idx = np.append(idx, fd.d * fd.n_u + p0)
        vals = np.append(vals, 0.0)
    return idx, vals


def solve_stokes(fd: FlowDiscretization, nu: float, bcs: dict, body_velocity=None, force=None, u_ref: float = 0.0,
                 **kw) -> FlowSolution:
    """Steady Stokes flow. ``bcs``: box face -> :class:`Velocity`, ``"wall"``, ``"slip"`` or ``"outflow"``."""
    t0 = time.perf_counter()
    sys = assemble_flow(fd, nu, u_ref, body_velocity, force, **kw)
    sys["body_velocity"] = body_velocity
    idx, vals = _constraints(fd, bcs)
    t1 = time.perf_counter()
    x = _solve_constrained(sys["L"], sys["b"], idx, vals)
    info = {**sys["info"], "unknowns": fd.n_unknowns, "solve_s": time.perf_counter() - t1,
            "total_s": time.perf_counter() - t0, "navier_stokes": False, "quadrature_s": fd.timings["quadrature_s"]}
    sol = FlowSolution(fd, x, nu, info, sys)
    _remove_mean_pressure(sol, bcs)
    return sol


def solve_navier_stokes(fd: FlowDiscretization, nu: float, bcs: dict, u_ref: float, body_velocity=None, force=None,
                        tol: float = 1e-8, max_iter: int = 30, initial: FlowSolution | None = None, **kw
                        ) -> FlowSolution:
    """Steady Navier-Stokes by Newton's method from the Stokes solution (or ``initial``); stops when the residual
    of the free unknowns falls below ``tol`` times the residual of the first iterate (or to round-off)."""
    t0 = time.perf_counter()
    sys = assemble_flow(fd, nu, u_ref, body_velocity, force, **kw)
    sys["body_velocity"] = body_velocity
    idx, vals = _constraints(fd, bcs)
    free = np.ones(fd.n_unknowns, bool)
    free[idx] = False
    L, b = sys["L"], sys["b"]
    if initial is not None:
        x = initial.x.copy()
        x[idx] = vals
    else:
        x = _solve_constrained(L, b, idx, vals)
    hist, r0 = [], None
    t_solve = 0.0
    for it in range(max_iter):
        cc, cd = _COO(fd.n_unknowns), _COO(fd.n_unknowns)
        _convection(fd, sys["sets"], x, cc, cd)
        C = cc.matrix()
        R = L @ x + C @ x - b
        rn = float(np.linalg.norm(R[free]))
        r0 = rn if r0 is None else r0
        hist.append(rn)
        if it == 0:                                  # round-off floor: the size of the terms of the residual
            floor = 1e-13 * float(np.linalg.norm((abs(L) @ np.abs(x) + abs(C) @ np.abs(x) + np.abs(b))[free]))
        if it > 0 and rn <= max(tol * r0, floor):
            break
        J = L + C + cd.matrix()
        t1 = time.perf_counter()
        dx = _solve_constrained(J, -R, idx, np.zeros(len(idx)))
        t_solve += time.perf_counter() - t1
        x = x + dx
    else:
        raise RuntimeError(f"Newton did not converge in {max_iter} iterations: residuals {hist}")
    info = {**sys["info"], "unknowns": fd.n_unknowns, "newton_iterations": len(hist) - 1, "residuals": hist,
            "solve_s": t_solve, "total_s": time.perf_counter() - t0, "navier_stokes": True,
            "quadrature_s": fd.timings["quadrature_s"]}
    sol = FlowSolution(fd, x, nu, info, sys)
    _remove_mean_pressure(sol, bcs)
    return sol


def _remove_mean_pressure(sol: FlowSolution, bcs: dict) -> None:
    if any(v == "outflow" for v in bcs.values() if isinstance(v, str)):
        return
    fd = sol.fd
    num = den = 0.0
    for cells, xi, W, *_ in sol._sys["sets"]:
        X = fd.grid.to_physical(np.repeat(cells, len(xi)), np.tile(xi, (len(cells), 1)))
        p = fd.evaluate_pressure(sol.x, X).reshape(len(cells), len(xi))
        num += float(np.sum(W * p))
        den += float(np.sum(W))
    sol.x[fd.d * fd.n_u:] -= num / den
