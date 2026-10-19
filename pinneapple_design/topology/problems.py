"""Density-based topology-optimization problems on structured grids of square bilinear (Q4) elements.

Grid convention: ``nelx x nely`` elements of size ``h``, element ``e = i * nely + j`` (i along x,
j along y, y upwards), node ``n = i * (nely + 1) + j``. Element matrices are integrated with 2x2
Gauss quadrature (selective one-point integration for the incompressibility penalty).

Every problem exposes ``n`` (number of elements), ``volumes`` and ``evaluate(x) -> (f, df/dx)`` on
physical densities ``x`` in [0, 1].

* :class:`Compliance2D` -- plane-stress linear elasticity, SIMP ``E = Emin + x^p (E0 - Emin)``,
  compliance ``c = f^T u``, ``dc/dx_e = -p x_e^(p-1) (E0 - Emin) u_e^T k_e u_e``. Presets:
  :meth:`Compliance2D.mbb` (half MBB beam) and :meth:`Compliance2D.cantilever`.
* :class:`ThermalCompliance2D` -- steady conduction ``k = kmin + x^p (1 - kmin)``, uniform heat source,
  isothermal sink on a boundary segment, objective ``f^T T``.
* :class:`StokesDissipation2D` -- Stokes flow in a Brinkman medium (Borrvall & Petersson, Int. J.
  Numer. Meth. Fluids 41 (2003) 77-107): inverse permeability
  ``alpha(x) = a_min + (a_max - a_min) q (1 - x) / (q + x)`` (x = 1 fluid), velocity-driven, objective the
  potential power ``0.5 u^T K u`` (viscous + Brinkman + incompressibility penalty) whose sensitivity is
  ``0.5 u_e^T (d alpha/dx_e) m_e u_e`` (the minimum of a quadratic energy over a fixed affine space).
  Incompressibility is imposed by a penalty with one-point (reduced) integration.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["Q4Grid", "Compliance2D", "ThermalCompliance2D", "StokesDissipation2D",
           "elasticity_ke", "conduction_ke"]

_GP = np.array([-1.0, 1.0]) / math.sqrt(3.0)


def _shape_grads(xi, eta, h):
    """dN/dx, dN/dy of the 4 bilinear shape functions on an h x h square (nodes CCW from bottom-left)."""
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return dxi * 2.0 / h, deta * 2.0 / h


def _shape(xi, eta):
    return 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])


def elasticity_ke(nu: float = 0.3, E: float = 1.0, h: float = 1.0, thickness: float = 1.0) -> np.ndarray:
    """8x8 plane-stress stiffness of a square Q4 element, dofs (ux, uy) per node."""
    D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    ke = np.zeros((8, 8))
    detJ = (h / 2) ** 2
    for xi in _GP:
        for eta in _GP:
            dx, dy = _shape_grads(xi, eta, h)
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dx, dy
            B[2, 0::2], B[2, 1::2] = dy, dx
            ke += B.T @ D @ B * detJ * thickness
    return ke


def conduction_ke(h: float = 1.0) -> np.ndarray:
    """4x4 conduction matrix (unit conductivity) of a square Q4 element."""
    ke = np.zeros((4, 4))
    for xi in _GP:
        for eta in _GP:
            dx, dy = _shape_grads(xi, eta, h)
            ke += (np.outer(dx, dx) + np.outer(dy, dy)) * (h / 2) ** 2
    return ke


class Q4Grid:
    def __init__(self, nelx: int, nely: int, h: float = 1.0):
        self.nelx, self.nely, self.h = nelx, nely, h
        i, j = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        i, j = i.ravel(), j.ravel()
        n0 = i * (nely + 1) + j
        self.enodes = np.column_stack([n0, n0 + nely + 1, n0 + nely + 2, n0 + 1])  # CCW
        self.n_nodes = (nelx + 1) * (nely + 1)
        self.n_el = nelx * nely
        self.centroids = np.column_stack([(i + 0.5) * h, (j + 0.5) * h])

    def node(self, i: int, j: int) -> int:
        return i * (self.nely + 1) + j

    def node_xy(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nelx + 1), np.arange(self.nely + 1), indexing="ij")
        return np.column_stack([i.ravel() * self.h, j.ravel() * self.h])

    def edofs(self, ndof: int) -> np.ndarray:
        return (ndof * self.enodes[:, :, None] + np.arange(ndof)).reshape(self.n_el, -1)

    def assemble(self, edof: np.ndarray, ke: np.ndarray, coef: np.ndarray, ndof_total: int):
        k = edof.shape[1]
        rows = np.repeat(edof, k, axis=1).ravel()
        cols = np.tile(edof, (1, k)).ravel()
        vals = (coef[:, None, None] * ke[None]).ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(ndof_total, ndof_total))


def _solve(K, f, fixed, values=None, ndof=None):
    ndof = K.shape[0]
    u = np.zeros(ndof)
    if values is not None:
        u[fixed] = values
    free = np.setdiff1d(np.arange(ndof), fixed)
    rhs = f[free] - K[free][:, fixed] @ u[fixed]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u


class Compliance2D:
    def __init__(self, nelx: int, nely: int, fixed_dofs, loads: dict, penal: float = 3.0,
                 E0: float = 1.0, Emin: float = 1e-9, nu: float = 0.3):
        self.grid = Q4Grid(nelx, nely)
        self.n = self.grid.n_el
        self.volumes = np.ones(self.n)
        self.ndof = 2 * self.grid.n_nodes
        self.edof = self.grid.edofs(2)
        self.ke = elasticity_ke(nu, 1.0)
        self.fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        self.f = np.zeros(self.ndof)
        for dof, val in loads.items():
            self.f[dof] += val
        self.penal, self.E0, self.Emin = penal, E0, Emin
        self.u = None

    @classmethod
    def mbb(cls, nelx: int = 60, nely: int = 20, **kw) -> Compliance2D:
        """Half MBB beam: symmetry (u_x = 0) on the left edge, roller (u_y = 0) at the bottom-right corner,
        unit downward load at the top-left corner."""
        g = Q4Grid(nelx, nely)
        left = [2 * g.node(0, j) for j in range(nely + 1)]
        fixed = left + [2 * g.node(nelx, 0) + 1]
        return cls(nelx, nely, fixed, {2 * g.node(0, nely) + 1: -1.0}, **kw)

    @classmethod
    def cantilever(cls, nelx: int = 80, nely: int = 40, **kw) -> Compliance2D:
        """Clamped left edge, unit downward load at the middle of the right edge."""
        g = Q4Grid(nelx, nely)
        fixed = [2 * g.node(0, j) + d for j in range(nely + 1) for d in (0, 1)]
        return cls(nelx, nely, fixed, {2 * g.node(nelx, nely // 2) + 1: -1.0}, **kw)

    def evaluate(self, x: np.ndarray):
        E = self.Emin + x ** self.penal * (self.E0 - self.Emin)
        K = self.grid.assemble(self.edof, self.ke, E, self.ndof)
        u = _solve(K, self.f, self.fixed)
        self.u = u
        ue = u[self.edof]
        ce = np.einsum("ei,ij,ej->e", ue, self.ke, ue)
        c = float(self.f @ u)
        dc = -self.penal * x ** (self.penal - 1) * (self.E0 - self.Emin) * ce
        return c, dc


class ThermalCompliance2D:
    """Square plate, uniform heat source ``q`` per unit area, ``T = 0`` on the middle ``sink_fraction`` of
    the left edge, insulated elsewhere."""

    def __init__(self, nelx: int = 60, nely: int = 60, sink_fraction: float = 0.1, q: float = 1.0,
                 penal: float = 3.0, kmin: float = 1e-3):
        self.grid = g = Q4Grid(nelx, nely)
        self.n = g.n_el
        self.volumes = np.ones(self.n)
        self.edof = g.enodes
        self.ke = conduction_ke()
        j0 = int(round(nely * (0.5 - sink_fraction / 2)))
        j1 = int(round(nely * (0.5 + sink_fraction / 2)))
        self.fixed = np.array([g.node(0, j) for j in range(j0, j1 + 1)])
        self.f = np.bincount(g.enodes.ravel(), weights=np.full(4 * self.n, q / 4.0), minlength=g.n_nodes)
        self.penal, self.kmin = penal, kmin
        self.T = None

    def evaluate(self, x: np.ndarray):
        k = self.kmin + x ** self.penal * (1.0 - self.kmin)
        K = self.grid.assemble(self.edof, self.ke, k, self.grid.n_nodes)
        T = _solve(K, self.f, self.fixed)
        self.T = T
        te = T[self.edof]
        ce = np.einsum("ei,ij,ej->e", te, self.ke, te)
        return float(self.f @ T), -self.penal * x ** (self.penal - 1) * (1.0 - self.kmin) * ce


class StokesDissipation2D:
    """Velocity-driven Stokes flow with Brinkman penalization on a ``Lx x 1`` box.

    ``inlets`` / ``outlets``: lists of ``(y_center, width, peak_speed)`` on the left / right edges with
    parabolic profiles (outflow given, as in the "double pipe" benchmark); no slip elsewhere.
    """

    def __init__(self, nelx: int, nely: int, inlets, outlets, mu: float = 1.0,
                 alpha_max: float | None = None, alpha_min: float | None = None, q: float = 0.1,
                 penalty: float = 1e4):
        self.grid = g = Q4Grid(nelx, nely, h=1.0 / nely)
        self.n = g.n_el
        self.volumes = np.ones(self.n)
        self.ndof = 2 * g.n_nodes
        self.edof = g.edofs(2)
        h = g.h
        self.mu = mu
        self.alpha_max = 2.5 * mu / 0.01 ** 2 if alpha_max is None else alpha_max
        self.alpha_min = 2.5 * mu / 100.0 ** 2 if alpha_min is None else alpha_min
        self.q = q
        # element matrices: viscous (vector Laplacian), Brinkman mass, reduced-integration divergence
        kv = np.zeros((8, 8))
        me = np.zeros((8, 8))
        for xi in _GP:
            for eta in _GP:
                dx, dy = _shape_grads(xi, eta, h)
                N = _shape(xi, eta)
                w = (h / 2) ** 2
                L = np.outer(dx, dx) + np.outer(dy, dy)
                kv[0::2, 0::2] += L * w
                kv[1::2, 1::2] += L * w
                M = np.outer(N, N) * w
                me[0::2, 0::2] += M
                me[1::2, 1::2] += M
        dx, dy = _shape_grads(0.0, 0.0, h)
        div = np.zeros(8)
        div[0::2], div[1::2] = dx, dy
        kd = np.outer(div, div) * h * h
        self.kv, self.me, self.kd = mu * kv, me, penalty * mu * kd
        # boundary conditions: all boundary nodes fixed (no slip / prescribed in- and outflow)
        xy = g.node_xy()
        Lx = nelx * h
        bnd = np.where((xy[:, 0] < 1e-12) | (xy[:, 0] > Lx - 1e-12) | (xy[:, 1] < 1e-12) | (xy[:, 1] > 1 - 1e-12))[0]
        vals = np.zeros(self.ndof)
        for side, specs in ((0.0, inlets), (Lx, outlets)):
            on = bnd[np.abs(xy[bnd, 0] - side) < 1e-12]
            for yc, wdt, umax in specs:
                t = (xy[on, 1] - yc) / (wdt / 2)
                inside = np.abs(t) < 1
                vals[2 * on[inside]] = umax * (1 - t[inside] ** 2)
        self.fixed = np.concatenate([2 * bnd, 2 * bnd + 1])
        self.fixed_values = vals[self.fixed]
        self.u = None

    def alpha(self, x):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * self.q * (1 - x) / (self.q + x)

    def dalpha(self, x):
        return -(self.alpha_max - self.alpha_min) * self.q * (1 + self.q) / (self.q + x) ** 2

    def evaluate(self, x: np.ndarray):
        g = self.grid
        a = self.alpha(x)
        ones = np.ones(self.n)
        K = (g.assemble(self.edof, self.kv + self.kd, ones, self.ndof)
             + g.assemble(self.edof, self.me, a, self.ndof))
        u = _solve(K, np.zeros(self.ndof), self.fixed, self.fixed_values)
        self.u = u
        ue = u[self.edof]
        phi = 0.5 * float(u @ (K @ u))
        mm = np.einsum("ei,ij,ej->e", ue, self.me, ue)
        return phi, 0.5 * self.dalpha(x) * mm

    @classmethod
    def double_pipe(cls, nely: int = 60, length: float = 1.5, **kw) -> StokesDissipation2D:
        """Two inlets and two outlets of width 1/6 centred at y = 1/4 and 3/4 (Borrvall & Petersson's
        double-pipe layout), unit peak speed."""
        nelx = int(round(length * nely))
        specs = [(0.25, 1.0 / 6.0, 1.0), (0.75, 1.0 / 6.0, 1.0)]
        return cls(nelx, nely, specs, specs, **kw)
