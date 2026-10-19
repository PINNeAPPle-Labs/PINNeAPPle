"""2D magnetostatics in the z-component of the magnetic vector potential, P1 triangles.

Strong form, with ``B = curl(A_z e_z) = (dA/dy, -dA/dx)`` and the constitutive law ``H = nu(|B|^2) (B - Br)``::

    -div( nu grad A ) = J_z + (curl of nu Br)_z

Weak form solved here, for every test function ``v``::

    int nu grad A . grad v  =  int J v  +  int nu (Br_x dv/dy - Br_y dv/dx)

It is the stationarity condition of the energy functional (per metre of stack)::

    Pi(A) = sum_e area_e [ w_e(|B_e|^2) - nu_e Br_e . B_e ] - int J A

with ``w = nu |B|^2 / 2`` for linear media and ``w = int_0^|B| H db`` for nonlinear iron. ``Pi`` is convex
for monotone B-H curves, so Newton's method with a backtracking line search on ``Pi`` converges from
any start. ``-Pi`` at the solution equals the magnetic co-energy up to a term that does not depend on
the rotor angle; this is what :func:`pinneapple_simulation.electromagnetics.postprocess.torque_coenergy`
differentiates.

Boundary conditions: Dirichlet values on any node set, and periodic / anti-periodic node pairs
``A[slave] = sign * A[master]`` (eliminated through a sparse map ``A = P a + A_D``). Nodes with neither
are natural (zero tangential H, i.e. flux crosses the boundary at right angles).

Magnets are linear (``mu_r``, ``Br``); iron may be nonlinear. Materials are passed per element.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh2D

__all__ = ["Geometry", "MagnetostaticSolution", "solve_magnetostatic", "constraint_map", "flux_density"]


@dataclass
class Geometry:
    """Per-element areas and barycentric gradients of a P1 mesh, plus sparse-assembly indices."""

    area: np.ndarray  # (M,)
    gx: np.ndarray  # (M, 3) d(phi_i)/dx
    gy: np.ndarray  # (M, 3) d(phi_i)/dy
    rows: np.ndarray
    cols: np.ndarray
    n: int

    @classmethod
    def of(cls, mesh: Mesh2D) -> Geometry:
        p = mesh.points[mesh.tris]
        x, y = p[..., 0], p[..., 1]
        det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        if np.any(det <= 0):
            raise ValueError("mesh has degenerate or clockwise triangles")
        gx = np.column_stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]]) / det[:, None]
        gy = np.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]]) / det[:, None]
        t = mesh.tris
        rows = np.repeat(t, 3, axis=1).ravel()
        cols = np.tile(t, (1, 3)).ravel()
        return cls(0.5 * det, gx, gy, rows, cols, mesh.n_nodes)

    def stiffness(self, coef: np.ndarray) -> sp.csr_matrix:
        """sum_e coef_e area_e grad(phi_i).grad(phi_j)."""
        ke = (self.gx[:, :, None] * self.gx[:, None, :] + self.gy[:, :, None] * self.gy[:, None, :])
        ke *= (coef * self.area)[:, None, None]
        return sp.csr_matrix((ke.ravel(), (self.rows, self.cols)), shape=(self.n, self.n))

    def outer(self, coef: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> sp.csr_matrix:
        """sum_e coef_e area_e (v.grad phi_i)(v.grad phi_j), v constant per element."""
        d = vx[:, None] * self.gx + vy[:, None] * self.gy
        ke = (coef * self.area)[:, None, None] * d[:, :, None] * d[:, None, :]
        return sp.csr_matrix((ke.ravel(), (self.rows, self.cols)), shape=(self.n, self.n))

    def load(self, J: np.ndarray, nu: np.ndarray, Br: np.ndarray, tris: np.ndarray) -> np.ndarray:
        """Right-hand side from current density J (A/m^2) and remanence Br (T) per element."""
        fe = (J * self.area / 3.0)[:, None] + (nu * self.area)[:, None] * (
            Br[:, 0:1] * self.gy - Br[:, 1:2] * self.gx)
        return np.bincount(tris.ravel(), weights=fe.ravel(), minlength=self.n)

    def curl(self, A: np.ndarray, tris: np.ndarray) -> np.ndarray:
        """Element flux density (M, 2): B = (dA/dy, -dA/dx)."""
        a = A[tris]
        return np.column_stack([(a * self.gy).sum(1), -(a * self.gx).sum(1)])


def flux_density(mesh: Mesh2D, A: np.ndarray) -> np.ndarray:
    """B per element (T) from nodal A_z (Wb/m)."""
    return Geometry.of(mesh).curl(A, mesh.tris)


def constraint_map(n: int, dirichlet: np.ndarray | None = None, dirichlet_values=None,
                   periodic: Sequence[tuple[int, int, float]] | None = None):
    """Sparse map ``A = P a + A_D``: Dirichlet nodes take their values, a periodic slave node takes
    ``sign * A[master]`` (chains are resolved), the remaining nodes are the unknowns ``a``."""
    is_d = np.zeros(n, bool)
    cplx = dirichlet_values is not None and np.iscomplexobj(dirichlet_values)
    AD = np.zeros(n, complex if cplx else float)
    if dirichlet is not None and len(dirichlet):
        dirichlet = np.asarray(dirichlet, dtype=np.int64)
        is_d[dirichlet] = True
        AD[dirichlet] = 0.0 if dirichlet_values is None else dirichlet_values
    master = np.arange(n)
    sign = np.ones(n)
    for s, m, f in periodic or []:
        if s == m:
            continue
        master[s] = m
        sign[s] = f
    for _ in range(8):  # resolve chains like corner nodes
        mm = master[master]
        if np.array_equal(mm, master):
            break
        sign = sign * sign[master]
        master = mm
    slave = master != np.arange(n)
    # a slave of a Dirichlet master is Dirichlet itself (homogeneous or consistent values expected)
    d_through = slave & is_d[master]
    is_d = is_d | d_through
    AD[d_through] = sign[d_through] * AD[master[d_through]]
    free = ~is_d & ~slave
    col = -np.ones(n, dtype=np.int64)
    col[free] = np.arange(free.sum())
    rows = np.where(free | (slave & ~is_d))[0]
    cols = col[np.where(free[rows], rows, master[rows])]
    vals = np.where(free[rows], 1.0, sign[rows])
    if np.any(cols < 0):
        raise ValueError("a periodic master node is itself constrained in an unsupported way")
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, int(free.sum())))
    return P, AD


@dataclass
class MagnetostaticSolution:
    A: np.ndarray
    B: np.ndarray
    nu: np.ndarray  # secant reluctivity per element at the solution
    energy_functional: float  # Pi at the solution, J/m
    iterations: int
    residual: float
    geometry: Geometry
    tangent: sp.spmatrix | None = None  # reduced tangent P^T K_t P at the solution
    P: sp.spmatrix | None = None
    info: dict = field(default_factory=dict)

    @property
    def coenergy(self) -> float:
        """-Pi: magnetic co-energy per metre up to an angle-independent constant (J/m)."""
        return -self.energy_functional


def _element_energy(B2, Bdot, nu_lin, nonlinear, area):
    w = 0.5 * nu_lin * B2
    for mask, curve in nonlinear:
        w[mask] = curve.energy(B2[mask])
    return float(np.sum(area * (w - nu_lin * Bdot)))


def solve_magnetostatic(mesh: Mesh2D, nu: np.ndarray, J: np.ndarray | None = None,
                        Br: np.ndarray | None = None,
                        nonlinear: Sequence[tuple[np.ndarray, object]] = (),
                        dirichlet: np.ndarray | None = None, dirichlet_values=None,
                        periodic: Sequence[tuple[int, int, float]] | None = None,
                        tol: float = 1e-9, max_iter: int = 60, A0: np.ndarray | None = None,
                        geometry: Geometry | None = None, keep_tangent: bool = False,
                        ) -> MagnetostaticSolution:
    """Solve the A_z magnetostatic problem.

    nu: (M,) reluctivity of linear elements (1/(mu0 mu_r)); for nonlinear elements it is ignored.
    J: (M,) source current density (A/m^2); Br: (M, 2) remanence (T), zero outside magnets.
    nonlinear: list of ``(element_mask, curve)`` with ``curve.nu``, ``curve.dnu``, ``curve.energy``.
    dirichlet: node indices (default: ``mesh.boundary`` with A=0); pass an empty array for none.
    Newton iterations stop when the reduced residual drops below ``tol`` times the load norm.
    """
    g = geometry or Geometry.of(mesh)
    M = mesh.n_elements
    nu_lin = np.asarray(nu, float).copy()
    J = np.zeros(M) if J is None else np.asarray(J, float)
    Br = np.zeros((M, 2)) if Br is None else np.asarray(Br, float)
    if dirichlet is None:
        dirichlet = mesh.boundary
    P, AD = constraint_map(mesh.n_nodes, dirichlet, dirichlet_values, periodic)
    f = g.load(J, nu_lin, Br, mesh.tris)
    f_J = np.bincount(mesh.tris.ravel(), weights=np.repeat(J * g.area / 3.0, 3), minlength=mesh.n_nodes)

    def state(A):
        B = g.curl(A, mesh.tris)
        B2 = (B * B).sum(1)
        nu_e = nu_lin.copy()
        dnu_e = np.zeros(M)
        for mask, curve in nonlinear:
            nu_e[mask] = curve.nu(B2[mask])
            dnu_e[mask] = curve.dnu(B2[mask])
        return B, B2, nu_e, dnu_e

    def functional(A):
        B, B2, _, _ = state(A)
        return _element_energy(B2, (B * Br).sum(1), nu_lin, nonlinear, g.area) - float(f_J @ A)

    def tangent(B, nu_e, dnu_e):
        K = g.stiffness(nu_e)
        Kt = K + g.outer(2.0 * dnu_e, -B[:, 1], B[:, 0]) if is_nl else K
        return K, Kt

    is_nl = len(nonlinear) > 0
    a = np.zeros(P.shape[1]) if A0 is None else P.T @ (np.asarray(A0, float) - AD)
    A = P @ a + AD
    B, B2, nu_e, dnu_e = state(A)
    K, Kt_full = tangent(B, nu_e, dnu_e)
    r = P.T @ (K @ A - f)
    scale = max(np.linalg.norm(P.T @ f), np.linalg.norm(P.T @ (K @ AD)), 1e-300)
    res = np.linalg.norm(r) / scale
    it = 0
    while it < max_iter and not (it > 0 and res < tol) and not (res == 0.0):
        it += 1
        d = spla.spsolve((P.T @ Kt_full @ P).tocsc(), -r)
        s = 1.0
        if is_nl:
            Pi0 = functional(A)
            slope = float(r @ d)
            # Near the solution the predicted decrease falls below the round-off of Pi (a sum of terms of size
            # |Pi|); the Armijo test is then noise, and the full Newton step is taken (quadratic convergence).
            if -slope > 1e3 * np.finfo(float).eps * max(abs(Pi0), 1e-300):
                while s > 1e-6 and functional(P @ (a + s * d) + AD) > Pi0 + 1e-4 * s * slope:
                    s *= 0.5
        a = a + s * d
        A = P @ a + AD
        B, B2, nu_e, dnu_e = state(A)
        K, Kt_full = tangent(B, nu_e, dnu_e)
        r = P.T @ (K @ A - f)
        res = np.linalg.norm(r) / scale
        if not is_nl:
            break
        if s * np.linalg.norm(d) <= 1e-15 * max(np.linalg.norm(a), 1e-300):
            break
    Kt = (P.T @ Kt_full @ P).tocsc() if keep_tangent else None
    Pi = _element_energy(B2, (B * Br).sum(1), nu_lin, nonlinear, g.area) - float(f_J @ A)
    return MagnetostaticSolution(A, B, nu_e, Pi, it, float(res), g,
                                 tangent=Kt if keep_tangent else None, P=P,
                                 info={"AD": AD})
