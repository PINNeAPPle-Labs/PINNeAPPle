"""Time-harmonic 2D eddy currents in A_z with solid conductors (linear media).

Phasors with ``exp(+j omega t)``. In a solid conductor ``c`` the current density is
``J = sigma (-j omega A + u_c)`` where ``u_c`` is the (uniform) voltage per metre along z, an extra
unknown fixed by the net current ``int_c J dS = I_c`` (``I_c = 0`` for an insulated piece such as a
magnet segment). The system::

    int nu grad A . grad v + j omega int sigma A v - sum_c u_c int_c sigma v = int J_s v
    -j omega int_c sigma A + u_c int_c sigma                                 = I_c

is assembled with P1 elements (consistent mass matrix) and solved with a sparse direct solver.
Losses are ``int |J|^2 / (2 sigma)`` (time average, peak phasors).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fem import Geometry, constraint_map
from .mesh import Mesh2D

__all__ = ["TimeHarmonicSolution", "solve_time_harmonic"]


@dataclass
class TimeHarmonicSolution:
    A: np.ndarray  # complex nodal potential
    u: np.ndarray  # voltage per metre of each conductor
    J_nodal: np.ndarray  # complex induced + imposed current density at the nodes of each conductor (NaN elsewhere)
    loss: np.ndarray  # time-averaged loss per conductor, W/m
    geometry: Geometry


def _mass(g: Geometry, coef: np.ndarray) -> sp.csr_matrix:
    me = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    ke = (coef * g.area)[:, None, None] * me[None]
    return sp.csr_matrix((ke.ravel(), (g.rows, g.cols)), shape=(g.n, g.n))


def solve_time_harmonic(mesh: Mesh2D, nu: np.ndarray, sigma: np.ndarray, omega: float,
                        conductors: Sequence[np.ndarray], currents: Sequence[complex],
                        J_source: np.ndarray | None = None, dirichlet: np.ndarray | None = None,
                        dirichlet_values=None) -> TimeHarmonicSolution:
    """Solve for A (complex). ``conductors``: element masks of solid conductors; ``currents``: their
    net current phasors (A). ``sigma`` (M,) is used inside conductors only."""
    g = Geometry.of(mesh)
    n = mesh.n_nodes
    M = mesh.n_elements
    nu = np.asarray(nu, float)
    sigma = np.asarray(sigma, float)
    in_cond = np.zeros(M, bool)
    for m in conductors:
        in_cond |= m
    sig = np.where(in_cond, sigma, 0.0)
    K = g.stiffness(nu).astype(complex) + 1j * omega * _mass(g, sig)
    f = np.zeros(n, complex)
    if J_source is not None:
        f += np.bincount(mesh.tris.ravel(), weights=np.repeat(np.asarray(J_source) * g.area / 3.0, 3),
                         minlength=n)
    nc = len(conductors)
    C = np.zeros((n, nc))  # C[:, c] = int_c sigma phi_i
    s_tot = np.zeros(nc)
    for k, m in enumerate(conductors):
        w = np.repeat(sig[m] * g.area[m] / 3.0, 3)
        C[:, k] = np.bincount(mesh.tris[m].ravel(), weights=w, minlength=n)
        s_tot[k] = float(np.sum(sig[m] * g.area[m]))
    Cs = sp.csr_matrix(C)
    big = sp.bmat([[K, -Cs], [-1j * omega * Cs.T, sp.diags(s_tot)]], format="csr")
    rhs = np.concatenate([f, np.asarray(currents, complex)])
    if dirichlet is None:
        dirichlet = mesh.boundary
    P, AD = constraint_map(n, dirichlet, dirichlet_values)
    Pb = sp.block_diag([P, sp.identity(nc)], format="csr")
    ADb = np.concatenate([AD, np.zeros(nc)])
    lhs = (Pb.T @ big @ Pb).tocsc()
    x = spla.spsolve(lhs, Pb.T @ (rhs - big @ ADb))
    full = Pb @ x + ADb
    A, u = full[:n], full[n:]
    loss = np.zeros(nc)
    Jn = np.full(n, np.nan + 0j)
    me = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    for k, m in enumerate(conductors):
        t = mesh.tris[m]
        Je = sig[m][:, None] * (-1j * omega * A[t] + u[k])  # nodal values per element (P1)
        q = np.einsum("mi,ij,mj->m", Je, me, Je.conj()).real
        loss[k] = float(np.sum(g.area[m] * q / (2.0 * sig[m])))
        Jn[t.ravel()] = Je.ravel()
    return TimeHarmonicSolution(A, u, Jn, loss, g)
