"""Linear plane-stress finite elements on any triangle mesh (constant-strain triangles), for parts whose geometry
comes from elsewhere (a B-rep face, a CAD tessellation).

    sol = solve_plane_stress(V2, F, E=68.9e9, nu=0.33, thickness=0.005,
                             fixed={"x": left_nodes_x, "y": [one_node]}, traction=[(edges, (sx, 0.0))])
    sol.u, sol.stress (per element: sxx, syy, sxy), sol.nodal_stress(), sol.von_mises()

Constant-strain triangles: the stress is constant per element and converges at first order in the element size;
:meth:`PlaneStressSolution.nodal_stress` averages it to the nodes (area weighted). ``patch_test`` checks the
element: a uniform stress state is reproduced exactly on any mesh. References: Zienkiewicz, Taylor & Zhu, *The Finite
Element Method*, 7th ed. (2013), ch. 6; Kirsch (1898) / Heywood's net-section form for a hole in a finite strip
(Pilkey & Pilkey, *Peterson's Stress Concentration Factors*, 3rd ed., 2008) for the stress concentration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["solve_plane_stress", "PlaneStressSolution", "boundary_edges", "patch_test", "heywood_kt_net"]


def heywood_kt_net(d_over_w: float) -> float:
    """Net-section stress concentration of a central hole in a strip under tension, ``2 + (1 - d/W)^3``.

    Returns
    -------
    Kt : float [-]
        Stress concentration factor on the net section."""
    return 2.0 + (1.0 - float(d_over_w)) ** 3


def boundary_edges(F) -> np.ndarray:
    """Edges ``(m, 2)`` that belong to one triangle only."""
    F = np.asarray(F, int)
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


@dataclass
class PlaneStressSolution:
    V: np.ndarray
    F: np.ndarray
    u: np.ndarray              # (n, 2) displacement [m]
    stress: np.ndarray         # (m, 3) sxx, syy, sxy per element [Pa]
    area: np.ndarray           # (m,)
    reaction: np.ndarray       # (n, 2) nodal reaction forces at the fixed dofs [N]

    def nodal_stress(self) -> np.ndarray:
        """Area-weighted average of the element stresses at each node, ``(n, 3)``."""
        n = len(self.V)
        acc, w = np.zeros((n, 3)), np.zeros(n)
        for k in range(3):
            np.add.at(acc, self.F[:, k], self.stress * self.area[:, None])
            np.add.at(w, self.F[:, k], self.area)
        return acc / np.maximum(w, 1e-300)[:, None]

    @staticmethod
    def _vm(s):
        return np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2)

    def von_mises(self, nodal: bool = True) -> np.ndarray:
        return self._vm(self.nodal_stress() if nodal else self.stress)


def _b_matrices(V, F):
    P = V[F]                                             # (m, 3, 2)
    x, y = P[..., 0], P[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    A2 = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
    B = np.zeros((len(F), 3, 6))
    B[:, 0, 0::2], B[:, 1, 1::2] = b, c
    B[:, 2, 0::2], B[:, 2, 1::2] = c, b
    return B / A2[:, None, None], np.abs(A2) / 2, np.sign(A2)


def solve_plane_stress(V, F, *, E: float, nu: float, thickness: float, fixed: dict, traction=(),
                       body_force=(0.0, 0.0)) -> PlaneStressSolution:
    """``V (n, 2)`` nodes [m], ``F (m, 3)`` triangles; ``fixed = {"x": nodes, "y": nodes}`` (zero displacement);
    ``traction``: ``[(edges (k, 2), (tx, ty) [Pa]), ...]`` on boundary edges; ``body_force`` [N/m^3]."""
    V, F = np.asarray(V, float)[:, :2], np.asarray(F, int)
    B, A, s = _b_matrices(V, F)
    if np.any(s < 0):                                   # orient every triangle counter-clockwise
        F = F.copy()
        F[s < 0] = F[s < 0][:, [0, 2, 1]]
        B, A, _ = _b_matrices(V, F)
    D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ke = thickness * A[:, None, None] * np.einsum("mki,kl,mlj->mij", B, D, B)
    dof = np.stack([2 * F, 2 * F + 1], -1).reshape(len(F), 6)
    n = 2 * len(V)
    K = sp.coo_matrix((Ke.ravel(), (np.repeat(dof, 6, 1).ravel(), np.tile(dof, (1, 6)).ravel())), (n, n)).tocsr()
    f = np.zeros(n)
    for edges, (tx, ty) in traction:
        edges = np.asarray(edges, int)
        L = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        for k in range(2):
            np.add.at(f, 2 * edges[:, k], 0.5 * L * thickness * tx)
            np.add.at(f, 2 * edges[:, k] + 1, 0.5 * L * thickness * ty)
    if any(body_force):
        for k in range(3):
            np.add.at(f, 2 * F[:, k], A * thickness * body_force[0] / 3)
            np.add.at(f, 2 * F[:, k] + 1, A * thickness * body_force[1] / 3)
    fix = np.zeros(n, bool)
    fix[2 * np.asarray(fixed.get("x", []), int)] = True
    fix[2 * np.asarray(fixed.get("y", []), int) + 1] = True
    free = ~fix
    u = np.zeros(n)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    r = K @ u - f
    reaction = np.where(fix, r, 0.0).reshape(-1, 2)
    U = u.reshape(-1, 2)
    eps = np.einsum("mij,mj->mi", B, u[dof])
    return PlaneStressSolution(V, F, U, eps @ D.T, A, reaction)


def patch_test(V, F, *, E: float = 1.0, nu: float = 0.3) -> float:
    """Impose the displacement of a uniform stress state on the boundary nodes and return the largest relative
    deviation of the element stresses from it (exact elements give round-off)."""
    V, F = np.asarray(V, float)[:, :2], np.asarray(F, int)
    eps0 = np.array([1e-3, -4e-4, 6e-4])                # exx, eyy, gxy
    ub = np.c_[eps0[0] * V[:, 0] + 0.5 * eps0[2] * V[:, 1], eps0[1] * V[:, 1] + 0.5 * eps0[2] * V[:, 0]]
    bn = np.unique(boundary_edges(F))
    B, A, s = _b_matrices(V, F)
    D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ke = A[:, None, None] * np.einsum("mki,kl,mlj->mij", B, D, B)
    dof = np.stack([2 * F, 2 * F + 1], -1).reshape(len(F), 6)
    n = 2 * len(V)
    K = sp.coo_matrix((Ke.ravel(), (np.repeat(dof, 6, 1).ravel(), np.tile(dof, (1, 6)).ravel())), (n, n)).tocsr()
    fix = np.zeros(n, bool)
    fix[2 * bn], fix[2 * bn + 1] = True, True
    u = np.zeros(n)
    u[fix] = ub.ravel()[fix]
    free = ~fix
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free][:, fix] @ u[fix]))
    sig = np.einsum("mij,mj->mi", B, u[dof]) @ D.T
    s0 = D @ eps0
    return float(np.abs(sig - s0).max() / np.abs(s0).max())
