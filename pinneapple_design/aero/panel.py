"""Hess-Smith panel method: incompressible, inviscid (potential) flow around an airfoil.

Constant-strength source panels plus one vortex strength shared by every panel; the flow-tangency condition at each
panel's midpoint and the Kutta condition (equal and opposite tangential velocities on the two trailing-edge panels)
give N + 1 equations for the N source strengths and the vortex strength. Written from the textbook formulation
(Hess & Smith 1967; Moran, "An Introduction to Theoretical and Computational Aerodynamics", 1984, ch. 4).

It is the inviscid reference for the airfoil benchmark: what any potential-flow model, a PINN included, can at best
reach (Cl without viscous decambering, zero drag by d'Alembert's paradox), and a check that such a model has solved
its own equations. Units: chord 1, free stream 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import surfaces

__all__ = ["PanelResult", "panel_solve", "airfoil_nodes"]


def airfoil_nodes(shape: np.ndarray, n_per_side: int = 100, close_te: bool = False) -> np.ndarray:
    """Clockwise panel nodes: lower surface trailing edge -> leading edge, then upper leading edge -> trailing edge,
    cosine-clustered at both ends. The blunt trailing edge of the CST shapes stays open (a gap of 0.25 % chord)
    unless ``close_te``: then both end nodes move to the mid-point of the gap. With the gap open the Kutta condition
    acts on panels that shrink below the gap as the panels are refined and Cl drifts down by about 0.8 % per
    doubling (NACA 2412, 4 deg: 0.728, 0.723, 0.717, 0.712, 0.706 for 50-800 panels per side); closed, it converges
    (0.745, 0.734, 0.731, 0.7296, 0.7292)."""
    b = np.linspace(0, np.pi, n_per_side + 1)
    x = 0.5 * (1 - np.cos(b))
    yu, yl = surfaces(shape, x)
    lower = np.stack([x[::-1], yl[::-1]], 1)
    upper = np.stack([x[1:], yu[1:]], 1)
    nodes = np.concatenate([lower, upper])
    if close_te:
        te = 0.5 * (nodes[0] + nodes[-1])
        nodes[0] = nodes[-1] = te
    return nodes


def _influence(px, py, x1, y1, x2, y2):
    """Velocities at points (P,) from unit-strength source and vortex panels (S,): four (P, S) arrays (us, vs, uv, vv)."""
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx                                             # left of the direction of travel: outward
    rx, ry = px[:, None] - x1[None], py[:, None] - y1[None]
    xl = rx * tx + ry * ty
    yl = rx * nx + ry * ny
    r1 = np.hypot(xl, yl)
    r2 = np.hypot(xl - L, yl)
    beta = np.arctan2(yl, xl - L) - np.arctan2(yl, xl)
    beta = np.where(beta > np.pi, beta - 2 * np.pi, np.where(beta < -np.pi, beta + 2 * np.pi, beta))
    lnr = np.log(np.maximum(r1, 1e-14) / np.maximum(r2, 1e-14))
    us_l, vs_l = lnr / (2 * np.pi), beta / (2 * np.pi)            # source panel, panel coordinates
    uv_l, vv_l = vs_l, -us_l                                     # vortex panel: the source field turned by 90 deg
    to_g = lambda u, v: (u * tx + v * nx, u * ty + v * ny)       # noqa: E731
    return (*to_g(us_l, vs_l), *to_g(uv_l, vv_l))


@dataclass
class PanelResult:
    nodes: np.ndarray
    sigma: np.ndarray
    gamma: float
    alpha_deg: float
    x_mid: np.ndarray
    y_mid: np.ndarray
    cp: np.ndarray
    cl_kutta_joukowski: float
    cl_pressure: float
    cd_pressure: float
    cm_quarter: float

    def velocity(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocity (u, v) at arbitrary points outside the airfoil."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        a = math.radians(self.alpha_deg)
        n = self.nodes
        us, vs, uv, vv = _influence(x, y, n[:-1, 0], n[:-1, 1], n[1:, 0], n[1:, 1])
        return (math.cos(a) + us @ self.sigma + self.gamma * uv.sum(1),
                math.sin(a) + vs @ self.sigma + self.gamma * vv.sum(1))


def panel_solve(shape: np.ndarray | None = None, alpha_deg: float = 0.0, *, nodes: np.ndarray | None = None,
                n_per_side: int = 100, close_te: bool = False) -> PanelResult:
    """Solve the potential flow around an airfoil given by CST ``shape`` (or by clockwise ``nodes``);
    ``close_te``: close the blunt CST trailing edge (see :func:`airfoil_nodes`), needed for a grid-converged Cl."""
    n = airfoil_nodes(shape, n_per_side, close_te) if nodes is None else np.asarray(nodes, float)
    x1, y1, x2, y2 = n[:-1, 0], n[:-1, 1], n[1:, 0], n[1:, 1]
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    L = np.hypot(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -ty, tx
    a = math.radians(alpha_deg)
    us, vs, uv, vv = _influence(xm, ym, x1, y1, x2, y2)
    N = len(xm)
    # own panel, control point on the outer side: beta = pi exactly, ln term 0
    i = np.arange(N)
    us[i, i] = 0.5 * nx
    vs[i, i] = 0.5 * ny
    uv[i, i] = 0.5 * tx
    vv[i, i] = 0.5 * ty
    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)
    A[:N, :N] = us * nx[:, None] + vs * ny[:, None]
    A[:N, N] = (uv * nx[:, None] + vv * ny[:, None]).sum(1)
    rhs[:N] = -(math.cos(a) * nx + math.sin(a) * ny)
    # Kutta: tangential velocities on the first (lower) and last (upper) trailing-edge panels cancel
    for k in (0, N - 1):
        A[N, :N] += us[k] * tx[k] + vs[k] * ty[k]
        A[N, N] += (uv[k] * tx[k] + vv[k] * ty[k]).sum()
        rhs[N] -= math.cos(a) * tx[k] + math.sin(a) * ty[k]
    sol = np.linalg.solve(A, rhs)
    sigma, gamma = sol[:N], float(sol[N])
    vt = (us * tx[:, None] + vs * ty[:, None]) @ sigma + gamma * (uv * tx[:, None] + vv * ty[:, None]).sum(1) \
        + math.cos(a) * tx + math.sin(a) * ty
    cp = 1 - vt ** 2
    fx, fy = -(cp * nx * L).sum(), -(cp * ny * L).sum()           # force per unit dynamic pressure
    cl_p = -fx * math.sin(a) + fy * math.cos(a)
    cd_p = fx * math.cos(a) + fy * math.sin(a)
    cm = ((xm - 0.25) * (cp * ny * L) - ym * (cp * nx * L)).sum()  # nose-up positive about the quarter chord
    return PanelResult(nodes=n, sigma=sigma, gamma=gamma, alpha_deg=alpha_deg, x_mid=xm, y_mid=ym, cp=cp,
                       cl_kutta_joukowski=float(2 * gamma * L.sum()), cl_pressure=float(cl_p), cd_pressure=float(cd_p),
                       cm_quarter=float(cm))
