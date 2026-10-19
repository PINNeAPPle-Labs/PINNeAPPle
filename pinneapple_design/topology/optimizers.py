"""Design-update rules: optimality criteria (OC) and the method of moving asymptotes (MMA).

OC: the classical fixed-point update for one volume constraint and a non-increasing objective
(compliance-type problems), with a bisection on the Lagrange multiplier.

MMA: K. Svanberg, "The method of moving asymptotes -- a new method for structural optimization",
Int. J. Numer. Meth. Eng. 24 (1987) 359-373. Each iteration replaces the objective and the
constraints ``f_i(x) <= 0`` by convex separable approximations ``r_i + sum p_ij/(U_j - x_j) + q_ij/(x_j - L_j)``
and solves the subproblem through its dual, a smooth concave maximization over ``lambda >= 0``
(here with L-BFGS-B). Elastic variables ``y_i >= 0`` with cost ``c y + y^2/2`` keep the subproblem
feasible (the dual then stays bounded). The coefficient and asymptote-update formulas follow the
paper and Svanberg's later public note on MMA/GCMMA (2007): ``p = (U-x)^2 (1.001 g+ + 0.001 g- + eps)``,
``q = (x-L)^2 (0.001 g+ + 1.001 g- + eps)``, asymptote factors 0.7 / 1.2 on oscillation / monotone steps.
This implementation is written from those formulas, not from any existing MMA code.
"""
from __future__ import annotations

import numpy as np

__all__ = ["oc_update", "MMA"]


def oc_update(x: np.ndarray, dfdx: np.ndarray, dvdx: np.ndarray, volume_target: float,
              volume_of, move: float = 0.2, eta: float = 0.5, xmin: float = 0.0, xmax: float = 1.0):
    """One OC step: ``x_new = clip(x (max(-df/dx, 0) / (lambda dv/dx))^eta)`` with ``volume_of(x_new) =
    volume_target`` found by bisection on ``lambda``. ``volume_of`` maps design variables to the volume."""
    g = np.maximum(-dfdx, 1e-30)
    lo, hi = 0.0, 1e9
    xn = x
    while (hi - lo) / (hi + lo + 1e-300) > 1e-6:
        lam = 0.5 * (lo + hi)
        xn = np.clip(x * (g / (lam * dvdx)) ** eta, np.maximum(xmin, x - move), np.minimum(xmax, x + move))
        if volume_of(xn) > volume_target:
            lo = lam
        else:
            hi = lam
    return xn


class MMA:
    """Method of moving asymptotes for ``min f0(x)  s.t.  f_i(x) <= 0,  xmin <= x <= xmax``."""

    def __init__(self, n: int, m: int, xmin=0.0, xmax=1.0, move: float = 0.5, asyinit: float = 0.5,
                 asyincr: float = 1.2, asydecr: float = 0.7, c: float = 1000.0, d: float = 1.0):
        self.n, self.m = n, m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.move, self.asyinit, self.asyincr, self.asydecr = move, asyinit, asyincr, asydecr
        self.c = np.full(m, c)
        self.d = np.full(m, d)
        self.k = 0
        self.xold1 = self.xold2 = None
        self.low = self.upp = None

    def _asymptotes(self, x):
        dx = self.xmax - self.xmin
        if self.k < 2:
            self.low = x - self.asyinit * dx
            self.upp = x + self.asyinit * dx
        else:
            s = (x - self.xold1) * (self.xold1 - self.xold2)
            gam = np.where(s > 0, self.asyincr, np.where(s < 0, self.asydecr, 1.0))
            self.low = x - gam * (self.xold1 - self.low)
            self.upp = x + gam * (self.upp - self.xold1)
            self.low = np.clip(self.low, x - 10 * dx, x - 0.01 * dx)
            self.upp = np.clip(self.upp, x + 0.01 * dx, x + 10 * dx)

    def update(self, x: np.ndarray, f0: float, df0: np.ndarray, f: np.ndarray, df: np.ndarray) -> np.ndarray:
        """New design from values and gradients at ``x``: ``f`` (m,), ``df`` (m, n)."""
        from scipy.optimize import minimize

        x = np.asarray(x, float)
        f = np.atleast_1d(np.asarray(f, float))
        df = np.atleast_2d(np.asarray(df, float))
        self._asymptotes(x)
        L, U = self.low, self.upp
        dx = self.xmax - self.xmin
        alpha = np.maximum.reduce([self.xmin, L + 0.1 * (x - L), x - self.move * dx])
        beta = np.minimum.reduce([self.xmax, U - 0.1 * (U - x), x + self.move * dx])
        eps = 1e-5 / np.maximum(dx, 1e-12)
        ux2, xl2 = (U - x) ** 2, (x - L) ** 2

        def pq(g):
            gp, gm = np.maximum(g, 0.0), np.maximum(-g, 0.0)
            return ux2 * (1.001 * gp + 0.001 * gm + eps), xl2 * (0.001 * gp + 1.001 * gm + eps)

        p0, q0 = pq(np.asarray(df0, float))
        P = np.empty((self.m, self.n))
        Q = np.empty((self.m, self.n))
        for i in range(self.m):
            P[i], Q[i] = pq(df[i])
        r0 = f0 - np.sum(p0 / (U - x) + q0 / (x - L))
        r = f - (P / (U - x)).sum(1) - (Q / (x - L)).sum(1)

        def primal(lam):
            Pj = p0 + lam @ P
            Qj = q0 + lam @ Q
            sp_, sq = np.sqrt(Pj), np.sqrt(Qj)
            xs = np.clip((sp_ * L + sq * U) / (sp_ + sq), alpha, beta)
            return xs, Pj, Qj

        def negdual(lam):
            xs, Pj, Qj = primal(lam)
            y = np.maximum(0.0, (lam - self.c) / self.d)
            W = (r0 + lam @ r + np.sum(Pj / (U - xs) + Qj / (xs - L))
                 + np.sum(self.c * y + 0.5 * self.d * y * y - lam * y))
            grad = r + (P / (U - xs)).sum(1) + (Q / (xs - L)).sum(1) - y
            return -W, -grad

        res = minimize(negdual, np.ones(self.m), jac=True, method="L-BFGS-B",
                       bounds=[(0.0, None)] * self.m, options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        xnew, _, _ = primal(res.x)
        self.xold2, self.xold1 = self.xold1, x.copy()
        self.k += 1
        self.lam = res.x
        return xnew
