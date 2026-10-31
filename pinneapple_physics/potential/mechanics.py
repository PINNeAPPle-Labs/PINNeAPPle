"""Analytical mechanics from a Lagrangian ``L(q, qd)``.

Depth 1 of the tree gives the generalised momenta ``p_i = dL/dqd_i`` and forces ``Q_i = dL/dq_i``; depth 2 the mass
matrix ``M_ij = d2L/dqd_i dqd_j`` and the coupling ``d2L/dqd_i dq_j``. The Euler-Lagrange equations
``d/dt dL/dqd = dL/dq`` give ``M qdd = Q - (d2L/dqd dq) qd`` (:func:`accelerations`), and the Hamiltonian is the
Legendre transform ``H(q, p) = p qd - L`` (:func:`hamiltonian`), defined where ``L`` is convex in ``qd``
(positive-definite mass matrix).

:func:`double_pendulum` is the planar double pendulum of two point masses on massless rods (angles from the
downward vertical); its equations of motion in closed form are in Goldstein, Poole & Safko, *Classical Mechanics*,
3rd ed. (2002), and in most textbooks.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .core import Potential, Response
from .legendre import LegendreResult, legendre

__all__ = ["lagrangian", "double_pendulum", "accelerations", "hamiltonian", "simulate", "energy"]


def _energy(qd: Sequence[str]):
    def E(d, x):
        return sum(x[v] * d[(v,)] for v in qd) - d[()]
    return E


def lagrangian(fn=None, q: Sequence[str] = ("q",), qd: Sequence[str] = ("qd",), *, expr=None, symbols=None,
               name: str = "L", q_units: Sequence[str] | None = None) -> Potential:
    """A Lagrangian with names ``p_i = dL/dqd_i``, ``Q_i = dL/dq_i`` and the energy ``E = qd . p - L``."""
    q, qd = list(q), list(qd)
    qu = list(q_units) if q_units else ["rad"] * len(q)
    units = {**dict(zip(q, qu, strict=True)), **{b: f"{u}/s" for b, u in zip(qd, qu, strict=True)}}
    names = {}
    for i, (a, b) in enumerate(zip(q, qd, strict=False), start=1):
        names[(b,)] = (f"p{i}", 1, None, f"generalised momentum dL/d{b}")
        names[(a,)] = (f"Q{i}", 1, None, f"generalised force dL/d{a}")
    rel = {(a, b): "Euler-Lagrange coupling symmetry" for a in q for b in qd}
    kw = dict(name=name, unit="J", names=names, relation_names=rel,
              responses={"E": Response(_energy(qd), 1, "J", "energy qd . dL/dqd - L")})
    if expr is not None:
        return Potential.from_sympy(expr, {s: units[str(s)] for s in symbols}, **kw)
    return Potential(fn, {v: units[v] for v in q + qd}, **kw)


def double_pendulum(m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.81) -> Potential:
    """``L = (m1 + m2) l1^2 qd1^2 / 2 + m2 l2^2 qd2^2 / 2 + m2 l1 l2 qd1 qd2 cos(q1 - q2)
    + (m1 + m2) g l1 cos q1 + m2 g l2 cos q2`` (SymPy, so the tree is also symbolic)."""
    import sympy as sp

    q1, q2, w1, w2 = sp.symbols("q1 q2 qd1 qd2", real=True)
    L = ((m1 + m2) * l1 ** 2 * w1 ** 2 / 2 + m2 * l2 ** 2 * w2 ** 2 / 2 + m2 * l1 * l2 * w1 * w2 * sp.cos(q1 - q2)
         + (m1 + m2) * g * l1 * sp.cos(q1) + m2 * g * l2 * sp.cos(q2))
    return lagrangian(q=("q1", "q2"), qd=("qd1", "qd2"), expr=L, symbols=(q1, q2, w1, w2))


def _split(L: Potential, q, qd):
    q = list(q) if q else [v for v in L.variables if not v.startswith("qd")]
    qd = list(qd) if qd else [v for v in L.variables if v.startswith("qd")]
    return q, qd


def accelerations(L: Potential, point, q: Sequence[str] | None = None, qd: Sequence[str] | None = None
                  ) -> np.ndarray:
    """``qdd`` from the Euler-Lagrange equations ``M qdd = dL/dq - (d2L/dqd dq) qd`` at ``point``; shape
    ``batch + (n,)``."""
    q, qd = _split(L, q, qd)
    z, bshape = L._prepare(point)
    d, _ = L.derivatives(z, 2)
    x = L.unflatten(z)
    n = len(q)
    M = torch.empty(z.shape[0], n, n, dtype=z.dtype)
    rhs = torch.empty(z.shape[0], n, dtype=z.dtype)
    for i in range(n):
        rhs[:, i] = d[(q[i],)] - sum(d[L.key((qd[i], q[j]))] * x[qd[j]] for j in range(n))
        for j in range(n):
            M[:, i, j] = d[L.key((qd[i], qd[j]))]
    a = torch.linalg.solve(M, rhs[..., None])[..., 0].detach().numpy()
    return a.reshape(bshape + (n,)) if bshape else a.reshape(n)


def energy(L: Potential, point) -> np.ndarray:
    """``E = qd . dL/dqd - L`` at ``point``."""
    from .core import tree

    return tree(L, 1, at=point, symbolic=False)["E"]


def hamiltonian(L: Potential, q: Sequence[str] | None = None, qd: Sequence[str] | None = None,
                bounds=None, guess=None, method: str = "auto") -> LegendreResult:
    """``H(q, p) = p qd - L`` by Legendre transform in the velocities (``qd_i -> p_i``)."""
    q, qd = _split(L, q, qd)
    swap = {b: f"p{i}" for i, b in enumerate(qd, start=1)}
    guess = guess or {b: 0.0 for b in qd}
    return legendre(L, swap, negate=True, name="H", bounds=bounds, guess=guess, method=method,
                    names={(f"p{i}",): (b, 1, L.variables[b].unit, f"velocity dH/dp{i}")
                           for i, b in enumerate(qd, start=1)})


def simulate(L: Potential, q0: Sequence[float], qd0: Sequence[float], t_end: float, dt: float,
             q: Sequence[str] | None = None, qd: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Integrate the Euler-Lagrange equations with classical fourth-order Runge-Kutta; returns ``t``, ``q``,
    ``qd`` (one row per step) and the energy ``E``."""
    q, qd = _split(L, q, qd)
    n = len(q)
    steps = int(round(t_end / dt))
    y = np.concatenate([np.asarray(q0, float), np.asarray(qd0, float)])

    def f(y):
        pt = {**{v: y[i] for i, v in enumerate(q)}, **{v: y[n + i] for i, v in enumerate(qd)}}
        return np.concatenate([y[n:], accelerations(L, pt, q, qd)])

    Y = np.empty((steps + 1, 2 * n))
    Y[0] = y
    for k in range(steps):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[k + 1] = y
    pts = {**{v: Y[:, i] for i, v in enumerate(q)}, **{v: Y[:, n + i] for i, v in enumerate(qd)}}
    return {"t": dt * np.arange(steps + 1), "q": Y[:, :n], "qd": Y[:, n:], "E": energy(L, pts)}
