"""Steady coupling of two physics participants through named interface fields (#1209).

A :class:`Participant` owns a discrete state and exposes three functions:

* ``solve(input) -> state``: its own solver, given the other participant's interface output;
* ``residual(state, input) -> r``: its discrete equations (used by the monolithic solve);
* ``output(state) -> field``: the interface field it hands to the other participant.

Participant ``a`` reads ``b``'s output and vice versa. The strategies:

* :func:`one_way`: ``a`` sees ``b``'s reference (uncoupled) output, ``b`` sees ``a``'s result; no feedback.
* :func:`gauss_seidel`: two-way partitioned, ``y -> b.output(b.solve(a.output(a.solve(y))))`` relaxed on the field
  ``y`` that ``a`` reads, with Aitken's dynamic relaxation (Irons & Tuck 1969; Kuttler & Wall 2008, eq. 44)
  or a fixed factor. The interface residual history is returned.
* :func:`monolithic`: both residuals solved together for both states (``scipy.optimize.root``, Powell hybrid with
  a finite-difference Jacobian), started from the one-way states.

:func:`coupling_strength` measures the contraction factor of the plain (unrelaxed) Gauss-Seidel map from its first
iterations (a power-iteration estimate of the spectral radius of the interface Jacobian), and
:func:`recommend_strategy` turns it and the measured one-way error into the strategy to use.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Participant:
    name: str
    field_in: str
    field_out: str
    solve: Callable[[np.ndarray], np.ndarray]
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray]
    output: Callable[[np.ndarray], np.ndarray]


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def one_way(a: Participant, b: Participant, y_ref: np.ndarray) -> dict[str, Any]:
    """``a`` solved with ``b``'s reference output ``y_ref``; ``b`` solved with ``a``'s result."""
    sa = a.solve(y_ref)
    sb = b.solve(a.output(sa))
    return {"strategy": "one_way", "state_a": sa, "state_b": sb, "iterations": 1, "converged": True,
            "residual_history": []}


def gauss_seidel(a: Participant, b: Participant, y0: np.ndarray, *, aitken: bool = True, omega0: float = 0.5,
                 omega: float = 1.0, tol: float = 1e-10, max_iter: int = 200) -> dict[str, Any]:
    """Two-way partitioned coupling; ``tol`` on ``|r_k| / |y_k|`` (``r_k = G(y_k) - y_k``)."""
    y = np.asarray(y0, float).copy()
    hist: list[float] = []
    r_prev = None
    w = float(omega0 if aitken else omega)
    omegas: list[float] = []
    sa = sb = None
    converged = False
    k = 0
    while k < max_iter:
        k += 1
        sa = a.solve(y)
        sb = b.solve(a.output(sa))
        r = b.output(sb) - y
        rel = _norm(r) / max(_norm(y), 1e-300)
        hist.append(rel)
        if not np.isfinite(rel):
            break
        if rel <= tol:
            converged = True
            break
        if aitken and r_prev is not None:
            dr = r - r_prev
            den = float(dr @ dr)
            if den > 0:
                w = -w * float(r_prev @ dr) / den
        omegas.append(w)
        y = y + w * r
        r_prev = r
    return {"strategy": "gauss_seidel_aitken" if aitken else f"gauss_seidel_omega_{omega:g}", "state_a": sa,
            "state_b": sb, "iterations": k, "converged": converged, "residual_history": hist,
            "relaxation_history": omegas}


def monolithic(a: Participant, b: Participant, sa0: np.ndarray, sb0: np.ndarray, *, tol: float = 1e-12
               ) -> dict[str, Any]:
    """Both discrete residuals solved together, started from ``(sa0, sb0)``."""
    from scipy.optimize import root
    na = sa0.size

    def F(z):
        sa, sb = z[:na], z[na:]
        return np.concatenate([a.residual(sa, b.output(sb)), b.residual(sb, a.output(sa))])
    z0 = np.concatenate([sa0, sb0])
    scale0 = max(_norm(F(z0)), 1e-300)
    sol = root(F, z0, method="hybr", tol=tol)
    res = _norm(F(sol.x))
    return {"strategy": "monolithic", "state_a": sol.x[:na], "state_b": sol.x[na:], "iterations": int(sol.nfev),
            "converged": bool(sol.success) or res <= 1e-9 * scale0, "residual_norm": res,
            "initial_residual_norm": scale0, "message": str(sol.message), "residual_history": []}


def coupling_strength(a: Participant, b: Participant, y0: np.ndarray, n: int = 6) -> dict[str, Any]:
    """Contraction factor of the unrelaxed Gauss-Seidel map, from the ratios of its first interface corrections."""
    y = np.asarray(y0, float).copy()
    steps = []
    for _ in range(n):
        y_new = b.output(b.solve(a.output(a.solve(y))))
        steps.append(_norm(y_new - y))
        y = y_new
        if not np.all(np.isfinite(y)):
            break
    floor = 1e-11 * max(_norm(y0), 1e-300)          # corrections below this are round-off, not the map
    ratios = [steps[i + 1] / steps[i] for i in range(len(steps) - 1) if steps[i + 1] > floor]
    rho = float(ratios[-1]) if ratios else 0.0
    feedback = steps[1] / max(_norm(y0), 1e-300) if len(steps) > 1 else 0.0
    return {"contraction": rho, "ratios": ratios, "first_corrections": steps,
            "feedback": float(feedback)}


def recommend_strategy(contraction: float, one_way_error: float, one_way_tolerance: float) -> tuple[str, str]:
    """``one_way`` when its measured error is within the tolerance; partitioned Gauss-Seidel with Aitken while the
    plain map contracts (rho < 1); monolithic otherwise."""
    if one_way_error <= one_way_tolerance:
        return "one_way", (f"one-way differs from the two-way solution by {one_way_error:.3g} "
                           f"(<= {one_way_tolerance:g}): the feedback is negligible")
    if contraction < 1.0:
        return "gauss_seidel_aitken", (f"two-way needed (one-way error {one_way_error:.3g}); the plain partitioned "
                                       f"map contracts (rho = {contraction:.3g} < 1), so Gauss-Seidel with Aitken "
                                       "relaxation converges and reuses each solver as is")
    return "monolithic", (f"two-way needed (one-way error {one_way_error:.3g}) and the plain partitioned map does "
                          f"not contract (rho = {contraction:.3g} >= 1): solve both residuals together")
