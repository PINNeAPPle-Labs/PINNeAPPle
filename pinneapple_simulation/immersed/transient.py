"""Transient heat conduction on an immersed discretisation: rho c dT/dt - div(k grad T) = f(X, t), by the theta
scheme (theta = 1 backward Euler, 1/2 Crank-Nicolson):

    (M / dt + theta A) T^{n+1} = (M / dt - (1 - theta) A) T^n + theta b^{n+1} + (1 - theta) b^n

A and b are the steady operator and load of :mod:`.heat` (Nitsche, flux, convection, small-cut stabilisation). The
mass matrix M is integrated with the same cut-cell rule; with the ghost penalty its small-cut conditioning is
controlled by a mass ghost term gamma_M rho c h^2 (the gradient ghost faces scaled by h^2), with the finite cell
method by the fictitious material. Sources and boundary data may depend on time: pass ``source(X, t)`` and
``bcs(t)`` (a function returning the list of conditions); constant data are assembled once.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from .assembly import Discretization, ghost_matrix, reference_cell, scatter
from .heat import assemble_heat, field_errors


def mass_matrix(disc: Discretization, rho_c: float = 1.0, stabilization: str = "ghost", ghost_mass: float = 0.01,
                alpha: float = 1e-6) -> sparse.csr_matrix:
    n = disc.n_dofs
    ref = reference_cell(disc.grid)
    inside = disc.quad.inside_cells
    M = scatter(disc.dofs.dofs_of(inside), np.broadcast_to(rho_c * ref["M"], (len(inside),) + ref["M"].shape), n)
    cut, W = disc.cut_weights()
    N, _, _ = disc.fitted_reference
    nb = N.shape[1]
    Mc = rho_c * (W @ np.einsum("qi,qj->qij", N, N).reshape(len(N), -1)).reshape(len(cut), nb, nb)
    if stabilization == "fcm":
        Mc = (1 - alpha) * Mc + alpha * rho_c * ref["M"][None]
    M = M + scatter(disc.dofs.dofs_of(cut), Mc, n)
    if stabilization == "ghost" and ghost_mass:
        M = M + ghost_mass * rho_c * disc.grid.size ** 2 * ghost_matrix(disc)
    return sparse.csr_matrix(M)


@dataclass
class TransientSolution:
    disc: Discretization
    times: np.ndarray
    states: list                       # recorded fields
    info: dict = field(default_factory=dict)

    @property
    def T(self) -> np.ndarray:
        return self.states[-1]

    def __call__(self, X: np.ndarray, k: int = -1) -> np.ndarray:
        return self.disc.evaluate(self.states[k], X)

    def errors(self, exact: Callable, grad: Callable | None = None, k: int = -1) -> dict:
        t = self.times[k]
        return field_errors(self.disc, self.states[k], lambda X: exact(X, t),
                            None if grad is None else (lambda X: grad(X, t)))


def solve_transient_heat(disc: Discretization, conductivity: float, rho_c: float, source: Any = 0.0,
                         bcs: Sequence | Callable = (), T0: Any = 0.0, dt: float = 1e-2, n_steps: int = 10,
                         theta: float = 0.5, stabilization: str = "ghost", record_every: int = 0,
                         solver: str = "direct") -> TransientSolution:
    """March ``n_steps`` of size ``dt`` from ``T0`` (a constant, or a callable of X interpolated at the nodes).
    ``source`` may be ``f(X)`` or ``f(X, t)``; ``bcs`` a list or ``bcs(t)``. The matrix is factorised once."""
    from scipy.sparse.linalg import splu

    from .bcs import _arity
    t0 = time.perf_counter()
    td_src = callable(source) and _arity(source) >= 2
    td_bcs = callable(bcs)

    def load(t):
        f = (lambda X: source(X, t)) if td_src else source
        return assemble_heat(disc, conductivity, f, bcs(t) if td_bcs else bcs, stabilization)

    A, b_prev, _ = load(0.0)
    M = mass_matrix(disc, rho_c, stabilization)
    if callable(T0):
        # nodal interpolation (T0 must be defined at the nodes of the active cells, outside the part included); an
        # L2 projection puts round-off-sized noise on small-cut unknowns that Crank-Nicolson does not damp
        # (measured on the 2-D manufactured case: the error then stays at the projection's 1.4e-3 for every dt)
        T = np.asarray(T0(disc.dofs.coords()), float)
    else:
        T = np.full(disc.n_dofs, float(T0))
    lhs = splu(sparse.csc_matrix(M / dt + theta * A))
    rhs_op = sparse.csr_matrix(M / dt - (1 - theta) * A)
    times, states = [0.0], [T.copy()]
    t_asm = time.perf_counter() - t0
    for k in range(1, n_steps + 1):
        t = k * dt
        b_new = load(t)[1] if (td_src or td_bcs) else b_prev
        T = lhs.solve(rhs_op @ T + theta * b_new + (1 - theta) * b_prev)
        b_prev = b_new
        if record_every and k % record_every == 0 or k == n_steps:
            times.append(t)
            states.append(T.copy())
    info = {"dofs": disc.n_dofs, "steps": n_steps, "dt": dt, "theta": theta, "setup_s": t_asm,
            "total_s": time.perf_counter() - t0, "quadrature_s": disc.timings.get("quadrature_s", 0.0)}
    return TransientSolution(disc, np.array(times), states, info)
