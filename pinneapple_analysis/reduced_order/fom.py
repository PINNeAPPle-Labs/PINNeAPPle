"""Full-order models (FOM) for the reduced-order-modeling workflow: 1-D periodic transport on ``[0, 2*pi)``.

``du/dt = -c u_x + mu u_xx + q (-1/2 (u^2)_x)``, second-order central differences on ``n`` points, so the linear
operator is affine in the parameter (``L(mu) = L0 + mu L1``) and the nonlinearity is a quadratic of the state:
exactly the structure POD-Galerkin can project and operator inference can learn.

* ``advection_diffusion``: ``c = 1``, ``q = 0`` (linear; ``mu`` is the diffusivity).
* ``burgers``: ``c = 0``, ``q = 1`` (viscous Burgers in conservative form; ``mu`` is the viscosity).

Both conserve the mass ``sum(u) dx`` exactly in the discrete model (the columns of ``D1`` and ``D2`` sum to zero),
which is the invariant the workflow checks in every reduced model. Time stepping is IMEX: Crank-Nicolson on the
linear part, second-order Adams-Bashforth on the quadratic part (forward Euler on the first step). The reduced
models use the same scheme and step, so their error against the FOM is the reduction error, not a scheme error.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _periodic_diff(n: int, dx: float) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    e = np.ones(n)
    d1 = sp.diags([-e[:-1], e[:-1]], [-1, 1], shape=(n, n), format="lil")
    d1[0, n - 1], d1[n - 1, 0] = -1.0, 1.0
    d2 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], shape=(n, n), format="lil")
    d2[0, n - 1], d2[n - 1, 0] = 1.0, 1.0
    return (d1.tocsc() / (2 * dx)), (d2.tocsc() / dx**2)


def imex_rollout(u0: np.ndarray, lin: np.ndarray | sp.spmatrix, quad: Callable[[np.ndarray], np.ndarray] | None,
                 dt: float, steps: int, save_every: int) -> np.ndarray:
    """IMEX CN/AB2 integration of ``du/dt = lin @ u + quad(u)``; returns the saved states ``(steps//save_every+1, n)``.

    Works for the sparse FOM operator and for a dense reduced operator alike (the same scheme for both)."""
    n = u0.size
    if sp.issparse(lin):
        eye = sp.identity(n, format="csc")
        lu = spla.splu((eye - 0.5 * dt * lin).tocsc())
        solve = lu.solve
        rhs_op = (eye + 0.5 * dt * lin).tocsr()
    else:
        import scipy.linalg as sla
        eye = np.eye(n)
        fac = sla.lu_factor(eye - 0.5 * dt * lin)

        def solve(b):
            return sla.lu_solve(fac, b)
        rhs_op = eye + 0.5 * dt * lin
    out = [u0.copy()]
    u = u0.copy()
    q_prev = None
    for k in range(1, steps + 1):
        rhs = rhs_op @ u
        if quad is not None:
            qn = quad(u)
            rhs = rhs + dt * (qn if q_prev is None else 1.5 * qn - 0.5 * q_prev)
            q_prev = qn
        u = solve(rhs)
        if k % save_every == 0:
            out.append(u.copy())
    return np.array(out)


@dataclass
class PeriodicTransportFOM:
    """The FOM: ``n`` grid points, step ``dt``, saved every ``save_every`` steps up to ``t_end``."""

    kind: str = "burgers"
    n: int = 512
    dt: float = 1e-3
    t_end: float = 2.0
    save_every: int = 10
    c: float = 0.0
    q: float = 1.0
    x: np.ndarray = field(init=False, repr=False)
    D1: sp.csc_matrix = field(init=False, repr=False)
    D2: sp.csc_matrix = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = 2 * np.pi / self.n
        self.x = np.arange(self.n) * self.dx
        self.D1, self.D2 = _periodic_diff(self.n, self.dx)

    @property
    def steps(self) -> int:
        return int(round(self.t_end / self.dt))

    @property
    def times(self) -> np.ndarray:
        return np.arange(0, self.steps + 1, self.save_every) * self.dt

    @property
    def snapshot_dt(self) -> float:
        return self.dt * self.save_every

    def u0(self) -> np.ndarray:
        return 1.0 + 0.5 * np.sin(self.x) + 0.25 * np.cos(2 * self.x)

    def L0(self) -> sp.csc_matrix:
        return (-self.c * self.D1).tocsc()

    def L1(self) -> sp.csc_matrix:
        return self.D2

    def quad(self, u: np.ndarray) -> np.ndarray:
        return -0.5 * self.q * (self.D1 @ (u * u))

    def mass(self, U: np.ndarray) -> np.ndarray:
        """The invariant: ``sum(u) dx`` of each state (rows of ``U``)."""
        return np.asarray(U).sum(axis=-1) * self.dx

    def solve(self, mu: float, t_end: float | None = None) -> np.ndarray:
        steps = self.steps if t_end is None else int(round(t_end / self.dt))
        lin = (self.L0() + mu * self.L1()).tocsc()
        return imex_rollout(self.u0(), lin, self.quad if self.q else None, self.dt, steps, self.save_every)


def make_fom(kind: str = "burgers", **kw) -> PeriodicTransportFOM:
    """``advection_diffusion`` (linear, ``c = 1``) or ``burgers`` (quadratic). ``kw`` overrides n, dt, t_end, ..."""
    if kind == "advection_diffusion":
        return PeriodicTransportFOM(kind=kind, c=1.0, q=0.0, **kw)
    if kind == "burgers":
        return PeriodicTransportFOM(kind=kind, c=0.0, q=1.0, **kw)
    raise ValueError(f"unknown FOM {kind!r}: use 'advection_diffusion' or 'burgers'")
