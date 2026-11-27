"""PoNN and BeNN: physics-informed optimal control and guidance.

Two ways of getting an optimal control out of a differential equation instead of out of a
shooting loop around a simulator:

* **PoNN** (:func:`pontryagin`) -- the Pontryagin Minimum Principle turns the problem into a
  two-point boundary value problem in the state ``x(t)`` and the costate ``lambda(t)``, solved here
  with X-TFC: the same fixed random ELM basis, Chebyshev-Gauss-Lobatto collocation and analytic
  basis derivatives that :mod:`pinneapple_simulation.numerical_solvers.xtfc_ivp` uses, with its
  single-initial-condition constrained expression generalised to a *mixed* boundary -- variables
  pinned at ``t0`` (the state), at ``tf`` (the costate transversality) or at both ends (a state
  that must arrive).  The boundary conditions hold identically for every coefficient vector, never
  as a penalised loss, exactly as in the IVP module.  For the linear-quadratic problems this module
  covers, the collocation residual is affine in the coefficients, so one least-squares solve is the
  whole solution.
* **BeNN** (:func:`bellman`) -- the Hamilton-Jacobi-Bellman residual is imposed on a value network
  ``V(t, x)`` whose terminal condition ``V(tf, x) = terminal cost`` is a TFC constrained expression
  (the basis is projected onto functions vanishing at ``tf``, so the condition holds to machine
  precision for any coefficients).  The policy comes out of the value gradient,
  ``u = -R^-1 B^T grad V``, so -- unlike PoNN -- it acts on states it was never solved for.

Trust: once a policy is learned, the HJB residual (:meth:`BeNN.hjb_residual`) evaluated along the
trajectory the policy is *actually* flying says whether that policy is still inside the region it
was fitted on: it is small where the fit holds and grows as the state leaves the box.
:func:`landing_demo` uses it to hand the vehicle over to a shrinking-horizon MPC
(:func:`mpc_control`) when it crosses a threshold.

Scope, stated plainly
---------------------
Implemented and checked against closed-form answers: linear-quadratic problems with an affine drift
(``xdot = A x + B u + c``) and a quadratic running cost, which covers the three published case
classes this module claims -- the LQR regulator (Riccati), the energy-optimal 1-D intercept and the
1-D planetary landing.  Not implemented: the non-linear dynamics of the three-dimensional intercept
and closed-loop guidance of the papers the issue names, and no table from those papers is
reproduced.  There is no CasADi/IPOPT here either, so the independent numerical reference is
:func:`direct_transcription` (RK4 plus SLSQP), not the IPOPT transcription the papers compare
against.

Two things are missing on purpose, both because measurement said so, and both spelt out where they
would have been:

* the *stationary* value network -- its HJB residual is satisfied to machine zero by every symmetric
  root of the algebraic Riccati equation, so a residual-only fit cannot tell the value function from
  its negative (see :func:`bellman`);
* a claim of accuracy on the BeNN *value* -- the gain comes out within a per cent of Riccati's and
  the cost to go within about 7 %, but not better, and the trust signal is an ordering of residuals
  rather than a prediction of the error.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, optimize

from pinneapple_simulation.numerical_solvers.xtfc_ivp import (
    ACTIVATIONS,
    _elm_basis,
    chebyshev_gauss_lobatto,
    solve_xtfc_ode,
)

from .tuning import lqr as _care_lqr

Array = np.ndarray

__all__ = [
    "BeNN",
    "LQProblem",
    "MinEnergyTransfer",
    "MinFuelLanding",
    "PoNN",
    "bellman",
    "direct_transcription",
    "landing_demo",
    "min_energy_transfer",
    "min_fuel_landing",
    "mpc_control",
    "pontryagin",
    "residual_report",
    "riccati_solution",
    "steady_state_lqr",
]


# --------------------------------------------------------------------------------------
# the problem
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LQProblem:
    """A control-affine problem with a quadratic cost::

        xdot = A x + B u + c,            x(t0) = x0
        J    = 1/2 int_t0^tf (x' Q x + u' R u) dt
               + 1/2 (x(tf) - xf)' Qf (x(tf) - xf)

    ``terminal_mask`` names the state components that must reach ``xf`` *exactly* at ``tf`` instead
    of paying a terminal cost; that constraint is carried by the constrained expression, so it
    holds to machine precision.  ``Qf = 0`` with no mask entry is the free final state of the
    regulator and of the intercept.

    The half in the cost is the convention that makes the BeNN policy read
    ``u = -R^-1 B^T grad V``.  The same ``P`` solves the algebraic Riccati equation either way, so
    the optimal cost is ``0.5 x0' P x0`` here and ``x0' P x0`` under the no-half convention of
    :func:`pinneapple_systems.control.tuning.lqr`, whose ``P`` and gain ``K`` are reused verbatim as
    the infinite-horizon reference.
    """

    A: Array
    B: Array
    Q: Array
    R: Array
    x0: Sequence[float]
    t_span: tuple[float, float]
    c: Sequence[float] | None = None
    Qf: Array | None = None
    xf: Sequence[float] | None = None
    terminal_mask: Sequence[bool] | None = None
    name: str = ""
    _Rinv: Array | None = field(default=None, repr=False, compare=False)
    _BRt: Array | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # ``A`` is *not* symmetrised: the drift ``A x`` of a mechanical system (``pdot = v``) is the
        # most asymmetric matrix there is, and symmetrising it silently solves another problem.  The
        # cost matrices are, since only their symmetric part enters ``x' Q x``.
        A = np.atleast_2d(np.asarray(self.A, float))
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError(f"A must be square, got {A.shape}")
        B = np.asarray(self.B, float).reshape(n, -1)
        m = B.shape[1]
        Q = _sym(np.atleast_2d(np.asarray(self.Q, float)))
        R = _sym(np.atleast_2d(np.asarray(self.R, float)))
        if Q.shape != (n, n) or R.shape != (m, m):
            raise ValueError(f"Q must be {(n, n)} and R {(m, m)}, got {Q.shape} and {R.shape}")
        if float(np.min(np.linalg.eigvalsh(R))) <= 0.0:
            raise ValueError("R must be positive definite")
        c = np.zeros(n) if self.c is None else np.asarray(self.c, float).reshape(n)
        Qf = (np.zeros((n, n)) if self.Qf is None
              else _sym(np.atleast_2d(np.asarray(self.Qf, float))))
        xf = np.zeros(n) if self.xf is None else np.asarray(self.xf, float).reshape(n)
        mask = (np.zeros(n, bool) if self.terminal_mask is None
                else np.asarray(self.terminal_mask, bool).reshape(n))
        x0 = np.asarray(self.x0, float).reshape(n)
        t0, tf = float(self.t_span[0]), float(self.t_span[1])
        if not tf > t0:
            raise ValueError("t_span must satisfy tf > t0")
        Rinv = np.linalg.inv(R)
        for key, value in (("A", A), ("B", B), ("Q", Q), ("R", R), ("c", c), ("Qf", Qf),
                           ("xf", xf), ("terminal_mask", mask), ("x0", x0), ("_Rinv", Rinv),
                           ("_BRt", Rinv @ B.T)):
            object.__setattr__(self, key, value)
        object.__setattr__(self, "t_span", (t0, tf))

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        """Input dimension."""
        return int(self.B.shape[1])

    @property
    def Rinv(self) -> Array:
        """Inverse of ``R`` (computed once, at construction)."""
        return self._Rinv

    @property
    def constrained(self) -> bool:
        """Whether any state component is pinned at the final time."""
        return bool(np.any(self.terminal_mask))

    def state_rate(self, x: Array, u: Array) -> Array:
        """``xdot = A x + B u + c`` for a batch of states ``x`` (N, n) and inputs ``u`` (N, m)."""
        return x @ self.A.T + u @ self.B.T + self.c

    def costate_rate(self, x: Array, lam: Array) -> Array:
        """``lambdadot = -Q x - A' lambda`` (the affine drift ``c`` has no state derivative)."""
        return -(x @ self.Q.T + lam @ self.A)

    def optimal_control(self, lam: Array) -> Array:
        """``u = -R^-1 B' lambda``, the Hamiltonian minimiser for a quadratic control cost."""
        return -lam @ self._BRt.T

    def running_cost(self, x: Array, u: Array) -> Array:
        """``1/2 (x' Q x + u' R u)`` at each row."""
        return (0.5 * np.einsum("pi,ij,pj->p", x, self.Q, x)
                + 0.5 * np.einsum("pi,ij,pj->p", u, self.R, u))

    def terminal_cost(self, x: Array) -> Array:
        """``1/2 (x - xf)' Qf (x - xf)`` at each row."""
        d = x - self.xf
        return 0.5 * np.einsum("pi,ij,pj->p", d, self.Qf, d)

    def with_state(self, x0: Sequence[float], t_span: tuple[float, float]) -> LQProblem:
        """The same problem re-started from ``x0`` over ``t_span`` (what an MPC step asks for)."""
        return LQProblem(A=self.A, B=self.B, Q=self.Q, R=self.R, x0=x0, t_span=t_span, c=self.c,
                         Qf=self.Qf, xf=self.xf, terminal_mask=self.terminal_mask, name=self.name)


def _sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


# --------------------------------------------------------------------------------------
# X-TFC constrained expression with a per-variable boundary
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Term:
    """One term of a variable's constrained expression.

    ``kind="basis"`` contributes ``sign * phi(x)`` to the coefficient columns of ``block``;
    ``kind="row"`` contributes ``sign * vec(x) * row`` to those same columns; ``kind="row_all"``
    contributes it to *all* the columns, which is what the costate's terminal coupling needs.
    """

    kind: str
    block: int = 0
    sign: float = 1.0
    vec: str = "one"
    row: Array | None = None


_VECTORS: dict[str, tuple[Callable[[Array], Array], Callable[[Array], Array]]] = {
    "one": (lambda x: np.ones_like(x), lambda x: np.zeros_like(x)),
    "ramp": (lambda x: 0.5 * (1.0 + x), lambda x: np.full_like(x, 0.5)),
}


class _MixedExpression:
    """TFC constrained expression with a different boundary condition per variable::

        y_i(x) = (terms of variable i) . beta + const_i(x)

    so the boundary conditions of a two-point problem hold for **every** ``beta``: a variable pinned
    at ``x = -1``, one pinned at ``x = +1``, one pinned at both ends (a linear support function is
    added and the basis is projected onto functions vanishing at both ends), and one whose terminal
    value is a linear combination of the other variables' terminal values -- the costate
    transversality ``lambda(tf) = Qf (x(tf) - xf)``.

    The construction is the one of :func:`pinneapple_simulation.numerical_solvers.xtfc_ivp.
    solve_xtfc_ode` -- the basis, its analytic derivative and the collocation are reused from
    there; only the projection differs, because an initial-value problem has one boundary and a
    Pontryagin problem has two.
    """

    def __init__(self, w: Array, b: Array, activation: str, terms: list[list[_Term]],
                 consts: list[tuple[Callable[[Array], Array], Callable[[Array], Array]]],
                 n_vars: int) -> None:
        self.w, self.b, self.activation = np.asarray(w, float), np.asarray(b, float), activation
        self.terms, self.consts, self.n_vars = terms, consts, n_vars
        self.n_basis = int(self.w.size)
        if len(terms) != n_vars or len(consts) != n_vars:
            raise ValueError("one term list and one constant per variable")

    def phi(self, x: Array) -> tuple[Array, Array]:
        """The X-TFC basis and its analytic derivative at the mapped points ``x``."""
        return _elm_basis(np.asarray(x, float), self.w, self.b, self.activation)

    def assemble(self, x: Array) -> tuple[Array, Array, Array, Array]:
        """``(A, const, dA, dconst)``, per variable ``i``::

            y_i      = A[i] beta + const[:, i]
            dy_i/dx  = dA[i] beta + dconst[:, i]

        with ``A`` of shape ``(n_vars, N, n_basis * n_vars)`` -- one coefficient matrix per output
        variable, since a term of one variable only fills that variable's own row.
        """
        x = np.asarray(x, float)
        n_pts, m = x.size, self.n_basis
        phi, dphi = self.phi(x)
        A = np.zeros((self.n_vars, n_pts, m * self.n_vars))
        dA = np.zeros_like(A)
        const = np.zeros((n_pts, self.n_vars))
        dconst = np.zeros_like(const)
        for i, terms in enumerate(self.terms):
            cols = slice(i * m, (i + 1) * m)
            for term in terms:
                value_fn, deriv_fn = _VECTORS[term.vec]
                if term.kind == "basis":
                    A[i, :, cols] += term.sign * phi
                    dA[i, :, cols] += term.sign * dphi
                elif term.kind == "row":
                    local = slice(term.block * m, (term.block + 1) * m)
                    A[i, :, local] += term.sign * value_fn(x)[:, None] * term.row[None, :]
                    dA[i, :, local] += term.sign * deriv_fn(x)[:, None] * term.row[None, :]
                elif term.kind == "row_all":
                    A[i] += term.sign * value_fn(x)[:, None] * term.row[None, :]
                    dA[i] += term.sign * deriv_fn(x)[:, None] * term.row[None, :]
                else:  # pragma: no cover - unreachable by construction
                    raise ValueError(f"unknown term kind {term.kind!r}")
            value_fn, deriv_fn = self.consts[i]
            const[:, i] = value_fn(x)
            dconst[:, i] = deriv_fn(x)
        return A, const, dA, dconst

    def evaluate(self, x: Array, beta: Array) -> tuple[Array, Array]:
        """``(y, dy/dx)`` at ``x`` for the flattened coefficient vector ``beta``."""
        A, const, dA, dconst = self.assemble(x)
        flat = np.asarray(beta, float).ravel()
        y = np.stack([A[i] @ flat for i in range(self.n_vars)], axis=1) + const
        dy = np.stack([dA[i] @ flat for i in range(self.n_vars)], axis=1) + dconst
        return y, dy


_Const = tuple[Callable[[Array], Array], Callable[[Array], Array]]


def _constant(value: float, slope: float = 0.0) -> _Const:
    """``value + slope * (1 + x) / 2`` as the ``(value, derivative)`` pair the assembler wants."""
    return (lambda x: np.full_like(x, value) + slope * 0.5 * (1.0 + x),
            lambda x: np.full_like(x, 0.5 * slope))


def _bvp_expression(problem: LQProblem, w: Array, b: Array, activation: str) -> _MixedExpression:
    """The constrained expression of the PMP two-point problem.

    Variables ``0..n-1`` are the state, pinned at ``t0`` and -- where ``terminal_mask`` says so --
    also at ``tf``; variables ``n..2n-1`` are the costate, whose terminal condition is the
    transversality ``lambda_i(tf) = sum_j Qf_ij (x_j(tf) - xf_j)`` -- except for the components the
    mask pins, whose costate endpoint is *free*: a hard terminal state constraint carries an
    unknown multiplier ``nu_i`` that enters ``lambda_i(tf)``, so imposing ``lambda_i(tf) = 0`` there
    (what ``Qf = 0`` would say) is a different, unsolvable problem.  Those costate variables get no
    projection at all, only the raw ELM basis.
    """
    n = problem.n
    m = int(np.asarray(w).size)
    phi_lo = _elm_basis(np.array([-1.0]), w, b, activation)[0][0]
    phi_hi = _elm_basis(np.array([1.0]), w, b, activation)[0][0]
    terms: list[list[_Term]] = [[] for _ in range(2 * n)]
    consts: list[_Const] = []

    for i in range(n):
        terms[i] = [_Term("basis", block=i), _Term("row", block=i, sign=-1.0, row=phi_lo)]
        if problem.terminal_mask[i]:
            terms[i].append(_Term("row", block=i, sign=-1.0, vec="ramp", row=phi_hi - phi_lo))
            consts.append(_constant(float(problem.x0[i]), float(problem.xf[i] - problem.x0[i])))
        else:
            consts.append(_constant(float(problem.x0[i])))

    # x_k(tf) is itself affine in beta: its row at x = +1 (no coupling, so only block k is filled)
    # is what the costate boundary is built from, and that is what makes
    # lambda(tf) = Qf (x(tf) - xf) hold identically for any beta.
    probe = _MixedExpression(w, b, activation, terms,
                             consts + [_constant(0.0) for _ in range(n)], 2 * n)
    a_hi, c_hi, _, _ = probe.assemble(np.array([1.0]))
    for i in range(n):
        j = n + i
        if problem.terminal_mask[i]:
            terms[j], offset = [_Term("basis", block=j)], 0.0
        else:
            terms[j] = [_Term("basis", block=j), _Term("row", block=j, sign=-1.0, row=phi_hi)]
            offset = 0.0
            for k in range(n):
                q = float(problem.Qf[i, k])
                if q == 0.0:
                    continue
                terms[j].append(_Term("row", block=k, sign=q, row=a_hi[k, 0, k * m:(k + 1) * m]))
                offset += q * (float(c_hi[0, k]) - float(problem.xf[k]))
        consts.append(_constant(offset))
    return _MixedExpression(w, b, activation, terms, consts, 2 * n)


# --------------------------------------------------------------------------------------
# closed-form references
# --------------------------------------------------------------------------------------
def riccati_solution(problem: LQProblem, t_query: Sequence[float] | None = None, *,
                     rtol: float = 1e-12, atol: float = 1e-12) -> dict:
    """The exact answer of the linear-quadratic problem, integrated from the terminal condition.

    Backward Riccati equation for the affine system ``xdot = A x + B u + c``::

        Pdot = -(A' P + P A - P B R^-1 B' P + Q),     P(tf) = Qf
        sdot = -(A - B R^-1 B' P)' s - P c,           s(tf) = -Qf xf
        qdot = 1/2 s' B R^-1 B' s - s' c,             q(tf) = 1/2 xf' Qf xf

    with ``V(t, x) = 1/2 x' P x + s' x + q`` and ``u = -R^-1 B' (P x + s)``.  One ODE integration:
    this is the closed-form statement of the problem, not a control solve, and it is the reference
    both networks are measured against.  It covers the terminal-*cost* formulation; a hard terminal
    constraint (``terminal_mask``) has :class:`MinEnergyTransfer` as its closed form instead.
    """
    n = problem.n
    p_terminal = np.concatenate([problem.Qf.ravel(), -problem.Qf @ problem.xf,
                                 [0.5 * float(problem.xf @ problem.Qf @ problem.xf)]])
    brb = problem.B @ problem.Rinv @ problem.B.T

    def rhs(_t: float, y: Array) -> Array:
        P = y[:n * n].reshape(n, n)
        s = y[n * n:n * n + n]
        Pdot = -(problem.A.T @ P + P @ problem.A - P @ brb @ P + problem.Q)
        sdot = -(problem.A - brb @ P).T @ s - P @ problem.c
        qdot = 0.5 * float(s @ brb @ s) - float(s @ problem.c)
        return np.concatenate([Pdot.ravel(), sdot, [qdot]])

    t0, tf = problem.t_span
    t_eval = np.array([t0, tf], float) if t_query is None else np.atleast_1d(
        np.asarray(t_query, float))
    sol = integrate.solve_ivp(rhs, (tf, t0), p_terminal, method="DOP853", rtol=rtol, atol=atol,
                              dense_output=True)
    if not sol.success:                                       # pragma: no cover - defensive
        raise RuntimeError(f"Riccati reference failed: {sol.message}")
    ys = sol.sol(t_eval)
    P = ys[:n * n].T.reshape(-1, n, n)
    s = ys[n * n:n * n + n].T
    q = ys[-1]
    x0 = problem.x0
    value = 0.5 * np.einsum("pi,pij,pj->p", np.broadcast_to(x0, (t_eval.size, n)), P,
                            np.broadcast_to(x0, (t_eval.size, n))) + s @ x0 + q
    gain = np.einsum("ij,pjk->pik", problem._BRt, P)           # R^-1 B' P(t)
    return {"t": t_eval, "P": P, "s": s, "q": q, "gain": gain, "value": value,
            "cost": float(value[0])}


def steady_state_lqr(problem: LQProblem) -> dict:
    """The infinite-horizon answer: the CARE solution, reusing the library's LQR helper.

    Valid for ``c = 0`` and no terminal constraint, where the value is ``1/2 x' P x`` and
    ``J* = 1/2 x0' P x0``; ``P`` and ``K = R^-1 B' P`` are the ones returned by
    :func:`pinneapple_systems.control.tuning.lqr`.
    """
    out = _care_lqr(problem.A, problem.B, problem.Q, problem.R)
    P, K = np.asarray(out["P"]), np.asarray(out["K"])
    return {"P": P, "gain": K, "poles": out["poles"],
            "cost": 0.5 * float(problem.x0 @ P @ problem.x0)}


@dataclass(frozen=True)
class MinEnergyTransfer:
    """Minimum ``1/2 int u^2 dt`` transfer of ``pdot = v, vdot = u - g`` between two endpoints.

    The Hamiltonian is affine in the costate, so the optimal acceleration is affine in time and the
    trajectory is the unique cubic through the endpoint conditions -- the closed-form answer
    :func:`pontryagin` has to reproduce::

        D  = pf - p0 - v0 T + 1/2 g T^2
        u0 = 6 D / T^2 - 2 (vf - v0 + g T) / T,   u1 = -6 D / T^2 + 4 (vf - v0 + g T) / T
        J  = T (u0^2 + u0 u1 + u1^2) / 6

    With ``vf=None`` the final velocity is free, ``lambda_v(T) = 0`` and so ``u1 = 0``: that is the
    energy-optimal intercept, whose control ``u0 (1 - t/T)`` gives ``J = u0^2 T / 6 = 6 D^2 / T^3``.
    """

    p0: float
    v0: float
    pf: float
    vf: float | None
    duration: float
    g: float
    u0: float
    u1: float
    cost: float
    fuel: float
    peak_thrust: float
    final_velocity: float

    def control(self, t: Array) -> Array:
        """The optimal acceleration at time ``t`` (scalar or array)."""
        t = np.asarray(t, float)
        return self.u0 + (self.u1 - self.u0) * t / self.duration

    def state(self, t: Array) -> tuple[Array, Array]:
        """Position and velocity at time ``t``."""
        t = np.asarray(t, float)
        slope = (self.u1 - self.u0) / self.duration
        v = self.v0 + (self.u0 - self.g) * t + 0.5 * slope * t ** 2
        p = self.p0 + self.v0 * t + 0.5 * (self.u0 - self.g) * t ** 2 + slope * t ** 3 / 6.0
        return p, v

    def summary(self) -> dict:
        """The numbers, as a dict (what the report and the tests quote)."""
        return {"cost": self.cost, "fuel": self.fuel, "peak_thrust": self.peak_thrust,
                "u0": self.u0, "u1": self.u1, "final_velocity": self.final_velocity}


def min_energy_transfer(p0: float, v0: float, pf: float, duration: float, *,
                        vf: float | None = None, g: float = 0.0) -> MinEnergyTransfer:
    """The closed-form minimum-energy transfer of :class:`MinEnergyTransfer`.

    ``vf=None`` is the free-final-velocity intercept; ``pf = vf = 0`` is a soft landing.
    """
    T = float(duration)
    if not T > 0.0:
        raise ValueError("duration must be positive")
    D = float(pf) - float(p0) - float(v0) * T + 0.5 * float(g) * T ** 2
    if vf is None:
        u0, u1 = 6.0 * D / T ** 2, 0.0
        final_v = float(v0) - float(g) * T + 0.5 * u0
    else:
        u0 = 6.0 * D / T ** 2 - 2.0 * (float(vf) - float(v0) + float(g) * T) / T
        u1 = -6.0 * D / T ** 2 + 4.0 * (float(vf) - float(v0) + float(g) * T) / T
        final_v = float(vf)
    cost = T * (u0 * u0 + u0 * u1 + u1 * u1) / 6.0
    # u is affine, so |u| integrates in closed form: it changes sign at most once.
    if (u0 >= 0.0) == (u1 >= 0.0):
        fuel = 0.5 * T * (abs(u0) + abs(u1))
    else:
        t_zero = T * abs(u0) / (abs(u0) + abs(u1))
        fuel = 0.5 * (abs(u0) * t_zero + abs(u1) * (T - t_zero))
    return MinEnergyTransfer(float(p0), float(v0), float(pf), vf, T, float(g), u0, u1, cost, fuel,
                             max(abs(u0), abs(u1)), final_v)


@dataclass(frozen=True)
class MinFuelLanding:
    """Bang-bang minimum-fuel landing in constant gravity (thrust up, ``0 <= a <= a_max``).

    Minimise ``int a dt`` for ``pdot = v, vdot = a - g`` from ``(h0, v0)`` to rest at ``h = 0`` with
    free final time.  The switching function ``sigma = 1 + lambda_v`` is linear in time
    (``lambda_p`` is constant), so the extremal has at most one switch and the structure is *coast,
    then full thrust*; the two touchdown conditions then give the burn time in closed form::

        c = a_max - g,   tau = sqrt((v0^2 + 2 g h0) / (a_max c))
        t_coast = (v0 + c tau) / g,   T = t_coast + tau,   fuel = a_max tau

    Valid for ``a_max > g`` and ``t_coast >= 0``; a steeper entry needs thrust from the first
    instant and is rejected rather than silently returned.
    """

    h0: float
    v0: float
    g: float
    a_max: float
    burn_time: float
    coast_time: float
    flight_time: float
    fuel: float

    def control(self, t: Array) -> Array:
        """The thrust acceleration at ``t``: zero while coasting, ``a_max`` while burning."""
        return np.where(np.asarray(t, float) < self.coast_time, 0.0, self.a_max)

    def state(self, t: Array) -> tuple[Array, Array]:
        """Altitude and velocity along the two arcs."""
        t = np.atleast_1d(np.asarray(t, float))
        c = self.a_max - self.g
        v_s = self.v0 - self.g * self.coast_time
        h_s = self.h0 + self.v0 * self.coast_time - 0.5 * self.g * self.coast_time ** 2
        late = t > self.coast_time
        s = np.where(late, t - self.coast_time, 0.0)
        h = np.where(late, h_s + v_s * s + 0.5 * c * s ** 2,
                     self.h0 + self.v0 * t - 0.5 * self.g * t ** 2)
        v = np.where(late, v_s + c * s, self.v0 - self.g * t)
        return h, v

    def summary(self) -> dict:
        """The numbers, as a dict."""
        return {"fuel": self.fuel, "burn_time": self.burn_time, "coast_time": self.coast_time,
                "flight_time": self.flight_time, "mean_thrust": self.fuel / self.flight_time}


def min_fuel_landing(h0: float, v0: float, g: float, a_max: float) -> MinFuelLanding:
    """The closed-form bang-bang landing of :class:`MinFuelLanding`."""
    if not a_max > g:
        raise ValueError("a_max must exceed g for a landing to be possible")
    if h0 <= 0.0:
        raise ValueError("h0 must be positive")
    c = a_max - g
    tau = float(np.sqrt((v0 * v0 + 2.0 * g * h0) / (a_max * c)))
    coast = (v0 + c * tau) / g
    if coast < 0.0:
        raise ValueError("entry too fast for a coast-then-burn extremal (coast time < 0): thrust "
                         "is needed from the first instant")
    return MinFuelLanding(float(h0), float(v0), float(g), float(a_max), tau, coast, coast + tau,
                          a_max * tau)


# --------------------------------------------------------------------------------------
# PoNN: the PMP two-point problem with X-TFC
# --------------------------------------------------------------------------------------
@dataclass
class PoNN:
    """The solved Pontryagin two-point boundary value problem.

    ``state``, ``costate`` and ``control`` are on the grid ``t``; ``cost`` is the cost functional
    evaluated on that grid plus the terminal cost; ``residual_rms`` is the RMS of the collocation
    residual (state and costate equations, in physical units) and ``boundary_error`` the largest
    violation of the boundary conditions, which the constrained expression makes zero by
    construction.  ``gain`` is the least-squares matrix with ``lambda(t) = gain x(t)``: for a linear
    problem the costate is linear in the state, so this is the feedback the open-loop solution
    corresponds to, and it is what the LQR test compares with Riccati's ``K``.
    """

    problem: LQProblem
    t: Array
    state: Array
    costate: Array
    control: Array
    cost: float
    gain: Array
    t_colloc: Array
    residual_rms: float
    residual_history: list[float] = field(default_factory=list)
    n_iter: int = 1
    n_basis: int = 0
    boundary_error: float = 0.0

    def state_at(self, t_query: Sequence[float]) -> Array:
        """State at arbitrary times (the same constrained expression, re-evaluated)."""
        return self._evaluate(t_query)[0]

    def costate_at(self, t_query: Sequence[float]) -> Array:
        """Costate at arbitrary times."""
        return self._evaluate(t_query)[1]

    def control_at(self, t_query: Sequence[float]) -> Array:
        """Open-loop control at arbitrary times."""
        return self.problem.optimal_control(self._evaluate(t_query)[1])

    def _evaluate(self, t_query: Sequence[float]) -> tuple[Array, Array]:
        t0, tf = self.problem.t_span
        tq = np.atleast_1d(np.asarray(t_query, float))
        x = -1.0 + 2.0 * (tq - t0) / (tf - t0)
        y, _ = self.expression.evaluate(x, self.beta)
        return y[:, :self.problem.n], y[:, self.problem.n:]

    def summary(self) -> dict:
        """The scalars the report quotes."""
        return {"cost": self.cost, "residual_rms": self.residual_rms,
                "boundary_error": self.boundary_error, "n_iter": self.n_iter,
                "n_features": int(self.beta.size)}


_PONTRYAGIN_INTERNALS = ("expression", "beta")


def pontryagin(problem: LQProblem, *, n_basis: int = 24, n_collocation: int = 120,
               activation: str = "tanh", seed: int = 0, linear: bool = True, max_iter: int = 40,
               tol: float = 1e-11, t_eval: Array | None = None,
               w_range: tuple[float, float] = (-5.0, 5.0),
               b_range: tuple[float, float] = (-5.0, 5.0)) -> PoNN:
    """Solve the Pontryagin two-point problem with X-TFC -- no gradient descent, no training loop.

    Builds the constrained expression of :func:`_bvp_expression`, collocates the state and the
    costate equations on Chebyshev-Gauss-Lobatto points (the nodes of the IVP solver) and solves the
    resulting system for the linear coefficients.  ``linear=True`` (right for every problem of this
    module) differences the collocation residual, which is *exact* for an affine residual, and
    returns the X-TFC minimiser from one least-squares solve; ``linear=False`` re-linearises at the
    new iterate (Gauss-Newton) up to ``max_iter``, which is what a nonlinear right-hand side needs.
    """
    n = problem.n
    t0, tf = problem.t_span
    rng = np.random.default_rng(seed)
    w = rng.uniform(w_range[0], w_range[1], n_basis)
    b = rng.uniform(b_range[0], b_range[1], n_basis)
    expr = _bvp_expression(problem, w, b, activation)

    x_colloc = chebyshev_gauss_lobatto(n_collocation)
    t_colloc = t0 + 0.5 * (x_colloc + 1.0) * (tf - t0)
    scale = 2.0 / (tf - t0)
    n_params = int(n_basis) * 2 * n

    def residual(beta: Array) -> Array:
        y, dy = expr.evaluate(x_colloc, beta)
        xs, lam = y[:, :n], y[:, n:]
        out = np.empty_like(y)
        out[:, :n] = scale * dy[:, :n] - problem.state_rate(xs, problem.optimal_control(lam))
        out[:, n:] = scale * dy[:, n:] - problem.costate_rate(xs, lam)
        return out.ravel()

    beta = np.zeros(n_params)
    history: list[float] = []
    current = residual(beta)
    rms = float(np.sqrt(np.mean(current ** 2)))
    history.append(rms)
    n_iter = 0
    for n_iter in range(1, (1 if linear else max_iter) + 1):
        if rms < tol:
            break
        jac = np.empty((current.size, n_params))
        probe = np.zeros(n_params)
        for k in range(n_params):
            probe[k] = 1.0
            jac[:, k] = residual(probe) - current
            probe[k] = 0.0
        step, *_ = np.linalg.lstsq(jac, -current, rcond=None)
        candidate = beta + step
        updated = residual(candidate)
        new_rms = float(np.sqrt(np.mean(updated ** 2)))
        if new_rms > rms and n_iter > 1:
            break                                          # divergence guard of the IVP module
        beta, current, rms = candidate, updated, new_rms
        history.append(rms)                                # residual *after* this solve
        if linear:
            break

    t_q = np.linspace(t0, tf, 201) if t_eval is None else np.atleast_1d(np.asarray(t_eval, float))
    x_q = -1.0 + 2.0 * (t_q - t0) / (tf - t0)
    y, _ = expr.evaluate(x_q, beta)
    xs, lam = y[:, :n], y[:, n:]
    u = problem.optimal_control(lam)
    cost = float(integrate.trapezoid(problem.running_cost(xs, u), t_q)
                 + problem.terminal_cost(xs[-1:])[0])
    result = PoNN(problem=problem, t=t_q, state=xs, costate=lam, control=u, cost=cost,
                  gain=_fit_gain(xs, lam), t_colloc=t_colloc,
                  residual_rms=history[-1] if history else 0.0, residual_history=history,
                  n_iter=n_iter, n_basis=int(n_basis),
                  boundary_error=_boundary_error(problem, xs[0], xs[-1], lam[-1]))
    result.expression, result.beta = expr, beta
    return result


def _fit_gain(state: Array, costate: Array) -> Array:
    """Least-squares ``Lambda`` with ``lambda(t) = Lambda x(t)`` (the feedback of a linear solution)."""
    norm = np.linalg.norm(state, axis=1)
    keep = norm > 1e-10 * max(1.0, float(np.max(norm)))
    if keep.sum() < state.shape[1]:
        keep = np.ones(state.shape[0], bool)
    matrix, *_ = np.linalg.lstsq(state[keep], costate[keep], rcond=None)
    return matrix.T


def _boundary_error(problem: LQProblem, x_first: Array, x_last: Array, lam_last: Array) -> float:
    """Largest violation of the boundary conditions of the two-point problem.

    The transversality is checked only on the components that are *not* pinned: where the state is
    pinned at ``tf`` the costate endpoint is free (it is the multiplier of that constraint), so
    there is nothing there to violate.
    """
    parts = [float(np.max(np.abs(x_first - problem.x0)))]
    if problem.constrained:
        parts.append(float(np.max(np.abs((x_last - problem.xf)[problem.terminal_mask]))))
    free = ~problem.terminal_mask
    if np.any(free):
        target = problem.Qf @ (x_last - problem.xf)
        parts.append(float(np.max(np.abs(lam_last - target)[free])))
    return max(parts)


# --------------------------------------------------------------------------------------
# BeNN: the HJB residual on a value network with a TFC terminal condition
# --------------------------------------------------------------------------------------
def _monomials(n: int, degree: int) -> list[tuple[int, ...]]:
    """All multi-indices of total degree ``1..degree`` in ``n`` variables."""
    out: list[tuple[int, ...]] = []

    def walk(prefix: list[int], remaining: int) -> None:
        if len(prefix) == n:
            if sum(prefix) > 0:
                out.append(tuple(prefix))
            return
        for power in range(remaining + 1):
            walk(prefix + [power], remaining - power)

    walk([], int(degree))
    return out


class _ValueFeatures:
    """Features of the value network: ELM ``tanh`` units plus polynomial enrichment.

    The ELM part is ``act(xi . W_p + wt_p tau + bias_p)`` with fixed random draws, as in X-TFC.  The
    enrichment is ``s^k prod_i xi_i^e_i`` with ``xi = (x - center) / scale``, ``tau`` the horizon
    mapped to ``[-1, 1]`` and ``s = (t - tf) / (tf - t0)`` the time-to-go as a fraction of the
    horizon.  The ``s`` factor is not decoration: a purely time-independent feature cancels in the
    terminal-condition projection, and a finite-horizon Riccati value is smooth in the time to go,
    so a few powers of it reproduce the time dependence the ELM part would otherwise have to carry
    alone.  ``s`` is normalised by the horizon for conditioning -- raw ``(t - tf)^k`` spans seven
    decades over a 10 s problem, which costs the Gauss-Newton most of its digits.
    """

    def __init__(self, n: int, n_features: int, poly_degree: int, poly_time_degree: int,
                 x_center: Array, x_scale: Array, t_span: tuple[float, float],
                 activation: str, seed: int, w_range: tuple[float, float],
                 b_range: tuple[float, float]) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation={activation!r} not in {list(ACTIVATIONS)}")
        rng = np.random.default_rng(seed)
        self.n, self.activation, self.t_span = n, activation, t_span
        self.horizon = float(t_span[1] - t_span[0])
        self.W = rng.uniform(w_range[0], w_range[1], (n_features, n))
        self.bias = rng.uniform(b_range[0], b_range[1], n_features)
        self.wt = rng.uniform(w_range[0], w_range[1], n_features)
        self.x_center, self.x_scale = np.asarray(x_center, float), np.asarray(x_scale, float)
        powers = _monomials(n, poly_degree)
        times = list(range(1, poly_time_degree + 1))
        self.poly = [(k, e) for k in times for e in powers]
        self.n_features = int(n_features)
        self.size = self.n_features + len(self.poly)

    def evaluate(self, t: Array, x: Array) -> tuple[Array, Array, Array]:
        """``(phi, dphi/dx, dphi/dt)`` at the paired points ``t`` (N,) and ``x`` (N, n)."""
        act, dact = ACTIVATIONS[self.activation]
        x = np.asarray(x, float)
        n_pts = x.shape[0]
        xi = (x - self.x_center) / self.x_scale
        t = np.asarray(t, float)
        z = xi @ self.W.T + self.bias
        t0, tf = self.t_span
        tau = -1.0 + 2.0 * (t - t0) / (tf - t0)
        z = z + tau[:, None] * self.wt[None, :]
        dtau = 2.0 / (tf - t0)
        phi = np.zeros((n_pts, self.size))
        dphi_dx = np.zeros((n_pts, self.size, self.n))
        dphi_dt = np.zeros((n_pts, self.size))
        phi[:, :self.n_features] = act(z)
        slope = dact(z)
        for k in range(self.n):
            dphi_dx[:, :self.n_features, k] = slope * self.W[None, :, k] / self.x_scale[k]
        dphi_dt[:, :self.n_features] = slope * self.wt[None, :] * dtau
        for j, (power, exponent) in enumerate(self.poly):
            col = self.n_features + j
            mono = np.ones(n_pts)
            for k in range(self.n):
                mono = mono * xi[:, k] ** exponent[k]
            to_go = (t - self.t_span[1]) / self.horizon
            time_factor = to_go ** power
            dtime = power * to_go ** (power - 1) / self.horizon if power > 0 else np.zeros(n_pts)
            phi[:, col] = time_factor * mono
            dphi_dt[:, col] = dtime * mono
            for k in range(self.n):
                if exponent[k] == 0:
                    continue
                term = exponent[k] * xi[:, k] ** (exponent[k] - 1)
                for i in range(self.n):
                    if i != k:
                        term = term * xi[:, i] ** exponent[i]
                dphi_dx[:, col, k] = time_factor * term / self.x_scale[k]
        return phi, dphi_dx, dphi_dt


@dataclass
class BeNN:
    """The solved Hamilton-Jacobi-Bellman problem: a value network with a closed-loop policy.

    ``value``, ``grad_value``, ``policy`` and ``hjb_residual`` take batches (``t`` with N entries,
    ``x`` with shape (N, n)).  ``scale`` is the mean absolute Hamiltonian term over the training box
    and turns the residual dimensionless (``hjb_residual(..., relative=True)``), which is what a
    trust threshold is set on.
    """

    problem: LQProblem
    features: _ValueFeatures
    beta: Array
    scale: float = 1.0
    box: Array | None = None
    residual_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    t_reference: float = 0.0

    # -- the constrained expression -------------------------------------------------
    def _psi(self, t: Array, x: Array) -> tuple[Array, Array, Array]:
        """The TFC-projected features and their derivatives.

        Finite horizon: the reference used for the projection is ``(tf, x)``, the same state at the
        terminal time, so ``psi(t, x) = phi(t, x) - phi(tf, x)`` vanishes at ``tf`` and its spatial
        derivative is ``dphi/dx(t, x) - dphi/dx(tf, x)`` -- the reference *moves with x*, so it does
        contribute to the gradient.
        """
        phi, dphi_dx, dphi_dt = self.features.evaluate(t, x)
        t_ref = np.full(x.shape[0], self.features.t_span[1])
        phi_ref, dphi_ref_dx, _ = self.features.evaluate(t_ref, x)
        return phi - phi_ref, dphi_dx - dphi_ref_dx, dphi_dt

    def value(self, t: Array, x: Array) -> Array:
        """``V(t, x)``, which equals the terminal cost exactly at ``t = tf``."""
        t, x = _as_rows(t, x)
        psi, _, _ = self._psi(t, x)
        return self._terminal_value(x) + psi @ self.beta

    def grad_value(self, t: Array, x: Array) -> Array:
        """``grad_x V(t, x)``, shape (N, n)."""
        t, x = _as_rows(t, x)
        _, dpsi_dx, _ = self._psi(t, x)
        return self._terminal_grad(x) + np.einsum("pcj,c->pj", dpsi_dx, self.beta)

    def dvalue_dt(self, t: Array, x: Array) -> Array:
        """``dV/dt`` along the horizon."""
        t, x = _as_rows(t, x)
        _, _, dpsi_dt = self._psi(t, x)
        return dpsi_dt @ self.beta

    def policy(self, t: Array, x: Array) -> Array:
        """The closed-loop policy ``u = -R^-1 B' grad V``, shape (N, m)."""
        return self.problem.optimal_control(self.grad_value(t, x))

    def hjb_residual(self, t: Array, x: Array, *, relative: bool = False) -> Array:
        """The HJB residual along ``(t, x)``, in cost-rate units (or divided by ``scale``)::

            dV/dt + grad V . (A x + c) - 1/2 grad V' B R^-1 B' grad V + 1/2 x' Q x

        Zero for the exact value function.  Evaluated along the trajectory the policy is actually
        flying, it is the cheapest statement about whether that policy is still inside the region
        it was fitted on.
        """
        t, x = _as_rows(t, x)
        grad = self.grad_value(t, x)
        drift = x @ self.problem.A.T + self.problem.c
        gb = grad @ self.problem.B
        residual = (self.dvalue_dt(t, x) + np.einsum("pj,pj->p", grad, drift)
                    - 0.5 * np.einsum("pm,pm->p", gb @ self.problem.Rinv, gb)
                    + 0.5 * np.einsum("pi,ij,pj->p", x, self.problem.Q, x))
        return residual / self.scale if relative else residual

    def hjb_jacobian(self, t: Array, x: Array) -> Array:
        """``d hjb_residual / d beta``, shape (N, n_features), in closed form.

        The residual is affine in ``beta`` except for the ``-1/2 grad' B R^-1 B' grad`` term, which
        is quadratic, so the Gauss-Newton Jacobian is analytic: with ``g = grad V`` and
        ``q = g B R^-1``,

            d r / d beta_c = dpsi/dt_c + (dpsi/dx_c . drift) - q . (dpsi/dx_c B).

        Supplying it is what makes the fit cheap: a finite-difference Jacobian costs one residual
        evaluation per feature per iteration (ordered 20x more), and this grid runs to hundreds of
        points.
        """
        t, x = _as_rows(t, x)
        _, dpsi_dx, dpsi_dt = self._psi(t, x)
        grad = self._terminal_grad(x) + np.einsum("pcj,c->pj", dpsi_dx, self.beta)
        drift = x @ self.problem.A.T + self.problem.c
        q = (grad @ self.problem.B) @ self.problem.Rinv
        dgb = np.einsum("pcj,jm->pcm", dpsi_dx, self.problem.B)
        return (dpsi_dt + np.einsum("pcj,pj->pc", dpsi_dx, drift)
                - np.einsum("pm,pcm->pc", q, dgb))

    # -- terminal condition ---------------------------------------------------------
    def _terminal_value(self, x: Array) -> Array:
        return self.problem.terminal_cost(x)

    def _terminal_grad(self, x: Array) -> Array:
        return (x - self.problem.xf) @ self.problem.Qf.T

    # -- what the reports and the tests quote ---------------------------------------
    def cost_to_go(self, x0: Array | None = None, t: float | None = None) -> float:
        """``V(t0, x0)`` from the network."""
        x0 = self.problem.x0 if x0 is None else np.atleast_1d(np.asarray(x0, float))
        t0 = self.problem.t_span[0] if t is None else float(t)
        return float(self.value(np.array([t0]), x0[None, :])[0])

    def gain(self, x0: Array | None = None, t: float | None = None) -> Array:
        """The Jacobian of the policy at ``x0``, negated: the ``K`` of ``u = -K x`` for small ``x``.

        The policy is exactly linear in the state only when the value is exactly quadratic, so this
        is the local gain, which is what the LQR comparison uses.
        """
        x0 = (self.problem.x0 if x0 is None
              else np.atleast_1d(np.asarray(x0, float)))
        t0 = self.problem.t_span[0] if t is None else float(t)
        eps = 1e-6 * np.maximum(1.0, np.abs(x0))
        columns = []
        for i in range(self.problem.n):
            probe = np.repeat(x0[None, :], 2, axis=0)
            probe[0, i] += eps[i]
            probe[1, i] -= eps[i]
            values = self.policy(np.full(2, t0), probe)
            columns.append((values[0] - values[1]) / (2.0 * eps[i]))
        return -np.asarray(columns).T

    def summary(self) -> dict:
        """The scalars the reports quote."""
        return {"cost_to_go": self.cost_to_go(), "scale": self.scale,
                "residual_rms": self.residual_history[-1] if self.residual_history else 0.0,
                "n_features": self.features.size, "n_iter": self.n_iter}


def _as_rows(t: Array, x: Array) -> tuple[Array, Array]:
    """Broadcast ``(t, x)`` to paired batches."""
    x = np.atleast_2d(np.asarray(x, float))
    t = np.atleast_1d(np.asarray(t, float))
    if t.size == 1 and x.shape[0] > 1:
        t = np.full(x.shape[0], float(t[0]))
    if x.shape[0] == 1 and t.size > 1:
        x = np.broadcast_to(x, (t.size, x.shape[1]))
    return t, x


def bellman(problem: LQProblem, *, box: Array | None = None, weighted: bool = False,
            n_features: int = 24, poly_degree: int = 2, poly_time_degree: int = 6, n_time: int = 22,
            n_space: int = 11, max_points: int = 400, activation: str = "tanh", seed: int = 0,
            w_range: tuple[float, float] = (-4.0, 4.0),
            b_range: tuple[float, float] = (-4.0, 4.0), max_nfev: int = 400) -> BeNN:
    """Fit a value network to the Hamilton-Jacobi-Bellman equation of ``problem``.

    ``V(t, x) = terminal cost(x) + (phi(t, x) - phi(tf, x)) . beta``: the terminal condition holds
    for any ``beta`` because the basis is projected onto functions vanishing at ``tf`` (TFC).  The
    residual is collocated on a Chebyshev grid in time and in each state of ``box``, and the
    coefficients are solved by Gauss-Newton (:func:`scipy.optimize.least_squares`) with the analytic
    Jacobian of :meth:`BeNN.hjb_jacobian`; no network is trained, the features are fixed as in
    X-TFC.

    ``box`` is the region the value is fitted on, and therefore the region the policy is valid on;
    it defaults to ``x0 +/- 3 max(|x0|, 1)``.  The residual of :meth:`BeNN.hjb_residual` is
    meaningful precisely as a statement about leaving it.

    ``weighted`` divides each collocation residual by the magnitude of the Hamiltonian terms at that
    point before the least-squares solve, which is the difference between a fit that works and one
    that does not when the value spans decades across ``box`` (a landing corridor).  It is off by
    default because on a well-scaled box it costs the regulator accuracy: measured on the double
    integrator over 10 s, the cost to go came out 6.9 % above the Riccati value unweighted and 27.6 %
    below it weighted.  :func:`landing_demo` turns it on for that reason.

    Only the *finite* horizon is offered, and the reason is measured, not stylistic: the stationary
    problem's HJB residual is solved exactly by every symmetric root of the algebraic Riccati
    equation, the stabilising one and the anti-stabilising one alike, so a residual-only fit has no
    way to tell the value function from minus it.  Fitted on the double integrator the stationary
    residual reaches machine zero on the anti-stabilising root (gain ``[-0.62, +0.62]`` against the
    Riccati ``[1.00, 1.73]``, cost to go ``-0.69`` against ``+0.87``).  A terminal condition removes
    the ambiguity -- the value at ``tf`` is fixed, so the finite-horizon fit is unique -- and that is
    the only form this module claims.  The infinite-horizon answer keeps its closed form in
    :func:`steady_state_lqr`.
    """
    n = problem.n
    if box is None:
        half = np.maximum(3.0 * np.abs(problem.x0), 1.0)
        box = np.column_stack([-half, half])
    box = np.asarray(box, float).reshape(n, 2)
    center = 0.5 * (box[:, 0] + box[:, 1])
    half_width = np.maximum(0.5 * (box[:, 1] - box[:, 0]), 1e-9)
    features = _ValueFeatures(n, n_features, poly_degree, poly_time_degree, center, half_width,
                              problem.t_span, activation, seed, w_range, b_range)

    # collocation: Chebyshev in time, tensor Chebyshev in the state box, thinned if too large
    t_nodes = chebyshev_gauss_lobatto(n_time, *problem.t_span)
    axes = [chebyshev_gauss_lobatto(n_space, lo, hi) for lo, hi in box]
    grid = np.stack([axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=1)
    if grid.shape[0] > max_points:
        keep = np.random.default_rng(seed + 1).choice(grid.shape[0], max_points, replace=False)
        grid = grid[np.sort(keep)]
    t_grid = np.repeat(t_nodes, grid.shape[0])
    x_grid = np.tile(grid, (t_nodes.size, 1))

    net = BeNN(problem=problem, features=features, beta=np.zeros(features.size), box=box,
               t_reference=problem.t_span[1])

    def hamiltonian_scale(beta: Array) -> Array:
        """Per-point magnitude of the Hamiltonian terms, evaluated at ``beta``.

        The HJB residual is not of one size across ``box``: on a landing corridor the terminal cost
        puts six or seven decades between the top of the box and the origin, and an unweighted
        least-squares fit spends every degree of freedom on the corner.  Dividing the residual of
        each collocation point by the size of the terms that live there is the per-point version of
        the normalisation the trust threshold is set on, applied inside the fit.
        """
        net.beta = beta
        grad = net.grad_value(t_grid, x_grid)
        drift = x_grid @ problem.A.T + problem.c
        gb = grad @ problem.B
        return (np.abs(np.einsum("pj,pj->p", grad, drift))
                + np.abs(0.5 * np.einsum("pm,pm->p", gb @ problem.Rinv, gb))
                + np.abs(0.5 * np.einsum("pi,ij,pj->p", x_grid, problem.Q, x_grid)))

    weight = hamiltonian_scale(np.zeros(features.size)) if weighted else np.ones(t_grid.size)
    if weighted:
        weight = np.maximum(weight, 1e-9 * float(np.max(weight))) if np.any(weight) else weight

    def residual(beta: Array) -> Array:
        net.beta = beta
        return net.hjb_residual(t_grid, x_grid) / weight

    def jacobian(beta: Array) -> Array:
        net.beta = beta
        return net.hjb_jacobian(t_grid, x_grid) / weight[:, None]

    solution = optimize.least_squares(residual, np.zeros(features.size), jac=jacobian, method="lm",
                                      xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=max_nfev)
    net.beta = solution.x
    net.n_iter = int(solution.nfev)
    raw = net.hjb_residual(t_grid, x_grid)
    net.residual_history = [float(np.sqrt(np.mean(solution.fun ** 2))),
                            float(np.sqrt(np.mean(raw ** 2)))]
    # the dimensionless scale: mean magnitude of the Hamiltonian terms over the training box
    net.scale = float(np.mean(hamiltonian_scale(solution.x))) + 1e-12
    return net


# --------------------------------------------------------------------------------------
# an independent numerical reference: direct transcription
# --------------------------------------------------------------------------------------
def direct_transcription(problem: LQProblem, *, n_knots: int = 24, sub_steps: int = 4,
                         max_iter: int = 300, smooth_abs: float = 0.0) -> dict:
    """Direct transcription with finite differences: an independent check of the closed forms.

    The control is piecewise linear on ``n_knots`` nodes of ``[t0, tf]``, the state is marched with
    RK4 (``sub_steps`` per interval) and the cost with the trapezoid rule; the states named by
    ``terminal_mask`` are enforced as SLSQP equality constraints.  Nothing here is shared with the
    constrained expressions or the Riccati integration, so agreeing with it is evidence about both.

    CasADi/IPOPT is the usual tool for this and is not a dependency of this library, so this is a
    finite-difference transcription of the same problem -- a quadratic program once discretised,
    which SLSQP solves to its tolerance.  ``smooth_abs > 0`` replaces the control cost by
    ``sqrt(u^2 + smooth_abs^2)``, which is how the min-fuel landing is transcribed (the exact
    problem is nondifferentiable at ``u = 0``).
    """
    n, m = problem.n, problem.m
    t = np.linspace(problem.t_span[0], problem.t_span[1], n_knots)
    h = (t[1] - t[0]) / sub_steps

    def march(z: Array) -> Array:
        u_nodes = np.asarray(z, float).reshape(n_knots, m)
        states = np.empty((n_knots, n))
        states[0] = problem.x0
        for i in range(n_knots - 1):
            x = states[i].copy()
            u_a, u_b = u_nodes[i], u_nodes[i + 1]
            for s in range(sub_steps):
                u_mid = u_a + (u_b - u_a) * ((s + 0.5) / sub_steps)
                k1 = problem.A @ x + problem.B @ u_mid + problem.c
                k2 = problem.A @ (x + 0.5 * h * k1) + problem.B @ u_mid + problem.c
                k3 = problem.A @ (x + 0.5 * h * k2) + problem.B @ u_mid + problem.c
                k4 = problem.A @ (x + h * k3) + problem.B @ u_mid + problem.c
                x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states[i + 1] = x
        return states

    def rate(states: Array, z: Array) -> Array:
        u_nodes = np.asarray(z, float).reshape(n_knots, m)
        value = (0.5 * np.einsum("pi,ij,pj->p", states, problem.Q, states)
                 + 0.5 * np.einsum("pi,ij,pj->p", u_nodes, problem.R, u_nodes))
        if smooth_abs > 0.0:
            value = value + np.sum(np.sqrt(u_nodes ** 2 + smooth_abs ** 2), axis=1)
        return value

    def cost(z: Array) -> float:
        states = march(z)
        return float(integrate.trapezoid(rate(states, z), t)
                     + problem.terminal_cost(states[-1:])[0])

    constraints = []
    if problem.constrained:
        index = np.flatnonzero(problem.terminal_mask)
        constraints.append({"type": "eq",
                            "fun": lambda z, index=index: march(z)[-1, index] - problem.xf[index]})
    solution = optimize.minimize(cost, np.zeros(n_knots * m), method="SLSQP",
                                 constraints=constraints,
                                 options={"maxiter": max_iter, "ftol": 1e-14})
    return {"cost": float(solution.fun), "control": np.asarray(solution.x).reshape(n_knots, m),
            "t": t, "state": march(solution.x), "success": bool(solution.success),
            "message": str(solution.message), "n_knots": int(n_knots)}


# --------------------------------------------------------------------------------------
# trust: the residual along a trajectory, and the MPC hand-over
# --------------------------------------------------------------------------------------
def residual_report(net: BeNN, trajectory: dict, *, threshold: float | None = None) -> dict:
    """HJB residual of ``net`` along a flown trajectory, as a table plus a verdict.

    ``trajectory`` holds ``t`` (N,) and ``state`` (N, n).  Returns the absolute and relative RMS and
    maximum residual, the first time the relative residual crossed ``threshold`` (``None`` when it
    did not) and ``within``, which is ``True`` when there was no crossing.
    """
    t = np.asarray(trajectory["t"], float)
    x = np.atleast_2d(np.asarray(trajectory["state"], float))
    raw = net.hjb_residual(t, x)
    rel = np.abs(net.hjb_residual(t, x, relative=True))
    crossed = None
    if threshold is not None:
        over = np.flatnonzero(rel > threshold)
        if over.size:
            crossed = float(t[over[0]])
    return {"rms": float(np.sqrt(np.mean(raw ** 2))), "max": float(np.max(np.abs(raw))),
            "relative_rms": float(np.sqrt(np.mean(rel ** 2))), "relative_max": float(np.max(rel)),
            "threshold": threshold, "crossed_at": crossed, "within": crossed is None}


def mpc_control(problem: LQProblem, t_now: float, x_now: Sequence[float], *, n_basis: int = 18,
                n_collocation: int = 80, seed: int = 0) -> dict:
    """One shrinking-horizon MPC step: re-solve the two-point problem from the current state.

    The terminal time is fixed by the mission and the horizon shrinks as time advances, which is the
    receding-horizon form for a fixed-final-time problem; the control applied is the first one of the
    re-solved Pontryagin solution.  For the linear-quadratic problems of this module this is the
    exact optimal feedback, which is what makes it a defensible backup for a learned policy that has
    left its region.
    """
    sub = problem.with_state(x_now, (float(t_now), problem.t_span[1]))
    solution = pontryagin(sub, n_basis=n_basis, n_collocation=n_collocation, seed=seed)
    return {"control": solution.control[0], "cost_to_go": solution.cost,
            "residual_rms": solution.residual_rms, "solution": solution}


def _closed_loop_rhs(problem: LQProblem, control_of: Callable[[Array, Array], Array],
                     free_control: Array) -> Callable[[Array, Array], Array]:
    """The right-hand side of ``xdot = A x + B u(t, x) + c`` for the X-TFC IVP solver."""

    def rhs(t: Array, x: Array) -> Array:
        u = control_of(np.asarray(t, float), np.atleast_2d(x))
        return np.atleast_2d(x) @ problem.A.T + u @ problem.B.T + problem.c

    return rhs


def _fly(net: BeNN, x0: Sequence[float], t0: float, tf: float, dt: float,
         steps: int = 12) -> dict:
    """Fly the value network's policy with one X-TFC IVP solve per step.

    The closed loop is integrated with
    :func:`pinneapple_simulation.numerical_solvers.xtfc_ivp.solve_xtfc_ode`, so the flown trajectory
    is the one the *policy* produces and not the one the value network was fitted on.
    """
    problem = net.problem
    rhs = _closed_loop_rhs(problem, net.policy, np.zeros(problem.m))
    times, states, controls = [float(t0)], [np.asarray(x0, float)], []
    t = float(t0)
    while t < tf - 1e-12:
        step = min(dt, tf - t)
        controls.append(net.policy(np.array([t]), states[-1][None, :])[0])
        out = solve_xtfc_ode(rhs, states[-1], (t, t + step), n_basis=steps, n_collocation=steps,
                             max_iter=20, tol=1e-12, seed=1)
        states.append(np.asarray(out["predict"]([t + step]), float)[0])
        times.append(t + step)
        t += step
    return {"t": np.array(times), "state": np.array(states),
            "control": np.array(controls) if controls else np.zeros((0, problem.m))}


def _fly_with_handover(net: BeNN, x0: Sequence[float], t0: float, tf: float, dt: float,
                       threshold: float, gust: tuple[float, Sequence[float]] | None = None,
                       steps: int = 12, seed: int = 0) -> dict:
    """Fly the learned policy and hand over to :func:`mpc_control` when the HJB residual crosses.

    ``gust=(time, jump)`` applies a state jump at ``time`` (a disturbance), reported back as
    ``disturbed_at``.  The moment the relative residual at the state the vehicle is actually in
    exceeds ``threshold``, the controller switches for the rest of the flight.
    """
    problem = net.problem
    times, states, controls, residuals = [float(t0)], [np.asarray(x0, float)], [], []
    handed_over_at: float | None = None
    disturbed_at: float | None = None
    t = float(t0)
    while t < tf - 1e-12:
        x_now = np.array(states[-1], float)
        if gust is not None and disturbed_at is None and t >= gust[0] - 1e-12:
            jump = np.atleast_1d(np.asarray(gust[1], float))
            x_now = x_now.copy()
            x_now[:jump.size] += jump
            states[-1] = x_now
            disturbed_at = t
        relative = float(abs(net.hjb_residual(np.array([t]), x_now[None, :], relative=True)[0]))
        residuals.append(relative)
        if handed_over_at is None and relative > threshold:
            handed_over_at = t
        step = min(dt, tf - t)
        if handed_over_at is None:
            applied = net.policy(np.array([t]), x_now[None, :])[0]
            rhs = _closed_loop_rhs(problem, net.policy, np.zeros(problem.m))
        else:
            applied = np.atleast_1d(mpc_control(problem, t, x_now, seed=seed)["control"]).astype(float)
            rhs = _closed_loop_rhs(problem, lambda _t, _x, u=applied: u[None, :],
                                   np.zeros(problem.m))
        controls.append(np.atleast_1d(applied).astype(float))
        out = solve_xtfc_ode(rhs, x_now, (t, t + step), n_basis=steps, n_collocation=steps,
                             max_iter=20, tol=1e-12, seed=1)
        states.append(np.asarray(out["predict"]([t + step]), float)[0])
        times.append(t + step)
        t += step
    return {"t": np.array(times), "state": np.array(states),
            "control": np.array(controls) if controls else np.zeros((0, problem.m)),
            "residuals": np.array(residuals), "handed_over_at": handed_over_at,
            "disturbed_at": disturbed_at}


# --------------------------------------------------------------------------------------
# the demo
# --------------------------------------------------------------------------------------
MOON_G = 1.62


def landing_demo(*, h0: float = 120.0, v0: float = -15.0, g: float = MOON_G,
                 a_max_factor: float = 4.0, corridor_altitude: float = 1.1,
                 corridor_velocity: float = 3.0, terminal_cost: tuple[float, float] = (0.5, 5.0),
                 n_features: int = 24, dt: float = 0.5, threshold_factor: float = 10.0,
                 gust_velocity: float = -20.0, gust_fraction: float = 0.45, seed: int = 0) -> dict:
    """The residual-triggered hand-over, on a 1-D lunar landing.

    Five measured pieces:

    1. the bang-bang minimum-fuel landing of :func:`min_fuel_landing` -- flight, burn and coast
       times and fuel, all closed form;
    2. a value network (:func:`bellman`) fitted on the *nominal descent corridor* around
       ``(h0, v0)`` over the closed-form flight time, and the HJB residual along the trajectory it
       flies from the nominal start;
    3. the same flight through a gust (a velocity jump part-way down) that takes the state out of
       the fitted corridor -- the residual along it is reported next to the nominal one, and that
       ordering *is* the trust claim;
    4. the hand-over: the relative residual crosses the threshold, the controller switches to
       :func:`mpc_control`, and the touchdown error is reported next to the one the learned policy
       alone achieves;
    5. the Riccati cost to go at the start, next to the network's.

    The terminal cost is deliberately modest (``terminal_cost``, on altitude and on velocity): a
    heavy ``Qf`` turns the value into a function spanning seven decades across the corridor, and the
    feature set of :class:`_ValueFeatures` cannot hold that -- measured on this corridor, the best
    possible projection of the Riccati value onto the features is 3.7 % off with ``Qf = diag(0.5, 5)``
    and 25 % off with ``Qf = 2000 I``.  The corridor is ``[-5, corridor_altitude * h0] x [v0 -
    corridor_velocity, 2]``, the region the nominal descent occupies; the gust is aimed outside it on
    purpose.

    Returns the pieces plus ``table``, a list of formatted lines.
    """
    landing = min_fuel_landing(h0, v0, g, a_max=a_max_factor * g)
    T = landing.flight_time
    problem = LQProblem(A=[[0.0, 1.0], [0.0, 0.0]], B=[[0.0], [1.0]], Q=np.zeros((2, 2)), R=[[1.0]],
                        x0=[h0, v0], t_span=(0.0, T), c=[0.0, -g],
                        Qf=np.diag(np.asarray(terminal_cost, float)), xf=[0.0, 0.0],
                        name="lunar landing")
    corridor = np.array([[-5.0, corridor_altitude * h0],
                         [v0 - corridor_velocity, 2.0]])
    net = bellman(problem, box=corridor, weighted=True, n_features=n_features, poly_degree=2,
                  poly_time_degree=6, n_time=20, n_space=13, seed=seed)

    nominal = _fly(net, [h0, v0], 0.0, T, dt)
    nominal_report = residual_report(net, nominal)
    threshold = max(threshold_factor * nominal_report["relative_rms"], 1e-3)

    gust = (gust_fraction * T, [0.0, gust_velocity])
    disturbed = _fly_with_handover(net, [h0, v0], 0.0, T, dt, threshold, gust=gust, seed=seed)
    disturbed_report = residual_report(net, disturbed, threshold=threshold)
    # the counterfactual: the same gust, the learned policy kept all the way down.  The state has to
    # come from the *nominal* trajectory -- ``disturbed`` already carries the jump, and adding it a
    # second time would launch the vehicle at twice the gust velocity.
    free = _fly(net, _state_after_gust(nominal, gust), gust[0], T, dt)

    reference = riccati_solution(problem)
    end = lambda traj: (float(traj["state"][-1, 0]), float(traj["state"][-1, 1]))  # noqa: E731
    handed = ("never" if disturbed["handed_over_at"] is None
              else f"{disturbed['handed_over_at']:.2f} s")
    table = [
        f"min-fuel (bang-bang): flight {T:.2f} s, coast {landing.coast_time:.2f} s, "
        f"burn {landing.burn_time:.2f} s, fuel {landing.fuel:.2f} m/s",
        f"HJB residual, nominal flight: rms {nominal_report['rms']:.3e}, "
        f"relative rms {nominal_report['relative_rms']:.3e}",
        f"HJB residual, after the gust: rms {disturbed_report['rms']:.3e}, "
        f"relative rms {disturbed_report['relative_rms']:.3e}",
        f"threshold {threshold:.3e} ({threshold_factor:.0f}x the nominal relative rms); "
        f"gust of {gust_velocity:+.1f} m/s at {gust[0]:.2f} s; hand-over at {handed}",
        f"touchdown, policy alone:  h {end(free)[0]:+.3f} m, v {end(free)[1]:+.3f} m/s",
        f"touchdown, with hand-over: h {end(disturbed)[0]:+.3f} m, "
        f"v {end(disturbed)[1]:+.3f} m/s",
        f"cost to go at the start: Riccati {reference['cost']:.2f}, "
        f"network {net.cost_to_go():.2f}",
    ]
    return {"problem": problem, "landing": landing, "net": net, "nominal": nominal,
            "disturbed": disturbed, "free": free, "nominal_report": nominal_report,
            "disturbed_report": disturbed_report, "threshold": threshold, "gust": gust,
            "reference": reference, "table": table}


def _state_after_gust(trajectory: dict, gust: tuple[float, Sequence[float]]) -> Array:
    """The state right after the gust, which is where the no-hand-over flight is restarted."""
    t = np.asarray(trajectory["t"], float)
    index = int(np.argmin(np.abs(t - gust[0])))
    jump = np.atleast_1d(np.asarray(gust[1], float))
    state = np.array(trajectory["state"][index], float)
    state[:jump.size] += jump
    return state
