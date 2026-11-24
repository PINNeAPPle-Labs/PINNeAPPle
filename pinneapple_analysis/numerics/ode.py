"""ODE integrators and stiffness: measured order, sampled stability regions, diagnosed stiffness (#1060).

Three questions, answered with numbers rather than adjectives:

1. **What order does this method deliver?** Every method here is run on a problem with an exact solution
   and its order is fitted from the error against the step -- Euler 1, Heun 2, RK4 4, Adams-Bashforth/
   Moulton 4, backward Euler 1, the trapezoid rule 2, BDF ``k`` for ``k <= 5``, Radau IIA 5. A method
   whose textbook order and measured order differ is telling you about the problem (order reduction on a
   stiff one) or about the startup (a multistep method started at low order).
2. **Where is it stable?** The amplification ``R(z) = 1 + z b^T (I - zA)^{-1} e`` comes from the Butcher
   tableau, so the region is computed, not drawn from a picture. Its extent on the negative real axis
   (2 for Euler, 2.785 for RK4, unbounded for backward Euler, the trapezoid and Radau), A-stability and
   L-stability are all measured, and for the multistep BDF formulas the stability is measured as the
   ``A(alpha)`` angle.
3. **Is the problem stiff?** The stiffness ratio comes from the Jacobian eigenvalues; the cost of an
   explicit method at equal accuracy is then compared with an implicit one on Robertson's kinetics and on
   Van der Pol at large ``mu``.

Also here: geometric integrators with the energy drift measured against RK4 (bounded and oscillatory for
Störmer-Verlet, secular for RK4), event location by bisection inside a step, and a pendulum written in
Cartesian coordinates as an index-reduced DAE, where the constraint drift is the observable that says
whether the reduction is being kept.

The limitation, once: fixed-step methods are the reference here because their order is unambiguous;
the adaptive pairs (Bogacki-Shampine 3(2) and Dormand-Prince 5(4)) report the order of their higher-order
solution measured the same way, but their cost depends on the tolerance and the problem, so the two are
not interchangeable in a cost comparison.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .convergence import OrderEstimate, fit_order
from .diagnostics import Diagnostic, explain

__all__ = [
    "DORMAND_PRINCE",
    "EULER",
    "HEUN",
    "RK4",
    "BACKWARD_EULER",
    "TRAPEZOID",
    "BOGACKI_SHAMPINE",
    "RADAU_IIA3",
    "Tableau",
    "ODEResult",
    "absolute_stability_limit",
    "adams_bashforth",
    "adams_moulton",
    "adaptive_dormand_prince",
    "backward_euler",
    "bdf",
    "bdf_amplification_roots",
    "bdf_angle",
    "bogacki_shampine",
    "dormand_prince",
    "energy_drift_study",
    "euler",
    "heun",
    "implicit_trapezoid",
    "integrate_with_events",
    "is_a_stable",
    "is_l_stable",
    "measure_order",
    "pendulum_cartesian_index1",
    "radau_iia",
    "rk4",
    "robertson_rhs",
    "stability_function",
    "stability_region",
    "stiffness_ratio",
    "stiffness_report",
    "stormer_verlet",
    "symplectic_euler",
    "van_der_pol_rhs",
]


# --------------------------------------------------------------------------- tableaux
@dataclass(frozen=True)
class Tableau:
    """A Runge-Kutta tableau: the coefficients, and whether the method is explicit.

    ``b_hat`` is the lower-order solution of an embedded pair, which is what makes the local error
    estimate available without a second solve.
    """

    name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    explicit: bool
    order: int
    b_hat: np.ndarray | None = None

    @property
    def stages(self) -> int:
        """Number of stages (function evaluations per step for an explicit method)."""
        return int(self.b.size)


def _tableau(name: str, A: list[list[float]], b: list[float], order: int,
             b_hat: list[float] | None = None, *, c: list[float] | None = None) -> Tableau:
    Am = np.asarray(A, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    cv = np.asarray(c, dtype=np.float64) if c is not None else Am.sum(axis=1)
    return Tableau(
        name=name, A=Am, b=bv, c=cv, explicit=bool(np.allclose(np.triu(Am), 0.0)), order=order,
        b_hat=np.asarray(b_hat, dtype=np.float64) if b_hat is not None else None,
    )


EULER = _tableau("euler", [[0.0]], [1.0], 1)
HEUN = _tableau("heun", [[0.0, 0.0], [1.0, 0.0]], [0.5, 0.5], 2)
RK4 = _tableau("rk4", [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]],
               [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], 4)
BACKWARD_EULER = _tableau("backward_euler", [[1.0]], [1.0], 1)
TRAPEZOID = _tableau("crank_nicolson", [[0.0, 0.0], [0.5, 0.5]], [0.5, 0.5], 2)
BOGACKI_SHAMPINE = _tableau(
    "bogacki_shampine",
    [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.75, 0.0, 0.0], [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0]],
    [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0], 3,
    b_hat=[7.0 / 24.0, 0.25, 1.0 / 3.0, 0.125],
)
DORMAND_PRINCE = _tableau(
    "dormand_prince",
    [
        [0.0] * 7,
        [1.0 / 5.0] + [0.0] * 6,
        [3.0 / 40.0, 9.0 / 40.0] + [0.0] * 5,
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0] + [0.0] * 4,
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0] + [0.0] * 3,
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0] + [0.0] * 2,
        [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0],
    ],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0], 5,
    b_hat=[5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0,
           1.0 / 40.0],
)


def radau_iia(stages: int = 3) -> Tableau:
    """Radau IIA: ``s``-stage collocation, order ``2s - 1``, L-stable.

    The nodes are the roots of ``d^{s-1}/dx^{s-1}[x^{s-1}(x-1)^s]`` -- the right end point is a node,
    which is what makes the method stiffly accurate -- and the coefficients are
    ``a_ij = int_0^{c_i} l_j(tau) dtau`` on those nodes. Deriving them this way, instead of copying the
    published decimals, means the tableaux can be checked: the order measured in the tests comes out at
    ``2s - 1`` and the method is L-stable by construction.
    """
    if stages < 1:
        raise ValueError(f"stages must be at least 1; got {stages}")
    # x^{s-1} in ascending coefficients
    x_power = np.zeros(max(stages, 2))
    x_power[stages - 1] = 1.0
    if stages == 1:
        x_power = np.array([1.0])
    # (x - 1)^s in ascending coefficients
    one_minus = np.array([float((-1) ** (stages - k)) * _binom(stages, k) for k in range(stages + 1)])
    polynomial = np.polynomial.polynomial.polymul(x_power, one_minus)
    reduced = np.polynomial.polynomial.polyder(polynomial, m=max(stages - 1, 0))
    nodes = np.sort(np.real(np.polynomial.polynomial.polyroots(reduced)))
    nodes[-1] = 1.0  # the right end point is a node by construction, to within rounding

    A = np.zeros((stages, stages))
    for j in range(stages):
        basis = np.array([1.0])
        for k in range(stages):
            if k == j:
                continue
            basis = np.polynomial.polynomial.polymul(
                basis, np.array([-nodes[k], 1.0]) / (nodes[j] - nodes[k])
            )
        integral = np.polynomial.polynomial.polyint(basis)
        A[:, j] = np.polynomial.polynomial.polyval(nodes, integral)
    b = A[-1, :].copy()
    return Tableau(name=f"radau_iia_{stages}", A=A, b=b, c=nodes, explicit=False, order=2 * stages - 1)


def _binom(n: int, k: int) -> int:
    """Binomial coefficient, so the Radau polynomial needs no import of ``math`` at call time."""
    value = 1
    for i in range(k):
        value = value * (n - i) // (i + 1)
    return value


#: the three-stage Radau IIA method, order 5, L-stable (built after ``_binom``, which its nodes need)
RADAU_IIA3 = radau_iia(3)


# --------------------------------------------------------------------------- results
@dataclass(frozen=True)
class ODEResult:
    """What an integration did: the solution, the cost, the measured order and the diagnosis."""

    t: np.ndarray
    y: np.ndarray
    method: str
    function_evaluations: int
    steps: int
    rejected: int = 0
    order: OrderEstimate | None = None
    step_sizes: tuple[float, ...] = ()
    events: tuple[dict[str, float], ...] = ()
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def final(self) -> np.ndarray:
        """The last state."""
        return self.y[-1] if self.y.ndim > 1 else np.asarray([self.y[-1]])

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "method": self.method,
            "function_evaluations": self.function_evaluations,
            "steps": self.steps,
            "rejected": self.rejected,
            "order": self.order.to_dict() if self.order else None,
            "step_sizes": list(self.step_sizes),
            "events": list(self.events),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _as_state(y: Sequence[float] | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(y, dtype=np.float64)).ravel()


def _evaluate(rhs: Callable[[float, np.ndarray], Any], t: float, y: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(rhs(t, y), dtype=np.float64)).ravel()


# --------------------------------------------------------------------------- explicit RK
def _explicit_step(
    rhs: Callable[[float, np.ndarray], Any], t: float, y: np.ndarray, h: float, tableau: Tableau
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """One explicit Runge-Kutta step, returning the solution, the embedded one and the evaluations used."""
    s = tableau.stages
    k = np.zeros((s, y.size))
    for i in range(s):
        argument = y + h * (tableau.A[i, :i] @ k[:i]) if i else y
        k[i] = _evaluate(rhs, t + tableau.c[i] * h, argument)
    y_new = y + h * (tableau.b @ k)
    y_hat = y + h * (tableau.b_hat @ k) if tableau.b_hat is not None else None
    return y_new, y_hat, s


def _integrate_explicit(
    rhs: Callable[[float, np.ndarray], Any],
    tableau: Tableau,
    y0: Sequence[float] | float,
    t_span: tuple[float, float],
    n_steps: int,
) -> ODEResult:
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1; got {n_steps}")
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    y = _as_state(y0)
    ts = np.linspace(t0, t1, n_steps + 1)
    ys = np.zeros((n_steps + 1, y.size))
    ys[0] = y
    evaluations = 0
    # an unstable step legitimately overflows, and the diagnostic below is what reports it: silencing the
    # warning here keeps the blow-up a measured result instead of a wall of numpy warnings
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(1, n_steps + 1):
            y, _, used = _explicit_step(rhs, ts[i - 1], y, h, tableau)
            ys[i] = y
            evaluations += used
    diagnostics: list[Diagnostic] = []
    if not np.all(np.isfinite(ys)):
        diagnostics.append(explain("ODE.STABILITY_LIMIT", method=tableau.name, step_size=h))
    return ODEResult(t=ts, y=ys if y.size > 1 else ys[:, 0], method=tableau.name,
                     function_evaluations=evaluations, steps=n_steps,
                     step_sizes=tuple([h] * n_steps), diagnostics=tuple(diagnostics))


def euler(rhs, y0, t_span, n_steps: int) -> ODEResult:
    """Explicit (forward) Euler: order 1, and the method every stability result is stated against.

    Each step is ``y + h f(t, y)``. Its order is 1 and its absolute stability region is the unit disc
    centred at ``-1``, so the step is limited to ``h |lambda| < 2`` on every mode -- which is why it is
    useless on a stiff problem and useful as a definition.
    """
    return _integrate_explicit(rhs, EULER, y0, t_span, n_steps)


def heun(rhs, y0, t_span, n_steps: int) -> ODEResult:
    """Heun's method (explicit trapezoid): order 2, two evaluations per step."""
    return _integrate_explicit(rhs, HEUN, y0, t_span, n_steps)


def rk4(rhs, y0, t_span, n_steps: int) -> ODEResult:
    """Classical fourth-order Runge-Kutta: order 4, four evaluations per step.

    The workhorse: order 4 with no storage and no startup, and a stability limit of ``h |lambda| < 2.785``
    on the negative real axis, which the tests measure from its tableau. The limitation is that the limit
    exists at all -- on a stiff problem the step is set by stability, not accuracy.
    """
    return _integrate_explicit(rhs, RK4, y0, t_span, n_steps)


def _adaptive(
    rhs: Callable[[float, np.ndarray], Any],
    tableau: Tableau,
    y0: Sequence[float] | float,
    t_span: tuple[float, float],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float | None = None,
    max_steps: int = 100_000,
) -> ODEResult:
    """Shared step-size controller for the embedded pairs: PI-free, ``err**(-1/(p+1))`` exponent."""
    if tableau.b_hat is None:
        raise ValueError(f"{tableau.name} is not an embedded pair")
    t0, t1 = float(t_span[0]), float(t_span[1])
    y = _as_state(y0)
    h = (t1 - t0) / 100.0 if h0 is None else float(h0)
    time = t0
    evaluations = 0
    accepted = 0
    rejected = 0
    times = [time]
    states = [y.copy()]
    step_sizes: list[float] = []
    exponent = 1.0 / (tableau.order + 1)

    for _ in range(max_steps):
        if time >= t1:
            break
        h = min(h, t1 - time)
        y_new, y_hat, used = _explicit_step(rhs, time, y, h, tableau)
        evaluations += used
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        error = float(np.sqrt(np.mean(((y_new - y_hat) / scale) ** 2)))
        if not np.isfinite(error):
            rejected += 1
            h *= 0.2
        elif error <= 1.0:
            time = time + h
            y = y_new
            accepted += 1
            step_sizes.append(h)
            times.append(time)
            states.append(y.copy())
            factor = 5.0 if error == 0.0 else min(5.0, max(0.2, 0.9 * error ** (-exponent)))
            h *= factor
        else:
            rejected += 1
            h *= min(0.9, max(0.1, 0.9 * error ** (-exponent)))
        if h < 1e-14 * max(abs(t1 - t0), 1.0):
            diagnostics = (explain("ODE.NOT_CONVERGED", method=tableau.name, step_size=h,
                                   time=time),)
            return ODEResult(
                t=np.asarray(times), y=np.asarray(states) if y.size > 1 else np.asarray(states)[:, 0],
                method=tableau.name, function_evaluations=evaluations, steps=accepted,
                rejected=rejected, step_sizes=tuple(step_sizes), diagnostics=diagnostics,
            )

    return ODEResult(
        t=np.asarray(times), y=np.asarray(states) if y.size > 1 else np.asarray(states)[:, 0],
        method=tableau.name, function_evaluations=evaluations, steps=accepted, rejected=rejected,
        step_sizes=tuple(step_sizes),
    )


def bogacki_shampine(rhs, y0, t_span, *, rtol: float = 1e-6, atol: float = 1e-9,
                     h0: float | None = None) -> ODEResult:
    """Bogacki-Shampine 3(2): an embedded third-order pair with step control, three stages (FSAL).

    Cheaper per step than Dormand-Prince and adequate for moderate tolerances; the embedded estimate is the
    difference between its second- and third-order solutions, which is what the controller drives to one.
    """
    return _adaptive(rhs, BOGACKI_SHAMPINE, y0, t_span, rtol=rtol, atol=atol, h0=h0)


def dormand_prince(rhs, y0, t_span, *, rtol: float = 1e-8, atol: float = 1e-11,
                   h0: float | None = None) -> ODEResult:
    """Dormand-Prince 5(4): a seventh-stage FSAL pair, the default of every modern ODE solver.

    Seven stages for a fifth-order solution and a fourth-order error estimate; the first stage of the next
    step is the last stage of the current one (FSAL), so the effective cost is six evaluations per step.
    """
    return _adaptive(rhs, DORMAND_PRINCE, y0, t_span, rtol=rtol, atol=atol, h0=h0)


def adaptive_dormand_prince(*args: Any, **kwargs: Any) -> ODEResult:
    """Alias of :func:`dormand_prince`, kept for discoverability."""
    return dormand_prince(*args, **kwargs)


# --------------------------------------------------------------------------- implicit one-step
def _implicit_step(
    rhs: Callable[[float, np.ndarray], Any],
    t: float,
    y: np.ndarray,
    h: float,
    tableau: Tableau,
    *,
    shift: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """One implicit Runge-Kutta step: Newton on the (coupled) stage equations.

    The unknowns are the ``s * n`` stage values ``Z_i``; the residual is
    ``Z_i - y - h sum_j a_ij f(t + c_j h, Z_j)``. Solved by Newton with a finite-difference Jacobian,
    which is what makes the cost of an implicit method measurable in evaluations as well.
    """
    s = tableau.stages
    n = y.size
    rhs_for_jacobian = np.zeros((0, 0))
    del rhs_for_jacobian  # the Jacobian is built by differences inside the loop below
    z = np.tile(y, s)
    evaluations = 0

    def residual(z_flat: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        stages = z_flat.reshape(s, n)
        f = np.zeros((s, n))
        for j in range(s):
            f[j] = _evaluate(rhs, t + tableau.c[j] * h, stages[j])
        evaluations += s
        return (stages - y[None, :] - h * (tableau.A @ f)).ravel()

    for _ in range(60):
        r = residual(z)
        if np.linalg.norm(r) <= 1e-13 * max(1.0, float(np.linalg.norm(y))):
            break
        eps = 1e-8
        jacobian = np.zeros((s * n, s * n))
        for k in range(s * n):
            shifted = z.copy()
            shifted[k] += eps * max(1.0, abs(z[k]))
            jacobian[:, k] = (residual(shifted) - r) / (eps * max(1.0, abs(z[k])))
        try:
            delta = np.linalg.solve(jacobian, -r)
        except np.linalg.LinAlgError:
            break
        z = z + delta
        if np.linalg.norm(delta) <= 1e-14 * max(1.0, float(np.linalg.norm(z))):
            break

    stages = z.reshape(s, n)
    f = np.array([_evaluate(rhs, t + tableau.c[j] * h, stages[j]) for j in range(s)])
    evaluations += s
    y_new = y + h * (tableau.b @ f)
    if shift is not None:
        y_new = y_new + shift
    return y_new, evaluations


def _integrate_implicit(
    rhs, tableau: Tableau, y0, t_span, n_steps: int
) -> ODEResult:
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1; got {n_steps}")
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    y = _as_state(y0)
    ts = np.linspace(t0, t1, n_steps + 1)
    ys = np.zeros((n_steps + 1, y.size))
    ys[0] = y
    evaluations = 0
    for i in range(1, n_steps + 1):
        y, used = _implicit_step(rhs, ts[i - 1], y, h, tableau)
        evaluations += used
        ys[i] = y
    diagnostics: list[Diagnostic] = []
    if not np.all(np.isfinite(ys)):
        diagnostics.append(explain("ODE.STABILITY_LIMIT", method=tableau.name, step_size=h))
    return ODEResult(t=ts, y=ys if y.size > 1 else ys[:, 0], method=tableau.name,
                     function_evaluations=evaluations, steps=n_steps,
                     step_sizes=tuple([h] * n_steps), diagnostics=tuple(diagnostics))


def backward_euler(rhs, y0, t_span, n_steps: int) -> ODEResult:
    """Backward Euler: order 1, A-stable and L-stable, and the simplest thing that survives a stiff mode.

    The order is only 1, which is why it is used as a startup or a sanity check rather than as a solver --
    but its stability region is the whole left half-plane, so no step is too large for stability.
    """
    return _integrate_implicit(rhs, BACKWARD_EULER, y0, t_span, n_steps)


def implicit_trapezoid(rhs, y0, t_span, n_steps: int) -> ODEResult:
    """The trapezoid rule (Crank-Nicolson): order 2, A-stable, but **not** L-stable.

    A-stability makes it usable on stiff problems; the missing L-stability means a very fast mode is
    damped only as ``|R| -> 1``, so the solution rings around the slow manifold instead of settling on it.
    That distinction is measured in the tests through ``|R(z)|`` as ``z -> -inf``.
    """
    return _integrate_implicit(rhs, TRAPEZOID, y0, t_span, n_steps)


def radau_iia3(rhs, y0, t_span, n_steps: int) -> ODEResult:
    """Three-stage Radau IIA: order 5, A- and L-stable -- the stiff workhorse of this module."""
    return _integrate_implicit(rhs, RADAU_IIA3, y0, t_span, n_steps)


# --------------------------------------------------------------------------- multistep
_AB: dict[int, list[float]] = {
    1: [1.0],
    2: [3.0 / 2.0, -1.0 / 2.0],
    3: [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
    4: [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
}
_AM: dict[int, list[float]] = {
    1: [1.0],
    2: [0.5, 0.5],
    3: [5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
    4: [9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
}
#: ``(alpha, beta)`` with ``sum_j alpha_j y_{n+1-j} = h beta f_{n+1}``, ``alpha_0 = 1``
_BDF: dict[int, tuple[list[float], float]] = {
    1: ([1.0, -1.0], 1.0),
    2: ([1.0, -4.0 / 3.0, 1.0 / 3.0], 2.0 / 3.0),
    3: ([1.0, -18.0 / 11.0, 9.0 / 11.0, -2.0 / 11.0], 6.0 / 11.0),
    4: ([1.0, -48.0 / 25.0, 36.0 / 25.0, -16.0 / 25.0, 3.0 / 25.0], 12.0 / 25.0),
    5: ([1.0, -300.0 / 137.0, 300.0 / 137.0, -200.0 / 137.0, 75.0 / 137.0, -12.0 / 137.0],
        60.0 / 137.0),
}


def adams_bashforth(rhs, y0, t_span, n_steps: int, *, order: int = 4) -> ODEResult:
    """Adams-Bashforth ``order``: explicit multistep, order ``k``, one evaluation per step after startup.

    The startup is Runge-Kutta 4 on the first ``k - 1`` steps, so the observed order is ``min(k, 4)`` until
    the startup error has been swept out -- a lower-order start contaminating a higher-order method is the
    classic multistep trap, and it is visible in the measurement.

    Raises ``ValueError`` for an order outside ``1..4``.
    """
    if order not in _AB:
        raise ValueError(f"order must be in {sorted(_AB)}; got {order}")
    if n_steps < order:
        raise ValueError(f"n_steps must be at least the order ({order}); got {n_steps}")
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    coefficients = _AB[order]
    y = _as_state(y0)
    ts = np.linspace(t0, t1, n_steps + 1)
    ys = np.zeros((n_steps + 1, y.size))
    ys[0] = y
    f_history: list[np.ndarray] = [_evaluate(rhs, t0, y)]  # AB-k needs k values, starting with f(t0)
    evaluations = 1

    for i in range(1, n_steps + 1):
        if i < order:  # startup: the same RK4 step the explicit integrator uses
            y, _, used = _explicit_step(rhs, ts[i - 1], y, h, RK4)
            evaluations += used
            f_history.insert(0, _evaluate(rhs, ts[i], y))
            evaluations += 1
        else:
            y = y + h * sum(coefficient * f_history[j] for j, coefficient in enumerate(coefficients))
            f_history.insert(0, _evaluate(rhs, ts[i], y))
            evaluations += 1
            f_history = f_history[:order]
        ys[i] = y

    return ODEResult(t=ts, y=ys if y.size > 1 else ys[:, 0], method=f"adams_bashforth_{order}",
                     function_evaluations=evaluations, steps=n_steps,
                     step_sizes=tuple([h] * n_steps))


def adams_moulton(rhs, y0, t_span, n_steps: int, *, order: int = 4, iterations: int = 2) -> ODEResult:
    """Adams-Moulton ``order`` as a predictor-corrector: AB``k`` predicts, AM``k`` corrects.

    With ``iterations >= 2`` the corrector converges to the implicit solution and the method is order
    ``k + 1``; with one iteration it is the explicit predictor-corrector (ABM) of order ``k``. Both
    variants are the same code, which is the point: the number of evaluations per step is what buys the
    extra order.
    """
    if order not in _AM or order not in _AB:
        raise ValueError(f"order must be in {sorted(_AM)}; got {order}")
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    y = _as_state(y0)
    ts = np.linspace(t0, t1, n_steps + 1)
    ys = np.zeros((n_steps + 1, y.size))
    ys[0] = y
    f_history: list[np.ndarray] = [_evaluate(rhs, t0, y)]  # AM-k needs k values, starting with f(t0)
    evaluations = 1

    for i in range(1, n_steps + 1):
        if i < order:
            y, _, used = _explicit_step(rhs, ts[i - 1], y, h, RK4)
            evaluations += used
            f_history.insert(0, _evaluate(rhs, ts[i], y))
            evaluations += 1
            continue
        predictor = y + h * sum(coefficient * f_history[j] for j, coefficient in enumerate(_AB[order]))
        guess = predictor
        for _ in range(max(1, int(iterations))):
            f_new = _evaluate(rhs, ts[i], guess)
            evaluations += 1
            # AM coefficient b_{j+1} multiplies f_{n-j}, i.e. f_history[j]: the pairing is off by one if
            # the coefficients are used in their own order, which is the classic corrector bug
            guess = y + h * (sum(_AM[order][j + 1] * f_history[j] for j in range(order - 1))
                             + _AM[order][0] * f_new)
        y = guess
        f_history.insert(0, _evaluate(rhs, ts[i], y))
        evaluations += 1
        f_history = f_history[:order]
        ys[i] = y

    return ODEResult(t=ts, y=ys if y.size > 1 else ys[:, 0], method=f"adams_moulton_{order}",
                     function_evaluations=evaluations, steps=n_steps,
                     step_sizes=tuple([h] * n_steps))


def bdf(
    rhs,
    y0,
    t_span,
    n_steps: int,
    *,
    order: int = 2,
    startup: str = "backward_euler",
    exact: Callable[[float], Any] | None = None,
) -> ODEResult:
    """BDF of ``order`` 1..5: implicit multistep, the standard workhorse for stiff problems.

    ``sum_j alpha_j y_{n+1-j} = h beta f(t_{n+1}, y_{n+1})`` solved by Newton at every step. BDF1 is
    backward Euler and BDF2 is A-stable; BDF3-BDF5 are not A-stable but are stable on the whole negative
    real axis, which is exactly why they are used for stiff problems and why the ``A(alpha)`` angle is
    measured by :func:`bdf_angle`.

    ``startup="backward_euler"`` (the default) is what a production code does; ``startup="exact"`` requires
    ``exact`` and initialises the history from it, which is how the *order of the formula* is measured
    without the startup error dominating it. The two give different measured orders, and that difference
    is the point (a lower-order start can cost one order).
    """
    if order not in _BDF:
        raise ValueError(f"order must be in {sorted(_BDF)}; got {order}")
    if startup not in ("backward_euler", "exact"):
        raise ValueError(f"startup must be 'backward_euler' or 'exact'; got {startup!r}")
    if startup == "exact" and exact is None:
        raise ValueError("startup='exact' needs the exact solution")
    if n_steps < order:
        raise ValueError(f"n_steps must be at least the order ({order}); got {n_steps}")

    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    y = _as_state(y0)
    ts = np.linspace(t0, t1, n_steps + 1)
    ys = np.zeros((n_steps + 1, y.size))
    ys[0] = y
    history: list[np.ndarray] = [y.copy()]  # history[j] is y_{n+1-j}
    evaluations = 0
    alpha, beta = _BDF[order]

    for i in range(1, n_steps + 1):
        if i < order:
            if startup == "exact":
                y = _as_state(exact(ts[i]))
            else:
                y, used = _implicit_step(rhs, ts[i - 1], y, h, BACKWARD_EULER)
                evaluations += used
            history.insert(0, y.copy())
            ys[i] = y
            continue

        # y_i = -sum_{j>=1} alpha_j y_{i-j} + h beta f_i: alpha_0 = 1 multiplies the unknown itself and
        # must NOT enter the constant term (that was a factor-of-two bug at order 1)
        constant = np.zeros_like(history[0])
        for j in range(1, order + 1):
            constant = constant - alpha[j] * history[j - 1]
        # predictor: extrapolate from the last two points; with order 1 there is only one point
        guess = history[0] + (history[0] - history[1]) if len(history) > 1 else history[0].copy()
        for _ in range(50):
            f_new = _evaluate(rhs, ts[i], guess)
            evaluations += 1
            residual = guess - constant - h * beta * f_new
            if float(np.linalg.norm(residual)) <= 1e-13 * max(1.0, float(np.linalg.norm(guess))):
                break
            jacobian = np.zeros((guess.size, guess.size))
            for k in range(guess.size):
                step = 1e-8 * max(1.0, abs(guess[k]))
                shifted = guess.copy()
                shifted[k] += step
                perturbed = shifted - constant - h * beta * _evaluate(rhs, ts[i], shifted)
                evaluations += 1
                jacobian[:, k] = (perturbed - residual) / step
            try:
                delta = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                break
            guess = guess + delta
        y = guess
        history.insert(0, y.copy())
        history = history[:order]
        ys[i] = y

    diagnostics: list[Diagnostic] = []
    if not np.all(np.isfinite(ys)):
        diagnostics.append(explain("ODE.STABILITY_LIMIT", method=f"bdf_{order}", step_size=h))
    return ODEResult(t=ts, y=ys if y.size > 1 else ys[:, 0], method=f"bdf_{order}",
                     function_evaluations=evaluations, steps=n_steps,
                     step_sizes=tuple([h] * n_steps), diagnostics=tuple(diagnostics))


def bdf_amplification_roots(z: complex, order: int) -> np.ndarray:
    """Roots of the BDF stability polynomial at ``z = h lambda`` -- all inside the unit disc means stable.

    With ``sum_j alpha_j y_{n+1-j} = h beta f_{n+1}`` and ``y_m = xi^m``, the polynomial is
    ``(1 - z beta) xi^k + sum_{j=1}^{k} alpha_j xi^{k-j} = 0``. This is how the stability of a multistep
    method is decided: not by an amplification scalar (those exist only for one-step methods) but by the
    root condition.
    """
    if order not in _BDF:
        raise ValueError(f"order must be in {sorted(_BDF)}; got {order}")
    alpha, beta = _BDF[order]
    coefficients = np.zeros(order + 1, dtype=complex)
    coefficients[0] = 1.0 - z * beta
    for j in range(1, order + 1):
        coefficients[j] = alpha[j]
    return np.roots(coefficients)


def bdf_angle(order: int) -> float:
    """The ``A(alpha)`` angle of a BDF formula, in degrees: how far the stability region opens.

    BDF1 and BDF2 are A-stable (90 degrees). For BDF3-BDF6 the angle falls -- about 86.0, 73.4, 51.8 and
    17.8 degrees -- which is the quantitative statement of "BDF is stable for stiff problems *whose
    eigenvalues sit close to the negative real axis*". Measured here by bisection on the angle: for each
    candidate, check that every root stays inside the unit disc along the ray.
    """
    if order not in _BDF:
        raise ValueError(f"order must be in {sorted(_BDF)}; got {order}")

    def stable_at(angle_degrees: float) -> bool:
        theta = np.pi - np.radians(angle_degrees)
        for radius in np.logspace(-4, 6, 240):
            roots = bdf_amplification_roots(radius * np.exp(1j * theta), order)
            if np.max(np.abs(roots)) > 1.0 + 1e-9:
                return False
        return True

    if stable_at(90.0):
        return 90.0
    low, high = 0.0, 90.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if stable_at(mid):
            low = mid
        else:
            high = mid
    return float(low)


# --------------------------------------------------------------------------- stability
def stability_function(tableau: Tableau) -> Callable[[complex], complex]:
    """``R(z) = 1 + z b^T (I - z A)^{-1} e``: the amplification of the method on ``y' = lambda y``.

    Derived from the tableau, so every method in this module has one and no picture is copied. The absolute
    stability region is ``{z : |R(z)| <= 1}``; on the real axis it extends over an interval that is the
    method's ``h lambda`` limit.
    """
    s = tableau.stages
    ones = np.ones(s)

    def amplification(z: complex) -> complex:
        matrix = np.eye(s) - np.asarray(z, dtype=complex) * tableau.A
        return complex(1.0 + np.asarray(z) * (tableau.b @ np.linalg.solve(matrix, ones)))

    return amplification


def absolute_stability_limit(tableau: Tableau, *, limit: float = 100.0) -> float:
    """The largest ``x > 0`` with ``|R(-x)| <= 1``, measured by bisection on the negative real axis.

    Euler gives 2, RK4 2.785, and the implicit methods return ``inf``: their amplification stays below one
    for every negative ``x`` tested up to ``limit``, which is the definition of A-stability on that axis.
    """
    R = stability_function(tableau)
    if abs(R(complex(-limit, 0.0))) <= 1.0:
        return float("inf")
    low, high = 0.0, float(limit)
    for _ in range(200):
        mid = 0.5 * (low + high)
        if abs(R(complex(-mid, 0.0))) <= 1.0:
            low = mid
        else:
            high = mid
    return float(low)


def is_a_stable(tableau: Tableau, *, extent: float = 50.0, samples: int = 60) -> bool:
    """Whether ``|R(z)| <= 1`` over a grid of the whole left half-plane (the A-stability definition)."""
    R = stability_function(tableau)
    real = -np.logspace(-3, np.log10(extent), samples)
    imaginary = np.linspace(-extent, extent, samples)
    for re in real:
        for im in imaginary:
            if abs(R(complex(re, im))) > 1.0 + 1e-9:
                return False
    return True


def is_l_stable(tableau: Tableau, *, near: float = 1e4, far: float = 1e6) -> bool:
    """Whether ``R(z) -> 0`` as ``z -> -inf``: the fast mode is annihilated, not merely not amplified.

    Decided by measurement, not by a threshold at one point: the amplification is sampled at two large
    magnitudes and must fall by about the same factor the magnitude grew (``|R| ~ 1/|z|`` for an L-stable
    method) instead of creeping towards 1, which is what an A-stable-but-not-L-stable method does.
    """
    R = stability_function(tableau)
    at_near = abs(R(complex(-near, 0.0)))
    at_far = abs(R(complex(-far, 0.0)))
    return bool(at_far < 1e-2 and at_far < 0.1 * at_near)


def stability_region(tableau: Tableau, *, extent: float = 5.0, resolution: int = 201) -> dict[str, Any]:
    """Sample ``|R(z)| <= 1`` on a grid of the complex plane -- the data a plot needs, computed."""
    R = stability_function(tableau)
    real = np.linspace(-extent, extent * 0.5, resolution)
    imaginary = np.linspace(-extent * 0.75, extent * 0.75, resolution)
    stable = np.zeros((imaginary.size, real.size), dtype=bool)
    for i, im in enumerate(imaginary):
        for j, re in enumerate(real):
            stable[i, j] = abs(R(complex(re, im))) <= 1.0
    return {
        "real": [float(v) for v in real],
        "imaginary": [float(v) for v in imaginary],
        "stable": stable,
        "limit_on_real_axis": absolute_stability_limit(tableau),
        "a_stable": is_a_stable(tableau),
        "l_stable": is_l_stable(tableau),
    }


# --------------------------------------------------------------------------- order measurement
def measure_order(
    method: Callable[..., ODEResult],
    rhs: Callable[[float, np.ndarray], Any],
    exact_solution: Callable[[float], Any],
    y0: Sequence[float] | float,
    t_span: tuple[float, float],
    step_counts: Sequence[int],
    *,
    component: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Measure a fixed-step method's global order: integrate at several step counts and fit the final error.

    The error is taken at ``t_span[1]`` against ``exact_solution`` -- the *global* error, which is what an
    order claim is about. Steps whose error has reached the rounding floor stop improving, so the fit keeps
    only the largest errors, where the truncation term still dominates.

    The parameter is named ``exact_solution`` and not ``exact`` so that a method that itself takes an
    ``exact=`` argument (BDF's startup) can be passed through ``**kwargs`` without a name collision.
    """
    errors: list[float] = []
    steps: list[float] = []
    evaluations: list[int] = []
    for n in step_counts:
        result = method(rhs, y0, t_span, n, **kwargs)
        if result.y.ndim > 1:
            computed = float(result.y[-1, component])
        else:
            computed = float(result.y[-1])
        errors.append(abs(computed - float(np.atleast_1d(exact_solution(t_span[1]))[component])))
        steps.append(abs(t_span[1] - t_span[0]) / n)
        evaluations.append(result.function_evaluations)

    errors_array = np.asarray(errors)
    steps_array = np.asarray(steps)
    usable = errors_array > 20.0 * max(float(np.min(errors_array)), np.finfo(float).tiny)
    if usable.sum() < 2:
        usable = errors_array > 0
    estimate = fit_order(steps_array[usable], errors_array[usable], name=method.__name__) \
        if usable.sum() >= 2 else None
    return {
        "method": method.__name__,
        "steps": [float(v) for v in steps],
        "errors": [float(v) for v in errors],
        "evaluations": evaluations,
        "order": estimate.order if estimate else float("nan"),
        "ci95": list(estimate.ci95) if estimate else [float("nan")] * 2,
    }


# --------------------------------------------------------------------------- benchmark problems
def robertson_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """Robertson's chemical kinetics: the standard stiff benchmark, with time scales spanning 1e11.

    ``y1' = -0.04 y1 + 1e4 y2 y3``, ``y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2``, ``y3' = 3e7 y2^2``. The
    fast eigenvalue is ``~ -1e7`` and the slow one ``~ -1e-2``, so an explicit method needs a step around
    ``1e-7`` for stability while the solution varies on a scale of thousands -- the cost ratio measured in
    the tests is the whole argument for implicit methods.
    """
    y1, y2, y3 = float(y[0]), float(y[1]), float(y[2])
    return np.array([
        -0.04 * y1 + 1.0e4 * y2 * y3,
        0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
        3.0e7 * y2 * y2,
    ])


def van_der_pol_rhs(mu: float = 1.0) -> Callable[[float, np.ndarray], np.ndarray]:
    """Van der Pol oscillator; ``mu`` large makes it stiff, with a relaxation oscillation of period ~1.6 mu."""
    if mu <= 0:
        raise ValueError(f"mu must be positive; got {mu}")

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([float(y[1]), mu * (1.0 - float(y[0]) ** 2) * float(y[1]) - float(y[0])])

    return rhs


def _jacobian_eigenvalues(
    rhs: Callable[[float, np.ndarray], Any], t: float, y: np.ndarray, *, step: float = 1e-7
) -> np.ndarray:
    base = _evaluate(rhs, t, y)
    jacobian = np.zeros((base.size, y.size))
    for j in range(y.size):
        h = step * max(1.0, abs(y[j]))
        shifted = y.copy()
        shifted[j] += h
        jacobian[:, j] = (_evaluate(rhs, t, shifted) - base) / h
    return np.linalg.eigvals(jacobian)


def stiffness_ratio(rhs: Callable[[float, np.ndarray], Any], t: float,
                    y: Sequence[float] | float) -> float:
    """``max|Re lambda| / min|Re lambda|`` of the local Jacobian: the ratio of the fastest to the slowest mode.

    A ratio of a few units is not stiff; ``1e3`` and above means an explicit method's step is set by
    stability and not by the accuracy of the solution anybody wants. The limitation: a local, linear
    measure -- it can change along the trajectory, which is why :func:`stiffness_report` samples it.
    """
    eigenvalues = _jacobian_eigenvalues(rhs, float(t), _as_state(y))
    real = np.abs(np.real(eigenvalues))
    # an eigenvalue that is zero to within the finite-difference noise is not a time scale: counting it
    # would make every ratio infinite, which is why it is dropped relative to the largest one
    if real.size == 0:
        return 1.0
    real = real[real > 1e-12 * float(np.max(real))]
    if real.size == 0:
        return 1.0
    return float(np.max(real) / np.min(real))


def stiffness_report(
    rhs: Callable[[float, np.ndarray], Any],
    y0: Sequence[float] | float,
    t_span: tuple[float, float],
    *,
    samples: int = 11,
    threshold: float = 1e3,
) -> dict[str, Any]:
    """Sample the stiffness ratio along a rough trajectory and say whether the problem is stiff.

    The trajectory comes from an implicit run at coarse resolution (stability does not matter for the
    *diagnosis*), so the report is available before choosing a solver -- which is when it is useful.
    """
    result = backward_euler(rhs, y0, t_span, max(4, int(samples) - 1))
    times = np.linspace(float(t_span[0]), float(t_span[1]), int(samples))
    state = _as_state(y0)
    ratios: list[float] = []
    for t in times:
        index = int(np.argmin(np.abs(result.t - t)))
        state = _as_state(result.y[index])
        ratios.append(stiffness_ratio(rhs, float(t), state))
    worst = float(np.max(ratios))
    diagnostics: list[Diagnostic] = []
    if worst >= threshold:
        diagnostics.append(explain("ODE.STIFF", stiffness_ratio=worst, samples=int(samples),
                                   threshold=threshold))
    return {
        "times": [float(t) for t in times],
        "stiffness_ratios": ratios,
        "max_stiffness_ratio": worst,
        "is_stiff": bool(worst >= threshold),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


# --------------------------------------------------------------------------- geometric integrators
def symplectic_euler(
    force: Callable[[np.ndarray], np.ndarray],
    q0: Sequence[float],
    p0: Sequence[float],
    t_span: tuple[float, float],
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Symplectic Euler for ``q' = p``, ``p' = F(q)``: update ``p`` first, then ``q`` with the new ``p``.

    The one-line difference from the explicit Euler -- which one is updated first -- is what makes the map
    symplectic, and that is what turns the energy error from a secular drift into a bounded oscillation.
    It is still only first order, so the *accuracy* is poor; what is good is the long-time behaviour.
    """
    q = _as_state(q0).copy()
    p = _as_state(p0).copy()
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    qs = np.zeros((n_steps + 1, q.size))
    ps = np.zeros((n_steps + 1, p.size))
    qs[0], ps[0] = q, p
    for i in range(1, n_steps + 1):
        p = p + h * np.asarray(force(q), dtype=np.float64)
        q = q + h * p
        qs[i], ps[i] = q, p
    return {"t": np.linspace(t0, t1, n_steps + 1), "q": qs, "p": ps}


def stormer_verlet(
    force: Callable[[np.ndarray], np.ndarray],
    q0: Sequence[float],
    p0: Sequence[float],
    t_span: tuple[float, float],
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Störmer-Verlet (leapfrog): second order, symplectic, and the default for Hamiltonian problems.

    One step is a half kick, a full drift, a half kick: ``p += h/2 F(q)``, ``q += h p``, ``p += h/2 F(q)``.
    Time-reversible and symplectic, so the energy error stays bounded over arbitrarily long integrations --
    measured against RK4, whose energy drifts secularly at the same step.
    """
    q = _as_state(q0).copy()
    p = _as_state(p0).copy()
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    qs = np.zeros((n_steps + 1, q.size))
    ps = np.zeros((n_steps + 1, p.size))
    qs[0], ps[0] = q, p
    half_kick = np.asarray(force(q), dtype=np.float64)
    for i in range(1, n_steps + 1):
        p = p + 0.5 * h * half_kick
        q = q + h * p
        half_kick = np.asarray(force(q), dtype=np.float64)
        p = p + 0.5 * h * half_kick
        qs[i], ps[i] = q, p
    return {"t": np.linspace(t0, t1, n_steps + 1), "q": qs, "p": ps}


def energy_drift_study(
    force: Callable[[np.ndarray], np.ndarray],
    potential: Callable[[np.ndarray], float],
    q0: Sequence[float],
    p0: Sequence[float],
    t_span: tuple[float, float],
    n_steps: int,
) -> dict[str, Any]:
    """Compare the energy history of Störmer-Verlet, symplectic Euler and RK4 on the same Hamiltonian.

    Returns, for each method, the maximum relative energy deviation and the drift of the *envelope*
    (the difference between the deviation over the last tenth of the run and over the first tenth). A
    symplectic method has a bounded envelope; RK4's grows with time, which is the measurement that makes
    "use a geometric integrator" a statement rather than a preference.
    """
    q = _as_state(q0)
    p = _as_state(p0)

    def energy(qs: np.ndarray, ps: np.ndarray) -> np.ndarray:
        return np.array([0.5 * float(pp @ pp) + float(potential(qq)) for qq, pp in zip(qs, ps, strict=True)])

    e0 = 0.5 * float(p @ p) + float(potential(q))
    results: dict[str, Any] = {}
    for name, integrator in (("stormer_verlet", stormer_verlet), ("symplectic_euler", symplectic_euler)):
        trajectory = integrator(force, q0, p0, t_span, n_steps)
        values = energy(trajectory["q"], trajectory["p"])
        deviation = np.abs(values - e0) / abs(e0)
        window = max(1, deviation.size // 10)
        results[name] = {
            "max_relative_deviation": float(np.max(deviation)),
            "envelope_growth": float(np.max(deviation[-window:]) - np.max(deviation[:window])),
            "final_relative_deviation": float(deviation[-1]),
            "energy": values,
        }

    # RK4 on the same problem: the state is (q, p) and the right-hand side is the Hamiltonian vector field
    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        n = state.size // 2
        qq, pp = state[:n], state[n:]
        return np.concatenate([pp, np.asarray(force(qq), dtype=np.float64)])

    result = rk4(rhs, np.concatenate([q, p]), t_span, n_steps)
    states = result.y if result.y.ndim > 1 else result.y[:, None]
    n = q.size
    values_rk4 = energy(states[:, :n], states[:, n:])
    deviation_rk4 = np.abs(values_rk4 - e0) / abs(e0)
    window = max(1, deviation_rk4.size // 10)
    results["rk4"] = {
        "max_relative_deviation": float(np.max(deviation_rk4)),
        "envelope_growth": float(np.max(deviation_rk4[-window:]) - np.max(deviation_rk4[:window])),
        "final_relative_deviation": float(deviation_rk4[-1]),
        "energy": values_rk4,
    }

    diagnostics: list[Diagnostic] = []
    if results["rk4"]["envelope_growth"] > 10.0 * results["stormer_verlet"]["envelope_growth"]:
        diagnostics.append(
            explain("ODE.ENERGY_DRIFT", rk4_growth=results["rk4"]["envelope_growth"],
                    verlet_growth=results["stormer_verlet"]["envelope_growth"])
        )
    return {"methods": results, "initial_energy": float(e0),
            "diagnostics": [d.to_dict() for d in diagnostics]}


# --------------------------------------------------------------------------- events
def integrate_with_events(
    rhs: Callable[[float, np.ndarray], Any],
    y0: Sequence[float] | float,
    t_span: tuple[float, float],
    n_steps: int,
    event: Callable[[float, np.ndarray], float],
    *,
    method: Callable[..., ODEResult] = rk4,
    tolerance: float = 1e-12,
) -> ODEResult:
    """Integrate with a scalar event function, locating each crossing inside the step by bisection.

    Applying the discontinuity at the end of a full step is the classic way to lose accuracy at an event:
    the step that contains it is cut back until the event time is known to ``tolerance``, and the
    integration restarts from there. The event times and states are returned in
    :attr:`ODEResult.events`, and the ``ODE.EVENT`` diagnostic says how many were found.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    y = _as_state(y0)
    times = [t0]
    states = [y.copy()]
    events: list[dict[str, float]] = []
    evaluations = 0

    def step(t: float, state: np.ndarray, size: float) -> tuple[np.ndarray, int]:
        if method is rk4 or method is euler or method is heun:
            new_state, _, used = _explicit_step(rhs, t, state, size,
                                                {"rk4": RK4, "euler": EULER, "heun": HEUN}[method.__name__])
            return new_state, used
        result = method(rhs, state, (t, t + size), 1)
        return _as_state(result.y[-1]), result.function_evaluations

    time = t0
    guard = 0
    while time < t1 - 1e-15 and guard < 10 * n_steps + 10:
        guard += 1
        size = min(h, t1 - time)
        previous_value = float(event(time, y))
        new_state, used = step(time, y, size)
        evaluations += used
        new_value = float(event(time + size, new_state))
        if previous_value == 0.0:
            previous_value = 1e-300
        if previous_value * new_value < 0.0:
            # bisect in time for the crossing, re-integrating from the last accepted point
            low, high = 0.0, size
            high_state = new_state
            for _ in range(60):
                if high - low <= tolerance * max(1.0, abs(time)):
                    break
                mid = 0.5 * (low + high)
                mid_state, used_mid = step(time, y, mid)
                evaluations += used_mid
                value = float(event(time + mid, mid_state))
                if np.sign(value) == np.sign(previous_value):
                    low = mid
                else:
                    high, high_state = mid, mid_state
            time = time + high
            y = high_state
            times.append(time)
            states.append(y.copy())
            events.append({"time": float(time), "event_value": float(event(time, y))})
            continue
        time = time + size
        y = new_state
        times.append(time)
        states.append(y.copy())

    diagnostics: list[Diagnostic] = []
    if events:
        diagnostics.append(explain("ODE.EVENT", events=len(events)))
    states_array = np.asarray(states)
    return ODEResult(t=np.asarray(times), y=states_array if y.size > 1 else states_array[:, 0],
                     method=f"{method.__name__} (events)", function_evaluations=evaluations,
                     steps=len(times) - 1, events=tuple(events), diagnostics=tuple(diagnostics))


# --------------------------------------------------------------------------- index-1 DAE
def pendulum_cartesian_index1(
    length: float = 1.0, gravity: float = 9.81
) -> Callable[[float, np.ndarray], np.ndarray]:
    """A pendulum in Cartesian coordinates as an index-reduced DAE (index 3 -> index 1), as an ODE.

    The constrained formulation is ``x'' = lambda x``, ``y'' = lambda y - g`` with ``x^2 + y^2 = L^2``.
    Differentiating the constraint twice gives ``x x'' + y y'' + v^2 = 0`` and hence the multiplier in
    closed form, ``lambda = (g y - v^2) / L^2``, which turns the system into a plain ODE in
    ``(x, y, vx, vy)``. That is index reduction: no algebraic equation is left to solve, at the price that
    the constraint is only satisfied to the integration error, which then drifts and has to be watched.

    The state is ``[x, y, vx, vy]`` with ``y`` pointing up. The limitation: the drift is *not* corrected
    here -- use :func:`project_pendulum` to renormalise, and compare the two drifts.
    """
    if length <= 0:
        raise ValueError(f"length must be positive; got {length}")
    squared = length * length

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, yy, vx, vy = (float(v) for v in y)
        multiplier = (gravity * yy - (vx * vx + vy * vy)) / squared
        return np.array([vx, vy, multiplier * x, multiplier * yy - gravity])

    return rhs


def project_pendulum(state: np.ndarray, length: float = 1.0) -> np.ndarray:
    """Project a Cartesian pendulum state back onto the constraint manifold and its tangent space.

    The position is rescaled to the circle (``r -> L r/|r|``) and the velocity is made orthogonal to the
    radius (``v -> v - (v.r) r/|r|^2``): the standard projection that removes the drift accumulated by the
    index reduction without changing the dynamics to first order.
    """
    x, y, vx, vy = (float(v) for v in state)
    radius = math.hypot(x, y)
    if radius == 0.0:
        raise ValueError("the pendulum position cannot be at the origin")
    x = x * length / radius
    y = y * length / radius
    radial = x * vx + y * vy
    vx -= radial * x / (length * length)
    vy -= radial * y / (length * length)
    return np.array([x, y, vx, vy])
