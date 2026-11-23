"""Roots of scalar equations and nonlinear systems, with the order of each method measured (#1057).

Five properties of an iterative root finder are reported here, and only the first is a matter of taste:

1. whether it converged, to what residual, and in how many evaluations;
2. the **order** it was measured at -- from its own iterates, not from the textbook;
3. for a linearly convergent method, the **contraction factor** it actually achieved;
4. which failure mode it hit, when it failed: a cycle, a flat derivative, a lost order at a multiple
   root, a stalled bracket, a singular Jacobian at a fold;
5. what fixed it, when something did.

The classic orders, which the tests re-derive rather than assume: bisection 1 (factor 1/2), regula falsi 1
(factor data-dependent, and it stalls), secant 1.618, Newton 2, Halley 3, Brent superlinear (~1.6 in
practice, since it switches between bisection and inverse quadratic interpolation).

Why the order is measured and not trusted
-----------------------------------------
Newton has order 2 only for a *simple* root. At a root of multiplicity ``m`` it degrades to linear with
factor ``1 - 1/m``, because ``f'`` vanishes there too; the fix is Newton on ``f/f'``, which restores
order 2. A method whose theory says 2 and which measures 1 here is not broken -- it is telling you the
root is multiple, which is a statement about the problem.

The limitation, stated once: every order measurement here is asymptotic in the number of iterations and
is computed from the tail of the error sequence. When the tail reaches the rounding floor
(``~eps * |root|``) the estimates are noise, so iterates at that level are excluded, which means a method
converging in three steps has too few clean iterates to measure its order at all -- NaN, not a guess.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .convergence import OrderEstimate, order_from_iterates
from .diagnostics import Diagnostic, explain

__all__ = [
    "ContinuationResult",
    "PolynomialRoots",
    "RootResult",
    "SystemRootResult",
    "bisection",
    "brent",
    "broyden",
    "cstr_residual",
    "fixed_point",
    "halley",
    "illinois",
    "jfnk",
    "natural_parameter_continuation",
    "newton",
    "newton_system",
    "numerical_jacobian",
    "poly_roots_companion",
    "pseudo_arclength_continuation",
    "regula_falsi",
    "scan_for_bracket",
    "secant",
    "wilkinson_sensitivity",
]

#: errors below this fraction of the root magnitude are at the rounding floor and excluded from order fits
_PLATEAU = 1e-12


@dataclass(frozen=True)
class RootResult:
    """What a scalar root finder did: the root, its history, the measured order, and why it stopped."""

    root: float
    method: str
    converged: bool
    iterations: int
    function_evaluations: int
    residuals: tuple[float, ...]
    iterates: tuple[float, ...]
    order: OrderEstimate | None = None
    linear_rate: float | None = None
    bracket: tuple[float, float] | None = None
    bracket_widths: tuple[float, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def residual(self) -> float:
        """The final ``|f(x)|``."""
        return self.residuals[-1] if self.residuals else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "root": self.root,
            "method": self.method,
            "converged": self.converged,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "residual": self.residual,
            "order": self.order.to_dict() if self.order else None,
            "linear_rate": self.linear_rate,
            "bracket": list(self.bracket) if self.bracket else None,
            "bracket_widths": [float(w) for w in self.bracket_widths],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass(frozen=True)
class SystemRootResult:
    """What a nonlinear-system solver did, with the order measured from the residual norms."""

    x: np.ndarray
    method: str
    converged: bool
    iterations: int
    function_evaluations: int
    residual_norms: tuple[float, ...]
    order: OrderEstimate | None = None
    linear_rate: float | None = None
    step_norms: tuple[float, ...] = ()
    iterates: tuple[np.ndarray, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def residual(self) -> float:
        """The final ``||F(x)||_2``."""
        return float(self.residual_norms[-1]) if self.residual_norms else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "x": [float(v) for v in np.ravel(self.x)],
            "method": self.method,
            "converged": self.converged,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "residual": self.residual,
            "step_norms": [float(v) for v in self.step_norms],
            "order": self.order.to_dict() if self.order else None,
            "linear_rate": self.linear_rate,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass(frozen=True)
class ContinuationResult:
    """A traced solution branch ``F(x, lambda) = 0``, with the folds it passed and the ones it hit."""

    path: np.ndarray  # shape (n_steps + 1, n + 1); last column is lambda
    method: str
    converged: bool
    folds: tuple[int, ...]
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def parameters(self) -> np.ndarray:
        """The lambda values along the path (not monotone for pseudo-arclength past a fold)."""
        return self.path[:, -1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "path": [[float(v) for v in row] for row in self.path],
            "method": self.method,
            "converged": self.converged,
            "folds": list(self.folds),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass(frozen=True)
class PolynomialRoots:
    """Companion-matrix roots of a polynomial, plus the conditioning of that root problem."""

    roots: np.ndarray
    coefficients: np.ndarray
    sensitive_index: int | None = None
    condition_numbers: np.ndarray | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "roots": [[float(r.real), float(r.imag)] for r in np.ravel(self.roots)],
            "coefficients": [float(c) for c in self.coefficients],
            "condition_numbers": [float(c) for c in np.ravel(self.condition_numbers)]
            if self.condition_numbers is not None else None,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# --------------------------------------------------------------------------- order from iterates
def _measure_order(
    iterates: Sequence[float] | np.ndarray, reference: float, method: str
) -> tuple[OrderEstimate | None, float | None]:
    """Measure the order of a scalar iteration, or its contraction factor when it is linear.

    Two regimes, decided from the data rather than declared by the caller. A superlinearly convergent
    method drives the ratio ``e_{k+1}/e_k`` to zero, so the triple-based order formula is stable; a
    linearly convergent one has a constant ratio, and the triple formula divides two nearly equal
    quantities and returns noise -- the ratio itself is then the measurement.

    Errors at the rounding floor (below ``1e-12 * max(1, |root|)``) are removed first: they are the
    plateau where the iteration cannot improve, and including them makes every method look linear.
    """
    x = np.asarray(iterates, dtype=np.float64).ravel()
    floor = _PLATEAU * max(1.0, abs(float(reference)))
    e = np.abs(x - float(reference))
    e = e[(e > floor) & np.isfinite(e)]
    if e.size < 3:
        return None, None

    ratios = e[1:] / e[:-1]
    tail = ratios[-3:]
    linear = bool(np.all(tail > 0.01) and np.std(tail) <= 0.25 * max(float(np.mean(tail)), 1e-30))

    if linear:
        rate = float(np.exp(np.mean(np.log(tail))))
        estimate = OrderEstimate(
            order=1.0, stderr=float(np.std(tail, ddof=1)) if tail.size > 1 else float("nan"),
            ci95=(float(np.min(tail)), float(np.max(tail))), r_squared=float("nan"),
            n_points=int(tail.size), name=f"{method} (linear)",
        )
        return estimate, rate

    if e.size < 3:
        return None, None
    try:
        return order_from_iterates(x[: e.size], float(reference), tail=3), None
    except ValueError:
        return None, None


def _result(
    method: str,
    root: float,
    iterates: list[float],
    residuals: list[float],
    *,
    converged: bool,
    evaluations: int,
    diagnostics: list[Diagnostic],
    bracket: tuple[float, float] | None = None,
    bracket_widths: tuple[float, ...] = (),
) -> RootResult:
    order, rate = _measure_order(iterates, root, method)
    return RootResult(
        root=float(root),
        method=method,
        converged=converged,
        iterations=max(len(iterates) - 1, 0),
        function_evaluations=int(evaluations),
        residuals=tuple(float(r) for r in residuals),
        iterates=tuple(float(v) for v in iterates),
        order=order,
        linear_rate=rate,
        bracket=bracket,
        bracket_widths=bracket_widths,
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- scalar methods
def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-12,
    max_iter: int = 200,
) -> RootResult:
    """Bisection: order 1 with factor exactly 1/2, and the only method here that cannot fail on a bracket.

    The interval halves every step, so after ``k`` steps the root is known to ``(b-a)/2**k`` -- a
    *guaranteed* bound, not an estimate. That is why the contraction factor is 0.5 to machine precision
    and why the order is 1 no matter how smooth ``f`` is.

    Raises ``ValueError`` if ``f(a)`` and ``f(b)`` have the same sign (no guaranteed root inside; see
    :func:`scan_for_bracket`).
    """
    fa, fb = float(f(a)), float(f(b))
    evaluations = 2
    if fa == 0.0:
        return _result("bisection", a, [a], [0.0], converged=True, evaluations=1, diagnostics=[])
    if fb == 0.0:
        return _result("bisection", b, [b], [0.0], converged=True, evaluations=1, diagnostics=[])
    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) have the same sign ({fa:.3e}, {fb:.3e}): the interval does not bracket a root"
        )

    lo, hi = float(a), float(b)
    mid = 0.5 * (lo + hi)
    iterates = [mid]
    residuals = [abs(float(f(mid)))]
    widths = [hi - lo]
    evaluations += 1
    diagnostics: list[Diagnostic] = []

    for _ in range(max_iter):
        fmid = float(f(mid))
        evaluations += 1
        if fmid == 0.0 or 0.5 * (hi - lo) <= xtol * max(1.0, abs(mid)):
            return _result("bisection", mid, iterates, residuals, converged=True,
                           evaluations=evaluations, diagnostics=diagnostics, bracket=(lo, hi),
                           bracket_widths=tuple(widths))
        if np.sign(fmid) == np.sign(fa):
            lo, fa = mid, fmid
        else:
            hi = mid
        mid = 0.5 * (lo + hi)
        iterates.append(mid)
        residuals.append(abs(float(f(mid))))
        widths.append(hi - lo)

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="bisection", iterations=max_iter,
                               bracket_width=hi - lo))
    return _result("bisection", mid, iterates, residuals, converged=False, evaluations=evaluations,
                   diagnostics=diagnostics, bracket=(lo, hi), bracket_widths=tuple(widths))


def scan_for_bracket(
    f: Callable[[float], float], a: float, b: float, *, samples: int = 100
) -> tuple[float, float] | None:
    """Find a sign change of ``f`` in ``[a, b]`` by scanning -- the step every global method needs first.

    Returns the first bracketing pair, or ``None``. This is a scan, not a root finder: it can miss an
    even number of roots inside one subinterval, which is stated here because it is the usual reason a
    bracket "cannot be found" for a function that visibly has roots.
    """
    xs = np.linspace(float(a), float(b), int(samples) + 1)
    prev_x, prev_f = float(xs[0]), float(f(float(xs[0])))
    for x in xs[1:]:
        fx = float(f(float(x)))
        if fx == 0.0:
            return (float(x), float(x))
        if prev_f == 0.0:
            return (prev_x, prev_x)
        if np.sign(fx) != np.sign(prev_f):
            return (prev_x, float(x))
        prev_x, prev_f = float(x), fx
    return None


def regula_falsi(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    variant: str = "plain",
    xtol: float = 1e-12,
    max_iter: int = 200,
) -> RootResult:
    """False position (secant-on-a-bracket). ``variant="illinois"`` halves the stale endpoint's value.

    The plain variant keeps one endpoint fixed forever, so it converges (linearly, order 1) but the
    *bracket* stops shrinking: it can look converged while the guaranteed accuracy is stuck. That is the
    ``ROOTS.STALLED`` diagnostic. The Illinois modification halves the function value at whichever
    endpoint has stayed put twice, which restores geometric shrinking of both ends.

    Raises ``ValueError`` for a non-bracketing interval.
    """
    fa, fb = float(f(a)), float(f(b))
    evaluations = 2
    if fa * fb > 0:
        raise ValueError(f"f(a) and f(b) have the same sign ({fa:.3e}, {fb:.3e}): no bracket")
    lo, hi = float(a), float(b)
    side = 0  # 0: no repetition, 1: hi stayed, -1: lo stayed
    iterates: list[float] = []
    residuals: list[float] = []
    widths = [hi - lo]
    diagnostics: list[Diagnostic] = []
    x = lo
    for _ in range(max_iter):
        x = (lo * fb - hi * fa) / (fb - fa)
        fx = float(f(x))
        evaluations += 1
        iterates.append(x)
        residuals.append(abs(fx))
        widths.append(hi - lo)
        if fx == 0.0 or abs(hi - lo) <= xtol * max(1.0, abs(x)):
            return _result("regula falsi" if variant == "plain" else f"regula falsi ({variant})", x,
                           iterates, residuals, converged=True, evaluations=evaluations,
                           diagnostics=diagnostics, bracket=(lo, hi), bracket_widths=tuple(widths))
        if np.sign(fx) == np.sign(fb):
            hi, fb = x, fx
            if variant == "illinois":
                if side == 1:
                    fa *= 0.5
                side = 1
        else:
            lo, fa = x, fx
            if variant == "illinois":
                if side == -1:
                    fb *= 0.5
                side = -1

    diagnostics.append(explain("ROOTS.STALLED", method=variant, bracket_width=hi - lo,
                               iterations=max_iter))
    return _result("regula falsi" if variant == "plain" else f"regula falsi ({variant})", x,
                   iterates, residuals, converged=False, evaluations=evaluations,
                   diagnostics=diagnostics, bracket=(lo, hi), bracket_widths=tuple(widths))


def illinois(f: Callable[[float], float], a: float, b: float, **kwargs: Any) -> RootResult:
    """Regula falsi with the Illinois modification -- the version that does not stall."""
    return regula_falsi(f, a, b, variant="illinois", **kwargs)


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    xtol: float = 1e-13,
    max_iter: int = 200,
) -> RootResult:
    """Secant method: order ``(1+sqrt(5))/2 = 1.618``, with no derivative and no bracket.

    It replaces ``f'`` by the slope through the last two iterates. The order follows from
    ``e_{k+1} ~ e_k e_{k-1}``, whose characteristic exponent is the golden ratio -- measurably lower than
    Newton's 2, which is exactly why the tests here compare the two on the same function.
    """
    x_prev, x = float(x0), float(x1)
    f_prev, fx = float(f(x_prev)), float(f(x))
    evaluations = 2
    iterates = [x_prev, x]
    residuals = [abs(f_prev), abs(fx)]
    diagnostics: list[Diagnostic] = []
    if fx == 0.0:
        return _result("secant", x, iterates, residuals, converged=True, evaluations=evaluations,
                       diagnostics=[])

    for _ in range(max_iter):
        denom = fx - f_prev
        if denom == 0.0:
            diagnostics.append(explain("ROOTS.FLAT_DERIVATIVE", method="secant", x=x,
                                       f_prev=f_prev, f_x=fx))
            return _result("secant", x, iterates, residuals, converged=False, evaluations=evaluations,
                           diagnostics=diagnostics)
        x_new = x - fx * (x - x_prev) / denom
        f_new = float(f(x_new))
        evaluations += 1
        iterates.append(x_new)
        residuals.append(abs(f_new))
        if abs(x_new - x) <= xtol * max(1.0, abs(x_new)) or f_new == 0.0:
            return _result("secant", x_new, iterates, residuals, converged=True,
                           evaluations=evaluations, diagnostics=diagnostics)
        x_prev, f_prev, x, fx = x, fx, x_new, f_new

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="secant", iterations=max_iter))
    return _result("secant", x, iterates, residuals, converged=False, evaluations=evaluations,
                   diagnostics=diagnostics)


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    *,
    xtol: float = 1e-13,
    max_iter: int = 100,
    multiplicity: int = 1,
) -> RootResult:
    """Newton's method: order 2 for a simple root, order 1 with factor ``1 - 1/m`` at a multiple root.

    Two failure modes are detected and returned, not raised:

    * a **cycle** -- the iterates repeat exactly, which happens for cubics started in a saddle's basin;
    * a **flat derivative** -- ``f'(x) = 0`` or so small that the step is not a step.

    ``multiplicity=m`` multiplies the step by ``m`` (``x - m f/f'``), which is the cheap fix at a known
    multiple root: it restores order 2 because it cancels the ``m``-fold vanishing of ``f'``. The
    alternative is Newton on ``f/f'``, which needs ``f''``.
    """
    x = float(x0)
    fx = float(f(x))
    dfx = float(df(x))
    evaluations = 2
    iterates = [x]
    residuals = [abs(fx)]
    diagnostics: list[Diagnostic] = []
    seen: dict[float, int] = {x: 0}

    for k in range(max_iter):
        if dfx == 0.0:
            diagnostics.append(explain("ROOTS.FLAT_DERIVATIVE", method="newton", x=x, f_x=fx))
            return _result("newton", x, iterates, residuals, converged=False, evaluations=evaluations,
                           diagnostics=diagnostics)
        x_new = x - max(int(multiplicity), 1) * fx / dfx
        f_new = float(f(x_new))
        dfx = float(df(x_new))
        evaluations += 2
        iterates.append(x_new)
        residuals.append(abs(f_new))
        if f_new == 0.0 or abs(x_new - x) <= xtol * max(1.0, abs(x_new)):
            return _result("newton", x_new, iterates, residuals, converged=True,
                           evaluations=evaluations, diagnostics=diagnostics)
        if x_new in seen:
            diagnostics.append(explain("ROOTS.NEWTON_CYCLE", method="newton", cycle_start=seen[x_new],
                                       period=k + 1 - seen[x_new], x=x_new))
            return _result("newton", x_new, iterates, residuals, converged=False,
                           evaluations=evaluations, diagnostics=diagnostics)
        seen[x_new] = k + 1
        x, fx = x_new, f_new

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="newton", iterations=max_iter,
                               residual=abs(fx)))
    return _result("newton", x, iterates, residuals, converged=False, evaluations=evaluations,
                   diagnostics=diagnostics)


def halley(
    f: Callable[[float], float],
    df: Callable[[float], float],
    d2f: Callable[[float], float],
    x0: float,
    *,
    xtol: float = 1e-13,
    max_iter: int = 100,
) -> RootResult:
    """Halley's method: order 3, from including the second derivative in the correction.

    ``x_{k+1} = x - 2 f f' / (2 f'^2 - f f'')``. The denominator is the derivative of ``f/f'``, which is
    also why applying Newton to ``f/f'`` (the multiple-root fix) is the same idea one order lower.
    """
    x = float(x0)
    iterates = [x]
    residuals = [abs(float(f(x)))]
    evaluations = 3
    diagnostics: list[Diagnostic] = []

    for _ in range(max_iter):
        fx = float(f(x))
        dfx = float(df(x))
        d2fx = float(d2f(x))
        denom = 2.0 * dfx * dfx - fx * d2fx
        if denom == 0.0:
            diagnostics.append(explain("ROOTS.FLAT_DERIVATIVE", method="halley", x=x, f_x=fx))
            return _result("halley", x, iterates, residuals, converged=False, evaluations=evaluations,
                           diagnostics=diagnostics)
        x_new = x - 2.0 * fx * dfx / denom
        f_new = float(f(x_new))
        evaluations += 3
        iterates.append(x_new)
        residuals.append(abs(f_new))
        if f_new == 0.0 or abs(x_new - x) <= xtol * max(1.0, abs(x_new)):
            return _result("halley", x_new, iterates, residuals, converged=True,
                           evaluations=evaluations, diagnostics=diagnostics)
        x = x_new

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="halley", iterations=max_iter))
    return _result("halley", x, iterates, residuals, converged=False, evaluations=evaluations,
                   diagnostics=diagnostics)


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-13,
    max_iter: int = 200,
) -> RootResult:
    """Brent's method: bisection's guarantee with inverse quadratic interpolation's speed.

    It keeps a bracket (so it never diverges), and each step tries inverse quadratic interpolation or a
    secant step, falling back to bisection whenever the trial point is not a sufficient improvement. The
    order is not a single number: it is ~1.84 at best and 1 whenever it falls back, so what is reported
    here is whatever the measured tail gives -- which is the honest answer for a hybrid.

    Raises ``ValueError`` for a non-bracketing interval.
    """
    a, b = float(a), float(b)
    fa, fb = float(f(a)), float(f(b))
    evaluations = 2
    if fa * fb > 0:
        raise ValueError(f"f(a) and f(b) have the same sign ({fa:.3e}, {fb:.3e}): no bracket")
    if abs(fa) < abs(fb):  # invariant: |f(b)| <= |f(a)|
        a, b, fa, fb = b, a, fb, fa
    c, fc = a, fa
    d = float(b) - float(a)
    mflag = True
    iterates = [b]
    residuals = [abs(fb)]
    diagnostics: list[Diagnostic] = []

    for _ in range(max_iter):
        if fb == 0.0 or abs(b - a) <= xtol * max(1.0, abs(b)):
            return _result("brent", b, iterates, residuals, converged=True,
                           evaluations=evaluations, diagnostics=diagnostics, bracket=(a, b))
        if fa != fc and fb != fc:
            # inverse quadratic interpolation through (a, fa), (b, fb), (c, fc)
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)  # secant step

        between = (min(b, (3.0 * a + b) / 4.0) < s < max(b, (3.0 * a + b) / 4.0))
        reject = (
            not between
            or (mflag and abs(s - b) >= abs(b - c) / 2.0)
            or (not mflag and abs(s - b) >= abs(c - d) / 2.0)
            or (mflag and abs(b - c) < xtol)
            or (not mflag and abs(c - d) < xtol)
        )
        if reject:
            s = 0.5 * (a + b)  # bisection fallback: the guarantee of convergence
            mflag = True
        else:
            mflag = False

        fs = float(f(s))
        evaluations += 1
        iterates.append(float(s))
        residuals.append(abs(fs))
        d, c, fc = c, b, fb
        if np.sign(fs) == np.sign(fa):
            a, fa = s, fs
        else:
            b, fb = s, fs
        if abs(fa) < abs(fb):  # restore |f(b)| <= |f(a)|
            a, b, fa, fb = b, a, fb, fa

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="brent", iterations=max_iter))
    return _result("brent", b, iterates, residuals, converged=False, evaluations=evaluations,
                   diagnostics=diagnostics, bracket=(a, b))


def fixed_point(
    g: Callable[[float], float],
    x0: float,
    *,
    xtol: float = 1e-13,
    max_iter: int = 1000,
) -> RootResult:
    """Fixed-point iteration ``x_{k+1} = g(x_k)``: order 1 with factor ``|g'(x*)|``.

    Converges if and only if ``|g'| < 1`` near the fixed point, and the measured contraction factor is
    that derivative -- which is why reformulating a fixed point so that ``|g'|`` is small (adding and
    subtracting a scaled residual) is the whole art of the method. Diverges quietly otherwise, so the
    iteration limit is part of the answer, not an error.
    """
    x = float(x0)
    iterates = [x]
    residuals = [abs(float(g(x)) - x)]  # the fixed-point residual |g(x) - x|, not the step
    diagnostics: list[Diagnostic] = []
    for _ in range(max_iter):
        x_new = float(g(x))
        iterates.append(x_new)
        residuals.append(abs(x_new - x))
        if abs(x_new - x) <= xtol * max(1.0, abs(x_new)):
            return _result("fixed point", x_new, iterates, residuals, converged=True,
                           evaluations=len(iterates), diagnostics=diagnostics)
        x = x_new
    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="fixed point", iterations=max_iter,
                               last_change=residuals[-1]))
    return _result("fixed point", x, iterates, residuals, converged=False,
                   evaluations=len(iterates), diagnostics=diagnostics)


# --------------------------------------------------------------------------- systems
def numerical_jacobian(
    F: Callable[[np.ndarray], np.ndarray], x: Sequence[float], *, step: float = 1e-7
) -> np.ndarray:
    """Forward-difference Jacobian of ``F`` at ``x``, one column per unknown.

    ``J[:, j] ~ (F(x + h e_j) - F(x)) / h`` with ``h`` relative to ``|x_j|``. The default step balances
    truncation (``~h``) against round-off (``~eps/h``) at the classical optimum ``sqrt(eps) ~ 1.5e-8``
    for a forward difference; the V-curve of that trade-off is measured in
    :mod:`pinneapple_analysis.numerics.quadrature`.
    """
    x0 = np.asarray(x, dtype=np.float64).ravel()
    f0 = np.asarray(F(x0), dtype=np.float64).ravel()
    J = np.zeros((f0.size, x0.size))
    for j in range(x0.size):
        h = step * max(1.0, abs(x0[j]))
        xp = x0.copy()
        xp[j] += h
        J[:, j] = (np.asarray(F(xp), dtype=np.float64).ravel() - f0) / h
    return J


def newton_system(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], np.ndarray] | None,
    x0: Sequence[float],
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
    line_search: bool = True,
) -> SystemRootResult:
    """Newton's method for ``F(x) = 0``, with an optional backtracking line search.

    The order is 2 for a simple root as long as the Jacobian is well conditioned, and it is measured on
    the **step norms** ``||x_{k+1} - x_k||`` -- the practical error estimate, because the residual norm
    only tracks the error once the asymptotic regime is reached (a far start reads an order of ~1.6). A
    **singular Jacobian** stops the iteration with the
    ``ROOTS.JACOBIAN_SINGULAR`` diagnosis -- either the model degenerates here, or the iteration reached
    a fold of the solution branch, where the fix is continuation, not a better linear solver.

    The line search halves the step until ``||F||`` decreases by at least ``1e-4`` of the linear
    prediction (Armijo). Without it Newton can overshoot into a region where ``F`` is meaningless, and
    with it the measured order drops to 1 on the steps where it engages -- which the order estimate will
    show, and which is the price of robustness.
    """
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    Fx = np.asarray(F(x), dtype=np.float64).ravel()
    evaluations = 1
    norms = [float(np.linalg.norm(Fx))]
    diagnostics: list[Diagnostic] = []
    rate: float | None = None

    steps: list[float] = []
    iterates: list[np.ndarray] = [x.copy()]

    def finish(*args: Any, **kwargs: Any) -> SystemRootResult:
        return _system_result(*args, steps=steps, iterates=iterates, **kwargs)

    for _ in range(max_iter):
        if norms[-1] <= tol:
            return finish("newton", x, norms, True, evaluations, diagnostics, rate)
        if J is None:
            jac = numerical_jacobian(F, x)
            evaluations += x.size  # the finite-difference Jacobian costs n residual evaluations
        else:
            jac = np.asarray(J(x), dtype=np.float64)
        try:
            step = np.linalg.solve(jac, -Fx)
        except np.linalg.LinAlgError:
            diagnostics.append(explain("ROOTS.JACOBIAN_SINGULAR", method="newton", x=list(x),
                                       residual=norms[-1]))
            return finish("newton", x, norms, False, evaluations, diagnostics, None)
        if not np.all(np.isfinite(step)):
            diagnostics.append(explain("ROOTS.JACOBIAN_SINGULAR", method="newton", x=list(x),
                                       residual=norms[-1]))
            return finish("newton", x, norms, False, evaluations, diagnostics, None)

        alpha = 1.0
        if line_search:
            for _trial in range(30):
                trial = x + alpha * step
                f_trial = np.asarray(F(trial), dtype=np.float64).ravel()
                evaluations += 1
                if float(np.linalg.norm(f_trial)) <= (1.0 - 1e-4 * alpha) * norms[-1]:
                    break
                alpha *= 0.5
            else:
                alpha = 1.0
        x = x + alpha * step
        steps.append(float(np.linalg.norm(alpha * step)))
        iterates.append(x.copy())
        Fx = np.asarray(F(x), dtype=np.float64).ravel()
        evaluations += 1
        norms.append(float(np.linalg.norm(Fx)))
        if norms[-1] == 0.0:
            return finish("newton", x, norms, True, evaluations, diagnostics, rate)
        if np.linalg.norm(alpha * step) <= tol * max(1.0, float(np.linalg.norm(x))):
            converged = norms[-1] <= max(tol, 1e-10)
            return finish("newton", x, norms, converged, evaluations, diagnostics, rate)

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="newton", iterations=max_iter,
                               residual=norms[-1]))
    return finish("newton", x, norms, False, evaluations, diagnostics, rate)


def _system_result(
    method: str,
    x: np.ndarray,
    norms: list[float],
    converged: bool,
    evaluations: int,
    diagnostics: list[Diagnostic],
    rate: float | None,
    steps: list[float] | None = None,
    iterates: list[np.ndarray] | None = None,
) -> SystemRootResult:
    """Join the history into an :class:`OrderEstimate`.

    The order is measured on the **step norms** ``||x_{k+1} - x_k||`` when they were recorded, because for
    a converging iteration the step is the practical estimate of the error (``||Delta x_k|| ~ e_k``), while
    the residual norm only tracks the error once the asymptotic regime is reached: from a far start it
    reports an order lower than the method's. With no steps, the residual norms are used instead, which
    is honest but weaker -- and fewer than three usable values give ``None``, not a guess.
    """
    history: list[float] = list(steps) if steps else list(norms)
    order, linear_rate = _measure_order(history, 0.0, method) if len(history) >= 3 else (None, None)
    if rate is not None:
        linear_rate = rate
    return SystemRootResult(
        x=x,
        method=method,
        converged=converged,
        iterations=max(len(norms) - 1, 0),
        function_evaluations=evaluations,
        residual_norms=tuple(float(v) for v in norms),
        order=order,
        linear_rate=linear_rate,
        step_norms=tuple(float(v) for v in (steps or [])),
        iterates=tuple(np.asarray(v, dtype=np.float64).copy() for v in (iterates or [])),
        diagnostics=tuple(diagnostics),
    )


def broyden(
    F: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    *,
    B0: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> SystemRootResult:
    """Broyden's good method: a rank-one approximation of the Jacobian, refreshed at every step.

    ``B_{k+1} = B_k + (y - B_k s) s^T / (s^T s)`` with ``s = x_{k+1} - x_k`` and ``y = F_{k+1} - F_k``.
    One function evaluation per step instead of the ``n+1`` a finite-difference Jacobian costs, at the
    price of superlinear (not quadratic) convergence: the measured order lands near 1.6-2.0 depending on
    how well ``B_0`` is scaled, which is why the initial matrix matters.

    The ``B_0`` default is a scaled identity (``F'(x_0)`` if supplied): an unscaled identity is the
    usual reason Broyden "does not converge" on a badly scaled system.
    """
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    n = x.size
    Fx = np.asarray(F(x), dtype=np.float64).ravel()
    evaluations = 1
    if B0 is None:
        jac = numerical_jacobian(F, x)
        evaluations += n
        B = jac
    else:
        B = np.asarray(B0, dtype=np.float64).copy()
        if B.shape != (n, n):
            raise ValueError(f"B0 must be {n}x{n}; got {B.shape}")
    norms = [float(np.linalg.norm(Fx))]

    steps: list[float] = []
    iterates: list[np.ndarray] = [x.copy()]

    def finish(*args: Any, **kwargs: Any) -> SystemRootResult:
        return _system_result(*args, steps=steps, iterates=iterates, **kwargs)
    diagnostics: list[Diagnostic] = []

    for _ in range(max_iter):
        if norms[-1] <= tol:
            return finish("broyden", x, norms, True, evaluations, diagnostics, None)
        try:
            s = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            diagnostics.append(explain("ROOTS.JACOBIAN_SINGULAR", method="broyden", x=list(x)))
            return finish("broyden", x, norms, False, evaluations, diagnostics, None)
        x_new = x + s
        F_new = np.asarray(F(x_new), dtype=np.float64).ravel()
        evaluations += 1
        y = F_new - Fx
        denom = float(s @ s)
        if denom > 0.0:
            B = B + np.outer(y - B @ s, s) / denom
        x, Fx = x_new, F_new
        steps.append(float(np.linalg.norm(s)))
        iterates.append(x.copy())
        norms.append(float(np.linalg.norm(Fx)))
        if norms[-1] == 0.0:
            return finish("broyden", x, norms, True, evaluations, diagnostics, None)
        if float(np.linalg.norm(s)) <= tol * max(1.0, float(np.linalg.norm(x))):
            return finish("broyden", x, norms, norms[-1] <= max(tol, 1e-8), evaluations,
                                  diagnostics, None)

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="broyden", iterations=max_iter,
                               residual=norms[-1]))
    return finish("broyden", x, norms, False, evaluations, diagnostics, None)


def jfnk(
    F: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
    krylov_dim: int = 20,
    forcing: float = 1e-6,
) -> SystemRootResult:
    """Jacobian-free Newton-Krylov: the Newton step solved by GMRES, matrix-vector products by differences.

    ``J v ~ (F(x + eps v) - F(x)) / eps`` needs no Jacobian at all, which is the point: for a system whose
    Jacobian is expensive or never formed, this is the only affordable Newton. The inner solve is
    inexact (``forcing`` sets the relative residual it targets), and the outer iteration is still Newton,
    so the order is ~2 as long as the inner solve is tighter than the outer error -- the "Eisenstat-Walker"
    condition, which is why an over-tolerant inner solve shows up as a measured order between 1 and 2.
    """
    from .linalg import gmres

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    Fx = np.asarray(F(x), dtype=np.float64).ravel()
    evaluations = 1
    norms = [float(np.linalg.norm(Fx))]
    diagnostics: list[Diagnostic] = []

    steps: list[float] = []
    iterates: list[np.ndarray] = [x.copy()]

    def finish(*args: Any, **kwargs: Any) -> SystemRootResult:
        return _system_result(*args, steps=steps, iterates=iterates, **kwargs)
    eps = float(np.sqrt(np.finfo(float).eps))

    def matvec(v: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        vhat = np.asarray(v, dtype=np.float64).ravel()
        scale = float(np.linalg.norm(vhat))
        if scale == 0.0:
            return np.zeros_like(vhat)
        h = eps * max(1.0, float(np.linalg.norm(x))) / scale
        f_shift = np.asarray(F(x + h * vhat), dtype=np.float64).ravel()
        evaluations += 1
        return (f_shift - Fx) / h

    for _ in range(max_iter):
        if norms[-1] <= tol:
            return finish("jfnk", x, norms, True, evaluations, diagnostics, None)
        step, info = gmres(matvec, -Fx, tol=forcing * max(norms[-1], tol), restart=krylov_dim)
        if info > 0:
            diagnostics.append(explain("LINALG.NO_CONVERGENCE", method="jfnk (inner GMRES)",
                                       iterations=krylov_dim))
            return finish("jfnk", x, norms, False, evaluations, diagnostics, None)
        x = x + step
        steps.append(float(np.linalg.norm(step)))
        iterates.append(x.copy())
        Fx = np.asarray(F(x), dtype=np.float64).ravel()
        evaluations += 1
        norms.append(float(np.linalg.norm(Fx)))
        if norms[-1] == 0.0:
            return finish("jfnk", x, norms, True, evaluations, diagnostics, None)
        if float(np.linalg.norm(step)) <= tol * max(1.0, float(np.linalg.norm(x))):
            return finish("jfnk", x, norms, norms[-1] <= max(tol, 1e-9), evaluations,
                                  diagnostics, None)

    diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="jfnk", iterations=max_iter,
                               residual=norms[-1]))
    return finish("jfnk", x, norms, False, evaluations, diagnostics, None)


# --------------------------------------------------------------------------- continuation
def _augmented_jacobian(
    F: Callable[[np.ndarray], np.ndarray], x: np.ndarray, lam: float, tangent: np.ndarray
) -> np.ndarray:
    """``[[F_x, F_lam], [t_x, t_lam]]`` -- the Jacobian of the pseudo-arclength system."""
    n = x.size
    base = np.asarray(F(x, lam) if _takes_lambda(F) else F(x), dtype=np.float64).ravel()
    J = np.zeros((n, n + 1))
    for j in range(n):
        h = 1e-7 * max(1.0, abs(x[j]))
        xp = x.copy()
        xp[j] += h
        J[:, j] = (np.asarray(F(xp, lam) if _takes_lambda(F) else F(xp), dtype=np.float64).ravel() - base) / h
    h_lam = 1e-7 * max(1.0, abs(lam))
    J[:, n] = (np.asarray(F(x, lam + h_lam) if _takes_lambda(F) else F(x), dtype=np.float64).ravel()
               - base) / h_lam
    return np.vstack([J, tangent[None, :]])


def _takes_lambda(F: Callable[..., np.ndarray]) -> bool:
    """Whether the residual accepts ``(x, lambda)`` or only ``x`` (a probe, done once)."""
    try:
        import inspect

        params = inspect.signature(F).parameters
        return len(params) >= 2
    except (TypeError, ValueError):  # pragma: no cover - builtins and C callables
        return True


def _call(F: Callable[..., np.ndarray], x: np.ndarray, lam: float) -> np.ndarray:
    return np.asarray(F(x, lam) if _takes_lambda(F) else F(x), dtype=np.float64).ravel()


def _tangent(F: Callable[..., np.ndarray], x: np.ndarray, lam: float) -> np.ndarray:
    """Unit tangent of the solution branch, as the null vector of the ``n x (n+1)`` Jacobian.

    The null vector comes from the last right singular vector -- the one direction in which ``F_x`` and
    ``F_lam`` combine to zero. That is well defined even at a fold, where ``F_x`` alone is singular and a
    natural-parameter step would have no direction: this is exactly the difference between the two
    continuation methods here.
    """
    n = x.size
    J = np.zeros((n, n + 1))
    base = _call(F, x, lam)
    for j in range(n):
        h = 1e-7 * max(1.0, abs(x[j]))
        xp = x.copy()
        xp[j] += h
        J[:, j] = (_call(F, xp, lam) - base) / h
    h_lam = 1e-7 * max(1.0, abs(lam))
    J[:, n] = (_call(F, x, lam + h_lam) - base) / h_lam
    _, _, vt = np.linalg.svd(J)
    t = vt[-1, :]
    norm = float(np.linalg.norm(t))
    return t / norm if norm > 0 else t


def natural_parameter_continuation(
    F: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    parameters: Sequence[float],
    *,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> ContinuationResult:
    """Continue a branch by fixing ``lambda`` and solving for ``x`` -- the method that stops at a fold.

    Each step is a Newton solve of ``F(x, lambda) = 0`` at the prescribed ``lambda``, started from the
    previous point. This is the natural thing to do and it is correct while the branch is a function of
    ``lambda``. At a turning point the Jacobian ``F_x`` is singular, so the solve either fails or takes
    an enormous step: that is the ``ROOTS.FOLD`` diagnostic, and it is not a bug to fix but a statement
    that this branch is not a function of ``lambda`` there.

    Raises ``ValueError`` if ``parameters`` is not monotone (the method has no meaning otherwise) or
    empty.
    """
    lams = np.asarray(parameters, dtype=np.float64).ravel()
    if lams.size < 2:
        raise ValueError("continuation needs at least two parameter values")
    diffs = np.diff(lams)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("natural-parameter continuation requires a monotone sequence of parameters")

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    path = [np.concatenate([x, [lams[0]]])]
    diagnostics: list[Diagnostic] = []
    converged = True
    folds: list[int] = []

    for i, lam in enumerate(lams[1:], start=1):
        def residual_at(z: np.ndarray, lam: float = lam) -> np.ndarray:
            return _call(F, z, lam)

        got = newton_system(residual_at, None, x, tol=tol, max_iter=max_iter)
        if not got.converged or any(d.code == "ROOTS.JACOBIAN_SINGULAR" for d in got.diagnostics):
            diagnostics.append(
                explain("ROOTS.FOLD", method="natural parameter", index=i, parameter=float(lam),
                        residual=got.residual)
            )
            folds.append(i - 1)
            converged = False
            break
        x = got.x
        path.append(np.concatenate([x, [lam]]))

    return ContinuationResult(
        path=np.asarray(path), method="natural parameter", converged=converged,
        folds=tuple(folds), diagnostics=tuple(diagnostics),
    )


def pseudo_arclength_continuation(
    F: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    lam0: float,
    *,
    steps: int = 40,
    ds: float = 0.1,
    tol: float = 1e-10,
    max_iter: int = 50,
    direction: int = 1,
) -> ContinuationResult:
    """Pseudo-arclength continuation: treats ``lambda`` as an unknown and passes through folds.

    The unknowns are ``z = (x, lambda)`` in ``R^{n+1}`` and the system is ``F(x, lambda) = 0`` plus the
    arclength constraint ``(z - z_k) . t_k = ds``, where ``t_k`` is the unit tangent of the branch. The
    extra equation is what makes the Jacobian nonsingular at a turning point, so the branch can bend
    back: ``lambda`` is allowed to decrease. :attr:`ContinuationResult.parameters` is therefore not
    monotone, which is the signature of having gone through a fold.

    ``direction`` (+1 or -1) fixes which way the parameter moves on the first step; the tangent's sign
    is otherwise arbitrary and a caller that wants "increase lambda until it turns" must say so.

    Raises ``ValueError`` for a non-positive ``ds`` or ``steps``, or a ``direction`` that is not +-1.
    """
    if ds <= 0 or steps <= 0:
        raise ValueError("ds and steps must be positive")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1; got {direction}")
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    lam = float(lam0)
    n = x.size
    tangent = _tangent(F, x, lam)
    if direction * tangent[-1] < 0.0:
        tangent = -tangent  # orient the first step so that lambda moves the way the caller asked
    z = np.concatenate([x, [lam]])
    path = [z.copy()]
    diagnostics: list[Diagnostic] = []
    folds: list[int] = []
    converged = True
    previous_tangent = tangent

    for k in range(steps):
        predictor = z + ds * tangent
        z_new = predictor.copy()
        ok = False
        for _ in range(max_iter):
            xk, lk = z_new[:n], z_new[n]
            residual = np.concatenate([_call(F, xk, lk), [float(tangent @ (z_new - z) - ds)]])
            if np.linalg.norm(residual) <= tol:
                ok = True
                break
            jac = _augmented_jacobian(F, xk, lk, tangent)
            try:
                delta = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                diagnostics.append(explain("ROOTS.JACOBIAN_SINGULAR", method="pseudo-arclength",
                                           index=k, parameter=float(lk)))
                converged = False
                break
            z_new = z_new + delta
            if np.linalg.norm(delta) <= tol:
                ok = np.linalg.norm(residual) <= max(tol, 1e-8)
                break
        if not ok:
            converged = False
            diagnostics.append(explain("ROOTS.NOT_CONVERGED", method="pseudo-arclength", index=k))
            break

        tangent_new = _tangent(F, z_new[:n], z_new[n])
        if float(tangent_new @ previous_tangent) < 0.0:
            tangent_new = -tangent_new  # keep the orientation along the branch
        # a fold is where lambda stops advancing: the tangent's lambda component changes sign
        if np.sign(tangent_new[-1]) != np.sign(tangent[-1]):
            folds.append(len(path))
            diagnostics.append(explain("ROOTS.FOLD", method="pseudo-arclength", index=len(path),
                                       parameter=float(z_new[n])))
        previous_tangent, tangent = tangent, tangent_new
        z = z_new
        path.append(z.copy())

    return ContinuationResult(
        path=np.asarray(path), method="pseudo-arclength", converged=converged,
        folds=tuple(folds), diagnostics=tuple(diagnostics),
    )


def cstr_residual(
    x: np.ndarray | float,
    Da: float,
    *,
    theta: float = 18.0,
    gamma: float = 0.6,
) -> np.ndarray:
    """Dimensionless steady state of a cooled CSTR with an Arrhenius rate -- the classic multiplicity demo.

    ``f(x) = -x + Da (1 - x) exp(theta x / (gamma + x))`` for the dimensionless conversion ``x`` and
    Damkohler number ``Da``. The exponential is a sigmoid in the relevant range, so ``f`` has two
    extrema and the branch of steady states is an S with two folds: at a critical ``Da`` the reactor
    jumps between the ignited and extinguished branches. Verified in the tests by tracing the branch with
    pseudo-arclength continuation and counting the folds -- the numbers come from the continuation, not
    from this docstring.

    The limitation: one reaction, one species, no energy balance solved explicitly -- the thermal effect
    enters only through ``theta``. It is a demonstration of multiplicity, not a reactor model.
    """
    xv = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return -xv + Da * (1.0 - xv) * np.exp(theta * xv / (gamma + xv))


# --------------------------------------------------------------------------- polynomials
def poly_roots_companion(coefficients: Sequence[float], *, sensitivity: bool = True) -> PolynomialRoots:
    """All roots of ``p(x) = c_0 + c_1 x + ... + c_n x^n`` as eigenvalues of the companion matrix.

    Trailing zeros in ``coefficients`` are stripped (a degree drop, not a root at infinity). The
    companion matrix is the standard ``n x n`` one with ``-c_k / c_n`` in the last row; eigenvalues are
    then computed by the QR algorithm, which is backward stable for the *eigenvalue* problem -- but the
    map from coefficients to that matrix is not, which is the whole point of
    :func:`wilkinson_sensitivity`.

    ``sensitivity=True`` also returns the per-root coefficient condition number
    ``kappa_i = sum_k |a_k r_i^k| / (|r_i| |p'(r_i)|)``: the relative root error per unit relative
    coefficient error. Large values mean the coefficients do not determine the root.
    """
    c = np.asarray(coefficients, dtype=np.float64).ravel()
    if c.size < 2:
        raise ValueError("need at least two coefficients (degree >= 1)")
    nonzero = np.nonzero(c)[0]
    if nonzero.size == 0:
        raise ValueError("the zero polynomial has no roots")
    c = c[: nonzero[-1] + 1]
    n = c.size - 1
    if n == 0:
        raise ValueError("a constant polynomial has no roots")

    C = np.zeros((n, n))
    if n > 1:
        C[1:, :-1] = np.eye(n - 1)  # subdiagonal of ones
    C[:, -1] = -c[:-1] / c[-1]      # last column carries the coefficients
    roots = np.linalg.eigvals(C)

    diagnostics: list[Diagnostic] = []
    conds: np.ndarray | None = None
    if sensitivity:
        dp = np.polyder(c[::-1])  # np.polyder takes descending coefficients; so does np.polyval
        conds = np.zeros(roots.size)
        for i, r in enumerate(roots):
            powers = np.array([r**k for k in range(c.size)])
            numerator = float(np.sum(np.abs(c * powers)))
            derivative = complex(np.polyval(dp, r))
            denom = abs(r) * abs(derivative)
            conds[i] = numerator / denom if denom > 0 else np.inf
        worst = int(np.argmax(conds))
        if conds[worst] > 1e10:
            diagnostics.append(
                explain("ROOTS.POLYNOMIAL_ILL_CONDITIONED", max_condition_number=float(conds[worst]),
                        root=complex(roots[worst]).__repr__())
            )
    return PolynomialRoots(roots=roots, coefficients=c, condition_numbers=conds,
                           diagnostics=tuple(diagnostics))


def _wilkinson_coefficients(degree: int) -> np.ndarray:
    """Coefficients of ``prod_{i=1..degree} (x - i)`` computed in exact integers before the float cast."""
    coeffs = [1]
    for i in range(1, degree + 1):
        new = [0] * (len(coeffs) + 1)
        for k, c in enumerate(coeffs):
            new[k] += -i * c
            new[k + 1] += c
        coeffs = new
    return np.asarray(coeffs, dtype=np.float64)  # ascending order; exact up to 2**53, then rounded


def wilkinson_sensitivity(*, degree: int = 20, relative_perturbation: float = 2.0**-23) -> dict[str, Any]:
    """Wilkinson's polynomial: how far the roots move when one coefficient moves in its last bit.

    ``prod_{i=1..20}(x-i)`` has roots 1..20, and its coefficients are huge (the constant term is 20!).
    Two measurements come out of it: the *unperturbed* roots already differ from the integers, because
    20! does not fit in 53 bits, so the coefficients are rounded before any root finding starts; and
    perturbing the coefficient of ``x^{n-1}`` by a single float32 ulp (``2**-23``) moves some roots by
    *whole units*. That is the standard demonstration that computing roots from coefficients is
    ill-posed, and the reason ``numpy.roots`` cannot be blamed for the answer.

    Returns the unperturbed and perturbed roots, the measured displacement of each, and the predicted
    displacement ``kappa * relative_perturbation * |r|`` from the coefficient condition number -- so the
    sensitivity is checked against theory rather than merely shown.
    """
    base = _wilkinson_coefficients(degree)
    reference = poly_roots_companion(base)
    # index of the x^{n-1} coefficient in ascending order: n-1
    perturbed = base.copy()
    index = degree - 1
    delta = relative_perturbation * abs(perturbed[index])
    perturbed[index] += delta
    shifted = poly_roots_companion(perturbed)

    order_ref = np.argsort(np.real(reference.roots))
    order_new = np.argsort(np.real(shifted.roots))
    r0 = np.real(reference.roots)[order_ref]
    r1 = np.real(shifted.roots)[order_new]
    displacement = np.abs(r1 - r0)
    predicted = np.abs(reference.condition_numbers[order_ref]) * relative_perturbation * np.abs(r0)
    return {
        "degree": degree,
        "relative_perturbation": relative_perturbation,
        "roots": r0.tolist(),
        "perturbed_roots": r1.tolist(),
        "displacement": displacement.tolist(),
        "predicted_displacement": predicted.tolist(),
        "max_displacement": float(displacement.max()),
        "max_condition_number": float(np.max(reference.condition_numbers)),
        "diagnostics": [d.to_dict() for d in poly_roots_companion(perturbed).diagnostics],
    }
