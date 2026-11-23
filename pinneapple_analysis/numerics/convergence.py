"""Observed convergence orders and error curves, measured rather than assumed.

Every method in this package is described by the same two numbers: the order it converges at and the
error it reaches for a given effort. Both are *measured* here, from error values the caller already
has, so a test can assert that a scheme delivers the order its textbook formula promises -- and a user
can discover that it does not on their own problem.

The measurement
---------------
If the error of a discretisation decays like ``e(h) = C h^p`` then ``log e = log C + p log h``: ``p`` is
the slope of a straight line in log-log. :func:`fit_order` reports that slope with its standard error
and 95 % confidence interval, because a two-point estimate of an order is not an estimate at all -- it
is a coincidence that happens to have units of order.

    >>> import numpy as np
    >>> from pinneapple_analysis.numerics.convergence import fit_order
    >>> h = np.array([0.1, 0.05, 0.025, 0.0125])
    >>> fit_order(h, 3.0 * h**2).order
    2.0000000000000004

The limitation, stated once for the whole module: a log-log slope is an *asymptotic* statement. It is
only the order of the scheme while the mesh is fine enough that the leading truncation term dominates
rounding and the higher-order terms. :meth:`OrderEstimate.agrees_with` is what a test should use, with
a tolerance, and the reported ``r_squared`` says how straight the line actually was.

The V-curve
-----------
Truncation error falls with ``h`` and round-off error grows like ``eps/h``, so the total error has a
minimum at some ``h_opt``. :func:`truncation_roundoff_balance` measures that curve and locates the
minimum instead of predicting it from a constant somebody guessed: the predicted and measured optimum
are compared, and a large ratio is itself a result worth reporting.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .diagnostics import Diagnostic, explain

__all__ = [
    "ErrorCurve",
    "OrderEstimate",
    "VCurveResult",
    "fit_order",
    "order_from_iterates",
    "richardson_extrapolate",
    "truncation_roundoff_balance",
]

#: two-sided 95 % Student-t quantiles, by degrees of freedom (1..30; 1.96 beyond, which is the limit)
_T975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


@dataclass(frozen=True)
class OrderEstimate:
    """A measured convergence order with the uncertainty of the measurement.

    ``order`` is the exponent ``p`` in ``error ~ h**p``; ``ci95`` is the 95 % confidence interval of that
    slope from the log-log least-squares fit; ``r_squared`` says how straight the log-log line was (1.0
    is a perfect power law, which is what an asymptotic regime looks like).
    """

    order: float
    stderr: float
    ci95: tuple[float, float]
    r_squared: float
    n_points: int
    steps: tuple[float, ...] = ()
    name: str = "error"

    def agrees_with(self, expected: float, *, rtol: float = 0.10) -> bool:
        """Whether the measured order is within ``rtol`` (relative) of ``expected``.

        This is the assertion a convergence test should make. Use a loose ``rtol`` on coarse meshes: an
        observed order of 1.85 for a second-order scheme is the normal asymptotic-range behaviour, while
        0.95 for a second-order scheme is a bug or a boundary condition that is only first order.
        """
        if not np.isfinite(self.order):
            return False
        return abs(self.order - expected) <= rtol * max(abs(expected), 1e-12)

    def __str__(self) -> str:
        lo, hi = self.ci95
        if not (np.isfinite(lo) and np.isfinite(hi)):
            spread = "interval unavailable"
        elif np.isnan(self.r_squared):
            spread = f"range [{lo:.3f}, {hi:.3f}]"  # a spread of estimates, not a regression interval
        else:
            spread = f"95% CI [{lo:.3f}, {hi:.3f}]"
        r2 = "" if np.isnan(self.r_squared) else f", R^2 = {self.r_squared:.4f}"
        return f"p = {self.order:.4f} ({spread}{r2}, n = {self.n_points})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the numerical-lab page."""
        return {
            "order": self.order,
            "stderr": self.stderr,
            "ci95": list(self.ci95),
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "steps": list(self.steps),
            "name": self.name,
        }


@dataclass(frozen=True)
class ErrorCurve:
    """An error measured at a sequence of step sizes (or costs), plus the order read off it."""

    steps: np.ndarray
    errors: np.ndarray
    name: str = "error"

    def __post_init__(self) -> None:
        if self.steps.shape != self.errors.shape:
            raise ValueError(f"steps and errors must have the same shape; got {self.steps.shape} and "
                             f"{self.errors.shape}")
        if self.steps.size < 2:
            raise ValueError("an error curve needs at least two points")

    def order(self, *, tail: int | None = None) -> OrderEstimate:
        """Fit ``error ~ C * step**p`` over the whole curve or its last ``tail`` (finest) points.

        Fitting the whole curve is the honest default when the points are already in the asymptotic
        regime; ``tail=3`` is what a test uses when it wants the fine-mesh order specifically.
        """
        if tail is None:
            return fit_order(self.steps, self.errors, name=self.name)
        return fit_order(self.steps[-tail:], self.errors[-tail:], name=self.name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form: the raw curve plus its order -- what the lab page plots."""
        return {
            "name": self.name,
            "steps": [float(s) for s in self.steps],
            "errors": [float(e) for e in self.errors],
            "order": self.order().to_dict(),
        }


@dataclass(frozen=True)
class VCurveResult:
    """The truncation/round-off trade-off, measured.

    ``optimal_step`` is the step at which the measured error is smallest, ``truncation_order`` the
    measured order on the truncation-dominated branch (steps below the optimum), and
    ``roundoff_slope`` the measured exponent on the round-off-dominated branch (positive: the error
    *grows* as the step shrinks, typically like ``eps/h`` or ``eps/sqrt(h)``).
    """

    steps: np.ndarray
    errors: np.ndarray
    optimal_step: float
    optimal_error: float
    truncation_order: float
    roundoff_slope: float
    predicted_optimal_step: float | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def is_balanced(self) -> bool:
        """Whether the measured error actually has an interior minimum (both branches visible)."""
        return bool(np.isfinite(self.optimal_step) and 0 < np.argmin(self.errors) < self.errors.size - 1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the numerical-lab page."""
        return {
            "steps": [float(s) for s in self.steps],
            "errors": [float(e) for e in self.errors],
            "optimal_step": self.optimal_step,
            "optimal_error": self.optimal_error,
            "truncation_order": self.truncation_order,
            "roundoff_slope": self.roundoff_slope,
            "predicted_optimal_step": self.predicted_optimal_step,
            "is_balanced": self.is_balanced,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _clean(steps: Sequence[float] | np.ndarray, errors: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and drop points that cannot enter a log-log fit.

    A zero or negative error is not a small error -- it is a measurement that saturated at the exact
    value (or a caller passing something else). Dropping it silently would hide that, so the caller of
    :func:`fit_order` is told through the returned mask implicitly: this helper only drops non-finite or
    non-positive values, and :func:`fit_order` raises if fewer than two usable points remain.
    """
    h = np.asarray(steps, dtype=np.float64).ravel()
    e = np.abs(np.asarray(errors, dtype=np.float64).ravel())
    keep = np.isfinite(h) & np.isfinite(e) & (h > 0.0) & (e > 0.0)
    return h[keep], e[keep]


def fit_order(
    steps: Sequence[float] | np.ndarray,
    errors: Sequence[float] | np.ndarray,
    *,
    name: str = "error",
) -> OrderEstimate:
    """Measure the convergence order ``p`` in ``error = C * step**p`` by log-log least squares.

    Parameters
    ----------
    steps : the step sizes (or mesh spacings, or cell counts -- anything that shrinks under refinement;
        for a cost axis the exponent means something else, so do not call it an order).
    errors : the corresponding errors, positive and finite. At least two *distinct* steps are needed;
        with two points the slope is exact but has no computable uncertainty, and ``stderr`` is reported
        as NaN rather than zero -- a two-point fit is not evidence of a power law.

    Raises
    ------
    ValueError
        if fewer than two distinct, usable points remain. Returning an order from one point (or from a
        zero error) would be fabricating the number this function exists to measure.
    """
    h, e = _clean(steps, errors)
    if h.size < 2:
        raise ValueError(
            "need at least two points with positive, finite step and error to measure an order; got "
            f"{h.size} usable of {np.size(steps)} -- a zero error usually means the computation reached the "
            "exact value (or the exact value was used by mistake)."
        )
    if np.allclose(h, h[0]):
        raise ValueError("all step sizes are equal: the order is not identifiable from a single step size")

    x = np.log(h)
    y = np.log(e)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    residual = y - (intercept + slope * x)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    n = h.size
    if n > 2:
        stderr = float(np.sqrt(ss_res / (n - 2) / sxx))
        t = _T975.get(n - 2, 1.960)
        ci = (slope - t * stderr, slope + t * stderr)
    else:
        stderr = float("nan")
        ci = (float("nan"), float("nan"))

    return OrderEstimate(
        order=float(slope),
        stderr=stderr,
        ci95=(float(ci[0]), float(ci[1])),
        r_squared=float(r_squared),
        n_points=int(n),
        steps=tuple(float(v) for v in h),
        name=name,
    )


def order_from_iterates(
    iterates: Sequence[float] | np.ndarray,
    limit: float,
    *,
    tail: int = 3,
) -> OrderEstimate:
    """Measure the convergence order of an iteration from its iterates.

    For an iteration with error ``e_k = x_k - limit`` converging as ``e_{k+1} = C e_k^p`` with ``p > 1``,
    three consecutive errors give ``p = log(e_{k+1}/e_k) / log(e_k/e_{k-1})``. The median of the last
    ``tail`` such estimates is returned, with the spread of those estimates as the standard error.

    Only valid for ``p > 1``: for a linearly convergent iteration (bisection, fixed point) the same
    formula divides two nearly equal quantities and returns noise, so the *contraction factor*
    ``e_{k+1}/e_k`` is the meaningful measurement -- see :func:`contraction_factors`. A ``limit`` equal to
    the last iterate, or an iteration that never got below round-off, makes the errors constant and the
    order meaningless; both produce NaN rather than a number.
    """
    x = np.asarray(iterates, dtype=np.float64).ravel()
    if x.size < 3:
        raise ValueError(f"need at least three iterates to measure an order from two consecutive errors; got {x.size}")
    e = np.abs(x - float(limit))
    if np.any(e <= 0.0):
        x = x[e > 0.0]
        e = e[e > 0.0]
    if e.size < 3:
        raise ValueError("the iteration reached the limit exactly; the order cannot be measured from the remaining iterates")

    estimates: list[float] = []
    for k in range(1, e.size - 1):
        num = float(np.log(e[k + 1] / e[k]))
        den = float(np.log(e[k] / e[k - 1]))
        if np.isfinite(num) and np.isfinite(den) and den != 0.0:
            estimates.append(num / den)
    if not estimates:
        raise ValueError("could not form a single order estimate from these iterates (errors are not decreasing)")

    chosen = estimates[-tail:] if len(estimates) >= tail else estimates
    order = float(np.median(chosen))
    spread = float(np.std(chosen, ddof=1)) if len(chosen) > 1 else float("nan")
    return OrderEstimate(
        order=order,
        stderr=spread,
        ci95=(float(min(chosen)), float(max(chosen))),
        r_squared=float("nan"),  # not a regression; the interval is the spread of the estimates
        n_points=len(chosen),
        name="iteration order",
    )


def contraction_factors(iterates: Sequence[float] | np.ndarray, limit: float, *, tail: int = 3) -> tuple[float, ...]:
    """The last ``tail`` ratios ``|x_{k+1} - limit| / |x_k - limit|`` -- the measured factor of a linear iteration.

    A bisection run gives ~0.5, a fixed-point iteration gives ~``|g'(x*)|``. This is the number that
    matters for a linearly convergent method; the order is 1 by construction.
    """
    x = np.asarray(iterates, dtype=np.float64).ravel()
    e = np.abs(x - float(limit))
    e = e[e > 0.0]
    if e.size < 2:
        raise ValueError("need at least two non-zero errors to form a contraction factor")
    ratios = e[1:] / e[:-1]
    return tuple(float(r) for r in ratios[-tail:])


def richardson_extrapolate(f_h: float, f_h2: float, order: float, *, r: float = 2.0) -> float:
    """Extrapolate two approximations at step ``h`` and ``h/r`` to ``h -> 0``.

    With ``f(h) = f_exact + C h^p`` and ``f(h/r) = f_exact + C (h/r)^p``,
    ``f_exact = f(h/r) + (f(h/r) - f(h)) / (r^p - 1)``. Exactly the Roache formula used by
    ``pinneapple_analysis.verification.convergence``, kept here so the differentiation and quadrature
    modules can extrapolate without depending on the verification package.
    """
    if r <= 1.0:
        raise ValueError(f"refinement ratio r must be > 1; got {r}")
    if not np.isfinite(order) or order <= 0.0:
        raise ValueError(f"order must be positive and finite to extrapolate; got {order}")
    return float(f_h2 + (f_h2 - f_h) / (r**order - 1.0))


def truncation_roundoff_balance(
    step_sizes: Sequence[float] | np.ndarray,
    error_fn: Callable[[float], float],
    *,
    eps: float = float(np.finfo(float).eps),
    scale: float = 1.0,
    name: str = "error",
) -> VCurveResult:
    """Evaluate ``error_fn(h)`` across a range of steps and locate the truncation/round-off minimum.

    The total error of a finite-difference-like computation is ``C h^p + eps * S / h^q`` (``q = 1`` for a
    first difference, ``q = 1/2`` for a centred one), so on a log-log plot it is a V: slope ``+p`` on the
    left, ``-q`` on the right. Both slopes and the vertex are *measured* here.

    ``scale`` is the magnitude of the quantity being differentiated (``|f(x)|`` for a derivative): the
    round-off term is relative to it, so the predicted optimum is
    ``h_opt = (q * eps * scale / (p * C))**(1/(p+q))`` with ``C`` taken from the measured truncation
    branch. The ratio of predicted to measured ``h_opt`` is what says whether the classical balance
    argument applies to this computation at all.

    Raises ``ValueError`` if fewer than four usable steps are given: with fewer, neither branch can be fit
    and any "optimum" would be the lowest of three arbitrary points.
    """
    h_all = np.asarray(step_sizes, dtype=np.float64).ravel()
    if h_all.size < 4:
        raise ValueError(f"a V-curve needs at least four step sizes to resolve both branches; got {h_all.size}")
    e_all = np.array([abs(float(error_fn(float(h)))) for h in h_all], dtype=np.float64)

    order_idx = np.argsort(h_all)
    h = h_all[order_idx]
    e = e_all[order_idx]
    i_min = int(np.argmin(e))
    optimal_step = float(h[i_min])
    optimal_error = float(e[i_min])

    diagnostics: list[Diagnostic] = []
    truncation_order = float("nan")
    roundoff_slope = float("nan")
    predicted: float | None = None

    if i_min >= 1:
        truncation_order = fit_order(h[: i_min + 1], np.maximum(e[: i_min + 1], np.finfo(float).tiny), name=name).order
    if i_min < h.size - 1 and i_min >= 1:
        roundoff_slope = fit_order(h[i_min:], np.maximum(e[i_min:], np.finfo(float).tiny), name=name).order
    if np.isfinite(truncation_order) and np.isfinite(roundoff_slope) and roundoff_slope < 0.0 and truncation_order > 0.0:
        # C from the truncation branch, using the model e = C h^p + r, with r read at the optimum.
        c = max((e[0] - optimal_error) / h[0] ** truncation_order, np.finfo(float).tiny)
        q = -roundoff_slope
        predicted = float((q * eps * scale / (truncation_order * c)) ** (1.0 / (truncation_order + q)))
        diagnostics.append(
            explain(
                "QUAD.ROUNDOFF_FLOOR",
                measured_optimal_step=optimal_step,
                predicted_optimal_step=predicted,
                truncation_order=truncation_order,
                roundoff_slope=roundoff_slope,
            )
        )

    return VCurveResult(
        steps=h,
        errors=e,
        optimal_step=optimal_step,
        optimal_error=optimal_error,
        truncation_order=float(truncation_order),
        roundoff_slope=float(roundoff_slope),
        predicted_optimal_step=predicted,
        diagnostics=tuple(diagnostics),
    )
