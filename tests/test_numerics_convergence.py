"""Tests for ``pinneapple_analysis.numerics.convergence`` and ``...diagnostics``.

Every assertion here measures something against a case whose answer is known in closed form: a synthetic
error law ``C h^p`` with a chosen ``p``, a synthetic iteration with a chosen order, and a difference
quotient whose truncation and round-off terms are known analytically. Checking that a function returns
without raising would test nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.numerics.convergence import (
    ErrorCurve,
    OrderEstimate,
    fit_order,
    order_from_iterates,
    richardson_extrapolate,
    truncation_roundoff_balance,
)
from pinneapple_analysis.numerics.diagnostics import (
    CATALOGUE,
    Diagnostic,
    catalogue,
    explain,
    has_code,
    worst,
)


# --------------------------------------------------------------------------- fit_order
@pytest.mark.parametrize("order", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
def test_fit_order_recovers_a_known_power_law(order: float) -> None:
    """A synthetic error ``3 h^p`` must give back ``p`` with an interval that contains it."""
    h = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    estimate = fit_order(h, 3.0 * h**order)
    assert isinstance(estimate, OrderEstimate)
    assert estimate.order == pytest.approx(order, abs=1e-9)
    assert estimate.r_squared == pytest.approx(1.0, abs=1e-12)
    lo, hi = estimate.ci95
    assert lo <= order <= hi
    assert estimate.n_points == h.size


def test_fit_order_interval_widens_when_the_data_is_noisy() -> None:
    """The confidence interval is a measurement of the fit, not a decoration: noise must widen it."""
    rng = np.random.default_rng(7)
    h = np.logspace(-3, -1, 12)
    clean = fit_order(h, h**2.0)
    noisy = fit_order(h, h**2.0 * np.exp(rng.normal(0.0, 0.20, h.size)))
    assert clean.stderr < noisy.stderr
    assert noisy.order == pytest.approx(2.0, abs=0.25)
    assert noisy.ci95[0] < 2.0 < noisy.ci95[1]


def test_fit_order_reports_nan_uncertainty_for_two_points() -> None:
    """Two points fix the slope exactly and say nothing about its uncertainty -- NaN, not zero."""
    estimate = fit_order([0.1, 0.05], [1e-2, 2.5e-3])
    assert estimate.order == pytest.approx(2.0, abs=1e-12)
    assert np.isnan(estimate.stderr)
    assert np.isnan(estimate.ci95[0]) and np.isnan(estimate.ci95[1])


def test_fit_order_rejects_a_zero_or_single_point() -> None:
    """A saturated error (exactly zero) cannot produce an order; inventing one would be a lie."""
    with pytest.raises(ValueError):
        fit_order([0.1, 0.05], [1e-3, 0.0])
    with pytest.raises(ValueError):
        fit_order([0.1, 0.1, 0.1], [1.0, 0.5, 0.25])


def test_agrees_with_uses_a_relative_tolerance() -> None:
    """The order a test should accept: 1.85 is a second-order scheme in its asymptotic range."""
    estimate = fit_order(np.array([0.1, 0.05, 0.025, 0.0125]), 2.0 * np.array([0.1, 0.05, 0.025, 0.0125]) ** 1.85)
    assert estimate.agrees_with(2.0, rtol=0.10)
    assert not estimate.agrees_with(2.0, rtol=0.05)
    assert not estimate.agrees_with(1.0)


# --------------------------------------------------------------------------- order_from_iterates
@pytest.mark.parametrize("order", [1.618, 2.0, 3.0])
def test_order_from_iterates_recovers_the_iteration_order(order: float) -> None:
    """Build errors that obey ``e_{k+1} = e_k^p`` and check the recovered ``p``."""
    limit = 1.0
    errors = [0.5]
    for _ in range(7):
        errors.append(errors[-1] ** order)
    iterates = [limit + e for e in errors if e > 1e-15]
    estimate = order_from_iterates(iterates, limit, tail=3)
    assert estimate.order == pytest.approx(order, rel=0.05)


def test_order_from_iterates_needs_three_iterates() -> None:
    with pytest.raises(ValueError):
        order_from_iterates([1.0, 1.1], 1.0)


# --------------------------------------------------------------------------- V-curve
def _forward_difference_error(h: float) -> float:
    """Absolute error of the forward difference of exp at x = 1: truncation h/2 e, round-off ~eps e/h."""
    x = 1.0
    return abs((np.exp(x + h) - np.exp(x)) / h - np.exp(x))


def test_v_curve_locates_the_truncation_roundoff_balance() -> None:
    """The measured optimum step must be where theory puts it: ``h* = sqrt(2 eps)`` for a forward difference.

    For ``(f(x+h) - f(x))/h`` the total error is ``h/2 |f''| + 2 eps |f|/h``, minimised at
    ``h* = sqrt(4 eps f/f'') = sqrt(2 eps) ~ 2.1e-8`` for ``f = exp`` at ``x = 1``. The measurement also
    has to show the two branches: order ~1 on the truncation side, slope ~-1 on the round-off side.
    """
    eps = np.finfo(float).eps
    steps = np.logspace(-12, -2, 60)
    curve = truncation_roundoff_balance(steps, _forward_difference_error, scale=np.e)
    assert curve.is_balanced
    # the constant in h* is model-dependent (the rounding of exp(1+h) - exp(1) is not exactly eps), so the
    # honest claim is the order of the optimum, within a factor of a few of sqrt(eps) = 1.5e-8
    assert 0.5 * np.sqrt(eps) < curve.optimal_step < 5.0 * np.sqrt(eps)
    assert curve.truncation_order == pytest.approx(1.0, abs=0.15)
    assert curve.roundoff_slope == pytest.approx(-1.0, abs=0.2)
    assert curve.predicted_optimal_step is not None
    assert 0.1 < curve.predicted_optimal_step / curve.optimal_step < 10.0
    assert [d.code for d in curve.diagnostics] == ["QUAD.ROUNDOFF_FLOOR"]


def test_v_curve_requires_four_steps() -> None:
    with pytest.raises(ValueError):
        truncation_roundoff_balance([1e-3, 1e-4, 1e-5], _forward_difference_error)


def test_error_curve_fits_the_fine_tail_separately() -> None:
    """A pre-asymptotic first point biases the global fit; the fine-mesh tail recovers the true order."""
    h = np.array([0.5, 0.1, 0.05, 0.025])
    errors = np.array([0.9, 3.0e-2, 7.5e-3, 1.875e-3])  # the coarse point sits above the h^2 line
    curve = ErrorCurve(steps=h, errors=errors)
    assert curve.order().order > 2.05
    assert curve.order(tail=3).order == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------- Richardson
def test_richardson_extrapolation_removes_the_leading_error_term() -> None:
    """Two second-order approximations extrapolate to fourth-order accuracy, measurably."""
    exact = np.cos(0.7)  # the central difference approximates the DERIVATIVE of sin, which is cos

    def central(h: float) -> float:
        return (np.sin(0.7 + h) - np.sin(0.7 - h)) / (2.0 * h)

    extrapolated = [richardson_extrapolate(central(h), central(h / 2.0), order=2.0) for h in (0.1, 0.05, 0.025)]
    raw = [abs(central(h) - exact) for h in (0.1, 0.05, 0.025)]
    improved = [abs(v - exact) for v in extrapolated]
    assert improved[-1] < raw[-1] / 100.0
    assert fit_order([0.1, 0.05, 0.025], improved).order == pytest.approx(4.0, abs=0.5)


def test_richardson_rejects_a_bad_ratio_or_order() -> None:
    with pytest.raises(ValueError):
        richardson_extrapolate(1.0, 2.0, order=2.0, r=1.0)
    with pytest.raises(ValueError):
        richardson_extrapolate(1.0, 2.0, order=0.0)


# --------------------------------------------------------------------------- diagnostics
def test_diagnostic_catalogue_is_self_consistent() -> None:
    """Every code is explained, and every explanation has a remedy and a valid severity."""
    codes = catalogue()
    assert len(codes) >= 30
    for code in codes:
        diagnostic = explain(code)
        assert isinstance(diagnostic, Diagnostic)
        assert diagnostic.severity in ("info", "warning", "error")
        assert diagnostic.title and diagnostic.explanation and diagnostic.remedy
        assert diagnostic.code == code
    assert set(codes) == set(CATALOGUE)


def test_explain_carries_the_measured_context_and_serialises() -> None:
    diagnostic = explain("ODE.STIFF", stiffness_ratio=1.2e6)
    assert diagnostic.context["stiffness_ratio"] == 1.2e6
    payload = diagnostic.to_dict()
    assert payload["code"] == "ODE.STIFF" and payload["context"]["stiffness_ratio"] == 1.2e6
    assert "ODE.STIFF" in diagnostic.format()


def test_unknown_code_is_an_error_not_a_silent_default() -> None:
    assert not has_code("NOPE.NOPE")
    with pytest.raises(KeyError):
        explain("NOPE.NOPE")


def test_worst_picks_the_most_severe() -> None:
    """info < warning < error, and the first of equal severities wins (the caller's order)."""
    diagnostics = [explain("QUAD.GIBBS"), explain("ODE.STIFF"), explain("ROOTS.NO_BRACKET")]
    assert worst(diagnostics).code == "ROOTS.NO_BRACKET"  # the only "error" in the list
    assert worst([explain("QUAD.GIBBS"), explain("ODE.STIFF")]).code == "ODE.STIFF"
    assert worst([]) is None
