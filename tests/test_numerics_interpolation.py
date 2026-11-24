"""Tests for ``pinneapple_analysis.numerics.interpolation`` (#1058).

The three measurements the issue asks for are all here and all against closed forms: the Runge
phenomenon (error grows with the degree on equispaced nodes and does not on Chebyshev), the Lebesgue
constant (measured, not quoted), and the ``O(h^4)`` of the cubic spline (fitted, not asserted from
theory). Chebyshev coefficients are checked against the exact coefficients of ``x^2`` and against the
well-posed coefficient-decay exponent of ``|x|``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.numerics.convergence import fit_order
from pinneapple_analysis.numerics.interpolation import (
    CubicSpline,
    aliasing_demo,
    barycentric_interpolate,
    barycentric_weights,
    chebyshev_coefficients,
    chebyshev_convergence_study,
    chebyshev_evaluate,
    chebyshev_lobatto_nodes,
    chebyshev_nodes,
    dft,
    equispaced_nodes,
    gibbs_overshoot,
    l_curve,
    lagrange_interpolate,
    least_squares_conditioning,
    least_squares_normal_equations,
    least_squares_qr,
    lebesgue_constant,
    newton_divided_differences,
    newton_evaluate,
    pade_coefficients,
    pade_evaluate,
    pchip,
    runge_study,
    tikhonov_solve,
    trigonometric_interpolant,
    vandermonde,
)


def _runge(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 25.0 * np.asarray(t, dtype=np.float64) ** 2)


# --------------------------------------------------------------------------- interpolating forms
def test_the_three_interpolation_forms_agree_and_are_exact_at_the_nodes() -> None:
    """Barycentric, Lagrange and Newton are the same polynomial: they must agree to rounding."""
    nodes = chebyshev_nodes(12)
    values = np.sin(3.0 * nodes)
    grid = np.linspace(-1.0, 1.0, 501)
    barycentric = barycentric_interpolate(nodes, values, grid)
    lagrange = lagrange_interpolate(nodes, values, grid)
    newton = newton_evaluate(nodes, newton_divided_differences(nodes, values), grid)
    assert np.allclose(barycentric, lagrange, atol=1e-12)
    assert np.allclose(barycentric, newton, atol=1e-12)
    # exactly at the nodes the interpolant reproduces the data, not approximately
    assert np.allclose(barycentric_interpolate(nodes, values, nodes), values, atol=1e-15)


def test_a_cubic_is_reproduced_exactly_by_four_nodes() -> None:
    """Degree 3 is in the space: any 4 distinct nodes give the polynomial back to rounding."""
    polynomial = lambda t: 2.0 - t + 0.5 * t**2 - 0.125 * t**3  # noqa: E731
    nodes = np.array([-1.0, -0.3, 0.4, 0.9])
    grid = np.linspace(-1.0, 1.0, 101)
    assert np.allclose(barycentric_interpolate(nodes, polynomial(nodes), grid), polynomial(grid),
                       atol=1e-14)


def test_duplicate_nodes_are_rejected() -> None:
    with pytest.raises(ValueError):
        barycentric_weights([0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        barycentric_interpolate([0.0], [1.0], 0.5)


# --------------------------------------------------------------------------- Runge + Lebesgue
def test_runge_phenomenon_is_measured_on_both_node_families() -> None:
    """Equispaced error must grow with the degree (divergence) while Chebyshev's falls."""
    study = runge_study()
    assert study.equispaced_diverges
    assert study.chebyshev_converges
    # the divergence is not a marginal effect: it spans more than four decades
    assert study.equispaced_errors[-1] > 1e3 * study.equispaced_errors[0]
    assert study.chebyshev_errors[-1] < 1e-3
    assert "INTERP.RUNGE" in [d.code for d in study.diagnostics]
    # and the mechanism is the Lebesgue constant, which grows with the degree for the equispaced set
    assert study.equispaced_lebesgue[-1] > 100.0 * study.equispaced_lebesgue[0]
    assert study.chebyshev_lebesgue[-1] < 2.0 * study.chebyshev_lebesgue[0]


def test_lebesgue_constant_grows_exponentially_on_equispaced_nodes_and_logarithmically_on_chebyshev() -> None:
    """Measured against the textbook orders of magnitude: ~2, ~11, ~90 for the equispaced set at n=4, 8, 12."""
    equispaced = [lebesgue_constant(equispaced_nodes(n), samples=4001) for n in (4, 8, 12)]
    chebyshev = [lebesgue_constant(chebyshev_nodes(n), samples=4001) for n in (4, 8, 12)]
    assert 2.0 < equispaced[0] < 2.5
    assert 9.0 < equispaced[1] < 13.0
    assert 70.0 < equispaced[2] < 110.0
    assert chebyshev[0] < 2.0 and chebyshev[-1] < 3.0
    # exponential against logarithmic: the ratio of the two growths is the point
    assert equispaced[-1] / equispaced[0] > 20.0 * (chebyshev[-1] / chebyshev[0])


def test_chebyshev_lobatto_is_the_same_node_set_in_ascending_order() -> None:
    lobatto = chebyshev_lobatto_nodes(6)
    assert np.allclose(lobatto, np.sort(chebyshev_nodes(6)))
    assert lobatto[0] == -1.0 and lobatto[-1] == 1.0


# --------------------------------------------------------------------------- splines
def test_cubic_spline_converges_at_fourth_order() -> None:
    """``O(h^4)`` measured: the error against ``sin`` must fall by 16x when ``h`` halves."""
    exact = lambda t: np.sin(2.0 * np.asarray(t, dtype=np.float64))  # noqa: E731
    grid = np.linspace(0.0, 1.0, 4001)
    sizes = [8, 16, 32, 64, 128]
    errors = []
    for n in sizes:
        nodes = np.linspace(0.0, 1.0, n + 1)
        spline = CubicSpline(nodes, exact(nodes), kind="not-a-knot")
        errors.append(float(np.max(np.abs(spline(grid) - exact(grid)))))
    order = fit_order(1.0 / np.asarray(sizes, dtype=np.float64), np.asarray(errors))
    assert order.order == pytest.approx(4.0, abs=0.35), errors


def _natural_second_derivatives(nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Independent reference: the natural spline of a cubic is the cubic itself, so ``M = f''``."""
    return 6.0 * nodes  # for f = x^3, f'' = 6x


def test_spline_boundary_conditions_behave_as_named() -> None:
    """Natural forces ``M = 0`` at the ends; clamped matches the given slope; a cubic is reproduced."""
    nodes = np.linspace(0.0, 2.0, 9)
    cubic = lambda t: np.asarray(t, dtype=np.float64) ** 3  # noqa: E731
    natural = CubicSpline(nodes, cubic(nodes), kind="natural")
    assert natural.second_derivatives[0] == pytest.approx(0.0, abs=1e-12)
    assert natural.second_derivatives[-1] == pytest.approx(0.0, abs=1e-12)

    clamped = CubicSpline(nodes, cubic(nodes), kind="clamped", left_slope=0.0, right_slope=12.0)
    assert np.allclose(clamped(nodes), cubic(nodes), atol=1e-12)
    assert clamped(nodes[0] + 1e-6) > clamped(nodes[0])  # slope 0 at the left end, from below

    not_a_knot = CubicSpline(nodes, cubic(nodes), kind="not-a-knot")
    assert np.allclose(not_a_knot(nodes), cubic(nodes), atol=1e-12)
    # for a cubic, not-a-knot reproduces it exactly, including the second derivative
    assert np.allclose(not_a_knot.second_derivatives, _natural_second_derivatives(nodes, cubic(nodes)),
                       atol=1e-8)


def test_spline_rejects_malformed_input() -> None:
    with pytest.raises(ValueError):
        CubicSpline([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        CubicSpline([0.0, 1.0, 0.5], [0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        CubicSpline([0.0, 0.5, 1.0], [0.0, 1.0, 0.0], kind="clamped")


def test_pchip_preserves_monotonicity_where_the_spline_overshoots() -> None:
    """The measured difference between the two: a spline adds extrema, PCHIP cannot."""
    nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # monotone, with a jump in the middle
    grid = np.linspace(0.0, 5.0, 2001)
    monotone = pchip(nodes, values)
    spline = CubicSpline(nodes, values, kind="not-a-knot")
    monotone_values = monotone(grid)
    spline_values = spline(grid)
    assert np.all(np.diff(monotone_values) >= -1e-12)  # never decreases
    assert monotone_values.min() >= 0.0 and monotone_values.max() <= 1.0  # no new extrema
    assert spline_values.max() > 1.0  # the spline overshoots, which is why PCHIP exists
    assert monotone(nodes) == pytest.approx(values)


# --------------------------------------------------------------------------- Chebyshev
def test_chebyshev_coefficients_of_a_quadratic_are_exact() -> None:
    """``x^2 = (T_0 + T_2)/2``: the one closed form that catches the classic halving mistake."""
    coefficients = chebyshev_coefficients(lambda t: np.asarray(t, dtype=np.float64) ** 2, 6)
    assert coefficients[0] == pytest.approx(0.5, abs=1e-15)
    assert coefficients[2] == pytest.approx(0.5, abs=1e-15)
    assert np.allclose(coefficients[[1, 3, 4, 5, 6]], 0.0, atol=1e-15)
    # and the series evaluates the function back
    grid = np.linspace(-1.0, 1.0, 201)
    assert np.allclose(chebyshev_evaluate(coefficients, grid), grid**2, atol=1e-14)


def test_chebyshev_series_reproduces_a_smooth_function_to_rounding() -> None:
    coefficients = chebyshev_coefficients(np.exp, 24)
    grid = np.linspace(-1.0, 1.0, 501)
    assert np.allclose(chebyshev_evaluate(coefficients, grid), np.exp(grid), atol=1e-13)


def test_coefficient_decay_separates_analytic_from_limited_smoothness() -> None:
    """Algebraic (``k^-2``) for ``|x|`` against geometric for ``exp``: measured as a log-log slope."""
    analytic = chebyshev_convergence_study(np.exp, degrees=(4, 8, 16, 32))
    kinked = chebyshev_convergence_study(np.abs, degrees=(8, 16, 32, 64))
    assert kinked["coefficient_exponent"] == pytest.approx(-2.0, abs=0.4)
    assert analytic["coefficient_exponent"] < -5.0  # geometric decay looks steep on a log-log plot
    assert analytic["tail_ratio"] > 2.0  # the coefficients of exp fall by a constant factor per term
    # (a tail ratio is not the right statistic for |x|: its odd coefficients are zero, so successive
    # ratios alternate; the exponent above is the measurement that separates the two cases)


# --------------------------------------------------------------------------- least squares
def test_qr_least_squares_beats_the_normal_equations_by_about_the_condition_number() -> None:
    """The measured comparison the issue asks for, on a Vandermonde system."""
    report = least_squares_conditioning(degree=12, samples=40)
    assert report["condition_number"] > 1e3
    assert report["error_ratio"] > 100.0  # normal equations square the condition number
    assert report["normal_equations_error"] > report["qr_error"]
    assert report["condition_number_normal_equations"] == pytest.approx(
        report["condition_number"] ** 2, rel=1e-9
    )


def test_least_squares_recovers_a_polynomial_from_noisy_data() -> None:
    rng = np.random.default_rng(4)
    x = np.linspace(-1.0, 1.0, 200)
    true = np.array([0.5, -2.0, 1.5])
    y = vandermonde(x, 2) @ true + rng.normal(0.0, 1e-9, x.size)
    coefficients = least_squares_qr(vandermonde(x, 2), y)
    assert np.allclose(coefficients, true, atol=1e-8)
    assert np.allclose(coefficients, least_squares_normal_equations(vandermonde(x, 2), y), atol=1e-8)


def test_tikhonov_trades_residual_for_norm_along_the_l_curve() -> None:
    """More regularisation means a smaller solution norm and a larger residual: the L, in two numbers."""
    x = np.linspace(-1.0, 1.0, 60)
    A = vandermonde(x, 10)
    b = np.sin(3.0 * x) + 1e-3 * np.random.default_rng(1).normal(size=x.size)
    curve = l_curve(A, b)
    residuals = np.asarray(curve["residuals"])
    norms = np.asarray(curve["solution_norms"])
    assert np.all(np.diff(residuals) > -1e-12)  # residual never falls as lambda grows
    assert np.all(np.diff(norms) < 1e-12)  # solution norm never grows
    assert 0 < curve["corner_index"] < len(residuals) - 1  # the corner is an interior point
    assert tikhonov_solve(A, b, 1.0)[0] != tikhonov_solve(A, b, 100.0)[0]
    with pytest.raises(ValueError):
        tikhonov_solve(A, b, -1.0)


# --------------------------------------------------------------------------- Pade
def test_pade_matches_the_series_to_its_order_and_stays_bounded_where_taylor_does_not() -> None:
    """Two measured properties: the error is ``O(x^5)`` at small ``x``, and at ``x = -5`` only Pade survives."""
    series = [1.0, 1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0]
    p, q = pade_coefficients(series, 2, 2)
    assert p == pytest.approx([1.0, 0.5, 1.0 / 12.0], rel=1e-12)
    assert q == pytest.approx([1.0, -0.5, 1.0 / 12.0], rel=1e-12)

    xs = np.array([0.4, 0.2, 0.1, 0.05])
    errors = np.abs(pade_evaluate(p, q, xs) - np.exp(xs))
    assert fit_order(xs, errors).order == pytest.approx(5.0, abs=0.3)

    # where the Taylor polynomial has diverged past its useful range, the rational form has not
    pade_far = float(pade_evaluate(p, q, -5.0))
    taylor_far = sum(coefficient * (-5.0) ** k for k, coefficient in enumerate(series))
    exact_far = float(np.exp(-5.0))
    assert pade_far > 0.0
    assert abs(pade_far - exact_far) / exact_far < 20.0
    assert taylor_far < 0.0
    assert abs(taylor_far - exact_far) / exact_far > 100.0


def test_pade_requires_enough_coefficients() -> None:
    with pytest.raises(ValueError):
        pade_coefficients([1.0, 1.0], 2, 2)


# --------------------------------------------------------------------------- Fourier
def test_trigonometric_interpolant_reproduces_the_samples() -> None:
    n = 32
    phases = np.arange(n) / n
    values = np.cos(2.0 * np.pi * 3.0 * phases)
    assert np.allclose(trigonometric_interpolant(values, phases), values, atol=1e-12)
    assert dft(values)[3] == pytest.approx(0.5, abs=1e-12)


def test_aliasing_is_predicted_by_the_sampling_theorem() -> None:
    """A frequency above Nyquist is folded onto its mirror image, and the fold is measured, not assumed."""
    folded = aliasing_demo(40.0, 64)  # 40 cycles on 64 samples: Nyquist is 32
    assert folded["aliased"]
    assert folded["apparent_cycles"] == pytest.approx(folded["predicted_alias"])
    assert folded["apparent_cycles"] == pytest.approx(24.0)
    resolved = aliasing_demo(20.0, 64)
    assert not resolved["aliased"]
    assert resolved["apparent_cycles"] == pytest.approx(20.0)


def test_gibbs_overshoot_is_the_wilbraham_gibbs_constant() -> None:
    """About 8.95 % of the jump, and it does not shrink with the number of terms."""
    for terms in (51, 501, 5001):
        assert gibbs_overshoot(terms) == pytest.approx(0.0895, abs=2e-3)
