"""Tests for ``pinneapple_analysis.numerics.quadrature`` (#1059).

Every rule is checked against an integral whose value is known exactly -- polynomials for the exactness
degree of the Gaussian rules, ``exp`` and ``sin`` for the convergence of the composite and spectral ones,
``pi/2`` for the endpoint-singular substitution -- and the *order* is always fitted, never asserted from
the textbook. The node sets are cross-checked against ``scipy.special``, which is an independent
implementation of the same orthogonal polynomials.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis.numerics.convergence import fit_order
from pinneapple_analysis.numerics.quadrature import (
    adaptive_simpson,
    clenshaw_curtis,
    compare_rules_at_equal_cost,
    complex_step_derivative,
    composite_midpoint,
    composite_simpson,
    composite_trapezoid,
    fd_weights,
    finite_difference,
    gauss_hermite,
    gauss_laguerre,
    gauss_legendre,
    gauss_legendre_nodes,
    gauss_lobatto,
    halton_sequence,
    integrate_endpoint_singularity,
    measure_differentiation_order,
    monte_carlo_integrate,
    optimal_step_study,
    quasi_monte_carlo_integrate,
    romberg,
    sampling_error_rate,
)


# --------------------------------------------------------------------------- finite differences
def test_fd_weights_reproduce_the_classic_stencils() -> None:
    """Fornberg's algorithm against the stencils everybody memorises."""
    assert fd_weights([-1.0, 0.0, 1.0], 1, 0.0) == pytest.approx([-0.5, 0.0, 0.5])
    assert fd_weights([-1.0, 0.0, 1.0], 2, 0.0) == pytest.approx([1.0, -2.0, 1.0])
    assert fd_weights([0.0, 1.0], 1, 0.0) == pytest.approx([-1.0, 1.0])
    assert fd_weights([-2.0, -1.0, 0.0, 1.0, 2.0], 1, 0.0) == pytest.approx(
        [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
    )
    # a staggered (Chebyshev) node set works too: no stencil has to be memorised
    chebyshev = np.cos(np.pi * np.arange(4) / 3)
    weights = fd_weights(chebyshev, 1, 0.0)
    assert sum(weights) == pytest.approx(0.0, abs=1e-14)  # a first derivative annihilates constants


def test_fd_weights_reject_duplicate_nodes() -> None:
    with pytest.raises(ValueError):
        fd_weights([0.0, 1.0, 1.0], 1, 0.0)
    with pytest.raises(ValueError):
        fd_weights([0.0, 1.0], -1, 0.0)


@pytest.mark.parametrize(
    ("scheme", "expected_order", "steps"),
    [
        ("forward", 1.0, np.logspace(-1, -8, 12)),
        ("central", 2.0, np.logspace(-1, -9, 14)),
        ("central4", 4.0, np.logspace(-0.5, -4, 12)),
        ("second-central", 2.0, np.logspace(-1, -7, 12)),
        ("second-central4", 4.0, np.logspace(-0.5, -3, 10)),
    ],
)
def test_stencil_orders_are_measured(scheme: str, expected_order: float, steps: np.ndarray) -> None:
    """The truncation order of each scheme, fitted on ``exp`` where the exact derivative is ``exp``."""
    report = measure_differentiation_order(np.exp, np.exp, 1.0, steps, scheme=scheme)
    assert report["order"] == pytest.approx(expected_order, abs=0.08), report


def test_refinement_order_of_magnitude_is_the_v_curve_vertex() -> None:
    """The measured optimal step must match the classical balance ``h* ~ (eps |f| / C)^(1/(p+1))``."""
    curve = optimal_step_study(np.exp, np.exp, 1.0, scheme="forward")
    eps = np.finfo(float).eps
    assert curve.is_balanced
    assert curve.truncation_order == pytest.approx(1.0, abs=0.15)
    assert curve.roundoff_slope == pytest.approx(-1.0, abs=0.2)
    assert 0.05 * math.sqrt(eps) < curve.optimal_step < 10.0 * math.sqrt(eps)


def test_complex_step_derivative_has_no_step_size_problem() -> None:
    """The error stays at the rounding level for every ``h``, which no real-step stencil achieves."""
    exact = float(np.exp(1.0))
    for h in (1e-10, 1e-15, 1e-20, 1e-30, 1e-50):
        assert abs(complex_step_derivative(np.exp, 1.0, h) - exact) < 1e-14
    # the forward difference with the same tiny steps is pure noise
    assert abs(finite_difference(np.exp, 1.0, 1e-15, scheme="forward") - exact) > 1e-3


# --------------------------------------------------------------------------- Newton-Cotes
def test_composite_rules_show_their_order() -> None:
    """Trapezoid and midpoint are second order, Simpson fourth -- each fitted, each on ``exp``."""
    exact = math.e - 1.0
    sizes = np.array([8.0, 16.0, 32.0, 64.0, 128.0])
    for rule, expected in ((composite_trapezoid, 2.0), (composite_midpoint, 2.0),
                           (composite_simpson, 4.0)):
        errors = np.array([abs(rule(np.exp, 0.0, 1.0, int(n)) - exact) for n in sizes])
        order = fit_order(1.0 / sizes, errors, name=rule.__name__)
        assert order.order == pytest.approx(expected, abs=0.15), (rule.__name__, errors)


def test_romberg_diagonal_gains_two_orders_per_column() -> None:
    """Richardson on the trapezoid rule: each extrapolation column removes one power of ``h^2``."""
    exact = math.e - 1.0
    report = romberg(np.exp, 0.0, 1.0, levels=6)
    assert abs(report["value"] - exact) < 1e-14
    first_column = np.array([abs(report["table"][k, 0] - exact) for k in range(1, 6)])
    # the trapezoid column falls by 4 per halving (second order); four levels therefore buy 4**4 = 256
    assert first_column[0] > first_column[-1] * 100.0
    assert first_column[1] / first_column[2] == pytest.approx(4.0, rel=0.1)
    # the exactness of the diagonal: a 2k-order rule built from k+1 trapezoid levels
    assert abs(report["diagonal"][-1] - exact) < 1e-15


# --------------------------------------------------------------------------- Gaussian rules
def test_gauss_legendre_nodes_match_scipy() -> None:
    scipy_special = pytest.importorskip("scipy.special")
    for n in (1, 2, 5, 10, 20):
        nodes, weights = gauss_legendre_nodes(n)
        reference_nodes, reference_weights = scipy_special.roots_legendre(n)
        assert np.allclose(nodes, reference_nodes, atol=1e-14)
        assert np.allclose(weights, reference_weights, atol=1e-14)


def test_gauss_legendre_is_exact_to_degree_two_n_minus_one() -> None:
    """The defining property: 10 nodes integrate ``x^19`` (and every lower power) exactly."""
    n = 10
    for power in range(0, 2 * n):
        exact = 2.0 / (power + 1) if power % 2 == 0 else 0.0
        computed = gauss_legendre(lambda t, p=power: np.asarray(t) ** p, -1.0, 1.0, n)
        assert computed == pytest.approx(exact, abs=1e-13), power


def test_gauss_legendre_converges_geometrically_for_an_analytic_integrand() -> None:
    exact = math.e - 1.0
    errors = [abs(gauss_legendre(np.exp, 0.0, 1.0, n) - exact) for n in (2, 4, 6, 8)]
    assert errors[-1] < 1e-10
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))


def test_gauss_lobatto_includes_the_endpoints_and_loses_one_order() -> None:
    """``n`` points are exact to degree ``2n-3``: 5 points integrate ``x^7`` but not ``x^8``."""
    assert gauss_lobatto(lambda t: np.asarray(t) ** 7, -1.0, 1.0, 5) == pytest.approx(0.0, abs=1e-14)
    assert gauss_lobatto(lambda t: np.asarray(t) ** 6, -1.0, 1.0, 5) == pytest.approx(2.0 / 7.0, abs=1e-13)
    assert gauss_lobatto(np.exp, 0.0, 1.0, 6) == pytest.approx(math.e - 1.0, abs=1e-12)
    with pytest.raises(ValueError):
        gauss_lobatto(np.exp, 0.0, 1.0, 1)


def test_gauss_laguerre_and_hermite_integrate_their_weighted_moments() -> None:
    """``int_0^inf e^-x x^k dx = k!`` and ``int_-inf^inf e^-x^2 x^2 dx = sqrt(pi)/2``."""
    for k in range(0, 7):
        exact = float(math.factorial(k))
        computed = gauss_laguerre(lambda t, p=k: np.asarray(t) ** p, 8)
        assert computed == pytest.approx(exact, rel=1e-11), k
    assert gauss_hermite(lambda t: np.asarray(t) ** 2, 4) == pytest.approx(math.sqrt(math.pi) / 2.0, rel=1e-12)
    assert gauss_hermite(lambda t: np.asarray(t) ** 4, 4) == pytest.approx(3.0 * math.sqrt(math.pi) / 4.0,
                                                                        rel=1e-12)
    # an odd integrand is integrated to rounding, not to truncation, because the nodes are symmetric
    assert gauss_hermite(lambda t: np.asarray(t) ** 5, 6) == pytest.approx(0.0, abs=1e-14)


def test_clenshaw_curtis_rivals_gauss_legendre_on_a_smooth_integrand() -> None:
    exact = math.e - 1.0
    gauss = abs(gauss_legendre(np.exp, 0.0, 1.0, 10) - exact)
    clenshaw = abs(clenshaw_curtis(np.exp, 0.0, 1.0, 10) - exact)
    assert clenshaw < 1e-12
    assert gauss < 1e-13
    assert abs(clenshaw_curtis(np.sin, 0.0, math.pi, 10) - 2.0) < 1e-9


# --------------------------------------------------------------------------- adaptive / singular
def test_adaptive_simpson_reports_an_honest_error_estimate() -> None:
    """The embedded estimate must be *conservative* -- at least as large as the error actually made."""
    report = adaptive_simpson(np.exp, 0.0, 1.0, tol=1e-12)
    exact = math.e - 1.0
    true_error = abs(report["value"] - exact)
    assert true_error < report["error_estimate"]  # the estimate is an upper bound here
    assert report["error_estimate"] < 1e-9
    assert report["evaluations"] > 10
    # on a harder integrand the estimate stays conservative and the value stays inside it
    harder = adaptive_simpson(lambda x: np.sqrt(np.abs(np.asarray(x) - 0.3)), 0.0, 1.0, tol=1e-10)
    exact_harder = 0.3**1.5 * 2.0 / 3.0 + 0.7**1.5 * 2.0 / 3.0
    assert abs(harder["value"] - exact_harder) <= harder["error_estimate"] + 1e-12


def test_substitution_restores_the_order_of_a_singular_integrand() -> None:
    """``x^-1/2/(1+x)`` on ``[0,1]`` is ``pi/2``; the raw rule converges at half order, the substituted one at 2.

    This is the measurement that makes the singularity diagnostic actionable: the rule is not broken, the
    integrand is outside its hypothesis.
    """
    integrand = lambda x: np.asarray(x, dtype=np.float64) ** -0.5 / (1.0 + np.asarray(x, dtype=np.float64))  # noqa: E731,E501
    exact = math.pi / 2.0
    sizes = np.array([40.0, 80.0, 160.0, 320.0])
    # the midpoint rule never evaluates the singular endpoint, so it can see the half-order behaviour
    raw_errors = np.array([abs(composite_midpoint(integrand, 0.0, 1.0, int(n)) - exact) for n in sizes])
    substituted_errors = np.array([
        abs(integrate_endpoint_singularity(lambda t: 1.0 / (1.0 + np.asarray(t)), 0.0, 1.0, exponent=0.5,
                                          n=int(n)) - exact)
        for n in sizes
    ])
    raw_order = fit_order(1.0 / sizes, raw_errors).order
    substituted_order = fit_order(1.0 / sizes, substituted_errors).order
    assert raw_order == pytest.approx(0.5, abs=0.15)
    assert substituted_order == pytest.approx(2.0, abs=0.2)
    assert substituted_errors[-1] < raw_errors[-1] / 100.0
    with pytest.raises(ValueError):
        integrate_endpoint_singularity(lambda t: np.asarray(t), 0.0, 1.0, exponent=1.5)


# --------------------------------------------------------------------------- sampling
def test_monte_carlo_and_halton_have_the_error_rates_their_theory_predicts() -> None:
    """Monte Carlo at ``N^-1/2`` in every dimension; Halton clearly better in one dimension."""
    report = sampling_error_rate(dimensions=(1, 2), sizes=(1_000, 10_000, 100_000))
    assert report["monte_carlo_exponents"]["1"] == pytest.approx(-0.5, abs=0.25)
    assert report["monte_carlo_exponents"]["2"] == pytest.approx(-0.5, abs=0.25)
    assert report["quasi_monte_carlo_exponents"]["1"] < report["monte_carlo_exponents"]["1"]
    assert report["quasi_monte_carlo_exponents"]["1"] < -1.0
    # the reported numbers are errors of an integral whose exact value is 1, so they start small
    assert all(v < 0.1 for v in report["monte_carlo_errors"]["1"])


def test_monte_carlo_reports_its_own_standard_error() -> None:
    """The standard error must bracket the true error about as often as its definition suggests."""
    exact = 0.5
    errors = []
    estimates = []
    for seed in range(40):
        report = monte_carlo_integrate(lambda p: p[:, 0], [0.0], [1.0], 2000, seed=seed)
        errors.append(abs(report["value"] - exact))
        estimates.append(report["standard_error"])
    errors = np.asarray(errors)
    estimates = np.asarray(estimates)
    # within two standard errors, in the great majority of the trials
    assert np.mean(errors < 2.0 * estimates) > 0.8
    with pytest.raises(ValueError):
        monte_carlo_integrate(lambda p: p[:, 0], [0.0], [1.0], 1)
    with pytest.raises(ValueError):
        quasi_monte_carlo_integrate(lambda p: p[:, 0], [0.0, 0.0], [1.0], 10)


def test_halton_sequence_is_low_discrepancy_and_deterministic() -> None:
    points = halton_sequence(8, 2)
    assert points.shape == (8, 2)
    assert points[:, 0] == pytest.approx([0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625])
    assert np.allclose(points, halton_sequence(8, 2))  # deterministic, unlike Monte Carlo
    # low discrepancy: the first level of a 2-D stratification is almost balanced
    quadrant = np.histogram2d(points[:, 0], points[:, 1], bins=2, range=[[0, 1], [0, 1]])[0]
    assert quadrant.min() >= 1
    with pytest.raises(ValueError):
        halton_sequence(0, 1)


# --------------------------------------------------------------------------- cost
def test_rules_compared_at_equal_cost_show_the_expected_exponents() -> None:
    """Simpson's fourth order and the trapezoid's second are visible in error against evaluations."""
    exact = math.e - 1.0
    comparison = compare_rules_at_equal_cost(np.exp, 0.0, 1.0, exact, budgets=(8, 16, 32, 64, 128, 256))
    trapezoid = comparison.rules["trapezoid"]
    simpson = comparison.rules["simpson"]
    assert trapezoid["error_exponent"] == pytest.approx(-2.0, abs=0.3)
    assert simpson["error_exponent"] == pytest.approx(-4.0, abs=0.4)
    assert simpson["errors"][-1] < trapezoid["errors"][-1]
    # the two spectral rules are at the rounding floor, where no exponent is measurable
    assert comparison.rules["gauss_legendre"]["errors"][-1] < 1e-13
    assert comparison.rules["clenshaw_curtis"]["errors"][-1] < 1e-13
    assert comparison.dominant in ("gauss_legendre", "clenshaw_curtis")
    assert not comparison.diagnostics
    payload = comparison.to_dict()
    assert set(payload["rules"]) == {"trapezoid", "simpson", "gauss_legendre", "clenshaw_curtis"}
