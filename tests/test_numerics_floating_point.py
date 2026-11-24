"""Tests for ``pinneapple_analysis.numerics.floating_point`` (#1056).

The verification strategy is the one the issue asks for: every claim is checked against a reference that
is *not* the algorithm under test -- the definition of epsilon (the halving loop), ``decimal`` at 60 digits
for the quadratic roots, ``math.fsum`` (exactly rounded) for summation and the variance, and ``mpmath``
where an independent arbitrary-precision reference is useful.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis.numerics.floating_point import (
    Interval,
    digits_lost,
    error_budget,
    interval_quadratic_roots,
    kahan_sum,
    log1p_expm1_comparison,
    machine_epsilon,
    machine_parameters,
    measured_condition_number,
    naive_sum,
    neumaier_sum,
    pairwise_sum,
    quadratic_roots,
    quadratic_roots_reference,
    relative_error,
    residual_precision_floor,
    summation_error_growth,
    ulp,
    variance_comparison,
    variance_naive,
    variance_welford,
)


# --------------------------------------------------------------------------- dtype parameters
@pytest.mark.parametrize(
    ("dtype", "expected_eps"),
    [("float16", 2.0**-10), ("float32", 2.0**-23), ("float64", 2.0**-52)],
)
def test_machine_epsilon_is_measured_by_its_definition(dtype: str, expected_eps: float) -> None:
    """``eps`` is the smallest value with ``1 + eps > 1``: the halving loop must return exactly that."""
    eps = machine_epsilon(dtype)
    assert eps == expected_eps
    one = np.dtype(dtype).type(1.0)
    assert np.dtype(dtype).type(one + np.dtype(dtype).type(eps)) > one
    assert np.dtype(dtype).type(one + np.dtype(dtype).type(eps / 2.0)) == one


def test_machine_parameters_match_the_dtype_definition() -> None:
    """The range is numpy's, the epsilon and the subnormal floor are measured here."""
    for dtype in ("float16", "float32", "float64"):
        params = machine_parameters(dtype)
        info = np.finfo(dtype)
        assert params.max_finite == pytest.approx(float(info.max))
        assert params.min_subnormal == pytest.approx(float(np.nextafter(np.dtype(dtype).type(0.0),
                                                                       np.dtype(dtype).type(1.0))))
        assert params.ulp_at_one == params.eps
        assert params.decimal_digits == pytest.approx(-math.log10(params.eps))
    # float16 carries three decimal digits and float64 sixteen: the ratio is what mixed precision costs
    assert machine_parameters("float16").decimal_digits < 4.0
    assert machine_parameters("float64").decimal_digits > 15.0


def test_ulp_scales_with_the_magnitude() -> None:
    """``ulp(x) = eps * 2**floor(log2|x|)``: the same relative spacing in every binade."""
    assert ulp(1.0) == pytest.approx(2.0**-52)
    assert ulp(1024.0) == pytest.approx(2.0**-52 * 1024.0)
    assert ulp(0.5) == pytest.approx(2.0**-53)
    assert ulp(np.inf) != ulp(np.inf)  # NaN for a non-finite input


# --------------------------------------------------------------------------- cancellation
def test_stable_quadratic_recovers_the_small_root_the_naive_formula_loses() -> None:
    """``x^2 - 1e8 x + 1``: the small root is 1e-8, and the naive formula cancels every digit of it."""
    naive = quadratic_roots(1.0, -1e8, 1.0, method="naive")
    stable = quadratic_roots(1.0, -1e8, 1.0, method="stable")
    assert naive.relative_errors[1] > 1e-3
    assert stable.relative_errors[1] < 1e-15
    assert stable.relative_errors[0] < 1e-15
    # Vieta: the product of the roots must be c/a in both cases (the stable form gets it by construction)
    assert stable.roots[0] * stable.roots[1] == pytest.approx(1.0, rel=1e-15)
    assert "FLOAT.CANCELLATION" in [d.code for d in naive.diagnostics]
    assert naive.diagnostics and not stable.diagnostics


def test_quadratic_reference_is_correct_at_50_digits() -> None:
    """The ``decimal`` reference must agree with ``mpmath`` -- the reference is not the algorithm."""
    mpmath = pytest.importorskip("mpmath")
    a, b, c = 1.0, -1e8, 1.0
    reference = quadratic_roots_reference(a, b, c, digits=50)
    with mpmath.workdps(60):
        disc = mpmath.sqrt(mpmath.mpf(b) ** 2 - 4 * mpmath.mpf(a) * mpmath.mpf(c))
        r_small = (mpmath.mpf(-b) - disc) / (2 * mpmath.mpf(a))
    assert relative_error(reference[1], float(r_small)) < 1e-25


def test_quadratic_rejects_a_negative_discriminant() -> None:
    with pytest.raises(ValueError):
        quadratic_roots(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        quadratic_roots(0.0, 1.0, 1.0)


def test_log1p_and_expm1_recover_what_the_naive_forms_discard() -> None:
    """For ``x`` at the rounding level of 1, ``log(1+x)`` returns 0 and ``exp(x)-1`` returns 0."""
    result = log1p_expm1_comparison(1e-16)
    assert result["log1p_naive"] == 0.0
    assert result["log1p_naive_error"] == pytest.approx(1e-16)
    assert result["log1p_stable_error"] == 0.0
    assert result["expm1_naive"] == 0.0
    assert result["expm1_stable_error"] == 0.0
    # away from the cancellation the two forms agree, which is why the naive one is dangerous
    far = log1p_expm1_comparison(1.0)
    assert far["log1p_naive_error"] < 1e-15


def test_welford_beats_the_naive_one_pass_variance() -> None:
    """Data with a mean of 1e6 and a variance of 1e-6: the shortcut loses every digit, Welford keeps 9."""
    rng = np.random.default_rng(0)
    values = 1e6 + rng.normal(0.0, 1e-3, 500)
    comparison = variance_comparison(values)
    assert comparison.naive_relative_error > 0.1
    assert comparison.welford_relative_error < 1e-6
    assert comparison.reference > 0.0
    assert "FLOAT.CANCELLATION" in [d.code for d in comparison.diagnostics]
    # the two agree to a few digits on benign data, so the failure is the data, not the formula
    benign = variance_comparison(rng.normal(0.0, 1.0, 500))
    assert benign.naive_relative_error < 1e-12


def test_variance_of_a_constant_sample_is_zero_in_both() -> None:
    assert variance_naive([3.0, 3.0, 3.0]) == pytest.approx(0.0, abs=1e-15)
    assert variance_welford([3.0, 3.0, 3.0]) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        variance_welford([])


# --------------------------------------------------------------------------- summation
def test_compensated_summation_matches_the_exactly_rounded_sum() -> None:
    """``fsum`` is exact, so the distance to it is the error caused by the order of the additions."""
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, 20_000)
    exact = math.fsum(float(v) for v in values)
    errors = {
        "naive": relative_error(naive_sum(values), exact),
        "pairwise": relative_error(pairwise_sum(values), exact),
        "kahan": relative_error(kahan_sum(values), exact),
        "neumaier": relative_error(neumaier_sum(values), exact),
    }
    assert errors["naive"] > 10.0 * max(errors["kahan"], errors["neumaier"], 1e-17)
    assert errors["kahan"] < 1e-15
    assert errors["neumaier"] < 1e-15
    assert errors["pairwise"] < errors["naive"]


def test_naive_summation_error_grows_with_more_terms_and_compensation_does_not() -> None:
    """The measured exponent is the point: ~1/2 for naive (a random walk of roundings), ~0 when compensated."""
    growth = summation_error_growth((10, 100, 1_000, 10_000, 100_000))
    assert growth.growth_exponents["naive"] > 0.25
    assert growth.growth_exponents["naive"] < 1.0
    # compensated sums sit at the floor, where the measured exponent is NaN ("no growth detectable")
    assert not (growth.growth_exponents["kahan"] > 0.1)
    assert not (growth.growth_exponents["neumaier"] > 0.1)
    assert growth.naive_error[-1] > growth.naive_error[1]
    assert growth.sizes[-1] == 100_000


def test_summation_of_an_empty_sequence_is_zero() -> None:
    assert naive_sum([]) == 0.0
    assert pairwise_sum([]) == 0.0
    assert kahan_sum([]) == 0.0
    assert neumaier_sum([]) == 0.0


# --------------------------------------------------------------------------- conditioning
def test_measured_condition_number_matches_the_analytic_one() -> None:
    """``cond(x/(1+x)) = 1/(1+x)`` and ``cond(atan) = x/((1+x^2) atan x)``: both are measured here."""
    assert measured_condition_number(lambda x: x / (1.0 + x), 1.0) == pytest.approx(0.5, rel=1e-5)
    assert measured_condition_number(np.arctan, 1.0) == pytest.approx(
        1.0 / (2.0 * math.atan(1.0)), rel=1e-5
    )
    assert measured_condition_number(math.exp, 10.0) == pytest.approx(10.0, rel=1e-4)
    with pytest.raises(ValueError):
        measured_condition_number(math.exp, 0.0)


def test_error_budget_names_the_responsible_party() -> None:
    """The verdict is the actionable part: reformulate the problem, or replace the algorithm."""
    benign = error_budget(condition_number=2.0, forward_error_rel=1e-16, backward_error_rel=1e-16)
    assert benign.verdict == "well-conditioned"

    ill_conditioned = error_budget(condition_number=1e12, forward_error_rel=1e-4, backward_error_rel=1e-16)
    assert ill_conditioned.verdict == "problem"
    assert "FLOAT.PROBLEM_ILL_CONDITIONED" in [d.code for d in ill_conditioned.diagnostics]

    unstable = error_budget(condition_number=10.0, forward_error_rel=1e-4, backward_error_rel=1e-16)
    assert unstable.verdict == "algorithm"
    assert "FLOAT.ALGORITHM_UNSTABLE" in [d.code for d in unstable.diagnostics]

    hopeless = error_budget(condition_number=1e20, forward_error_rel=1.0, backward_error_rel=1e-16)
    assert "FLOAT.NO_SIGNIFICANT_DIGITS" in [d.code for d in hopeless.diagnostics]
    assert hopeless.digits_lost == pytest.approx(0.0, abs=1e-12)


def test_error_budget_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        error_budget(-1.0, 1e-16, 1e-16)


def test_digits_lost_is_the_negative_log_of_the_relative_error() -> None:
    assert digits_lost(1.0, 1.0) == float("inf")
    assert digits_lost(1.0 + 1e-8, 1.0) == pytest.approx(8.0, abs=0.01)
    assert relative_error(0.0, 0.0) == 0.0


# --------------------------------------------------------------------------- intervals
def test_interval_bounds_contain_the_exact_result() -> None:
    """Outward rounding makes the enclosure a guarantee, not an estimate: check it over many trials."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        a, b = rng.uniform(-1e6, 1e6, 2)
        product = Interval.point(a) * Interval.point(b)
        assert product.contains(a * b)
        if b != 0.0:
            assert (Interval.point(a) / Interval.point(b)).contains(a / b)
    # and the width of the result is the rounding of the operation, not more
    assert (Interval.point(1e10) * Interval.point(1e-10)).relative_width < 1e-15


def test_interval_arithmetic_bounds_a_sqrt_and_refuses_a_zero_divisor() -> None:
    root = Interval.point(2.0).sqrt()
    assert root.contains(math.sqrt(2.0))
    with pytest.raises(ZeroDivisionError):
        Interval.point(1.0) / Interval(-1.0, 1.0)
    with pytest.raises(ValueError):
        Interval(1.0, 0.0)
    with pytest.raises(ValueError):
        Interval(-1.0, 4.0).sqrt()


def test_interval_quadratic_enclosure_widens_exactly_where_the_formula_is_unstable() -> None:
    """The small root's enclosure is wide because the naive formula cannot determine it -- that is the point."""
    plus, minus = interval_quadratic_roots(1.0, -1e8, 1.0)
    reference = quadratic_roots_reference(1.0, -1e8, 1.0)
    assert plus.contains(reference[0])
    assert minus.contains(reference[1])
    # the large root is known to ~15 digits, the small one to a handful
    assert plus.relative_width < 1e-14
    assert minus.relative_width > 1e-4
    with pytest.raises(ValueError):
        interval_quadratic_roots(1.0, 0.0, 1.0)


# --------------------------------------------------------------------------- precision floor
def test_residual_floor_separates_float32_from_float64() -> None:
    """A residual that is mathematically zero shows the dtype's floor: float32 sits ~1e8 above float64.

    ``(x^3-1)/(x-1) - (x^2+x+1) = 0`` exactly in real arithmetic; what is left is the rounding of two
    algebraically identical expressions, which is precisely the situation of a PINN residual.
    """

    def residual(x: np.ndarray) -> np.ndarray:
        return (x**3 - np.asarray(1.0, dtype=x.dtype)) / (x - np.asarray(1.0, dtype=x.dtype)) - (
            x**2 + x + np.asarray(1.0, dtype=x.dtype)
        )

    xs = np.linspace(1.1, 2.0, 40)
    floor32 = residual_precision_floor(residual, xs, dtype="float32")
    floor64 = residual_precision_floor(residual, xs, dtype="float64")
    assert floor32.max_abs_residual > 1e3 * floor64.max_abs_residual
    assert floor64.max_abs_residual < 1e-13
    assert floor32.at_floor
    assert floor32.digits_lost_vs_float64 == pytest.approx(9.0, abs=0.5)
    assert "FLOAT.MIXED_PRECISION_RESIDUAL" in [d.code for d in floor32.diagnostics]
    assert not floor64.diagnostics
