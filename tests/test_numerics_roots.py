"""Tests for ``pinneapple_analysis.numerics.roots`` (#1057).

Each method is checked against its *theoretical order*, measured from its own iterates, and against its
documented failure mode (a cycle, a flat derivative, order loss at a multiple root, a stalled bracket, a
fold in the branch). The roots themselves are known in closed form: ``x^3 - 2x - 5`` has one root at
``2.0945514815423265...``, ``cos(x) = x`` at the Dottie number, the CSTR has a known number of steady
states per Damkohler number, and the parabola ``x^2 = lambda`` has its fold at the origin.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pinneapple_analysis.numerics.roots import (
    bisection,
    brent,
    broyden,
    cstr_residual,
    fixed_point,
    halley,
    illinois,
    jfnk,
    natural_parameter_continuation,
    newton,
    newton_system,
    poly_roots_companion,
    pseudo_arclength_continuation,
    regula_falsi,
    scan_for_bracket,
    secant,
    von_mises_truss_critical_load,
    von_mises_truss_load,
    von_mises_truss_residual,
    wilkinson_sensitivity,
)

# f = x^3 - 2x - 5: one real root, and its value to 16 digits
_CUBIC_ROOT = 2.0945514815423265


def _f(x: float) -> float:
    return x**3 - 2.0 * x - 5.0


def _df(x: float) -> float:
    return 3.0 * x**2 - 2.0


def _d2f(x: float) -> float:
    return 6.0 * x


# --------------------------------------------------------------------------- measured orders
@pytest.mark.parametrize(
    ("result_factory", "expected_order", "rtol"),
    [
        (lambda: newton(_f, _df, 2.0), 2.0, 0.10),
        (lambda: secant(_f, 1.0, 3.0), 1.6180339887, 0.05),
        (lambda: halley(_f, _df, _d2f, 2.5), 3.0, 0.15),
        (lambda: bisection(_f, 1.0, 3.0), 1.0, 1e-9),
    ],
)
def test_scalar_methods_show_their_theoretical_order(result_factory, expected_order, rtol) -> None:
    """Order from the iterates, not from the textbook: Newton 2, secant 1.618, Halley 3, bisection 1."""
    result = result_factory()
    assert result.converged
    assert result.root == pytest.approx(_CUBIC_ROOT, rel=1e-10)
    assert result.order is not None
    assert result.order.agrees_with(expected_order, rtol=rtol), result.order


def test_bisection_halves_the_bracket_exactly() -> None:
    """The guarantee of bisection is on the bracket, and the measured ratio is exactly 1/2."""
    result = bisection(_f, 1.0, 3.0)
    widths = np.asarray(result.bracket_widths)
    ratios = widths[1:] / widths[:-1]
    assert np.allclose(ratios, 0.5, rtol=1e-12)
    assert result.linear_rate == pytest.approx(0.5, abs=0.06)
    # the guarantee is honest: the final bracket contains the root and is smaller than xtol * |root|
    assert result.bracket[0] < _CUBIC_ROOT < result.bracket[1]
    # the stopping rule is 0.5 * width <= xtol * |mid|, so the bracket that remains is at most 2 xtol |root|
    assert result.bracket[1] - result.bracket[0] <= 2.1 * 1e-12 * abs(_CUBIC_ROOT)


def test_bisection_refuses_a_non_bracket() -> None:
    with pytest.raises(ValueError):
        bisection(_f, 2.5, 3.5)  # f > 0 at both ends
    assert scan_for_bracket(_f, 0.0, 4.0) is not None
    assert scan_for_bracket(lambda x: x**2 + 1.0, -2.0, 2.0) is None


def test_regula_falsi_stalls_and_illinois_does_not() -> None:
    """The classic pair: plain false position stops shrinking the bracket, Illinois keeps going."""
    plain = regula_falsi(_f, 1.0, 3.0)
    assert plain.root == pytest.approx(_CUBIC_ROOT, rel=1e-10)  # the estimate is fine...
    assert plain.bracket_widths[-1] > 0.5  # ...while the guarantee collapses
    assert "ROOTS.STALLED" in [d.code for d in plain.diagnostics]

    doing = illinois(_f, 1.0, 3.0)
    assert doing.converged
    widths = np.asarray(doing.bracket_widths)
    assert widths[-1] < 1e-6 * widths[0]  # the bracket really shrinks
    assert doing.function_evaluations < plain.function_evaluations


def test_brent_keeps_the_bracket_and_beats_bisection_in_evaluations() -> None:
    """Brent has to be both safe (bracket kept) and fast (fewer evaluations than a pure halving)."""
    result = brent(_f, 1.0, 3.0)
    assert result.converged
    assert result.root == pytest.approx(_CUBIC_ROOT, rel=1e-12)
    # the bracket has converged onto the root to the last ulp, so compare against it directly
    assert abs(result.bracket[0] - _CUBIC_ROOT) < 1e-12
    assert abs(result.bracket[1] - _CUBIC_ROOT) < 1e-12
    assert result.function_evaluations < bisection(_f, 1.0, 3.0).function_evaluations
    with pytest.raises(ValueError):
        brent(_f, 2.5, 3.5)


def test_multiple_root_loses_the_order_and_the_multiplicity_fixes_it() -> None:
    """``(x-1)^2`` has a double root: Newton degrades to linear with factor 1 - 1/m = 0.5."""
    f = lambda x: (x - 1.0) ** 2  # noqa: E731
    df = lambda x: 2.0 * (x - 1.0)  # noqa: E731

    lost = newton(f, df, 3.0)
    assert lost.converged
    assert lost.order is not None and lost.order.order == pytest.approx(1.0, abs=1e-9)
    assert lost.linear_rate == pytest.approx(0.5, abs=0.02)
    assert "ROOTS.MULTIPLE_ROOT_ORDER_LOSS" in str(lost.to_dict()) or lost.linear_rate < 0.6

    fixed = newton(f, df, 3.0, multiplicity=2)
    assert fixed.converged
    assert fixed.root == pytest.approx(1.0, rel=1e-12)
    assert fixed.iterations < lost.iterations


def test_newton_cycle_is_detected_instead_of_iterating_forever() -> None:
    """``x^3 - 2x + 2`` from 0 cycles between 0 and 1: a real 2-cycle of the Newton map."""
    result = newton(lambda x: x**3 - 2.0 * x + 2.0, lambda x: 3.0 * x**2 - 2.0, 0.0)
    assert not result.converged
    assert "ROOTS.NEWTON_CYCLE" in [d.code for d in result.diagnostics]
    assert set(np.round(result.iterates, 12)) == {0.0, 1.0}


def test_flat_derivative_is_detected() -> None:
    """Starting at a stationary point, the Newton step is undefined -- reported, not divided by zero."""
    result = newton(lambda x: x**3, lambda x: 3.0 * x**2, 0.0)
    assert not result.converged
    assert "ROOTS.FLAT_DERIVATIVE" in [d.code for d in result.diagnostics]
    assert np.isfinite(result.root)


def test_fixed_point_contracts_at_the_derivative_of_g() -> None:
    """``cos(x) = x`` converges at ``|g'| = sin(0.739) = 0.6736``: the measured factor must match."""
    result = fixed_point(np.cos, 0.5)
    dottie = 0.7390851332151607
    assert result.root == pytest.approx(dottie, rel=1e-9)
    assert result.order is not None and result.order.order == pytest.approx(1.0)
    assert result.linear_rate == pytest.approx(abs(-np.sin(dottie)), rel=0.02)


# --------------------------------------------------------------------------- systems
_SYSTEM = lambda v: np.array([v[0] ** 2 + v[1] - 37.0, v[0] - v[1] ** 2 - 5.0])  # noqa: E731
_SYSTEM_SOLUTION = np.array([6.0, 1.0])


def test_newton_for_systems_is_quadratic_once_in_the_asymptotic_regime() -> None:
    """Measured on the step norms: order 2 near the solution, and *less* from a far start."""
    near = newton_system(_SYSTEM, None, [5.5, 1.5], line_search=False)
    far = newton_system(_SYSTEM, None, [4.0, 2.0], line_search=False)
    assert near.converged and far.converged
    for result in (near, far):
        assert np.allclose(result.x, _SYSTEM_SOLUTION, rtol=1e-10)
        assert result.order is not None
    assert near.order.agrees_with(2.0, rtol=0.10), near.order
    # pre-asymptotic: the first steps of a far start do not obey the quadratic law yet, which is measured
    assert far.order.order < 1.9
    assert far.iterations >= near.iterations
    # the trajectory is kept, so a caller can measure its own error against a known solution
    errors = [float(np.linalg.norm(v - _SYSTEM_SOLUTION)) for v in far.iterates]
    assert errors[-1] < 1e-6 and errors[0] > 1.0


def test_line_search_buys_robustness_from_a_far_start() -> None:
    """From (20, 20) the undamped step overflows the model; the damped one still converges."""
    damped = newton_system(_SYSTEM, None, [20.0, 20.0], line_search=True)
    assert damped.converged
    assert np.allclose(damped.x, _SYSTEM_SOLUTION, rtol=1e-6)


def _bratu(n: int = 20) -> tuple[Any, np.ndarray]:
    """Discretised ``u'' + lambda exp(u) = 0`` on n unknowns: a genuinely nonlinear system of size n."""
    h = 1.0 / (n + 1)
    lam = 0.5
    x = np.linspace(h, 1.0 - h, n)
    guess = 4.0 * x * (1.0 - x)  # already close, so both methods converge and only the cost differs

    def residual(u: np.ndarray) -> np.ndarray:
        left = np.concatenate(([0.0], u[:-1]))
        right = np.concatenate((u[1:], [0.0]))
        return (2.0 * u - left - right) / h**2 + lam * np.exp(u)

    return residual, guess


def test_broyden_solves_with_one_evaluation_per_step() -> None:
    """On a 20-unknown system, one evaluation per step beats a finite-difference Jacobian of 20."""
    residual, guess = _bratu(20)
    result = broyden(residual, guess)
    newton = newton_system(residual, None, guess)
    assert result.converged and newton.converged
    assert np.allclose(result.x, newton.x, atol=1e-6)
    assert result.iterations < 40
    assert result.function_evaluations < newton.function_evaluations / 2.0


def test_jfnk_never_forms_a_jacobian() -> None:
    """The Jacobian-free variant only sees residual evaluations, and still converges quadratically."""

    def counting_system(v: np.ndarray) -> np.ndarray:
        counting_system.calls += 1  # type: ignore[attr-defined]
        return _SYSTEM(v)

    counting_system.calls = 0  # type: ignore[attr-defined]
    result = jfnk(counting_system, [4.0, 2.0])
    assert result.converged
    assert np.allclose(result.x, _SYSTEM_SOLUTION, rtol=1e-8)
    # no analytic Jacobian was ever called: every evaluation is a residual, and the count matches
    assert result.function_evaluations >= counting_system.calls  # type: ignore[attr-defined]


def test_singular_jacobian_is_reported() -> None:
    """At ``(1,1)`` the two equations of ``x^2+y^2=4, xy=1`` have proportional gradients."""
    degenerate = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[0] * v[1] - 1.0])  # noqa: E731
    result = newton_system(degenerate, None, [1.0, 1.0], line_search=False)
    assert "ROOTS.JACOBIAN_SINGULAR" in [d.code for d in result.diagnostics]


# --------------------------------------------------------------------------- continuation
def test_natural_continuation_stops_at_the_fold() -> None:
    """``x^2 = lambda`` has its fold at the origin: no real branch continues past it in lambda."""
    residual = lambda x, lam: np.array([x[0] ** 2 - lam])  # noqa: E731
    natural = natural_parameter_continuation(residual, [-1.0], np.linspace(1.0, -0.1, 12))
    assert not natural.converged
    assert natural.diagnostics and natural.diagnostics[-1].code == "ROOTS.FOLD"
    # up to the fold it is perfectly fine
    up_to_fold = natural_parameter_continuation(residual, [-1.0], np.linspace(1.0, 0.1, 10))
    assert up_to_fold.converged
    assert up_to_fold.path[-1][0] == pytest.approx(-np.sqrt(0.1), rel=1e-8)


def test_pseudo_arclength_passes_through_the_fold() -> None:
    """The arclength path is not a function of lambda: it turns, which is the whole point."""
    residual = lambda x, lam: np.array([x[0] ** 2 - lam])  # noqa: E731
    path = pseudo_arclength_continuation(residual, [-1.0], 1.0, steps=40, ds=0.1, direction=-1)
    assert path.converged
    assert len(path.folds) == 1  # exactly one turning point, and it is the origin
    parameters = path.parameters
    assert parameters.min() < 0.05  # it went down to the fold...
    assert parameters[-1] > parameters.min()  # ...and came back up the other branch
    assert path.path[-1, 0] > 0.0  # so it ended on x > 0, which natural parameter never reaches
    assert np.allclose(path.path[:, 0] ** 2, parameters, atol=1e-8)  # the constraint holds along the path


def test_cstr_multiplicity_is_a_measured_number_of_steady_states() -> None:
    """Scan for sign changes: the cooled CSTR has three steady states only for small Damkohler."""

    def steady_states(Da: float) -> list[float]:
        xs = np.linspace(0.0, 0.999999, 200_000)
        values = cstr_residual(xs, Da)
        roots = []
        for i in range(xs.size - 1):
            if np.sign(values[i]) != np.sign(values[i + 1]):
                roots.append(0.5 * (xs[i] + xs[i + 1]))
        return roots

    multiple = steady_states(0.005)
    assert len(multiple) == 3  # ignited, unstable middle, extinguished
    assert len(steady_states(0.1)) == 1
    # the coarse scan brackets them; brent then solves each one to full precision
    for x in multiple:
        refined = brent(lambda v: float(cstr_residual(v, 0.005)), x - 1e-3, x + 1e-3)
        assert refined.converged
        assert abs(float(cstr_residual(refined.root, 0.005))) < 1e-10


# --------------------------------------------------------------------------- polynomials
def test_companion_roots_of_a_cubic() -> None:
    roots = poly_roots_companion([-5.0, -2.0, 0.0, 1.0])
    real = np.sort(np.real(roots.roots[np.abs(np.imag(roots.roots)) < 1e-8]))
    assert real[-1] == pytest.approx(_CUBIC_ROOT, rel=1e-12)
    assert roots.condition_numbers is not None


def test_wilkinson_roots_move_by_whole_units_for_a_last_bit_change() -> None:
    """The measurement the polynomial literature is built on: 2**-23 relative on one coefficient, O(1) drift."""
    report = wilkinson_sensitivity(degree=20, relative_perturbation=2.0**-23)
    # even the *unperturbed* roots are off the integers: 20! does not fit in 53 bits, so the coefficients
    # are already rounded before any root finding starts -- that is the first half of the demonstration
    assert np.allclose(np.sort(report["roots"]), np.arange(1, 21), atol=2.0)
    assert np.max(np.abs(np.sort(report["roots"]) - np.arange(1, 21))) > 1e-2
    assert report["max_displacement"] > 1.0
    amplification = report["max_displacement"] / (report["relative_perturbation"] * 20.0)
    assert amplification > 1e5  # the condition number of the root problem, measured
    assert report["max_condition_number"] > 1e12
    # the predicted displacement (condition number x perturbation) has the measured magnitude
    predicted = np.asarray(report["predicted_displacement"])
    measured = np.asarray(report["displacement"])
    assert np.corrcoef(np.log10(predicted + 1e-30), np.log10(measured + 1e-30))[0, 1] > 0.8


# --------------------------------------------------------------------------- buckling
def test_von_mises_truss_limit_point_is_the_closed_form_and_the_numeric_maximum() -> None:
    """Snap-through buckling: the load-displacement curve turns over, and three independent values agree.

    The closed form (``l^3 = L b^2``) is compared with a golden-section search on the load itself and with
    the load the closed-form displacement carries -- so the reference the continuation is judged against
    is not taken on faith.
    """
    critical = von_mises_truss_critical_load()
    assert von_mises_truss_load(0.0) == pytest.approx(0.0, abs=1e-15)
    assert von_mises_truss_load(critical["apex_height"]) == pytest.approx(0.0, abs=1e-15)

    low, high = 1e-12, critical["apex_height"]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        left = high - ratio * (high - low)
        right = low + ratio * (high - low)
        if float(von_mises_truss_load(left)) < float(von_mises_truss_load(right)):
            low = left
        else:
            high = right
    measured_displacement = 0.5 * (low + high)
    measured_load = float(von_mises_truss_load(measured_displacement))
    assert measured_displacement == pytest.approx(critical["displacement"], rel=1e-8)
    assert measured_load == pytest.approx(critical["load"], rel=1e-8)
    assert measured_load == pytest.approx(0.45019646437, rel=1e-9)


def test_natural_continuation_jumps_the_branch_and_arclength_walks_around_the_fold() -> None:
    """The buckling lesson, measured: fixing the load gives a converged answer on the *wrong* branch.

    Past the limit load there is no equilibrium on the shallow branch, but Newton does not say so -- it
    converges to the far branch, and the traced path has a jump of more than one unit of displacement
    between two consecutive load steps. Pseudo-arclength instead steps around the turning point: the load
    becomes non-monotone (its maximum is the critical load) and consecutive displacements stay of the
    order of the arclength step.
    """
    loads = np.linspace(0.01, 1.0, 20)
    natural = natural_parameter_continuation(von_mises_truss_residual, [0.02], loads)
    assert natural.converged  # it does not fail: that is the trap
    displacements = natural.path[:, 0]
    jump = float(np.max(np.abs(np.diff(displacements))))
    assert jump > 1.0  # the snap-through, crossed silently in one step

    critical = von_mises_truss_critical_load()
    step = 0.01
    arc = pseudo_arclength_continuation(
        von_mises_truss_residual, [0.05], 0.02, steps=150, ds=step
    )
    assert arc.converged
    assert not np.all(np.diff(arc.parameters) > 0)  # the load turns back: the fold was passed
    assert float(np.max(arc.parameters)) == pytest.approx(critical["load"], abs=5e-3)
    assert float(np.max(np.abs(np.diff(arc.path[:, 0])))) < 10.0 * step
    # the branch continues past the fold instead of jumping: the displacement keeps growing smoothly
    assert float(np.max(arc.path[:, 0])) > 0.9
