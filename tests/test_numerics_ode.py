"""Tests for ``pinneapple_analysis.numerics.ode`` (#1060).

Every method's order is fitted from its own global error on ``y' = -2y`` (exact solution ``exp(-2t)``),
the stability limits come from the tableau and are checked against the classical numbers (2 for Euler,
2.785 for RK4, unbounded for the A-stable ones), the BDF angles against the published ones, and the cost
of an explicit against an implicit method is measured on Robertson's kinetics by counting evaluations to
reach the same accuracy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis.numerics.ode import (
    BACKWARD_EULER,
    BOGACKI_SHAMPINE,
    DORMAND_PRINCE,
    EULER,
    RADAU_IIA3,
    RK4,
    TRAPEZOID,
    absolute_stability_limit,
    adams_bashforth,
    backward_euler,
    adams_moulton,
    bdf,
    bdf_amplification_roots,
    bdf_angle,
    bogacki_shampine,
    dormand_prince,
    energy_drift_study,
    euler,
    heun,
    implicit_trapezoid,
    integrate_with_events,
    is_a_stable,
    is_l_stable,
    measure_order,
    pendulum_cartesian_index1,
    project_pendulum,
    radau_iia3,
    rk4,
    robertson_rhs,
    stability_function,
    stability_region,
    stiffness_ratio,
    stormer_verlet,
    symplectic_euler,
    van_der_pol_rhs,
)

_DECAY = lambda t, y: -2.0 * y  # noqa: E731
_DECAY_EXACT = lambda t: np.exp(-2.0 * t)  # noqa: E731
_COUNTS = (40, 80, 160, 320)


def _order(method, **kwargs) -> float:
    return float(
        measure_order(method, _DECAY, _DECAY_EXACT, [1.0], (0.0, 2.0), _COUNTS, **kwargs)["order"]
    )


# --------------------------------------------------------------------------- measured orders
@pytest.mark.parametrize(
    ("method", "expected"),
    [(euler, 1.0), (heun, 2.0), (rk4, 4.0), (implicit_trapezoid, 2.0), (radau_iia3, 5.0)],
)
def test_one_step_methods_deliver_their_order(method, expected: float) -> None:
    """Global error against step size: the order is fitted, and it is what the tableau claims."""
    assert _order(method) == pytest.approx(expected, abs=0.15)


def test_radau_iia_tableau_is_the_three_stage_collocation() -> None:
    """Nodes, weights and the stiff-accuracy property, all checked against the construction itself."""
    assert np.allclose(RADAU_IIA3.c, [0.1550510257, 0.6449489743, 1.0], atol=1e-9)
    assert np.allclose(RADAU_IIA3.b, RADAU_IIA3.A[-1, :])  # stiffly accurate: b is the last row of A
    assert RADAU_IIA3.order == 5


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_adams_bashforth_reaches_its_order_despite_the_startup(order: int) -> None:
    """The RK4 startup costs nothing here because it is itself fourth order (the cap is at 4)."""
    assert _order(adams_bashforth, order=order) == pytest.approx(float(order), abs=0.15)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_adams_moulton_reaches_its_order(order: int) -> None:
    """With ``k`` coefficients the corrector is order ``k`` (BE, trapezoid, then 3 and 4).

    The RK4 startup costs nothing at these orders because its own error is ``O(h^4)``; a fifth-order
    corrector would need a fifth-order start, which is the multistep constraint worth knowing.
    """
    corrected = _order(adams_moulton, order=order)
    single_pass = _order(adams_moulton, order=order, iterations=1)
    assert corrected == pytest.approx(float(order), abs=0.2)
    # one corrector pass is the same order with a larger error constant; the fitted exponent can land a
    # few tenths above the order when the two leading error terms have opposite signs
    assert abs(single_pass - float(order)) <= 0.5


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_bdf_reaches_its_order_with_an_exact_start(order: int) -> None:
    """With the history initialised exactly, BDF ``k`` converges at order ``k`` -- up to 5."""
    assert _order(bdf, order=order, startup="exact", exact=_DECAY_EXACT) == pytest.approx(
        float(order), abs=0.2
    )


def test_bdf_started_at_low_order_is_what_a_production_code_gets() -> None:
    """Backward-Euler startup still gives the right order for BDF2 (order 1 start, order 2 method)."""
    assert _order(bdf, order=2) == pytest.approx(2.0, abs=0.2)


# --------------------------------------------------------------------------- stability
def test_stability_limits_on_the_negative_real_axis_are_measured() -> None:
    """Euler 2, RK4 2.785: the numbers every stability chapter quotes, recovered from the tableaux."""
    assert absolute_stability_limit(EULER) == pytest.approx(2.0, abs=1e-6)
    assert absolute_stability_limit(RK4) == pytest.approx(2.7852935634, abs=1e-6)
    # the embedded pairs are the truncated exponential to their order, so their limits are 2.513 and 3.307
    assert absolute_stability_limit(BOGACKI_SHAMPINE) == pytest.approx(2.5127, abs=1e-3)
    assert absolute_stability_limit(DORMAND_PRINCE) == pytest.approx(3.3066, abs=1e-3)
    # the A-stable methods have no limit on that axis at all
    for tableau in (BACKWARD_EULER, TRAPEZOID, RADAU_IIA3):
        assert absolute_stability_limit(tableau) == float("inf")


def test_a_and_l_stability_flags_separate_the_implicit_methods() -> None:
    """The trapezoid rule is A-stable but not L-stable; backward Euler and Radau are both."""
    assert not is_a_stable(EULER) and not is_a_stable(RK4)
    assert is_a_stable(BACKWARD_EULER) and is_l_stable(BACKWARD_EULER)
    assert is_a_stable(TRAPEZOID) and not is_l_stable(TRAPEZOID)
    assert is_a_stable(RADAU_IIA3) and is_l_stable(RADAU_IIA3)


def test_bdf_angles_match_the_published_values() -> None:
    """90, 90, 86.0, 73.4, 51.8 degrees: BDF3 and beyond are only stable close to the real axis."""
    expected = {1: 90.0, 2: 90.0, 3: 86.0, 4: 73.4, 5: 51.8}
    for order, angle in expected.items():
        assert bdf_angle(order) == pytest.approx(angle, abs=0.5)


def test_bdf_root_condition_agrees_with_the_amplification_of_backward_euler() -> None:
    """BDF1 must reproduce ``R(z) = 1/(1-z)``: the two ways of computing stability have to agree."""
    for z in (complex(-1.0, 0.0), complex(-2.0, 1.0), complex(0.5, -0.5)):
        roots = bdf_amplification_roots(z, 1)
        assert abs(roots[0] - 1.0 / (1.0 - z)) < 1e-12


def test_stability_region_sampling_carries_the_same_verdict() -> None:
    region = stability_region(RK4, extent=4.0, resolution=41)
    assert region["limit_on_real_axis"] == pytest.approx(2.7852935634, abs=1e-6)
    assert region["a_stable"] is False
    assert len(region["real"]) == 41 and len(region["imaginary"]) == 41
    assert np.asarray(region["stable"]).any()
    # RK4 is unstable well inside the left half-plane at large imaginary parts
    R = stability_function(RK4)
    assert abs(R(complex(-2.0, 3.0))) > 1.0


# --------------------------------------------------------------------------- adaptive pairs
def test_embedded_pairs_reach_the_requested_tolerance() -> None:
    """The controller drives its own estimate to ``rtol``; the true error must be of that order."""
    exact = math.exp(-2.0)
    for solver, rtol in ((bogacki_shampine, 1e-6), (dormand_prince, 1e-9)):
        result = solver(_DECAY, [1.0], (0.0, 1.0), rtol=rtol)
        assert abs(float(result.y[-1]) - exact) < 100.0 * rtol
        assert result.steps > 5
        assert result.rejected >= 0


def test_dormand_prince_is_cheaper_than_a_fixed_step_rk4_at_equal_accuracy() -> None:
    """Both reach ``1e-9``; the adaptive fifth-order pair does it with far fewer evaluations."""
    exact = math.exp(-2.0)
    adaptive = dormand_prince(_DECAY, [1.0], (0.0, 1.0), rtol=1e-9)
    fixed = rk4(_DECAY, [1.0], (0.0, 1.0), 256)
    assert abs(float(adaptive.y[-1]) - exact) < 1e-8
    assert abs(float(fixed.y[-1]) - exact) < 1e-8
    assert adaptive.function_evaluations < fixed.function_evaluations


# --------------------------------------------------------------------------- stiffness
def test_stiffness_ratio_separates_a_stiff_problem_from_a_benign_one() -> None:
    """Van der Pol at ``mu = 1000`` has a ratio around ``mu`` squared; the decay problem has 1."""
    assert stiffness_ratio(_DECAY, 0.0, [1.0]) == pytest.approx(1.0, abs=1e-6)
    stiff = stiffness_ratio(van_der_pol_rhs(1000.0), 0.0, [2.0, 0.0])
    assert stiff > 1e6


def test_stability_limit_bites_exactly_where_the_linear_analysis_says() -> None:
    """``y' = -1e6 y``: RK4 blows up at ``h|lambda| = 10`` and 100, and survives at 2.8 and 2.5."""
    lhs = lambda t, y: -1e6 * y  # noqa: E731
    for steps, product in ((100, 100.0), (1000, 10.0)):
        blown = rk4(lhs, [1.0], (0.0, 0.01), steps)
        assert not np.all(np.isfinite(blown.y))
        assert "ODE.STABILITY_LIMIT" in [d.code for d in blown.diagnostics]
    for steps, product in ((3600, 2.78), (4000, 2.5)):
        surviving = rk4(lhs, [1.0], (0.0, 0.01), steps)
        assert np.all(np.isfinite(surviving.y))
        assert not surviving.diagnostics
    # backward Euler is stable at every one of those steps, including the ones that blow RK4 up
    for steps in (5, 100):
        assert np.all(np.isfinite(backward_euler(lhs, [1.0], (0.0, 0.01), steps).y))


def test_implicit_costs_less_than_explicit_on_robertson_at_equal_accuracy() -> None:
    """The measured cost argument: BDF2 reaches ``1e-8`` with two orders of magnitude fewer evaluations.

    RK4 cannot take a step larger than ``2.785 / 3e7`` for stability, so its cost is fixed by the fast
    mode and not by the accuracy anybody asked for. BDF2 has no such floor.
    """
    t_span = (0.0, 1e-3)
    reference = bdf(robertson_rhs, [1.0, 0.0, 0.0], t_span, 4000, order=2)

    explicit = rk4(robertson_rhs, [1.0, 0.0, 0.0], t_span, 3600)  # just inside its stability limit
    implicit = bdf(robertson_rhs, [1.0, 0.0, 0.0], t_span, 40, order=2)
    assert np.all(np.isfinite(explicit.y))
    error_explicit = float(np.linalg.norm(explicit.y[-1] - reference.y[-1]))
    error_implicit = float(np.linalg.norm(implicit.y[-1] - reference.y[-1]))
    assert error_explicit < 1e-10
    assert error_implicit < 1e-7
    assert implicit.function_evaluations < explicit.function_evaluations / 20.0


# --------------------------------------------------------------------------- geometric integrators
def _kepler_force(q: np.ndarray) -> np.ndarray:
    radius = float(np.linalg.norm(q))
    return -np.asarray(q, dtype=np.float64) / radius**3


def test_symplectic_integrators_keep_the_energy_bounded_where_rk4_drifts() -> None:
    """Over 200 orbits the envelope of the energy error grows for RK4 and stays flat for Verlet."""
    study = energy_drift_study(
        _kepler_force, lambda q: -1.0 / float(np.linalg.norm(q)), [1.0, 0.0], [0.0, 1.0],
        (0.0, 200.0 * math.pi), 20000,
    )
    verlet = study["methods"]["stormer_verlet"]
    rk4_result = study["methods"]["rk4"]
    assert abs(verlet["envelope_growth"]) < 1e-8  # bounded: no secular trend
    assert rk4_result["envelope_growth"] > 1e-7  # secular: grows with time
    assert rk4_result["envelope_growth"] > 1e3 * abs(verlet["envelope_growth"])
    assert "ODE.ENERGY_DRIFT" in [d["code"] for d in study["diagnostics"]]


def test_stormer_verlet_is_second_order_and_symplectic_euler_first() -> None:
    """The drift maps have their own orders: measure them on a harmonic oscillator against the exact orbit."""
    force = lambda q: -np.asarray(q, dtype=np.float64)  # noqa: E731
    exact = np.cos(1.0)
    sizes = [100, 200, 400, 800]
    verlet_errors = [
        abs(float(stormer_verlet(force, [1.0], [0.0], (0.0, 1.0), n)["q"][-1][0]) - exact)
        for n in sizes
    ]
    euler_errors = [
        abs(float(symplectic_euler(force, [1.0], [0.0], (0.0, 1.0), n)["q"][-1][0]) - exact)
        for n in sizes
    ]
    steps = np.array([1.0 / n for n in sizes])
    from pinneapple_analysis.numerics.convergence import fit_order

    assert fit_order(steps, np.asarray(verlet_errors)).order == pytest.approx(2.0, abs=0.2)
    assert fit_order(steps, np.asarray(euler_errors)).order == pytest.approx(1.0, abs=0.2)


# --------------------------------------------------------------------------- events
def test_event_time_is_located_to_the_tolerance() -> None:
    """A free-falling body released at ``y = 1`` hits ``y = 0`` at ``sqrt(2/g)``; the event must find it."""
    gravity = 9.81
    rhs = lambda t, y: np.array([y[1], -gravity])  # noqa: E731
    event = lambda t, y: float(y[0])  # noqa: E731
    result = integrate_with_events(rhs, [1.0, 0.0], (0.0, 1.0), 20, event, tolerance=1e-12)
    assert len(result.events) == 1
    assert result.events[0]["time"] == pytest.approx(math.sqrt(2.0 / gravity), abs=1e-9)
    assert "ODE.EVENT" in [d.code for d in result.diagnostics]


# --------------------------------------------------------------------------- index-1 DAE
def test_index_reduced_pendulum_keeps_the_constraint_to_integration_error() -> None:
    """Index reduction turns the DAE into an ODE whose constraint drifts -- and the drift is measurable."""
    rhs = pendulum_cartesian_index1(1.0, 9.81)
    result = rk4(rhs, [1.0, 0.0, 0.0, 0.0], (0.0, 10.0), 20000)
    states = result.y
    radii = np.hypot(states[:, 0], states[:, 1])
    drift = float(np.max(np.abs(radii - 1.0)))
    assert drift < 1e-9  # RK4 at this resolution keeps it tight
    # a coarser run drifts more (the drift is the integration error, not a modelling term)
    coarse = rk4(rhs, [1.0, 0.0, 0.0, 0.0], (0.0, 10.0), 500)
    assert float(np.max(np.abs(np.hypot(coarse.y[:, 0], coarse.y[:, 1]) - 1.0))) > drift

    projected = np.array([project_pendulum(state) for state in coarse.y])
    assert np.allclose(np.hypot(projected[:, 0], projected[:, 1]), 1.0, atol=1e-14)
    # the projection also removes the radial velocity, which is what the constraint implies
    radial_velocity = np.abs(projected[:, 0] * projected[:, 2] + projected[:, 1] * projected[:, 3])
    assert float(np.max(radial_velocity)) < 1e-12


def test_pendulum_energy_is_conserved_by_the_index_reduced_form() -> None:
    """The reduced ODE has no dissipation term, so the energy must stay constant to integration error."""
    gravity, length = 9.81, 1.0
    rhs = pendulum_cartesian_index1(length, gravity)
    result = rk4(rhs, [1.0, 0.0, 0.0, 0.0], (0.0, 5.0), 10000)
    states = result.y
    energy = 0.5 * (states[:, 2] ** 2 + states[:, 3] ** 2) + gravity * states[:, 1]
    initial = energy[0]
    # the pendulum starts at rest at the bottom, where this energy is zero: the scale of the conserved
    # quantity is therefore g*L (the potential at the top), not the initial value
    assert float(np.max(np.abs(energy - initial))) / (gravity * length) < 1e-9


def test_bdf_and_radau_reject_bad_arguments() -> None:
    with pytest.raises(ValueError):
        bdf(_DECAY, [1.0], (0.0, 1.0), 10, order=6)
    with pytest.raises(ValueError):
        bdf(_DECAY, [1.0], (0.0, 1.0), 10, order=2, startup="exact")
    with pytest.raises(ValueError):
        bdf(_DECAY, [1.0], (0.0, 1.0), 10, order=2, startup="magic")
    with pytest.raises(ValueError):
        adams_bashforth(_DECAY, [1.0], (0.0, 1.0), 10, order=5)
    with pytest.raises(ValueError):
        van_der_pol_rhs(0.0)
    with pytest.raises(ValueError):
        pendulum_cartesian_index1(length=0.0)
    with pytest.raises(ValueError):
        project_pendulum(np.array([0.0, 0.0, 0.0, 0.0]))
    assert DORMAND_PRINCE.order == 5 and BOGACKI_SHAMPINE.order == 3
