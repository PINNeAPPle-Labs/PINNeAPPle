"""Tests for ``pinneapple_analysis.numerics.pde`` (#1062).

Consistency, stability and convergence are measured separately on the same schemes: the truncation error
by substituting the exact solution into the discrete operator, the stability by the von Neumann
amplification factor (with the CFL limit found by bisection and the blow-up then demonstrated just above
it), and the convergence by the observed order against the exact solution. The schemes are then compared
where they differ: upwind's first order and numerical diffusion against Lax-Wendroff's second order and
dispersion, and a TVD limiter that does not increase the total variation where Lax-Wendroff does.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis.numerics.pde import (
    advection_1d,
    advection_convergence,
    advection_stability_limit,
    heat_1d_crank_nicolson,
    heat_1d_exact_mode,
    heat_1d_explicit,
    heat_cfl_limit,
    heat_convergence,
    lax_equivalence_demo,
    manufactured_solution_study,
    max_amplification,
    numerical_diffusion_coefficient,
    poisson_1d,
    poisson_2d,
    total_variation,
    truncation_error_study,
    von_neumann_amplification,
    wave_1d,
)


# --------------------------------------------------------------------------- von Neumann
def test_amplification_factors_reproduce_their_formulas() -> None:
    """The analytic factors at ``theta = pi`` and ``theta = 0``: no scheme-specific magic anywhere."""
    assert float(von_neumann_amplification("heat_explicit", np.pi, r=0.4)) == pytest.approx(1.0 - 1.6)
    assert float(von_neumann_amplification("heat_explicit", 0.0, r=0.4)) == pytest.approx(1.0)
    # upwind at theta = pi: |1 - 2C|, which is why C = 1 is the limit
    assert abs(complex(von_neumann_amplification("upwind", np.pi, courant=0.5))) == pytest.approx(0.0, abs=1e-12)
    assert abs(complex(von_neumann_amplification("upwind", np.pi, courant=1.2))) == pytest.approx(1.4)
    # the Crank-Nicolson factor is bounded by one for every r, which is unconditional stability
    for r in (0.1, 1.0, 10.0, 100.0):
        assert max_amplification("heat_crank_nicolson", r=r) <= 1.0 + 1e-12
    with pytest.raises(ValueError):
        von_neumann_amplification("nope", 0.0)


def test_cfl_limits_are_measured_by_bisection() -> None:
    """``r <= 1/2`` for the explicit heat scheme, ``C <= 1`` for the three advection schemes."""
    assert heat_cfl_limit() == pytest.approx(0.5, abs=1e-9)
    for scheme in ("upwind", "lax_friedrichs", "lax_wendroff"):
        assert advection_stability_limit(scheme) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        advection_stability_limit("nope")


# --------------------------------------------------------------------------- heat
def test_explicit_heat_blows_up_just_above_its_limit() -> None:
    """``r = 0.51`` amplifies the highest mode by ``|1 - 4r| = 1.04`` per step, and that is what is seen.

    A smooth initial datum would *not* blow up -- the unstable modes are simply not there -- so the datum
    here carries a grid-scale component of amplitude ``1e-6``, which is what a boundary condition or
    round-off supplies in practice. The measured growth over ``n`` steps must be ``|1 - 4r|^n``.
    """
    alpha, nx = 0.1, 40
    dx = 1.0 / nx
    steps = 400
    grid_scale = lambda x: np.sin(np.pi * x) + 1e-6 * np.where(np.arange(x.size) % 2 == 0, 1.0, -1.0)  # noqa: E731

    stable = heat_1d_explicit(alpha, nx, 0.49 * dx**2 / alpha, steps=steps, initial=grid_scale)
    unstable = heat_1d_explicit(alpha, nx, 0.51 * dx**2 / alpha, steps=steps, initial=grid_scale)
    assert stable.stable and not unstable.stable
    assert unstable.amplification == pytest.approx(1.04, abs=1e-12)
    # the seed of 1e-6, amplified by 1.04 per step for 400 steps, ends above the smooth part itself
    assert float(np.max(np.abs(unstable.u))) > 1.0
    assert float(np.max(np.abs(unstable.u))) > 5.0 * float(np.max(np.abs(stable.u)))
    assert "PDE.CFL_VIOLATION" in [d.code for d in unstable.diagnostics]
    assert not stable.diagnostics
    # and the smooth datum alone does not blow up, which is the trap this test exists to document
    smooth = heat_1d_explicit(alpha, nx, 0.51 * dx**2 / alpha, steps=steps)
    assert float(np.max(np.abs(smooth.u))) < 1.0


def test_crank_nicolson_is_unconditionally_stable_where_explicit_is_not() -> None:
    """At ``r = 5`` -- ten times the explicit limit -- Crank-Nicolson is stable and the explicit one is not.

    Its accuracy there is the *spatial* discretisation, ``O(dx^2)``, so the comparison is against the exact
    solution evaluated at the time the scheme actually reached, ``steps * dt``. The second point is the
    honest one this test exists to make: the explicit run only explodes because the datum carries a
    grid-scale component of ``1e-6``. On the smooth datum alone it survives eight steps -- the unstable
    mode is simply not there -- which is why a stability limit is a statement about the operator and not
    about a particular run.
    """
    alpha, nx, steps = 0.1, 40, 8
    dx = 1.0 / nx
    dt = 5.0 * dx**2 / alpha  # r = 5, ten times the explicit limit

    implicit = heat_1d_crank_nicolson(alpha, nx, dt, steps=steps)
    exact = heat_1d_exact_mode(alpha, steps * dt, implicit.x)
    assert implicit.stable
    assert float(np.max(np.abs(implicit.u - exact))) < 1e-4
    # and refining the mesh takes that error down, which is what makes it an accuracy limit and not a
    # stability one (measured: 3.7e-5 at nx=40, 2.8e-6 at nx=80, 1.8e-7 at nx=160 over three steps)
    coarse_three = heat_1d_crank_nicolson(alpha, nx, dt, steps=3)
    coarse_error = float(np.max(np.abs(coarse_three.u - heat_1d_exact_mode(alpha, 3 * dt, coarse_three.x))))
    finer = heat_1d_crank_nicolson(alpha, 2 * nx, 5.0 * (0.5 * dx) ** 2 / alpha, steps=3)
    fine_dt = 5.0 * (0.5 * dx) ** 2 / alpha
    fine_error = float(np.max(np.abs(finer.u - heat_1d_exact_mode(alpha, 3 * fine_dt, finer.x))))
    assert fine_error < coarse_error / 4.0

    grid_scale = lambda x: np.sin(np.pi * x) + 1e-6 * np.where(np.arange(x.size) % 2 == 0, 1.0, -1.0)  # noqa: E731
    explicit = heat_1d_explicit(alpha, nx, dt, steps=steps, initial=grid_scale)
    assert not explicit.stable
    assert float(np.linalg.norm(explicit.u)) > 1e3  # 19^8 times the 1e-6 seed is 1e5
    smooth = heat_1d_explicit(alpha, nx, dt, steps=steps)
    assert float(np.linalg.norm(smooth.u)) < 1e2  # damped, not amplified: the mode is absent


def test_heat_convergence_is_second_order_in_the_mesh() -> None:
    """With ``dt = O(dx^2)`` the error is ``O(dx^2)`` for both schemes, and the fit must say 2."""
    explicit = heat_convergence(scheme="explicit")
    implicit = heat_convergence(scheme="crank_nicolson")
    assert explicit["order"] == pytest.approx(2.0, abs=0.1), explicit
    assert implicit["order"] == pytest.approx(2.0, abs=0.1), implicit
    assert explicit["r_squared"] > 0.999


# --------------------------------------------------------------------------- advection
def test_advection_schemes_blow_up_above_courant_one() -> None:
    """``C = 1.05`` grows for every scheme, ``C = 0.8`` does not -- the limit is measured, not assumed."""
    for scheme in ("upwind", "lax_friedrichs", "lax_wendroff"):
        stable = advection_1d(1.0, 100, 0.8, scheme=scheme, steps=200)
        unstable = advection_1d(1.0, 100, 1.05, scheme=scheme, steps=200)
        assert stable.stable and not unstable.stable
        assert unstable.amplification > 1.0
        assert float(np.max(np.abs(unstable.u))) > 10.0 * float(np.max(np.abs(stable.u)))
        assert "PDE.CFL_VIOLATION" in [d.code for d in unstable.diagnostics]


def test_observed_order_of_advection_on_a_smooth_datum() -> None:
    """Upwind is first order, Lax-Wendroff second: measured on ``sin(2 pi x)``, where the order exists."""
    upwind = advection_convergence(scheme="upwind")
    wendroff = advection_convergence(scheme="lax_wendroff")
    assert upwind["order"] == pytest.approx(1.0, abs=0.15), upwind
    assert wendroff["order"] == pytest.approx(2.0, abs=0.2), wendroff


def test_numerical_diffusion_of_upwind_matches_the_modified_equation() -> None:
    """The measured amplitude decay of one Fourier mode must equal ``|G(theta)|`` from the analysis."""
    velocity, nx, courant = 1.0, 200, 0.8
    dx = 1.0 / nx
    steps = 100
    initial = lambda x: np.sin(2.0 * np.pi * x)  # noqa: E731
    result = advection_1d(velocity, nx, courant, scheme="upwind", steps=steps, initial=initial)
    amplitude = float(np.max(np.abs(result.u)))
    theta = 2.0 * np.pi * dx  # the mode the sine represents on this grid
    predicted = abs(complex(von_neumann_amplification("upwind", theta, courant=courant))) ** steps
    assert amplitude == pytest.approx(predicted, rel=1e-3)
    # and the smearing is the diffusivity the modified equation predicts, not a free parameter
    coefficient = numerical_diffusion_coefficient("upwind", velocity, dx, courant)
    assert coefficient == pytest.approx(0.5 * velocity * dx * (1.0 - courant))
    assert numerical_diffusion_coefficient("lax_wendroff", velocity, dx, courant) == 0.0
    diffusion = numerical_diffusion_coefficient("lax_friedrichs", velocity, dx, courant)
    assert diffusion > coefficient  # Lax-Friedrichs diffuses more than upwind at the same Courant number


def test_tvd_limiter_keeps_the_total_variation_and_lax_wendroff_does_not() -> None:
    """On a step, the TVD flux creates no new extremum while Lax-Wendroff overshoots by ~10 %."""
    initial = lambda x: np.where((x >= 0.2) & (x <= 0.4), 1.0, 0.0)  # noqa: E731
    tvd = advection_1d(1.0, 100, 0.8, scheme="tvd", steps=100, initial=initial)
    wendroff = advection_1d(1.0, 100, 0.8, scheme="lax_wendroff", steps=100, initial=initial)
    initial_variation = total_variation(initial(tvd.x))
    assert tvd.total_variation <= 1.05 * initial_variation  # nearly preserved
    assert wendroff.total_variation > 1.4 * initial_variation  # raised by the oscillations
    assert wendroff.overshoot == pytest.approx(0.174, abs=0.02)  # 17 % above the plateau
    assert tvd.overshoot <= 1e-9  # no new extremum at all
    assert "PDE.OSCILLATION" in [d.code for d in wendroff.diagnostics]


def test_total_variation_includes_the_periodic_wrap() -> None:
    assert total_variation(np.array([0.0, 1.0, 0.0])) == pytest.approx(2.0)
    assert total_variation(np.full(5, 0.5)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- wave
def test_wave_cfl_limit_and_the_exact_case_at_courant_one() -> None:
    """``C <= 1``; at ``C = 1`` the leapfrog scheme is exact, at ``C = 0.5`` it is dispersive but stable.

    The datum ``sin(2 pi x)`` with zero initial velocity splits into two counter-propagating halves, so the
    exact solution is ``0.5[sin(2 pi (x - ct)) + sin(2 pi (x + ct))]`` -- and it passes through zero at
    ``t = 1/4``, which is why the amplitude at one time is a poor test and the error against that solution
    is the right one.
    """
    nx, steps = 200, 100
    exact_at_one = wave_1d(1.0, nx, 1.0, steps=steps)
    unstable = wave_1d(1.0, nx, 1.05, steps=50)
    assert exact_at_one["stable"]
    assert unstable["amplification"] > 1.0
    assert "PDE.CFL_VIOLATION" in [d["code"] for d in unstable["diagnostics"]]

    def error(courant: float, n_steps: int, n: int) -> float:
        result = wave_1d(1.0, n, courant, steps=n_steps)
        positions = result["x"]
        t = n_steps * courant / n
        exact = 0.5 * (np.sin(2.0 * np.pi * (positions - t)) + np.sin(2.0 * np.pi * (positions + t)))
        return float(np.max(np.abs(result["u"] - exact)))

    assert error(1.0, steps, nx) < 1e-12  # exact: the discrete and the continuous characteristics coincide
    # at Courant 1/2 the scheme is dispersive, and the error is second order in the mesh at a fixed time
    spacings, errors = [], []
    for n in (100, 200, 400, 800):
        n_steps = int(round(0.25 * n / 0.5))  # a fixed physical time, otherwise the time shrinks with the mesh
        spacings.append(1.0 / n)
        errors.append(error(0.5, n_steps, n))
    from pinneapple_analysis.numerics.convergence import fit_order

    fit = fit_order(np.asarray(spacings), np.asarray(errors))
    assert fit.order == pytest.approx(2.0, abs=0.05), fit


# --------------------------------------------------------------------------- Poisson + manufactured
def test_poisson_1d_is_second_order_against_a_manufactured_solution() -> None:
    """``-u'' = pi^2 sin(pi x)`` has ``u = sin(pi x)``: the observed order must be 2."""
    source = lambda x: np.pi**2 * np.sin(np.pi * x)  # noqa: E731
    solver = lambda n: poisson_1d(source, n)[1]  # noqa: E731
    exact = lambda x: np.sin(np.pi * x)  # noqa: E731
    report = manufactured_solution_study(solver, exact, (10, 20, 40, 80))
    assert report["observed_order"] == pytest.approx(2.0, abs=0.1), report
    assert report["ci95"][0] < 2.0 < report["ci95"][1]
    assert not report["diagnostics"]


def test_poisson_2d_is_second_order_against_a_manufactured_solution() -> None:
    """``-(u_xx + u_yy) = 2 pi^2 sin(pi x) sin(pi y)``: the 5-point stencil is second order too."""
    source = lambda x, y: 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)  # noqa: E731

    def solver(n: int) -> np.ndarray:
        x, _, field = poisson_2d(source, n)
        return field[:, 0]  # the first column is u(x, 0) = 0 everywhere, which is trivial
        # (kept only to make the signature explicit; the study below uses the interior diagonal)
    del solver

    spacings: list[float] = []
    errors: list[float] = []
    for n in (8, 16, 32):
        x, y, field = poisson_2d(source, n)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
        spacings.append(1.0 / n)
        errors.append(float(np.max(np.abs(field - exact))))
    from pinneapple_analysis.numerics.convergence import fit_order

    order = fit_order(np.asarray(spacings), np.asarray(errors))
    assert order.order == pytest.approx(2.0, abs=0.2), errors
    with pytest.raises(ValueError):
        manufactured_solution_study(lambda n: np.zeros(n + 1), np.sin, (10,))


def test_manufactured_solution_study_flags_an_inconsistent_solver() -> None:
    """A solver that returns a wrong field at every mesh has no order: the diagnostic says so."""
    report = manufactured_solution_study(lambda n: np.full(n + 1, 0.3), np.sin, (10, 20, 40, 80))
    assert abs(report["observed_order"]) < 1.0
    assert "PDE.INCONSISTENT" in [d["code"] for d in report["diagnostics"]]


# --------------------------------------------------------------------------- consistency / Lax
def test_truncation_error_is_second_order_for_both_heat_schemes() -> None:
    """Consistency is a property of the scheme: with ``r`` fixed both are ``O(dx^2)``."""
    explicit = truncation_error_study("heat_explicit")
    implicit = truncation_error_study("heat_crank_nicolson")
    assert explicit["consistency_order"] == pytest.approx(2.0, abs=0.2), explicit
    assert implicit["consistency_order"] == pytest.approx(2.0, abs=0.2), implicit
    with pytest.raises(ValueError):
        truncation_error_study("nope")


def test_lax_equivalence_reports_consistency_stability_and_convergence_separately() -> None:
    """Consistent and stable converges; consistent and unstable does not -- all three measured."""
    demo = lax_equivalence_demo()
    assert demo["is_consistent"]
    assert demo["stable_amplification"] <= 1.0 + 1e-12
    assert demo["unstable_amplification"] > 1.0
    assert demo["is_convergent"]
    assert demo["diverges_when_unstable"]
    assert demo["unstable_error"] > demo["stable_error"]
    assert demo["stable_error"] < 1e-3


def test_heat_and_advection_reject_bad_arguments() -> None:
    with pytest.raises(ValueError):
        heat_1d_explicit(0.1, 2, 1e-3)
    with pytest.raises(ValueError):
        heat_1d_crank_nicolson(0.1, 2, 1e-3)
    with pytest.raises(ValueError):
        advection_1d(1.0, 4, 0.5)
    with pytest.raises(ValueError):
        advection_1d(1.0, 100, 0.5, scheme="nope")
    with pytest.raises(ValueError):
        poisson_1d(np.sin, 2)
    with pytest.raises(ValueError):
        poisson_2d(lambda x, y: x * y, 2)
    with pytest.raises(ValueError):
        wave_1d(1.0, 4, 0.5)
    assert math.isclose(total_variation([1.0, 2.0]), 2.0)
