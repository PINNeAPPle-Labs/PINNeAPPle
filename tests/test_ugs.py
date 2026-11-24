"""pinneapple_simulation.ugs: the PVT, the grids, the wells, the reference simulator and what verifies it."""
import numpy as np
import pytest

from pinneapple_simulation.ugs import (
    CartesianGrid,
    Control,
    DeliverabilityProblem,
    Gas,
    Period,
    RadialGrid,
    Rock,
    Schedule,
    SimConfig,
    SlightlyCompressible,
    UGSModel,
    Well,
    equal_allocation,
    evaluate_allocation,
    gaussian_field,
    generate_dataset,
    operating_point,
    optimise_peak_deliverability,
    run_benchmark,
    uniform_baseline,
    verification,
)
from pinneapple_simulation.ugs.benchmark import DAYS_INJECTION, DAYS_WITHDRAWAL, Scenario, metrics
from pinneapple_simulation.ugs.simulator import SECONDS_PER_DAY

# ----------------------------------------------------------------------------------------------- PVT


def test_papay_and_hall_yarborough_agree_over_the_storage_range():
    gas = Gas(T=320.0)
    ppc, tpc = gas.critical
    assert 4.0e6 < ppc < 5.5e6 and 200.0 < tpc < 240.0        # Sutton for sg = 0.65
    for p in (2.0e6, 5.0e6, 1.0e7, 2.0e7, 3.0e7):
        z = float(gas.z(p))
        z_hall = float(Gas(T=320.0, z_model="hall-yarborough").z(p))
        assert 0.6 < z < 1.05
        assert abs(z_hall - z) < 0.12 * z, f"the two z correlations disagree by more than 12 % at {p:g} Pa"


def test_formation_volume_factor_and_density_are_consistent():
    gas = Gas(T=320.0)
    p = 1.0e7
    assert gas.index(p) == pytest.approx(1.0 / gas.Bg(p))
    assert float(gas.Bg(p)) == pytest.approx((gas.p_sc / gas.T_sc) * float(gas.z(p)) * gas.T / p)
    # a real gas is denser than an ideal one at the same pressure and temperature
    assert float(gas.rho(p)) > p * gas.mw / (8.31446261815324 * gas.T)
    assert float(gas.rho(2 * p)) > float(gas.rho(p))


def test_viscosity_is_in_the_range_a_gas_viscosity_is_measured_in():
    gas = Gas(T=320.0)
    for p in (1.0e6, 1.0e7, 3.0e7):
        assert 9e-6 < float(gas.mu(p)) < 35e-6


def test_gas_derivatives_match_finite_differences():
    gas = Gas(T=320.0)
    p = 1.2e7
    h = 1.0e3
    np.testing.assert_allclose(float(gas.dindex_dp(p)),
                               (float(gas.index(p + h)) - float(gas.index(p - h))) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(float(gas.dmu_dp(p)),
                               (float(gas.mu(p + h)) - float(gas.mu(p - h))) / (2 * h), rtol=1e-4)


def test_slightly_compressible_fluid_is_the_linear_limit():
    fluid = SlightlyCompressible(viscosity=2.0e-5)
    assert float(fluid.index(1.0e7)) == 1.0 and float(fluid.dindex_dp(1.0e7)) == 0.0
    assert float(fluid.mu(3.0e7)) == pytest.approx(2.0e-5)


def test_rock_compressibility_moves_the_porosity_linearly():
    rock = Rock(cr=1.0e-8, p_ref=1.0e7)
    assert float(rock.porosity(0.2, 1.0e7)) == pytest.approx(0.2)
    assert float(rock.porosity(0.2, 1.2e7)) == pytest.approx(0.2 * (1.0 + 1.0e-8 * 2.0e6))
    assert float(rock.dporosity_dp(0.2, 1.2e7)) == pytest.approx(0.2 * 1.0e-8)


# ----------------------------------------------------------------------------------------------- grids


def test_cartesian_transmissibility_is_the_harmonic_mean_over_the_distance():
    k1, k2, dx = 1.0e-13, 3.0e-13, 10.0
    grid = CartesianGrid((2,), (dx,), permeability=1.0e-13)
    grid.k[0][:] = [[[k1]], [[k2]]]
    grid = CartesianGrid((2,), (dx,), permeability=(np.array([k1, k2]).reshape(2, 1, 1),) * 3,
                         porosity=0.2)
    expected = 1.0 * (2.0 * k1 * k2 / (k1 + k2)) / dx      # area 1 m2 (dy = dz = 1 m), centre to centre
    assert grid.conn_T[0] == pytest.approx(expected)
    assert grid.conn_a.tolist() == [0] and grid.conn_b.tolist() == [1]


def test_cartesian_volumes_and_connection_counts_in_two_dimensions():
    grid = CartesianGrid((4, 5), (20.0, 25.0), permeability=1.0e-13, porosity=0.2)
    assert grid.n_cells == 20
    assert grid.volumes.sum() == pytest.approx(4 * 5 * 20.0 * 25.0 * 1.0)
    assert len(grid.conn_T) == 3 * 5 + 4 * 4             # x faces + y faces
    assert grid.pore_volume() == pytest.approx(0.2 * grid.volumes.sum())


def test_peaceman_well_index_reduces_to_the_one_dimensional_radius():
    grid = CartesianGrid((5,), (40.0,), permeability=1.0e-13, porosity=0.2)
    k, dz, r_w = 1.0e-13, 1.0, 0.1
    expected = 2 * np.pi * k * dz / np.log(0.2 * 40.0 / r_w)
    assert grid.well_index(2, r_w) == pytest.approx(expected)


def test_radial_grid_volumes_and_transmissibility():
    grid = RadialGrid(40, r_w=0.1, r_out=1.0e3, height=10.0, permeability=1.0e-13, porosity=0.2)
    assert grid.volumes.sum() == pytest.approx(np.pi * (1.0e3 ** 2 - 0.1 ** 2) * 10.0, rel=1e-12)
    k = 1.0e-13
    expected = 2 * np.pi * k * 10.0 / np.log(grid.centre[1] / grid.centre[0])
    assert grid.conn_T[0] == pytest.approx(expected)
    assert grid.well_index(0, 0.1) == pytest.approx(2 * np.pi * k * 10.0 / np.log(grid.centre[0] / 0.1))


def test_gaussian_field_is_lognormal_with_the_requested_correlation_length():
    sigma = 0.6
    field = gaussian_field((80, 80), (10.0,), 100.0, sigma, seed=3)
    logs = np.log(field)
    assert field.min() > 0.0
    assert logs.std() == pytest.approx(sigma, rel=0.1)
    assert logs.mean() == pytest.approx(-0.5 * sigma ** 2, abs=0.1)

    def lag(n):
        return float(np.corrcoef(logs[:, :-n].ravel(), logs[:, n:].ravel())[0, 1])

    assert lag(1) > 0.8                       # one cell, far inside a 100 m correlation length
    assert lag(1) > lag(10) > lag(40)         # it decorrelates as the separation grows


# ----------------------------------------------------------------------------------------------- wells


def test_operating_point_honours_a_rate_that_fits():
    well = Well("W", 0, bhp_min=5.0e6, rate_max=1.0)
    q, bhp, dq = operating_point(well, Control("rate", -0.3), p_cell=1.0e7, mobility=2.0e7,
                                 dmobility_dp=0.0, wi=1.0e-12)
    assert q == pytest.approx(-0.3)
    assert bhp == pytest.approx(1.0e7 - 0.3 / (1.0e-12 * 2.0e7))
    assert dq == 0.0


def test_operating_point_switches_to_the_pressure_floor():
    # a well index times mobility of 2e-5 m3/(s Pa): 10 m3/s costs the whole 0.5 MPa down to the floor, so a
    # target of 20 m3/s cannot be met and the floor takes over
    well = Well("W", 0, bhp_min=9.5e6)
    q, bhp, _dq = operating_point(well, Control("rate", -20.0), p_cell=1.0e7, mobility=2.0e7,
                                  dmobility_dp=0.0, wi=1.0e-12)
    assert bhp == pytest.approx(9.5e6)
    assert q == pytest.approx(-10.0)
    assert abs(q) < 20.0                       # the floor delivers less than the target


def test_operating_point_caps_the_rate_and_the_pressure():
    well = Well("W", 0, bhp_min=1.0e6, bhp_max=2.0e7, rate_max=0.4)
    q, _bhp, _dq = operating_point(well, Control("rate", -2.0), 1.0e7, 2.0e7, 0.0, 1.0e-12)
    assert q == pytest.approx(-0.4)
    well = Well("W", 0, bhp_max=1.2e7)
    q, bhp, _dq = operating_point(well, Control("bhp", 3.0e7), 1.0e7, 2.0e7, 0.0, 1.0e-12)
    assert bhp == pytest.approx(1.2e7) and q == pytest.approx(1.0e-12 * 2.0e7 * 2.0e6)


def test_operating_point_is_shut_in_without_a_control():
    well = Well("W", 0)
    assert operating_point(well, None, 1.0e7, 2.0e7, 0.0, 1.0e-12) == (0.0, 1.0e7, 0.0)


def test_operating_point_derivative_matches_a_finite_difference():
    well = Well("W", 0, bhp_min=9.0e6, bhp_max=2.0e7)
    gas = Gas(T=320.0)
    p = 1.2e7
    wi = 1.0e-12
    mobility = float(gas.index(p) / gas.mu(p))
    dmobility = float((gas.dindex_dp(p) * gas.mu(p) - gas.index(p) * gas.dmu_dp(p)) / gas.mu(p) ** 2)
    q, _bhp, dq = operating_point(well, Control("bhp", 9.5e6), p, mobility, dmobility, wi)
    h = 1.0e3
    q_plus = operating_point(well, Control("bhp", 9.5e6), p + h,
                             float(gas.index(p + h) / gas.mu(p + h)), dmobility, wi)[0]
    q_minus = operating_point(well, Control("bhp", 9.5e6), p - h,
                              float(gas.index(p - h) / gas.mu(p - h)), dmobility, wi)[0]
    assert dq == pytest.approx((q_plus - q_minus) / (2 * h), rel=1e-4)


# ----------------------------------------------------------------------------------------------- simulator


def line_model(n_cells: int = 11, length: float = 40.0):
    grid = CartesianGrid((n_cells,), (length,), permeability=1.0e-13, porosity=0.2)
    return UGSModel(grid=grid, fluid=Gas(T=320.0), rock=Rock(cr=1.0e-9, p_ref=1.2e7),
                    wells=[Well("P1", 0, r_w=0.1, bhp_min=5.0e6, rate_max=5.0)], p0=1.2e7)


def test_mass_is_conserved_to_the_tolerance_of_the_solver():
    model = line_model()
    result = model.simulator().run(Schedule((Period(10.0, {"P1": Control("rate", -1.0e-3)}),)),
                                   SimConfig(dt_days=1.0))
    assert result.converged, result.reason
    assert abs(result.mass_balance_error) < 1e-10
    assert len(result.newton_iterations) == 10
    assert max(result.newton_iterations) <= 8


def test_the_gas_in_place_falls_by_exactly_what_the_wells_took():
    model = line_model()
    result = model.simulator().run(Schedule((Period(8.0, {"P1": Control("rate", -1.0e-3)}),)))
    produced = -float(result.well_cumulative["P1"][-1])
    assert result.gas_in_place[0] - result.gas_in_place[-1] == pytest.approx(produced, rel=1e-12)


def test_a_closed_reservoir_ends_where_the_material_balance_says():
    model = line_model()
    sim = model.simulator()
    result = sim.run(Schedule((Period(8.0, {"P1": Control("rate", -1.0e-3)}),)))
    gip_end = sim.gas_in_place_uniform(float(result.mean_pressure[-1]))
    assert gip_end == pytest.approx(result.gas_in_place[-1], rel=2e-3)


def test_the_jacobian_the_newton_loop_uses_is_the_derivative_of_the_residual():
    model = line_model(n_cells=6)
    sim = model.simulator()
    rng = np.random.default_rng(0)
    p = model.p0 * (1.0 + 0.01 * rng.normal(size=sim.grid.n_cells))
    n_old = sim._state(model.p0)[0].copy()                                    # noqa: SLF001
    controls = {"P1": Control("rate", -1.0e-3)}
    dt = SECONDS_PER_DAY
    R, J, _ = sim._assemble(p, n_old, dt, controls, jacobian=True)           # noqa: SLF001
    step = 1.0 * np.ones_like(p)
    R_plus, _, _ = sim._assemble(p + 1e-3 * step, n_old, dt, controls, jacobian=False)   # noqa: SLF001
    R_minus, _, _ = sim._assemble(p - 1e-3 * step, n_old, dt, controls, jacobian=False)  # noqa: SLF001
    numerical = (R_plus - R_minus) / (2e-3)
    analytic = J @ step
    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-12 * np.abs(numerical).max())


def test_the_well_falls_back_to_its_pressure_floor_when_the_rate_is_beyond_it():
    # a one-metre-thick sheet of 100 m cells: the well index is 1.2e-13 m3, so a rate of 2 m3/s would need
    # 2.2 MPa of drawdown and the 0.1 MPa down to the floor cannot carry it
    grid = CartesianGrid((11, 11), (100.0, 100.0), permeability=1.0e-13, porosity=0.2)
    well = Well("P1", 0, r_w=0.1, bhp_min=11.9e6)
    model = UGSModel(grid=grid, fluid=Gas(T=320.0), rock=Rock(cr=1.0e-9, p_ref=1.2e7),
                     wells=[well], p0=1.2e7)
    sim = model.simulator()
    result = sim.run(Schedule((Period(3.0, {"P1": Control("rate", -2.0)}),)))
    assert result.converged, result.reason
    assert result.well_bhp["P1"][-1] == pytest.approx(11.9e6, rel=1e-6)
    delivered = -result.well_rate["P1"][-1] / SECONDS_PER_DAY
    assert 0.0 < delivered < 2.0
    # the rate the floor allows is the index times the mobility at the cell, times the 0.1 MPa it has left
    p_cell = float(result.pressure[-1][0])
    mobility = float(model.fluid.index(p_cell) / model.fluid.mu(p_cell))
    assert delivered == pytest.approx(sim.wi["P1"] * mobility * (p_cell - 11.9e6), rel=1e-6)


def test_non_convergence_is_reported_not_hidden():
    # a well with no pressure floor asked to take 100 m3/s out of a reservoir that holds 21 000 m3: the step
    # needs a negative pressure, and the run says so instead of passing the last guess off as an answer
    grid = CartesianGrid((11,), (40.0,), permeability=1.0e-13, porosity=0.2)
    model = UGSModel(grid=grid, fluid=Gas(T=320.0), rock=Rock(cr=1.0e-9, p_ref=1.2e7),
                     wells=[Well("P1", 0)], p0=1.2e7)
    result = model.simulator().run(Schedule((Period(1.0, {"P1": Control("rate", -100.0)}),)))
    assert not result.converged
    assert "pressure below" in result.reason
    assert result.n_steps == 0                       # nothing was accepted, and the record says so
    assert "NOT converged" in result.summary()


# ----------------------------------------------------------------------------------------------- verification


@pytest.fixture(scope="module")
def verification_report():
    return verification.run(quick=True)


def test_the_line_source_drawdown_matches_theis(verification_report):
    theis = verification_report["theis"]
    assert theis["ok"] and theis["infinite_acting"]
    assert theis["relative_error"] < 0.02            # measured 8e-3 at the quick resolution
    assert theis["max_abs_error"] > 0.0              # a real error, not a tuned zero


def test_the_closed_reservoir_falls_at_the_material_balance_slope(verification_report):
    pss = verification_report["pseudo_steady_state"]
    assert pss["ok"] and pss["r2"] > 0.999
    assert pss["slope_relative_error"] < 1e-5        # measured 1.5e-8


def test_p_over_z_is_a_straight_line_against_cumulative_production(verification_report):
    mb = verification_report["material_balance"]
    assert mb["ok"] and mb["r2"] > 0.999
    assert mb["gas_in_place_relative_error"] < 0.01  # measured 2.1e-3
    assert mb["fraction_produced"] > 0.2             # a real drawdown, not a line fitted to noise


def test_the_observed_orders_of_accuracy_are_the_nominal_ones(verification_report):
    time_order = verification_report["time_convergence"]["observed_order"]
    grid_order = verification_report["grid_convergence"]["observed_order"]
    assert 0.8 < time_order < 1.4                    # backward Euler, first order
    assert 1.5 < grid_order < 2.5                    # two-point flux, second order


def test_every_case_conserves_mass(verification_report):
    conservation = verification_report["mass_conservation"]
    assert conservation["ok"]
    assert conservation["worst_abs"] < 1e-8
    assert verification.summary(verification_report).startswith("line source:")


# ----------------------------------------------------------------------------------------------- benchmark


def test_the_benchmark_dataset_and_the_tank_baseline():
    dataset = generate_dataset(n_scenarios=5, seed=1)
    assert dataset.n_scenarios >= 6                   # five sampled plus the extrapolation cases
    assert dataset.pressure.shape[1] == len(dataset.times_days)
    assert all(dataset.converged)
    assert abs(dataset.mass_balance_error) < 1e-9
    assert set(dataset.splits()) >= {"train", "test", "extrapolation"}

    from pinneapple_simulation.ugs import TankBaseline
    prediction = TankBaseline(dataset.model).predict(dataset)
    assert abs(prediction["mass_balance_error"]) < 1e-9
    assert np.allclose(prediction["pressure"][:, 0, :], dataset.model.p0)

    report = metrics(dataset, prediction)
    assert 0.0 < report["relative_l2"] < 0.5          # a baseline with a real, non-zero error
    assert report["bhp_error_MPa"] > 0.0
    # the tank gets the average pressure right -- that is what the field metric sees -- and misses the wells
    assert report["relative_l2"] < report["bhp_error"]


def test_the_benchmark_says_the_surrogate_is_missing():
    report = run_benchmark(n_scenarios=2, seed=2)
    assert report["surrogate"]["present"] is False
    assert "not in this pass" in report["surrogate"]["reason"]
    assert report["speedup"]["break_even_evaluations"] is None
    assert report["dataset"]["all_converged"]
    assert report["baseline"]["metrics"]["all"]["r2"] < 1.0


# ----------------------------------------------------------------------------------------------- optimisation


def tiny_model(n_wells: int = 2) -> UGSModel:
    """A small model with two distant wells and a permeability that rises across it, so the two are unequal.

    A gaussian field is not used here on purpose: on a 7 x 7 grid a correlation length of the size of the
    domain leaves one or two modes and the log-normal becomes degenerate (permeabilities three orders below the
    mean). A linear ramp is the controlled heterogeneity this test wants.
    """
    shape = (7, 7)
    ramp = 0.6 + 0.8 * np.arange(shape[0]) / (shape[0] - 1)
    permeability = 2.0e-14 * ramp[:, None] * np.ones(shape[1])[None, :]
    grid = CartesianGrid(shape, (40.0, 40.0), permeability=permeability, porosity=0.2)
    cells = [int(np.ravel_multi_index((1, 1), shape)), int(np.ravel_multi_index((5, 5), shape))][:n_wells]
    wells = [Well(f"W{i}", c, r_w=0.1, bhp_min=5.0e6, bhp_max=2.8e7)
             for i, c in enumerate(cells)]
    return UGSModel(grid=grid, fluid=Gas(T=320.0), rock=Rock(cr=2.0e-9, p_ref=1.2e7), wells=wells, p0=1.2e7)


def tiny_problem(model: UGSModel) -> DeliverabilityProblem:
    return DeliverabilityProblem(model, days_pre=3.0, days_post=2.0, pre_rate=0.8, post_rate=0.8,
                                 capacity=4.0, p_floor=8.0e6, rate_max=1.2, bhp_min=5.0e6)


def test_the_allocation_must_meet_the_demand_before_the_peak():
    model = tiny_model()
    problem = tiny_problem(model)
    starving = evaluate_allocation(problem, np.array([1.0, 0.0]))
    assert not starving.feasible
    assert "short of the demand" in starving.reason
    even = uniform_baseline(problem)
    assert even.feasible, even.reason
    assert even.delivered_before_peak == pytest.approx(even.target_before_peak, rel=0.02)


def test_the_optimiser_finds_the_exhaustive_optimum_on_a_small_model():
    model = tiny_model()
    problem = tiny_problem(model)
    best_peak = -np.inf
    for share in np.linspace(0.0, 1.0, 21):
        result = evaluate_allocation(problem, np.array([share, 1.0 - share]))
        if result.feasible and result.peak > best_peak:
            best_peak = result.peak
    assert best_peak > 0.0, "no feasible allocation in the exhaustive sweep"
    search = optimise_peak_deliverability(problem, n_samples=8, seed=0, refine_rounds=1, capital_search=1)
    assert search["ok"], search.get("reason")
    assert search["optimum"]["feasible"]
    assert search["optimum"]["peak_m3_per_day"] >= 0.99 * best_peak
    assert search["optimum"]["peak_m3_per_day"] >= search["baseline"]["peak_m3_per_day"]


def test_every_reported_schedule_carries_a_simulator_verification():
    model = tiny_model()
    problem = tiny_problem(model)
    search = optimise_peak_deliverability(problem, n_samples=6, seed=2, refine_rounds=1, capital_search=1)
    assert search["verified_by_reference_simulator"] is True
    weights = [search["optimum"]["weights"][w.name] for w in model.wells]
    again = evaluate_allocation(problem, weights)
    assert again.peak == pytest.approx(search["optimum"]["peak_m3_per_day"], rel=1e-12)
    assert again.feasible
    margins = search["constraint_margins"]
    assert margins["bhp_margin_MPa"] >= -1e-6
    assert margins["peak_over_capacity"] <= 1.0 + 1e-9


def test_a_repeated_scenario_is_a_repeated_season():
    model = tiny_model()
    problem = tiny_problem(model)
    first = evaluate_allocation(problem, equal_allocation(2))
    second = evaluate_allocation(problem, equal_allocation(2))
    assert first.peak == second.peak
    assert first.weights == second.weights
    assert first.mass_balance_error == pytest.approx(second.mass_balance_error)


def test_the_benchmark_season_is_injection_then_withdrawal():
    model = tiny_model()
    from pinneapple_simulation.ugs.benchmark import default_scenarios
    scenario: Scenario = default_scenarios(model, n=1, seed=0)[0]
    assert scenario.schedule.days == pytest.approx(DAYS_INJECTION + 2.0 + DAYS_WITHDRAWAL)
    assert scenario.split == "train"
    assert scenario.schedule.period_index(1.0) == 0
    assert scenario.schedule.period_index(DAYS_INJECTION + 1.0) == 1
    assert scenario.schedule.period_index(DAYS_INJECTION + 3.0) == 2
