"""Stirred exothermic reactor as a digital-twin benchmark case (#179): plant, reference, twin, protocol."""
from __future__ import annotations

import pytest

from pinneapple_systems.digital_twin.stirred_reactor import (
    AUTHOR,
    Faults,
    Inputs,
    LumpedTwin,
    Reactor,
    Scenario,
    alarm_scores,
    coolant_for_margin,
    inputs_of,
    make_cases,
    reference_trust,
    results_table,
    run_benchmark,
    run_reference,
    run_twin,
    thermo_from_pair,
    unagitated_reading,
)

U0 = inputs_of(Reactor())


def test_thermodynamics_is_the_post_pair():
    """The reaction thermodynamics is *derived* from the post's two points, so the equilibrium
    conversion there is exactly the pair: that is the debt the case states, not a measurement."""
    R = Reactor()
    dh, ds = thermo_from_pair(48.0, 0.75, 75.0, 0.20)
    assert dh == pytest.approx(R.delta_h) and ds == pytest.approx(R.delta_s)
    assert dh < 0 and ds < 0                                   # exothermic, entropy-losing
    for temperature_c, conversion in ((48.0, 0.75), (75.0, 0.20)):
        k = R.equilibrium_constant(temperature_c + 273.15)
        assert k / (1.0 + k) == pytest.approx(conversion, abs=0.01)
    assert R.equilibrium_constant(348.15) < R.equilibrium_constant(321.15)     # K falls as T rises


def test_agitation_is_a_cooling_variable():
    """The only path agitation has into the model is the tank-side film coefficient."""
    R = Reactor()
    h = [R.film_coefficient(n) for n in (0.0, 1.0, 2.5, 5.0)]
    assert h == sorted(h) and h[0] == pytest.approx(R.h_natural)               # the natural-convection floor
    assert R.overall_coefficient(5.0) > R.overall_coefficient(0.0) * 5.0
    duty = R.duty_coefficient(5.0) * (321.0 - U0.t_coolant_k)
    assert R.cooling_duty(321.0, U0) == pytest.approx(duty)                    # linear in (T - T_cin)
    assert R.duty_coefficient(5.0, (1.0, 1.0, 0.0, 1.0, 1.0)) < R.duty_coefficient(5.0)


def test_the_stationary_point_sits_where_the_case_is_defined_to_operate():
    R = Reactor()
    st = R.steady_state(U0)
    assert st["n_roots"] == 1.0                                                # one stationary point here
    assert st["t_c"] == pytest.approx(AUTHOR["agitated"]["temperature_c"], abs=1.0)
    assert st["conversion"] == pytest.approx(AUTHOR["agitated"]["conversion"], abs=0.02)
    assert st["t_alarm_c"] > st["t_c"] and st["t_trip_c"] > st["t_alarm_c"]     # margin below the alarm
    for key in ("c_a", "c_b"):
        assert st[key] > 0.0
    assert st["c_a"] + st["c_b"] == pytest.approx(R.c_feed, rel=1e-12)          # the feed carries no B


def test_the_unagitated_branch_is_a_prediction_and_differs_from_the_post():
    """Nothing in the sizing touched the impeller-off branch, so this is what the model says."""
    R = Reactor()
    un = unagitated_reading(R, U0)
    assert un["t_c"] > R.steady_state(U0)["t_c"] + 5.0                        # hotter with the impeller off
    assert un["conversion"] < R.steady_state(U0)["conversion"] - 0.1          # and less converted
    assert un["ua_w_k_stopped"] < un["ua_w_k_agitated"] / 5.0                  # the film coefficient collapsed
    # to sit at the post's 75 °C the same tank needs its feed preheated far above the 40 °C used here
    assert un["feed_for_75c_c"] > 60.0
    assert un["conversion_at_75c"] == pytest.approx(AUTHOR["unagitated"]["conversion"], abs=0.01)


def test_reference_closes_its_balances_and_two_routes_agree():
    """Why the reference is the reference: stationary algebra against a marched transient, tolerance
    convergence, and the first law over a run with a fault in it."""
    trust = reference_trust(Reactor(), rtol=1e-12)
    assert trust["steady_agreement_k"] < 1e-2                                  # algebraic vs marched [K]
    assert trust["tolerance_difference_k"] < 1e-5                              # rtol 1e-10 vs 1e-12 [K]
    assert trust["energy_relative_residual"] < 1e-2
    assert trust["steady_roots"] == 1


def test_reference_energy_balance_closes_over_a_fault_run():
    R = Reactor()
    sc = Scenario("bal", "cooling", 5, hours=1.0, dt_s=60.0, onset_s=1200.0,
                  faults=Faults(tubes_lost=(2,), impeller_tripped=True))
    run = run_reference(R, sc)
    eb = run.energy_balance()
    sensible = R.rho * R.cp * R.volume_m3 * float(run.y_true[-1, 2] - run.y_true[0, 2])
    lhs = sensible + eb["cooling_j"] + eb["wall_j"] - eb["feed_j"]
    # the residual is the trapezoidal rule's on this grid (60 s), not the model's: on the
    # protocol's 30 s grid the same run closes to 2e-3 (benchmark.reference_trust)
    assert eb["grid_s"] == 60.0
    assert abs(eb["generated_j"] - lhs) / abs(eb["generated_j"]) < 2e-2


def test_readings_and_faults_move_the_right_way():
    R = Reactor()
    base = run_reference(R, Scenario("ok", "baseline", 11, hours=1.5, dt_s=60.0))
    tube = run_reference(R, Scenario("tube", "cooling", 11, hours=1.5, dt_s=60.0, onset_s=1200.0,
                                     faults=Faults(tubes_lost=(2,))))
    trip = run_reference(R, Scenario("trip", "cooling", 11, hours=1.5, dt_s=60.0, onset_s=1200.0,
                                     faults=Faults(impeller_tripped=True)))
    assert base.temperature_c.max() < trip.temperature_c.max()                # stopping the impeller is hotter
    assert tube.temperature_c.max() > base.temperature_c.max()                # one tube out is warmer
    assert tube.effective_phi(len(tube.t) - 1) < 0.95                          # and the twin can see it
    assert base.tube_scale.min() == 1.0
    assert tube.tube_scale[-1].min() == 0.0


def test_twin_beats_the_instrument_and_the_open_loop():
    R = Reactor()
    sc = Scenario("t", "cooling", 21, hours=2.5, dt_s=30.0, onset_s=1800.0, faults=Faults(tubes_lost=(2,)))
    run = run_twin(R, sc)
    s = run.summary()
    assert s["temperature_error_rms_k"] < s["sensor_temperature_error_rms_k"]
    assert s["temperature_error_rms_k"] < s["open_loop_temperature_error_rms_k"]
    # the latent cooling fraction finds the isolated tube, and it is still walking towards the true
    # 0.788 when this short run ends: the protocol's five hours close the last five hundredths
    assert s["phi_end"] < 0.9 and abs(s["phi_end"] - s["phi_truth_end"]) < 0.08
    assert s["latency_median_ms"] < 1000.0                                     # one sample, well inside 30 s
    assert s["samples"] == len(run.t)


def test_twin_alarms_when_the_impeller_stops_and_stays_quiet_when_it_should():
    R = Reactor()
    quiet = run_twin(R, Scenario("quiet", "baseline", 31, hours=1.2, dt_s=30.0, onset_s=1e9))
    assert not quiet.alarm.any()
    loud = run_twin(R, Scenario("trip", "cooling", 32, hours=5.0, dt_s=30.0, onset_s=3600.0,
                                faults=Faults(impeller_tripped=True)))
    scores = alarm_scores(loud)
    assert scores["truth_alarmed"] and scores["detected"] and not scores["false_alarm"]
    assert scores["lead_s"] > 0.0                                              # the twin is early, not late
    assert scores["twin_alarm_forecast_at_s"] is not None                       # the forecast is what caught it


def test_margin_family_walks_the_coolant_towards_the_alarm():
    R = Reactor()
    wide = coolant_for_margin(R, U0, 12.0)
    tight = coolant_for_margin(R, U0, 2.0)
    assert wide < tight < U0.t_coolant_c + 30.0
    hot = R.steady_state(Inputs(feed_m3_s=U0.feed_m3_s, t_feed_k=U0.t_feed_k, t_coolant_k=tight + 273.15,
                                n_impeller=U0.n_impeller))
    assert hot["t_c"] == pytest.approx(R.t_alarm_c - 2.0, abs=0.3)


def test_protocol_has_the_three_perturbations_and_the_table_has_the_three_columns():
    cases = make_cases(0, hours=0.6, dt_s=60.0)
    kinds = {c.kind for c in cases}
    assert {"noise", "cooling", "margin"} <= kinds
    assert any(c.faults.tubes_lost for c in cases)                             # a failed cooling tube
    assert any(c.faults.impeller_tripped for c in cases)
    assert all(c.noise in {"tight", "typical", "poor"} for c in cases)
    assert sorted({c.noise for c in cases if c.kind == "noise"}) == ["poor", "tight", "typical"]
    bench = run_benchmark(cases=cases, seed=0)
    table = results_table(bench)
    assert "state error" in table and "latency" in table and "alarm" in table
    for key in ("temperature_error_rms_k", "latency_median_ms", "cases_that_crossed_the_alarm"):
        assert bench["overall"][key] is not None
    assert bench["overall"]["cases"] == len(cases)
    assert bench["not_measurable"] and all("why" in item for item in bench["not_measurable"])
    comp = bench["author_comparison"]
    assert comp["agitated"]["author_c"] == AUTHOR["agitated"]["temperature_c"]
    twin = LumpedTwin(Reactor())
    assert twin.horizon_s > 0.0


def test_a_case_crosses_and_the_twin_is_scored_on_it():
    """One margin case driven past the alarm: the truth crosses and the twin has to say so."""
    R = Reactor()
    sc = Scenario("cross", "margin", 41, hours=4.0, dt_s=60.0, onset_s=1e9,
                  coolant_start_c=coolant_for_margin(R, U0, 6.0),
                  coolant_end_c=coolant_for_margin(R, U0, -4.0), margin_k=(6.0, -4.0))
    run = run_twin(R, sc)
    scores = alarm_scores(run)
    assert scores["truth_alarmed"]
    assert not scores["missed"]
    assert run.forecast_margin_k[-1] < run.forecast_margin_k[0] - 6.0          # the margin was eaten
    assert run.forecast_peak_c.max() >= R.t_alarm_c
