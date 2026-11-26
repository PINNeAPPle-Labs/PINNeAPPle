"""The twin benchmark (#57): every section measured on the heated channel, against a truth the plant kept.

The protocol is stated in the result (sizes, plant parameters, sensor noise, seeds) and every number is reproducible
from it; these tests check that each section measures what it says, that the sections that cannot be measured say why,
and that a case with one sensor reports the failure modes it cannot run instead of inventing them.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_systems.digital_twin.benchmark import (  # noqa: E402
    CASES,
    benchmark_case,
    cases,
    list_benchmark,
    twin_benchmark,
)

SMALL = dict(n_steps=120, members=50)


@pytest.fixture(scope="module")
def bench():
    return twin_benchmark(**SMALL)


def test_the_protocol_is_stated_and_the_case_is_the_one_asked_for(bench):
    p = bench["protocol"]
    assert p["case"] == "heated_channel" and p["model"] == "heated_channel"
    assert p["space"] == "synthetic" and "simulated plant" in p["truth"]
    assert p["n_steps"] == 120 and p["members"] == 50 and p["seed"] == 0
    assert p["plant_params"] and p["sensor_std"] and p["limits"]
    assert set(bench["sections"]) == {"state_error", "latency", "noise", "sensor_failure", "anomaly", "rul",
                                      "decision"}
    assert set(bench["verdicts"]) == set(bench["sections"])
    assert "heated_channel" in cases() and CASES["heated_channel"]["failure_drop"] < 1.0


def test_state_error_is_scored_against_the_truth_and_beats_persistence(bench):
    se = bench["sections"]["state_error"]
    assert se["rmse"] < se["persistence_rmse"] / 10          # a filter that cannot beat persistence is not a twin
    assert se["beats_persistence"] is True
    assert abs(se["parameter_relative_error"]) < 0.10        # the plant's parameter is recovered
    assert se["readings"] > 100


def test_latency_is_measured_against_the_data_period_and_the_recovery_path_against_its_window(bench):
    la = bench["sections"]["latency"]
    assert la["mean_ms"] > 0 and la["p95_ms"] >= la["mean_ms"] and la["max_ms"] >= la["p95_ms"]
    assert la["budget_ms"] == pytest.approx(1000.0)          # one reading per second, the case's data period
    assert 0.0 <= la["over_budget"] <= 1.0
    assert la["recalibration_s"] > 0 and la["recalibration_window_s"] == pytest.approx(50.0)
    assert isinstance(la["recalibration_fits_the_window"], bool)


def test_noise_section_grows_with_the_plant_noise_and_is_consistent_at_the_nominal_level(bench):
    rows = {row["sensor_scale"]: row for row in bench["sections"]["noise"]}
    assert set(rows) == {0.5, 1.0, 2.0, 4.0}
    assert rows[1.0]["rmse_ratio"] == pytest.approx(1.0)
    assert rows[0.5]["rmse"] < rows[1.0]["rmse"] < rows[2.0]["rmse"] < rows[4.0]["rmse"]
    assert rows[1.0]["consistent"] is True                   # told the right noise: inside the band
    assert rows[4.0]["nis"] > rows[1.0]["nis"]               # told too little noise: the innovations blow up


def test_a_lost_reading_is_recorded_and_a_lying_sensor_is_detected(bench):
    rows = {row["mode"]: row for row in bench["sections"]["sensor_failure"]}
    assert set(rows) == {"dropout", "outage", "stuck", "bias"}
    for mode in ("dropout", "outage"):
        assert rows[mode]["gaps"] > 0 and rows[mode]["handled"] is True
        assert rows[mode]["rmse_ratio"] < 2.0                # one live thermocouple still anchors the state
    for mode in ("stuck", "bias"):
        assert rows[mode]["alarmed"] is True and rows[mode]["handled"] is True
        assert rows[mode]["rmse_ratio"] > 5.0                # a lying sensor is not believed, but it hurts
        assert rows[mode]["first_event"] in ("anomaly", "drift")
    assert all(row["finite_state"] for row in rows.values())


def test_anomaly_rate_and_false_alarms_are_measured(bench):
    an = bench["sections"]["anomaly"]
    assert an["spikes"] == 18 and 0 <= an["detected"] <= an["spikes"]
    assert 0.0 <= an["detection_rate"] <= 1.0
    assert an["false_per_100"] < 5.0
    assert bench["verdicts"]["anomaly"] in ("pass", "review")


def test_rul_either_predicts_or_says_why_it_cannot(bench):
    ru = bench["sections"]["rul"]
    if ru.get("status") == "not_run":
        assert ru["verdict"] == "not_measurable" and ru["reason"]
        assert "facts" in ru or "rows" in ru
    else:
        assert ru["rows"] and all("fit_quality" in row for row in ru["rows"])
        assert ru["verdict"] in ("pass", "partial", "fail")


def test_the_decision_section_prices_the_twin_against_never_acting(bench):
    de = bench["sections"]["decision"]
    assert de["n"] > 100 and 0.0 <= de["agreement"] <= 1.0
    assert de["cost"]["perfect"] == 0.0
    assert de["cost"]["twin"] <= de["cost"]["never_act"]
    assert sum(de["counts"].values()) == de["n"]
    truth_actions = {a for a in ("safe", "warn", "stop") if any(k.startswith(a + "->") and v
                                                                for k, v in de["counts"].items())}
    assert len(truth_actions) == 3                            # every action class is exercised
    assert de["limits"]["warn"] < de["limits"]["stop"] and de["basis"]


def test_the_results_table_has_every_section(bench):
    md = list_benchmark(bench)
    assert md.startswith("# Twin benchmark: heated_channel")
    for header in ("state error", "latency", "noise", "sensor failure", "anomaly", "RUL", "decision"):
        assert f"| {header} |" in md, header
    assert "Verdicts:" in md and "synthetic" not in md.splitlines()[0]
    assert md.count("|") > 60


def test_a_case_with_one_sensor_reports_the_modes_it_cannot_run():
    r = twin_benchmark(model="thermal_plant", n_steps=100, members=40)
    assert r["protocol"]["model"] == "thermal_plant" and len(r["protocol"]["outputs"]) == 1
    rows = {row["mode"]: row for row in r["sections"]["sensor_failure"]}
    for mode in ("outage", "stuck", "bias"):
        assert rows[mode]["status"] == "not_run" and "one output" in rows[mode]["reason"]
    assert r["sections"]["decision"]["n"] > 50


def test_an_unknown_case_is_refused():
    with pytest.raises(ValueError, match="case:"):
        benchmark_case("twin_of_nothing")
    with pytest.raises(ValueError, match="case:"):
        twin_benchmark(case="twin_of_nothing")


def test_the_case_limits_come_from_the_plants_own_physics():
    spec = benchmark_case("heated_channel")
    assert spec["limits"]["warn"] < spec["limits"]["stop"]
    assert spec["limits"]["warn"] > 20.0                    # above the ambient it starts from
    assert spec["steady"] > spec["limits"]["stop"]          # the trip level is inside the plant's own range
    assert np.isfinite(spec["steady"]) and spec["failure_value"] < spec["model"].nominal["H"]
