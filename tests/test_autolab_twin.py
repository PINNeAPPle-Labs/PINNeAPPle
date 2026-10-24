"""Autonomous Physics Lab, demo 3 (pipe-loop twin): plant physics, residuals, Kalman, diagnosis of the three causes."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.digital_twin.pipe_loop import (
    INPUTS,
    SENSORS,
    SIGMA,
    Diagnoser,
    Loop,
    LossTracker,
    Residuals,
    Scenario,
    Surrogate,
    make_cases,
    quasi_steady_valid,
    run_plant,
    steady_state,
)
from pinneapple_systems.digital_twin.pipe_loop.plant import readings

U0 = {"n": 0.9, "u_B": 0.7, "u_C": 0.6, "T_in": 25.0, "Q_heat_kW": 600.0}


def test_plant_obeys_its_balances():
    st = steady_state(Loop(), U0)
    assert st["newton_residual_m"] < 1e-9
    assert abs(st["Q_A"] - st["Q_B"] - st["Q_C"]) < 1e-15
    assert st["H_pump"] > st["H_tee"] > max(st["H_B"], st["H_C"])                # head falls along the flow
    assert st["T_A"] > U0["T_in"] and st["T_B"] < st["T_A"]                     # heater adds, the branch loses
    obstructed = steady_state(Loop(), U0, {"B": 8.0})
    assert obstructed["Q_B"] < st["Q_B"] and obstructed["Q_C"] > st["Q_C"]       # flow moves to the free branch


def test_physics_residuals_vanish_on_true_readings_and_point_at_the_fault():
    loop = Loop()
    res = Residuals(loop)
    y = readings(steady_state(loop, U0))
    e = res.evaluate(y, U0)
    assert np.abs(e["z"]).max() < 1e-3                                           # exact readings: no residual
    yb = y.copy()
    yb[SENSORS.index("Q_B")] += 10 * SIGMA[SENSORS.index("Q_B")]
    z = res.evaluate(yb, U0)["z"]
    d = e["D_sensor"][:, SENSORS.index("Q_B")]
    assert np.allclose(z, 10 * d, atol=0.3)                                      # a bias moves along its direction
    yo = readings(steady_state(loop, U0, {"B": 4.0}))
    zo = res.evaluate(yo, U0)["z"]
    assert abs(zo[2]) > 10 and np.abs(np.delete(zo, 2)).max() < 0.5             # obstruction B: only Darcy B moves


def test_kalman_recovers_an_obstruction():
    loop = Loop()
    kf = LossTracker(loop)
    y = readings(steady_state(loop, U0, {"B": 5.0}))
    for _ in range(80):
        k, s = kf.step(y, U0)
    assert abs(k[1] - 5.0) < 0.1 and abs(k[0]) < 0.1 and abs(k[2]) < 0.1 and s[1] < 0.5


def test_quasi_steady_validity():
    assert quasi_steady_valid(Loop(), 60.0) and not quasi_steady_valid(Loop(), 0.5)


@pytest.fixture(scope="module")
def diag():
    pytest.importorskip("sklearn")
    loop = Loop()
    d = Diagnoser(loop, Surrogate(n_train=60).fit(loop))
    return d.calibrate([run_plant(loop, Scenario(f"c{i}", "normal", 500 + i, duration_s=2400.0)) for i in range(3)])


@pytest.mark.parametrize("sc", [
    Scenario("ok", "normal", 11, duration_s=2400.0),
    Scenario("obs", "process", 12, duration_s=2400.0, onset_s=900.0, target="C", magnitude=6.0),
    Scenario("drift", "sensor", 13, duration_s=2400.0, onset_s=900.0, target="P_tee", magnitude=0.06, mode="drift"),
    Scenario("ood", "domain", 14, duration_s=2400.0, onset_s=900.0, target="T_in", magnitude=55.0)],
    ids=lambda s: s.name)
def test_the_twin_names_the_cause(diag, sc):
    d = diag.diagnose(run_plant(diag.loop, sc))
    assert d["kind"] == sc.kind
    assert d.get("target", "") == sc.target
    if sc.kind == "process":
        assert abs(d["size"] - sc.magnitude) < 0.5


def test_cases_are_controlled_and_seeded():
    a, b = make_cases(3, seed=1), make_cases(3, seed=1)
    assert [x.__dict__ for x in a] == [x.__dict__ for x in b]
    assert sorted({c.kind for c in a}) == ["domain", "normal", "process", "sensor"]
    assert all(c.target in INPUTS for c in a if c.kind == "domain")


def test_flow_node_is_registered():
    from pinneapple_flow.registry import REGISTRY
    assert [p.name for p in REGISTRY.get("autolab.pipe_twin").outputs] == ["report", "figure", "metrics"]
