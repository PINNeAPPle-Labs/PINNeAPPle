"""Heated-tank plant, controlled operator inference and the digital twin runtime (pinneapple_systems.operations)."""
import json

import numpy as np
import pytest

from pinneapple_systems.operations.plant import (
    SENSOR_NAMES,
    Fault,
    Scenario,
    SensorNoise,
    TankParams,
    demo_scenario,
    energy_balance_residual,
    inputs_for_steady_state,
    mass_balance_residual,
    simulate,
    steady_state,
)
from pinneapple_systems.operations.rom import ControlledOpInf, fit_controlled_opinf, residual_sigma
from pinneapple_systems.operations.twin import (
    AlertRules,
    LumpedTankModel,
    SurrogateModel,
    TwinRuntime,
    TwinStore,
    model_from_dict,
)

QUIET = SensorNoise({k: 0.0 for k in SENSOR_NAMES})


def _const(q, Q, x0=(1.0, 20.0), duration=20000.0, dt=10.0):
    return Scenario(duration=duration, dt=dt, x0=x0, q_in=lambda t: q, Q=lambda t: Q)


def test_simulation_reaches_closed_form_steady_state():
    p = TankParams()
    out = simulate(_const(0.0066, 2.8e5), p, noise=QUIET)
    h, T = steady_state(p, 0.0066, 2.8e5, 15.0, 20.0)
    assert out["true_h"][-1] == pytest.approx(h, rel=1e-4)
    assert out["true_T"][-1] == pytest.approx(T, abs=1e-3)
    u = inputs_for_steady_state(p, h, T, 15.0, 20.0)
    assert u == pytest.approx([0.0066, 2.8e5], rel=1e-9)


def test_balances_close_without_faults_and_measure_the_faults():
    p = TankParams()
    sc = demo_scenario(3600.0, 10.0, seed=2)
    clean = simulate(sc, p, noise=QUIET)
    r_m = mass_balance_residual(clean["t"], clean["h"], clean["q_in"], clean["q_out"], p.area)
    args = [clean[k] for k in ("t", "h", "T", "q_in", "q_out", "Q", "T_in", "T_amb")]
    r_e = energy_balance_residual(*args, p)
    steady = slice(5, -5)                                   # central differences, away from the ends
    # input steps make the difference quotient approximate: compare with the size of the terms
    assert np.median(np.abs(r_m[steady])) < 1e-3 * 0.0066
    assert np.median(np.abs(r_e[steady])) < 2e-3 * 2.8e5

    leak = 4e-4
    faulty = simulate(sc, p, faults=[Fault("leak", 1200.0, 2400.0, leak)], noise=QUIET)
    r_m = mass_balance_residual(faulty["t"], faulty["h"], faulty["q_in"], faulty["q_out"], p.area)
    inside = (faulty["t"] > 1300) & (faulty["t"] < 2300)
    assert np.median(r_m[inside]) == pytest.approx(-leak, rel=0.02)

    heater = simulate(sc, p, faults=[Fault("heater", 1200.0, 2400.0, 0.2)], noise=QUIET)
    args = [heater[k] for k in ("t", "h", "T", "q_in", "q_out", "Q", "T_in", "T_amb")]
    r_e = energy_balance_residual(*args, p)
    assert np.median(r_e[inside] / heater["Q"][inside]) == pytest.approx(-0.2, abs=0.01)


def test_fault_validation():
    with pytest.raises(ValueError):
        Fault("explosion", 0.0)
    with pytest.raises(ValueError):
        Fault("sensor_bias", 0.0, 10.0, 1.0)            # no target
    with pytest.raises(ValueError):
        Fault("leak", 10.0, 5.0, 1e-4)


def _campaign(seeds, dt=10.0):
    return [simulate(demo_scenario(7200, dt, seed=s), noise=QUIET, seed=s) for s in seeds]


def test_controlled_opinf_held_out_accuracy_and_round_trip():
    trs = _campaign(range(10, 18))
    rom = fit_controlled_opinf(trs, ["h", "T"], ["q_in", "Q", "T_in", "T_amb"], 10.0, holdout=2)
    v = rom.validation
    assert v["n_test_trajectories"] == 2
    assert v["rollout_rel_rmse"]["h"] < 0.02 and v["rollout_rel_rmse"]["T"] < 0.1
    again = ControlledOpInf.from_dict(json.loads(json.dumps(rom.to_dict())))
    x, u = np.array([1.1, 24.0]), np.array([0.006, 2.5e5, 15.0, 20.0])
    assert again.step(x, u) == pytest.approx(rom.step(x, u))
    sig = residual_sigma(rom, trs[-2:])
    assert sig["h"] == pytest.approx(v["one_step_rmse"]["h"], rel=0.05)
    with pytest.raises(ValueError):
        fit_controlled_opinf(trs[:1], ["h", "T"], ["q_in", "Q", "T_in", "T_amb"], 10.0)


def test_controlled_opinf_casadi_step_matches_numpy():
    ca = pytest.importorskip("casadi")
    rom = fit_controlled_opinf(_campaign(range(3)), ["h", "T"], ["q_in", "Q", "T_in", "T_amb"], 10.0, holdout=1)
    x, v = ca.SX.sym("x", 2), ca.SX.sym("v", 4)
    f = ca.Function("f", [x, v], [rom.casadi_step(x, v)])
    x0, u0 = [1.05, 26.0], [0.0062, 3.1e5, 14.0, 21.0]
    assert np.array(f(x0, u0)).ravel() == pytest.approx(rom.step(x0, u0), rel=1e-12)


def _run_twin(model, faults, seed=3, duration=5400.0, rules=None):
    out = simulate(demo_scenario(duration, 10.0, seed=seed), faults=faults, seed=seed)
    tw = TwinRuntime(model, ["h", "T"], {"h": 0.003, "T": 0.05},
                     {"h": 2e-4, "T": 0.01, "q_leak": 2e-5, "eta": 2e-3}, rules=rules, seed=0)
    for i in range(len(out["t"])):
        tw.ingest({k: out[k][i] for k in ("t", "h", "T", "q_in", "Q", "T_in", "T_amb")})
    return tw, out


def test_twin_quiet_without_faults():
    tw, _ = _run_twin(LumpedTankModel(), [], seed=8)
    raised = [a for a in tw.alerts if not a.cleared]
    assert raised == [], [a.message for a in raised]
    assert tw.status == "normal"
    assert tw.last_drift["decision"] == "ok"


def test_twin_estimates_a_leak_and_a_heater_loss():
    tw, out = _run_twin(LumpedTankModel(), [Fault("leak", 1800.0, 6000.0, 6e-4)])
    assert tw.history[-1]["state"]["q_leak"] == pytest.approx(6e-4, rel=0.15)
    assert any(a.kind == "parameter" and a.data["parameter"] == "q_leak" for a in tw.alerts)
    tw, _ = _run_twin(LumpedTankModel(), [Fault("heater", 1800.0, 6000.0, 0.25)])
    assert tw.history[-1]["state"]["eta"] == pytest.approx(0.75, abs=0.03)
    assert any(a.kind == "parameter" and a.data["parameter"] == "eta" for a in tw.alerts)


def test_twin_forecast_band_contains_the_plant():
    p = TankParams()
    sc = _const(0.0072, 3.0e5, x0=(1.0, 24.0), duration=3000.0)
    out = simulate(sc, p, seed=1)
    tw = TwinRuntime(LumpedTankModel(p, estimate=()), ["h", "T"], {"h": 0.003, "T": 0.05}, {"h": 2e-4, "T": 0.01})
    n0 = 100
    for i in range(n0):
        tw.ingest({k: out[k][i] for k in ("t", "h", "T", "q_in", "Q", "T_in", "T_amb")})
    fc = tw.forecast(60)
    truth_h = out["true_h"][n0:n0 + 60]
    inside = (np.array(fc["p05"]["h"]) - 0.002 <= truth_h) & (truth_h <= np.array(fc["p95"]["h"]) + 0.002)
    assert inside.mean() > 0.9


def test_surrogate_twin_flags_a_valve_fault_and_revalidation_fails():
    rom = fit_controlled_opinf(_campaign(range(20, 28)), ["h", "T"], ["q_in", "Q", "T_in", "T_amb"], 10.0,
                               holdout=2)
    tw, _ = _run_twin(SurrogateModel(rom), [Fault("valve", 1800.0, 6000.0, 0.3)])
    kinds = {a.kind for a in tw.alerts}
    assert {"consistency", "revalidation"} <= kinds
    assert tw.revalidate()["passed"] is False


def test_store_and_model_round_trip(tmp_path):
    store = TwinStore(str(tmp_path / "twin.sqlite"))
    m = LumpedTankModel(estimate=("q_leak",))
    out = simulate(demo_scenario(600.0, 10.0, seed=1), seed=1)
    tw = TwinRuntime(model_from_dict(m.to_dict()), ["h", "T"], {"h": 0.003, "T": 0.05},
                     {"h": 2e-4, "T": 0.01, "q_leak": 2e-5}, store=store, asset="t1",
                     rules=AlertRules(forecast_steps=0))
    for i in range(len(out["t"])):
        tw.ingest({k: out[k][i] for k in ("t", "h", "T", "q_in", "Q", "T_in", "T_amb")})
    assert store.count("estimates", "t1") == len(out["t"])
    rows = store.read("estimates", "t1", since=300.0)
    assert rows[0]["t"] >= 300.0 and set(rows[0]["state"]) == {"h", "T", "q_leak"}
    with pytest.raises(ValueError):
        tw.ingest({"t": 0.0, "h": 1.0, "T": 20.0, "q_in": 0.006, "Q": 2e5, "T_in": 15.0, "T_amb": 20.0})
    with pytest.raises(KeyError):
        tw.ingest({"t": 1e6, "h": 1.0})
    snap = tw.snapshot()
    assert snap["asset"] == "t1" and snap["n_updates"] == len(out["t"])


def test_live_twin_over_embedded_mqtt_and_rest():
    import time

    pytest.importorskip("paho.mqtt.client")
    from pinneapple_systems.operations.live import LiveTwin, demo_record, make_twin, replay

    lt = LiveTwin(make_twin(LumpedTankModel()), speed=1000.0).start()
    try:
        deadline = time.time() + 10
        while lt.twin.n_updates < 30 and time.time() < deadline:
            time.sleep(0.1)
        assert lt.twin.n_updates >= 30 and lt.broker is not None and lt.broker.n_published >= 30
        f = lt.inject_fault("leak", 5e-4)
        assert f["kind"] == "leak" and lt.sim.faults
        snap = lt.snapshot()
        assert snap["running"] and len(snap["history"]["t"]) == len(snap["history"]["h"])
        assert snap["forecast"]["t"]
    finally:
        lt.stop()
    rest = LiveTwin(make_twin(LumpedTankModel()), source="rest").start()
    rec = demo_record(600.0)
    for i in range(len(rec["t"])):
        rest.push({k: rec[k][i] for k in ("t", "h", "T", "q_in", "Q", "T_in", "T_amb")})
    rest.stop()
    assert rest.twin.n_updates == len(rec["t"])
    with pytest.raises(RuntimeError):
        rest.inject_fault("leak", 1e-4)
    tw, res = replay(demo_record(), LumpedTankModel())
    raised = [a for a in res["alerts"] if not a["cleared"]]
    assert len(raised) == 1 and raised[0]["data"]["parameter"] == "q_leak"   # one alert, no flapping
    with pytest.raises(ValueError, match="lacks columns"):
        replay({"t": [0.0, 1.0]}, LumpedTankModel())
