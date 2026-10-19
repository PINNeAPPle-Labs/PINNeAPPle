"""Physics anomaly detection, physics-based control, the series flow and the embedded MQTT broker."""
import json
import queue
import time

import numpy as np
import pytest

from pinneapple_systems.component_modeling.control import PIDController
from pinneapple_systems.operations.anomaly import (
    BalanceRule,
    DetectorConfig,
    detect,
    fault_benchmark,
    run_benchmark,
    score_events,
    severity_of,
    tank_balance_rules,
    tank_constants,
)
from pinneapple_systems.operations.control import (
    MODE,
    ControlScenario,
    ControlSpec,
    LinearMPC,
    NonlinearMPC,
    PIDBaseline,
    compare_controllers,
    recommend,
    run_closed_loop,
)
from pinneapple_systems.operations.plant import Fault, demo_scenario, simulate
from pinneapple_systems.operations.twin import LumpedTankModel

SENSORS = ["h", "T", "q_in", "q_out", "Q"]


def _cfg(**kw):
    kw.setdefault("normal_until", 7200.0)
    return DetectorConfig(sensors=SENSORS, balances=tank_balance_rules(), constants=tank_constants(), **kw)


def test_leak_breaks_the_mass_balance_by_the_leak_flow():
    leak = 4e-4
    data = simulate(demo_scenario(14400.0, 10.0, seed=5), faults=[Fault("leak", 9000.0, 11400.0, leak)], seed=5)
    res = detect(data, _cfg(methods=("balance",)))
    assert len(res.events) == 1
    ev = res.events[0]
    assert 9000.0 <= ev.start <= 9600.0 and ev.end >= 11000.0
    mass = next(e for e in ev.evidence if e["channel"] == "balance:mass")
    assert mass["mean_excess"] == pytest.approx(-leak, rel=0.15)
    assert ev.explanations[0].startswith("mass balance broken by -")
    assert "something leaves" in ev.explanations[0]


def test_fault_free_data_raise_no_physics_events():
    data = simulate(demo_scenario(14400.0, 10.0, seed=6), seed=6)
    res = detect(data, _cfg(methods=("balance", "model")), model=LumpedTankModel(estimate=()),
                 model_states=["h", "T"], model_inputs=["q_in", "Q", "T_in", "T_amb"])
    assert res.events == []


def test_detector_input_errors():
    data = simulate(demo_scenario(1800.0, 10.0, seed=1), seed=1)
    with pytest.raises(ValueError, match="normal window"):
        detect(data, _cfg(normal_until=100.0))
    with pytest.raises(ValueError, match="unknown methods"):
        detect(data, DetectorConfig(normal_until=1200.0, methods=("magic",)))
    bad = DetectorConfig(normal_until=1200.0, methods=("balance",),
                         balances=[BalanceRule("x", "__import__('os')", "q_in")])
    with pytest.raises(ValueError, match="balance 'x'"):
        detect(data, bad)
    assert [severity_of(r) for r in (1.5, 3, 5, 9)] == ["low", "medium", "high", "critical"]


def test_benchmark_on_held_out_runs():
    res = run_benchmark(seeds=(201,), methods_sets={"balance": ("balance",), "combined": ("statistical",
                        "timeseries", "balance", "model")})
    comb, bal = res["results"]["combined"], res["results"]["balance"]
    assert comb["event_recall"] >= 5 / 6
    # regression guard, not a claim: the measured rate is 0.1-0.6 per hour depending on the run (statistical and
    # time-series channels at operating changes); the app reports the measured figure
    assert comb["false_alarms_per_hour"] <= 1.0
    assert bal["point_precision"] >= 0.9                          # balances rarely fire without a fault
    assert bal["per_fault_kind"]["valve"] == "0/1"                # q_out is measured: the valve keeps the balance


def test_score_events_counts():
    data, faults = fault_benchmark(seed=202)
    res = detect(data, _cfg(methods=("balance",)))
    sc = score_events(res, faults)
    assert sc["tp"] + sc["fn"] > 0 and 0 <= sc["point_precision"] <= 1
    assert len(sc["per_fault"]) == len(faults)


def test_pid_anti_windup_and_limits():
    pid = PIDController(1.0, 1.0, 0.0, setpoint=10.0, output_limits=(0.0, 1.0))
    for _ in range(100):
        u = pid.update(0.0, 1.0)
    assert u == 1.0 and pid._integral < 2.0                      # integral frozen while saturated
    legacy = PIDController(1.0, 0.1, 0.0, setpoint=1.0)
    assert legacy.update(0.0, 1.0) == pytest.approx(1.1)          # unbounded law unchanged


def test_mpc_beats_pid_on_the_same_plant():
    spec = ControlSpec()
    sc = ControlScenario(duration=1800.0)
    pid = run_closed_loop(PIDBaseline(spec), spec, sc)
    lmpc = run_closed_loop(LinearMPC(spec), spec, sc)
    assert lmpc["metrics"]["iae_T_K_s"] < 0.6 * pid["metrics"]["iae_T_K_s"]
    assert lmpc["metrics"]["max_violation_T_K"] == 0.0
    assert lmpc["failed_solves"] == 0
    tr = lmpc["trajectory"]
    assert np.all(tr["Q"] <= spec.Q_bounds[1] + 1e-6) and np.all(np.abs(np.diff(tr["Q"])) <= spec.dQ_max + 1e-3)


def test_nonlinear_mpc_backends_agree():
    pytest.importorskip("casadi")
    spec = ControlSpec()
    est = {"state": {"h": 1.1, "T": 26.0, "q_leak": 0.0, "eta": 1.0}}
    reading = {"q_in": 0.0063, "Q": 3.0e5, "T_in": 15.0, "T_amb": 20.0}
    u_ca, _ = NonlinearMPC(spec, backend="casadi").act(700.0, reading, est)
    u_sp, _ = NonlinearMPC(spec, backend="scipy").act(700.0, reading, est)
    assert u_ca[1] == pytest.approx(u_sp[1], rel=0.05, abs=5e3)
    rec = recommend(est["state"], reading, spec, t=700.0)
    assert rec["mode"] == MODE and "no command path" in rec["mode"]
    assert len(rec["plan"]["pred_T"]) == spec.horizon


def test_compare_controllers_rejects_unknown():
    with pytest.raises(ValueError):
        compare_controllers(controllers=("pid", "magic"))
    with pytest.raises(ValueError):
        compare_controllers(controllers=("smpc",))


def test_series_flow_writes_valid_contracts(tmp_path):
    from pinneapple_data.contracts import load
    from pinneapple_systems.operations.series import run_series_flow

    s = run_series_flow(tmp_path, apps_dir=None)
    assert [st["day"] for st in s["steps"]] == [13, 14, 15, 17, 20]
    for name in ("campaign.study.json", "uncertainty.study.json", "optimize.study.json", "twin.study.json",
                 "tank_rom.json.modelref.json", "campaign_runs.dataset.json", "operating.space.json"):
        assert load(tmp_path / name).kind
    opt = load(tmp_path / "optimize.study.json")
    assert abs(opt.metrics["gap_to_closed_form"]) < 0.05
    twin = s["steps"][-1]
    assert twin["revalidation_requested_at"] is not None and twin["revalidation_requested_at"] > 2400.0


def test_mqtt_broker_round_trip_with_paho():
    mqtt = pytest.importorskip("paho.mqtt.client")
    from pinneapple_systems.digital_twin.io.mqtt_broker import MQTTBroker, topic_matches
    from pinneapple_systems.digital_twin.io.stream import MQTTStream

    assert topic_matches("plant/+/sensors", "plant/a/sensors") and not topic_matches("plant/+", "plant/a/b")
    assert topic_matches("plant/#", "plant/a/b") and not topic_matches("#", "$SYS/x")
    broker = MQTTBroker(port=0, username="u", password="p").start()
    try:
        q = queue.Queue()
        stream = MQTTStream("127.0.0.1", "plant/+/sensors", "tank", ["h", "T"], port=broker.port,
                            username="u", password="p")
        stream.start(q)
        pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        pub.username_pw_set("u", "p")
        pub.connect("127.0.0.1", broker.port)
        pub.loop_start()
        deadline = time.time() + 5
        while not broker.clients or len(broker.clients) < 2:
            assert time.time() < deadline
            time.sleep(0.05)
        time.sleep(0.2)
        for i in range(3):
            pub.publish("plant/t1/sensors", json.dumps({"timestamp": i, "h": 1.0 + i, "T": 25.0}), qos=1)
        pub.publish("other/topic", b"{}")
        got = []
        while len(got) < 3 and time.time() < deadline:
            try:
                got.append(q.get(timeout=0.2).values["h"])
            except queue.Empty:
                pass
        assert got == [1.0, 2.0, 3.0]
        codes = []
        bad = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        bad.username_pw_set("u", "wrong")
        bad.on_connect = lambda c, ud, flags, rc, props: codes.append(rc)
        bad.connect("127.0.0.1", broker.port)
        bad.loop_start()
        while not codes and time.time() < deadline + 5:
            time.sleep(0.05)
        bad.loop_stop()
        assert codes and codes[0].is_failure and not bad.is_connected()
        pub.loop_stop()
        pub.disconnect()
        stream.stop()
    finally:
        broker.stop()


def test_study_results_record_only_checks_that_ran(tmp_path):
    from pinneapple_data.contracts import load
    from pinneapple_systems.operations.contracts import (
        anomaly_study,
        control_study,
        load_model,
        save_model,
    )

    data = simulate(demo_scenario(10800.0, 10.0, seed=7), faults=[Fault("leak", 8000.0, 10000.0, 5e-4)], seed=7)
    st = anomaly_study(detect(data, _cfg(methods=("balance",))))
    assert st.metrics["n_events"] >= 1 and st.passed
    st.save(tmp_path / "a.json")
    assert load(tmp_path / "a.json").study_type == "anomaly"
    cmp = compare_controllers(ControlSpec(), ControlScenario(duration=600.0), controllers=("pid", "rules"))
    cs = control_study(cmp)
    assert cs.table.columns["controller"] == ["PID", "Rule-based"]
    assert all(c.status in ("PASS", "FAIL") for c in cs.checks)
    ref = save_model(LumpedTankModel(), tmp_path / "m.json", "tank")
    assert ref.metrics == {} and load_model(ref, tmp_path).kind == "lumped_tank"
    (tmp_path / "m.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(ref, tmp_path)


def test_from_dict_round_trips():
    spec = ControlSpec.from_dict({"horizon": 10, "setpoints": [[600, 1.1, 28], [0, 1.2, 25]]})
    assert spec.setpoints[0] == (0.0, 1.2, 25.0) and spec.reference(700) == (1.1, 28.0)
    with pytest.raises(ValueError):
        ControlSpec.from_dict({"horizn": 10})
    sc = ControlScenario.from_dict(ControlScenario().to_dict())
    assert sc.faults[0].kind == "valve" and sc.T_in(3000) == 11.0
    cfg = DetectorConfig.from_dict({"normal_until": 100.0, "balances": [{"name": "m", "accumulation": "A*h",
                                                                          "inflow": "q_in"}]})
    assert cfg.balances[0].name == "m"
    with pytest.raises(ValueError):
        DetectorConfig.from_dict({"normal_until": 1, "bogus": 2})


def test_intervals_from_labels_score_like_faults():
    from pinneapple_systems.operations.anomaly import intervals_from_labels

    iv = intervals_from_labels([0, 10, 20, 30, 40, 50], ["", "leak", "leak", "0", "normal", "bias"])
    assert [(i.kind, i.start, i.end) for i in iv] == [("leak", 10.0, 30.0), ("bias", 50.0, 60.0)]
    data, faults = fault_benchmark(seed=203)
    res = detect(data, _cfg(methods=("balance",)))
    a = score_events(res, faults)
    b = score_events(res, intervals_from_labels(data["t"], data["fault"]))
    assert a["tp"] == b["tp"] and a["event_recall"] == b["event_recall"]
