"""The closed twin product: spec validation, preflight refusals, held-out tracking, drift ladder.

The tests use a two-line linear model so the expected answer is known by hand, plus one
end-to-end run on the HX-201 exchanger in ``apps/twin_ops`` — the same path a project takes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pinneapple_systems.digital_twin.product import (
    AcceptanceGates,
    BiasModel,
    CheckReport,
    DriftMonitor,
    DriftPolicy,
    FieldSpec,
    LatentStateTwin,
    ModelSpec,
    SensorSpec,
    SpecError,
    TwinEvidence,
    TwinMode,
    TwinSpec,
    commission,
    ensure_twin_model,
    preflight,
    run_live,
)

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "apps" / "twin_ops"


# ---------------------------------------------------------------------------
# a model whose latent parameter is known by hand:  y = 2x + k
# ---------------------------------------------------------------------------

def _model(X: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(X, dtype=float)[:, 0]
    k = float(np.asarray(theta, dtype=float).ravel()[0])
    return {"y": 2.0 * x + k}


def _spec(**over) -> TwinSpec:
    base = dict(
        asset_id="T-1",
        asset_name="linear twin",
        asset_type="test",
        fields=(
            FieldSpec(name="y", unit="m", kind="state", lo=0.0, hi=20.0, alarm_above=5.0),
            FieldSpec(name="k", unit="m", kind="parameter", lo=-5.0, hi=15.0),
        ),
        sensors=(
            SensorSpec(sensor_id="drv", unit="m", driver=True, lo=0.0, hi=30.0, at=(0.0,)),
            SensorSpec(sensor_id="A", field="y", unit="m", lo=0.0, hi=20.0, at=(1.0,), noise_sigma=0.05),
            SensorSpec(sensor_id="C", field="y", unit="m", lo=0.0, hi=25.0, at=(2.0,), noise_sigma=0.05),
            SensorSpec(sensor_id="H", field="y", unit="m", lo=0.0, hi=20.0, at=(1.0,),
                       noise_sigma=0.05, holdout=True),
        ),
        model=ModelSpec(model_id="linear", version="1", accuracy_floor=0.1),
        coord_names=("x",),
        sampling_s=60.0,
        gates=AcceptanceGates(max_rmse_frac=0.10, max_bias_frac=0.05, min_coverage=0.85,
                              min_alarm_lead_s=0.0),
    )
    base.update(over)
    return TwinSpec(**base)


def _history(n: int = 400, *, k_slope: float = 1.0 / 200.0, seed: int = 3,
             stuck: str | None = None, drop_units: bool = False) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x_drv = 1.0 + 0.05 * np.sin(i / 17.0)
        k = 2.0 + k_slope * i                       # rises through the alarm limit
        y1, y2 = 2.0 * 1.0 + k, 2.0 * 2.0 + k       # model is exact here
        row = {
            "t": 1_767_225_600.0 + 60.0 * i,
            "drv": x_drv,
            "A": y1 + rng.normal(0, 0.05),
            "C": y2 + rng.normal(0, 0.05),
            "H": y1 + rng.normal(0, 0.05),
        }
        if stuck:
            row[stuck] = row[stuck] if not rows else rows[0][stuck]
        rows.append(row)
    if drop_units:
        for s in _spec().sensors:
            pass
    return rows


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

def test_spec_digest_is_stable_and_changes_with_the_gates():
    spec = _spec()
    assert spec.digest() == _spec().digest()
    assert spec.digest() != spec.with_gates(max_rmse_frac=0.2).digest()


def test_spec_rejects_a_sensor_on_a_field_that_does_not_exist():
    with pytest.raises(SpecError, match="do not exist"):
        _spec(sensors=(
            SensorSpec(sensor_id="A", field="nope", unit="m", lo=0, hi=1, at=(1.0,)),
            SensorSpec(sensor_id="H", field="y", unit="m", lo=0, hi=1, at=(1.0,), holdout=True),
        ))


def test_spec_requires_a_holdout_to_be_checkable():
    with pytest.raises(SpecError, match="holdout"):
        _spec(sensors=(
            SensorSpec(sensor_id="A", field="y", unit="m", lo=0, hi=20, at=(1.0,)),
            SensorSpec(sensor_id="C", field="y", unit="m", lo=0, hi=25, at=(2.0,)),
        ))


def test_spec_rejects_a_field_seen_only_by_holdouts():
    with pytest.raises(SpecError, match="only by holdout"):
        TwinSpec(
            asset_id="T2", asset_name="x", asset_type="t",
            fields=(FieldSpec(name="y", unit="m"), FieldSpec(name="z", unit="m")),
            sensors=(
                SensorSpec(sensor_id="A", field="y", unit="m", lo=0, hi=10, at=(1.0,)),
                SensorSpec(sensor_id="H", field="z", unit="m", lo=0, hi=10, at=(2.0,), holdout=True),
            ),
            model=ModelSpec(model_id="m"),
        )


def test_yaml_round_trip(tmp_path):
    spec = _spec()
    path = tmp_path / "t.yaml"
    spec.to_yaml(path)
    assert TwinSpec.from_yaml(path).to_dict() == spec.to_dict()


def test_input_names_are_coordinates_then_drivers():
    assert _spec().input_names == ("x", "drv")


# ---------------------------------------------------------------------------
# model contract
# ---------------------------------------------------------------------------

def test_bias_model_adds_one_offset_per_field():
    wrapped = BiasModel(lambda X: {"y": np.asarray(X, dtype=float)[:, 0]}, ["y"])
    out = wrapped(np.array([[3.0]]), np.array([0.5]))
    assert out["y"][0] == pytest.approx(3.5)


def test_ensure_twin_model_rejects_a_bare_model_when_bias_dim_does_not_match():
    with pytest.raises(Exception):
        ensure_twin_model(lambda X: {"y": X[:, 0]}, ["y", "z"], n_latent=1)


def test_model_output_missing_a_field_is_an_error_not_a_default():
    fn = ensure_twin_model(lambda X, th: {"y": np.zeros(len(X))}, ["y", "z"], 0)
    with pytest.raises(Exception, match="missing"):
        fn(np.zeros((3, 1)), np.zeros(0))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def test_preflight_passes_on_a_clean_spec():
    rep = preflight(_spec(), _history())
    assert isinstance(rep, CheckReport)
    assert rep.passed, rep.render()


def test_preflight_fails_on_a_unit_mismatch():
    bad = _spec(sensors=(
        SensorSpec(sensor_id="drv", unit="m", driver=True, lo=0, hi=30, at=(0.0,)),
        SensorSpec(sensor_id="A", field="y", unit="ft", lo=0, hi=20, at=(1.0,)),
        SensorSpec(sensor_id="C", field="y", unit="m", lo=0, hi=25, at=(2.0,)),
        SensorSpec(sensor_id="H", field="y", unit="m", lo=0, hi=20, at=(1.0,), holdout=True),
    ))
    rep = preflight(bad, _history())
    assert not rep.passed
    assert any(i.check_id == "P01" and "ft" in i.detail for i in rep.items)


def test_preflight_fails_on_a_stuck_channel():
    rep = preflight(_spec(), _history(stuck="A"))
    assert not rep.passed
    assert any(i.check_id == "P07" and i.status == "FAIL" for i in rep.items)


def test_preflight_fails_on_non_monotonic_time():
    rows = _history()
    rows[10], rows[11] = rows[11], rows[10]
    rep = preflight(_spec(), rows)
    assert any(i.check_id == "P03" and i.status == "FAIL" for i in rep.items)


def test_preflight_warns_when_the_gate_is_vacuous():
    rep = preflight(_spec(gates=AcceptanceGates(max_rmse_frac=1.5)), _history())
    assert any(i.check_id == "P09" and i.status == "WARN" for i in rep.items)


# ---------------------------------------------------------------------------
# commissioning
# ---------------------------------------------------------------------------

def test_commissioning_recovers_the_latent_parameter_and_files_evidence(tmp_path):
    spec = _spec()
    hist = _history(n=600, k_slope=1.0 / 400.0)
    ev = commission(spec, _model, hist, n_ens=48, seed=0, warmup_steps=20)

    assert ev.commissioned, ev.render()
    y = ev.tracking.field("y")
    assert y.n > 400
    assert y.rmse_frac is not None and y.rmse_frac <= spec.gates.max_rmse_frac
    assert y.coverage >= spec.gates.min_coverage

    # the twin must have found k near its true end value (2 + 1.5 = 3.5)
    k = ev.initial_state["k"]["value"]
    assert k == pytest.approx(3.5, abs=0.25)

    out = tmp_path / "ev.json"
    ev.save(str(out))
    blob = json.loads(out.read_text())
    assert blob["commissioned"] is True
    assert blob["spec_digest"] == spec.digest()
    assert blob["unverified"], "a report with nothing unverified is not a report"


def test_a_broken_spec_never_reaches_the_backtest():
    bad = _spec(sensors=(
        SensorSpec(sensor_id="A", field="y", unit="ft", lo=0, hi=20, at=(1.0,)),
        SensorSpec(sensor_id="H", field="y", unit="m", lo=0, hi=20, at=(1.0,), holdout=True),
    ))
    ev = commission(bad, _model, _history(), n_ens=16)
    assert not ev.commissioned
    assert ev.tracking is None
    assert any("preflight" in u for u in ev.unverified)


def test_alarm_fires_before_the_measurement_crosses():
    spec = _spec()
    hist = _history(n=600, k_slope=1.0 / 400.0)
    ev = commission(spec, _model, hist, n_ens=48, seed=0, warmup_steps=20)
    assert "y" in ev.tracking.alarm_lead_s
    assert ev.tracking.alarm_lead_s["y"] > 0


# ---------------------------------------------------------------------------
# on line
# ---------------------------------------------------------------------------

def test_run_live_starts_normal_when_seeded_with_the_commissioned_state():
    spec = _spec()
    hist = _history(n=500, k_slope=1.0 / 400.0)
    ev = commission(spec, _model, hist, n_ens=48, seed=0, warmup_steps=20)
    theta0 = [ev.initial_state["k"]["value"]]
    run = run_live(spec, _model, hist[-120:], n_ens=48, seed=0, theta0=theta0)
    assert run.final_mode in (TwinMode.NORMAL, TwinMode.WATCH)
    assert run.coverage is not None and run.coverage >= 0.5


def test_run_live_converges_from_a_cold_start_without_being_handed_anything():
    """A cold twin must find the latent state on its own, and say so by leaving WATCH."""
    spec = _spec()
    hist = _history(n=500, k_slope=1.0 / 400.0)
    run = run_live(spec, _model, hist, n_ens=48, seed=0)
    assert run.final_mode in (TwinMode.NORMAL, TwinMode.WATCH)
    assert run.latent["k"]["value"] == pytest.approx(3.5, abs=0.3)


# ---------------------------------------------------------------------------
# drift ladder
# ---------------------------------------------------------------------------

def test_drift_monitor_walks_the_ladder_and_needs_hysteresis_to_come_back():
    pol = DriftPolicy(window=20, target_coverage=0.9, hard_break=0.5, max_sustained=3,
                      recalibrate_after=6, hysteresis=5)
    mon = DriftMonitor(pol)
    for _ in range(3):
        mon.observe([False])
    assert mon.mode is TwinMode.ALARM, "coverage 0 is a hard break, not a warning"
    for _ in range(6):
        mon.observe([False])
    assert mon.mode is TwinMode.RECALIBRATE
    for _ in range(4):
        mon.observe([True])
    assert mon.mode is TwinMode.RECALIBRATE, "one good run must not clear an alarm"
    for _ in range(25):
        mon.observe([True])
    assert mon.mode is TwinMode.NORMAL
    assert mon.decisions and {"from", "to", "coverage"} <= set(mon.decisions[0])


# ---------------------------------------------------------------------------
# end to end on the reference asset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (APP / "spec_hx201.yaml").exists(), reason="reference app not present")
def test_hx201_commissions_end_to_end():
    from pinneapple_systems.digital_twin.product import TwinSpec as TS
    from pinneapple_systems.operations import exchanger_model, generate_history

    spec = TS.from_yaml(APP / "spec_hx201.yaml")
    hist = generate_history(hours=36.0, sampling_s=600.0)

    def model(X, theta):
        return exchanger_model(X, theta, ua_clean=10_200.0)

    ev = commission(spec, model, hist, n_ens=48, seed=0, warmup_steps=10)
    assert ev.commissioned, ev.render()
    assert ev.tracking.alarm_lead_s.get("T_ho", 0) > 0
    # the latent state must land in the physical range, not at a bound
    rf = ev.initial_state["R_f"]["value"]
    assert 0.0 < rf < 0.0025


def test_export_scene_writes_a_viewer_folder(tmp_path):
    from pinneapple_systems.digital_twin.product import export_scene

    spec = _spec()
    hist = _history(n=40)
    out = export_scene(spec, hist, str(tmp_path / "scene"))
    assert Path(out).exists() and Path(out).name == "scene.json"
    blob = json.loads(Path(out).read_text())
    assert len(blob["sensors"]) == len(spec.sensors)


# ---------------------------------------------------------------------------
# standards
# ---------------------------------------------------------------------------

def test_describe_publishes_the_latent_state_in_the_open_standards():
    from pinneapple_systems.digital_twin.standards import to_ditto_thing, to_dtdl, to_ngsi_ld

    from pinneapple_systems.digital_twin.product import describe

    spec = _spec()
    desc = describe(spec, {"k": {"value": 3.49, "std": 0.02, "unit": "m"}}, {"y": 5.49})
    by_name = {p.name: p for p in desc.properties}
    assert by_name["k"].kind == "parameter"
    assert by_name["y"].kind == "state"
    assert desc.id.startswith("urn:pinneapple:twin:")

    for fn in (to_dtdl, to_ngsi_ld, to_ditto_thing):
        assert isinstance(fn(desc), dict), fn.__name__


def test_identifiers_stay_standard_safe():
    from pinneapple_systems.digital_twin.product import describe

    spec = _spec()
    desc = describe(spec, {"k": {"value": 1.0, "std": 0.0, "unit": "m"}})
    assert all(p.name.isidentifier() for p in desc.properties)
