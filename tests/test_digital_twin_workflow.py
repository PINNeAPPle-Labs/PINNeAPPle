"""The digital-twin workflow (#1193): on a simulated plant with a known parameter drift the drift is detected within
the stated delay and the recalibration (inverse workflow) recovers the parameter; NIS sits inside its chi-square
band before the drift and after the recalibration; NEES over independent plants; anomalies; forecast; CSV replay;
the transform analysis against the dense formula; negative controls."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_systems.digital_twin.workflow import (  # noqa: E402
    chi2_band,
    cusum_arl0,
    digital_twin_workflow,
    etkf_analysis,
    simulate_plant,
    thermal_plant,
)


def _status(r):
    return {c["code"]: c["status"] for c in r["checks"]}


def test_etkf_matches_the_dense_transform_and_the_kalman_mean():
    rng = np.random.default_rng(0)
    N, n, m = 60, 4, 2
    E = rng.standard_normal((N, n))
    hx = E[:, :2] * 2 + 0.1 * rng.standard_normal((N, m))
    y, r = np.array([0.3, -0.2]), np.array([0.5, 0.2])
    xm, hm = E.mean(0), hx.mean(0)
    A, HA = E - xm, hx - hm
    C = HA / r
    lam, V = np.linalg.eigh((N - 1) * np.eye(N) + C @ HA.T)
    Pt = (V / lam) @ V.T
    dense = xm + ((V * np.sqrt((N - 1) / lam)) @ V.T + (Pt @ C @ (y - hm))[:, None]).T @ A
    Ea = etkf_analysis(E, y, hx, r)
    assert np.abs(Ea - dense).max() < 1e-12
    # the analysis mean is the Kalman update with the ensemble covariances
    Pxy, Pyy = A.T @ HA / (N - 1), HA.T @ HA / (N - 1) + np.diag(r)
    assert np.allclose(Ea.mean(0), xm + Pxy @ np.linalg.solve(Pyy, y - hm), atol=1e-12)


def test_statistics_helpers():
    lo, hi = chi2_band(1, 100)
    assert lo == pytest.approx(0.7422, abs=1e-3) and hi == pytest.approx(1.2956, abs=1e-3)
    assert cusum_arl0(0.5, 8.0) > 5e3                       # rare false alarms over a few hundred readings
    assert cusum_arl0(0.5, 4.0) < cusum_arl0(0.5, 8.0)


def test_simulated_drift_detected_recalibrated_and_filter_consistent(tmp_path):
    r = digital_twin_workflow(seed=0, out_dir=str(tmp_path))
    st = _status(r)
    assert r["verdict"] == "pass", r["checks"]
    assert set(st) >= {"twin.nis_consistent", "twin.nis_after_recalibration", "twin.nees_consistent",
                       "twin.drift_detected", "twin.recalibrated", "twin.anomaly_flagged", "twin.forecast_coverage"}
    drift = next(e for e in r["events"] if e["kind"] == "drift")
    assert 200 <= drift["step"] <= 220
    ua = r["recalibrations"][-1]["applied"]["UA"]
    assert ua == pytest.approx(35.0, rel=0.02)
    lo, hi = r["recalibrations"][-1]["intervals"]["UA"]
    assert lo <= 35.0 <= hi
    assert 120 in [e["step"] for e in r["events"] if e["kind"] == "anomaly"]
    assert (tmp_path / "dashboard.html").stat().st_size > 10_000


def test_detection_and_recovery_over_seeds():
    """Five more plants (fresh noise): detection within 20 steps, no false alarm, the recalibrated UA within 2 % of
    the truth in each; the 95 % interval holds it in at least four."""
    hits = 0
    for seed in range(1, 6):
        r = digital_twin_workflow(seed=seed, nees_runs=4, horizon=0)
        st = _status(r)
        assert st["twin.drift_detected"] == "pass", r["checks"]
        rc = r["recalibrations"][-1]
        assert rc["applied"]["UA"] == pytest.approx(35.0, rel=0.02)
        lo, hi = rc["intervals"]["UA"]
        hits += int(lo <= 35.0 <= hi)
    assert hits >= 4


def test_ramp_drift_is_detected_and_tracked():
    r = digital_twin_workflow(seed=0, drift={"param": "UA", "at": 200, "to": 35.0, "ramp": 40}, spike={},
                              nees_runs=4, max_detection_delay=40)
    assert _status(r)["twin.drift_detected"] == "pass", r["checks"]
    assert r["recalibrations"] and r["recalibrations"][-1]["applied"]["UA"] == pytest.approx(35.0, rel=0.03)


def test_no_drift_no_alarm_and_a_wrong_noise_is_inconsistent():
    r = digital_twin_workflow(seed=0, drift={}, spike={}, nees_runs=4, horizon=0)
    assert not [e for e in r["events"] if e["kind"] == "drift"]
    assert _status(r)["twin.nis_consistent"] == "pass"
    # negative control: the sensor is 3x noisier than the model says -> NIS far above its band
    m = thermal_plant()
    m.sensor_std = [0.6]
    sim = simulate_plant(m, n_steps=300, seed=3)
    m2 = thermal_plant()
    r2 = digital_twin_workflow(model=m2, source="arrays", data=sim["record"], horizon=0)
    st2 = _status(r2)
    assert st2["twin.nis_consistent"] == "fail"
    assert r2["nis"]["before"] > 3.0
    assert "recalibration_refused" in [e["kind"] for e in r2["events"]]     # the model cannot fit that noise
    assert st2["twin.truth"] == "inconclusive"               # replayed data: no truth to score drift recovery


def test_csv_replay(tmp_path):
    sim = simulate_plant(thermal_plant(), n_steps=150, seed=5, plant_params={"UA": 50.0})
    p = tmp_path / "plant.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "Q", "T"])
        for row in zip(sim["record"]["t"], sim["record"]["Q"], sim["record"]["T"], strict=True):
            w.writerow([f"{v:.10g}" for v in row])
    r = digital_twin_workflow(source="csv", csv_path=str(p), horizon=30)
    assert r["n_steps"] == 150 and r["source"] == "csv"
    assert _status(r)["twin.nis_consistent"] == "pass"
    assert len(r["forecast"]["mean"]) == 30
    assert all(lo <= mu <= hi for lo, mu, hi in zip(r["forecast"]["lo"], r["forecast"]["mean"], r["forecast"]["hi"],
                                                     strict=True))
    assert math.isfinite(r["what_if"][1]["peak_mean"]) and r["what_if"][1]["peak_mean"] > r["what_if"][2]["peak_mean"]


def test_flow_node_runs(tmp_path):
    from pinneapple_flow import run
    from pinneapple_flow.nodes_ext.digital_twin import build_flow
    f = build_flow(n_steps=300, horizon=40, nees_runs=4, drift={"param": "UA", "at": 150, "to": 35.0, "ramp": 0},
                   spike={"at": 90, "size": 12.0}).flow({})
    res = run(f, workdir=tmp_path)
    assert res.status == "done", res.nodes
