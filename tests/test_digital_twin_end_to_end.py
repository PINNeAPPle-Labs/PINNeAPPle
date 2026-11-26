"""The twin end to end on a real sensor stream (#84): the provenance of every record, the commissioned river model,
a run over a simulated stream that scores every check against its truth, a real feed's gaps carried instead of
filled in, and the drift verdict -- stable on a clean recession, RETRAIN when the sensor goes noisy.

No test in this file reaches the network: the USGS path is exercised with a recorded-shaped payload through
``fetch=``, and the live call is the ``PINNEAPPLE_NETWORK`` test in ``tests/test_river_recession.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_systems.digital_twin import end_to_end as E  # noqa: E402
from pinneapple_systems.digital_twin.io import usgs  # noqa: E402


def _fake_series(n: int = 800, dt_minutes: int = 15, q0: float = 1600.0, skip=(300, 301)) -> usgs.Series:
    """A gauge series shaped like a real IV response, on a 30 ft3/s grid, with two readings missing."""
    start = datetime(2026, 11, 17, tzinfo=timezone.utc)
    ts, vs = [], []
    for i in range(n):
        if i in skip:
            continue
        ts.append(start + timedelta(minutes=dt_minutes * i))
        vs.append(round((q0 - 0.4 * i) / 30.0) * 30.0)      # receding, on the feed's grid
    return usgs.Series(site_id="01646500", site_name="TEST RIVER", parameter_cd=usgs.DISCHARGE, unit="ft3/s",
                       timestamps=ts, values=vs, fetched_at=datetime.now(timezone.utc))


def test_usgs_record_keeps_the_citation_and_carries_gaps():
    rec = E.usgs_record("01646500", "P5D", fetch=lambda *a, **k: _fake_series())
    o = rec["origin"]
    assert o["kind"] == "real" and "POTOMAC" not in o["site_name"].upper() and o["site_name"] == "TEST RIVER"
    assert o["url"].endswith("/01646500/") and "public domain" in o["source"]
    assert rec["dt_unit"] == "h" and rec["dt"] == pytest.approx(0.25)
    q = np.asarray(rec["Q"], float)
    assert len(q) == len(rec["t"])
    assert int(np.isnan(q).sum()) == 2                       # the two readings that did not arrive, not filled in
    assert np.allclose(np.asarray(rec["Q_base"], float), 0.0)  # no inflow is measured by this gauge


def test_synthetic_record_is_labelled_synthetic_and_differs_in_provenance():
    rec = E.synthetic_river_record(hours=48.0, step_minutes=15)
    assert rec["origin"]["kind"] == "synthetic"
    assert "not a measurement" in rec["origin"]["note"]
    assert not rec["origin"]["url"]


def test_river_model_is_commissioned_on_the_record():
    rec = E.synthetic_river_record(q0=2000.0, k_per_hour=0.0035, hours=200.0, step_minutes=15, noise=0.005)
    m = E.river_model(rec)
    assert m.name == "linear_reservoir" and m.params == ["k"]
    assert m.nominal["k"] == pytest.approx(0.0035, rel=0.25)         # identified, not assumed
    assert m.sensor_std[0] > 0 and m.process_std[0] > 0
    with pytest.raises(ValueError):                                  # nothing to commission a twin on
        E.river_model({"t": [0.0, 1.0], "Q": [1.0, np.nan], "Q_base": [0.0, 0.0], "dt": 1.0})


def test_end_to_end_on_a_simulated_stream_scores_every_check():
    r = E.end_to_end_twin(source="simulated", n_steps=300, members=150, nees_runs=20, horizon=40,
                          recalibration_window=50)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert r["verdict"] == "pass", r["checks"]
    assert set(st) >= {"twin.nis_consistent", "twin.nees_consistent", "twin.drift_detected", "twin.recalibrated",
                       "twin.anomaly_flagged", "twin.forecast_coverage"}
    assert r["origin"]["kind"] == "simulated" and "synthetic" in r["origin"]["note"]
    assert r["model"]["name"] == "heated_channel"
    assert r["drift"]["status"] == "ok" and r["drift"]["verdict"] in ("STABLE", "REVIEW", "RETRAIN")


def test_missing_readings_are_carried_as_gaps_not_interpolated():
    rec = E.synthetic_river_record(hours=240.0, step_minutes=15, noise=0.004)
    q = np.asarray(rec["Q"], float)
    q[400:430] = np.nan                                     # half a day of no readings at all
    q[600] = np.nan                                         # one reading dropped
    rec["Q"] = q.tolist()
    r = E.end_to_end_twin(source="record", record=rec, members=100)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["twin.missing_readings"] == "pass"
    assert len(r["gaps"]) == 31
    assert any(e["kind"] == "missing" for e in r["events"])
    assert np.all(np.isfinite(np.asarray(r["trace"]["x"], float)))     # the twin kept a finite state throughout
    assert int(np.sum(~np.asarray(r["trace"]["assimilated"], bool))) >= 31


def test_drift_is_stable_on_a_steady_record_and_alarms_when_the_sensor_goes_noisy():
    # a steady river under a steady reading noise: the twin's additive budget holds, nothing is flagged
    steady = E.synthetic_river_record(q0=2000.0, k_per_hour=1e-9, hours=240.0, step_minutes=15, noise=0.004)
    stable = E.end_to_end_twin(source="record", record=steady, members=120)["drift"]
    assert stable["status"] == "ok"
    assert stable["verdict"] == "STABLE", stable["windows"]
    assert stable["flagged"] == 0
    assert stable["thresholds"]["input"] > 0 and stable["thresholds"]["error"] > 0   # calibrated, not hand-set
    assert stable["thresholds"]["prediction"] == 0.0        # the monitored model predicts zero by construction
    # the same record with the reading noise multiplied after 60 % of it: the residuals stop looking like the ones
    # the twin was validated on, which is the whole point of monitoring them
    noisy = E.synthetic_river_record(q0=2000.0, k_per_hour=1e-9, hours=240.0, step_minutes=15, noise=0.004)
    q = np.asarray(noisy["Q"], float)
    cut = int(0.6 * len(q))
    q[cut:] = q[cut:] + 60.0 * np.random.default_rng(3).standard_normal(len(q) - cut)
    noisy["Q"] = q.tolist()
    moved = E.end_to_end_twin(source="record", record=noisy, members=120)["drift"]
    assert moved["status"] == "ok" and moved["verdict"] == "RETRAIN", moved
    assert any("error_drift" in w["alarms"] for w in moved["windows"])
    assert moved["flagged"] > stable["flagged"]
    assert max(w["statistics"]["error_drift"] for w in moved["windows"]) > \
        max(w["statistics"]["error_drift"] for w in stable["windows"])


def test_drift_check_reports_a_record_too_short_instead_of_guessing():
    rec = E.synthetic_river_record(hours=6.0, step_minutes=15)
    d = E.end_to_end_twin(source="record", record=rec, members=60)["drift"]
    assert d["status"] == "not_run" and "few" in d["reason"] or "calibrat" in d["reason"]


def test_end_to_end_writes_the_provenance_page(tmp_path):
    r = E.end_to_end_twin(source="synthetic_river", members=80, out_dir=str(tmp_path))
    page = (tmp_path / "twin_end_to_end.html").read_text()
    assert "Where the readings came from" in page and "SYNTHETIC" in page
    assert r["origin_page"].endswith("twin_end_to_end.html")
    assert (tmp_path / "dashboard.html").exists()
    assert r["origin"]["kind"] == "synthetic" and r["origin"]["dt_unit"] == "h"


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="source:"):
        E.end_to_end_twin(source="opcua")
    with pytest.raises(ValueError, match="record="):
        E.end_to_end_twin(source="record")
