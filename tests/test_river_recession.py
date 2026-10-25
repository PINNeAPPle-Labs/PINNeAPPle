"""pinneapple_systems.digital_twin.recession and .io.usgs (ported from the tests of PINNeAPPle-apps
poc_predictive_twin). The live USGS test needs PINNEAPPLE_NETWORK=1."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from pinneapple_systems.digital_twin import recession as R
from pinneapple_systems.digital_twin.io import usgs


def test_recession_equation_and_half_life():
    assert R.recession(1000.0, 0.01, 0.0) == pytest.approx(1000.0)
    assert R.recession(1000.0, 0.01, math.log(2) / 0.01) == pytest.approx(500.0)


def test_fit_recovers_a_known_rate_from_noisy_readings():
    rng = np.random.default_rng(0)
    h = np.linspace(0, 72, 289)
    q = 1500.0 * np.exp(-0.005 * h) * (1 + 0.003 * rng.standard_normal(h.size))
    fit = R.fit_recession(h + 10.0, q)
    assert fit["k_per_hour"] == pytest.approx(0.005, rel=0.03)
    assert fit["q0"] == pytest.approx(1500.0, rel=0.01) and fit["t0_hours"] == 10.0 and fit["r_squared"] > 0.99
    with pytest.raises(ValueError):
        R.fit_recession([0, 1], [1.0, -1.0])


def _detect(res, hours=None):
    h = np.arange(len(res)) * 0.25 if hours is None else hours
    return R.detect_divergence(h, np.asarray(res) + 100.0, np.full(len(res), 100.0), 0.0, 1.0)


def test_sustained_excursion_is_flagged_with_its_direction():
    up = _detect([0.0] * 8 + [5.0] * 8)
    assert up["detected"] and up["direction"] == "rising_above_forecast" and up["detect_hour"] == 2.0
    down = _detect([0.0] * 8 + [-5.0] * 8)
    assert down["detected"] and down["direction"] == "falling_below_forecast"


def test_single_spike_and_noise_are_not_flagged():
    assert not _detect([0.0] * 8 + [9.0] + [0.0] * 8)["detected"]
    assert not _detect(np.random.default_rng(1).normal(0, 0.5, 96))["detected"]


def test_twin_verdicts_on_synthetic_records():
    calm = R.synthetic_recession(seed=1)
    r = R.run_twin(calm["timestamps"], calm["values"])
    assert r["verdict"] == "MATCHES_FORECAST" and r["calibration"]["regime"] == "RECESSION"
    assert r["calibration"]["half_life_hours"] == pytest.approx(math.log(2) / 0.004, rel=0.05)
    storm = R.synthetic_recession(bump_at_hour=84, bump=0.08, seed=1)
    r = R.run_twin(storm["timestamps"], storm["values"])
    assert r["verdict"] == "DIVERGENCE_DETECTED" and r["detection"]["direction"] == "rising_above_forecast"
    rising = R.synthetic_recession(k_per_hour=-0.002, seed=3)
    assert R.run_twin(rising["timestamps"], rising["values"])["verdict"] == "NO_CLEAN_RECESSION"
    with pytest.raises(ValueError):
        R.run_twin(calm["timestamps"][:5], calm["values"][:5])


def test_water_level_from_gage_height():
    q = R.synthetic_recession(seed=1)
    st = R.synthetic_recession(q0=5.0, k_per_hour=0.002, noise=0.001, seed=2)
    w = R.run_twin(q["timestamps"], q["values"], stage_timestamps=st["timestamps"], stage_values=st["values"])
    assert w["water_level"]["forecast_now_ft"] == pytest.approx(w["water_level"]["current_ft"], rel=0.01)


def test_parse_iv_payload_offline():
    payload = {"value": {"timeSeries": [{"sourceInfo": {"siteName": "TEST RIVER"},
                                         "variable": {"unit": {"unitCode": "ft3/s"}},
                                         "values": [{"value": [
                                             {"dateTime": "2026-01-01T00:15:00.000-05:00", "value": "1210"},
                                             {"dateTime": "2026-01-01T00:00:00.000-05:00", "value": "1215"},
                                             {"dateTime": "2026-01-01T00:30:00.000-05:00", "value": "bad"}]}]}]}}
    s = usgs.parse_iv(payload, "00000000", usgs.DISCHARGE)
    assert s.site_name == "TEST RIVER" and s.values == [1215.0, 1210.0] and s.unit == "ft3/s"
    with pytest.raises(usgs.USGSDataError):
        usgs.parse_iv({"value": {"timeSeries": []}}, "x", usgs.DISCHARGE)


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_NETWORK"), reason="set PINNEAPPLE_NETWORK=1 (live USGS API)")
def test_live_usgs_gauge_runs_through_the_twin():
    s = usgs.fetch_iv("01646500", usgs.DISCHARGE, "P5D")
    assert len(s.values) > 100 and "POTOMAC" in s.site_name.upper()
    r = R.run_twin(s.timestamps, s.values)
    assert r["verdict"] in ("DIVERGENCE_DETECTED", "MATCHES_FORECAST", "NO_CLEAN_RECESSION")
