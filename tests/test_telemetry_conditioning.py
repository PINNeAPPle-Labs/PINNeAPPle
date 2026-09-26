"""Causal telemetry conditioning: filter maths, outliers, gaps, per-signal profiles."""
import math

import numpy as np
import pytest

from pinneapple_systems.digital_twin import Observation, SignalProfile, StreamingConditioner, TelemetryConditioner


def test_first_order_filter_matches_closed_form_step_response_with_irregular_sampling():
    tau = 10.0
    c = StreamingConditioner(SignalProfile(tau_s=tau, hampel_window=0))
    c.update(0.0, 0.0)
    rng = np.random.default_rng(0)
    t = 0.0
    while t < 30.0:
        t += float(rng.uniform(0.05, 0.8))  # irregular sampling
        y = c.update(t, 1.0).value
    # exact response of dy/dt = (1 - y)/tau to a unit step, independent of the sample times
    assert y == pytest.approx(1 - math.exp(-t / tau), rel=1e-9)


def test_hampel_removes_spikes_and_flags_them():
    c = StreamingConditioner("pressure", tau_s=0.0)
    t = np.arange(0, 20, 0.1)
    x = np.sin(0.5 * t)
    spikes = [50, 120, 170]
    x_bad = x.copy()
    x_bad[spikes] += 8.0
    out = [c.update(ti, xi) for ti, xi in zip(t, x_bad)]
    flagged = [i for i, s in enumerate(out) if "outlier" in s.flags]
    assert flagged == spikes
    err = np.max(np.abs(np.array([s.value for s in out]) - x))
    assert err < 0.15  # replaced by the rolling median, far from the 8.0 spike


def test_gaps_hold_then_go_stale_and_out_of_range_is_missing():
    c = StreamingConditioner(SignalProfile(tau_s=0.0, hampel_window=0, valid_range=(0.0, 100.0), max_gap_s=5.0))
    assert c.update(0.0, 42.0).value == 42.0
    s = c.update(1.0, 500.0)  # out of range -> treated as missing -> hold
    assert s.value == 42.0 and s.flags == ["out_of_range", "gap_filled"]
    assert c.fill(4.0).value == 42.0
    stale = c.fill(6.0)
    assert math.isnan(stale.value) and "stale" in stale.flags and not stale.ok


def test_rate_limit():
    c = StreamingConditioner(SignalProfile(tau_s=0.0, hampel_window=0, max_rate=2.0))
    c.update(0.0, 0.0)
    s = c.update(1.0, 10.0)
    assert s.value == pytest.approx(2.0) and "rate_limited" in s.flags


def test_control_profile_never_alters_values_and_thermal_smooths_more_than_pressure():
    ctrl = StreamingConditioner("control")
    vals = [0.0, 1.0, 1.0, 0.0, 5.0, 5.0]
    assert [ctrl.update(i, v).value for i, v in enumerate(vals)] == vals
    th, pr = StreamingConditioner("thermal"), StreamingConditioner("pressure")
    th.update(0.0, 0.0), pr.update(0.0, 0.0)
    assert th.update(1.0, 1.0).value < pr.update(1.0, 1.0).value  # tau 30 s vs 1 s


def test_telemetry_conditioner_on_observations_and_profile_resolution():
    tc = TelemetryConditioner({"temperature": "thermal", "PT-101.pressure": "pressure", "valve": "control"})
    assert tc._get("any", "temperature").profile.kind == "thermal"
    assert tc._get("PT-101", "pressure").profile.kind == "pressure"
    assert tc._get("valve", "cmd").profile.kind == "control"
    obs = Observation(timestamp=0.0, sensor_id="PT-101", values={"pressure": 5.0})
    out = tc.condition_observation(obs)
    assert out.values["pressure"] == 5.0 and out.metadata["raw_values"] == {"pressure": 5.0}


def test_real_step_change_is_accepted_after_a_few_rejections():
    c = StreamingConditioner("pressure", tau_s=0.0)
    t = np.arange(0, 10, 0.1)
    x = np.where(t < 5, 2.0, 6.0) + 0.01 * np.sin(7 * t)  # valve opens: pressure steps up for good
    out = [c.update(ti, xi) for ti, xi in zip(t, x)]
    k = int(np.argmax(t >= 5))
    assert [("outlier" in s.flags) for s in out[k:k + 3]] == [True, True, True]
    assert "level_shift" in out[k + 3].flags
    assert all(abs(s.value - xi) < 0.05 for s, xi in zip(out[k + 3:], x[k + 3:]))  # tracks the new level
