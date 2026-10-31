"""Critical slowing down and early-warning indicators (pinneapple_analysis.dynamics.early_warning).

References: the stationary Ornstein-Uhlenbeck process (lag-1 autocorrelation exp(-theta dt), variance
sigma^2 / (2 theta)); stochastic normal forms of the saddle-node, transcritical and pitchfork bifurcations driven
towards (not past) the bifurcation; negative controls with constant stability (plain, accelerating trend, growing
noise); an exact exponential relaxation for the recovery rate. Detection and false-positive claims are made on fixed
seeds and small replicate sets; the rates themselves are measured in the benchmark.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics.early_warning import (
    INTERPRETATION,
    AutocorrelationAnalyzer,
    CriticalSlowingDown,
    EarlyWarningAnalyzer,
    RecoveryRateEstimator,
    RollingStatistics,
    TransitionDiagnostics,
    detrend,
    ramp_series,
)


def _series(kind, seed=1, **kw):
    sig = {"saddle_node": 0.002, "transcritical": 0.003}.get(kind, 0.05)
    return ramp_series(kind, seed=seed, sigma=sig, end=0.1, **kw)


# =========================================================================== detrending
def test_loess_removes_a_linear_trend_exactly_and_gaussian_is_biased_at_the_ends():
    t = np.linspace(0, 100, 501)
    trend = 3.0 + 0.05 * t
    r_lo, tr_lo = detrend(t, trend, "loess", 0.1)
    r_ga, _ = detrend(t, trend, "gaussian", 0.1)
    assert np.max(np.abs(r_lo)) < 1e-9 and np.allclose(tr_lo, trend)
    assert abs(r_ga[0]) > 0.1 and abs(r_ga[250]) < 1e-9          # edge bias ~ slope x bandwidth, interior exact
    x = trend.copy()
    x[[10, 200]] = np.nan
    r, _ = detrend(t, x, "linear")
    assert np.isnan(r[[10, 200]]).all() and np.nanmax(np.abs(r)) < 1e-9
    with pytest.raises(ValueError):
        detrend(t, x, "spline")


# =========================================================================== windowed statistics
def test_vectorised_and_looped_windows_agree():
    rng = np.random.default_rng(0)
    t = np.arange(300.0)
    x = np.cumsum(rng.standard_normal(300)) * 0.1 + rng.standard_normal(300)
    fast = RollingStatistics(0.3).compute(t, x, ("variance", "skewness", "kurtosis", "cv"), raw=x + 10)
    ac_fast = AutocorrelationAnalyzer(0.3).compute(t, x)
    x_nan = x.copy()
    x_nan[0] = np.nan                                                # forces the looped path
    slow = RollingStatistics(0.3, min_count=5).compute(t, x_nan, ("variance", "skewness", "kurtosis", "cv"),
                                                       raw=x_nan + 10)
    ac_slow = AutocorrelationAnalyzer(0.3, min_count=5).compute(t, x_nan)
    for k in ("variance", "skewness", "kurtosis", "cv"):              # windows that do not contain sample 0
        assert np.allclose(fast[k].values[1:], slow[k].values[1:], rtol=1e-10, equal_nan=True)
    assert np.allclose(ac_fast.values[1:], ac_slow.values[1:], rtol=1e-10)
    from pinneapple_analysis.dynamics.early_warning import _windows

    i = 10
    seg = x[_windows(t, 0.3, "fraction")[i]]
    assert fast["variance"].values[i] == pytest.approx(seg.var(ddof=1))
    assert ac_fast.values[i] == pytest.approx(np.corrcoef(seg[:-1], seg[1:])[0, 1])


def test_stationary_ou_estimates_match_theory():
    ac, va = [], []
    for seed in range(6):
        s = ramp_series("stable", seed=seed, n=2000, t_end=1000.0, sigma=0.1, start=0.5)
        r = EarlyWarningAnalyzer(detrend="linear", surrogate="none").analyze((s["t"], s["x"]))
        ac.append(np.nanmean(r.details["indicators"]["ac1"].values))
        va.append(np.nanmean(r.details["indicators"]["variance"].values))
    dt = 1000.0 / 1999
    assert abs(np.mean(ac) - math.exp(-0.5 * dt)) < 4 * np.std(ac, ddof=1) / math.sqrt(6) + 0.005
    assert abs(np.mean(va) - 0.01) < 4 * np.std(va, ddof=1) / math.sqrt(6)


# =========================================================================== detection and controls
@pytest.mark.parametrize("kind", ["saddle_node", "transcritical", "pitchfork", "ou_decreasing"])
def test_slowing_down_is_detected_before_each_bifurcation(kind):
    s = _series(kind)
    assert s["tipped_at"] is None and s["approaching_bifurcation"]
    r = EarlyWarningAnalyzer(n_surrogates=49).analyze((s["t"], s["x"]), seed=0)
    ind = r.details["indicators"]
    assert ind["variance"].significant() and ind["ac1"].significant()
    assert ind["ac1"].values[-1] > ind["ac1"].values[0]
    rep = TransitionDiagnostics(r).report()
    assert rep["verdict"] == "indicator signal (not a prediction)" and rep["is_prediction"] is False
    assert rep["interpretation"] == INTERPRETATION


def test_stable_control_gives_no_signal():
    s = _series("stable", seed=5)
    r = EarlyWarningAnalyzer(n_surrogates=49).analyze((s["t"], s["x"]), seed=0)
    assert r.estimate["n_significant_rising"] == 0
    assert TransitionDiagnostics(r).report()["verdict"] == "no signal"


def test_trend_false_positive_without_detrending_and_its_removal():
    s = _series("stable_trend", seed=2)
    raw = EarlyWarningAnalyzer(detrend="none", n_surrogates=49).analyze((s["t"], s["x"]), seed=0)
    assert raw.details["indicators"]["ac1"].significant()          # a trend alone "warns"
    assert raw.diagnostic("no_detrending").status == "warn"
    assert TransitionDiagnostics(raw).report()["verdict"] == "inconclusive"
    clean = EarlyWarningAnalyzer(detrend="loess", n_surrogates=49).analyze((s["t"], s["x"]), seed=0)
    assert not clean.details["indicators"]["ac1"].significant()


def test_growing_noise_raises_variance_but_not_autocorrelation():
    s = _series("stable_growing_noise", seed=3)
    r = EarlyWarningAnalyzer(n_surrogates=49).analyze((s["t"], s["x"]), seed=0)
    ind = r.details["indicators"]
    assert ind["variance"].significant() and not ind["ac1"].significant()


def test_critical_slowing_down_bundle_and_recovery_rate():
    s = _series("ou_decreasing")
    r = CriticalSlowingDown(n_surrogates=19).analyze((s["t"], s["x"]), seed=0)
    rr = r.details["indicators"]["recovery_rate"]
    assert np.all(rr.values[np.isfinite(rr.values)] < 0) and rr.tau > 0.5     # approaches 0 from below
    assert rr.values[-1] == pytest.approx(-np.mean(s["rate"][-rr.t.size:]), rel=0.6)


def test_recovery_rate_from_a_pulse():
    t = np.linspace(0, 20, 200)
    x = 2.0 + 1.5 * np.exp(-0.35 * t)
    fit = RecoveryRateEstimator.from_pulse(t, x)
    assert fit["rate"] == pytest.approx(0.35, rel=1e-6) and fit["x_eq"] == pytest.approx(2.0, rel=1e-6)
    noisy = x + np.random.default_rng(0).normal(0, 0.02, t.size)
    f2 = RecoveryRateEstimator.from_pulse(t, noisy, x_eq=2.0)
    assert abs(f2["rate"] - 0.35) < 4 * f2["rate_se"]


def test_spectral_reddening():
    s = _series("ou_decreasing")
    r = EarlyWarningAnalyzer(indicators=("low_frequency_power", "spectral_exponent"), surrogate="none").analyze(
        (s["t"], s["x"]))
    assert r.estimate["low_frequency_power"]["tau"] > 0.5 and r.estimate["spectral_exponent"]["tau"] > 0.5


# =========================================================================== sampling, missing data, sensitivity
def test_irregular_sampling_with_pair_autocorrelation():
    s = ramp_series("stable", seed=4, n=4000, t_end=1000.0, sigma=0.1, start=0.5)
    rng = np.random.default_rng(4)
    keep = np.sort(rng.choice(s["t"].size, 1500, replace=False))
    t, x = s["t"][keep], s["x"][keep]
    tr = dyn.from_array(t, x, ("x",), kind="stochastic")
    lag = 2.0
    fail = EarlyWarningAnalyzer().analyze(tr)
    assert not fail.applicable and "lag_time" in fail.failures[0]
    r = EarlyWarningAnalyzer(detrend="linear", surrogate="none", lag_time=lag).analyze(tr)
    ac = np.nanmean(r.details["indicators"]["ac1"].values)
    assert ac == pytest.approx(math.exp(-0.5 * lag), abs=0.05)
    spec = EarlyWarningAnalyzer(indicators=("spectral_exponent",), surrogate="none").analyze(tr)
    assert not spec.applicable


def test_missing_values_are_tolerated_and_reported():
    s = _series("ou_decreasing")
    x = s["x"].copy()
    x[np.random.default_rng(0).choice(x.size, 100, replace=False)] = np.nan
    r = EarlyWarningAnalyzer(n_surrogates=19).analyze((s["t"], x), seed=0)
    assert r.applicable and r.diagnostic("missing_fraction").value == pytest.approx(0.1)
    assert r.estimate["ac1"]["tau"] > 0.5
    short = EarlyWarningAnalyzer().analyze((np.arange(20.0), np.random.default_rng(0).normal(size=20)))
    assert not short.applicable


def test_window_bandwidth_and_sampling_sensitivity():
    s = _series("ou_decreasing")
    an = EarlyWarningAnalyzer()
    sens = an.sensitivity((s["t"], s["x"]), windows=(0.25, 0.5, 0.75), bandwidths=(0.05, 0.2))
    assert sens["tau"]["variance"].shape == (3, 2) and sens["fraction_positive"]["ac1"] == 1.0
    samp = EarlyWarningAnalyzer(n_surrogates=9).sampling_sensitivity((s["t"], s["x"]), factors=(1, 4))
    assert set(samp) == {1, 4} and samp[1]["ac1"][0] > 0.5
    r = EarlyWarningAnalyzer(n_surrogates=19).analyze((s["t"], s["x"]), seed=0)
    fragile = {"fraction_positive": {"variance": 0.5, "ac1": 1.0}}
    rep = TransitionDiagnostics(r, fragile).report()
    assert rep["verdict"] == "inconclusive" and any("50%" in d for d in rep["doubts"])


# =========================================================================== surrogates, reproducibility, failures
@pytest.mark.parametrize("kind", ["phase", "iaaft"])
def test_other_surrogates(kind):
    s = _series("ou_decreasing")
    r = EarlyWarningAnalyzer(surrogate=kind, n_surrogates=9).analyze((s["t"], s["x"]), seed=0)
    assert r.estimate["ac1"]["p_value"] == pytest.approx(0.1)          # floor 1/(9+1): all surrogates below
    assert kind in r.uncertainty["ac1"]["kind"]


def test_reproducible_and_seeded():
    s = _series("stable", seed=6)
    a = EarlyWarningAnalyzer(n_surrogates=19).analyze((s["t"], s["x"]), seed=3)
    b = EarlyWarningAnalyzer(n_surrogates=19).analyze((s["t"], s["x"]), seed=3)
    assert a.estimate["ac1"]["p_value"] == b.estimate["ac1"]["p_value"]
    assert a.reproducibility.input_hash == b.reproducibility.input_hash
    with pytest.raises(ValueError):
        EarlyWarningAnalyzer(surrogate="bootstrap")
    with pytest.raises(ValueError):
        ramp_series("hopf")


def test_ramp_series_reports_tipping():
    s = ramp_series("saddle_node", sigma=0.05, end=0.02, seed=0)       # noise wider than the shrinking basin
    assert s["tipped_at"] is not None and np.isnan(s["x"][-1])
    assert np.allclose(s["ac1_theory"], np.exp(-s["rate"] * s["dt"]))


def test_contract_and_card():
    s = _series("stable", seed=7)
    r = EarlyWarningAnalyzer(n_surrogates=9).analyze(dyn.from_array(s["t"], s["x"], ("x",)), seed=0)
    assert {a.name for a in r.unchecked_assumptions} >= {"local bifurcation", "stationary noise"}
    assert INTERPRETATION in r.limits and EarlyWarningAnalyzer.card.role == "validation"
    s_ = r.to_study_result()
    assert s_.metrics["estimate.ac1.n_windows"] > 0
