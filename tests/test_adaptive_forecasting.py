"""Adaptive forecasting: online model switching/combination without look-ahead (prequential)."""
import numpy as np
import pytest

from pinneapple_systems.time_series.adaptive import (
    AdaHedge,
    AdaptiveForecaster,
    FixedShare,
    HoltDamped,
    HoltWintersAdditive,
    LagRegressorExpert,
    RidgeAR,
    SimpleExpSmoothing,
    Theta,
    default_experts,
)


def regime_series(seed=0, n=900):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.empty(n)
    a, b = n // 3, 2 * n // 3
    y[:a] = 10 + 5 * np.sin(2 * np.pi * t[:a] / 12) + rng.normal(0, 0.5, a)          # seasonal
    y[a:b] = y[a - 1] + 0.4 * np.arange(1, b - a + 1) + rng.normal(0, 0.5, b - a)     # trend
    y[b:] = y[b - 1] + np.cumsum(rng.normal(0, 1.5, n - b))                           # random walk
    return y


@pytest.fixture(scope="module")
def run():
    return AdaptiveForecaster(default_experts(12), horizon=3).run(regime_series(), start=60)


def test_ensemble_beats_every_single_model_across_regime_changes(run):
    for h in (1, 3):
        e = run.errors(h)
        assert e["ensemble"]["mae"] < 0.95 * e["best_single_in_hindsight"]["mae"]


def test_weights_move_to_the_model_that_fits_each_regime(run):
    def top(a, b):
        m = (run.origins >= a) & (run.origins < b)
        w = run.weights[m].mean(0)
        return {run.expert_names[i] for i in np.argsort(-w)[:2]}
    assert top(150, 300) & {"ar", "holt_winters", "seasonal_naive"}
    assert top(400, 600) & {"holt_winters", "drift", "holt_damped", "ar"}
    assert "naive" in top(700, 900) or "ses_0.7" in top(700, 900)
    assert len(run.switches()) > 1


def test_no_look_ahead_forecasts_ignore_future_data():
    y = regime_series(1, 400)
    y2 = y.copy()
    y2[250:] += 1000.0
    r1 = AdaptiveForecaster(default_experts(12), horizon=2).run(y, start=40)
    r2 = AdaptiveForecaster(default_experts(12), horizon=2).run(y2, start=40)
    k = int(np.searchsorted(r1.origins, 249))
    np.testing.assert_array_equal(r1.forecast[: k + 1], r2.forecast[: k + 1])
    np.testing.assert_array_equal(r1.weights[: k + 1], r2.weights[: k + 1])
    assert not np.allclose(r1.forecast[k + 1], r2.forecast[k + 1])


def test_matches_the_best_model_when_one_model_is_right():
    rng = np.random.default_rng(2)
    t = np.arange(500)
    y = 10 + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 500)
    e = AdaptiveForecaster(default_experts(12), horizon=1).run(y, start=60).errors(1)
    assert e["ensemble"]["mae"] < 1.08 * e["best_single_in_hindsight"]["mae"]


def test_adaptive_conformal_interval_keeps_coverage_under_shift(run):
    for h in (1, 3):
        assert abs(run.coverage(h) - 0.9) < 0.04
    assert np.all(run.upper[-1] > run.lower[-1])


def test_select_mode_switches_and_stays_competitive(run):
    rs = AdaptiveForecaster(default_experts(12), horizon=1, mode="select").run(regime_series(), start=60)
    assert len(rs.switches()) > 2
    e = rs.errors(1)
    assert e["ensemble"]["mae"] < e["best_single_in_hindsight"]["mae"]
    f = AdaptiveForecaster(default_experts(12), horizon=1, mode="select")
    for v in regime_series()[:100]:
        f.update(v)
    out = f.forecast()
    assert out["forecast"][0] == out["expert_forecasts"][out["active"]][0]


def test_failing_or_external_experts():
    class Broken:
        def fit(self, y):
            raise RuntimeError("boom")

        def predict(self, h):
            return np.zeros(h)

    sklearn = pytest.importorskip("sklearn.linear_model")
    ex = {"ses": SimpleExpSmoothing(0.3), "broken": Broken(),
          "ridge_lags": LagRegressorExpert(sklearn.Ridge(alpha=1.0), n_lags=12, refit_every=20)}
    f = AdaptiveForecaster(ex, horizon=2)
    r = f.run(regime_series(3, 300), start=40)
    assert f.failures["broken"] > 0 and np.all(np.isfinite(r.forecast))
    assert r.weights[-1][r.expert_names.index("broken")] < 0.2
    with pytest.raises(RuntimeError):
        AdaptiveForecaster(ex, horizon=2, on_error="raise").run(regime_series(3, 100), start=40)


def test_fast_experts_on_series_they_should_get_exactly():
    t = np.arange(120, dtype=float)
    line = 3 + 0.5 * t
    np.testing.assert_allclose(HoltDamped(0.5, 0.5, 1.0).fit(line).predict(3), 3 + 0.5 * np.arange(120, 123), atol=1e-6)
    np.testing.assert_allclose(Theta(0.5).fit(np.full(50, 7.0)).predict(4), 7.0)
    np.testing.assert_allclose(SimpleExpSmoothing(0.3).fit(np.full(30, 2.0)).predict(2), 2.0)
    seas = 5 + np.tile([1.0, -1.0, 2.0, 0.0], 30)
    np.testing.assert_allclose(HoltWintersAdditive(4, 0.3, 0.1, 0.3).fit(seas).predict(4), seas[:4], atol=1e-6)
    ar = 10 + np.sin(2 * np.pi * t / 12)
    np.testing.assert_allclose(RidgeAR(p=12, ridge=1e-10).fit(ar).predict(5), 10 + np.sin(2 * np.pi * np.arange(120, 125) / 12), atol=1e-4)


def test_aggregators_basic_properties():
    fs = FixedShare(3, eta=2.0, alpha=0.1)
    for _ in range(50):
        fs.update(np.array([0.0, 1.0, 1.0]))
    assert fs.w[0] > 0.8 and fs.w.min() >= 0.1 / 3 - 1e-12        # fixed share keeps every model reachable
    ah = AdaHedge(2)
    for _ in range(20):
        ah.update(np.array([0.0, 1.0]))
    assert ah.weights()[0] > 0.999                                 # follows the leader while it never errs
    ah.update(np.array([1.0, 0.0]))
    assert 0 < ah.weights()[1] < 0.5
