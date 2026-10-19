"""Library pieces of the near-random forecasting study: generators with known ground truth, predictability
measures, surrogates, purged splits, forecast comparison tests and the forecaster zoo."""
import numpy as np
import pytest

from pinneapple_systems.time_series import predictability as P
from pinneapple_systems.time_series import synthetic as syn
from pinneapple_systems.time_series.metrics_ext import (
    benjamini_hochberg,
    diebold_mariano,
    holm,
    pit_ensemble,
    pit_gaussian,
    pit_histogram,
    skill_score,
)
from pinneapple_systems.time_series.validation import (
    PurgedKFoldSplitter,
    PurgedWalkForwardSplitter,
    ShuffledKFoldSplitter,
)

pytest.importorskip("sklearn")


# --- ground truth -----------------------------------------------------------

def test_lorenz63_lyapunov_matches_literature():
    # largest exponent of Lorenz-63 at the classical parameters is 0.9056 (Sprott 2003, Chaos and Time-Series
    # Analysis, Table A.1); a short Benettin run gets within 10 %
    s = syn.lorenz63(n=200, lyapunov_steps=20000)
    assert abs(s.lyapunov - 0.9056) / 0.9056 < 0.10
    assert s.obs.shape == (200, 3) and np.all(s.noise_std > 0)


def test_ou_exact_sampling_has_the_right_ar1_coefficient_and_variance():
    s = syn.ornstein_uhlenbeck(n=40000, dt=0.1, theta=1.0, sigma=1.0, noise=0.0, seed=1)
    x = s.state[:, 0]
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1] - np.exp(-0.1)) < 0.01
    assert abs(x.var() - 0.5) < 0.05                                        # sigma^2 / (2 theta)


def test_oscillator_transition_conserves_stationary_variance():
    Phi, Q = syn.oscillator_transition(1.0, 0.1, 0.5, 0.25)
    # stationary covariance of the continuous model: var(x) = sigma^2 / (4 zeta omega^3), var(v) = omega^2 var(x)
    S = np.diag([0.25 / (4 * 0.1), 0.25 / (4 * 0.1)])
    assert np.allclose(Phi @ S @ Phi.T + Q, S, atol=1e-10)


def test_vector_fields_accept_numpy_and_torch():
    torch = pytest.importorskip("torch")
    x = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(syn.lorenz63_rhs(x), syn.lorenz63_rhs(torch.tensor(x)).numpy())
    y = np.arange(8.0)[None]
    assert np.allclose(syn.lorenz96_rhs(y), syn.lorenz96_rhs(torch.tensor(y)).numpy())


# --- predictability ---------------------------------------------------------

def test_ar_fit_recovers_order_and_coefficients():
    rng = np.random.default_rng(0)
    x = np.zeros(5000)
    for t in range(2, 5000):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + rng.standard_normal()
    f = P.fit_ar(x, max_order=8, criterion="bic")
    assert f.order == 2 and np.allclose(f.coef, [0.6, -0.3], atol=0.05)


def test_entropies_separate_noise_from_a_sine():
    rng = np.random.default_rng(0)
    wn = rng.standard_normal(3000)
    sine = np.sin(np.arange(3000) * 0.1)
    assert P.permutation_entropy(wn) > 0.98 and P.permutation_entropy(sine) < 0.3
    assert P.sample_entropy(wn) > 2 * P.sample_entropy(sine)


def test_iaaft_keeps_amplitudes_and_spectrum():
    rng = np.random.default_rng(0)
    x = syn.lorenz63(n=2048, noise=0.0, lyapunov_steps=10).obs[:, 0]
    s = P.iaaft_surrogate(x, rng)
    assert np.allclose(np.sort(s), np.sort(x))
    a, b = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s))
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.1


def test_surrogate_test_rejects_iaaft_for_chaos_but_not_for_a_linear_process():
    lor = syn.lorenz63(n=3000, noise=0.0, lyapunov_steps=10).obs[:, 0]
    ou = syn.ornstein_uhlenbeck(n=3000, noise=0.0).obs[:, 0]
    npe = lambda s: P.nonlinear_prediction_error(s, dim=4, delay=3)  # noqa: E731
    assert P.surrogate_test(lor, npe, kind="iaaft", n_surrogates=19).p_value <= 0.05
    assert P.surrogate_test(ou, npe, kind="iaaft", n_surrogates=19).p_value > 0.05
    assert P.surrogate_test(ou, npe, kind="shuffle", n_surrogates=19).p_value <= 0.05


def test_rosenstein_estimates_the_lorenz_exponent():
    s = syn.lorenz63(n=5000, noise=0.0, lyapunov_steps=10)
    r = P.rosenstein_lyapunov(s.obs[:, 0], dim=5, delay=3, dt=s.dt)
    assert 0.6 < r["lyapunov"] < 1.3                                     # true 0.906; the fit window matters


# --- splits and comparisons ---------------------------------------------------

def test_purged_walk_forward_keeps_training_labels_out_of_the_validation_block():
    t = np.arange(1000)
    for s in PurgedWalkForwardSplitter(n_folds=3, val_size=100, purge=16).split(t):
        assert max(t[s.train_idx]) + 16 < t[s.val_idx[0]]
    folds = list(PurgedKFoldSplitter(n_folds=5, purge=10, embargo=10).split(t))
    for s in folds:
        lo, hi = t[s.val_idx[0]], t[s.val_idx[-1]]
        tr = t[s.train_idx]
        assert not np.any((tr >= lo - 10) & (tr <= hi + 10))
    sh = list(ShuffledKFoldSplitter(n_folds=4).split(t))
    assert sorted(np.concatenate([s.val_idx for s in sh]).tolist()) == list(range(1000))


def test_diebold_mariano_detects_a_better_forecast_and_holds_size():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(800)
    r = diebold_mariano(0.5 * y + 0.2 * rng.standard_normal(800), y, h=1)
    assert r.p_value_better < 1e-3
    rej = np.mean([diebold_mariano(rng.standard_normal(300), rng.standard_normal(300), h=3).p_value < 0.05
                   for _ in range(400)])
    assert rej < 0.09                                                      # nominal 5 %


def test_multiple_comparison_adjustments():
    p = [0.01, 0.04, 0.03, 0.5]
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 0.5])
    assert np.allclose(benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.5])
    assert skill_score(1.0, 2.0) == 0.5


def test_pit_is_uniform_for_calibrated_forecasts():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    assert pit_histogram(pit_gaussian(0.0, 1.0, y))["deviation"] < 0.05
    assert pit_histogram(pit_gaussian(0.0, 0.5, y))["deviation"] > 0.3                    # too narrow
    ens = rng.standard_normal((50, 20000))
    assert pit_histogram(pit_ensemble(ens, y))["deviation"] < 0.05


# --- forecasters --------------------------------------------------------------

@pytest.fixture(scope="module")
def lorenz():
    from pinneapple_systems.time_series.forecast_lab import SeriesData, forecast_origins
    s = syn.lorenz63(n=2400, noise=0.05, lyapunov_steps=10, seed=3)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    H = [1, 4, 16]
    return data, H, forecast_origins(0, 1500, H, min_history=16), forecast_origins(1500, 1900, H), \
        forecast_origins(1900, 2400, H, stride=4)


def _skill(m, data, te, H):
    from pinneapple_systems.time_series.forecast_lab import targets_at
    y = targets_at(data, te, H)
    f = m.predict(data, te)
    pers = np.sqrt(np.mean((data.y[te][:, None] - y) ** 2, axis=0))
    return 1 - np.sqrt(np.mean((f.mean - y) ** 2, axis=0)) / pers, f, y


def test_fitted_lorenz_ode_recovers_parameters_and_beats_persistence(lorenz):
    from pinneapple_systems.time_series.forecast_lab import ODEForecaster
    data, H, tr, va, te = lorenz
    m = ODEForecaster(syn.lorenz63_rhs, {"sigma": 8.0, "rho": 22.0, "beta": 2.0}).fit(data, tr, va, H)
    p = m.params_
    assert abs(p["sigma"] - 10) < 0.5 and abs(p["rho"] - 28) < 1 and abs(p["beta"] - 8 / 3) < 0.2
    sk, f, y = _skill(m, data, te, H)
    assert np.all(sk > 0.3)
    assert f.std is not None and np.all(f.crps(y) >= 0)


def test_data_driven_and_hybrid_models_fit_and_beat_persistence_at_one_step(lorenz):
    from pinneapple_systems.time_series.forecast_lab import (
        GBMForecaster,
        NeuralForecaster,
        ODEForecaster,
        ResidualCorrectionForecaster,
    )
    from pinneapple_systems.time_series.forecast_lab.presets import lorenz63_missing_xz_rhs
    data, H, tr, va, te = lorenz
    for m in (GBMForecaster(n_lags=4, max_iter=100),
              NeuralForecaster("mlp", n_lags=4, hidden=32, epochs=20, patience=5),
              ResidualCorrectionForecaster(ODEForecaster(lorenz63_missing_xz_rhs, {"sigma": 8., "rho": 22.,
                                                                                   "beta": 2.}), "gbm", n_lags=4)):
        m.fit(data, tr, va, H)
        sk, _, _ = _skill(m, data, te, H)
        assert sk[0] > 0.2, m.name
        assert m.n_params() > 0


def test_kalman_ssm_recovers_oscillator_parameters():
    from pinneapple_systems.time_series.forecast_lab import (
        LinearSSMForecaster,
        SeriesData,
        forecast_origins,
    )
    s = syn.stochastic_oscillator(n=4000, omega=1.0, zeta=0.05, sigma=0.3, noise=0.1, seed=2)
    data = SeriesData(s.obs, 0, s.dt)
    H = [1, 8]
    m = LinearSSMForecaster("oscillator").fit(data, forecast_origins(0, 3000, H, min_history=2), None, H)
    assert abs(m.params_["omega"] - 1.0) < 0.05 and abs(m.params_["zeta"] - 0.05) < 0.03
    f = m.predict(data, forecast_origins(3000, 4000, H, stride=5))
    assert f.std.shape == f.mean.shape


def test_protocol_never_lets_training_or_validation_labels_reach_the_test_period():
    from pinneapple_systems.time_series.forecast_lab.study import Protocol
    pr = Protocol([1, 5, 20], min_history=10)
    s = pr.final_split(3000)
    assert s["train"].max() + 20 < s["val_start"]
    assert s["val"].max() + 20 < s["dev_end"] <= s["test"].min()
    for f in pr.cv_folds(3000):
        assert f["train"].max() + 20 < f["inner_val"].min()
        assert f["inner_val"].max() < f["val"].min() and f["val"].max() + 20 < s["dev_end"]


def test_run_zoo_scores_and_tests_a_small_zoo():
    from pinneapple_systems.time_series.forecast_lab import (
        ARForecaster,
        Climatology,
        GBMForecaster,
        Persistence,
        SeriesData,
    )
    from pinneapple_systems.time_series.forecast_lab.study import ModelSpec, Protocol, run_zoo
    s = syn.ornstein_uhlenbeck(n=2000, noise=0.0, seed=4)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    specs = [ModelSpec("persistence", "baseline", Persistence), ModelSpec("climatology", "baseline", Climatology),
             ModelSpec("ar", "baseline", ARForecaster),
             ModelSpec("gbm", "data", lambda **k: GBMForecaster(**k), [{"n_lags": 4}, {"n_lags": 8}])]
    res = run_zoo(data, specs, Protocol([1, 10], min_history=8, n_folds=2), log=lambda *_: None)
    ar = res["models"]["ar"]
    assert ar["test"]["skill_rmse"][1] > 0.1                   # mean reversion beats persistence at h = 10
    assert res["models"]["gbm"]["n_configs"] == 2
    assert "p_holm" in res["models"]["gbm"]["dm_vs_persistence"][0]
    assert set(res["family_representative"]) == {"baseline", "data"}


def test_periodic_baselines_follow_a_drifting_cycle():
    from pinneapple_systems.time_series.forecast_lab import (
        PhaseClimatology,
        SeasonalNaive,
        SeriesData,
        dominant_period,
        forecast_origins,
        targets_at,
    )
    rng = np.random.default_rng(0)
    t = np.arange(4000)
    phase = 2 * np.pi * t / 12.5 + np.cumsum(0.02 * rng.standard_normal(4000))   # period 12.5, drifting phase
    y = np.sin(phase) + 0.1 * rng.standard_normal(4000)
    assert abs(dominant_period(y[:3000]) - 12.5) < 0.3
    data = SeriesData(y[:, None])
    H = [1, 6, 40]
    tr, te = forecast_origins(0, 3000, H, min_history=40), forecast_origins(3000, 4000, H, stride=3)
    yt = targets_at(data, te, H)
    clim = np.sqrt(np.mean(yt ** 2, axis=0))                                       # mean-zero climatology
    for m, upto in ((SeasonalNaive(), 2), (PhaseClimatology(fit_cycles=2), 3)):
        # seasonal naive looks back k whole (integer) periods, so it only works within about one period
        m.fit(data, tr, None, H)
        rmse = np.sqrt(np.mean((m.predict(data, te).mean - yt) ** 2, axis=0))
        assert np.all(rmse[:upto] < 0.6 * clim[:upto]), (m.name, rmse, clim)
    assert SeasonalNaive().fit(data, tr, None, H).info["period"] in (12, 13)


def test_mackey_glass_lyapunov_is_positive_and_converging():
    from pinneapple_systems.time_series.forecast_lab.pipeline import lyapunov_study
    st = lyapunov_study(dt_ints=(0.1,), total_times=(2000.0,), seeds=(0, 1), log=lambda *_: None)
    row = st["rows"][0]
    assert all(0.002 < v < 0.009 for v in row["values"]) and row["std"] < 0.002
