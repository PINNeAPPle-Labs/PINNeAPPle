"""Visual pack of the near-random forecasting study: render data, 2-D figures, the YouTube aggregation and the
in-process Cycles renderer (skipped without ``bpy``)."""
import numpy as np
import pytest

from pinneapple_systems.time_series import synthetic as syn

pytest.importorskip("sklearn")


def test_youtube_daily_aggregates_sums_per_trending_day():
    from pinneapple_systems.time_series.datasets.public import youtube_daily_aggregates
    csv_text = ("video_id,trending_date,category_id,views,likes\n"
                "a,17.14.11,24,100,5\nb,17.14.11,10,50,1\nc,17.16.11,24,10,0\n")
    y = youtube_daily_aggregates(csv_text)
    assert y.dates == ["2017-11-14", "2017-11-16"]
    assert y.n_entries.tolist() == [2, 1] and y.total_views.tolist() == [150.0, 10.0]
    assert y.category_counts[24].tolist() == [1, 1] and y.category_counts[10].tolist() == [1, 0]
    assert y.missing_days() == ["2017-11-15"]


def test_calendar_grid_places_days_by_week_and_weekday():
    from pinneapple_systems.time_series.forecast_lab.plots import calendar_grid
    cal = calendar_grid(["2018-01-01", "2018-01-03", "2018-01-09"], [1.0, 2.0, 3.0])   # Mon, Wed, Tue next week
    g = cal["grid"]
    assert g.shape == (7, 2) and g[0, 0] == 1.0 and g[2, 0] == 2.0 and g[1, 1] == 3.0
    assert np.isnan(g).sum() == 11 and cal["week_starts"][0] == "2018-01-01"


def test_ensemble_fibres_diverge_for_chaos_and_stay_bundled_for_a_damped_linear_system():
    from pinneapple_systems.time_series.forecast_lab.renders import ensemble_fibres
    s = syn.lorenz63(n=2000, noise=0.0, lyapunov_steps=10)
    lor = ensemble_fibres(syn.ode_stepper(syn.lorenz63_rhs, 0.02, 2), s.state[1500], n_members=20, n_steps=250,
                          scale=s.state.std(axis=0))
    osc = syn.stochastic_oscillator(n=2000, noise=0.0)
    lin = ensemble_fibres(syn.linear_sde_stepper(*syn.oscillator_transition(1.0, 0.05, 0.3, 0.25)), osc.state[1000],
                          n_members=20, n_steps=150, scale=osc.state.std(axis=0))
    assert lor["trajectories"].shape == (251, 20, 2)
    assert lor["spread"][-1] > 20 * lor["spread"][0]            # about 5.6 Lyapunov times of growth
    assert lin["spread"][-1] < lin["spread"][0]                  # same forcing, damped: members converge


def test_embedding_views_rank_the_original_below_its_surrogates():
    from pinneapple_systems.time_series.forecast_lab.renders import embedding_views
    s = syn.lorenz63(n=3000, noise=0.0, lyapunov_steps=10)
    v = embedding_views(s.obs[:, 0], n_points=500)
    assert set(v["embeddings"]) == {"original", "iaaft", "shuffled"}
    assert v["embeddings"]["original"].shape == (500, 3)
    assert v["npe"]["original"] < v["npe"]["iaaft"] < v["npe"]["shuffled"]


def test_dense_forecast_covers_every_horizon_and_reports_coverage():
    from pinneapple_systems.time_series.forecast_lab import GBMForecaster, SeriesData
    from pinneapple_systems.time_series.forecast_lab.renders import dense_forecast
    from pinneapple_systems.time_series.forecast_lab.study import Protocol
    s = syn.ornstein_uhlenbeck(n=1500, noise=0.0, seed=2)
    d = dense_forecast(SeriesData(s.obs, 0, s.dt), Protocol([1], min_history=8, test_stride=5),
                       lambda: GBMForecaster(n_lags=4, max_iter=50), 6)
    assert d["horizons"] == [1, 2, 3, 4, 5, 6] and d["quantiles"].shape[0] == 5
    assert np.all(np.diff(d["quantiles"], axis=0) >= 0)
    assert all(0.6 < c <= 1.0 for c in d["coverage90"])


def test_landscape_tiles_and_surprise_record_from_a_small_zoo():
    from pinneapple_systems.time_series.forecast_lab import (
        ARForecaster,
        Climatology,
        GBMForecaster,
        Persistence,
        SeriesData,
    )
    from pinneapple_systems.time_series.forecast_lab.pipeline import surprise_record
    from pinneapple_systems.time_series.forecast_lab.renders import landscape_tiles
    from pinneapple_systems.time_series.forecast_lab.study import ModelSpec, Protocol, run_zoo
    s = syn.ornstein_uhlenbeck(n=1200, noise=0.0, seed=3)
    dates = [str(np.datetime64("2020-01-01") + i) for i in range(1200)]
    data = SeriesData(s.obs, 0, s.dt, s.name, meta={"dates": dates})
    specs = [ModelSpec("persistence", "baseline", Persistence), ModelSpec("climatology", "baseline", Climatology),
             ModelSpec("ar", "baseline", ARForecaster), ModelSpec("gbm", "data", lambda: GBMForecaster(n_lags=4))]
    zoo = run_zoo(data, specs, Protocol([1, 5], min_history=8), keep_forecasts=["persistence", "climatology", "ar",
                                                                                "gbm"], log=lambda *_: None)
    rec = surprise_record(zoo, data)
    assert rec["model"] == "gbm" and len(rec["dates"]) == len(rec["abs_error_model"]) > 0
    assert rec["dates"][0] == dates[rec["target_index"][0]]
    tiles = landscape_tiles([{"case": "ou", "title": "OU", "truth": {}, "dt": s.dt, "zoo": zoo}])
    assert tiles[0]["families"] == ["data"] and len(tiles[0]["skill"][0]) == 2


def test_two_d_figures_are_written(tmp_path):
    from pinneapple_systems.time_series.forecast_lab import plots
    rng = np.random.default_rng(0)
    rec = {"title": "t", "dates": [str(np.datetime64("2018-01-01") + i) for i in range(40)], "model": "m",
           "baseline": "b", "horizon": 1, "abs_error_model": rng.random(40), "abs_error_baseline": rng.random(40)}
    assert (tmp_path / "c.png").name in plots.plot_surprise_calendar([rec], str(tmp_path / "c.png"))
    mem = {"title": "t", "horizon": 1, "train_index": list(range(10)), "test_index": list(range(10, 20)),
           "truth_train": list(rng.random(10)), "truth_test": list(rng.random(10)),
           "variants": {"a": {"pred_train": list(rng.random(10)), "pred_test": list(rng.random(10)),
                              "rmse": {"train": 0.1, "test": 0.5}}}}
    plots.plot_memorise_vs_learn([mem], str(tmp_path / "m.png"))
    assert (tmp_path / "c.png").stat().st_size > 0 and (tmp_path / "m.png").stat().st_size > 0


def test_cycles_scene_renders_a_tiny_image(tmp_path):
    pytest.importorskip("bpy")
    from pinneapple_blender.cycles import CyclesScene
    sc = CyclesScene(resolution=(48, 32), samples=2)
    x = np.linspace(-1, 1, 8)
    X, Y = np.meshgrid(x, x)
    sc.add_surface(X, Y, X * Y, np.stack([X * 0 + 0.8, X * 0 + 0.3, X * 0 + 0.1], -1))
    sc.add_plane(0.0, (2, 2), alpha=0.4)
    sc.add_tube(np.c_[x, x, x], 0.02, (0.1, 0.2, 0.9))
    sc.add_points(np.zeros((5, 3)), 0.05)
    sc.fit_camera((-1, -1, -1), (1, 1, 1))
    sc.light()
    out = sc.render(str(tmp_path / "t.png"))
    assert (tmp_path / "t.png").stat().st_size > 0 and out.endswith("t.png")
