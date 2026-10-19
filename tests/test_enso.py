"""Library pieces of the ENSO study (pinneapple_systems.time_series.enso): conceptual and intermediate models
checked against theory, data utilities, forecasters, physics-informed networks and the ENSO scores."""
import numpy as np
import pytest

from pinneapple_systems.time_series.enso import data as D
from pinneapple_systems.time_series.enso import evaluation as E
from pinneapple_systems.time_series.enso import intermediate as IM
from pinneapple_systems.time_series.enso import oscillators as O

torch = pytest.importorskip("torch")


# --- conceptual models ------------------------------------------------------

def test_recharge_oscillator_period_and_growth_match_eigenvalues():
    p = O.ROParams.constant(R=-0.05, F1=0.02, F2=1.5, eps=0.03)
    e = O.ro_eigen(**p.mean())
    assert abs(e["growth"] + 0.04) < 1e-12
    x = O.simulate_recharge(p, [1.0, 0.0], 600, sub=4, stochastic=False)[0, 0, :, 0]
    st = O.oscillation_stats(x, 1.0)
    assert abs(st["period"] / e["period"] - 1) < 1e-4
    assert abs(st["growth"] / e["growth"] - 1) < 1e-3


def test_recharge_fit_recovers_seasonal_and_multiplicative_parameters():
    p = O.ROParams({"R": (-0.05, 0.08, 0.03), "F1": (0.02, 0, 0), "F2": (1.5, 0, 0), "eps": (0.03, 0, 0)},
                   sigma_T=0.2, sigma_h=3.0, B=0.3)
    X = O.simulate_recharge(p, [0, 0], 6000, seed=2)[0, 0]
    f = O.fit_recharge(X[:, 0], X[:, 1], np.arange(6001) % 12, seasonal=["R"], multiplicative=True)
    assert abs(f.coef["R"][1] - 0.08) < 0.02 and abs(f.coef["F2"][0] - 1.5) < 0.1
    assert abs(f.coef["F1"][0] - 0.02) < 0.003 and 0.0 < f.B < 0.5
    assert abs(f.sigma_h - 3.0) < 0.3


def test_delayed_oscillator_matches_its_lambert_w_root():
    a, b, tau = 1 / 6, 1 / 3, 6.0
    r = O.do_roots(a, b, tau)[0]
    x = O.simulate_delayed(a, b, 0.0, tau, np.full(25, 0.01), 2400, 0.25)
    st = O.oscillation_stats(x, 0.25, skip=400)
    assert abs(st["period"] / (2 * np.pi / r.imag) - 1) < 1e-3
    assert abs(st["growth"] / r.real - 1) < 0.01


# --- intermediate model -----------------------------------------------------

def test_kelvin_wave_speed_and_second_order_convergence():
    k = IM.kelvin_wave_test()
    assert abs(k["rel_error"]) < 0.01
    conv = IM.convergence_study(factors=(1, 2))
    assert 1.7 < conv["observed_order"][0] < 2.3


def test_long_rossby_wave_moves_west_at_a_third_of_c():
    r = IM.rossby_wave_test()
    assert r["speed_measured"] < 0 and abs(r["rel_error"]) < 0.06


def test_gill_response_converges_to_the_closed_form():
    g = IM.gill_test(nlons=(72, 288))
    e0, e1 = g["rows"]
    assert e1["rel_l2_u"] < e0["rel_l2_u"] and e1["rel_l2_u"] < 1e-3 and e1["rel_l2_p"] < 1e-3
    assert g["pattern"]["u_east_of_heating"] < 0 < g["pattern"]["u_west_of_heating"]   # easterlies east, westerlies west


def test_coupled_model_runs_nudges_and_uses_a_low_rank_gill_operator(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_DATA_CACHE", str(tmp_path))
    g = IM.OceanGrid(dlon=5.0, dlat=2.0, lat_max=20.0)
    p = IM.CoupledParams(gamma=np.full(g.nx, 0.02), kappa=np.full(g.nx, 5.0), mu=0.01)
    m = IM.CoupledModel(p, IM.ShallowWaterOcean(g, dtype="float32"), dt_hours=12.0, atm_every=4)
    T = np.random.default_rng(0).standard_normal((g.ny, g.nx))
    u, v = m.gill_winds(T)
    tx, ty = m.stress(T)
    assert np.allclose(tx, 0.01 * u, atol=1e-5 * np.abs(u).max())
    obs = np.ones((2, g.ny, g.nx))
    s, Tm, Hm = m.run(m.zeros((3,)), 2, T_obs=obs, nudge_days=2.0)
    assert Tm.shape == (2, 3, g.ny, g.nx) and np.abs(Tm[-1] - 1).max() < 0.2


# --- data utilities ---------------------------------------------------------

def test_d20_from_profile_and_month_range():
    z = np.array([0, 50, 100, 150, 200.0])
    T = np.array([28, 26, 22, 18, 15.0])
    assert abs(D.d20_from_profile(z, T) - 125.0) < 1e-9
    assert np.isnan(D.d20_from_profile(z, T - 10))                     # never above 20 C
    assert D.month_range("1997-11", "1998-02") == ["1997-11", "1997-12", "1998-01", "1998-02"]


def test_manifest_raises_on_a_changed_file(tmp_path):
    m = D.Manifest(tmp_path / "m.json")
    m.check("u", b"abc")
    m.save()
    pinned = D.Manifest(tmp_path / "n.json", tmp_path / "m.json")
    with pytest.raises(ValueError):
        pinned.check("u", b"abd")


def test_anomalies_and_box_mean():
    t = D.month_range("2000-01", "2003-12")
    x = np.tile(np.arange(12.0), 4) + 1.0 * (np.arange(48) >= 24)
    a, clim = D.anomalies(x, t, ("2000-01", "2001-12"))
    assert np.allclose(a[:24], 0) and np.allclose(a[24:], 1)
    g = D.Gridded(t[:1], np.array([-1.0, 1.0]), np.array([200.0, 210.0]), {"sst": np.ones((1, 2, 2))})
    assert np.allclose(D.box_mean(g, "sst", (-5, 5, 190, 240)), 1.0)


def test_eof_fill_reconstructs_missing_bins_of_low_rank_data():
    from pinneapple_systems.time_series.enso.dataset import eof_fill
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2)) @ rng.standard_normal((2, 10))
    Y = X.copy()
    Y[150:, 7:] = np.nan
    F = eof_fill(Y, np.arange(200) < 150, n_eofs=2, ridge=1e-6)
    assert np.abs(F[150:, 7:] - X[150:, 7:]).max() < 1e-3


def test_localized_enkf_analysis_matches_the_kalman_update():
    from pinneapple_analysis.state_estimation import enkf_analysis, gaspari_cohn
    rng = np.random.default_rng(0)
    ens = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=40000)
    a = enkf_analysis(ens, np.array([1.0]), ens[:, :1], np.array([0.5]), rng=rng)
    assert np.allclose(a.mean(0), [2 / 3, 1 / 3], atol=0.02)
    assert abs(np.var(a[:, 0]) - 1 / 3) < 0.02
    assert gaspari_cohn(np.array([0.0]))[0] == 1 and gaspari_cohn(np.array([2.0]))[0] == 0


# --- forecasters and scores -------------------------------------------------

@pytest.fixture(scope="module")
def ro_data():
    from pinneapple_systems.time_series.forecast_lab import SeriesData
    p = O.ROParams.constant(R=-0.05, F1=0.15, F2=0.2, eps=0.03, sigma_T=0.2, sigma_h=1.5)
    X = O.simulate_recharge(p, [0, 0], 600, seed=3)[0, 0]
    rng = np.random.default_rng(1)
    pattern = rng.standard_normal((4, 6))
    maps = X[:, 0, None, None] * pattern[None] + 0.05 * rng.standard_normal((len(X), 4, 6))
    meta = {"columns": ["nino34", "wwv"], "month_of_year": np.arange(len(X)) % 12, "sst_maps": maps,
            "grid": IM.OceanGrid(lon0=180, lon1=240, dlon=10, lat_max=2, dlat=1)}
    return SeriesData(X, 0, 1.0, "ro", meta)


def test_recharge_lim_and_ensemble_forecasters_beat_persistence(ro_data):
    from pinneapple_systems.time_series.enso import forecasters as F
    from pinneapple_systems.time_series.forecast_lab import Persistence
    from pinneapple_systems.time_series.forecast_lab.study import Protocol, evaluate
    H = [1, 6, 12]
    sp = Protocol(H, min_history=12).final_split(len(ro_data))
    pers = evaluate(Persistence().fit(ro_data, sp["train"], sp["val"], H), ro_data, sp["test"])["rmse"]
    models = [F.RechargeOscillatorForecaster(stochastic=False), F.RechargeOscillatorForecaster(n_members=50),
              F.LIMForecaster(n_eofs=2), F.EnsembleForecaster([F.RechargeOscillatorForecaster(stochastic=False),
                                                               F.LIMForecaster(n_eofs=2)], n_draw=20)]
    for m in models:
        ev = evaluate(m.fit(ro_data, sp["train"], sp["val"], H), ro_data, sp["test"])
        assert ev["rmse"][1] < 0.8 * pers[1], m.name
        assert ev["crps"][1] < ev["rmse"][1]


def test_column_subset_runs_a_forecast_lab_ode_on_two_columns(ro_data):
    from pinneapple_systems.time_series.enso import forecasters as F
    from pinneapple_systems.time_series.forecast_lab import SeriesData
    from pinneapple_systems.time_series.forecast_lab.physical import ODEForecaster
    from pinneapple_systems.time_series.forecast_lab.study import Protocol
    data = SeriesData(np.column_stack([ro_data.obs, ro_data.obs[:, :1] ** 2]), 0, 1.0, "x",
                      {**ro_data.meta, "columns": ["nino34", "wwv", "junk"]})
    m = F.ColumnSubset(ODEForecaster(F.ro_rhs, {"R": -0.1, "F1": 0.1, "F2": 0.1, "eps": 0.1}), ["nino34", "wwv"])
    sp = Protocol([1, 6], min_history=12).final_split(len(data))
    m.fit(data, sp["train"], sp["val"], [1, 6])
    assert abs(m.inner.params_["F2"] - 0.2) < 0.05
    assert m.predict(data, sp["test"]).mean.shape == (len(sp["test"]), 2)


def test_pinn_identifies_the_recharge_oscillator_and_reports_residuals(ro_data):
    from pinneapple_systems.time_series.enso import pinn as PI
    r = PI.identify_recharge(ro_data.obs[:240, 0], ro_data.obs[:240, 1], epochs=300, n_col=256, closure=True)
    p = r["params"].mean()
    assert all(np.isfinite(list(p.values()))) and p["F1"] > 0 and p["F2"] > 0
    assert len(r["residual_rms_per_sd"]) == 2 and r["closure"](np.array([0.0, 1.0])).shape == (2,)


def test_pinn_extrapolation_returns_one_forecast_per_origin_and_horizon(ro_data):
    from pinneapple_systems.time_series.enso.pinn import PINNExtrapolationForecaster
    from pinneapple_systems.time_series.forecast_lab.study import Protocol
    sp = Protocol([1, 3], min_history=12).final_split(len(ro_data))
    m = PINNExtrapolationForecaster(window=12, epochs=60, n_col=64)
    m.fit(ro_data, sp["train"], None, [1, 3])
    f = m.predict(ro_data, sp["test"][:5])
    assert f.mean.shape == (5, 2) and np.isfinite(f.mean).all()
    assert "residual_rms_forecast_part" in m.info["chunks"][0]


def test_equatorial_wave_pinn_trains_and_reports_every_residual():
    from pinneapple_systems.time_series.enso.pinn import EquatorialWavePINN, project_forcing
    rng = np.random.default_rng(0)
    lat = np.arange(-19.5, 20, 1.0)
    lon_f = np.arange(120, 281, 4.0)
    tx = 0.01 * rng.standard_normal((6, len(lat), len(lon_f)))
    X0, X2 = project_forcing(tx, lat)
    lon_o = np.arange(135, 280, 10.0)
    pin = EquatorialWavePINN(lon_o, rng.standard_normal((6, len(lon_o))), rng.standard_normal((6, len(lon_o))),
                             lon_f, X0, X2, tx[:, 19:21].mean(1), hidden=16, layers=2, n_freq=2)
    out = pin.fit(epochs=30, n_col=128)
    assert set(out["residual_rms"]) == {"kelvin", "rossby", "east_bc", "west_bc", "sst"}
    assert pin.predict_grid(lon_o)["h"].shape == (6, len(lon_o))


def test_categorical_and_target_month_scores():
    y = np.array([[1.0], [-1.0], [0.0], [0.7], [-0.6], [0.2]])
    f = np.array([[0.8], [-0.2], [0.1], [0.6], [-0.9], [0.9]])
    c = E.categorical(f, y)["per_horizon"][0]
    assert np.array(c["table"]).sum() == 6 and c["hit_rate"]["el_nino"] == 1.0
    assert abs(c["hit_rate"]["la_nina"] - 0.5) < 1e-12
    r = E.by_target_month(np.tile(f, (4, 1)), np.tile(y, (4, 1)), np.arange(24), np.arange(24) % 12, [1], min_n=2)
    assert np.asarray(r["acc"]).shape == (12, 1)


def test_merge_zoo_equals_one_run():
    from pinneapple_systems.time_series import synthetic as syn
    from pinneapple_systems.time_series.forecast_lab import (
        ARForecaster,
        Climatology,
        Persistence,
        SeriesData,
    )
    from pinneapple_systems.time_series.forecast_lab.study import (
        ModelSpec,
        Protocol,
        merge_zoo,
        run_zoo,
    )
    s = syn.ornstein_uhlenbeck(n=800, noise=0.0, seed=4)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    specs = [ModelSpec("persistence", "baseline", Persistence), ModelSpec("climatology", "baseline", Climatology),
             ModelSpec("ar", "data", ARForecaster)]
    P = Protocol([1, 5], min_history=8, n_folds=2)
    q = lambda *_: None  # noqa: E731
    whole = run_zoo(data, specs, P, keep_forecasts=[x.name for x in specs], log=q)
    merged = merge_zoo([run_zoo(data, [x], P, keep_forecasts=[x.name], log=q) for x in specs])
    a, b = whole["models"]["ar"], merged["models"]["ar"]
    assert np.allclose(a["test"]["skill_rmse"], b["test"]["skill_rmse"])
    assert np.allclose([d["p_holm"] for d in a["dm_vs_persistence"]], [d["p_holm"] for d in b["dm_vs_persistence"]])


def test_slab_scene_has_the_surfaces_and_the_sst_field():
    from pinneapple_systems.time_series.enso.render import slab_scene
    lon, lat = np.arange(121, 280, 2.0), np.arange(-19.5, 20, 1.0)
    sc = slab_scene(lon, lat, 25 + np.random.default_rng(0).random((len(lat), len(lon))),
                    np.full((len(lat), len(lon)), 120.0))
    names = [s.name for s in sc.surfaces]
    assert names == ["sea_surface", "thermocline", "equator_curtain", "floor"]
    assert "sst" in sc.surfaces[0].fields and len(sc.surfaces[0].fields["sst"]) == len(sc.surfaces[0].vertices)


def test_skill_against_climatology_tests_every_model_with_holm():
    from pinneapple_systems.time_series import synthetic as syn
    from pinneapple_systems.time_series.enso.pipeline import skill_against
    from pinneapple_systems.time_series.forecast_lab import (
        ARForecaster,
        Climatology,
        Persistence,
        SeriesData,
    )
    from pinneapple_systems.time_series.forecast_lab.study import ModelSpec, Protocol, run_zoo
    s = syn.ornstein_uhlenbeck(n=800, noise=0.0, seed=4)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    specs = [ModelSpec("persistence", "baseline", Persistence), ModelSpec("climatology", "baseline", Climatology),
             ModelSpec("ar", "data", ARForecaster)]
    z = run_zoo(data, specs, Protocol([1, 5], min_history=8, n_folds=2), keep_forecasts=[x.name for x in specs],
                log=lambda *_: None)
    out = skill_against(z, "climatology")
    assert out["models"]["climatology"]["skill_rmse"] == [0.0, 0.0]
    ar = out["models"]["ar"]
    assert ar["skill_rmse"][0] > 0.3 and ar["significant"][0] and len(ar["p_holm"]) == 2      # OU dt = 0.1: AR(1)


def test_slab_scene_colours_the_thermocline_by_its_depth_anomaly():
    from pinneapple_systems.time_series.enso.render import slab_scene
    lon, lat = np.arange(121, 280, 2.0), np.arange(-19.5, 20, 1.0)
    sst = 25 + np.zeros((len(lat), len(lon)))
    sst[19, 3] = np.nan                                                   # a land cell on the band's wall row
    sc = slab_scene(lon, lat, sst, np.full((len(lat), len(lon)), 120.0),
                    d20_anom=np.linspace(-30, 30, len(lon))[None].repeat(len(lat), 0))
    th = next(x for x in sc.surfaces if x.name == "thermocline")
    assert "dz" in th.fields and th.fields["dz"].min() == -30
    wall = next(x for x in sc.surfaces if x.name == "equator_curtain")
    assert np.isfinite(wall.fields["sst"]).all() and np.allclose(wall.fields["sst"], 25)
