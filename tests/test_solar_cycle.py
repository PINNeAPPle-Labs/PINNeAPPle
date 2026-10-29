"""pinneapple_systems.time_series.solar_cycle on a synthetic SILSO-shaped series (the source app's tests/test_pipeline.py,
without its SILSO fixture, which is CC BY-NC and not shipped here). The live SILSO download runs only with
PINNEAPPLE_NETWORK=1; the ``tfc`` basis only when ``tfc`` is installed."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pandas")

from pinneapple_systems.time_series import solar_cycle as sc  # noqa: E402

BACKENDS = ["pinneapple"] + (["pyemd"] if sc._pyemd() else [])


@pytest.fixture(scope="module")
def series():
    return sc.synthetic_daily(seed=3)


def test_synthetic_series_has_the_silso_shape(series):
    assert len(series) > 7000 and (series["Sunspot"] >= 0).all() and series.index.is_monotonic_increasing
    csv = series[sc.SILSO_COLUMNS].to_csv(sep=";", header=False, index=False).encode()
    back = sc.parse_silso_csv(csv)
    assert len(back) == len(series) and back.index[0].year == 2005


def test_parse_drops_missing_days():
    raw = b"2020;1;1;2020.001;-1;-1.0;0;1\n2020;1;2;2020.004;12;1.0;5;1\n"
    df = sc.parse_silso_csv(raw)
    assert len(df) == 1 and df["Sunspot"].iloc[0] == 12


def test_preprocess_round_trips(series):
    t, y, (lo, hi) = sc.preprocess(series)
    assert t[0] == 0.0 and t[-1] == 1.0 and y.min() == 0.0 and y.max() == 1.0
    np.testing.assert_allclose(lo + y * (hi - lo), series["Sunspot"].to_numpy(dtype=float), atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_emd_rows_sum_to_the_signal(series, backend):
    _, y, _ = sc.preprocess(series)
    dec = sc.decompose(y, "emd", max_imf=6, backend=backend)
    assert dec.imfs.shape[0] >= 2 and dec.imfs.shape[1] == y.size
    np.testing.assert_allclose(dec.imfs.sum(axis=0), y, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eemd_small_ensemble_reconstructs_on_average(series, backend):
    _, y, _ = sc.preprocess(series)
    dec = sc.decompose(y[:3000], "eemd", max_imf=5, trials=4, noise_width=0.05, seed=0, backend=backend)
    assert dec.method == "eemd" and float(np.mean(np.abs(dec.imfs.sum(axis=0) - y[:3000]))) < 0.06


@pytest.mark.parametrize("backend", BACKENDS)
def test_index_zero_is_not_the_eleven_year_cycle(series, backend):
    """The source app's finding: the IMF near 11 years is not index 0 (index 0 is days-scale noise)."""
    _, y, _ = sc.preprocess(series)
    stats = sc.imf_statistics(sc.decompose(y, "emd", max_imf=6, backend=backend).imfs, dt_days=1.0)
    assert stats[0]["dominant_period_days"] < 30
    best, warning = sc.closest_to_schwabe_cycle(stats)
    assert best["imf_index"] != 0
    dist = [abs(s["dominant_period_days"] - sc.SCHWABE_CYCLE_DAYS) for s in stats]
    assert best["imf_index"] == int(np.argmin(dist))
    assert (warning is not None) == (min(dist) / sc.SCHWABE_CYCLE_DAYS > 0.4)


def test_hilbert_amplitude_is_scipy_abs_hilbert():
    from scipy.signal import hilbert

    t = np.linspace(0, 1, 500)
    imf = np.sin(2 * np.pi * 7 * t)
    np.testing.assert_allclose(sc.hilbert_analysis(imf, t).amplitude, np.abs(hilbert(imf)))


def test_end_point_constraint_holds_on_non_periodic_data():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 300)
    y = 0.2 + 0.6 * t + 0.02 * rng.standard_normal(300)
    fit = sc.periodic_elm_fit(t, y, hidden_units=40, seed=0)
    assert fit.endpoint_error < 1e-9 and fit.rmse < 0.2 and fit.forecast.shape == fit.forecast_t.shape


def test_periodic_signal_is_fitted_closely():
    t = np.linspace(0, 1, 400)
    fit = sc.periodic_elm_fit(t, 0.5 + 0.3 * np.sin(2 * np.pi * t), hidden_units=60, seed=1)
    assert fit.rmse < 0.02 and fit.endpoint_error < 1e-9


def test_run_pipeline_uses_the_closest_imf(series):
    """On this synthetic series the closest IMF is often far from 11 years (and warned about); on the real
    2005-2025 SILSO slice PyEMD's IMF 6 at 9.8 years was picked (validation run, data not shipped)."""
    out = sc.run_pipeline(series, max_imf=6, hidden_units=40, forecast_horizon_frac=0.5, seed=1)
    best, warning = sc.closest_to_schwabe_cycle(out["imf_statistics"])
    assert out["imf_index_used"] == best["imf_index"] != 0 and out["auto_select_warning"] == warning
    assert out["endpoint_constraint_error"] < 1e-9 and out["forecast_amplitude"].shape == out["forecast_t"].shape
    assert sc.run_pipeline(series, max_imf=6, imf_index=0, hidden_units=20)["imf_index_used"] == 0
    with pytest.raises(ValueError):
        sc.run_pipeline(series, max_imf=4, imf_index=99)


def test_tfc_basis_when_installed():
    pytest.importorskip("tfc")
    t = np.linspace(0, 1, 200)
    fit = sc.periodic_elm_fit(t, 0.5 + 0.3 * np.sin(2 * np.pi * t), hidden_units=40, backend="tfc")
    assert fit.endpoint_error < 1e-6 and fit.backend == "tfc"


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_NETWORK"), reason="network (set PINNEAPPLE_NETWORK=1)")
def test_live_silso_download():
    df = sc.fetch_silso_daily()
    assert len(df) > 70_000 and df.index[0].year == 1818
