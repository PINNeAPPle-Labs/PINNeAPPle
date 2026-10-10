"""Global weather forecasting pieces that do not need the network or a trained model: spherical padding, forcing,
regridding, scores, horizons, events and the visuals."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_physics.weather.evaluate import acc, first_crossing, horizons, rmse  # noqa: E402
from pinneapple_physics.weather.events import EVENTS, coarsen, region_mask  # noqa: E402
from pinneapple_physics.weather.model import (  # noqa: E402
    SphereUNet,
    sphere_pad,
    static_fields,
    toa_insolation,
)

LAT = np.linspace(87.1875, -87.1875, 32)
LON = np.arange(64) * 5.625


def test_sphere_pad_is_periodic_in_longitude_and_crosses_the_poles():
    x = torch.arange(32 * 64, dtype=torch.float32).reshape(1, 1, 32, 64)
    p = sphere_pad(x, 1)
    assert p.shape == (1, 1, 34, 66)
    assert torch.equal(p[0, 0, 1:-1, 0], x[0, 0, :, -1]) and torch.equal(p[0, 0, 1:-1, -1], x[0, 0, :, 0])
    # the row beyond the north pole is the first row on the other side of the pole (half a circle away)
    assert torch.equal(p[0, 0, 0, 1:-1], torch.roll(x[0, 0, 0], 32))


def test_network_starts_as_persistence_and_keeps_the_shape():
    m = SphereUNet(5, 7, widths=(16, 32), blocks=1, patch=2, refine=8)
    x = torch.randn(2, 5, 32, 64)
    y = m(x, x, torch.randn(2, 7, 32, 64))
    assert y.shape == x.shape and torch.allclose(y, x)          # zero-initialised output layer


def test_insolation_follows_the_sun():
    s = toa_insolation(np.array(["2020-06-21T12"], dtype="datetime64[ns]"), LAT, LON)[0]
    assert s.min() == 0 and 0.95 < s.max() <= 1
    i, j = np.unravel_index(np.argmax(s), s.shape)
    assert 15 < LAT[i] < 30 and (LON[j] < 15 or LON[j] > 345)     # sub-solar point near (23N, 0E) at noon UTC
    assert static_fields(LAT, LON, np.zeros((2, 32, 64))).shape == (5, 32, 64)


def test_coarsen_is_an_area_weighted_cell_mean():
    src_lat = np.arange(90, -90.01, -1.0)
    src_lon = np.arange(0, 360, 1.0)
    f = np.cos(np.deg2rad(src_lon))[None, :] * np.ones((len(src_lat), 1)) + 3.0
    c = coarsen(f, src_lat, src_lon, LAT, LON)
    assert c.shape == (32, 64)
    assert np.allclose(c[:, 0], c[0, 0]) and abs(c.mean() - 3.0) < 1e-2
    assert np.allclose(coarsen(np.full((len(src_lat), len(src_lon)), 7.0), src_lat, src_lon, LAT, LON), 7.0)


def test_scores_and_horizons():
    w = np.cos(np.deg2rad(LAT))
    w = w / w.mean()
    rng = np.random.default_rng(0)
    o = rng.normal(size=(32, 64))
    clim = np.zeros((32, 64))
    assert rmse(o, o, w) == 0 and abs(acc(o, o, clim, w) - 1) < 1e-12
    assert abs(acc(-o, o, clim, w) + 1) < 1e-12
    leads = np.arange(0, 241, 6.0)
    a = np.exp(-leads / 240.0)                                   # ACC = 0.6 at 240 ln(1/0.6) = 122.6 h
    assert abs(first_crossing(leads, a, 0.6) - 240 * np.log(1 / 0.6)) < 1.0
    assert first_crossing(leads, np.ones_like(leads), 0.6) == np.inf
    h = horizons(leads, a, rmse_curve=leads, clim_rmse=np.full_like(leads, 120.0))
    assert abs(h["no_skill_hours"] - 120.0) < 1e-9 and h["useful_hours"] > h["no_skill_hours"]


def test_events_are_well_formed():
    assert len({e.key for e in EVENTS}) == len(EVENTS)
    for e in EVENTS:
        assert np.datetime64(e.peak) > np.datetime64(e.init)
        assert region_mask(LAT, LON, e.region).sum() >= 2, e.key


def test_forecast_gif_and_skill_figure(tmp_path):
    from pinneapple_physics.weather.viz import forecast_gif, orthographic, skill_figure

    img = orthographic(np.ones((32, 64)), LAT, LON, size=64)
    assert np.isnan(img[0, 0]) and img[32, 32] == 1
    truth = 280 + np.random.default_rng(1).normal(size=(3, 32, 64))
    times = np.array(["2021-06-28T00", "2021-06-28T06", "2021-06-28T12"], dtype="datetime64[ns]")
    leads = np.arange(0, 241, 6.0)
    acc_curve = np.exp(-leads / 300)
    hz = horizons(leads, acc_curve, leads, np.full_like(leads, 200.0))
    g = forecast_gif(tmp_path / "e.gif", "t2m", truth, truth + 0.5, LAT, LON, times, skill_leads_h=leads,
                     skill_acc=acc_curve, horizon=hz, region=(40, 62, 225, 255), title="test", dpi=40)
    assert g.stat().st_size > 1000
    s = skill_figure(tmp_path / "s.png", leads, {"PINNeAPPle": acc_curve}, {"PINNeAPPle": hz["useful_hours"]}, "z500")
    assert s.stat().st_size > 1000


def test_earth2_style_globes(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from pinneapple_physics.weather.viz import _globe_rgb, coastlines, earth2_gif, upsample_sphere

    f = np.cos(np.deg2rad(LAT))[:, None] * np.ones((32, 64))
    up, la, lo = upsample_sphere(f, LAT, LON, factor=2)
    assert up.shape == (64, 128) and abs(up.max() - f.max()) < 0.05
    img = _globe_rgb(up, la, lo, (0.0, 0.0), 64, plt.get_cmap("turbo"), Normalize(0, 1), coastlines())
    assert img.shape == (64, 64, 4) and img[32, 32, 3] == 1 and img[0, 0, 3] == 0
    truth = 280 + np.random.default_rng(2).normal(size=(2, 32, 64))
    times = np.array(["2021-06-28T00", "2021-06-28T06"], dtype="datetime64[ns]")
    g = earth2_gif(tmp_path / "g.gif", "t2m", truth, truth + 0.3, LAT, LON, times, lead_hours=np.array([0.0, 6.0]),
                   horizon={"useful_hours": 90.0, "no_skill_hours": 100.0}, title="test", substeps=2, size=48, dpi=30)
    assert g.stat().st_size > 1000

