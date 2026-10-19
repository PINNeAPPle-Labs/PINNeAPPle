"""MIL-F-8785C / MIL-HDBK-1797 turbulence: intensities, spectra, and the generators against their target PSDs."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from pinneapple_simulation.flight.turbulence import (
    FT,
    KT,
    DiscreteGust,
    DrydenFilter,
    FrozenField3D,
    FrozenLine,
    ShearWind,
    isotropic_line_spectra,
    pratt_gust_load,
    psd,
    psd_temporal,
    turbulence_spec,
    welch_psd,
)


def test_low_altitude_intensities_and_lengths():
    s = turbulence_spec(100 * FT, "moderate")
    assert s.sigma_w == pytest.approx(0.1 * 30 * KT)
    f = 0.177 + 0.000823 * 100
    assert s.sigma_u == pytest.approx(s.sigma_w / f ** 0.4)
    assert s.L_w == pytest.approx(100 * FT)
    assert s.L_u == pytest.approx(100 / f ** 1.2 * FT)


def test_high_altitude_table_and_lengths():
    s = turbulence_spec(15000 * FT, "moderate", "dryden")
    assert s.sigma_w == pytest.approx(8.0 * FT) and s.L_w == pytest.approx(1750 * FT)
    v = turbulence_spec(15000 * FT, "severe")
    assert v.sigma_u == v.sigma_v == v.sigma_w == pytest.approx(22.1 * FT) and v.L_u == pytest.approx(2500 * FT)
    mid = turbulence_spec(1500 * FT, "light")                          # interpolated between 1000 and 2000 ft
    lo, hi = turbulence_spec(1000 * FT, "light"), turbulence_spec(2000 * FT, "light")
    assert mid.sigma_w == pytest.approx(0.5 * (lo.sigma_w + hi.sigma_w))


@pytest.mark.parametrize("model", ["dryden", "von_karman"])
@pytest.mark.parametrize("c", "uvw")
def test_spectra_integrate_to_the_variance(model, c):
    s = turbulence_spec(3000.0, "severe", model)
    var = quad(lambda W, c=c: psd(s, W, c), 0, np.inf, limit=500)[0]
    assert var == pytest.approx(s.sigma["uvw".index(c)] ** 2, rel=1e-4)


def _band_ratio(x, d, target, lo, hi, nseg=32):
    W, P = welch_psd(x, d, nseg)
    sel = (W > lo) & (W < hi)
    edges = np.geomspace(lo, hi, 9)
    r = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        k = (W >= a) & (W < b)
        if k.sum() >= 2:
            r.append(P[k].mean() / target(W[k]).mean())
    assert sel.sum() > 10
    return np.array(r)


@pytest.mark.parametrize("model", ["dryden", "von_karman"])
def test_frozen_line_matches_target_psd_and_variance(model):
    s = turbulence_spec(3000.0, "severe", model)
    line = FrozenLine(s, length=400_000.0, dx=2.0, seed=3)
    for i, c in enumerate("uvw"):
        x = line.series[i]
        assert x.std() == pytest.approx(s.sigma[i], rel=0.02)            # exact amplitudes: band-limited variance
        r = _band_ratio(x, 2.0, lambda W, c=c: psd(s, W, c), 4 / s.lengths[i], 0.5)
        assert np.all(np.abs(r - 1) < 0.25), r


def test_dryden_filter_variance_and_psd():
    s = turbulence_spec(3000.0, "moderate", "dryden")
    V, dt = 100.0, 0.02
    y = DrydenFilter(s, V, dt, seed=1).series(200_000)
    for i, c in enumerate("uvw"):
        assert y[:, i].std() == pytest.approx(s.sigma[i], rel=0.08)
        r = _band_ratio(y[:, i], dt, lambda w, c=c: psd_temporal(s, w, V, c), 4 * V / s.lengths[i], 10.0)
        assert np.all(np.abs(r - 1) < 0.3), r


def test_isotropic_von_karman_projects_onto_the_mil_spectra():
    L = 762.0
    s = turbulence_spec(3000.0, "severe")
    k = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    F11, F22 = isotropic_line_spectra(k, L, 1.0)
    unit = s.sigma_w ** 2
    np.testing.assert_allclose(F11, psd(s, k, "u") / unit, rtol=1e-3)
    np.testing.assert_allclose(F22, psd(s, k, "w") / unit, rtol=1e-3)


def test_frozen_field_3d_divergence_free_variance_and_spectrum():
    s = turbulence_spec(3000.0, "severe")
    f = FrozenField3D(s, dx_min=2.0, seed=2, n_shells=96, n_dirs=32)
    rng = np.random.default_rng(0)
    P = rng.uniform(-5e4, 5e4, (4000, 3))
    U = f.components(P)
    assert 0.8 < f.variance_fraction <= 1.0
    np.testing.assert_allclose(U.std(0), s.sigma * math.sqrt(f.variance_fraction), rtol=0.15)
    assert np.abs(np.einsum("ij,ij->i", f.k, f.s)).max() < 1e-12 * np.abs(f.k).max()   # k . sigma = 0
    h = 1e-3
    div = sum((f.components(P[:200] + h * e)[:, i] - f.components(P[:200] - h * e)[:, i]) / (2 * h)
              for i, e in enumerate(np.eye(3)))
    grad = np.abs((f.components(P[:200] + h * np.eye(3)[1])[:, 2] - f.components(P[:200] - h * np.eye(3)[1])[:, 2])
                  / (2 * h))
    assert np.abs(div).max() < 1e-4 * grad.max()
    # along a long line the w spectrum (transverse) follows the MIL von Karman w form in the resolved band
    x = np.arange(0, 200_000, 4.0)
    w = f.components(np.c_[x, np.zeros_like(x), np.zeros_like(x)])[:, 2]
    r = _band_ratio(w, 4.0, lambda W: psd(s, W, "w"), 5 / s.L_w, 0.2, nseg=16)
    assert np.all(np.abs(r - 1) < 0.5), r


def test_spanwise_gust_difference_grows_with_separation():
    s = turbulence_spec(3000.0, "severe")
    f = FrozenField3D(s, dx_min=1.0, seed=5)
    x = np.arange(0, 100_000, 5.0)
    d = []
    for dy in (2.0, 10.0, 40.0):
        a = f.components(np.c_[x, np.zeros_like(x), np.zeros_like(x)])[:, 2]
        b = f.components(np.c_[x, np.full_like(x, dy), np.zeros_like(x)])[:, 2]
        d.append(np.std(a - b))
    assert d[0] < d[1] < d[2]


def test_discrete_gust_and_wind_profile():
    g = DiscreteGust(amplitude=10.0, length=50.0, start=100.0, shape="full")
    P = np.c_[[100.0, 150.0, 200.0, 260.0], np.zeros(4), np.zeros(4)]
    np.testing.assert_allclose(g.velocity(0, P)[:, 2], [0.0, -10.0, 0.0, 0.0], atol=1e-12)
    r = DiscreteGust(amplitude=10.0, length=50.0, shape="ramp").profile(np.array([25.0, 50.0, 500.0]))
    np.testing.assert_allclose(r, [5.0, 10.0, 10.0])
    w = ShearWind(w20=10.0, from_deg=90.0)                              # from the east: blows towards the west
    v = w.velocity(0, np.array([[0, 0, -20 * FT]]))[0]
    np.testing.assert_allclose(v, [0.0, -10.0, 0.0], atol=1e-9)


def test_pratt_formula():
    p = pratt_gust_load(wing_loading=1000.0, cl_alpha=5.0, mac=1.5, U_de=15.24, V_e=60.0)
    mu = 2 * 1000.0 / (1.225 * 1.5 * 5.0 * 9.80665)
    assert p["mu"] == pytest.approx(mu) and p["K_g"] == pytest.approx(0.88 * mu / (5.3 + mu))
    assert p["dn"] == pytest.approx(p["K_g"] * 1.225 * 15.24 * 60.0 * 5.0 / 2000.0)
