"""Earth-system blocks: spherical shallow water (Williamson cases), Richards equation, two-layer climate EBM."""
import numpy as np
import pytest

from pinneapple_simulation.geophysics import (
    CELIA_1990_SOIL,
    Boundary,
    BrooksCorey,
    ClappHornberger,
    Gardner,
    SphericalHarmonics,
    TwoLayerEBM,
    fit_two_layer,
    gardner_steady_infiltration,
    height_errors,
    rossby_haurwitz_speed,
    solve_richards,
    williamson_case2,
    williamson_case5,
    williamson_case6,
)
from pinneapple_simulation.geophysics.sphere_shallow_water import legendre_tables

# ------------------------------------------------------------------------------------------- spherical transforms


@pytest.fixture(scope="module")
def sh21():
    return SphericalHarmonics(21)


def test_legendre_functions_are_orthonormal_and_transforms_round_trip(sh21):
    G = np.einsum("j,jmn,jmk->mnk", sh21.w, sh21.P, sh21.P)
    for m in range(sh21.M + 1):
        k = sh21.mask[m]
        np.testing.assert_allclose(G[m][np.ix_(k, k)], np.eye(k.sum()), atol=1e-12)
    rng = np.random.default_rng(0)
    s = (rng.normal(size=sh21.n.shape) + 1j * rng.normal(size=sh21.n.shape)) * sh21.mask
    s[0] = s[0].real                                     # m = 0 coefficients of a real field are real
    np.testing.assert_allclose(sh21.analysis(sh21.synthesis(s)), s, atol=1e-12)


def test_legendre_derivative_table_matches_finite_differences():
    mu = np.linspace(-0.95, 0.95, 41)
    P, H = legendre_tables(mu, 21, 21)
    Pp, _ = legendre_tables(mu + 1e-6, 21, 21)
    Pm, _ = legendre_tables(mu - 1e-6, 21, 21)
    dP = (Pp - Pm)[:, :, :22] / 2e-6
    np.testing.assert_allclose(H, (1 - mu ** 2)[:, None, None] * dP, atol=1e-6 * np.abs(H).max())


def test_laplacian_and_velocity_inversion_of_solid_body_rotation(sh21):
    u0 = 30.0
    lon, lat = sh21.grid()
    div, curl = sh21.div_curl(u0 * np.cos(lat) ** 2, 0 * lat)
    np.testing.assert_allclose(sh21.synthesis(curl), 2 * u0 / sh21.a * np.sin(lat), atol=1e-12)   # zeta = 2u0 sin/a
    assert np.abs(div).max() < 1e-15
    U, V = sh21.uv_cos(curl, div)
    np.testing.assert_allclose(U, u0 * np.cos(lat) ** 2, atol=1e-9)
    assert np.abs(V).max() < 1e-9


# ------------------------------------------------------------------------------------------- Williamson cases


@pytest.mark.parametrize("alpha", [0.0, np.pi / 4, np.pi / 2 - 0.05])
def test_case2_steady_geostrophic_flow_stays_steady(sh21, alpha):
    m = williamson_case2(sh21, alpha=alpha)
    h0, d0 = m.fields()["h"], m.diagnostics()
    m.run(2 * 86400, 1800.0)
    err = height_errors(sh21, m.fields()["h"], h0)
    assert err["l2"] < 1e-10 and err["linf"] < 1e-9
    d1 = m.diagnostics()
    assert abs(d1["mass"] / d0["mass"] - 1) < 1e-13 and abs(d1["energy"] / d0["energy"] - 1) < 1e-10


def test_case6_rossby_haurwitz_wave_moves_east_and_conserves(sh21):
    m = williamson_case6(sh21)
    f0, d0 = m.fields(), m.diagnostics()
    m.run(86400.0, 900.0)
    f1, d1 = m.fields(), m.diagnostics()
    j = np.argmin(np.abs(sh21.lat - np.deg2rad(45)))
    a, b = f0["h"][j] - f0["h"][j].mean(), f1["h"][j] - f1["h"][j].mean()
    shift = (np.angle(np.fft.rfft(a)[4]) - np.angle(np.fft.rfft(b)[4])) / 4       # eastward displacement [rad]
    ratio = shift / (rossby_haurwitz_speed() * 86400)
    assert 0.8 < ratio < 1.0                             # divergence slows the wave relative to the barotropic speed
    assert abs(d1["mass"] / d0["mass"] - 1) < 1e-13
    assert abs(d1["energy"] / d0["energy"] - 1) < 1e-6
    assert abs(d1["potential_enstrophy"] / d0["potential_enstrophy"] - 1) < 1e-4


def test_case5_flow_over_mountain_is_stable_and_mass_conserving(sh21):
    m = williamson_case5(sh21)
    d0 = m.diagnostics()
    m.run(2 * 86400, 900.0)
    f, d1 = m.fields(), m.diagnostics()
    assert np.all(np.isfinite(f["h"])) and f["h"].min() > 0
    assert abs(d1["mass"] / d0["mass"] - 1) < 1e-13 and abs(d1["energy"] / d0["energy"] - 1) < 1e-6
    assert 4500 < (f["h"] + f["hs"]).min() and (f["h"] + f["hs"]).max() < 6000


# ------------------------------------------------------------------------------------------- Richards equation


@pytest.mark.parametrize("soil", [CELIA_1990_SOIL, BrooksCorey(0.05, 0.40, -20.0, 0.5, 1e-3),
                                  ClappHornberger(0.45, -30.0, 5.0, 1e-3), Gardner(0.05, 0.40, 0.05, 1e-3)])
def test_moisture_capacity_is_the_derivative_of_the_retention_curve(soil):
    h = np.linspace(-500, -40, 50)
    fd = (soil.theta(h + 1e-4) - soil.theta(h - 1e-4)) / 2e-4
    np.testing.assert_allclose(soil.C(h), fd, rtol=1e-6, atol=1e-12)
    assert np.all(np.diff(soil.K(h)) > 0)                # conductivity grows towards saturation


def test_richards_matches_gardner_steady_infiltration_with_second_order_convergence():
    soil, L, q0 = Gardner(0.05, 0.40, 0.02, 1e-3), 200.0, 4e-4
    errs = []
    for n in (50, 100, 200):
        r = solve_richards(soil, L, n, -50.0, 5e6, Boundary("flux", -q0), Boundary("head", 0.0), dt0=10.0,
                           t_out=np.array([0.0, 5e6]))
        errs.append(np.max(np.abs(r.h[-1] - gardner_steady_infiltration(soil, r.z + L, q0))))
        assert abs(r.flux_bottom[-1] + q0) < 1e-9        # steady: what enters at the top leaves at the bottom
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 1e-3 and np.all(order > 1.8)


def test_richards_mixed_form_conserves_mass_in_celia_infiltration():
    for bottom in (Boundary("head", -1000.0), Boundary("free_drainage")):
        r = solve_richards(CELIA_1990_SOIL, 100.0, 100, -1000.0, 86400.0, Boundary("head", -75.0), bottom,
                           dt0=1.0, dt_max=600.0)
        assert abs(r.mass_balance_ratio - 1) < 1e-6
        front = -r.z[r.h[-1] > -500].min()               # wetting front depth after one day [cm]
        assert 40 < front < 70
        assert np.all(np.diff(r.h[-1]) >= -1e-6)         # head increases monotonically towards the wet surface


# ------------------------------------------------------------------------------------------- two-layer EBM


def test_ebm_exact_integration_matches_closed_form_and_equilibrium():
    t = np.arange(1, 301)
    for eps in (1.0, 1.3):
        m = TwoLayerEBM(lam=1.13, gamma=0.73, C=7.3, C0=106.0, eps=eps)
        np.testing.assert_allclose(m.simulate(np.full(300, 3.71))["T"], m.step_response(t), atol=1e-12)
        md = m.modes()
        assert abs(md["a_fast"] + md["a_slow"] - 1) < 1e-12 and md["tau_fast"] < 10 < 100 < md["tau_slow"]
    m = TwoLayerEBM(lam=1.13, gamma=0.73, C=7.3, C0=106.0)
    assert abs(m.simulate(np.full(5000, 3.71))["T"][-1] - m.ecs()) < 1e-6
    assert 0 < m.tcr() < m.ecs()
    run = m.simulate(np.full(10, 3.71))
    assert abs(run["N"][0] - (3.71 - m.lam * run["T"][0])) < 1e-12


def test_ebm_fit_recovers_parameters_from_abrupt_forcing_and_tcr_from_a_ramp():
    m = TwoLayerEBM(lam=1.13, gamma=0.73, C=7.3, C0=106.0)
    rng = np.random.default_rng(1)
    f4 = fit_two_layer(m.simulate(np.full(150, 7.42))["T"] + rng.normal(0, 0.05, 150), np.full(150, 7.42))
    assert abs(f4["params"]["lam"] / m.lam - 1) < 0.1 and abs(f4["ecs"] - m.ecs()) < 3 * f4["ecs_stderr"] + 0.1
    F = np.linspace(0, 2.7, 170)
    fr = fit_two_layer(m.simulate(F)["T"] + rng.normal(0, 0.1, 170), F)
    assert abs(fr["tcr"] / m.tcr() - 1) < 0.1
    assert abs(fr["correlation"][0, 3]) > 0.9            # lambda vs deep-ocean capacity: not identifiable from a ramp
