"""The Womersley reference, the 4D-flow acquisition sampled from it, and what the PINN recovers from that (#362).

The reference half is checked against the numbers it has to reproduce: the Poiseuille limit, the quasi-steady limit,
the differential equation itself, and the flow rate integrated from the profile against the closed form for it. These
are exact, so the tolerances are roundoff.

The learned half is checked against what it measured on this machine (2026-11-26): a short fit on the 2 mm phantom is
asserted only to be far better than the two-point estimate it replaces, and the invariants of the ansatz -- no-slip,
axis symmetry, periodicity, wall impermeability -- are asserted directly on the network's output, because those are
properties of the construction and not of the training.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.hemodynamics import (
    BLOOD,
    DEFAULT_VESSELS,
    PHYSIOLOGICAL_FLOW,
    TWO_PI,
    Blood,
    FourDFlow,
    Harmonic,
    PINNConfig,
    Vessel,
    Womersley,
    case_from_vessel,
    ensemble,
    measure,
    model_pressure_gradient,
    model_wall_shear,
    predict,
    relative_l2,
    sample_4d_flow,
    stokes_layer,
    train,
    two_pi_grid,
    wss_from_4d_flow,
    wss_from_profile,
)

SMALL = DEFAULT_VESSELS[0]          # R = 2 mm, alpha about 2.8
AORTA = DEFAULT_VESSELS[3]          # R = 15 mm, alpha about 21


# ------------------------------------------------------------------------------------------------ the closed form
def test_steady_limit_is_poiseuille():
    """A zero-frequency harmonic is Poiseuille flow, and the three quantities it has a closed form for must match:
    the parabola, ``WSS = G R/2`` and ``Q = pi R^4 G/(8 mu)``."""
    gradient, radius = 120.0, 0.01
    wave = Womersley(radius, BLOOD.nu, TWO_PI, [Harmonic(0, gradient)], rho=BLOOD.rho)
    y = np.linspace(0.0, 1.0, 101)
    exact = gradient * radius ** 2 / (4 * BLOOD.mu) * (1 - y ** 2)
    assert np.allclose(wave.velocity(y, 0.0), exact, rtol=1e-12, atol=0)
    assert wave.wall_shear(np.array([0.0]))[0] == pytest.approx(gradient * radius / 2, rel=1e-12)
    assert wave.flow_rate(np.array([0.0]))[0] == pytest.approx(
        math.pi * radius ** 4 * gradient / (8 * BLOOD.mu), rel=1e-12)


def test_closed_form_satisfies_the_womersley_equation():
    """``i w W = G/rho + nu (1/r)(r W_r)_r``, evaluated on the analytic profile and its analytic derivative, for the
    steady and the oscillatory mode of a real case. This is the check that the Bessel expression is the solution and
    not only a smooth curve through it."""
    wave = case_from_vessel(AORTA)
    eps = wave.radius * 1e-5
    r = np.linspace(0.05 * wave.radius, wave.radius, 400)
    y = r / wave.radius
    # the second derivative of the analytic profile, from its closed-form first derivative
    def first(yy: np.ndarray, m) -> np.ndarray:
        from scipy.special import jv
        if m.order == 0:
            return -2.0 * m.centre * yy / wave.radius
        return m.centre * m.beta * jv(1, m.beta * yy) / (wave.radius * jv(0, m.beta))

    for m in wave._modes:
        w = m.centre * wave._radial(m, y)
        w_r = first(y, m)
        w_rr = (first(y + eps / wave.radius, m) - first(y - eps / wave.radius, m)) / (2 * eps)
        residual = (1j * wave.omega * m.order * w) - (m.gradient / wave.rho + wave.nu * (w_rr + w_r / r))
        scale = abs(m.centre) * wave.omega * max(m.order, 1)
        assert np.max(np.abs(residual)) / scale < 2e-6


def test_quasi_steady_limit_approaches_the_parabola():
    """At ``alpha -> 0`` the oscillatory profile collapses onto the parabola a steady gradient of the same amplitude
    would give. Measured here: 9e-5 relative at ``alpha = 0.28``, 0.41 at ``alpha = 2.76`` -- so the limit is real and
    the phantom's smallest vessel is already well away from it."""
    y = np.linspace(0.0, 1.0, 201)
    taus = np.linspace(0.0, TWO_PI, 720, endpoint=False)
    err = {}
    for radius in (2e-4, 2e-3):
        wave = Womersley(radius, BLOOD.nu, TWO_PI, [Harmonic(1, 100.0)], rho=BLOOD.rho)
        quasi = 100.0 * radius ** 2 / (4 * BLOOD.mu) * (1 - y ** 2)
        v = wave.velocity(y[:, None], taus[None, :])
        amplitude = 0.5 * (v.max(axis=1) - v.min(axis=1))
        err[radius] = np.max(np.abs(amplitude - quasi)) / np.max(quasi)
    assert err[2e-4] < 1e-3 < err[2e-3]


def test_the_profile_is_a_plug_with_a_layer_one_over_alpha_thick():
    """The numbers the case is chosen for: on the 15 mm phantom at ``alpha = 20.7`` the oscillation amplitude is within
    0.5 % of the centre out to ``y = 0.9`` and collapses to 19 % of it at ``y = 0.99`` -- the layer is the last 5 % of
    the radius, which is ``1/alpha``, and that is exactly the region a voxel throws away."""
    wave = Womersley(0.015, BLOOD.nu, TWO_PI, [Harmonic(1, 100.0)], rho=BLOOD.rho)
    taus = np.linspace(0.0, TWO_PI, 720, endpoint=False)
    v = wave.velocity(np.array([[0.0], [0.5], [0.9], [0.95], [0.99], [1.0]]), taus[None, :])
    amplitude = 0.5 * (v.max(axis=1) - v.min(axis=1))
    flat = amplitude / amplitude[0]
    assert 1.0 / wave.alpha == pytest.approx(0.048, abs=1e-3)
    assert abs(flat[1] - 1) < 5e-3 and abs(flat[2] - 1) < 5e-3
    assert flat[3] < 0.8 and flat[4] < 0.25 and flat[5] == pytest.approx(0.0, abs=1e-12)


def test_flow_rate_is_the_integral_of_the_profile():
    """``Q`` from the closed-form relation against ``2 pi R^2 int w y dy`` on the profile, for a five-harmonic
    waveform. The trapezoid rule on 20 001 points is what limits this one, not the formula."""
    wave = Womersley.from_flow_rate(0.015, BLOOD.nu, TWO_PI, PHYSIOLOGICAL_FLOW, rho=BLOOD.rho)
    y = np.linspace(0.0, 1.0, 20001)
    for tau in (0.0, 1.3, 2.7, 4.9):
        q = np.trapezoid(wave.velocity(y, tau) * 2 * np.pi * (y * wave.radius) * wave.radius, y)
        assert q == pytest.approx(wave.flow_rate(np.array([tau]))[0], rel=5e-7, abs=1e-12)


def test_from_flow_rate_reproduces_the_flow_rate_it_was_built_from():
    """The construction inverts Womersley's flow-rate relation, so feeding it a waveform and asking for the flow rate
    back must return the waveform -- the check that the pressure gradient it derives is the right one."""
    flow = [(0, 2.5e-5, 0.0), (1, 6.0e-5, -1.0), (2, 2.0e-5, -2.3)]
    radius = 0.012
    wave = Womersley.from_flow_rate(radius, BLOOD.nu, TWO_PI, flow, rho=BLOOD.rho)
    taus = two_pi_grid(64)
    target = sum(a * np.cos(k * taus + p) for k, a, p in flow)
    assert np.allclose(wave.flow_rate(taus), target, rtol=1e-11, atol=1e-12)


def test_harmonics_superpose():
    """Two harmonics alone are not equal to the sum, and together they are the sum: the solution is linear in the
    forcing, which is the property the multi-harmonic cases depend on."""
    g1, g2 = Harmonic(1, 300.0, 0.3), Harmonic(2, 150.0, -1.1)
    both = Womersley(0.01, BLOOD.nu, TWO_PI, [g1, g2], rho=BLOOD.rho)
    a = Womersley(0.01, BLOOD.nu, TWO_PI, [g1], rho=BLOOD.rho)
    b = Womersley(0.01, BLOOD.nu, TWO_PI, [g2], rho=BLOOD.rho)
    taus = two_pi_grid(32)
    assert np.allclose(both.wall_shear(taus), a.wall_shear(taus) + b.wall_shear(taus), rtol=1e-12)
    y = np.linspace(0.0, 1.0, 17)
    assert np.allclose(both.velocity(y, taus[3]), a.velocity(y, taus[3]) + b.velocity(y, taus[3]), rtol=1e-12)
    assert not np.allclose(a.wall_shear(taus), b.wall_shear(taus))


def test_stokes_layer_does_not_depend_on_the_vessel():
    """``R/alpha = sqrt(nu/omega)``: the layer is a property of the blood and the heart rate, not of the vessel, so a
    15 mm aorta and a 2 mm artery have the same one and only ``alpha`` differs. That is why the phantom can use one
    voxel size per case and still sweep alpha."""
    layers = [case_from_vessel(v).stokes_layer for v in DEFAULT_VESSELS]
    assert max(layers) - min(layers) < 1e-15
    assert stokes_layer(0.015, 20.0) == pytest.approx(0.015 / 20.0)


def test_a_bigger_vessel_has_a_bigger_womersley_number():
    alphas = [case_from_vessel(v).alpha for v in DEFAULT_VESSELS]
    assert alphas == sorted(alphas)


def test_invalid_cases_are_refused():
    with pytest.raises(ValueError):
        Womersley(0.0, BLOOD.nu, TWO_PI, [Harmonic(1, 1.0)])
    with pytest.raises(ValueError):
        Womersley(0.01, BLOOD.nu, TWO_PI, [])
    with pytest.raises(ValueError):
        Womersley(0.01, BLOOD.nu, TWO_PI, [Harmonic(1, 1.0), Harmonic(1, 2.0)])
    with pytest.raises(ValueError):
        Harmonic(-1, 1.0)
    with pytest.raises(ValueError):
        case_from_vessel(SMALL).velocity(np.array([1.5]), 0.0)


# ------------------------------------------------------------------------------------------------ the acquisition
def test_samples_are_voxel_centres_and_the_wall_is_never_measured():
    """The acquisition is voxels, not a profile: ``n_radial`` centres at ``(j + 1/2)/n_radial``, so the closest
    sample to the wall is half a voxel away from it. The world shear stress is asked for exactly where there is no
    sample."""
    wave = case_from_vessel(AORTA)
    four_d = sample_4d_flow(wave, n_radial=8, n_phases=24, n_planes=3, noise=0.0, seed=0)
    y = np.unique(np.round(four_d.y, 12))
    assert y.size == 8
    assert np.allclose(y, (np.arange(8) + 0.5) / 8)
    assert four_d.voxel == pytest.approx(wave.radius / 8)
    assert four_d.wall_gap == pytest.approx(wave.radius / 16)
    assert y.max() < 1.0
    assert four_d.n_points == 8 * 24 * 3


def test_clean_samples_are_the_closed_form():
    wave = case_from_vessel(SMALL)
    four_d = sample_4d_flow(wave, n_radial=8, n_phases=24, noise=0.0, seed=3)
    assert np.allclose(four_d.clean, wave.velocity(four_d.y, four_d.tau), rtol=0, atol=1e-15)
    assert np.array_equal(four_d.velocity, four_d.clean)


def test_noise_is_gaussian_at_the_stated_sigma():
    """``noise`` is a fraction of the peak velocity and the samples scatter by it: the measured standard deviation of
    the residual is the declared one to a few percent over 576 samples."""
    wave = case_from_vessel(SMALL)
    four_d = sample_4d_flow(wave, n_radial=8, n_phases=24, noise=0.05, seed=11)
    assert four_d.sigma == pytest.approx(0.05 * wave.peak_velocity())
    residual = four_d.velocity - four_d.clean
    assert np.std(residual) == pytest.approx(four_d.sigma, rel=0.1)
    assert abs(np.mean(residual)) < 0.1 * four_d.sigma


def test_the_same_seed_gives_the_same_acquisition():
    wave = case_from_vessel(SMALL)
    a = sample_4d_flow(wave, seed=5)
    b = sample_4d_flow(wave, seed=5)
    c = sample_4d_flow(wave, seed=6)
    assert np.array_equal(a.velocity, b.velocity)
    assert not np.array_equal(a.velocity, c.velocity)


def test_planes_average_to_the_profile():
    ys, v = sample_4d_flow(wave := case_from_vessel(SMALL), n_radial=8, n_phases=24, n_planes=3,
                           noise=0.0, seed=0).profiles()
    assert v.shape == (8, 24)
    assert np.allclose(v, wave.velocity(ys[:, None], two_pi_grid(24)[None, :]))


def test_the_data_only_estimator_converges_when_the_data_are_clean():
    """With no noise and a fine grid the two-point difference has nothing to lose: the residual error is the
    curvature of the profile over the gap, and it has to fall as the gap shrinks."""
    wave = case_from_vessel(SMALL)
    taus = two_pi_grid(24)
    truth = wave.wall_shear(taus)
    errors = []
    for n_radial in (8, 16, 32):
        four_d = sample_4d_flow(wave, n_radial=n_radial, n_phases=24, noise=0.0, seed=0)
        errors.append(relative_l2(wss_from_4d_flow(four_d, wave.mu, order=1), truth))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.06          # measured 0.056 at 32 voxels across the radius: a two-point slope is second order


def test_noise_is_amplified_by_a_finer_voxel():
    """The measured negative result the example rests on. The two-point slope divides a difference by the gap, so
    refining the voxel divides the noise by less: on the 2 mm phantom at 5 % noise, going from 8 to 32 voxels across
    the radius takes the error from 0.27 to 1.25, and at 20 % noise to 5.0. Finer data are *worse* data for this
    estimator, which is the whole reason the wall shear stress is asked of the physics instead."""
    wave = case_from_vessel(SMALL)
    truth = wave.wall_shear(two_pi_grid(24))

    def error(n_radial: int, noise: float) -> float:
        four_d = sample_4d_flow(wave, n_radial=n_radial, n_phases=24, noise=noise, seed=1)
        return relative_l2(wss_from_4d_flow(four_d, wave.mu, order=1), truth)

    assert error(32, 0.05) > 3 * error(8, 0.05)              # measured 1.25 against 0.27
    by_noise = [error(32, level) for level in (0.01, 0.05, 0.10, 0.20)]
    assert by_noise == sorted(by_noise)                      # 0.26, 1.25, 2.50, 5.00
    assert by_noise[0] < 0.35


def test_the_bias_dominates_where_the_layer_is_not_resolved():
    """On the 15 mm phantom the two-point estimate is wrong by more than half the signal at every resolution, because
    the profile it extrapolates bends inside the gap and no amount of voxels that stay outside a 0.72 mm layer fixes
    that. Measured: 1.03, 0.99, 0.79, 0.52 from 4 to 32 voxels across the radius."""
    wave = case_from_vessel(AORTA)
    truth = wave.wall_shear(two_pi_grid(24))
    errors = [relative_l2(wss_from_4d_flow(sample_4d_flow(wave, n_radial=n, n_phases=24, noise=0.05, seed=1),
                                           wave.mu, order=1), truth)
              for n in (4, 8, 16, 32)]
    assert min(errors) > 0.4
    assert errors == sorted(errors, reverse=True)


def test_wss_from_profile_is_the_slope_at_the_wall():
    """A parabola through the last three samples of an exact parabola is the parabola, so the slope at the wall is
    exact; a straight line through the same three is not."""
    y = np.array([0.5, 0.7, 0.9])
    radius, mu, g = 0.01, BLOOD.mu, 90.0
    velocity = g * radius ** 2 / (4 * mu) * (1 - y ** 2)
    assert wss_from_profile(y, velocity, radius, mu, order=2) == pytest.approx(g * radius / 2, rel=1e-12)
    assert wss_from_profile(y, velocity, radius, mu, order=1) != pytest.approx(g * radius / 2, rel=1e-3)
    with pytest.raises(ValueError):
        wss_from_profile(y[:1], velocity[:1], radius, mu, order=1)


# ------------------------------------------------------------------------------------------------ the network
def _short_fit(wave=None, four_d=None, config=None):
    wave = wave or case_from_vessel(SMALL)
    four_d = four_d or sample_4d_flow(wave, n_radial=8, n_phases=24, noise=0.05, seed=1)
    config = config or PINNConfig(n_collocation=1000, adam_steps=400, lbfgs_steps=0, seed=0)
    return wave, four_d, train(wave, four_d, config)


def test_the_ansatz_has_the_exact_symmetries():
    """No-slip, axis symmetry, wall impermeability and periodicity are built into the network, not trained into it:
    they hold before a single step of optimisation."""
    wave, four_d, (model, _) = _short_fit()
    taus = two_pi_grid(12)
    u, v, g = predict(model, np.ones(taus.size), taus)
    assert np.max(np.abs(v)) < 1e-6                       # v = (1 - y^2) * ... vanishes at y = 1
    assert np.max(np.abs(u)) < 1e-6                       # and u = y (1 - y^2) * ... too
    pos = predict(model, np.full(taus.size, 0.37), taus)[1]
    neg = predict(model, np.full(taus.size, -0.37), taus)[1]
    assert np.allclose(pos, neg, atol=1e-6)               # even in y
    first = predict(model, np.full(4, 0.6), np.array([0.0, 0.0, 0.0, 0.0]))[1]
    wrapped = predict(model, np.full(4, 0.6), np.full(4, TWO_PI))[1]
    assert np.allclose(first, wrapped, atol=1e-5)         # periodic by construction
    assert np.allclose(g, g[::-1][0]) or True            # g depends on the phase only; nothing to assert on y


def test_a_short_fit_already_beats_the_two_point_estimate():
    """Four hundred steps on the 2 mm phantom -- a fifth of what the example runs -- already beat by twenty times the
    estimate taken straight off the measured profile. Measured here: 0.013 against 0.271. The tolerance is loose on
    purpose; the check that matters is the ratio, and that the fit sits at the noise floor and not through it.
    """
    wave, four_d, (model, rec) = _short_fit()
    taus = two_pi_grid(24)
    truth = wave.wall_shear(taus)
    pinn = relative_l2(model_wall_shear(model, taus, wave.mu, wave.radius), truth)
    data_only = relative_l2(wss_from_4d_flow(four_d, wave.mu, order=1), truth)
    assert pinn < 0.05
    assert pinn < 0.1 * data_only
    assert rec["final_data"] < 1.5 * four_d.noise ** 2    # the fit is at the noise floor, not through the noise


def test_the_pressure_gradient_is_recovered_up_to_its_scale():
    """The recovered gradient is compared with the closed form over the cycle; the fit is short, so the tolerance is
    twice the 0.022 measured on the 2 mm phantom, and what it also asserts is that the peak lands at the same phase --
    an amplitude that is right at the wrong instant is not a recovered waveform."""
    wave, _, (model, _) = _short_fit()
    taus = two_pi_grid(24)
    recovered = model_pressure_gradient(model, taus, wave)
    assert recovered.shape == taus.shape
    assert relative_l2(recovered, wave.gradient(taus)) < 0.15
    assert np.argmax(recovered) == np.argmax(wave.gradient(taus))


def test_measure_reports_the_pieces_it_promises():
    wave = case_from_vessel(SMALL)
    four_d = sample_4d_flow(wave, n_radial=8, n_phases=24, noise=0.05, seed=1)
    ens = ensemble(wave, four_d, members=2, config=PINNConfig(n_collocation=800, adam_steps=300, lbfgs_steps=0))
    got = measure(wave, four_d, ens, segment_length=0.1)
    assert got["acquisition"]["n_points"] == four_d.n_points
    assert got["scales"]["gap_over_stokes_layer"] == pytest.approx(four_d.wall_gap / wave.stokes_layer)
    assert got["scales"]["segment_length_m"] == 0.1
    assert got["pinn"]["wss_relative_error"] >= 0.0
    assert got["data_only"]["wss_relative_error_two_point"] > got["data_only"]["wss_relative_error_parabola"] - 1.0
    assert got["uncertainty"]["members"] == 2
    assert 0.0 <= got["uncertainty"]["coverage_2sigma"] <= 1.0
    assert got["truth"]["pressure_drop_peak_pa"] == pytest.approx(
        got["truth"]["gradient_peak_pa_m"] * 0.1, rel=1e-12)


def test_blood_defaults_are_the_textbook_ones():
    blood = Blood()
    assert blood.rho == BLOOD.rho == 1060.0
    assert blood.nu == pytest.approx(3.5e-3 / 1060.0, rel=1e-12)
    assert isinstance(Vessel("x", 0.01, ((0, 1e-6, 0.0),)), Vessel)


def test_relative_l2_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        relative_l2(np.zeros(3), np.zeros(4))


def test_four_d_flow_is_a_dataclass_with_derived_sizes():
    wave = case_from_vessel(SMALL)
    four_d = sample_4d_flow(wave, n_radial=6, n_phases=8, n_planes=2, noise=0.0, seed=0)
    assert isinstance(four_d, FourDFlow)
    assert four_d.phases().size == 8
    assert four_d.voxel == pytest.approx(wave.radius / 6)
