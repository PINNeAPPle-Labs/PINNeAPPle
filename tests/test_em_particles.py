"""Charged particles in E and B fields (issue #1106): Boris pusher and closed-form references."""
import math

import numpy as np

from pinneapple_simulation.electromagnetics.constants import MU0
from pinneapple_simulation.electromagnetics.particles import (
    C0,
    M_E,
    M_P,
    Q_E,
    cyclotron_frequency,
    cyclotron_study,
    drift_study,
    exb_drift,
    gradb_drift,
    larmor_radius,
    loss_cone_angle,
    magnetic_bottle,
    magnetic_bottle_study,
    magnetic_dipole,
    mass_spectrometer_study,
    penning_frequencies,
    penning_study,
    push,
    uniform,
)


def test_closed_forms_literal_calls():
    """Plain-number calls (also the Studio's smoke inputs)."""
    assert math.isclose(cyclotron_frequency(1.602176634e-19, 1.67262192595e-27, 1.0), 9.578833e7, rel_tol=1e-6)
    assert math.isclose(larmor_radius(1.602176634e-19, 9.1093837139e-31, 1.0, 1.0e6), 5.6856e-6, rel_tol=1e-4)
    assert np.allclose(exb_drift([0.0, 100.0, 0.0], [0.0, 0.0, 0.001]), [1.0e5, 0.0, 0.0])
    assert np.allclose(gradb_drift(1.602176634e-19, 1.67262192595e-27, 4.0e4, [0.0, 0.0, 0.001], [5.0e-5, 0.0, 0.0]),
                       [0.0, 1.67262192595e-27 * 1.6e9 / (2 * 1.602176634e-19 * 1e-3) * 5e-2, 0.0], rtol=1e-9)
    assert math.isclose(math.degrees(loss_cone_angle(0.01, 0.05)), 26.565051, rel_tol=1e-6)
    f = penning_frequencies(1.602176634e-19, 1.67262192595e-27, 1.0, 10.0, 0.005)
    assert math.isclose(f["modified_cyclotron"] + f["magnetron"], f["cyclotron"], rel_tol=1e-12)
    assert math.isclose(f["modified_cyclotron"] * f["magnetron"], f["axial"] ** 2 / 2, rel_tol=1e-9)


def test_boris_conserves_energy_exactly_in_pure_B_at_any_step():
    v = 0.9 * C0
    for steps in (7, 50):                                   # even a very coarse step keeps |u|
        out = push([[0, 0, 0]], [[v, 0, 0.1 * v]], -Q_E, M_E, B=uniform((0.3, -0.2, 1.0)),
                   dt=2 * math.pi / cyclotron_frequency(Q_E, M_E, 1.0, v) / steps, n_steps=300)
        ke = out["kinetic_energy"][:, 0]
        assert np.max(np.abs(ke - ke[0])) / ke[0] < 1e-12


def test_relativistic_gyration_radius_frequency_and_second_order_phase():
    s = cyclotron_study(kinetic_energy_eV=1.0e6, B=1.0, particle="electron", steps_per_turn=200, turns=5)
    assert s["gamma"] > 2.9
    assert math.isclose(s["radius_m"], s["radius_exact_m"], rel_tol=1e-4)
    assert s["radius_exact_m"] > 2.5 * s["radius_nonrelativistic_m"]       # gamma matters
    e1 = abs(s["frequency_rad_s"] / s["frequency_exact_rad_s"] - 1)
    e2 = abs(cyclotron_study(steps_per_turn=400, turns=5)["frequency_rad_s"] / s["frequency_exact_rad_s"] - 1)
    assert e1 < 2e-4 and 3.0 < e1 / e2 < 5.0                               # phase error ~ dt^2


def test_exb_and_gradb_drifts():
    d = drift_study(B=1.0e-3, E=100.0, grad_scale=20.0, energy_eV=10.0)
    assert math.isclose(d["exb_drift_m_s"], d["exb_drift_exact_m_s"], rel_tol=1e-4)
    assert d["larmor_over_grad_scale"] < 0.05
    assert math.isclose(d["gradb_drift_m_s"], d["gradb_drift_exact_m_s"], rel_tol=0.02)


def test_magnetic_bottle_loss_cone_and_adiabatic_invariant():
    b = magnetic_bottle_study(B0=0.01, L=1.0, z_mirror=2.0, energy_eV=100.0, n_particles=24, bounces=2.0)
    assert math.isclose(b["loss_cone_deg"], 26.565051, rel_tol=1e-6)
    assert b["agreement"] == 1.0                                           # escaped <=> pitch inside the cone
    assert b["mu_relative_variation"] < 0.05


def test_bottle_field_is_divergence_free():
    B = magnetic_bottle(0.01, 1.0)
    x = np.random.default_rng(0).normal(0, 0.3, (50, 3))
    h = 1e-6
    div = sum((B(x + h * e, 0)[:, i] - B(x - h * e, 0)[:, i]) / (2 * h) for i, e in enumerate(np.eye(3)))
    assert np.max(np.abs(div)) < 1e-8


def test_dipole_field_matches_equatorial_closed_form():
    m = 8.0e22
    r = 2 * 6.371e6
    B = magnetic_dipole((0, 0, -m))(np.array([[r, 0, 0]]), 0.0)[0]
    assert math.isclose(B[2], MU0 / (4 * math.pi) * m / r**3, rel_tol=1e-12) and abs(B[0]) < 1e-20


def test_penning_trap_eigenfrequencies():
    p = penning_study(B=1.0, V0=10.0, d=5e-3, particle="proton")
    for k in ("axial", "modified_cyclotron", "magnetron"):
        assert math.isclose(p["measured"][k], p["exact"][k], rel_tol=2e-3), k


def test_mass_spectrometer_landing_points():
    ms = mass_spectrometer_study(B=0.5, energy_eV=2000.0, masses_u=[12.0, 13.0, 14.0])
    for ion in ms["ions"]:
        assert math.isclose(ion["landing_m"], ion["landing_exact_m"], rel_tol=1e-5)
    assert all(s > 0 for s in ms["separation_m"])


def test_stop_freezes_particles_and_reports_time():
    out = push([[0, 0, 0], [0, 0, 0]], [[0, 0, 1e5], [0, 0, -2e5]], Q_E, M_P, dt=1e-7, n_steps=100,
               stop=lambda x, v: x[:, 2] > 0.5)
    assert out["stopped"].tolist() == [True, False]
    assert math.isclose(out["t_stop"][0], 5.0e-6, rel_tol=0.03)


def test_dipole_field_line_is_tangent_to_the_field():
    from pinneapple_simulation.electromagnetics.particles import dipole_field_line
    pts = dipole_field_line(3.0, n=400)
    assert np.allclose(np.linalg.norm(pts[[0, -1]], axis=1), 6.371e6, rtol=1e-9)
    tang = np.gradient(pts, axis=0)[50:350]
    B = magnetic_dipole()(pts[50:350], 0.0)
    cos = np.abs((tang * B).sum(1)) / (np.linalg.norm(tang, axis=1) * np.linalg.norm(B, axis=1))
    assert cos.min() > 0.9999


def test_small_helpers_literal_calls():
    from pinneapple_simulation.electromagnetics.particles import lorentz_factor, pitch_velocity
    assert math.isclose(lorentz_factor(0.6 * 299792458.0), 1.25, rel_tol=1e-12)
    v = pitch_velocity(1000.0, 30.0, [0.0, 0.0, 1.0], 0.0)
    assert math.isclose(np.linalg.norm(v), 1000.0) and math.isclose(v[2], 1000.0 * math.cos(math.radians(30)))


def test_radiation_belt_bounce_and_drift_periods():
    from pinneapple_simulation.electromagnetics.particles import dipole_trap_study
    d = dipole_trap_study(energy_MeV=10.0, L_shell=2.0, pitch_deg=30.0, species="proton", duration_s=8.0,
                          steps_per_gyration=25, save_every=20)
    assert not d["lost"] and d["drift_direction"] == "westward"            # protons drift west
    assert math.isclose(d["bounce_period_s"], d["bounce_period_approx_s"], rel_tol=0.05)
