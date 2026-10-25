"""pinneapple_physics.astro.conjunction and presets.astrophysics.cw_exact_velocity, against independent references:
finite differences of the exact CW position, the Rayleigh closed form and Monte Carlo for the collision
probability, and the conserved energy of the J2 potential."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_physics.astro.conjunction import (  # noqa: E402
    collision_probability_2d,
    collision_probability_centered,
    cw_closest_approach,
    cw_mean_motion,
    propagate_two_body_j2,
    screen_cw_conjunction,
)
from pinneapple_physics.pde_environment.presets.astrophysics import (  # noqa: E402
    _cw_exact_state,
    cw_exact_velocity,
)


def _fd_velocity(t, n, s0, h=1e-3):
    p, m = _cw_exact_state(np.array([t + h]), n, *s0), _cw_exact_state(np.array([t - h]), n, *s0)
    return np.array([(p[k][0] - m[k][0]) / (2 * h) for k in ("x", "y", "z")])


def test_cw_velocity_is_the_derivative_of_the_exact_position():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        n = rng.uniform(5e-4, 2e-3)
        s0 = (*rng.uniform(-2, 2, 3), *rng.uniform(-0.01, 0.01, 3))
        t = rng.uniform(10.0, 6000.0)
        v = cw_exact_velocity(np.array([t]), n, *s0)
        worst = max(worst, float(np.abs(_fd_velocity(t, n, s0) - [v["vx"][0], v["vy"][0], v["vz"][0]]).max()))
    assert worst < 1e-8


def test_the_old_example_sign_of_vy_fails_the_same_check():
    """examples/pde_environment/06 used vy = ... + 2 s vx0 ...; a finite difference shows it is wrong."""
    n, s0, t = 0.0011, (1.0, 0.0, 0.2, 0.004, -0.0015, 0.0008), 1500.0
    s, c = math.sin(n * t), math.cos(n * t)
    x0, _, _, vx0, vy0, _ = s0
    old_vy = 6 * n * (c - 1) * x0 + 2 * s * vx0 + (4 * c - 3) * vy0
    fd_vy = _fd_velocity(t, n, s0)[1]
    assert abs(old_vy - fd_vy) > 1e-3
    assert abs(cw_exact_velocity(np.array([t]), n, *s0)["vy"][0] - fd_vy) < 1e-9


def test_centered_quadrature_matches_the_rayleigh_closed_form():
    for r_c in (0.005, 0.01, 0.05, 0.1, 0.2):
        num = collision_probability_2d(0.0, 0.0, 0.05, 0.05, r_c)
        assert num == pytest.approx(collision_probability_centered(r_c, 0.05), rel=1e-7)


def test_offset_anisotropic_quadrature_matches_monte_carlo():
    xm, zm, sx, sz, r = 0.3, -0.1, 0.4, 0.15, 0.25
    pc = collision_probability_2d(xm, zm, sx, sz, r)
    rng = np.random.default_rng(1)
    n = 2_000_000
    pts = rng.normal([xm, zm], [sx, sz], size=(n, 2))
    mc = float(np.mean((pts ** 2).sum(axis=1) <= r * r))
    se = math.sqrt(mc * (1 - mc) / n)
    assert abs(pc - mc) < 4 * se


def test_closest_approach_is_a_true_minimum():
    n = cw_mean_motion(6771.0)
    s0 = (0.05, 1.0, 0.1, 0.0002, -0.0019, 0.00005)
    period = 2 * math.pi / n
    ca = cw_closest_approach(n, s0, period, n_samples=2000)
    assert ca.perpendicularity_residual < 1e-5
    t = np.linspace(0, period, 200_001)
    p = _cw_exact_state(t, n, *s0)
    dense_min = float(np.sqrt(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2).min())
    assert ca.miss_distance_km <= dense_min + 1e-9


def test_source_app_worked_example_is_reproduced():
    """SatScreen's README example: a = 6771 km, miss ~0.3729 km at ~423.5 s, Pc ~3.79e-5 (r_c 5 m, sigma 0.5 km)."""
    out = screen_cw_conjunction(cw_mean_motion(6771.0), (0.05, 1.0, 0.1, 0.0002, -0.0019, 0.00005),
                                hard_body_radius_km=0.005, sigma_km=0.5)
    ca = out["closest_approach"]
    assert ca["miss_distance_km"] == pytest.approx(0.3729, abs=1e-4)
    assert ca["t_tca_s"] == pytest.approx(423.5, abs=0.5)
    assert out["collision_probability"] == pytest.approx(3.79e-5, rel=5e-3)
    assert not out["hard_body_violated"] and out["warnings"] == ["sigma is larger than the miss distance: "
                                                                 "the 2-D Pc is hard to interpret in this regime"]


def _j2_energy(s, mu, J2, Re):
    x, y, z, vx, vy, vz = s
    r = math.sqrt(x * x + y * y + z * z)
    potential = -mu / r * (1 - J2 * (Re / r) ** 2 * (3 * (z / r) ** 2 - 1) / 2)
    return 0.5 * (vx * vx + vy * vy + vz * vz) + potential


def test_j2_propagation_conserves_the_energy_of_the_j2_potential():
    mu, J2, Re, a, inc = 398600.4418, 1.08262668e-3, 6378.137, 7000.0, math.radians(98.7)
    v = math.sqrt(mu / a)
    s0 = (a, 0.0, 0.0, 0.0, v * math.cos(inc), v * math.sin(inc))
    period = 2 * math.pi * math.sqrt(a ** 3 / mu)
    traj = propagate_two_body_j2(s0, np.linspace(0, 3 * period, 300), mu=mu, J2=J2, Re=Re)
    e = np.array([_j2_energy(s, mu, J2, Re) for s in traj])
    assert np.abs(e - e[0]).max() / abs(e[0]) < 1e-9
    # the J2 term is not zero: with J2 = 0 the energy of the J2 potential is not conserved along the path
    kep = propagate_two_body_j2(s0, np.linspace(0, 3 * period, 300), mu=mu, J2=0.0, Re=Re)
    assert np.abs(kep[-1, :3] - traj[-1, :3]).max() > 1.0


def test_j2_free_orbit_closes_after_one_period():
    mu, a = 398600.4418, 7000.0
    v = math.sqrt(mu / a)
    s0 = (a, 0.0, 0.0, 0.0, v, 0.0)
    period = 2 * math.pi * math.sqrt(a ** 3 / mu)
    end = propagate_two_body_j2(s0, [0.0, period], mu=mu, J2=0.0)[-1]
    assert np.abs(end[:3] - s0[:3]).max() < 1e-4
