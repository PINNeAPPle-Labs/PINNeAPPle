"""Non-axisymmetric emission in the Schwarzschild ray tracer: azimuth per sample, (r, theta, phi) fields, an
azimuthal modulation, the camera orbit without a re-trace, and orbiting clumps (issue #1091)."""
import math

import numpy as np
import pytest

from pinneapple_physics.blackhole.raytrace import Camera, keplerian_omega, orbiting_clumps, render


def _grid(nr=48, nth=64, r_in=2.05, r_out=40.0):
    rf = np.geomspace(r_in, r_out, nr + 1)
    tf = np.linspace(0, math.pi, nth + 1)
    return {"r_faces": rf, "theta_faces": tf, "r": np.sqrt(rf[1:] * rf[:-1]), "theta": 0.5 * (tf[1:] + tf[:-1])}


def _thin_disc(g, r_lo=6.0, r_hi=30.0):
    r, th = g["r"][:, None], g["theta"][None, :]
    rho = np.where((np.abs(th - math.pi / 2) < 0.06) & (r > r_lo) & (r < r_hi), 1.0, 0.0) * np.ones_like(th)
    vphi = np.minimum(np.sqrt(1 / np.maximum(r - 2, 1e-6)), 0.95) * (rho > 0)
    vel = np.stack([np.zeros_like(rho), np.zeros_like(rho), vphi])
    return rho, np.ones_like(rho), vel


@pytest.fixture(scope="module")
def traced():
    g = _grid()
    cam = Camera(width=64, height=40, r_cam=120.0, inclination_deg=80.0, fov_deg=26.0)
    return g, cam.trace(g, r_max=35.0)


def test_samples_carry_their_azimuth(traced):
    g, p = traced
    assert p.phi is not None and p.phi.shape == p.ray.shape
    assert np.all(np.abs(p.phi) <= math.pi + 1e-6)
    # the near side of the disc faces the camera (phi = 0) and the far side is behind the hole (|phi| ~ pi)
    assert (np.abs(p.phi) < 0.5).any() and (np.abs(p.phi) > math.pi - 0.5).any()


def test_a_phi_constant_field_renders_like_the_axisymmetric_one(traced):
    g, p = traced
    rho, T, vel = _thin_disc(g)
    a = render(p, rho, T, vel, return_intensity=True)
    b = render(p, np.repeat(rho[..., None], 16, 2), np.repeat(T[..., None], 16, 2), np.repeat(vel[..., None], 16, 3),
               return_intensity=True)
    c = render(p, rho, T, vel, modulation=np.ones((len(g["r"]), 32)), return_intensity=True)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-12) and np.allclose(a, c, rtol=1e-6, atol=1e-12)


def test_camera_orbit_equals_the_flow_turned_back(traced):
    """Static, axisymmetric metric: moving the camera by +alpha sees the flow turned by +alpha from the old camera."""
    g, p = traced
    rho, T, vel = _thin_disc(g)
    m = orbiting_clumps(g["r"], t=0.0, n_phi=64, seed=3)
    k = 11                                                     # whole cells, so the rolled grid is exact
    alpha = k * 2 * math.pi / 64
    turned = np.roll(m, k, axis=1)                             # pattern rotated by +alpha
    a = render(p, rho, T, vel, modulation=m, return_intensity=True)
    b = render(p, rho, T, vel, modulation=turned, camera_azimuth=alpha, return_intensity=True)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-12)
    assert not np.allclose(a, render(p, rho, T, vel, modulation=turned, return_intensity=True), rtol=1e-3)


def test_clumps_orbit_at_the_keplerian_rate_and_shear():
    r = np.array([8.0, 16.0])
    kw = dict(n_phi=360, n_clumps=6, r_range=(7.0, 18.0), turbulence=0.0, seed=1)
    m0 = orbiting_clumps(r, 0.0, **kw)
    period_in = 2 * math.pi / keplerian_omega(8.0)
    assert abs(period_in - 2 * math.pi * 8.0 ** 1.5) < 1e-9
    m1 = orbiting_clumps(r, period_in, **kw)
    assert np.allclose(m1[0], m0[0], atol=1e-9)                # inner ring back where it started ...
    assert not np.allclose(m1[1], m0[1], atol=1e-3)            # ... the outer ring is not: differential rotation
    half = orbiting_clumps(r, period_in / 2, **kw)
    assert np.allclose(np.roll(m0[0], 180), half[0], atol=1e-6)  # half an orbit later it is on the other side
    assert m0.min() >= 0


def test_a_clump_is_brighter_on_the_approaching_side(traced):
    """Camera at phi = 0 on +x, gas turning towards +phi: the side at phi = -pi/2 comes towards the camera."""
    g, p = traced
    rho, T, vel = _thin_disc(g)
    r = g["r"][:, None]
    phi = (np.arange(128) + 0.5) / 128 * 2 * math.pi

    def spot(phi0):
        d = np.angle(np.exp(1j * (phi[None, :] - phi0)))
        return np.exp(-0.5 * ((r - 12.0) / 1.0) ** 2) * np.exp(-0.5 * (12.0 * d / 2.0) ** 2) + 1e-9

    tot = {s: render(p, rho, T, vel, modulation=spot(s * math.pi / 2), return_intensity=True).sum() for s in (-1, 1)}
    ratio = tot[-1] / tot[1]
    assert ratio > 2.0
    flat = {s: render(p, rho, T, vel, modulation=spot(s * math.pi / 2), doppler=False, return_intensity=True).sum()
            for s in (-1, 1)}
    assert abs(flat[-1] / flat[1] - 1) < 0.15                  # without beaming the two sides are about equal


def test_select_keeps_the_image_when_only_empty_samples_go(traced):
    g, p = traced
    rho, T, vel = _thin_disc(g)
    from pinneapple_physics.blackhole.raytrace import _bilinear
    keep = _bilinear(rho, p.ir.astype(float), p.ith.astype(float)) > 0
    q = p.select(keep)
    assert len(q.ray) < len(p.ray)
    assert np.allclose(render(q, rho, T, vel, return_intensity=True), render(p, rho, T, vel, return_intensity=True))


def test_paths_without_azimuth_are_refused_for_3d_fields(traced):
    from dataclasses import replace
    g, p = traced
    rho, T, vel = _thin_disc(g)
    with pytest.raises(ValueError, match="azimuth"):
        render(replace(p, phi=None), rho, T, vel, modulation=np.ones((len(g["r"]), 8)))


def test_render_series_matches_render_frame_by_frame(traced):
    from pinneapple_physics.blackhole.raytrace import render_series
    g, p = traced
    rho, T, vel = _thin_disc(g)
    sky = np.random.default_rng(0).uniform(0, 0.2, (32, 64, 3))
    frames = [(orbiting_clumps(g["r"], t, n_phi=96, seed=2), az) for t, az in ((0.0, 0.0), (40.0, 0.3), (90.0, -0.7))]
    kw = dict(emissivity_power=1.0, exposure=1.5, scale=0.01, gamma=0.8, sky=sky, sky_gain=0.5)
    got = list(render_series(p, rho, T, vel, frames, **kw))
    for (m, az), img in zip(frames, got, strict=True):
        assert np.allclose(img, render(p, rho, T, vel, modulation=m, camera_azimuth=az, **kw), atol=1e-9)
    inten = next(render_series(p, rho, T, vel, frames[:1], emissivity_power=1.0, return_intensity=True))
    assert np.allclose(inten, render(p, rho, T, vel, modulation=frames[0][0], emissivity_power=1.0,
                                     return_intensity=True))
    with pytest.raises(ValueError, match="axisymmetric"):
        next(render_series(p, rho[..., None], T, vel, frames))


def test_orbiting_clumps_literal_call():
    """Plain-number call (also the Studio's smoke input)."""
    m = orbiting_clumps([7.0, 9.0, 12.0, 16.0], 50.0, n_phi=64, n_clumps=8, seed=4)
    assert m.shape == (4, 64) and m.min() >= 0 and m.max() > 1
