"""Yee FDTD (``pinneapple_simulation.electromagnetics.fdtd``): stability, energy, boundaries, sources and the
total-field / scattered-field box. The accuracy cases against closed forms (Fresnel, dispersive media, cavities,
waveguide, plasma, cylinder scattering, Hertzian dipole, Floquet, CPML) are in the verification suite
(``tests/test_em_verification.py``)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics import fdtd as F
from pinneapple_simulation.electromagnetics.constants import C0
from pinneapple_simulation.electromagnetics.media import Debye, Drude, Medium


def test_courant_limit_is_enforced_and_is_the_real_stability_limit():
    dx = 1e-3
    g = F.YeeGrid((40, 40, 1), (dx, dx, dx))
    assert g.dt == pytest.approx(0.99 * dx / (C0 * math.sqrt(2)))
    with pytest.raises(ValueError, match="Courant"):
        F.YeeGrid((40, 40, 1), (dx, dx, dx), dt=1.01 * g.dt_max)

    def grow(dt_factor):
        gg = F.YeeGrid((40, 40, 1), (dx, dx, dx), dt=dt_factor * g.dt_max, allow_unstable=True)
        s = F.FDTD(gg)
        rng = np.random.default_rng(0)
        s.E["Ez"][1:-1, 1:-1, 0] = rng.standard_normal((38, 38))
        e0 = np.abs(s.E["Ez"]).max()
        s.run(400)
        return np.abs(s.E["Ez"]).max() / e0

    assert grow(0.999) < 10.0         # bounded below the limit, even for grid-scale noise
    assert grow(1.02) > 1e6           # exponential blow-up just above it


def test_lossless_cavity_conserves_energy_and_conductor_dissipates_it():
    """FDTD( discrete energy: an exact invariant of the Yee scheme in a lossless cavity; dissipation with sigma."""
    dx = 1e-2
    g = F.YeeGrid((31, 25, 1), (dx, dx, dx))
    s = F.FDTD(g)
    s.add_source("Ez", (10, 9, 0), F.gaussian_pulse(30 * g.dt, 8 * g.dt), "soft")
    s.run(100)
    e = []
    for _ in range(30):
        s.run(50)
        e.append(s.energy())
    assert (max(e) - min(e)) / np.mean(e) < 1e-12           # the scheme's discrete energy is exactly conserved
    lossy = F.FDTD(g)
    lossy.set_material(np.ones(g.shape, bool), Medium(1.0, sigma=0.05))
    lossy.add_source("Ez", (10, 9, 0), F.gaussian_pulse(30 * g.dt, 8 * g.dt), "soft")
    lossy.run(100)
    e1 = lossy.energy()
    lossy.run(500)
    assert lossy.energy() < 0.2 * e1


@pytest.mark.parametrize("wall,sign", [("pec", -1.0), ("pmc", 1.0)])
def test_pec_and_pmc_walls_reflect_with_opposite_signs(wall, sign):
    """FDTD( walls: the analytic reflection coefficients r = -1 (PEC) and +1 (PMC)."""
    dx = 1e-3
    N = 400
    g = F.YeeGrid((N, 1, 1), (dx, dx, dx))
    s = F.FDTD(g, {"x": ("cpml", wall), "y": "periodic", "z": "periodic"}, npml=20)
    s.add_source("Ey", (100, 0, 0), F.gaussian_pulse(60 * g.dt, 15 * g.dt), "soft")
    p = s.add_probe("Ey", (200, 0, 0))
    s.run(1200)
    v = np.asarray(p.values)
    first, second = v[:500], v[500:]                         # incident, then reflected from the wall
    peak_in = first[np.argmax(np.abs(first))]
    peak_out = second[np.argmax(np.abs(second))]
    assert np.sign(peak_out) == sign * np.sign(peak_in)
    assert abs(peak_out) == pytest.approx(abs(peak_in), rel=0.02)


def test_periodic_axis_wraps_a_pulse_around():
    """FDTD( in 1-D at the magic time step reproduces the exact solution (d'Alembert) after one period."""
    dx = 1e-3
    N = 200
    g = F.YeeGrid((N, 1, 1), (dx, dx, dx), courant=1.0)    # 1-D at the magic time step: exact transport
    s = F.FDTD(g, {"x": "periodic", "y": "periodic", "z": "periodic"})
    x = np.arange(N)
    s.E["Ey"][:, 0, 0] = np.exp(-((x - 50) / 8.0) ** 2)     # H = 0: two halves travel to +x and -x
    E0 = s.E["Ey"][:, 0, 0].copy()
    s.run(N)                                                 # each half goes once around and they meet again
    assert np.max(np.abs(s.E["Ey"][:, 0, 0] - E0)) < 1e-6


def test_tfsf_box_leaks_nothing_and_reproduces_the_incident_wave_2d_and_3d():
    """FDTD( TF/SF: zero leakage and the total field equals the reference solution (the incident wave)."""
    f0 = 1e9
    dx = C0 / f0 / 20
    for shape, box in (((80, 70, 1), ((20, 60), (15, 55), (0, 1))), ((40, 36, 34), ((12, 28), (10, 26), (9, 25)))):
        g = F.YeeGrid(shape, (dx, dx, dx))
        s = F.FDTD(g, {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=8)
        s.add_plane_wave(F.PlaneWaveTFSF("x", "Ez", F.modulated_gaussian(f0, 0.6 * f0), box))
        inc = s.incident_monitor([f0])
        mon = s.add_dft(["Ez"], [f0])
        leak = 0.0
        for _ in range(500):
            s.step()
            sf = np.abs(s.E["Ez"]).copy()
            sl = tuple(slice(lo, hi) for lo, hi in box)
            sf[sl] = 0.0
            leak = max(leak, float(sf.max()))
        assert leak < 1e-12
        Ez = mon.phasor("Ez")[0]
        i = np.arange(box[0][0], box[0][1])
        ratio = Ez[i, box[1][0] + 3, box[2][0]] / inc.phasor("Ey")[0, i, 0, 0]
        np.testing.assert_allclose(ratio, 1.0, atol=1e-10)


def test_fill_fraction_and_partial_poles():
    g = F.YeeGrid((20, 20, 1), (1.0, 1.0, 1.0))
    fr = F.fill_fraction(g, lambda x, y, z: x < 9.5, 8)       # node 9's cell is split at x = 9.5: half inside
    assert fr[5, 5, 0] == 1.0 and fr[12, 5, 0] == 0.0 and fr[9, 5, 0] == pytest.approx(1.0)
    fr = F.fill_fraction(g, lambda x, y, z: x < 9.0, 8)
    assert fr[9, 5, 0] == pytest.approx(0.5)
    s = F.FDTD(g)
    s.set_material(fr, Medium(3.0, poles=(Debye(4.0, 1e-9), Drude(1e10))))
    assert s.eps_inf[9, 5, 0] == pytest.approx(2.0)             # arithmetic mixing with vacuum
    half = [p for m, p in s._poles if m[9, 5, 0]]
    assert {type(p).__name__ for p in half} == {"Debye", "Drude"}
    deb = next(p for p in half if isinstance(p, Debye))
    dru = next(p for p in half if isinstance(p, Drude))
    assert deb.delta_eps == pytest.approx(2.0) and dru.omega_p ** 2 == pytest.approx(0.5e20)


def test_numerical_dispersion_relation():
    f = 1e9
    w = 2 * math.pi * f
    dx = C0 / f / 10
    g = F.YeeGrid((10, 10, 1), (dx, dx, dx))
    k_axis = F.numerical_dispersion(w, g.dt, (dx, dx, dx), (1, 0, 0))
    k_diag = F.numerical_dispersion(w, g.dt, (dx, dx, dx), (1, 1, 0))
    k0 = w / C0
    # the Yee grid is slowest along the axes; both converge to k0 as dx -> 0
    assert k_axis > k_diag > k0 * (1 - 1e-12)
    fine = F.numerical_dispersion(w, g.dt / 4, (dx / 4,) * 3, (1, 0, 0))
    assert abs(fine / k0 - 1) < abs(k_axis / k0 - 1) / 10


def test_snapshot_layout_feeds_the_maxwell_checks():
    from pinneapple_physics import maxwell_checks as mc

    nx = 31
    dx = 1.0 / (nx - 1)
    s = F.FDTD(F.YeeGrid((nx, nx, 1), (dx, dx, dx)))
    x = np.arange(nx) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    s.E["Ez"][:, :, 0] = np.sin(np.pi * X) * np.sin(np.pi * Y)
    frames, ts = [], []
    for _ in range(60):
        s.step()
        t, f = s.snapshot()
        frames.append(f)
        ts.append(t)
    fields = {c: np.stack([fr[c][:, :, 0] for fr in frames], -1) for c in ("Ez", "Hx", "Hy")}
    rep, checks = mc.maxwell_trust_report({"x": x, "y": x, "t": np.array(ts)}, fields)
    assert rep.decisao == "APPROVED", [c.to_dict() for c in checks]
    bad = dict(fields, Hx=1.3 * fields["Hx"])
    rep_bad, _ = mc.maxwell_trust_report({"x": x, "y": x, "t": np.array(ts)}, bad)
    assert rep_bad.decisao == "REJECT"


def test_fdtd_reflection_matches_the_closed_form_fresnel_coefficient():
    """FDTD( 1-D, eps = 4 half-space: |r| against the closed form (n1 - n2)/(n1 + n2) = 1/3, second order."""
    from pinneapple_simulation.electromagnetics.verification import _fdtd_reflection_1d

    freqs = np.linspace(0.6, 1.4, 5) * 1e9
    errs = [np.max(np.abs(np.abs(_fdtd_reflection_1d(n, Medium(4.0), 1e9, 0.8, 12, freqs)) - 1 / 3))
            for n in (20, 40)]
    assert errs[1] < 1e-2 and 3.5 < errs[0] / errs[1] < 4.6


def test_near_to_far_field_reproduces_the_dipole_closed_form():
    """NearToFarField( around a one-cell current: E_theta follows the closed form eta k I l sin(theta) / (4 pi) and
    the radiated power eta k^2 |I l|^2 / (12 pi) (10 cells per wavelength)."""
    from pinneapple_simulation.electromagnetics.constants import ETA0
    from pinneapple_simulation.electromagnetics.waves import hertzian_dipole_power

    f0 = 1e9
    lam = C0 / f0
    dx = lam / 10
    N = int(1.2 * lam / dx) + 28
    c = N // 2
    s = F.FDTD(F.YeeGrid((N, N, N), (dx, dx, dx)), {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=10)
    w = F.modulated_gaussian(f0, 0.5 * f0)
    s.add_current("Ez", (c, c, c), w, 1.0)
    nt = F.NearToFarField(s, [f0], ((14, N - 14),) * 3)
    s.run_until(14 / f0)
    tj = (np.arange(s.n) + 0.5) * s.dt
    Il = np.sum(np.array([w(x) for x in tj]) * np.exp(-2j * np.pi * f0 * tj)) * s.dt * dx ** 3
    th = np.linspace(0.05, np.pi - 0.05, 19)
    Et, Ep = nt.far_field(th, 0.3 + 0 * th)                    # any azimuth: the dipole is symmetric
    exact = ETA0 * (2 * np.pi / lam) * abs(Il) * np.sin(th) / (4 * np.pi)
    assert np.max(np.abs(np.abs(Et[0]) - exact)) / exact.max() < 0.02
    # cross-polarisation from the cubic grid's anisotropy (it vanishes at phi = 0 by symmetry): 0.7 % here
    assert np.max(np.abs(Ep[0])) < 1e-2 * exact.max()
    assert float(nt.radiated_power()[0]) == pytest.approx(hertzian_dipole_power(Il, 2 * np.pi * f0), rel=0.015)
