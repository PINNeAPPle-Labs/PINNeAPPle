"""Electromagnetics core: constants and units, closed-form references against independent computations,
dispersive / anisotropic / hysteretic materials, the electrostatic solver, and the eddy-current source fix.

Every reference is checked against something computed another way (numerical integration, a solver, an
independent library, a limit), and convergence orders are measured, not assumed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from pinneapple_simulation.electromagnetics import constants as K
from pinneapple_simulation.electromagnetics import hysteresis as hy
from pinneapple_simulation.electromagnetics import media as md
from pinneapple_simulation.electromagnetics import mie
from pinneapple_simulation.electromagnetics import statics as st
from pinneapple_simulation.electromagnetics import units as un
from pinneapple_simulation.electromagnetics import waves as wv
from pinneapple_simulation.electromagnetics.electrostatics import capacitance, solve_electrostatic
from pinneapple_simulation.electromagnetics.fem import Geometry, solve_magnetostatic
from pinneapple_simulation.electromagnetics.materials import NU0
from pinneapple_simulation.electromagnetics.mesh import Ring, polar_mesh, rect_mesh


def orders(errors):
    e = np.asarray(errors, float)
    return np.log2(e[:-1] / e[1:])


# ------------------------------------------------------------------------------------------------ constants
def test_constants_are_consistent_and_shared_with_the_checker():
    from pinneapple_physics import maxwell_checks as mc

    assert K.EPS0 * K.MU0 * K.C0 ** 2 == pytest.approx(1.0, rel=1e-15)
    assert K.ETA0 == pytest.approx(376.730313412, rel=1e-10)        # CODATA 2022 characteristic impedance
    assert mc.MU0 == K.MU0 and mc.EPS0 == pytest.approx(K.EPS0, rel=1e-15)
    assert abs(K.MU0 / (4e-7 * math.pi) - 1) < 1e-9                  # 2019 SI: mu0 is measured, ~4 pi 1e-7


def test_maxwell_equations_are_dimensionally_consistent():
    assert un.check_equations() == []
    bad = {"faraday": [("d/dx", "E"), ("B",)]}                       # forgot d/dt
    assert un.check_equations(bad)
    probs = {p.where: p.message for p in un.check_units({"E": "V/m", "B": "mT", "H": "T", "J": "A/m^2"})}
    assert set(probs) == {"B", "H"} and "convert" in probs["B"] and "dimension" in probs["H"]
    assert un.dimension("Wb/m") == un.dimension("T*m")


# ------------------------------------------------------------------------------------------------ statics
def test_biot_savart_polygon_converges_to_the_loop_at_second_order():
    """biot_savart( of an inscribed polygon against the closed-form loop field (elliptic integrals)."""
    pts = np.array([[0.03, 0.01, 0.04], [0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.12, -0.05, -0.02]])
    exact = st.loop_field(pts, 0.1, 2.0)
    errs = []
    for n in (45, 90, 180, 360):
        s, e = st.circle_segments(0.1, n)
        errs.append(np.max(np.abs(st.biot_savart(pts, s, e, 2.0) - exact)) / np.max(np.abs(exact)))
    assert np.all(np.abs(orders(errs) - 2.0) < 0.05)
    # on the axis the elliptic-integral field reduces to the textbook formula
    z = np.linspace(-0.3, 0.3, 7)
    B = st.loop_field(np.column_stack([0 * z, 0 * z, z]), 0.1, 2.0)
    np.testing.assert_allclose(B[:, 2], st.loop_on_axis_B(z, 0.1, 2.0), rtol=1e-12)
    np.testing.assert_allclose(B[:, :2], 0.0, atol=1e-18)


def test_long_segment_is_the_infinite_wire_and_helmholtz_pair():
    """Closed-form limits: a long segment is the infinite wire; two loops give the Helmholtz pair."""
    p = np.array([[0.02, 0.01, 0.0], [-0.05, 0.03, 0.1]])
    L = 1e4
    B = st.biot_savart(p, [[0, 0, -L]], [[0, 0, L]], 3.0)
    np.testing.assert_allclose(B, st.infinite_wire_B(p, 3.0), rtol=1e-8)
    # two coaxial loops at +-R/2 give the Helmholtz-coil centre field
    R = 0.2
    Bc = st.loop_on_axis_B(0.1, R, 1.0) + st.loop_on_axis_B(-0.1, R, 1.0)
    assert Bc == pytest.approx(st.helmholtz_coil_center_B(R, 1.0), rel=1e-12)


def test_finite_solenoid_helix_approaches_current_sheet():
    """A Biot-Savart helix against the closed-form current-sheet solenoid."""
    z = np.array([0.0, 0.03, 0.05, 0.08])
    sheet = st.finite_solenoid_on_axis_B(z, 0.02, 0.1, 200, 1.5)
    s, e = st.helix_segments(0.02, 0.1, 200, n_per_turn=96)
    B = st.biot_savart(np.column_stack([0 * z, 0 * z, z]), s, e, 1.5)
    # discrete turns: ripple of order (pitch / radius)^2 near the axis, well below 1 %
    np.testing.assert_allclose(B[:, 2], sheet, rtol=5e-3)
    # long solenoid centre -> mu0 n I
    assert st.finite_solenoid_on_axis_B(0.0, 0.01, 10.0, 1e4, 1.0) == pytest.approx(K.MU0 * 1e3, rel=1e-5)


def test_mutual_inductance_matches_neumann_double_integral():
    """Maxwell's closed-form mutual inductance against Neumann's integral done numerically."""
    r1, r2, d = 0.1, 0.07, 0.05

    def integrand(t):  # Neumann: M = mu0/(4 pi) oint oint dl1.dl2 / |r1 - r2|, reduced to one integral
        return r1 * r2 * math.cos(t) / math.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * math.cos(t) + d * d)

    M = K.MU0 / (4 * math.pi) * 2 * math.pi * quad(integrand, 0, 2 * math.pi, epsabs=1e-14)[0]
    assert st.mutual_inductance_coaxial_loops(r1, r2, d) == pytest.approx(M, rel=1e-10)


# ------------------------------------------------------------------------------------------------ capacitors
def test_coax_capacitance_fem_second_order():
    """solve_electrostatic( against the closed-form coax capacitance 2 pi eps / ln(b/a)."""
    a, b, eps_r = 1e-3, 3.5e-3, 2.1
    errs = []
    for n in (16, 32, 64, 128):
        rings = [Ring(a + (b - a) * k / (n // 4), n) for k in range(n // 4 + 1)]
        m = polar_mesh(rings, lambda ly, r, p: "dielectric")
        s = solve_electrostatic(m, eps_r, {"inner": (m.rings[0], 1.0), "outer": (m.rings[-1], 0.0)})
        errs.append(abs(capacitance(s, 1.0) / st.coax_capacitance(a, b, eps_r) - 1))
    assert errs[-1] < 5e-4
    assert np.all(np.abs(orders(errs) - 2.0) < 0.05)


def test_parallel_plate_two_dielectrics_exact_on_any_mesh():
    """solve_electrostatic( reproduces the exact solution (piecewise linear) of a layered capacitor."""
    from pinneapple_physics.closed_form.electromagnetics import two_dielectric_capacitor

    d = 1e-3
    for n in (5, 11):
        m = rect_mesh(np.linspace(0, 0.01, n), np.linspace(0, d, n), lambda x, y: "a" if y < d / 2 else "b")
        eps_r = np.where(m.mask("a"), 1.0, 4.0)
        y = m.points[:, 1]
        s = solve_electrostatic(m, eps_r, {"ground": (np.flatnonzero(y == 0), 0.0),
                                           "plate": (np.flatnonzero(y == d), 1.0)})
        np.testing.assert_allclose(s.phi, two_dielectric_capacitor(y, 1.0, d, d / 2, 1.0, 4.0), atol=1e-12)
        # series capacitors per unit length along z for a 1 cm wide strip
        C = 0.01 / (d / 2 / (K.EPS0 * 1.0) + d / 2 / (K.EPS0 * 4.0))
        assert capacitance(s, 1.0) == pytest.approx(C, rel=1e-10)
        # normal D continuous across the interface, E jumps by eps2/eps1
        Dy = s.D[:, 1]
        np.testing.assert_allclose(Dy, Dy.mean(), rtol=1e-9)
    assert st.parallel_plate_capacitance(1e-2, 1e-3, 4.0) == pytest.approx(4 * K.EPS0 * 10, rel=1e-15)


# ------------------------------------------------------------------------------------------------ Mie
def test_mie_against_miepython_and_bohren_huffman():
    """mie.mie( against an independent implementation (miepython) as reference solution."""
    miepython = pytest.importorskip("miepython")
    for m, x in [(1.55, 5.213), (1.5 - 0.1j, 3.0), (1.33, 50.0), (4 - 3j, 1.2), (1.05, 0.01)]:
        qe, qs, qb, g = miepython.efficiencies_mx(m, x)   # miepython uses m = n - i k, the convention here
        r = mie.mie(m, x)
        assert r.qext == pytest.approx(qe, rel=1e-7) and r.qsca == pytest.approx(qs, rel=1e-7)
        assert r.qback == pytest.approx(qb, rel=1e-7) and r.g == pytest.approx(g, abs=1e-7)


def test_mie_limits_and_energy_conservation():
    """mie.mie( and mie_pec( against the published value of Bohren & Huffman and analytic limits."""
    # Bohren & Huffman appendix A test case: m = 1.55, radius 0.525 um, wavelength 0.6328 um
    r = mie.mie(1.55, 2 * math.pi * 0.525 / 0.6328)
    assert (round(r.qsca, 5), round(r.qext, 5), round(r.qback, 5)) == (3.10543, 3.10543, 2.92534)
    # Rayleigh limit Q_sca = 8/3 x^4 |(m^2 - 1)/(m^2 + 2)|^2, relative error O(x^2)
    errs = [abs(mie.mie(1.5, x).qsca / mie.rayleigh_qsca(1.5, x) - 1) for x in (0.1, 0.05, 0.025)]
    assert np.all(np.abs(orders(errs) - 2.0) < 0.05)
    errs = [abs(mie.mie_pec(x).qsca / (10 / 3 * x ** 4) - 1) for x in (0.1, 0.05, 0.025)]
    assert np.all(np.abs(orders(errs) - 2.0) < 0.05)
    # lossless: extinction = scattering; absorbing (Im m < 0 here): Q_abs > 0
    assert mie.mie(2.0, 4.0).qabs == pytest.approx(0.0, abs=1e-12)
    assert mie.mie(1.5 - 0.05j, 4.0).qabs > 0
    # optical theorem: Q_ext = (4 / x^2) Re S(0)
    r = mie.mie(1.7 - 0.02j, 6.0)
    S1, _ = r.amplitudes(0.0)
    assert r.qext == pytest.approx(4 / r.x ** 2 * S1[0].real, rel=1e-12)
    # extinction paradox: Q_ext -> 2 for a large sphere
    assert abs(mie.mie(1.33, 2000.0).qext - 2.0) < 0.02


# ------------------------------------------------------------------------------------------------ waves
@pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.0, 3.4)])
def test_fresnel_energy_brewster_and_tir(n1, n2):
    """wv.fresnel( against closed-form energy conservation, Brewster and total internal reflection."""
    th = np.linspace(0, 1.5, 40)
    rs, rp, ts, tp = wv.fresnel(n1, n2, th)
    st_ = n1 * np.sin(th) / n2
    prop = st_ < 1
    ct = np.sqrt(1 - st_[prop] ** 2 + 0j)
    f = (n2 * ct).real / (n1 * np.cos(th[prop]))
    np.testing.assert_allclose(np.abs(rs[prop]) ** 2 + f * np.abs(ts[prop]) ** 2, 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(rp[prop]) ** 2 + f * np.abs(tp[prop]) ** 2, 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(rs[~prop]), 1.0, atol=1e-12)      # total internal reflection
    assert abs(wv.fresnel(n1, n2, wv.brewster_angle(n1, n2))[1]) < 1e-12
    assert rp[0] == pytest.approx(-rs[0])                                  # Born & Wolf sign convention


def test_multilayer_matches_fresnel_and_quarter_wave_coatings():
    """wv.multilayer( against wv.fresnel( and closed-form quarter-wave stacks."""
    lam = 1.0e-6
    for th, pol, idx in ((0.3, "s", 0), (0.3, "p", 1)):
        r = wv.multilayer([], [], lam, th, pol, n_in=1.0, n_out=1.5)
        assert r.R == pytest.approx(abs(wv.fresnel(1.0, 1.5, th)[idx]) ** 2, rel=1e-12)
    ns = 1.52
    n_ar = math.sqrt(ns)
    assert wv.multilayer([n_ar], [lam / (4 * n_ar)], lam, n_out=ns).R == pytest.approx(0.0, abs=1e-15)
    # Bragg mirror of N (H L) pairs on a substrate: the stack's input admittance is Y = (nH/nL)^(2N) ns, so
    # R = ((1 - Y)/(1 + Y))^2 (Born & Wolf sec. 1.6.5)
    nH, nL, N = 2.3, 1.38, 6
    stack = [nH, nL] * N
    r = wv.multilayer(stack, [lam / (4 * n) for n in stack], lam, n_out=ns)
    b = (nH / nL) ** (2 * N) * ns
    assert r.R == pytest.approx(((1 - b) / (1 + b)) ** 2, rel=1e-12)
    # lossless stack: R + T = 1 over a band
    r = wv.multilayer(stack, [lam / (4 * n) for n in stack], np.linspace(0.6e-6, 1.6e-6, 50), 0.4, "p", n_out=ns)
    np.testing.assert_allclose(r.R + r.T, 1.0, atol=1e-12)


def test_waveguide_and_cavity_cutoffs():
    # WR-90: a = 22.86 mm, TE10 cut-off c / (2 a) = 6.557 GHz
    assert wv.rect_waveguide_cutoff(1, 0, 22.86e-3, 10.16e-3) == pytest.approx(6.557e9, rel=1e-4)
    modes = wv.rect_waveguide_modes(22.86e-3, 10.16e-3, 4)
    assert [(k, m, n) for _, k, m, n in modes] == [("TE", 1, 0), ("TE", 2, 0), ("TE", 0, 1), ("TE", 1, 1)]
    # circular guide: TE11 has the lowest cut-off, x'_11 = 1.8412; TM01: x_01 = 2.4048
    a = 0.01
    assert wv.circular_waveguide_modes(a, 1)[0][1:] == ("TE", 1, 1)
    assert wv.circular_waveguide_cutoff("TE", 1, 1, a) == pytest.approx(1.841184 * K.C0 / (2 * math.pi * a), rel=1e-6)
    assert wv.circular_waveguide_cutoff("TM", 0, 1, a) == pytest.approx(2.404826 * K.C0 / (2 * math.pi * a), rel=1e-6)
    # cube cavity: lowest resonance is three-fold degenerate (TE101, TE011, TM110)
    f, labels = wv.rect_cavity_modes(0.1, 0.1, 0.1, 1)[0]
    assert f == pytest.approx(K.C0 / (2 * 0.1) * math.sqrt(2), rel=1e-6) and len(labels) == 3
    # propagation constant: guide wavelength from beta
    fc = wv.rect_waveguide_cutoff(1, 0, 22.86e-3, 10.16e-3)
    beta = wv.propagation_constant(10e9, fc)
    assert beta == pytest.approx(2 * math.pi * 10e9 / K.C0 * math.sqrt(1 - (fc / 10e9) ** 2))


def test_hertzian_dipole_power_from_full_fields_at_any_radius():
    """Closed-form radiated power against the flux of the complete dipole fields."""
    omega = 2 * math.pi * 1e9
    k = omega / K.C0
    Il = 1e-3
    for r in (0.05 / k, 0.5 / k, 20 / k):          # near field included: only the real power crosses the sphere
        th = np.linspace(0, math.pi, 2001)
        _, Eth, Hph = wv.hertzian_dipole_fields(r, th, Il, omega)
        Sr = 0.5 * np.real(Eth * np.conj(Hph))
        P = np.trapezoid(Sr * 2 * math.pi * r * r * np.sin(th), th)
        assert P == pytest.approx(wv.hertzian_dipole_power(Il, omega), rel=1e-6)


def test_half_wave_dipole_resistance_from_pattern_integration():
    """Closed-form radiation resistance (Cin) against integration of the pattern."""
    th = np.linspace(1e-9, math.pi - 1e-9, 200001)
    F = wv.dipole_pattern(th, 0.5)
    # P = eta |I0|^2 / (4 pi) int F^2 sin dth (Balanis 4-68) and R = 2 P / |I0|^2
    R = 2 * K.ETA0 / (4 * math.pi) * np.trapezoid(F ** 2 * np.sin(th), th)
    assert R == pytest.approx(wv.half_wave_dipole_radiation_resistance(), rel=1e-6)
    assert wv.half_wave_dipole_radiation_resistance() == pytest.approx(73.08, abs=0.01)
    assert wv.half_wave_dipole_directivity() == pytest.approx(1.641, abs=1e-3)


def test_skin_depth_exact_and_good_conductor():
    w = 2 * math.pi * 60.0
    d = wv.skin_depth(w, 5.8e7)
    assert d == pytest.approx(8.53e-3, rel=2e-3)        # copper at 60 Hz: 8.5 mm
    assert wv.skin_depth(w, 5.8e7, exact=True) == pytest.approx(d, rel=1e-12)
    # poor conductor: exact differs from the good-conductor formula
    assert wv.skin_depth(2 * math.pi * 1e9, 0.01, eps_r=80, exact=True) > 1.5 * wv.skin_depth(2 * math.pi * 1e9, 0.01)


def test_cylinder_scattering_series_satisfies_helmholtz_and_interface_conditions():
    """The analytic solution (series) checked against its own PDE and interface conditions."""
    k0, a = 2 * math.pi, 0.3
    for eps in (4.0, 2.5 - 0.5j):
        phi = np.linspace(0, 2 * math.pi, 9)
        d = 1e-7
        f = lambda r, phi=phi, eps=eps: wv.cylinder_scattering_tm(r * np.cos(phi), r * np.sin(phi), k0, a, eps)  # noqa: E731
        np.testing.assert_allclose(f(a + d), f(a - d), atol=1e-5)
        dout = (f(a + 2 * d) - f(a + d)) / d
        din = (f(a - d) - f(a - 2 * d)) / d
        np.testing.assert_allclose(dout, din, atol=1e-3 * np.max(np.abs(dout)))
    pec = wv.cylinder_scattering_tm(a * np.cos(phi), a * np.sin(phi), k0, a, None)
    np.testing.assert_allclose(pec, 0.0, atol=1e-10)
    # Helmholtz with the local wavenumber, outside and inside (5-point Laplacian, h^2 error)
    h = 1e-3
    for (X, Y), kk in (((0.7, 0.2), k0), ((0.1, 0.05), 2 * k0)):
        g = lambda x, y: wv.cylinder_scattering_tm(x, y, k0, a, 4.0)  # noqa: E731
        lap = (g(X + h, Y) + g(X - h, Y) + g(X, Y + h) + g(X, Y - h) - 4 * g(X, Y)) / h ** 2
        assert abs(lap + kk ** 2 * g(X, Y)) < 1e-4 * kk ** 2 * abs(g(X, Y))


def test_spp_dispersion_asymptote():
    wp = 1.37e16
    metal = md.drude_metal(wp)
    w = np.array([0.3, 0.5, 0.65, 0.7]) * wp
    k = wv.spp_wavenumber(w, metal.eps_r(w), 1.0)
    assert np.all(k.real > w / K.C0) and np.all(np.diff(k.real) > 0)  # bound, below the light line in omega
    wsp = wp / math.sqrt(2)
    assert wv.spp_wavenumber(0.999 * wsp, metal.eps_r(0.999 * wsp), 1.0).real > 10 * wsp / K.C0


# ------------------------------------------------------------------------------------------------ media
def test_dispersive_models_and_ade_coefficients():
    """md.Drude( and the .ade( recursions against the closed-form susceptibility, second order in dt."""
    d = md.Drude(1e15, 1e14)
    w = 3e14
    assert d.chi(w) == pytest.approx(-1e30 / (w * w - 1j * w * 1e14))
    medium = md.Medium(2.0, poles=(d, md.Lorentz(1.5, 4e15, 1e14), md.Debye(70.0, 8e-12)), sigma=1.0)
    assert md.is_passive(medium, np.logspace(9, 16, 50))
    assert not md.is_passive(md.Medium(poles=(md.Drude(1e15, -1e14),)), [3e14])   # negative damping: gain
    assert medium.n(w).imag <= 0
    # the trapezoidal ADE of each pole reproduces its susceptibility at second order in dt
    for pole, w in ((d, 3e14), (md.Debye(70.0, 8e-12), 2e10)):
        errs = []
        for dt in (1e-16, 5e-17, 2.5e-17) if isinstance(pole, md.Drude) else (1e-12, 5e-13, 2.5e-13):
            ka, kb = pole.ade(dt)
            z = np.exp(1j * w * dt)
            jw = 2 / dt * (1 - 1 / z) / (1 + 1 / z)
            if isinstance(pole, md.Drude):    # J = dP/dt
                chi = kb * (1 + 1 / z) / (1 - ka / z) / (jw * K.EPS0)
            else:                             # P
                chi = kb * (1 + 1 / z) / (1 - ka / z) / K.EPS0
            errs.append(abs(chi / pole.chi(w) - 1))
        assert np.all(np.abs(orders(errs) - 2.0) < 0.05)
    lz = md.Lorentz(1.5, 4e15, 1e14)
    errs = []
    for dt in (2e-17, 1e-17, 5e-18):
        a1, a2, a3 = lz.ade(dt)
        z = np.exp(1j * 1e15 * dt)
        errs.append(abs(a3 / (1 - a1 / z - a2 / z ** 2) / z / K.EPS0 / lz.chi(1e15) - 1))
    assert np.all(np.abs(orders(errs) - 2.0) < 0.1)


def test_anisotropic_uniaxial_indices_and_reciprocity():
    """Indices against the closed-form uniaxial ellipse 1/n^2 = cos^2/no^2 + sin^2/ne^2."""
    no, ne = 1.5, 1.7
    u = md.Anisotropic.uniaxial(no, ne)
    for th in (0.0, 0.4, 1.0, math.pi / 2):
        n = u.indices([math.sin(th), 0, math.cos(th)])
        ext = 1 / math.sqrt(math.cos(th) ** 2 / no ** 2 + math.sin(th) ** 2 / ne ** 2)
        np.testing.assert_allclose(sorted(n.real), sorted([no, ext]), rtol=1e-12)
    R = md.rotation([1, 1, 0], 0.7)
    ur = u.rotated(R)
    s = R @ np.array([math.sin(0.4), 0, math.cos(0.4)])
    np.testing.assert_allclose(ur.indices(s), u.indices([math.sin(0.4), 0, math.cos(0.4)]), rtol=1e-12)
    assert u.is_reciprocal
    gyro = md.Anisotropic(np.array([[2, 0.3j, 0], [-0.3j, 2, 0], [0, 0, 2]]))   # magneto-optic, Hermitian
    assert not gyro.is_reciprocal


# ------------------------------------------------------------------------------------------------ hysteresis
def test_frohlich_kennelly_closed_forms():
    """Closed-form inverse, energy and derivative of FrohlichKennelly against quadrature and differences."""
    fk = hy.FrohlichKennelly.from_saturation(2000.0, 1.8)
    H = np.logspace(0, 6, 30)
    np.testing.assert_allclose(fk.h(fk.b(H)), H, rtol=1e-12)
    assert fk.nu_init == pytest.approx(1 / (K.MU0 * 2000.0), rel=1e-12)
    for b in (0.3, 1.5, 2.5):
        assert fk.energy(b * b) == pytest.approx(quad(lambda t: float(fk.h(t)), 0, b, epsabs=1e-12)[0], rel=1e-9)
        s, ds = b * b, 1e-6 * b * b
        assert fk.dnu(s) == pytest.approx((fk.nu(s + ds) - fk.nu(s - ds)) / (2 * ds), rel=1e-5)
    # polarisation saturates at 1/beta = Js
    assert float(fk.b(1e9) - K.MU0 * 1e9) == pytest.approx(1.8, rel=1e-4)


def test_frohlich_kennelly_iron_ring_in_the_newton_solver():
    """solve_magnetostatic( with FrohlichKennelly iron: B follows the curve at the analytic H = I / (2 pi r)."""
    fk = hy.FrohlichKennelly.from_saturation(3000.0, 1.9)
    a, r1, r2, R, n = 0.005, 0.01, 0.02, 0.03, 64
    radii = np.unique(np.concatenate([np.linspace(a / 4, a, 4), np.linspace(a, R, n // 2)[1:], [r1, r2]]))
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii],
                      lambda ly, r, p: "cu" if r < a else ("fe" if r1 < r < r2 else "air"))
    g = Geometry.of(mesh)
    J = np.where(mesh.mask("cu"), 1.0, 0.0)
    J *= 3000.0 / np.sum(J * g.area)
    sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J, nonlinear=[(mesh.mask("fe"), fk)])
    assert sol.residual < 1e-9 and sol.iterations < 20
    c = mesh.centroids()[mesh.mask("fe")]
    r = np.hypot(c[:, 0], c[:, 1])
    Bmag = np.hypot(*sol.B[mesh.mask("fe")].T)
    H = 3000.0 / (2 * math.pi * r)                           # Ampere: H independent of the material
    np.testing.assert_allclose(Bmag, fk.b(H), rtol=0.03)     # P1 element field vs centroid radius


def test_jiles_atherton_loop_properties_and_integrator_convergence():
    """JilesAtherton( loops: symmetry, analytic limit c = 1 (anhysteretic), integrator convergence."""
    ja = hy.JilesAtherton(Ms=1.6e6, a=1100.0, k=400.0, c=0.2, alpha=1.6e-3)
    loop, i0 = ja.major_loop(5000.0, 800)
    last = hy.HysteresisLoop(loop.H[i0:], loop.M[i0:])
    assert last.loss_per_cycle() > 0 and last.coercivity() > 0 and last.remanence() > 0
    # anti-symmetric major loop: the descending branch is minus the ascending one
    n = (len(last.H) - 1) // 2
    np.testing.assert_allclose(last.H[:n + 1], -last.H[n:], atol=1e-9)
    np.testing.assert_allclose(last.M[:n + 1], -last.M[n:], atol=1e-3 * ja.Ms)
    # wider pinning (k) -> wider loop
    wide = hy.JilesAtherton(1.6e6, 1100.0, 800.0, 0.2, 1.6e-3)
    lw, iw = wide.major_loop(5000.0, 800)
    assert hy.HysteresisLoop(lw.H[iw:], lw.M[iw:]).coercivity() > last.coercivity()
    # fully reversible (c = 1): the path stays on the anhysteretic curve, no loss
    rev = hy.JilesAtherton(1.6e6, 1100.0, 400.0, 1.0, 1.6e-3)
    lr, ir = rev.major_loop(5000.0, 400)
    assert abs(hy.HysteresisLoop(lr.H[ir:], lr.M[ir:]).loss_per_cycle()) < 1e-4 * last.loss_per_cycle()
    np.testing.assert_allclose(lr.M, rev.anhysteretic(lr.H + rev.alpha * lr.M), atol=1e-4 * rev.Ms)
    # RK4 sub-steps converge (irregularly: the pinning switch delta_M makes the right-hand side non-smooth)
    H = np.concatenate([np.linspace(0, 5e3, 101), np.linspace(5e3, -5e3, 201)[1:], np.linspace(-5e3, 5e3, 201)[1:]])
    ref = ja.trace(H, substeps=256).M
    errs = [np.max(np.abs(ja.trace(H, substeps=s).M - ref)) / ja.Ms for s in (2, 8, 32)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-8


# ------------------------------------------------------------------------------------------------ eddy FDM fix
def test_eddy_current_fdm_coil_field_matches_loop_formula():
    """Coil field of eddy_current against the analytic loop formula (closed form)."""
    from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
        MU0,
        annular_current_source,
        axial_flux_density,
        solve_axisymmetric_eddy_current,
    )

    errs = []
    for n in (121, 241, 481):            # box 20 coil radii wide: A = 0 on it truncates the field by < 1e-3
        r = np.linspace(0, 0.6, n)
        z = np.linspace(-0.6, 0.6, 2 * n - 1)
        J = annular_current_source(r, z, 0.03, 0.0, 0.004, 0.002, 10.0, 0.5)
        dr, dz = r[1] - r[0], z[1] - z[0]
        assert np.sum(J) * dr * dz == pytest.approx(5.0, rel=1e-12)          # exactly N I, any alignment
        RR = np.ones((n, len(z)))
        A = solve_axisymmetric_eddy_current(r, z, 0.0, MU0 * RR, 0 * RR, J)
        Bz = axial_flux_density(A, r, z).real
        # reference: the coil's own cross-section as a sum of filament loops on the band nodes
        k = int(np.argmin(np.abs(z - 0.05)))
        ii, jj = np.nonzero(J)
        ref = np.sum(J[ii, jj] * dr * dz * MU0 * r[ii] ** 2 / (2 * (r[ii] ** 2 + (z[jj] - z[k]) ** 2) ** 1.5))
        errs.append(abs(Bz[0, k] / ref - 1))
    assert errs[-1] < 1e-3 and errs[0] > errs[-1]


def test_eddy_current_density_sign_satisfies_ampere():
    """eddy_current density against Ampere's law on the solver's own stencil (exact discrete identity)."""
    from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
        MU0,
        annular_current_source,
        build_axisymmetric_helmholtz_system,
        eddy_current_density,
        solve_axisymmetric_eddy_current,
    )

    r = np.linspace(0, 0.05, 41)
    z = np.linspace(-0.05, 0.05, 81)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    sigma = np.where(ZZ < 0, 3.5e7, 0.0)
    mu = MU0 * np.ones_like(RR)
    omega = 2 * math.pi * 5e3
    Js = annular_current_source(r, z, 0.02, 0.01, 0.004, 0.003, 50.0, 1.0)
    A = solve_axisymmetric_eddy_current(r, z, omega, mu, sigma, Js)
    Je = eddy_current_density(A, omega, sigma)
    # Ampere: L[A] = -mu (J_source + J_eddy) at every interior node, with the same stencil (k2 = 0)
    L, _ = build_axisymmetric_helmholtz_system(r, z, np.zeros_like(RR, complex), np.zeros_like(RR, complex))
    lhs = (L @ A.ravel()).reshape(A.shape)[1:-1, 1:-1]
    rhs = (-mu * (Js + Je))[1:-1, 1:-1]
    assert np.max(np.abs(lhs - rhs)) < 1e-8 * np.max(np.abs(rhs))


def test_loop_self_inductance_from_filament_mutual_inductance():
    """Closed-form thin-loop inductance mu0 R (ln(8R/a) - 2) against Maxwell's formula for the axis filament and
    the inner-edge filament (the classical external-inductance construction), converging as a / R -> 0."""
    R = 0.1
    errs = []
    for a in (1e-3, 5e-4, 2.5e-4):
        L = st.loop_self_inductance(R, a, internal=False)
        errs.append(abs(st.mutual_inductance_coaxial_loops(R, R - a, 0.0) / L - 1))
    assert errs[-1] < 2e-3 and errs[0] > errs[1] > errs[2]
    assert st.loop_self_inductance(R, 1e-3) - st.loop_self_inductance(R, 1e-3, internal=False) == \
        pytest.approx(K.MU0 * R / 4, rel=1e-12)
