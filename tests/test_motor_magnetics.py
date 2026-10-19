"""Verification of pinneapple_simulation.electromagnetics against closed-form references.

Every tolerance below was set from a run of this code (the measured error is quoted next to it) and
every reference is either a closed-form solution (pinneapple_simulation/electromagnetics/analytic.py)
or a consistency identity (two independent torque methods, periodicity, power balance).
"""
import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics import (
    MU0,
    NU0,
    Geometry,
    Layer,
    LinearMaterial,
    RadialLayers,
    Ring,
    coax_field,
    illustrative_steel,
    magnetized_cylinder,
    polar_mesh,
    skin_effect_wire,
    solve_magnetostatic,
    solve_time_harmonic,
    wire_field,
)
from pinneapple_simulation.electromagnetics.analytic import round_conductor_eddy_loss_low_freq
from pinneapple_simulation.electromagnetics.machine import SPMMachine, belt_winding
from pinneapple_simulation.electromagnetics.postprocess import (
    IronLossModel,
    back_emf,
    classical_eddy_coefficient,
    iron_loss,
    magnet_eddy_loss_resistance_limited,
    torque_maxwell_stress,
)


def _disc(a, R, n, inner_layers=None, outer_layers=None, geom=True):
    k_in = inner_layers or n // 8
    radii_in = np.linspace(a / k_in, a, k_in)
    radii_out = (np.geomspace if geom else np.linspace)(a, R, outer_layers or n // 4)[1:]
    return [Ring(0.0, 1)] + [Ring(r, n) for r in np.concatenate([radii_in, radii_out])]


# ----------------------------------------------------------------------------- meshes
def test_zipper_mesh_is_valid_and_band_keeps_element_count():
    rings = [Ring(0.0, 1), Ring(0.01, 12, rotating=True), Ring(0.02, 24, rotating=True),
             Ring(0.025, 24, rotating=True), Ring(0.03, 24), Ring(0.04, 24)]
    areas, counts = [], []
    for theta in (0.0, 0.07, 0.2, 1.0):
        m = polar_mesh(rings, lambda ly, r, p: "air", theta=theta)
        assert np.all(m.areas() > 0)
        areas.append(m.areas().sum())
        counts.append(m.n_elements)
    assert len(set(counts)) == 1
    n = 24
    polygon = 0.5 * n * 0.04 ** 2 * math.sin(2 * math.pi / n)
    assert np.allclose(areas, polygon, rtol=1e-12)


# ----------------------------------------------------------------------------- statics
def test_round_wire_matches_amperes_law_and_converges_second_order():
    """solve_magnetostatic( against the closed-form wire potential (Ampere's law), second order."""
    a, R, current = 0.01, 0.05, 100.0
    errs = []
    for n in (32, 64):
        mesh = polar_mesh(_disc(a, R, n), lambda ly, r, p: "cu" if r < a else "air")
        g = Geometry.of(mesh)
        J = np.where(mesh.mask("cu"), 1.0, 0.0)
        J *= current / np.sum(J * g.area)
        sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J)
        A_ex, _ = wire_field(np.hypot(*mesh.points.T), current, a, R)
        errs.append(np.abs(sol.A - A_ex).max() / np.abs(A_ex).max())
    assert errs[1] < 2e-3  # measured 1.34e-3
    assert errs[0] / errs[1] > 3.5  # measured 4.3


def test_coaxial_cable_field():
    """B is piecewise constant on P1 elements, so its L2 error is first order."""
    a, b, c, R, current = 0.004, 0.008, 0.010, 0.015, 50.0
    errs = []
    for n, k in ((96, 8), (192, 16)):
        radii = np.unique(np.concatenate([np.linspace(a / k, a, k), np.linspace(a, b, k),
                                          np.linspace(b, c, k), np.linspace(c, R, k // 2)]))
        rings = [Ring(0.0, 1)] + [Ring(r, n) for r in radii]

        def cls(ly, r, p):
            return "in" if r < a else ("out" if b < r < c else "air")

        mesh = polar_mesh(rings, cls)
        g = Geometry.of(mesh)
        J = np.zeros(mesh.n_elements)
        for name, cur in (("in", current), ("out", -current)):
            m = mesh.mask(name)
            J[m] = cur / g.area[m].sum()
        sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J)
        rc = np.hypot(*mesh.centroids().T)
        B_ex = coax_field(rc, current, a, b, c)
        errs.append(np.sqrt(np.sum(g.area * (np.hypot(*sol.B.T) - B_ex) ** 2) / np.sum(g.area * B_ex ** 2)))
    assert errs[1] < 0.015  # measured 1.13e-2 (2.30e-2 on the coarse mesh)
    assert errs[0] / errs[1] > 1.8  # measured 2.04


def test_magnetized_cylinder_uniform_inner_field():
    a, R, Br, alpha = 0.01, 0.06, 1.2, 0.3
    mesh = polar_mesh(_disc(a, R, 128), lambda ly, r, p: "pm" if r < a else "air")
    m = mesh.mask("pm")
    Brv = np.zeros((mesh.n_elements, 2))
    Brv[m] = Br * np.array([math.cos(alpha), math.sin(alpha)])
    sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), Br=Brv)
    A_ex, B_in = magnetized_cylinder(*mesh.points.T, Br, a, R, alpha)
    assert np.abs(sol.A - A_ex).max() / np.abs(A_ex).max() < 2e-3  # measured 1.44e-3
    g = sol.geometry
    B_avg = (g.area[m, None] * sol.B[m]).sum(0) / g.area[m].sum()
    assert np.allclose(B_avg, B_in * np.array([math.cos(alpha), math.sin(alpha)]), rtol=3e-3)  # 1.4e-3


def test_antiperiodic_half_model_equals_full_model():
    a, R, Br, n = 0.01, 0.06, 1.2, 64
    full = polar_mesh(_disc(a, R, n), lambda ly, r, p: "pm" if r < a else "air")
    half = polar_mesh(_disc(a, R, n // 2, inner_layers=n // 8, outer_layers=n // 4), lambda ly, r, p: "pm" if r < a else "air", span=math.pi)
    sols = []
    for mesh, periodic, dn in (
        (full, None, full.boundary),
        (half, [(int(r[-1]), int(r[0]), -1.0) for r in half.rings[1:]], np.concatenate([half.boundary, [0]])),
    ):
        Brv = np.zeros((mesh.n_elements, 2))
        Brv[mesh.mask("pm")] = (Br, 0.0)
        sols.append(solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), Br=Brv, dirichlet=dn,
                                        periodic=periodic))
    A_full_ex, _ = magnetized_cylinder(*full.points.T, Br, a, R)
    A_half_ex, _ = magnetized_cylinder(*half.points.T, Br, a, R)
    e_full = np.abs(sols[0].A - A_full_ex).max() / np.abs(A_full_ex).max()
    e_half = np.abs(sols[1].A - A_half_ex).max() / np.abs(A_half_ex).max()
    assert e_half == pytest.approx(e_full, rel=1e-6)


@pytest.mark.parametrize("current", [50.0, 2000.0, 20000.0])
def test_nonlinear_iron_shell_follows_bh_curve(current):
    """A wire inside an iron ring: H = I / (2 pi r) whatever the material, so B = B(H) on the curve."""
    from scipy.optimize import brentq

    steel = illustrative_steel()
    a, r1, r2, R, n = 0.005, 0.01, 0.02, 0.03, 64
    radii = np.unique(np.concatenate([np.linspace(a / 4, a, 4), np.linspace(a, R, n // 2)[1:], [r1, r2]]))
    rings = [Ring(0.0, 1)] + [Ring(r, n) for r in radii]
    mesh = polar_mesh(rings, lambda ly, r, p: "cu" if r < a else ("fe" if r1 < r < r2 else "air"))
    g = Geometry.of(mesh)
    J = np.where(mesh.mask("cu"), 1.0, 0.0)
    J *= current / np.sum(J * g.area)
    sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J, nonlinear=[(mesh.mask("fe"), steel)])
    assert sol.residual < 1e-9
    m = mesh.mask("fe")
    H = current / (2 * math.pi * np.hypot(*mesh.centroids()[m].T))
    B_ex = np.array([brentq(lambda b, h=h: float(steel.h(b)) - h, 0.0, 10.0) for h in H])
    err = np.sqrt(np.sum(g.area[m] * (np.hypot(*sol.B[m].T) - B_ex) ** 2) / np.sum(g.area[m] * B_ex ** 2))
    assert err < 4e-3  # measured 3.0e-3, 1.9e-4, 1.2e-3


# ----------------------------------------------------------------------------- slotless machine
def _slotless(n, theta, p=2, Br0=1.1, J0=3e6, phJ=0.7, mur_fe=1000.0, mur_m=1.05):
    r_ri, r_m, r_s, r_w, R_o = 0.03, 0.034, 0.035, 0.038, 0.05
    g = r_s - r_m

    def nr(r0, r1, k):
        return list(np.linspace(r0, r1, k + 1)[1:])

    rot = nr(0, r_ri, 6) + nr(r_ri, r_m, 6) + [r_m + g / 5, r_m + 2 * g / 5]
    sta = [r_m + 3 * g / 5, r_m + 4 * g / 5, r_s] + nr(r_s, r_w, 6) + nr(r_w, R_o, 6)
    rings = ([Ring(0.0, 1)] + [Ring(r, max(16, n // 4) if r < 0.02 else n, rotating=True) for r in rot]
             + [Ring(r, n) for r in sta])

    def cls(ly, r, ph):
        return "rfe" if r < r_ri else "pm" if r < r_m else "gap" if r < r_s else "wind" if r < r_w else "sfe"

    mesh = polar_mesh(rings, cls, theta=theta, region_names=["rfe", "pm", "gap", "wind", "sfe"])
    c = mesh.centroids()
    phc = np.arctan2(c[:, 1], c[:, 0])
    nu = np.full(mesh.n_elements, NU0)
    nu[mesh.mask("rfe", "sfe")] = NU0 / mur_fe
    nu[mesh.mask("pm")] = NU0 / mur_m
    m = mesh.mask("pm")
    Br = np.zeros((mesh.n_elements, 2))
    amp = Br0 * np.cos(p * (phc - theta))
    Br[m, 0], Br[m, 1] = (amp * np.cos(phc))[m], (amp * np.sin(phc))[m]
    J = np.where(mesh.mask("wind"), J0 * np.cos(p * phc - phJ), 0.0)
    layers = [Layer(0, r_ri, mur_fe),
              Layer(r_ri, r_m, mur_m, Br_cos=Br0 * math.cos(p * theta), Br_sin=Br0 * math.sin(p * theta)),
              Layer(r_m, r_s), Layer(r_s, r_w, J_cos=J0 * math.cos(phJ), J_sin=J0 * math.sin(phJ)),
              Layer(r_w, R_o, mur_fe)]
    ref = RadialLayers(layers, p)
    return mesh, solve_magnetostatic(mesh, nu, J=J, Br=Br), ref, (r_m, g, r_s)


def test_slotless_spm_torque_and_gap_field_match_subdomain_solution():
    """torque_maxwell_stress( and co-energy torque against the exact solution (subdomain method) of a slotless machine."""
    n = 360
    h = 2 * math.pi / n
    theta = 0.3 * h  # rotor off the grid: the moving band is skewed
    mesh, sol, ref, (r_m, g, r_s) = _slotless(n, theta)
    T_ref = ref.torque(r_m + g / 2)
    assert ref.torque(r_m + 0.1 * g) == pytest.approx(T_ref, rel=1e-10)  # constant across the gap
    rc = np.hypot(*mesh.centroids().T)
    T1 = torque_maxwell_stress(mesh, sol.B, r_m, r_m + 2 * g / 5, elements=np.where((rc > r_m) & (rc < r_m + 2 * g / 5))[0])
    T2 = torque_maxwell_stress(mesh, sol.B, r_m + 3 * g / 5, r_s, elements=np.where((rc > r_m + 3 * g / 5) & (rc < r_s))[0])
    T_mst = 0.5 * (T1 + T2)
    W = [_slotless(n, theta + k * h)[1].coenergy for k in (-2, -1, 1, 2)]
    T_coe = (W[0] - 8 * W[1] + 8 * W[2] - W[3]) / (12 * h)
    assert T_mst == pytest.approx(T_ref, rel=1e-3)  # measured 4.6e-4
    assert T_coe == pytest.approx(T_ref, rel=1e-3)  # measured 1.9e-4
    gm = mesh.mask("gap")
    c = mesh.centroids()[gm]
    phc = np.arctan2(c[:, 1], c[:, 0])
    Br_fem = sol.B[gm, 0] * np.cos(phc) + sol.B[gm, 1] * np.sin(phc)
    Br_ref, _ = ref.B_polar(np.hypot(*c.T), phc)
    assert np.abs(Br_fem - Br_ref).max() / np.abs(Br_ref).max() < 0.02


# ----------------------------------------------------------------------------- slotted SPM machine
@pytest.fixture(scope="module")
def spm_linear():
    return SPMMachine(n_theta=360, iron=LinearMaterial(2000.0), slot_opening=2 / 15, slot_width=0.4)


def test_belt_winding_is_balanced():
    w = belt_winding(24, 2)
    for ph in range(3):
        sides = [s for p, s in w if p == ph]
        assert len(sides) == 8 and sum(sides) == 0
    assert w[:6] == [(0, 1), (0, 1), (2, -1), (2, -1), (1, 1), (1, 1)]


def test_maxwell_stress_and_coenergy_torques_agree(spm_linear):
    m = spm_linear
    th = math.radians(1.3)
    cur = m.phase_currents(th, 30.0, 0.0)
    T_mst = m.torque(m.solve(th, cur))
    T_coe = m.torque_coenergy(th, cur)
    assert T_mst > 10.0
    assert T_coe == pytest.approx(T_mst, rel=3e-3)  # measured 16.80 vs 16.82 N m (0.09 %)
    for deg in (2.7, 4.1):
        t = math.radians(deg)
        c_mst = m.torque(m.solve(t))
        c_coe = m.torque_coenergy(t, (0.0, 0.0, 0.0))
        assert abs(c_mst - c_coe) < 0.03 * 0.36  # measured |diff| <= 0.009 N m, cogging peak ~0.36 N m


def test_cogging_torque_is_periodic_and_zero_mean():
    """With an even number of grid steps per slot pitch the mesh repeats exactly every cogging period,
    so the computed cogging torque must repeat to solver precision."""
    m = SPMMachine(n_theta=480, iron=LinearMaterial(2000.0), slot_opening=0.2, slot_width=0.4)
    period = m.cogging_period()
    assert period == pytest.approx(math.radians(15.0))
    th = np.arange(0, 2 * period - 1e-12, period / 10)
    T = m.cogging(th)
    assert np.allclose(T[:10], T[10:], atol=1e-6 * np.abs(T).max())
    assert abs(T[:10].mean()) < 0.02 * np.abs(T).max()
    assert np.abs(T).max() > 0.1


def test_flux_linkage_balance_back_emf_and_power(spm_linear):
    """Linear iron: T(I) - T_cog = sum_k i_k dpsi_k/dtheta up to the small slotting reluctance torque."""
    m = spm_linear
    steps = 36
    th = (2 * math.pi / m.pole_pairs) * np.arange(steps) / steps
    psi = m.no_load_flux(th)
    # balanced winding: phase B is phase A delayed by 120 electrical degrees, phase C by 240
    assert np.allclose(psi[:, 1], np.roll(psi[:, 0], steps // 3), atol=1e-6 * np.abs(psi).max())
    assert np.allclose(psi[:, 2], np.roll(psi[:, 0], 2 * steps // 3), atol=1e-6 * np.abs(psi).max())
    rpm = 1500.0
    w = rpm * 2 * math.pi / 60
    e = back_emf(psi, th, w)
    k = 4
    cur = np.array(m.phase_currents(th[k], 5.0, 0.0))
    T = m.torque(m.solve(th[k], tuple(cur)))
    T_cog = m.torque(m.solve(th[k]))
    assert T - T_cog == pytest.approx(float(cur @ e[k]) / w, rel=0.03)


def test_back_emf_is_the_exact_derivative_of_a_band_limited_flux_linkage():
    n, p, w = 36, 2, 157.0
    th = (2 * math.pi / p) * np.arange(n) / n
    psi = 0.2 * np.cos(p * th + 0.3) + 0.03 * np.cos(3 * p * th - 1.0) + 0.01 * np.sin(5 * p * th)
    de = -0.2 * p * np.sin(p * th + 0.3) - 0.09 * p * np.sin(3 * p * th - 1.0) + 0.05 * p * np.cos(5 * p * th)
    assert np.allclose(back_emf(psi, th, w), w * de, atol=1e-12)


def test_torque_vs_current_angle_peaks_near_q_axis(spm_linear):
    m = spm_linear
    gam = np.radians(np.arange(-90, 91, 30))
    T = m.torque_vs_current_angle(30.0, gam, theta=math.radians(1.3))
    assert int(np.argmax(T)) == 3  # gamma = 0
    assert T[0] < 0.2 * T[3] and T[-1] < 0.2 * T[3]


def test_nonlinear_machine_torques_agree():
    m = SPMMachine(n_theta=360, slot_opening=2 / 15, slot_width=0.4)
    th = math.radians(1.3)
    cur = m.phase_currents(th, 30.0, 0.0)
    T_mst = m.torque(m.solve(th, cur))
    assert m.torque_coenergy(th, cur) == pytest.approx(T_mst, rel=3e-3)  # measured 16.564 vs 16.574


# ----------------------------------------------------------------------------- losses
def test_iron_loss_formula_on_alternating_and_rotating_fields():
    model = IronLossModel(k_h=0.02, alpha=1.0, beta=2.0, k_c=5e-5, k_e=1e-4, density=7650.0)
    n, f, B = 64, 50.0, 1.3
    t = np.arange(n) / n
    alt = np.zeros((n, 1, 2))
    alt[:, 0, 0] = B * np.sin(2 * math.pi * t)
    rot = np.zeros((n, 1, 2))
    rot[:, 0, 0] = B * np.cos(2 * math.pi * t)
    rot[:, 0, 1] = B * np.sin(2 * math.pi * t)
    area = np.array([2.0])
    p_alt, _ = iron_loss(alt, f, model, area, np.array([True]))
    p_rot, _ = iron_loss(rot, f, model, area, np.array([True]))
    expected = 2.0 * 7650.0 * model.specific_loss(f, B)
    assert p_alt == pytest.approx(expected, rel=1e-12)
    assert p_rot == pytest.approx(2 * expected, rel=1e-12)  # two orthogonal alternating components
    assert classical_eddy_coefficient(2e6, 0.35e-3, 7650.0) == pytest.approx(
        math.pi ** 2 * 2e6 * 0.35e-3 ** 2 / (6 * 7650.0))


def test_skin_effect_in_round_wire_matches_bessel_solution():
    """solve_time_harmonic( against the closed-form (Bessel) impedance of a round wire."""
    a, R, sigma = 0.005, 0.02, 5.8e7
    delta = a / 3
    omega = 2 / (MU0 * sigma * delta ** 2)
    n = 128
    radii = np.concatenate([np.linspace(a / (n // 4), a, n // 4), np.geomspace(a, R, n // 8)[1:]])
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii], lambda ly, r, p: "cu" if r < a else "air")
    m = mesh.mask("cu")
    sol = solve_time_harmonic(mesh, np.full(mesh.n_elements, NU0), np.full(mesh.n_elements, sigma),
                              omega, [m], [1.0])
    _, Z = skin_effect_wire(0.0, 1.0, a, sigma, omega)
    Z_total = Z + 1j * omega * MU0 / (2 * math.pi) * math.log(R / a)  # plus the external inductance
    assert sol.u[0].real == pytest.approx(Z_total.real, rel=3e-3)  # measured 7.5e-4
    assert sol.u[0].imag == pytest.approx(Z_total.imag, rel=3e-3)  # measured 8.6e-4
    R_dc = 1 / (sigma * sol.geometry.area[m].sum())
    assert 2 * sol.loss[0] / R_dc == pytest.approx(Z.real * sigma * math.pi * a * a, rel=3e-3)


def test_low_frequency_eddy_loss_in_conductor_in_transverse_field():
    a, R, sigma, f, B = 0.01, 0.05, 1e6, 50.0, 1.0
    omega = 2 * math.pi * f
    n = 128
    radii = np.concatenate([np.linspace(a / (n // 8), a, n // 8), np.linspace(a, R, n // 8)[1:]])
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii], lambda ly, r, p: "pm" if r < a else "air")
    m = mesh.mask("pm")
    y = mesh.points[:, 1]
    ref = round_conductor_eddy_loss_low_freq(B, omega, sigma, a)
    sol = solve_time_harmonic(mesh, np.full(mesh.n_elements, NU0), np.full(mesh.n_elements, sigma), omega,
                              [m], [0.0], dirichlet=mesh.boundary, dirichlet_values=(-1j * B * y)[mesh.boundary])
    assert sol.loss[0] == pytest.approx(ref, rel=3e-3)  # measured 8.4e-4 (delta / a = 7.1)
    t = np.arange(16) / 16 / f
    A_hist = np.array([B * np.sin(omega * tt) * y for tt in t])
    P = magnet_eddy_loss_resistance_limited(mesh, Geometry.of(mesh), A_hist, 1 / f, sigma, [m])
    assert P == pytest.approx(ref, rel=3e-3)  # measured 8.0e-4


def test_machine_losses_are_positive_and_scale_with_speed():
    m = SPMMachine(n_theta=240, iron=LinearMaterial(2000.0), slot_opening=2 / 15, slot_width=0.4)
    model = IronLossModel(k_h=0.02, k_c=5e-5, k_e=1e-4)
    lo = m.losses(1000.0, 10.0, 0.0, model, steps=12)
    hi = m.losses(2000.0, 10.0, 0.0, model, steps=12)
    for k in ("stator_iron_W", "rotor_iron_W", "magnet_eddy_W"):
        assert lo[k] > 0
    assert hi["magnet_eddy_W"] == pytest.approx(4 * lo["magnet_eddy_W"], rel=1e-9)  # resistance limited: f^2
    assert hi["stator_iron_W"] > 2 * lo["stator_iron_W"]
    assert lo["rotor_iron_W"] < lo["stator_iron_W"]


def test_rotor_elements_keep_identity_and_shape_when_the_rotor_turns():
    m = SPMMachine(n_theta=120, slots=12, magnet_thickness=0.0, rotor_core_inner=0.01, rotor_core_layers=4)
    m0 = m.mesh(0.0)
    rot = m.rotor_mask(m0)
    for t in (0.05, 0.3, 1.0, 4.0):
        mt = m.mesh(t)
        c, s = math.cos(t), math.sin(t)
        assert np.array_equal(m.rotor_mask(mt), rot)
        back = mt.centroids() @ np.array([[c, -s], [s, c]])  # rotate by -t
        assert np.abs(back[rot] - m0.centroids()[rot]).max() < 1e-12


def test_salient_rotor_design_maxwell_and_coenergy_torques_agree():
    m = SPMMachine(n_theta=240, slots=24, magnet_thickness=0.0, rotor_core_inner=0.01, rotor_core_layers=8,
                   iron=LinearMaterial(1000.0), airgap=0.0006)
    mesh = m.mesh(0.0)
    c = mesh.centroids()[m.design_elements(mesh)]
    phi = np.arctan2(c[:, 1], c[:, 0])
    x = np.where(np.cos(2 * m.pole_pairs * phi) > 0.0, 1.0, 1e-3)  # four salient poles
    th = math.radians(2.3)
    cur = m.phase_currents(th, 60.0, 0.0, phi0=math.radians(30.0))
    T_mst = m.torque(m.solve(th, cur, design=x))
    T_coe = m.torque_coenergy(th, cur, design=x)
    assert abs(T_mst) > 5.0
    # measured 0.65 % apart at n_theta=240 (0.13 % at 480): sharp iron corners next to a 0.6 mm gap
    assert T_coe == pytest.approx(T_mst, rel=1e-2)


# ----------------------------------------------------------------------------- adjoint
@pytest.mark.parametrize("iron", ["linear", "nonlinear"])
def test_rotor_torque_design_gradient_matches_finite_differences(iron):
    kw = {"iron": LinearMaterial(2000.0)} if iron == "linear" else {}
    m = SPMMachine(n_theta=120, slots=12, magnet_thickness=0.0, rotor_core_inner=0.01, rotor_core_layers=4, **kw)
    el = m.design_elements()
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.3, 0.9, len(el))
    th = 0.013
    cur = m.phase_currents(th, 40.0, 0.5, phi0=0.0)
    T, dT, _ = m.torque_and_design_gradient(th, cur, rho)
    big = np.argsort(-np.abs(dT))[:4]
    for i in big:
        h = 1e-5
        rp, rm = rho.copy(), rho.copy()
        rp[i] += h
        rm[i] -= h
        fd = (m.torque_and_design_gradient(th, cur, rp)[0] - m.torque_and_design_gradient(th, cur, rm)[0]) / (2 * h)
        assert dT[i] == pytest.approx(fd, rel=1e-5)


# ----------------------------------------------------------------------------- gmsh (optional)
def test_magnetized_cylinder_on_gmsh_mesh():
    gmsh = pytest.importorskip("gmsh")
    from pinneapple_simulation.electromagnetics import mesh_from_gmsh

    a, R, Br = 0.01, 0.06, 1.2
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cyl")
        inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
        outer = gmsh.model.occ.addDisk(0, 0, 0, R, R)
        out, _ = gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
        gmsh.model.occ.synchronize()
        surfs = [t for d, t in out if d == 2]
        mags = [s for s in surfs if abs(gmsh.model.occ.getMass(2, s) - math.pi * a * a) < 1e-9]
        airs = [s for s in surfs if s not in mags]
        gmsh.model.addPhysicalGroup(2, mags, name="pm")
        gmsh.model.addPhysicalGroup(2, airs, name="air")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.0015)
        gmsh.model.mesh.generate(2)
        mesh = mesh_from_gmsh()
    finally:
        gmsh.finalize()
    m = mesh.mask("pm")
    Brv = np.zeros((mesh.n_elements, 2))
    Brv[m] = (Br, 0.0)
    sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), Br=Brv)
    _, B_in = magnetized_cylinder(0.0, 0.0, Br, a, R)
    g = sol.geometry
    B_avg = (g.area[m, None] * sol.B[m]).sum(0) / g.area[m].sum()
    assert B_avg[0] == pytest.approx(B_in, rel=5e-3)
    assert abs(B_avg[1]) < 5e-3 * B_in
