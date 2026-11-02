"""Core coverage for four closed-form modules (#1129, toward 1.0): pump NPSH, electromagnetics, heat exchangers and
pipe insulation. Every check is against something outside the function itself: a published table (IAPWS-IF97,
Incropera), an exact law (Hagen-Poiseuille, Ampere, Maxwell's equations, the energy balance of an exchanger) or an
algebraic identity (an inverse, a round trip).
"""
import math

import numpy as np
import pytest

from pinneapple_physics.closed_form import electromagnetics as EM
from pinneapple_physics.closed_form import heat_exchanger as HX
from pinneapple_physics.closed_form import pipe_insulation as PI
from pinneapple_physics.closed_form import pump_npsh as P

G = 9.80665


# ------------------------------------------------------------------------------------------------ pump NPSH
def test_water_properties_against_published_tables():
    assert P.iapws97_psat(26.85) == pytest.approx(3.53658941e3, rel=1e-8)      # IAPWS-IF97 verification, 300 K
    w = P.water_density(20.0)
    assert w["rho"] == pytest.approx(998.21, rel=1e-3)                          # IAPWS-95 reference, 20 C
    assert w["mu"] == pytest.approx(1.0016e-3, rel=1e-2)                        # linear interpolation in a table


def test_laminar_pipe_loss_is_hagen_poiseuille():
    rho, mu, d, L, q = 1000.0, 1.0, 0.05, 20.0, 1e-3                            # a viscous oil: Re ~ 25
    r = P.pipe_loss(q, d, L, 0.0, 0.0, rho, mu)
    assert r["re"] < 2300 and r["f"] == pytest.approx(64 / r["re"])
    assert r["h"] == pytest.approx(128 * mu * L * q / (math.pi * rho * G * d ** 4), rel=1e-12)   # exact


def test_quadratic_fit_and_duty_point():
    q = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    c = P.fit_quadratic(q, 50.0 + 10.0 * q - 2000.0 * q ** 2)
    assert np.allclose(c, [50.0, 10.0, -2000.0], atol=1e-9)                    # exact data, exact coefficients
    qd = P.duty_point(np.array([50.0, 0.0, -2000.0]), 20.0, lambda x: 1000.0 * x * x, q_max=1.0)
    assert qd == pytest.approx(0.1, rel=1e-10)                                  # 50 - 2000 q^2 = 20 + 1000 q^2
    with pytest.raises(ValueError):
        P.duty_point(np.array([10.0, 0.0, -1.0]), 20.0, lambda x: 0.0, q_max=1.0)


def test_npsh_available_from_its_terms():
    r = P.npsh_available(101325.0, 26.85, -3.0, 0.5)
    rho = P.water_density(26.85)["rho"]
    assert r["pv"] == pytest.approx(3.53658941e3, rel=1e-8)
    assert r["npsha"] == pytest.approx((101325.0 - 3536.58941) / (rho * G) - 3.0 - 0.5, rel=1e-9)
    assert r["h_surface"] - r["h_vap"] + -3.0 - 0.5 == pytest.approx(r["npsha"])


# ------------------------------------------------------------------------------------------------ electromagnetics
def test_wire_potential_obeys_amperes_law():
    a, J, nu = 0.01, 2e6, 1.0 / (4e-7 * math.pi)
    r = np.array([0.3, 0.7, 1.5, 3.0]) * a
    h = 1e-7
    B = -(EM.wire_az(r + h, 0 * r, a, J, nu) - EM.wire_az(r - h, 0 * r, a, J, nu)) / (2 * h)   # B_phi = -dA/dr
    current = np.minimum(r, a) ** 2 * math.pi * J
    assert np.allclose(B, current / (nu * 2 * math.pi * r), rtol=1e-6)          # mu0 I_enc / (2 pi r)
    assert EM.wire_az(a, 0.0, a, J, nu) == pytest.approx(0.0, abs=1e-12)        # gauge A = 0 at r = a


def test_tm_cavity_mode_satisfies_maxwell_and_the_walls():
    Lx, Ly, eps, mu = 2.0, 1.0, 2.0, 1.5
    x, y, t, h = 0.37, 0.61, 0.23, 1e-5

    def f(x, y, t):
        return EM.tm_cavity_mode(x, y, t, 2, 1, Lx, Ly, eps, mu)
    dEz_dt = (f(x, y, t + h)[0] - f(x, y, t - h)[0]) / (2 * h)
    dHy_dx = (f(x + h, y, t)[2] - f(x - h, y, t)[2]) / (2 * h)
    dHx_dy = (f(x, y + h, t)[1] - f(x, y - h, t)[1]) / (2 * h)
    assert eps * dEz_dt == pytest.approx(dHy_dx - dHx_dy, rel=1e-6)             # Ampere-Maxwell (z)
    dHx_dt = (f(x, y, t + h)[1] - f(x, y, t - h)[1]) / (2 * h)
    dEz_dy = (f(x, y + h, t)[0] - f(x, y - h, t)[0]) / (2 * h)
    assert mu * dHx_dt == pytest.approx(-dEz_dy, rel=1e-6)                      # Faraday (x)
    walls = np.array([[0.0, 0.3], [Lx, 0.3], [0.5, 0.0], [0.5, Ly]])
    assert np.allclose(f(walls[:, 0], walls[:, 1], 0.4)[0], 0.0, atol=1e-12)    # PEC: tangential E = 0


def test_skin_effect_decays_by_e_per_skin_depth():
    delta = 0.002
    A = EM.skin_effect_slab(np.array([0.0, delta]), delta, A0=3.0)
    assert A[0] == 3.0 and abs(A[1]) == pytest.approx(3.0 / math.e) and np.angle(A[1]) == pytest.approx(-1.0)
    x, h = 0.0013, 1e-7
    d2 = (EM.skin_effect_slab(x + h, delta) - 2 * EM.skin_effect_slab(x, delta) + EM.skin_effect_slab(x - h, delta)) / h ** 2
    assert d2 == pytest.approx(2j / delta ** 2 * EM.skin_effect_slab(x, delta), rel=1e-4)   # nu A'' = j w sigma A


def test_a_cylinder_of_vacuum_does_not_scatter():
    pytest.importorskip("scipy")
    x, y = np.array([-2.0, 0.1, 1.7]), np.array([0.5, -0.2, 2.0])
    ez = EM.cylinder_tm_total(x, y, 3.0, 0.5, 1.0 + 0.0j)
    assert np.allclose(ez, np.exp(-1j * 3.0 * x), atol=1e-10)                   # eps_r = 1: the plane wave itself


# ------------------------------------------------------------------------------------------------ heat exchangers
def test_ntu_from_effectiveness_inverts_the_effectiveness():
    assert HX.ntu_from_effectiveness(0.75, 1.0) == pytest.approx(0.75 / 0.25, rel=1e-9)   # counterflow, C_r = 1
    for arr in ("counterflow", "parallel", "crossflow"):
        eps = HX.effectiveness(1.3, 0.6, arr)
        assert HX.ntu_from_effectiveness(eps, 0.6, arr) == pytest.approx(1.3, rel=1e-9)
    with pytest.raises(ValueError):
        HX.ntu_from_effectiveness(0.9, 1.0, "parallel")                         # parallel flow tops out at 0.5


def test_lmtd_correction_closes_the_energy_balance_of_a_1_2_exchanger():
    """Q = U A F LMTD must equal eps C_min dT_max for the same 1-2 exchanger."""
    ntu, cr, ch = 1.5, 0.5, 1.0
    eps = HX.effectiveness(ntu, cr, "shell_tube")
    thi, tci = 100.0, 0.0
    q = eps * ch * (thi - tci)
    tho, tco = thi - q / ch, tci + q / (ch / cr)
    lmtd = ((thi - tco) - (tho - tci)) / math.log((thi - tco) / (tho - tci))
    F = HX.lmtd_correction_1_2(thi, tho, tci, tco)
    assert 0 < F < 1 and ntu * ch * F * lmtd == pytest.approx(q, rel=1e-9)


def test_gnielinski_regimes_and_dittus_boelter():
    lam = HX.gnielinski(1000.0, 5.0)
    assert lam == {"nu": 3.66, "f": 0.064, "regime": "laminar"}
    tur = HX.gnielinski(1e5, 0.7)
    assert tur["regime"] == "turbulent" and tur["f"] == pytest.approx((0.790 * math.log(1e5) - 1.64) ** -2)
    db = 0.023 * 1e5 ** 0.8 * 0.7 ** 0.4                                          # Dittus-Boelter, heating
    assert abs(tur["nu"] / db - 1) < 0.15                                         # the two correlations agree
    assert HX.gnielinski(2500.0, 0.7)["regime"] == "transitional"


def test_kern_shell_side_geometry():
    r = HX.kern_shell(10.0, 3.4e-4, 0.19, 2840.0, shell_id=0.5, baffle_spacing=0.2, tube_od=0.019, pitch=0.025,
                      layout="square")
    assert r["a_s"] == pytest.approx(0.5 * 0.006 * 0.2 / 0.025)
    assert r["de"] == pytest.approx(1.27 / 0.019 * (0.025 ** 2 - 0.785 * 0.019 ** 2), rel=5e-3)   # Kern 1.27/d (rounded)
    assert r["re"] == pytest.approx(r["G"] * r["de"] / 3.4e-4) and r["h"] == pytest.approx(r["nu"] * 0.19 / r["de"])
    tri = HX.kern_shell(10.0, 3.4e-4, 0.19, 2840.0, shell_id=0.5, baffle_spacing=0.2, tube_od=0.019, pitch=0.025)
    assert tri["de"] == pytest.approx(1.10 / 0.019 * (0.025 ** 2 - 0.917 * 0.019 ** 2), rel=2e-2)  # Sinnott: within 2 %


# ------------------------------------------------------------------------------------------------ pipe insulation
def test_air_properties_against_incropera():
    a = PI.air(26.85)                                                            # 300 K, Incropera table A.4
    assert a["nu"] == pytest.approx(15.89e-6, rel=1e-3) and a["k"] == pytest.approx(0.0263, rel=1e-3)
    assert a["pr"] == pytest.approx(0.707, rel=1e-3) and a["beta"] == pytest.approx(1 / 300.0)


def test_capital_recovery_factor():
    assert PI.crf(0.10, 10) == pytest.approx(0.162745, rel=1e-5)                # published annuity factor
    assert PI.crf(0.0, 8) == pytest.approx(1 / 8)


def test_surface_coefficient_radiation_and_regimes():
    s = PI.surface_coefficient(60.0, 20.0, 0.2, 0.9)
    T1, T2 = 333.15, 293.15
    assert s["h_r"] == pytest.approx(0.9 * 5.670374419e-8 * (T1 ** 4 - T2 ** 4) / 40.0, rel=1e-4)
    assert s["regime"] == "natural" and s["re"] == 0.0 and s["h"] == pytest.approx(s["h_r"] + s["h_cv"])
    assert PI.surface_coefficient(60.0, 20.0, 0.2, 0.9, wind=8.0)["regime"] == "forced"


def test_economic_thickness_minimises_the_annual_cost():
    r = PI.economic_thickness(0.1, 180.0, 20.0, [0.0, 0.02, 0.04, 0.06, 0.08, 0.12], k0=0.035, k1=1e-4, eps=0.9,
                              wind=2.0, hours=8000, energy_price=0.05, efficiency=0.9, cost_per_m3=2000.0,
                              cost_per_m2=30.0, rate=0.08, years=10, t_surface_max=50.0)
    rows = r["rows"]
    assert all(row["total"] == pytest.approx(row["energy_cost"] + row["annual_capital"]) for row in rows)
    assert r["economic"]["total"] == min(row["total"] for row in rows if row["thickness"] > 0)
    assert [row["q"] for row in rows] == sorted((row["q"] for row in rows), reverse=True)   # more insulation, less loss
    safe = r["thinnest_safe"]
    assert safe["t_surface"] <= 50.0 and all(row["t_surface"] > 50.0 for row in rows if row["thickness"] < safe["thickness"])
