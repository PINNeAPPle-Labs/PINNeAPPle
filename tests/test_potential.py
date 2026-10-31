"""pp.potential: derivative trees, Legendre transforms, Maxwell relations and consistency checks, against closed
forms (ideal gas, van der Waals, Neo-Hookean, double pendulum), CoolProp's Peng-Robinson backend (optional) and the
Maxwell-stress torque of the surface-PM machine FEA."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.potential import (
    consistency_check,
    convexity,
    hyperelastic,
    legendre,
    maxwell_relations,
    mechanics,
    thermo,
    tree,
)

R = thermo.R_GAS


# ----------------------------------------------------------------------------- thermodynamics
def test_ideal_gas_tree_matches_closed_forms():
    M = 0.039948  # argon, kg/mol
    A = thermo.ideal_gas(c=1.5, molar_mass=M)
    T = np.array([250.0, 300.0, 450.0])
    V = np.array([0.01, 0.0249, 0.08])
    t = tree(A, 2, at={"T": T, "V": V})
    assert t["p"] == pytest.approx(R * T / V, rel=1e-13)
    assert t["S"] == pytest.approx(1.5 * R * np.log(T / 298.15) + R * np.log(V / 0.024789570296023), rel=1e-12)
    assert t["cv"] == pytest.approx(1.5 * R, rel=1e-13)
    assert t["cp"] == pytest.approx(2.5 * R, rel=1e-13)
    assert t["kappa_T"] == pytest.approx(V / (R * T), rel=1e-13)
    assert t["alpha_p"] == pytest.approx(1 / T, rel=1e-13)
    assert t["w"] == pytest.approx(np.sqrt(5 / 3 * R * T / M), rel=1e-13)
    assert t["H"] == pytest.approx(t["U"] + t["p"] * V, rel=1e-13)
    assert max(t.symbolic_agreement().values()) < 1e-13  # autodiff and SymPy derivatives agree
    assert [n.symbol for n in t.level(1)] == ["S", "p"]


def test_maxwell_relation_text_and_self_consistency():
    A = thermo.ideal_gas()
    (rel,) = maxwell_relations(A)
    assert rel.text == "(dS/dV)_T = (dp/dT)_V" and rel.name == "Maxwell relation of the Helmholtz energy"
    rep = consistency_check(A, A, at={"T": np.linspace(250, 450, 7), "V": np.linspace(0.01, 0.05, 7)})
    assert rep.ok and rep.rows[0]["rel_p95"] < 1e-6  # finite differences of an exact potential


def test_gibbs_by_legendre_symbolic_and_numeric_agree_to_third_order():
    A = thermo.ideal_gas(c=2.5)
    Gs = thermo.gibbs(A)
    assert Gs.method == "symbolic"
    Gn = thermo.gibbs(A, method="numeric", bounds={"T": (200, 500), "V": (1e-4, 1.0)})
    assert Gn.method == "numeric" and Gn.convexity.ok
    pt = {"T": np.array([250.0, 300.0, 420.0]), "p": np.array([5e4, 1e5, 3e6])}
    ts, tn = tree(Gs.potential, 3, at=pt), tree(Gn.potential, 3, at=pt)
    G0 = np.abs(ts[()])
    for k, node in ts.nodes.items():
        dim = G0 / np.prod([pt[v] for v in k], axis=0) if k else G0  # magnitude scale |G| / prod(x)
        assert (np.abs(tn.nodes[k].raw - node.raw) <= 1e-9 * np.abs(node.raw) + 1e-12 * dim).all(), k
    assert ts["V"] == pytest.approx(R * pt["T"] / pt["p"], rel=1e-13)
    assert ts["cp"] == pytest.approx(3.5 * R, rel=1e-12)
    assert ts["kappa_T"] == pytest.approx(1 / pt["p"], rel=1e-12)
    assert Gn.residual(**pt).max() < 1e-6 * 3e6
    # G(T, p) = A(T, V) + p V at V = R T / p
    V = R * pt["T"] / pt["p"]
    assert ts[()] == pytest.approx(A(T=pt["T"], V=V) + pt["p"] * V, rel=1e-12)


def test_van_der_waals_properties_critical_point_and_stability_region():
    a, b = 0.3640, 4.267e-5  # CO2 van der Waals constants, Pa m6/mol2 and m3/mol (CRC Handbook)
    A = thermo.van_der_waals(a, b, c=2.5)
    cp_ = thermo.vdw_critical_point(a, b)
    T = np.array([280.0, 320.0, 400.0])
    V = np.array([2e-4, 1.5e-4, 4e-4])
    t = tree(A, 3, at={"T": T, "V": V})
    assert t["p"] == pytest.approx(R * T / (V - b) - a / V ** 2, rel=1e-12)
    assert t["dpdV_T"] == pytest.approx(-R * T / (V - b) ** 2 + 2 * a / V ** 3, rel=1e-11)
    assert t["cv"] == pytest.approx(2.5 * R, rel=1e-12)
    assert t["cp"] - t["cv"] == pytest.approx(R / (1 - 2 * a * (V - b) ** 2 / (R * T * V ** 3)), rel=1e-11)
    assert max(t.symbolic_agreement().values()) < 1e-12
    # critical point: (dp/dV)_T = (d2p/dV2)_T = 0, i.e. A_VV = A_VVV = 0
    tc = tree(A, 3, at={"T": cp_["T"], "V": cp_["V"]})
    assert abs(tc[("V", "V")]) < 1e-9 * R * cp_["T"] / cp_["V"] ** 2
    assert abs(tc[("V", "V", "V")]) < 1e-9 * R * cp_["T"] / cp_["V"] ** 3
    assert tc["p"] == pytest.approx(cp_["p"], rel=1e-12)
    # convexity in V fails exactly between the spinodals below T_c and nowhere above it
    rep = convexity(A, "V", {"T": (250.0, 330.0), "V": (6e-5, 6e-4)}, n_grid=161)
    assert not rep.ok
    dV = (6e-4 - 6e-5) / 160
    for fixed, runs in rep.intervals("V"):
        sp = thermo.vdw_spinodal(a, b, fixed["T"])
        if sp is None:
            assert runs == []
        else:
            assert len(runs) == 1
            assert runs[0][0] == pytest.approx(sp[0], abs=dV) and runs[0][1] == pytest.approx(sp[1], abs=dV)
    with pytest.warns(UserWarning, match="not convex"):
        thermo.gibbs(A, bounds={"T": (250.0, 330.0), "V": (6e-5, 6e-4)}, method="numeric")
    above = convexity(A, "V", {"T": (cp_["T"] * 1.01, 400.0), "V": (6e-5, 6e-4)}, n_grid=41)
    assert above.ok


def test_consistency_check_flags_properties_that_no_potential_produces():
    a, b = 0.3640, 4.267e-5
    A = thermo.van_der_waals(a, b, c=2.5)
    ig = thermo.ideal_gas(c=2.5)
    T = np.linspace(320, 420, 9)
    V = np.linspace(2e-4, 1e-3, 9)

    def molar_to_specific(t, M=1.0):
        return {"p": t["p"], "s": t["S"], "u": t["U"], "cv": t["cv"], "cp": t["cp"], "w": t["w"]}

    def consistent(pt):
        return molar_to_specific(tree(A, 2, at={"T": pt["T"], "V": pt["v"]}, symbolic=False))

    def mixed(pt):  # van der Waals pressure with ideal-gas entropy and energy: not from one potential
        out = consistent(pt)
        t = tree(ig, 2, at={"T": pt["T"], "V": pt["v"]}, symbolic=False)
        out["s"], out["u"] = t["S"], t["U"]
        return out

    good = consistency_check(consistent, "helmholtz_T_v", at={"T": T, "v": V}, tol=1e-4)
    assert good.ok, str(good)
    bad = consistency_check(mixed, "helmholtz_T_v", at={"T": T, "v": V}, tol=1e-4)
    failed = {r["name"] for r in bad.rows if not r["ok"]}
    assert failed == {"Maxwell relation (Helmholtz)", "energy equation"}
    # the same check on tabulated data (np.gradient on the grid)
    Tg, Vg = np.linspace(320, 420, 41), np.geomspace(2e-4, 1e-3, 41)
    TT, VV = np.meshgrid(Tg, Vg, indexing="ij")
    tab = consistent({"T": TT, "v": VV})
    grid = consistency_check({"axes": {"T": Tg, "v": Vg}, "values": tab}, "helmholtz_T_v", tol=1e-2)
    assert grid.ok, str(grid)
    bad_tab = mixed({"T": TT, "v": VV})
    bad_grid = consistency_check({"axes": {"T": Tg, "v": Vg}, "values": bad_tab}, "helmholtz_T_v", tol=1e-2)
    assert not bad_grid.ok


def test_peng_robinson_against_coolprop():
    CP = pytest.importorskip("CoolProp.CoolProp")
    AS = CP.AbstractState("PR", "CO2")
    M, Rg = AS.molar_mass(), AS.gas_constant()
    A = thermo.peng_robinson(AS.T_critical(), AS.p_critical(), AS.acentric_factor(), c=2.5, R=Rg, molar_mass=M)
    for T, rho in [(280.0, 50.0), (310.0, 300.0), (320.0, 700.0), (400.0, 150.0)]:
        AS.update(CP.DmassT_INPUTS, rho, T)
        t = tree(A, 2, at={"T": T, "V": M / rho})
        assert t["p"] == pytest.approx(AS.p(), rel=1e-11)
        assert t["dpdT_V"] == pytest.approx(AS.first_partial_deriv(CP.iP, CP.iT, CP.iDmolar), rel=1e-11)
        # residual heat capacity: the ideal-gas parts differ (constant c here, CoolProp's own c_p0(T) there)
        assert t["cv"] - 2.5 * Rg == pytest.approx(AS.cvmolar() - (AS.cp0molar() - Rg), rel=1e-10)


# ----------------------------------------------------------------------------- hyperelasticity
def _deformations(n=5, seed=0):
    rng = np.random.default_rng(seed)
    F = np.eye(3) + 0.25 * rng.standard_normal((n, 3, 3))
    F[np.linalg.det(F) < 0, 0] *= -1
    return F


def test_neo_hookean_stress_and_tangent_match_closed_forms():
    mu, lam = 0.8e6, 2.4e6
    W = hyperelastic.neo_hookean(mu, lam)
    F = _deformations()
    t = tree(W, 2, at={"F": F})
    P_ref = hyperelastic.neo_hookean_stress(F, mu, lam)
    A_ref = hyperelastic.neo_hookean_tangent(F, mu, lam)
    assert np.abs(t["P"] - P_ref).max() < 1e-9 * np.abs(P_ref).max()
    assert t["A"].shape == (5, 3, 3, 3, 3)
    assert np.abs(t["A"] - A_ref).max() < 1e-9 * np.abs(A_ref).max()
    assert hyperelastic.major_symmetry(t["A"]) < 1e-14
    J = np.linalg.det(F)
    assert np.allclose(t["sigma"], np.einsum("niJ,nkJ->nik", P_ref, F) / J[:, None, None], rtol=1e-9, atol=1e-3)
    t0 = tree(W, 2, at={"F": np.eye(3)})
    assert np.abs(t0["P"]).max() < 1e-9 and np.allclose(t0["A"], hyperelastic.isotropic_tangent(mu, lam), atol=1e-6)


def test_mooney_rivlin_small_strain_moduli():
    c10, c01, K = 0.3e6, 0.1e6, 5e6
    W = hyperelastic.mooney_rivlin(c10, c01, K)
    t = tree(W, 2, at={"F": np.eye(3)})
    mu = 2 * (c10 + c01)
    lam = K - 2 * mu / 3
    assert np.allclose(t["A"], hyperelastic.isotropic_tangent(mu, lam), rtol=1e-10, atol=1e-6)
    assert hyperelastic.major_symmetry(tree(W, 2, at={"F": _deformations(seed=1)})["A"]) < 1e-14


# ----------------------------------------------------------------------------- analytical mechanics
def _double_pendulum_reference(q1, q2, w1, w2, m1, m2, l1, l2, g):
    """Closed-form accelerations of the planar double pendulum (point masses, angles from the vertical)."""
    d = q1 - q2
    den = 2 * m1 + m2 - m2 * np.cos(2 * d)
    a1 = (-g * (2 * m1 + m2) * np.sin(q1) - m2 * g * np.sin(q1 - 2 * q2)
          - 2 * np.sin(d) * m2 * (w2 ** 2 * l2 + w1 ** 2 * l1 * np.cos(d))) / (l1 * den)
    a2 = 2 * np.sin(d) * (w1 ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(q1) + w2 ** 2 * l2 * m2 * np.cos(d)) / (
        l2 * den)
    return np.stack([a1, a2], axis=-1)


def test_double_pendulum_equations_energy_and_hamiltonian():
    pars = dict(m1=1.3, m2=0.7, l1=1.1, l2=0.9, g=9.81)
    L = mechanics.double_pendulum(**pars)
    rng = np.random.default_rng(3)
    pt = {"q1": rng.uniform(-2, 2, 6), "q2": rng.uniform(-2, 2, 6), "qd1": rng.uniform(-3, 3, 6),
          "qd2": rng.uniform(-3, 3, 6)}
    acc = mechanics.accelerations(L, pt)
    ref = _double_pendulum_reference(pt["q1"], pt["q2"], pt["qd1"], pt["qd2"], **pars)
    assert np.abs(acc - ref).max() < 1e-11 * np.abs(ref).max()
    t = tree(L, 2, at=pt)
    M = np.stack([[t[("qd1", "qd1")], t[("qd1", "qd2")]], [t[("qd1", "qd2")], t[("qd2", "qd2")]]]).transpose(2, 0, 1)
    assert (np.linalg.eigvalsh(M) > 0).all()  # L convex in the velocities: the Hamiltonian exists
    m1, m2, l1, l2, g = (pars[k] for k in ("m1", "m2", "l1", "l2", "g"))
    E_ref = ((m1 + m2) * l1 ** 2 * pt["qd1"] ** 2 / 2 + m2 * l2 ** 2 * pt["qd2"] ** 2 / 2
             + m2 * l1 * l2 * pt["qd1"] * pt["qd2"] * np.cos(pt["q1"] - pt["q2"])
             - (m1 + m2) * g * l1 * np.cos(pt["q1"]) - m2 * g * l2 * np.cos(pt["q2"]))
    assert t["E"] == pytest.approx(E_ref, rel=1e-12)
    for method in ("symbolic", "numeric"):
        H = mechanics.hamiltonian(L, method=method).potential
        pq = {"q1": pt["q1"], "q2": pt["q2"], "p1": t["p1"], "p2": t["p2"]}
        th = tree(H, 1, at=pq, symbolic=False)
        assert th[()] == pytest.approx(E_ref, rel=1e-10)
        assert th["qd1"] == pytest.approx(pt["qd1"], rel=1e-9, abs=1e-12)  # Hamilton: qd = dH/dp
    run = mechanics.simulate(L, [1.2, -0.4], [0.0, 0.5], t_end=5.0, dt=0.005)
    assert np.abs(run["E"] - run["E"][0]).max() < 1e-6 * abs(run["E"][0])  # measured 6.9e-8 relative


# ----------------------------------------------------------------------------- electric machines
def test_spm_torque_from_coenergy_potential_matches_maxwell_stress():
    from pinneapple_physics.potential.machines import machine_coenergy
    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    from pinneapple_simulation.electromagnetics.materials import LinearMaterial

    m = SPMMachine(n_theta=360, iron=LinearMaterial(2000.0), slot_opening=2 / 15, slot_width=0.4)
    for deg in (1.3, 2.7):
        th = math.radians(deg)
        cur = m.phase_currents(th, 30.0, 0.0)
        cm = machine_coenergy(m, th, cur)
        r = cm.at_operating_point()
        ms = m.solve(th, cur)
        T_mst = m.torque(ms)
        assert T_mst > 10.0
        assert r["torque"] == pytest.approx(T_mst, rel=3e-3)  # measured 16.854 vs 16.837 N m at 1.3 deg (0.10 %)
        # flux linkages from the same potential against the FEA winding integral (linear iron: exact)
        assert np.allclose(r["psi"], m.flux_linkage(ms), rtol=1e-8, atol=1e-10)
        assert np.allclose(r["L"], r["L"].T) and (np.linalg.eigvalsh(r["L"]) > 0).all()
    e = np.zeros(3)
    e[0] = 1.0
    c = np.array(cur)
    dpsi = (m.flux_linkage(m.solve(th, tuple(c + e))) - m.flux_linkage(m.solve(th, tuple(c - e)))) / 2.0
    assert np.allclose(r["L"][0], dpsi, rtol=1e-6)


def test_public_namespace():
    import pinneapple as pp

    assert pp.potential.tree is tree and pp.potential.legendre is legendre
