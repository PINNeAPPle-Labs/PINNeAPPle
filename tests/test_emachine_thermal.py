"""Lumped thermal network and demagnetisation check (``machines.thermal``): closed forms of conduction and of a
first-order network, time-integration orders, energy balance, the seven-node machine model, and the reverse field in
the magnets of the FEA."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines.thermal import (
    NDFEB_HCJ_MIN_KAM,
    ThermalNetwork,
    cylinder_conductance,
    demagnetisation_margin,
    intrinsic_coercivity,
    machine_network,
)


def test_cylinder_conductance_is_the_limit_of_thin_shells():
    k, r1, r2, L = 28.0, 0.08, 0.13, 0.1
    G = cylinder_conductance(k, r1, r2, L)
    errs = []
    for n in (8, 16, 32):
        r = np.linspace(r1, r2, n + 1)
        rm = 0.5 * (r[1:] + r[:-1])
        R_series = np.sum(np.diff(r) / (2 * math.pi * rm * L * k))  # flat slabs at the mid radius
        errs.append(abs(1 / R_series - G) / G)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 1e-4 and np.all(np.abs(orders - 2) < 0.05)


def test_steady_chain_matches_series_resistances():
    net = ThermalNetwork.empty(["a", "b", "c"], [1.0, 1.0, 1.0])
    net.connect("a", "b", 2.0).connect("b", "c", 4.0).to_boundary("c", 8.0, 40.0)
    T = net.steady([10.0, 0.0, 6.0])
    # the 10 W of a cross a-b and b-c; all 16 W leave through the boundary of c
    Tc = 40.0 + 16.0 / 8.0
    Tb = Tc + 10.0 / 4.0
    Ta = Tb + 10.0 / 2.0
    assert T == pytest.approx([Ta, Tb, Tc], abs=1e-12)


def test_first_order_transient_orders_and_energy():
    C, G, P, Tb, T0 = 500.0, 2.0, 100.0, 20.0, 20.0
    net = ThermalNetwork.empty(["n"], [C]).to_boundary("n", G, Tb)
    tau = C / G
    exact = Tb + P / G * (1 - math.exp(-tau / tau))
    errs = {"euler": [], "trapezoidal": []}
    for m in errs:
        for dt in (tau / 10, tau / 20, tau / 40):
            t, T = net.transient([P], tau, dt, T0, method=m)
            errs[m].append(abs(T[-1, 0] - exact))
    o_e = math.log2(errs["euler"][1] / errs["euler"][2])
    o_t = math.log2(errs["trapezoidal"][1] / errs["trapezoidal"][2])
    assert abs(o_e - 1) < 0.1 and abs(o_t - 2) < 0.1
    # energy balance of the trapezoidal rule (exact for it): P t = C dT + G int (T - Tb) dt
    t, T = net.transient([P], 3 * tau, tau / 50, T0)
    dt = t[1] - t[0]
    out = G * np.sum(0.5 * (T[1:, 0] + T[:-1, 0]) - Tb) * dt
    assert P * t[-1] == pytest.approx(C * (T[-1, 0] - T0) + out, rel=1e-12)


def test_machine_network_is_physical():
    net = machine_network(prius_2004())
    P = {"yoke": 250.0, "teeth": 150.0, "winding": 600.0, "end_winding": 400.0, "rotor": 30.0, "magnets": 40.0}
    T = net.steady([P[n] for n in net.names])
    Tm = dict(zip(net.names, T, strict=True))
    assert min(T) > 65.0  # above the coolant everywhere
    assert Tm["winding"] > Tm["teeth"] > Tm["yoke"]
    K, b = net.matrix()
    # total heat to the coolant equals the total loss
    q_out = sum(G * (T[i] - Tb) for i, G, Tb in net.boundaries)
    assert q_out == pytest.approx(sum(P.values()), rel=1e-12)


def test_grade_coercivities():
    assert NDFEB_HCJ_MIN_KAM["SH"] == pytest.approx(1591.5, abs=0.1)  # 20 kOe
    assert intrinsic_coercivity(1.6e6, 120.0, -0.0055) == pytest.approx(1.6e6 * 0.45)


def test_demagnetisation_grows_with_negative_d_current():
    m = IPMMachine(prius_2004(), n_gap=288, magnet_temperature=150.0)
    Hcj = intrinsic_coercivity(NDFEB_HCJ_MIN_KAM["H"] * 1e3, 150.0)
    out = [demagnetisation_margin(m, m.solve(0.0, m.currents_dq(0.0, i_d, 0.0)), Hcj) for i_d in (0.0, -250.0, -600.0)]
    H = [o["H_rev_max"] for o in out]
    assert H[0] < H[1] < H[2]
    assert out[0]["demagnetised_fraction"] == 0.0 and out[0]["margin"] > 0


@pytest.mark.parametrize("name", ["Thermal network: first-order transient", "Thermal network: cylindrical conduction"])
def test_thermal_verification_case(name):
    from pinneapple_simulation.electromagnetics.machines import verification as mv

    r = mv.run([name], quick=True)[0]
    assert r.passed, r.row() + " " + r.detail


def test_demagnetisation_refinement_shrinks_the_magnet_elements():
    from pinneapple_simulation.electromagnetics.machines import prius_2004
    from pinneapple_simulation.electromagnetics.machines.thermal import demagnetisation_refinement

    out = demagnetisation_refinement(prius_2004(), 100.0, {"d": -250.0}, {"H": 6.0e5}, gradings=(0.2, 0.1), n_gap=192)
    assert out[1]["magnet_elements"] > out[0]["magnet_elements"]
    assert out[1]["magnet_element_size_m"] < out[0]["magnet_element_size_m"]
    for lv in out:
        r = lv["cases"]["d"]["H"]
        assert 0.0 <= r["demagnetised_fraction"] <= r["fraction_above_margin"] <= 1.0
        assert r["ratio_max"] > 0
