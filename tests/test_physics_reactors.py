"""Reactor models, sizing, safety and the closed-loop design (#1179), against closed forms and textbook results."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.reactors.kinetics import R_GAS, Feed, Network, Reaction
from pinneapple_physics.reactors.models import Bed, Cooling, cstr, cstr_startup, pfr
from pinneapple_physics.reactors.sizing import (
    cstrs_for,
    levenspiel,
    parametric_sensitivity,
    semenov_number,
    size_cstr,
    size_pfr,
)

FIRST = Network(("A", "B"), (Reaction({"A": -1, "B": 1}, A=0.1),), cp={"A": 75.0, "B": 75.0})
LIQ = Feed({"A": 1.0}, 300.0, "liquid", v=1e-3)          # c_A0 = 1000 mol/m^3, tau = V / 1e-3


def test_isothermal_first_and_second_order_match_the_closed_forms():
    tau = 20.0
    assert pfr(FIRST, LIQ, 0.02).conversion("A")[-1] == pytest.approx(1 - math.exp(-0.1 * tau), rel=1e-7)
    assert cstr(FIRST, LIQ, 0.02)[0].conversion("A", LIQ) == pytest.approx(0.1 * tau / (1 + 0.1 * tau), rel=1e-9)
    second = Network(("A", "B"), (Reaction({"A": -1, "B": 1}, A=1e-3, orders={"A": 2}),), cp={"A": 75, "B": 75})
    k_c0_tau = 1e-3 * 1000 * tau
    assert pfr(second, LIQ, 0.02).conversion("A")[-1] == pytest.approx(k_c0_tau / (1 + k_c0_tau), rel=1e-7)


def test_sizing_volumes_and_the_levenspiel_area():
    s = size_pfr(FIRST, LIQ, "A", 0.9)
    assert s["reached"] and s["V"] == pytest.approx(math.log(10) / 0.1 * 1e-3, rel=1e-7)
    c = size_cstr(FIRST, LIQ, "A", 0.9)
    assert c["V"] == pytest.approx(9 / 0.1 * 1e-3, rel=1e-6)                      # X / (k (1 - X)) * v
    lv = levenspiel(FIRST, LIQ, "A", s["V"], n=2001)
    assert lv["area"] == pytest.approx(s["V"], rel=1e-4)
    train = cstrs_for(FIRST, LIQ, "A", 0.9, n_max=3)
    n_tau = [t["V_total"] / 1e-3 for t in train]                                 # n ((1-X)^(-1/n) - 1) / k
    assert n_tau == pytest.approx([n * (0.1 ** (-1 / n) - 1) / 0.1 for n in (1, 2, 3)], rel=1e-6)
    assert n_tau[0] > n_tau[1] > n_tau[2] > math.log(10) / 0.1                    # approaching the PFR


def test_series_reactions_peak_at_the_closed_form_residence_time():
    k1, k2 = 0.2, 0.05
    net = Network(("A", "B", "C"), (Reaction({"A": -1, "B": 1}, A=k1), Reaction({"B": -1, "C": 1}, A=k2)),
                  cp={"A": 75, "B": 75, "C": 75})
    prof = pfr(net, LIQ, 0.06, n=6001)
    tau_peak = prof.V[int(prof.F[:, 1].argmax())] / 1e-3
    assert tau_peak == pytest.approx(math.log(k2 / k1) / (k2 - k1), abs=0.02)


EXO = Network(("A", "B"), (Reaction({"A": -1, "B": 1}, A=1e6, Ea=6e4, dH=-5e4),), cp={"A": 150.0, "B": 150.0})


def test_adiabatic_line_and_the_three_steady_states_of_a_cstr():
    prof = pfr(EXO, LIQ, 0.05)
    assert np.abs(prof.T - (300 + 5e4 / 150 * prof.conversion("A"))).max() < 1e-6
    states = cstr(EXO, LIQ, 0.01)
    assert [s.stable for s in states] == [True, False, True]
    assert states[0].T < states[1].T < states[2].T
    start = cstr_startup(EXO, LIQ, 0.01)                                         # cold start stays on the low state
    assert start["settled"] and start["T"] == pytest.approx(states[0].T, abs=1e-3)
    hot = cstr_startup(EXO, LIQ, 0.01, T_start=700.0)                            # a hot start ignites
    assert hot["T"] == pytest.approx(states[2].T, abs=1e-3)


def test_kirchhoff_heat_of_reaction_follows_dcp():
    net = Network(("A", "B"), (Reaction({"A": -1, "B": 1}, A=1.0, dH=-1e4, T_ref=300.0),), cp={"A": 100, "B": 80})
    assert net.dH_at(400.0)[0] == pytest.approx(-1e4 + (80 - 100) * 100.0)


def test_ergun_isothermal_pressure_profile_matches_its_closed_form():
    inert = Network(("N2",), (), cp={"N2": 29.1}, molar_mass={"N2": 0.028014})
    feed = Feed({"N2": 0.02}, 500.0, "gas", P=2e5)
    bed = Bed(diameter=0.0254, particle=4e-3, voidage=0.4, mu=2.6e-5)
    prof = pfr(inert, feed, bed.area * 3.0, None, bed)
    G = 0.02 * 0.028014 / bed.area
    rho0 = 2e5 * 0.028014 / (R_GAS * 500.0)
    beta0 = G / (rho0 * bed.particle) * 0.6 / 0.4 ** 3 * (150 * 0.6 * bed.mu / bed.particle + 1.75 * G)
    z = prof.V / bed.area
    assert prof.P == pytest.approx(2e5 * np.sqrt(1 - 2 * beta0 * z / 2e5), rel=1e-6)


def test_propylene_glycol_adiabatic_cstr_reproduces_foglers_example():
    from pinneapple_physics.reactors.presets import propylene_glycol_cstr
    c = propylene_glycol_cstr()
    (s,) = cstr(c["network"], c["feed"], c["V"], Cooling())
    assert s.conversion("PO", c["feed"]) == pytest.approx(0.85, abs=0.005)        # Fogler Ex. 12-3: X = 0.85
    assert s.T * 1.8 == pytest.approx(613.0, abs=1.0)                             # T = 613 R


def test_the_packed_bed_has_a_runaway_boundary_above_its_operation():
    from pinneapple_physics.reactors.presets import phthalic_anhydride_bed
    b = phthalic_anhydride_bed(0.011)
    ps = parametric_sensitivity(b["network"], b["feed"], b["V"], b["cooling"], Tc_range=(600.0, 660.0), n=31)
    assert ps["peak_inside_scan"] and 10.0 < ps["margin"] < 60.0
    assert ps["T_max"][-1] - ps["T_max"][0] > 100.0                              # the hot spot runs away past it
    assert semenov_number(b["network"], b["feed"], b["cooling"])["psi"] > 0


def test_closed_loop_design_meets_its_limits_and_says_which_model_it_used():
    from pinneapple_physics.reactors.design import ReactorSpec, design_reactor, to_json
    from pinneapple_physics.reactors.presets import phthalic_anhydride_bed
    b = phthalic_anhydride_bed(0.011)
    V0 = b["V"]
    spec = ReactorSpec(b["network"], b["feed"], "X", 0.7, {"V": (0.3 * V0, 3 * V0), "Tc": (610.0, 645.0)},
                      cooling=b["cooling"], desired="PA", T_limit=673.15, margin_min=5.0,
                      assumptions=b["assumptions"])
    for force in (False, True):
        r = to_json(design_reactor(spec, n_doe=24, maxiter=15, popsize=8, force_surrogate=force,
                                   sensitivity=not force))
        st = {c["code"]: c["status"] for c in r["checks"]}
        assert st["design.conversion"] == "pass" and st["design.hot_spot"] == "pass", r["checks"]
        assert st["model.converged"] == st["model.stoichiometry"] == st["model.levenspiel"] == "pass"
        assert r["outputs"]["X"] >= 0.7 - 1e-6 and r["outputs"]["T_max"] <= 673.15 + 1e-6
        assert r["cost"]["solve_seconds"] > 0
        if force:
            assert r["surrogate"]["family"].startswith("kpi_regressor")
            assert all(k.startswith("surrogate.held_out_error") or k in st for k in st)
        else:
            assert r["surrogate"] is None and st["safety.runaway_margin"] == "pass"
            assert r["recommendation"]["recommended"][0] in ("none", "kpi_regressor")


def test_the_flows_are_bundled_and_the_planner_proposes_them():
    import json
    from pathlib import Path

    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "reactor.design" in REGISTRY.types()
    root = Path(__file__).resolve().parents[1] / "pinneapple_flow" / "flows"
    for name in ("reactor_sizing_packed_bed", "reactor_sizing_cstr"):
        d = json.loads((root / f"{name}.json").read_text())
        assert d["nodes"][0]["type"] == "reactor.design"
    assert match("quero dimensionar um reator de leito fixo com risco de runaway")[0][1].id == "reactor.sizing"
