"""Wing planform design (#1181): the lattice against lifting-line theory and Munk's bound, the friction build-up, and
the closed loop meeting its limits with less drag than the baseline wing."""
from __future__ import annotations

import math

import pytest

from pinneapple_design.aero.wing_design import WingSpec, design_wing, evaluate, lifting_line


def test_lattice_tends_to_the_lifting_line_at_high_aspect_ratio_and_stays_below_munks_bound():
    s = WingSpec(span_max=None)
    gaps = []
    for ar in (6.0, 20.0, 40.0):
        x = {"aspect_ratio": ar, "taper": 1.0, "twist": 0.0}
        y = evaluate(s, x, nc=6, ns=40)
        ll = lifting_line(s, x)
        gaps.append(abs(y["CL_alpha"] - ll["CL_alpha"]) / ll["CL_alpha"])
        assert y["e"] <= 1.0 + 1e-3
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.015           # measured: 7.0 %, 2.0 %, 1.0 %


def test_taper_near_045_is_nearly_elliptic_and_washout_moves_the_stall_inboard():
    s = WingSpec(span_max=None)
    rect = evaluate(s, {"aspect_ratio": 8.0, "taper": 1.0, "twist": 0.0})
    tap = evaluate(s, {"aspect_ratio": 8.0, "taper": 0.45, "twist": 0.0})
    assert tap["e"] > rect["e"] and tap["e"] > 0.99
    tip = evaluate(s, {"aspect_ratio": 8.0, "taper": 0.3, "twist": 0.0})
    washed = evaluate(s, {"aspect_ratio": 8.0, "taper": 0.3, "twist": -5.0})
    assert washed["stall_station"] < tip["stall_station"]


def test_friction_build_up_follows_raymer():
    y = evaluate(WingSpec(), {"aspect_ratio": 7.5, "taper": 0.7, "twist": -3.0})
    p = y["profile"]
    assert p["Cf"] == pytest.approx(0.455 / math.log10(p["Re"]) ** 2.58 / (1 + 0.144 * p["mach"] ** 2) ** 0.65)
    assert 0.005 < y["CD0"] < 0.012 and p["t_c"] == pytest.approx(0.12, abs=0.002)


def test_closed_loop_beats_the_baseline_within_its_limits():
    r = design_wing(WingSpec(), maxiter=12, popsize=8)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert all(st[k] == "pass" for k in ("design.stall", "design.root_bending", "design.span", "model.converged",
                                         "model.munk", "model.lifting_line"))
    assert st["verify.cfd"] == "inconclusive"                        # stated, never claimed
    assert r["outputs"]["drag_N"] < r["baseline"]["drag_N"]
    assert r["recommendation"]["recommended"][0] == "none"           # a 30 ms lattice: no learned model


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "aero.wing_design" in REGISTRY.types()
    assert match("otimizar a asa do avião: alongamento e afilamento")[0][1].id == "aero.wing_design"
