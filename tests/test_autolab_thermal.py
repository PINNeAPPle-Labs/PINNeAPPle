"""Autonomous Physics Lab, demo 2 (board cooling): agent loop, mission parsing, coupled reference, surrogate, loop."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_autolab import LLMPlanner, Step, TemplatePlanner, Tool, Toolbox, execute
from pinneapple_design.thermal.board_cooling import (
    DEFAULT_TEXT,
    CoolingDesign,
    CoolingSurrogate,
    default_mission,
    holdout,
    parse_mission,
    simulate,
    simulate_many,
)
from pinneapple_design.thermal.board_cooling.verify import verify
from pinneapple_physics.pcb import Board, Component, Environment, solve

FAST = {"n_cells": 18, "hs_grid": 14}


# ---------------------------------------------------------------- agent loop
def _box():
    return Toolbox("t", [Tool("add", lambda st, a=1, b=2: {"sum": a + b}, "add", {"a": "x", "b": "y"}),
                         Tool("boom", lambda st: 1 / 0, "fails")])


def test_toolbox_rejects_unknown_tools_and_arguments():
    box = _box()
    assert box.check(Step("add", {"a": 1})) is None
    assert "unknown tool" in box.check(Step("rm", {}))
    assert "unknown argument" in box.check(Step("add", {"c": 3}))


def test_execute_logs_evidence_and_stops_on_error():
    box = _box()
    run = execute(TemplatePlanner([Step("add", {"a": 3}), Step("boom"), Step("add")]).plan("m", box), box)
    assert run.state["add"] == {"sum": 5} and not run.ok
    assert [e.tool for e in run.evidence] == ["add", "boom"]
    assert run.evidence[1].error.startswith("ZeroDivisionError")
    json.dumps(run.to_dict())


def test_llm_planner_validates_and_falls_back():
    box = _box()
    tpl = TemplatePlanner([Step("add")])
    good = LLMPlanner(tpl, llm=lambda p, s: json.dumps({"steps": [{"tool": "add", "args": {"a": 5}}]}),
                      required=("add",)).plan("m", box)
    assert good.planner.startswith("llm") and good.steps[0].args == {"a": 5}
    bad = LLMPlanner(tpl, llm=lambda p, s: json.dumps({"steps": [{"tool": "rm -rf", "args": {}}]})).plan("m", box)
    assert "fallback" in bad.planner and "unknown tool" in bad.notes[0]
    missing = LLMPlanner(tpl, llm=lambda p, s: json.dumps({"steps": [{"tool": "boom"}]}),
                         required=("add",)).plan("m", box)
    assert "missing required step" in missing.notes[0]
    offline = LLMPlanner(tpl, llm=lambda p, s: (_ for _ in ()).throw(ConnectionError("no network"))).plan("m", box)
    assert "ConnectionError" in offline.notes[0] and offline.steps[0].tool == "add"


# ---------------------------------------------------------------- mission
def test_parse_mission_keeps_the_words_of_each_value():
    m = parse_mission(DEFAULT_TEXT)
    assert (m.hot_component().power_w, m.tj_limit_c, m.t_ambient_c) == (40.0, 90.0, 35.0)
    assert (m.max_width_mm, m.max_depth_mm, m.max_height_mm, m.fan_power_budget_w) == (70, 70, 30, 0.02)
    assert all(v != "default" for v in m.provenance.values())
    e = parse_mission("A 45 W FPGA on a 200 x 120 mm board, junction below 100 °C, air at 40 °C, "
                      "heat sink up to 80 x 80 x 35 mm, fans 0.1 W")
    assert (e.hot_component().power_w, e.tj_limit_c, e.t_ambient_c, e.max_height_mm, e.fan_power_budget_w,
            e.board.width_mm) == (45.0, 100.0, 40.0, 35.0, 0.1, 200.0)
    assert parse_mission("cool it").provenance["power_w"] == "default"


def test_design_space_samples_are_feasible():
    sp = default_mission().space
    X = sp.sample(20, np.random.default_rng(1))
    assert len(X) == 20 and all(sp.feasible(x) for x in X)
    assert all(CoolingDesign.from_vector(x).fin_gap_mm >= 1.0 for x in X)


# ---------------------------------------------------------------- reference
def test_heat_sink_resistance_on_the_case_top_cools_the_junction():
    b, env = Board(80, 60), Environment(25.0, 1.0)
    bare = solve(b, [Component("U1", 40, 30, 5.0, "BGA 35x35")], env, n_cells=20)
    sink = solve(b, [Component("U1", 40, 30, 5.0, "BGA 35x35", r_top_k_w=1.0)], env, n_cells=20)
    assert sink["components"][0]["t_junction_c"] < bare["components"][0]["t_junction_c"] - 5
    assert sink["energy_balance_rel_error"] < 1e-8
    with pytest.raises(ValueError, match="r_top_k_w"):
        solve(b, [Component("U1", 40, 30, 5.0, r_top_k_w=-1.0)], env, n_cells=12)


def test_coupled_simulation_balances_energy_and_converges():
    m = default_mission()
    bare = simulate(m, None, fidelity=FAST, pessimistic=False)
    r = simulate(m, CoolingDesign(60, 60, 4, 16, 1.0, 25, 2.0), fidelity=FAST)
    assert r["coupling_converged"] and r["energy_balance_board"] < 1e-8
    assert r["tj_c"]["SoC"] < bare["tj_c"]["SoC"] - 100
    assert all(r["tj_pessimistic_c"][n] > r["tj_c"][n] for n in r["tj_c"])        # weaker cooling, hotter parts
    split = r["heat_split_w"]
    total = split["board_top_face"] + split["board_bottom_face"] + split["component_tops"] + split["chassis"]
    assert abs(total - sum(c.power_w for c in m.components)) < 1e-6 * total
    assert 0 < r["q_sink_w"] < 40 and r["map_c"].shape == (40, 64)


# ---------------------------------------------------------------- surrogate and verification
def test_surrogate_beats_the_nearest_snapshot_on_held_out_designs():
    pytest.importorskip("sklearn")                       # the Gaussian processes of ParametricPOD(regressor="gpr")
    m = default_mission()
    X = m.space.sample(28, np.random.default_rng(0))
    data = simulate_many(m, X, fidelity=FAST, workers=2)
    h = holdout(data, m.t_ambient_c, test_fraction=0.25)
    assert h["n_test"] == 7 and np.isfinite(h["surrogate"]["tj_mae_k"])
    assert h["surrogate"]["tj_mae_k"] < h["nearest_snapshot"]["tj_mae_k"]
    sur = CoolingSurrogate(m.t_ambient_c).fit(data["X"], data["Y"], data["maps"])
    p = sur.predict(X[:2])
    assert p["map"].shape == (2, 40, 64) and np.all(p["tj_std"] >= 0)
    assert np.allclose(p["tj"], data["Y"][:2], atol=0.5)                          # interpolates its snapshots


def test_verification_flags_a_design_that_misses_the_limit():
    m = default_mission()
    weak = CoolingDesign(36, 36, 2, 8, 0.8, 12, 0.5)                              # small sink, slow fans
    v = verify(m, weak, fidelity=FAST)
    assert v["tj_fine_pessimistic"]["SoC"] > m.tj_limit_c
    assert v["decision"] == "REJECT"
    assert any(c.name == "limit_tj_SoC" and c.status == "contradicts" and c.critical for c in v["checks"])


def test_flow_node_is_registered():
    from pinneapple_flow.registry import REGISTRY
    spec = REGISTRY.get("autolab.board_cooling")
    assert [p.name for p in spec.outputs] == ["report", "figure", "metrics"]
