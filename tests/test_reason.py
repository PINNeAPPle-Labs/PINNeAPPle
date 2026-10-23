"""pinneapple_reason: closed values of the failure modes, the engine reaching the expected chain, reports, and the
Flow / MCP / workbench integrations."""
import json
import math

import pytest

import pinneapple_reason as pr
from pinneapple_reason import LIBRARY, Concept, Requirement, cases, reason, tr
from pinneapple_reason.chain import Chain
from pinneapple_reason.library import Countermeasure, FailureMode, Library, iter_known_answers
from pinneapple_reason.modes._util import verdict
from pinneapple_reason.props import dew_point, saturation_pressure


# ------------------------------------------------------------------------------------------------ closed values
@pytest.mark.parametrize("mode,case", list(iter_known_answers()), ids=lambda x: getattr(x, "name", ""))
def test_known_answers(mode, case):
    v = mode.evaluate(case["params"])
    got = v.quantities[case["quantity"]][0]
    assert got == pytest.approx(case["expected"], rel=case["rtol"]), case["source"]


def test_every_mode_has_countermeasures_that_exist():
    for m in LIBRARY.modes.values():
        assert m.countermeasures, m.name
        for c in m.countermeasures:
            assert c in LIBRARY.countermeasures, (m.name, c)
    assert len(LIBRARY.domains()) >= 8


def test_psychrometrics_against_tables():
    # IAPWS-95 saturation pressures: 0.61121 kPa (0 °C), 2.3392 kPa (20 °C), 7.3849 kPa (40 °C); Magnus within 0.4 %
    for t, p in ((0.0, 611.21), (20.0, 2339.2), (40.0, 7384.9)):
        assert saturation_pressure(t) == pytest.approx(p, rel=4e-3)
    assert dew_point(20.0, 0.5) == pytest.approx(9.27, abs=0.1)
    assert dew_point(25.0, 1.0) == pytest.approx(25.0, abs=1e-9)


def test_capillary_rise_and_flat_roof_leak_numbers():
    reqs, c, _ = cases.roof()
    v = LIBRARY.mode("lap_water_ingress").evaluate(c)
    h = v.quantities
    cap = 2 * 0.0728 * math.cos(math.radians(20)) / (998.2 * 9.80665 * 1e-3)
    assert h["h_capillary"][0] == pytest.approx(cap, rel=1e-12)
    assert h["h_film"][0] == pytest.approx(150e-3 / 3600 * 600, rel=1e-12)     # flat roof: the storm ponds
    assert h["lap_rise"][0] == 0.0 and not v.passed


def test_minimum_slope_holds_the_head():
    from pinneapple_reason.modes.water import lap_heads, minimum_slope
    _, c, _ = cases.roof()
    a = minimum_slope(c.params)
    hd = lap_heads({**c.params, "slope_deg": a})
    assert hd["rise"] == pytest.approx(hd["h_cap"] + hd["h_wind"] + hd["h_film"], rel=1e-6)
    assert 10 < a < 25


def test_roof_network_energy_balance_and_ceiling_effect():
    from pinneapple_reason.modes.thermal import roof_network
    _, c, _ = cases.roof()
    p = {**c.params, "slope_deg": 22.0}
    for elems in ((), ("ceiling",)):
        for case in ("night", "day"):
            r = roof_network({**p, "_elements": elems}, case)
            assert abs(r["balance_residual"]) / r["balance_scale"] < 1e-8
    bare = roof_network({**p, "_elements": ()}, "night")
    ceil = roof_network({**p, "_elements": ("ceiling",)}, "night")
    assert bare["t_room_surface"] < dew_point(p["t_room_night"], p["rh_room_night"]) < ceil["t_room_surface"]
    assert bare["t_sky"] < bare["t_tile_top"] < bare["t_out"]                 # radiative cooling below the air


def test_moc_instantaneous_closure_is_joukowsky_and_slow_closure_is_gentler():
    from pinneapple_simulation.numerical_solvers.water_hammer_moc import (
        valve_closure_moc,
        wave_speed,
    )
    a = wave_speed(2.18e9, 998.2, 0.3, 0.006, 210e9, 0.91)
    kw = dict(length=1000.0, diameter=0.3, wave_speed=a, q0=0.1, h_reservoir=50.0, n_reaches=20)
    r = valve_closure_moc(friction_factor=0.0, t_close=0.0, **kw)
    assert r["max_head"] - r["valve_head_steady"] == pytest.approx(r["joukowsky"], rel=1e-9)
    assert r["min_head"] == pytest.approx(50.0 - r["joukowsky"], rel=1e-9)
    peaks = [valve_closure_moc(friction_factor=0.015, t_close=t, **kw)["max_head"] for t in (1.0, 5.0, 20.0)]
    assert peaks[0] > peaks[1] > peaks[2] > 50.0


def test_beam_frequency_and_lateral_buckling_numbers():
    from pinneapple_reason.modes.structural import chi, frequency
    _, c, _ = cases.footbridge()
    f = frequency(c.params)
    assert f["f1"] == pytest.approx(math.pi / (2 * 30 ** 2) * math.sqrt(f["EI"] / f["m"]), rel=1e-12)
    assert chi(0.2, 0.49) == pytest.approx(1.0)
    assert chi(1.0, 0.21) == pytest.approx(0.6656, abs=1e-4)                   # EN 1993-1-1 buckling curve a at 1.0
    assert chi(1.0, 0.34) == pytest.approx(0.5970, abs=1e-4)                   # curve b at 1.0
    v = LIBRARY.mode("pedestrian_resonance").evaluate(c)
    names = {ck.name: ck for ck in v.self_checks}
    assert names["solver vs closed form: Rayleigh frequency from FD deflection"].status == "supports"


# ------------------------------------------------------------------------------------------------ engine
EXPECTED = {
    "roof": [("lap_water_ingress", "increase_slope"), ("underside_condensation", "ceiling_ventilated_attic"),
             ("radiant_heat_gain", "radiant_barrier")],
    "footbridge": [("deflection", "deepen_girders"), ("lateral_torsional_buckling", "lateral_bracing"),
                   ("pedestrian_resonance", "tuned_mass_damper")],
    "pumped_main": [("pump_cavitation", "lower_pump"), ("water_hammer", "slow_valve_closure")],
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_cases_reach_the_expected_chain(name):
    ch = pr.run_case(name)
    assert ch.resolved and ch.final.passed
    assert [(m, c) for _, m, c in ch.path()] == EXPECTED[name]
    assert pr.trust_report(ch).decisao == "APPROVED"


def test_power_board_joule_heatsink_fan():
    reqs, c, title = cases.power_board()
    c = c.with_changes({"board_cells": 20})
    c.version = 0
    ch = reason(reqs, c, title)
    assert ch.resolved
    assert [x[2] for x in ch.path()] == ["heatsink_natural", "forced_air"]
    v0, v1 = ch.steps[0].verifications[0], ch.steps[1].verifications[0]
    assert v0.demand > v1.demand > 125.0 > ch.final.verifications[0].demand
    assert ch.steps[0].options[[o.countermeasure for o in ch.steps[0].options].index("forced_air")].feasible is False


def _toy_library():
    lib = Library()

    def ev(p, lim):
        x = p["x"]
        return verdict(demand=1.0, capacity=x, unit="1", demand_label=tr("d", "d"), capacity_label=tr("c", "c"),
                       mechanism=tr("m", "m"), method="toy")

    def ev2(p, lim):
        return verdict(demand=p.get("y", 0.0), capacity=1.0, unit="1", demand_label=tr("d", "d"),
                       capacity_label=tr("c", "c"), mechanism=tr("m", "m"), method="toy")
    lib.add_mode(FailureMode("toy", "toy", tr("t", "t"), tr("t", "t"), ev, countermeasures=("cheap", "dear", "half")))
    lib.add_mode(FailureMode("side", "toy", tr("s", "s"), tr("s", "s"), ev2, countermeasures=("undo",)))
    lib.add_countermeasure(Countermeasure("cheap", "a", tr("a", "a"), tr("a", "a"), ("toy",),
                                          lambda p, v: ({"x": 2.0, "y": 2.0}, (), ()), lambda b, a: (1.0, "u", "")))
    lib.add_countermeasure(Countermeasure("dear", "b", tr("b", "b"), tr("b", "b"), ("toy",),
                                          lambda p, v: ({"x": 2.0}, (), ()), lambda b, a: (5.0, "u", "")))
    lib.add_countermeasure(Countermeasure("half", "c", tr("c", "c"), tr("c", "c"), ("toy",),
                                          lambda p, v: ({"x": p["x"] + 0.4}, (), ()), lambda b, a: (0.1, "u", ""),
                                          repeatable=True))
    lib.add_countermeasure(Countermeasure("undo", "d", tr("d", "d"), tr("d", "d"), ("side",),
                                          lambda p, v: ({"y": 0.0}, (), ()), lambda b, a: (1.0, "u", "")))
    return lib


def test_engine_prefers_clean_fix_then_cost():
    lib = _toy_library()
    reqs = [Requirement("R1", tr("a", "a"), ["toy"]), Requirement("R2", tr("b", "b"), ["side"])]
    ch = reason(reqs, Concept("toy", {"x": 0.5}), "toy", library=lib)
    # "cheap" fixes toy but breaks side; "dear" fixes cleanly; "half" only improves -> dear
    assert ch.path()[0] == (0, "toy", "dear") and ch.resolved
    assert ch.steps[0].options[0].regressions == ["R2/side"]


def test_engine_goes_on_with_the_best_improvement_and_detects_no_solution():
    lib = _toy_library()
    lib.countermeasures.pop("dear")
    lib.countermeasures.pop("cheap")
    ch = reason([Requirement("R1", tr("a", "a"), ["toy"])], Concept("toy", {"x": 0.5}), "toy", library=lib)
    assert [c for _, _, c in ch.path()] == ["half", "half"] and ch.resolved     # 0.5 -> 0.9 -> 1.3
    lib.countermeasures.pop("half")
    ch = reason([Requirement("R1", tr("a", "a"), ["toy"])], Concept("toy", {"x": 0.5}), "toy", library=lib)
    assert ch.status == "unresolved" and ch.final.finding.mode == "toy"


def test_llm_advisor_only_adds_library_names():
    from pinneapple_reason.llm import LLMAdvisor

    class Fake:
        def extract(self, text, schema):
            assert "catalog" in schema
            return {"countermeasures": ["make_it_waterproof", "longer_lap", "ceiling_ventilated_attic"]}
    reqs, c, title = cases.roof()
    adv = LLMAdvisor(Fake())
    ch = reason(reqs, c, title, advisor=adv)
    by = {o.countermeasure: o.proposed_by for o in ch.steps[0].options}
    assert by["longer_lap"] == "library" and by["ceiling_ventilated_attic"] == "llm"
    assert "make_it_waterproof" not in by and adv.log[0]["kept"] == ["longer_lap", "ceiling_ventilated_attic"]
    assert ch.path()[0][2] == "increase_slope"                                # the physics still decides


# ------------------------------------------------------------------------------------------------ data and reports
def test_chain_round_trip_and_explain(tmp_path):
    ch = pr.run_case("pumped_main")
    p = ch.save(tmp_path / "c.json")
    back = Chain.load(p)
    assert back.to_dict() == json.loads(p.read_text())
    assert back.path() == ch.path()
    text = back.explain("pt")
    assert "Versão 0" in text and "NPSH" in text and "lower_pump" in text
    assert "Version 1" in back.explain("en")


def test_report_files_and_trust_card(tmp_path):
    ch = pr.run_case("roof")
    paths = pr.write_report(ch, tmp_path / "roof", lang="pt")
    for k in ("chain", "md", "html", "trust", "fig_timeline", "fig_margins", "fig_demand_capacity",
              "fig_roof_min_slope"):
        assert paths[k].exists() and paths[k].stat().st_size > 0, k
    html = paths["html"].read_text()
    assert html.count("data:image/png;base64,") >= 4 and "prefers-color-scheme: dark" in html
    md = paths["md"].read_text()
    assert "Linha do tempo" in md and "increase_slope" in md and "Cartão de confiança" in md
    trust = json.loads(paths["trust"].read_text())
    assert trust["report"]["decisao"] == "APPROVED"
    assert set(trust["card"]) == {"data_geometry", "model", "physics", "benchmark", "uncertainty", "decision"}
    assert trust["card"]["physics"]["status"] == "answered"


# ------------------------------------------------------------------------------------------------ integrations
def test_flow_nodes_match_library_and_run(tmp_path):
    import pinneapple_flow as pf
    from pinneapple_flow.nodes import reason as rn
    assert rn.MODES == sorted(LIBRARY.modes) and rn.COUNTERMEASURES == sorted(LIBRARY.countermeasures)
    for t in ("reason.requirement", "reason.concept", "reason.verify", "reason.failure_mode", "reason.countermeasure",
              "reason.chain", "reason.report"):
        assert t in pf.REGISTRY
    f = pf.Flow.load("pinneapple_flow/flows/reason_roof.json")
    r = pf.run(f, workdir=tmp_path)
    assert all(n["status"] == "done" for n in r.nodes.values()), {k: n.get("error") for k, n in r.nodes.items()}
    assert r.output("ch", "metrics")["status"] == "resolved"
    assert r.output("chk", "finding")["mode"] == "lap_water_ingress"
    assert r.output("cm", "metrics")["options"][r.output("cm", "metrics")["chosen"]]["countermeasure"] == \
        "increase_slope"


def test_mcp_and_workbench_tools(tmp_path):
    from pinneapple_llm.workbench.tools import TOOL_SPECS, ApprovalPolicy, Workspace
    from pinneapple_mcp import tools as T
    lib = T.reason_library("structures")
    assert {m["name"] for m in lib["modes"]} >= {"deflection", "lateral_torsional_buckling", "bending_strength"}
    d = T.reason_check("pump_cavitation", {"suction_lift": 5.0, "suction_d": 0.15, "suction_length": 12.0,
                                           "pump_q": 0.05, "npsh_required": 4.0, "altitude": 800.0,
                                           "suction_k": 3.0})
    assert d["passed"] is False and "NPSH" in d["mechanism_text"]
    assert {"reason_chain", "reason_library", "reason_check"} <= set(TOOL_SPECS)
    ws = Workspace(tmp_path, ApprovalPolicy("yolo"))
    out = ws.call("reason_chain", {"case": "pumped_main", "out": "rep", "lang": "en"})
    assert "slow_valve_closure" in out and (tmp_path / "rep" / "report.html").exists()
    assert ApprovalPolicy("ask").needs_approval("reason_chain", {})
