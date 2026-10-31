"""pinneapple_autolab.agent: mission reading, the closed toolbox, the step planners, the claim audit, replay."""
import json

import numpy as np
import pytest

from pinneapple_autolab import AutonomousLab, FunctionSimulator, Step, act
from pinneapple_autolab.agent import (
    LabPlanner,
    MissionDomain,
    MissionRefused,
    audit_claims,
    interpret_mission,
    lab_toolbox,
    run_agent,
)
from pinneapple_autolab.replay import ReplaySimulator
from pinneapple_design.design_optimizer.study import Variable


def toy(p):
    a, b = p["a"], p["b"]
    return {"drag": 1 + (a - 0.3) ** 2 + 0.5 * b ** 2 + 0.2 * a * b, "down": 2 + a + 0.8 * b - 0.3 * a ** 2}


DOM = MissionDomain("toy", [Variable("a", 0, 1), Variable("b", 0, 1)], {"a": 0.5, "b": 0.5},
                    {"drag": ["drag"], "down": ["downforce"]}, {"drag": "N", "down": "N"},
                    {"speed": (70.0, "m/s", r"(\d+(?:\.\d+)?)\s*m/s")}, {"drag": 0.03, "down": 0.03},
                    {"speed_m_s": 70.0}, default_budget=12, default_reserve=1)
TEXT = "Find a lower-drag design at 70 m/s. Preserve the required downforce and validate it. 12 CFD runs."
LAB = {"families": ("gp", "quadratic"), "nsga_budget": 200, "n_pool": 100}


def sim():
    return FunctionSimulator(toy, ["drag", "down"])


def test_interpret_reads_objective_limit_budget_and_refuses_other_conditions():
    r = interpret_mission(TEXT, DOM)
    m = r.mission
    assert [o.name for o in m.objectives] == ["drag"] and m.objectives[0].sense == "min"
    assert [(c.name, c.lower_frac_of_baseline) for c in m.constraints] == [("down", 1.0)]
    assert m.budget.max_runs == 12 and any("budget = 12" in x["what"] for x in r.readings)
    with pytest.raises(MissionRefused, match="80 m/s"):
        interpret_mission("Lower drag at 80 m/s.", DOM)
    with pytest.raises(MissionRefused, match="no objective"):
        interpret_mission("Make it nice.", DOM)


def test_toolbox_refuses_what_it_does_not_declare(tmp_path):
    r = interpret_mission(TEXT, DOM)
    lab = AutonomousLab(r.mission, sim(), str(tmp_path), **LAB)
    box = lab_toolbox(lab, r)
    steps = iter([Step("simulate_baseline", {"speed": 80}), Step("set_boundary_conditions", {"v": 90}), None])
    run = act(TEXT, box, type("P", (), {"name": "script", "next": lambda self, *a: next(steps)})())
    assert [e.ok for e in run.evidence] == [False, False] and lab.ledger.used_runs == 0
    assert "unknown argument(s) ['speed']" in run.evidence[0].error and "unknown tool" in run.evidence[1].error


def test_deterministic_agent_runs_the_mission_and_its_answer_is_supported(tmp_path):
    res = run_agent(TEXT, sim(), str(tmp_path), domain=DOM, n_doe=6, lab_options=LAB)
    assert res.refused is None and res.audit["supported"], res.audit
    tools = [s["tool"] for s in res.steps]
    assert tools[:4] == ["read_mission", "plan", "simulate_baseline", "space_filling"]
    assert "simulate_candidate" in tools and "verify_best" in tools and tools[-1] == "answer"
    assert res.summary["budget"]["used_runs"] == 12 and res.audit["recommendation_ok"]
    assert res.audit["recommended_run"] == res.summary["best"]["id"]


def test_a_rogue_llm_is_refused_and_its_invented_number_flagged(tmp_path):
    script = iter([
        {"tool": "simulate_baseline", "args": {"speed": 90}},                       # refused
        {"tool": "change_speed", "args": {"v": 90}},                                # no such tool
        {"tool": "plan", "args": {}},
        {"tool": "simulate_baseline", "args": {}},
        {"tool": "answer", "args": {"text": "Drag is now 0.123 N, a 40 % cut.", "recommended_run": "run001-baseline"}},
        {"done": True},
    ])
    res = run_agent(TEXT, sim(), str(tmp_path), domain=DOM, llm=lambda prompt, system: json.dumps(next(script)),
                    lab_options=LAB)
    errs = [s["error"] for s in res.steps if s["error"]]
    assert any("unknown argument(s) ['speed']" in e for e in errs) and any("unknown tool 'change_speed'" in e for e in errs)
    assert not res.audit["supported"] and set(res.audit["unsupported_numbers"]) == {"0.123", "40"}
    assert res.audit["recommendation_ok"] is True          # the baseline itself meets "down >= baseline"


def test_audit_accepts_rounded_values():
    class L:
        class ledger:
            @staticmethod
            def to_dict():
                return {"runs": []}
    a = audit_claims("drag 3287.4 N (3287 N)", [{"tool": "status", "result": {"drag": 3287.4417}}], L, None)
    assert a["supported"] and np.isfinite(a["n_numbers_checked"])


def test_replay_answers_only_exact_designs_and_labels_them(tmp_path):
    lab = AutonomousLab(interpret_mission(TEXT, DOM).mission, sim(), str(tmp_path / "a"), **LAB)
    lab.plan(4)
    lab.baseline()
    lab.doe(2)
    calls = []
    real = FunctionSimulator(lambda p: calls.append(p) or toy(p), ["drag", "down"])
    rs = ReplaySimulator(real, lab.ledger.save())
    hit = rs.run(lab.ledger.runs[1].params, str(tmp_path / "x"))
    miss = rs.run({"a": 0.123, "b": 0.456}, str(tmp_path / "y"))
    assert hit.meta["replayed_from"]["run"] == lab.ledger.runs[1].id and hit.outputs == lab.ledger.runs[1].outputs
    assert not calls[:-1] and len(calls) == 1 and miss.outputs["drag"] == pytest.approx(toy({"a": 0.123, "b": 0.456})["drag"])
    assert LabPlanner.name == "deterministic"


def test_audit_ignores_run_provenance():
    """A number in a run's start time or wall time is not a fact: "40" is unsupported even when a run started at
    minute 40 and took 40 s."""
    from types import SimpleNamespace

    from pinneapple_autolab.agent import audit_claims
    run = {"id": "run001-baseline", "params": {"a": 1.5}, "outputs": {"drag": 0.25}, "predicted": {},
           "started": "2026-10-30T11:40:40", "cost_s": 40.0, "message": "took 40 s", "meta": {"wall": 40}}
    lab = SimpleNamespace(ledger=SimpleNamespace(to_dict=lambda: {"runs": [run]}, runs=[]))
    a = audit_claims("Drag is 0.25 at a = 1.5, a 40 % cut.", [], lab)
    assert a["unsupported_numbers"] == ["40"] and not a["supported"]
