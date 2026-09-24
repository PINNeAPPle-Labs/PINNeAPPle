"""Tests for the ProblemSpec adapter, the pp.decide API, the decision tree and the
Arena decision mode (no network, no real LLM)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pinneapple_decision import (
    CallableExecutor,
    DecisionLevel,
    EvidenceStore,
    ExecutionResult,
    ModelSelector,
    PhysicsDecider,
    PhysicsDecisionEngine,
    adapt_problem,
    physics_ai_tree,
)
from pinneapple_decision import api

ROOT = Path(__file__).resolve().parents[1]

# The user's example, verbatim.
USER_AIRFOIL = {"name": "airfoil_flow", "reynolds": 1e5, "steady": True, "geometry": "naca0012",
                "target": "pressure_field"}


def _run_py(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------- adapter

def test_user_airfoil_example_excludes_fno_pino_with_reason():
    d = PhysicsDecider().decide(USER_AIRFOIL, ModelSelector.choice())
    for opt in ("fno", "pino"):
        assert opt in d.excluded and opt not in d.probabilities
        assert "must_support_geometry" in d.excluded[opt]
        assert "structured_grid" in d.excluded[opt] and "unstructured_mesh" in d.excluded[opt]
    facts = d.diagnostics["problem_facts"]
    assert facts["inferred"]["representation"]["value"] == "unstructured_mesh"
    assert "naca0012" in facts["inferred"]["representation"]["rule"]
    assert facts["given"]["time_dependent"] == {"value": False, "from": "steady"}
    assert facts["given"]["reynolds"] == 1e5
    assert facts["given"]["target_kind"]["value"] == "field"
    # nothing about data or an analytical solution was said: unknown, not filled in
    for key in ("n_simulations", "has_analytical_solution", "has_reference_data", "dimension"):
        assert facts["unknown"][key].startswith("unknown")
    assert any("not known for this problem" in r for r in d.rationale)


def test_unrecognized_and_unparseable_fields_are_unknown_not_invented():
    raw = {"name": "mystery case", "geometry": "blob_shape_17", "reynolds": "high",
           "turbulence_intensity": 0.05}
    ad = adapt_problem(raw)
    assert "representation" not in ad.problem and "reynolds" not in ad.problem
    assert ad.unknown["representation"].startswith("unknown") and "blob_shape_17" in ad.unknown["representation"]
    assert ad.unknown["reynolds"].startswith("unknown") and "could not parse" in ad.unknown["reynolds"]
    assert ad.unrecognized == ["turbulence_intensity"]
    assert ad.problem["turbulence_intensity"] == 0.05  # kept for the LLM, nothing derived from it
    assert ad.status("representation") == "unknown"

    d = PhysicsDecider().decide(raw, ModelSelector.choice())
    assert d.excluded == {}  # no geometry fact -> the geometry constraint has nothing to check
    assert "representation" in d.diagnostics["problem_facts"]["unknown"]
    assert d.diagnostics["problem_facts"]["unrecognized"] == ["turbulence_intensity"]


def test_synonyms_map_to_canonical_facts():
    ad = adapt_problem({"Re": "2e4", "transient": True, "mesh": "cartesian", "dim": "3D",
                        "target": ["drag", "lift"], "n_sims": 30})
    p = ad.problem
    assert p["reynolds"] == 2e4 and p["time_dependent"] is True
    assert p["representation"] == "structured_grid" and p["dimension"] == 3
    assert p["target_kind"] == "kpi" and p["n_simulations"] == 30
    assert ad.inferred["has_reference_data"] == {"value": True, "rule": "n_simulations=30"}


def test_real_problemspec_is_adapted():
    from pinneapple_problemdesign.schema import DataSpec, GeometrySpec, ProblemSpec

    spec = ProblemSpec(title="Airfoil surrogate", goal="Cl/Cd over angle of attack",
                       task_type="neural_operator", outputs=["lift coefficient", "drag coefficient"],
                       geometry=GeometrySpec(domain="NACA 0012 airfoil", aoa_sweep_deg=[0.0, 4.0, 8.0]),
                       data=DataSpec(sources=["200 OpenFOAM RANS runs"]))
    ad = adapt_problem(spec)
    assert ad.source == "ProblemSpec"
    assert ad.problem["representation"] == "unstructured_mesh"
    assert ad.problem["needs_parameter_generalization"] is True
    assert ad.problem["has_reference_data"] is True and ad.problem["target_kind"] == "kpi"
    assert ad.problem["is_inverse"] is False
    assert "n_simulations" in ad.unknown  # "200 runs" in free text is not parsed into a count
    d = PhysicsDecider().decide(spec, ModelSelector.choice())
    assert "fno" in d.excluded and "pino" in d.excluded


def test_adapter_is_idempotent_on_canonical_dicts():
    first = adapt_problem(USER_AIRFOIL).problem
    again = adapt_problem(first).problem
    assert {k: again[k] for k in first} == first


# ---------------------------------------------------------------- pp API

def test_pp_decide_is_lazy_and_does_not_import_torch():
    # torch is blocked in this subprocess: `import pinneapple` must still work, must not
    # import the decision layer, and pp.decide must work without torch.
    r = _run_py("""
        import sys, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import pinneapple as pp
        assert "pinneapple_decision" not in sys.modules, "import pinneapple loaded pinneapple_decision"
        d = pp.decide({"name": "airfoil_flow", "reynolds": 1e5, "steady": True, "geometry": "naca0012",
                       "target": "pressure_field"}, objective="minimize_prediction_error")
        assert d.choice == "model" and "fno" in d.excluded and "pino" in d.excluded, d
        assert "torch" not in sys.modules
        assert "pinneapple_analysis" not in sys.modules
        print("OK", d.selected)
    """)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "OK" in r.stdout


def test_decision_layer_never_imports_torch():
    r = _run_py("""
        import sys
        import pinneapple_decision as pd
        d = pd.decide({"name": "airfoil_flow", "reynolds": 1e5, "geometry": "naca0012"})
        pd.PhysicsDecisionEngine().decide({"reynolds": 100}, objective="next training experiment")
        assert "torch" not in sys.modules, [m for m in sys.modules if m.startswith("torch")][:5]
        print("OK")
    """)
    assert r.returncode == 0, r.stderr[-3000:]


def test_pp_decide_execute_verify_in_process():
    import pinneapple as pp

    def run(decision, problem):
        return ExecutionResult("ok", metrics={"residual": 1e-4, "rel_l2": 0.01})

    engine = pp.decision_engine(executor=CallableExecutor(run))
    try:
        d = pp.decide(USER_AIRFOIL, objective="minimize_prediction_error")
        assert d.choice == "model" and set(d.excluded) == {"fno", "pino"}
        result = pp.execute(d)
        v = pp.verify(result)
        assert v.passed and engine.store.find(d.id).verification is v
    finally:
        api.decision_engine(reset=True)


# ---------------------------------------------------------------- tree

def _passing(decision, problem):
    return ExecutionResult("ok", metrics={"residual": 1e-4, "rel_l2": 0.01})


def test_tree_follows_branch_by_fact():
    engine = PhysicsDecisionEngine(executor=CallableExecutor(_passing))
    problem = {"name": "airfoil surrogate", "geometry": "naca0012", "has_analytical_solution": False,
               "n_simulations": 200, "parametric": True}
    run = engine.run_tree(problem)
    assert run.outcome == "accepted" and run.experiments == 1
    steps = {s.node: s for s in run.path}
    assert steps["has_analytical_solution"].resolved_by == "fact" and steps["has_analytical_solution"].selected == "no"
    assert steps["has_reference_data"].resolved_by == "fact" and steps["has_reference_data"].selected == "yes"
    assert "physics_model" not in steps
    assert steps["surrogate_model"].resolved_by == "decider"
    assert run.plan["approach"] == "data_driven_surrogate" and run.plan["model"] in ("deeponet", "mesh_graph_net")
    assert run.decided_by_fact() == ["has_analytical_solution", "has_reference_data"]

    fact_ev = engine.store.find(steps["has_reference_data"].decision_id)
    assert fact_ev.decision.level is DecisionLevel.FACT and fact_ev.decision.backend == "fact:has_reference_data"
    assert fact_ev.decision.diagnostics["tree"]["fact_status"] == "inferred"  # from n_simulations=200
    model_ev = engine.store.find(steps["surrogate_model"].decision_id)
    assert "fno" in model_ev.decision.excluded  # constraints still apply inside the tree
    assert model_ev.verification is not None and model_ev.verification.passed


def test_tree_marks_when_it_needed_the_decider():
    engine = PhysicsDecisionEngine(executor=CallableExecutor(_passing))
    run = engine.run_tree(USER_AIRFOIL)
    assert run.decided_by_fact() == []
    assert run.decided_by_decider()[:2] == ["has_analytical_solution", "has_reference_data"]
    first = engine.store.find(run.path[0].decision_id).decision
    assert first.level is DecisionLevel.L0 and first.backend == "rules"
    tree_diag = first.diagnostics["tree"]
    assert tree_diag["resolved_by"] == "decider" and tree_diag["fact_status"] == "unknown"
    assert "unknown" in tree_diag["note"]
    assert first.selected == "no"  # stated hint: body-fitted geometry has no closed form
    assert any("decision under uncertainty" in r for r in first.rationale)


def test_tree_analytical_branch_runs_baseline_first():
    seen = []

    def run(decision, problem):
        seen.append(dict(problem["experiment"]))
        return ExecutionResult("ok", metrics={"residual": 1e-5, "rel_l2": 1e-3})

    engine = PhysicsDecisionEngine(executor=CallableExecutor(run))
    out = engine.run_tree({"geometry": "unit square", "has_analytical_solution": True})
    assert out.outcome == "accepted" and out.experiments == 1
    assert seen == [{"approach": "analytical_baseline"}]


def test_tree_goes_back_to_the_loop_when_validation_fails(tmp_path):
    def run(decision, problem):
        if problem["experiment"].get("training_strategy") == "adaptive_sampling":
            return ExecutionResult("ok", metrics={"residual": 5e-4, "rel_l2": 0.02})
        return ExecutionResult("ok", metrics={"residual": 3e-2, "rel_l2": 0.2}, diagnostics=["residual_localized"])

    store = EvidenceStore(tmp_path / "tree.json")
    engine = PhysicsDecisionEngine(executor=CallableExecutor(run), store=store)
    problem = {"description": "PINN #17, 2D lid-driven cavity", "reynolds": 100,
               "has_analytical_solution": False, "has_reference_data": False}
    out = engine.run_tree(problem, max_experiments=3)
    kinds = [(s.node, s.passed) for s in out.path if s.kind == "validate"]
    assert kinds == [("validate", False), ("validate", True)]
    assert [s.node for s in out.path].count("train") == 2
    nxt = next(s for s in out.path if s.node == "next_experiment")
    assert nxt.selected == "adaptive_sampling" and nxt.resolved_by == "decider"
    assert out.outcome == "accepted" and out.plan["training_strategy"] == "adaptive_sampling"

    saved = json.loads((tmp_path / "tree.json").read_text())
    nodes = [e["decision"]["diagnostics"]["tree"]["node"] for e in saved]
    assert nodes == ["has_analytical_solution", "has_reference_data", "physics_model", "next_experiment"]
    assert saved[2]["verification"]["passed"] is False and saved[3]["verification"]["passed"] is True

    engine2 = PhysicsDecisionEngine(executor=CallableExecutor(run))
    budget = engine2.run_tree(problem, max_experiments=1)
    assert budget.outcome == "budget_exhausted" and budget.experiments == 1
    assert budget.path[-1].node == "train" and "budget exhausted" in budget.path[-1].note


def test_default_tree_is_well_formed():
    tree = physics_ai_tree()
    assert tree.root == "has_analytical_solution"
    assert {"accept", "train", "validate", "next_experiment"} <= set(tree.nodes)


# ---------------------------------------------------------------- Arena decision mode

def _tiny_arena_cfg(**decision):
    from pinneapple_arena import ArenaConfig

    tr = {"epochs": 5, "lr": 1e-2, "scheduler": "none", "seed": 0}
    return ArenaConfig.from_dict({
        "problem": {"name": "poisson_2d", "grid_n": 10, "n_col": 64, "n_bc": 32,
                    "n_train_supervised": 64, "n_mesh_nodes": 40},
        "models": [
            {"name": "PINN", "type": "vanilla_pinn", "network": {"hidden": [8, 8]}, "training": tr},
            {"name": "DeepONet", "type": "deeponet", "network": {"hidden": [8, 8]}, "training": tr},
            {"name": "MGN", "type": "meshgraphnet", "network": {"hidden_dim": 8, "n_message_passing": 1},
             "training": {**tr, "epochs": 2}},
        ],
        "output": {"save_figures": False},
        "decision": decision,
    })


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_arena_decision_mode_respects_budget_and_records_verified_evidence(tmp_path):
    from pinneapple_arena import Arena

    arena = Arena(_tiny_arena_cfg(), device="cpu")
    report = arena.run_decision(budget=2, evidence_path=str(tmp_path / "arena_evidence.json"))

    assert len(report.ran) == 2 and report.budget == 2
    assert report.stop_reason == "budget_exhausted" and report.accepted is None
    assert len(report.not_trained) == 1
    assert [t.name for t in arena.train_results] == report.ran  # only the chosen models were trained
    assert report.ran[0] == "PINN"  # rules: no simulation data known -> physics-residual PINN first
    for ev in report.evidence:
        assert ev.verification is not None and ev.verification.checks  # verified, with checks that ran
        assert ev.decision.level is DecisionLevel.L0 and ev.decision.calibrated_confidence is None
    # the second decision could not re-pick the first model
    assert "not_run_before" in report.evidence[1].decision.excluded["PINN"]
    # a supervised model reports no PDE residual: that check did not run, it is not a pass
    deeponet = next(ev for ev in report.evidence if ev.decision.selected == "DeepONet")
    assert "residual" not in deeponet.verification.checks
    facts = report.evidence[0].decision.diagnostics["problem_facts"]
    assert facts["given"]["has_analytical_solution"] is True

    saved = json.loads((tmp_path / "arena_evidence.json").read_text())
    assert [e["decision"]["selected"] for e in saved] == report.ran
    assert all(e["verification"] is not None for e in saved)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_arena_decision_mode_from_config_stops_when_verified():
    from pinneapple_arena import Arena

    arena = Arena(_tiny_arena_cfg(enabled=True, budget=3, thresholds={"rel_l2": 100.0}), device="cpu")
    arena.run()
    report = arena.decision_report
    assert report.stop_reason == "verified" and len(report.ran) == 1
    assert report.accepted == report.ran[0] and len(arena.train_results) == 1


def test_arena_catalog_family_mapping():
    from pinneapple_arena.decision_mode import catalog_family

    assert catalog_family("vanilla_pinn") == "pinn" and catalog_family("siren") == "pinn"
    assert catalog_family("fno2d") == "fno" and catalog_family("meshgraphnet") == "mesh_graph_net"
    assert catalog_family("inverse_pinn") == "inverse_pinn" and catalog_family("lstm") is None
