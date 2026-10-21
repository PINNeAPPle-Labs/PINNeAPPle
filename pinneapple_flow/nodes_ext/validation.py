"""Validation nodes for ``examples/physics_check``, ``examples/vs_physicsnemo`` (07, the check replay),
``examples/model_zoo``, ``examples/proof_graph`` and ``examples/validation_arena``: the physics linter
(:func:`pinneapple_check.run`) against the rules each fixture must break, the zoo's evaluate / validate / fine-tune,
the proof graph of the PT-chain no-go question, and the red team / blue team arena."""
from __future__ import annotations

import json

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from .examples import example_dir


def _example_path(spec: str):
    ex, _, rel = spec.partition("/")
    root = example_dir(ex)
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise StepFailed(f"{spec!r} leaves the example folder")
    if not p.exists():
        raise StepFailed(f"{spec!r} does not exist")
    return p


@node("check.physics", "Physics checks of results and models (pinneapple check)", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("path", "str", "physics_check/fail", help="<example>/<folder> holding a pinneapple-check.toml"),
              Param("expected", "str", "expected.json",
                    help="file in that folder listing the rules each target must break (empty: none expected)")],
      calls="pinneapple_check.engine.run (Report.violations, exit_code)")
def physics_check(ctx, inputs, params):
    """Run every rule on the declared targets (OpenFOAM cases, fields, QoI tables, surrogates, models) and compare
    the rules each target breaks with the example's list (examples/physics_check, the CI gate's fixtures)."""
    from pinneapple_check.engine import run
    folder = _example_path(params["path"])
    rep = run([folder])
    got: dict[str, set] = {}
    for o in rep.violations:
        got.setdefault(str(o.target), set()).add(o.rule)
    ctx.metric("targets", len(rep.targets))
    ctx.metric("violations", len(rep.violations))
    ctx.metric("exit_code_warning", int(rep.exit_code("warning")))
    exp = {}
    if params["expected"]:
        f = folder / params["expected"]
        exp = json.loads(f.read_text()) if f.is_file() else {}
    keys = sorted(set(exp) | set(got))
    rows = [[k, sorted(got.get(k, [])), sorted(exp.get(k, [])), sorted(got.get(k, [])) == sorted(exp.get(k, []))]
            for k in keys]
    ctx.table("rules broken per target", ["target", "this run", "expected", "same"], rows)
    ctx.metric("targets_with_violations", len(got))
    ctx.metric("targets_expected_to_fail", len(exp))
    ctx.metric("targets_matching_expected", sum(r[3] for r in rows))
    ctx.metric("targets_compared", len(rows))
    ctx.summary(f"{len(rep.violations)} violations on {len(rep.targets)} targets; {sum(r[3] for r in rows)} of "
                f"{len(rows)} targets break exactly the expected rules")
    return {"metrics": ctx.report["metrics"]}


@node("zoo.evaluate", "Model zoo: evaluate, validate, fine-tune", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("name", "str", "airfoil_cst_rans_mlp", help="pp.zoo.list()"),
              Param("validate_inputs", "dict", {"T0": 0.169591, "T1": 0.144959, "T2": 0.131433, "M0": 0.024922,
                                                "M1": 0.071088, "M2": 0.05896, "alpha_deg": [4.0, 18.0]},
                    help="inputs for pp.zoo.validate (here NACA 2412 at 4 and 18 deg)"),
              Param("finetune_split", "str", "naca_reference", help="dataset split for the fine-tune (empty: skip)"),
              Param("epochs", "int", 50, minimum=1), Param("lr", "float", 1e-5)],
      calls="pp.zoo.info / load / evaluate / validate / finetune, pp.datasets.load (pinneapple_hub.zoo)")
def zoo_evaluate(ctx, inputs, params):
    """examples/model_zoo/quickstart.py: the card's held-out metrics reproduced, the physics checks and validity
    range, the comparison with XFOIL, and a guarded fine-tune."""
    import pinneapple as pp
    name = params["name"]
    ev = pp.zoo.evaluate(name)
    ctx.metric("matches_card", int(bool(ev["matches_card"])), "the card's held-out metrics reproduced")
    for k, v in ev["metrics"].items():
        if isinstance(v, (int, float)):
            ctx.metric(f"eval.{k}", float(v))
    model = pp.zoo.load(name)
    try:
        xf = pp.datasets.load("airfoil_rans_cst_re4e6", split="xfoil_naca2412")
        p = model.predict({k: xf[k] for k in model.input_names})
        ctx.metric("xfoil_max_abs_dCl", float(np.max(np.abs(p["Cl"] - xf["Cl"]))), "against XFOIL 6.99")
    except Exception as exc:  # noqa: BLE001 - only the airfoil card has this split
        ctx.warn(f"no XFOIL comparison: {exc}")
    rep = pp.zoo.validate(name, inputs=params["validate_inputs"])
    ctx.metric("validate_decision", rep.decisao)
    if params["finetune_split"]:
        ft = pp.zoo.finetune(name, pp.datasets.load("airfoil_rans_cst_re4e6", split=params["finetune_split"]),
                             epochs=int(params["epochs"]), lr=float(params["lr"]))
        ctx.metric("finetune_accepted", int(bool(ft.accepted)))
    ctx.summary(f"card metrics reproduced: {ev['matches_card']}; validate: {rep.decisao}")
    return {"metrics": ctx.report["metrics"]}


@node("proof.pt_chain", "Proof graph: the PT-chain no-go question", category="Validation",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("max_deg", "int", 4, minimum=1, maximum=8), Param("n_numeric", "int", 300, minimum=10),
              Param("search_iters", "int", 40, minimum=0)],
      calls="pinneapple_analysis.proof.pt_chains.build_pt_chain_graph + ProofGraph.run / report / plot")
def pt_chain(ctx, inputs, params):
    """A graph of assumptions, lemmas and a theorem with exact (sympy) and numerical checks: does the no-go theorem
    for linear exceptional-point tuning extend to PT tetramers (examples/proof_graph/pt_chain_no_go.py)?"""
    from pinneapple_analysis.proof.pt_chains import build_pt_chain_graph
    g = build_pt_chain_graph(max_deg=int(params["max_deg"]), n_numeric=int(params["n_numeric"]),
                             search_iters=int(params["search_iters"]))
    status = g.run()
    counts: dict[str, int] = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    for s, n in sorted(counts.items()):
        ctx.metric(f"status.{s}", n)
    ctx.table("nodes", ["node", "status"], [[k, v] for k, v in status.items()])
    (ctx.path("report.md")).write_text(g.report())
    p = g.plot(str(ctx.path("pt_chain_no_go.png")))
    ctx.report["figures"].append({"path": str(p or ctx.path("pt_chain_no_go.png")), "name": "pt_chain_no_go",
                                  "caption": "proof graph"})
    from ..runtime import artifact
    ctx.summary(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return {"metrics": ctx.report["metrics"], "figure": artifact(ctx.path("pt_chain_no_go.png"), "png")}


@node("validate.arena", "Validation arena: red team / blue team", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("problem", "enum", "darcy_2d", choices=["darcy_2d", "burgers_1d"]),
              Param("budget", "enum", "small", choices=["small", "full"]), Param("rounds", "int", 2, minimum=1),
              Param("alpha", "float", 0.01, minimum=1e-4, maximum=0.5), Param("method", "enum", "de",
                                                                                choices=["de", "cmaes", "lbfgs"]),
              Param("seed", "int", 0)],
      calls="pinneapple_analysis.validation.arena.run_arena")
def arena(ctx, inputs, params):
    """The red team looks for wrong outputs that pass every check (escapes) and correct ones that fail (false
    alarms); the blue team calibrates a combined test and adds checks until every stored escape is caught
    (examples/validation_arena/run_arena.py)."""
    from pinneapple_analysis.validation.arena import run_arena
    r = run_arena(params["problem"], budget=None if params["budget"] == "full" else params["budget"],
                  rounds=int(params["rounds"]), alpha=float(params["alpha"]), method=params["method"],
                  seed=int(params["seed"]))
    for rr in r.rounds:
        k = f"round{rr.round}"
        ctx.metric(f"{k}.escapes_found", sum(1 for f in rr.escapes if f.found))
        ctx.metric(f"{k}.heldout_false_alarm_before", float(rr.heldout_false_alarm_before))
        ctx.metric(f"{k}.heldout_false_alarm_after", float(rr.heldout_false_alarm_after))
        a, b = rr.stored_escapes_rejected
        ctx.metric(f"{k}.stored_escapes_rejected", f"{a}/{b}")
        for when in ("before", "after"):
            ms = getattr(rr, f"mutation_{when}") or {}
            if isinstance(ms.get("score"), (int, float)):
                ctx.metric(f"{k}.mutation_score_{when}", float(ms["score"]))
    ctx.metric("passed", int(bool(r.passed)))
    ctx.summary(f"{params['problem']}: {'passed' if r.passed else 'failed: ' + '; '.join(r.failures())}")
    return {"metrics": ctx.report["metrics"]}


@node("validate.audit_model", "Audit a model through the arena's defence", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("problem", "enum", "darcy_2d", choices=["darcy_2d", "burgers_1d"]),
              Param("n_cases", "int", 20, minimum=1), Param("seed", "int", 0)],
      calls="pinneapple_analysis.validation.arena.audit_model + fault_injection.get_problem(...).predict_reference")
def audit(ctx, inputs, params):
    """The reference solver itself judged by the shipped defence: it should have no escapes and no false alarms
    (examples/validation_arena/external_model.py)."""
    from pinneapple_analysis.validation.arena import audit_model
    from pinneapple_analysis.validation.fault_injection import get_problem
    a = audit_model(params["problem"], get_problem(params["problem"]).predict_reference,
                    n_cases=int(params["n_cases"]), seed=int(params["seed"]))
    ctx.metric("escapes", len(a.escapes))
    ctx.metric("false_alarms", len(a.false_alarms))
    ctx.metric("max_true_error", float(np.max(a.true_error)))
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="check.physics", title="Lint results and models for physics (pinneapple check)", domain="validation",
        example="physics_check", keywords={"pinneapple check": 4, "physics linter": 3, "lint": 2, "physics checks": 2,
                                           "ci gate": 2, "conservation check": 1.5}, min_score=3,
        nodes=("check.physics", "examples.compare"), references=("oberkampf2010vv",),
        template="physics_check/fail_and_pass", description="examples/physics_check"))
    register_capability(Capability(
        id="zoo.evaluate", title="Use a pre-trained, verified model from the zoo", domain="validation",
        example="model_zoo", keywords={"zoo": 3, "pre-trained": 2.5, "pretrained": 2.5, "model card": 2,
                                       "fine-tune": 1.5, "airfoil surrogate": 2}, min_score=3,
        nodes=("zoo.evaluate", "examples.compare"), references=("oberkampf2010vv",),
        template="model_zoo/quickstart", description="examples/model_zoo/quickstart.py"))
    register_capability(Capability(
        id="proof.pt_chain", title="Proof graph: PT-symmetric chains, exceptional-point tuning", domain="validation",
        example="proof_graph", keywords={"proof graph": 4, "no-go": 3, "exceptional point": 2.5, "pt-symmetric": 2.5,
                                         "pt chain": 3, "theorem": 1.5, "lemma": 1.5}, min_score=3,
        nodes=("proof.pt_chain",), uncited="Zenodo 23287622 (the PT-dimer paper) is quoted by the example, not in "
                                          "the flow reference registry",
        template="proof_graph/pt_chain_no_go", quick=True, description="examples/proof_graph/pt_chain_no_go.py"))
    register_capability(Capability(
        id="validate.arena", title="Red team / blue team for the validation checks", domain="validation",
        example="validation_arena", keywords={"red team": 3, "blue team": 3, "arena": 2, "escapes": 2,
                                              "false alarm": 2, "adversarial": 1.5}, min_score=3,
        nodes=("validate.arena", "validate.audit_model"), references=("oberkampf2010vv",),
        template="validation_arena/run_arena", quick=True, description="examples/validation_arena"))


_register_capabilities()
