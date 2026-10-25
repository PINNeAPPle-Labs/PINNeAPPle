"""Physics learning verification node: does a learned thermal model know the physics, beyond its loss? (library:
:mod:`pinneapple_analysis.validation.physics_learning`)."""
from __future__ import annotations

from ..registry import Param, Port, node

_TAGS = ("section:validation", "thermal", "PINN", "surrogate", "verification", "interventions", "trust")

_MODELS = ("reference", "smooth error 0.3 % (mode 1)", "smoothed interfaces", "frozen outside the box",
           "reversed TIM response", "cooling learned from power", "3 % energy created in the lid", "leaky sides",
           "saturating power response", "POD-GPR surrogate, correlated training data",
           "POD-GPR surrogate, independent training data", "detection study (all defects, severities, budgets)")


@node("analysis.physics_learning", "Physics learning verification of a learned thermal model", category="Analysis",
      outputs=[Port("metrics", "metrics")],
      params=[Param("model", "enum", _MODELS[2], choices=list(_MODELS),
                    help="the model under test on the die / TIM / lid section (or the whole detection study)"),
              Param("severity", "float", 1.0, minimum=0.0, help="size of a deliberate defect"),
              Param("points", "int", 48, minimum=4, maximum=400, help="informative points checked"),
              Param("seed", "int", 0)],
      calls="pinneapple_analysis.validation.physics_learning.verify / detection_study", tags=_TAGS)
def physics_learning(ctx, inputs, params):
    """Check a model with a verifier independent of its training loss -- discrete residual, interface flux,
    boundary conditions, energy balance, symmetry, scaling and causal interventions against a finite-volume
    reference -- and report each check, its provenance (architecture or learned), the decision, and what the usual
    metrics (held-out error, physics loss) would have said."""
    from pinneapple_analysis.validation.physics_learning import baselines, detection_study, verify
    from pinneapple_analysis.validation.physics_learning_cases import (
        Section,
        defective_models,
        good_models,
        surrogate_model,
    )
    name = params["model"]
    if name.startswith("detection study"):
        r = detection_study(seed=int(params["seed"]))
        s = r["summary"]
        rows = [[g, s[g]["n"], s[g]["verifier"], s[g]["flag_loss"], s[g]["flag_heldout"]] for g in ("defect", "control")]
        ctx.table("Detection rate", ["models", "n", "verifier", "physics loss > limit", "held-out L2 > limit"], rows)
        ctx.summary(f"verifier detects {s['defect']['verifier']:.0%} of {s['defect']['n']} defective models; "
                    f"physics loss {s['defect']['flag_loss']:.0%}, held-out L2 {s['defect']['flag_heldout']:.0%}")
        return {"metrics": {"defect_verifier": s["defect"]["verifier"], "defect_flag_loss": s["defect"]["flag_loss"],
                            "defect_flag_heldout": s["defect"]["flag_heldout"], "defects": s["defect"]["n"],
                            "controls_flagged": s["control"]["verifier"]}}
    case = Section()
    sev = float(params["severity"])
    pool = {m.name: m for m in good_models(case) + defective_models(case, sev)}
    pool["smooth error 0.3 % (mode 1)"] = good_models(case, amplitudes=(0.003,), modes=(1,))[1]
    if name.startswith("POD-GPR"):
        m = surrogate_model(case, correlated="correlated" in name, seed=int(params["seed"]))
    else:
        m = pool[name]
    r = verify(m, case, k=int(params["points"]), seed=int(params["seed"]))
    b = baselines(m, case)
    ctx.table("Checks", ["check", "passed", "value", "limit", "provenance"],
              [[f.check, f.passed, f.value, f.limit, f.provenance] for f in r.findings])
    ctx.summary(f"{m.name}: {r.trust.decisao}; {len(r.failed)} checks failed, {len(r.out_of_scope)} out of scope; "
                f"held-out L2 {b['heldout_rel_l2']:.2g}, physics loss {b['physics_loss']:.2g}")
    return {"metrics": {"detected": r.detected, "decision": r.trust.decisao, "failed": [f.check for f in r.failed],
                        "out_of_scope": [f.check for f in r.out_of_scope], **b}}
