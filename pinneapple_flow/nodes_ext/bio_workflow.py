"""The standard bio workflow (:mod:`pinneapple_physics.biology.workflow`, #1205): an SBML or library model (SIR /
SEIR, Hodgkin-Huxley, FitzHugh-Nagumo, two-compartment PK with Emax PD, Michaelis-Menten) checked for conservation,
positivity and the SBML round trip, simulated with a stiffness-aware solver, verified against closed forms,
analysed for local and Sobol sensitivity, identified from data through the inverse workflow (SBINN optional), used
for an intervention, dosing or stimulus design and re-verified under parameter uncertainty."""
from __future__ import annotations

from ..registry import Param, Port, node

MODELS = ["sir", "seir", "hodgkin_huxley", "fitzhugh_nagumo", "pk_two_compartment", "michaelis_menten", "sbml"]


@node("bio.workflow", "Bio model: checks, sensitivity, identification, design under uncertainty",
      category="Inverse", outputs=[Port("metrics", "metrics")],
      params=[Param("model", "enum", "sir", choices=MODELS,
                    help="a library model, or 'sbml' with the sbml parameter"),
              Param("sbml", "str", "", help="SBML path or text, or a shipped BioModels id (e.g. 12)"),
              Param("model_params", "dict", {}, help="parameter overrides of the model builder"),
              Param("t_end", "float", 0.0, minimum=0.0, help="horizon (0: the model's default)"),
              Param("identify", "list", [], help="parameters to identify (empty: the model's default)"),
              Param("noise", "float", 0.03, minimum=0.0, help="synthetic-data noise, fraction of the range"),
              Param("design", "dict", {}, help="design constraints (capacity, c_min / c_max, amp_max, ...)"),
              Param("uncertainty", "dict", {}, help="parameter -> coefficient of variation (empty: defaults)"),
              Param("target_probability", "float", 0.9, minimum=0.5, maximum=0.999),
              Param("n_sobol", "int", 32, minimum=0, advanced=True),
              Param("n_design", "int", 32, minimum=1, advanced=True),
              Param("n_reverify", "int", 100, minimum=1, advanced=True),
              Param("sbinn_epochs", "int", 0, minimum=0, advanced=True,
                    help="> 0: also fit an SBINN for that many epochs"),
              Param("seed", "int", 0)],
      calls="pinneapple_physics.biology.workflow.bio_workflow")
def bio_workflow_node(ctx, inputs, params):
    from pinneapple_physics.biology.workflow import bio_workflow
    ident = list(params.get("identify") or []) or None
    ep = int(params.get("sbinn_epochs") or 0)
    r = bio_workflow(model=str(params["model"]), sbml=str(params.get("sbml") or "") or None,
                     model_params=dict(params.get("model_params") or {}),
                     t_end=float(params.get("t_end") or 0.0) or None, identify=ident, noise=float(params["noise"]),
                     design=dict(params.get("design") or {}),
                     uncertainty=dict(params.get("uncertainty") or {}) or None,
                     target_probability=float(params["target_probability"]), n_sobol=int(params["n_sobol"]),
                     n_design=int(params["n_design"]), n_reverify=int(params["n_reverify"]),
                     sbinn={"epochs": ep, "epochs_data": max(1, ep // 5)} if ep else None, seed=int(params["seed"]))
    ctx.metric("solver", r["solver"]["chosen"], r["solver"]["why"])
    for k, v in r["qoi"].items():
        ctx.metric(f"qoi.{k}", v)
    for k, v in r["closed_form"].items():
        if isinstance(v, (int, float)):
            ctx.metric(f"closed_form.{k}", v)
    if r["sobol"]:
        s = r["sobol"]
        ctx.table(f"sobol ({s['output']}, +-{s['range']:.0%})", ["parameter", "S1", "ST", "local d ln q / d ln p"],
                  [[n, a, b, r["local_sensitivity"][n][s["output"]]] for n, a, b in zip(s["names"], s["S1"], s["ST"],
                                                                                         strict=True)])
    idn = r["identification"]
    if idn.get("estimated"):
        ctx.table("identification", ["parameter", "estimate", "95 % low", "95 % high", "truth"],
                  [[k, v, *(idn["intervals"][k] or [None, None]), (idn.get("truth") or {}).get(k)]
                   for k, v in idn["estimate"].items()])
    d = r["design"]
    if d.get("run") and d.get("feasible"):
        ctx.metric("design", d["summary"])
        ctx.metric("reverify.probability", d["reverify_probability"], f"{d['n_reverify']} fresh draws")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Bio model workflow: checks, sensitivity, identification, design re-verified under uncertainty"
    fb = FlowBuilder(title, "An SBML or library biological model (epidemic, excitable cell, PK/PD, enzyme network) "
                            "checked for conservation, positivity and the SBML round trip, simulated with a "
                            "stiffness-aware solver, verified against closed forms (SIR final size, PK AUC, the HH "
                            "threshold), analysed for local and Sobol sensitivity, identified from data, used for an "
                            "intervention or dosing design and re-verified under parameter uncertainty "
                            "(pinneapple_physics.biology.workflow).")
    fb.add("bio", "bio.workflow", dict(params), label="Bio workflow")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("bio", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="bio.workflow", title="Bio model workflow (standard)", domain="biology",
        keywords={"sbml": 3, "biomodels": 3, "epidemic": 3, "epidemia": 3, "epidemiológico": 3, "sir": 2,
                  "seir": 3, "pharmacokinetics": 3, "farmacocinética": 3, "pharmacodynamics": 3,
                  "farmacodinâmica": 3, "dosing": 3, "dosagem": 3, "posologia": 3, "dose regimen": 3,
                  "regime de dose": 3, "hodgkin-huxley": 3, "hodgkin huxley": 3, "fitzhugh": 3, "neuron": 2,
                  "neurônio": 2, "action potential": 3, "potencial de ação": 3, "michaelis-menten": 3,
                  "michaelis": 2, "enzyme kinetics": 3, "cinética enzimática": 3, "systems biology": 3,
                  "biologia de sistemas": 3, "intervention": 1, "intervenção": 1, "vaccination": 1, "vacinação": 1},
        min_score=3, nodes=("bio.workflow",), build=lambda text, det: build_flow(),
        inputs=("the model (SBML or a library model) and its parameters", "measurements of an observable, if any",
                "the design constraints and the parameter uncertainty"),
        uncited="the method cites Kermack & McKendrick (1927), Hodgkin & Huxley (1952), Rinzel & Miller (1980), "
                "Holford & Sheiner (1981), Schnell & Mendoza (1997) and Wilks (1941) in the library module; they are "
                "not in the flow reference file yet",
        description="pinneapple_physics.biology.workflow.bio_workflow"))


_register_capabilities()
