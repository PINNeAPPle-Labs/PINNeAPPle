"""The standard multiphysics-coupling workflow (:mod:`pinneapple_simulation.multiphysics.workflow`, #1209): two
physics participants through named interface fields, coupled one-way, two-way partitioned (Gauss-Seidel with Aitken
relaxation) and monolithically; the strategy recommended from the measured coupling strength; convergence,
interface balance and a closed-form coupled solution checked."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("multiphysics.couple", "Multiphysics coupling: one-way, partitioned (Aitken), monolithic, checked",
      category="Solvers", outputs=[Port("metrics", "metrics")],
      params=[Param("case", "enum", "joule_wire", choices=["joule_wire", "bimetal_strip", "conjugate_channel"],
                    help="electro-thermal wire, thermal-structural bimetal strip, or fluid-thermal channel (CHT)"),
              Param("case_params", "dict", {}, help="overrides of the case's inputs (see the library builders)"),
              Param("one_way_tolerance", "float", 0.01, minimum=0.0,
                    help="relative error of the quantities of interest at which one-way is accepted"),
              Param("coupling_tolerance", "float", 1e-10, minimum=1e-14, advanced=True),
              Param("balance_tolerance", "float", 1e-3, minimum=0.0, advanced=True),
              Param("closed_form_tolerance", "float", 1e-3, minimum=0.0, advanced=True),
              Param("max_iter", "int", 200, minimum=1, advanced=True)],
      calls="pinneapple_simulation.multiphysics.workflow.multiphysics_workflow")
def multiphysics_couple(ctx, inputs, params):
    from pinneapple_simulation.multiphysics.workflow import multiphysics_workflow as run
    from pinneapple_simulation.multiphysics.workflow import residual_table
    r = run(case=str(params["case"]), case_params=dict(params.get("case_params") or {}),
            one_way_tolerance=float(params["one_way_tolerance"]), coupling_tolerance=float(params["coupling_tolerance"]),
            balance_tolerance=float(params["balance_tolerance"]),
            closed_form_tolerance=float(params["closed_form_tolerance"]), max_iter=int(params["max_iter"]))
    ctx.metric("coupling.contraction", r["coupling_strength"]["contraction"],
               "spectral radius of the plain partitioned map, measured from its first iterations")
    ctx.metric("recommended", r["recommended"], r["why"])
    for k, v in r["one_way_error"].items():
        ctx.metric(f"one_way_error.{k}", v, "relative to the two-way solution")
    for name, s in r["strategies"].items():
        ctx.metric(f"{name}.iterations", s["iterations"], f"{s['seconds'] * 1e3:.3g} ms")
    ctx.table("strategies", ["strategy", "converged", "iterations", "ms"] + list(r["strategies"]["one_way"]["qoi"]),
              [[k, s["converged"], s["iterations"], s["seconds"] * 1e3] + list(s["qoi"].values())
               for k, s in r["strategies"].items()])
    ctx.table("interface_residual", ["iteration", "gauss_seidel", "gauss_seidel_aitken"], residual_table(r))
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Multiphysics coupling: one-way, partitioned with Aitken, monolithic, checked"
    fb = FlowBuilder(title, "Two physics participants through named interface fields, coupled one-way, two-way "
                            "partitioned (Gauss-Seidel with Aitken relaxation) and monolithically; the strategy from "
                            "the measured coupling strength; convergence, interface balance and a closed-form "
                            "coupled solution (pinneapple_simulation.multiphysics.workflow).")
    fb.add("couple", "multiphysics.couple", dict(params), label="Multiphysics coupling")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("couple", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="multiphysics.couple", title="Multiphysics coupling (standard workflow)", domain="multiphysics",
        keywords={"multiphysics": 3, "multifísica": 3, "multifisica": 3, "coupled": 1, "acoplado": 1,
                  "coupling": 2, "acoplamento": 2, "thermal stress": 3, "tensão térmica": 3, "joule heating": 3,
                  "aquecimento joule": 3, "efeito joule": 3, "conjugate heat transfer": 3,
                  "transferência de calor conjugada": 3, "bimetal": 3, "bimetálica": 3, "aitken": 3,
                  "partitioned": 2, "particionado": 2, "monolithic": 2, "monolítico": 2,
                  "fluid-structure": 2, "fluido-estrutura": 2, "electro-thermal": 3, "eletrotérmico": 3},
        min_score=3, nodes=("multiphysics.couple",), build=lambda text, det: build_flow(),
        inputs=("the physics to couple and the interface fields", "the materials and loads",
                "how close one-way must be to accept it"),
        uncited="the method cites Irons & Tuck (1969), Kuttler & Wall (2008) and Timoshenko (1925) in the library "
                "module; they are not in the flow reference file yet",
        description="pinneapple_simulation.multiphysics.workflow.multiphysics_workflow"))


_register_capabilities()
