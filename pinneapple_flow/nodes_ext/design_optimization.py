"""The standard design-optimisation workflow (:mod:`pinneapple_design.design_optimizer.workflow`, #1188): geometry
and parameters of a library tool (a gmsh-parametrized geometry solved by FEM by default) optimised under
constraints, the optimum re-solved on refined meshes with the mesh error stated, the coarse-mesh error corrected,
checks (convergence, constraints, KKT sensitivities, robustness to tolerances, the known optimum) and the STEP of
the optimum."""
from __future__ import annotations

from pathlib import Path

from ..registry import Param, Port, node

TOOL = "pinneapple_design.design_optimizer.cantilever:tapered_cantilever"
REFERENCE = "pinneapple_design.design_optimizer.cantilever:tapered_cantilever_beam"
EXPORT = "pinneapple_design.design_optimizer.cantilever:export_step"
BOUNDS = {"h_root": [0.04, 0.20], "h_tip": [0.02, 0.10], "thickness": [0.01, 0.04]}
OBJECTIVE = {"name": "mass_kg", "sense": "min"}
CONSTRAINTS = [{"name": "stress_max_Pa", "upper": 150e6}, {"name": "tip_deflection_m", "upper": 8e-3}]
FIDELITY = {"param": "n_through", "coarse": 4, "levels": [8, 16, 32]}


@node("design.optimization", "Design optimization (geometry + parameters, re-verified on refined meshes)",
      category="Design", outputs=[Port("metrics", "metrics")],
      params=[Param("tool", "str", TOOL, help="library tool id 'module:function' (keyword inputs, dict outputs)"),
              Param("bounds", "dict", BOUNDS, help="design variable -> [low, high] (dimensions and parameters)"),
              Param("objective", "dict", OBJECTIVE, help="{name, sense: min | max}"),
              Param("constraints", "list", CONSTRAINTS, help="[{name, upper and/or lower}] on the tool outputs"),
              Param("fixed", "dict", {}, help="inputs held fixed"),
              Param("fidelity", "dict", FIDELITY,
                    help="{param: the tool's mesh keyword, coarse: search value, levels: refined values (x2 each)}"),
              Param("reference", "str", REFERENCE, help="a model with a known optimum to compare with ('' none)"),
              Param("export", "str", EXPORT, help="writes the geometry of the optimum ('' none)"),
              Param("method", "str", "auto", choices=["auto", "scipy-slsqp", "scipy-de", "nsga2", "bayesian",
                                                         "random", "gradient"]),
              Param("budget", "int", 60, minimum=10, help="tool calls per start and per search"),
              Param("n_starts", "int", 2, minimum=1, maximum=8),
              Param("robust", "bool", True, help="back the limits off by the 95th-percentile spread under the "
                                                 "tolerances and re-optimise"),
              Param("tolerances", "dict", {}, help="variable -> +- tolerance (default 1 % of each dimension)"),
              Param("seed", "int", 0)],
      calls="pinneapple_design.design_optimizer.workflow.design_optimization (methods.run, verify.surrogate_assisted, "
            "architecture_recommendation.recommend_architecture, cantilever.tapered_cantilever)")
def design_optimization(ctx, inputs, params):
    """Run the workflow and write the geometry of the optimum next to the run."""
    from pinneapple_design.design_optimizer.workflow import DEFAULT_CANTILEVER
    from pinneapple_design.design_optimizer.workflow import design_optimization as run
    r = run(tool=str(params["tool"]), bounds={k: list(v) for k, v in dict(params["bounds"]).items()},
            objective=dict(params["objective"]), constraints=list(params["constraints"] or []),
            fixed=dict(params["fixed"] or {}), fidelity=dict(params["fidelity"] or {}) or None,
            reference=str(params.get("reference") or "") or None, export=str(params.get("export") or "") or None,
            method=str(params["method"]), budget=int(params["budget"]), n_starts=int(params["n_starts"]),
            robust=bool(params["robust"]), tolerances=dict(params.get("tolerances") or {}) or None,
            seed=int(params["seed"]), out_dir=str(Path(ctx.workdir) / "optimum"),
            units=DEFAULT_CANTILEVER["units"] if str(params["tool"]) == TOOL else None)
    ctx.metric("method", r["method"], r["method_reason"])
    ctx.metric("cost.solve_seconds", r["cost"]["solve_seconds"], "measured: one tool call at the search mesh")
    for k, v in (r.get("design") or {}).items():
        ctx.metric(f"design.{k}", v, "optimum")
    for o, v in (r.get("verified") or {}).items():
        ctx.metric(f"verified.{o}", v["value"], f"Richardson over the meshes {r['fidelity']['levels']}; grid "
                                                f"convergence index {v['gci']}")
    if r.get("robust"):
        ctx.metric("robust.objective_cost", r["robust"]["objective_cost"],
                   "relative objective paid for 95 % feasibility under the tolerances")
    if r.get("reference"):
        ctx.metric("reference.objective", r["reference"]["objective"], r["reference"]["method"])
    if r.get("export"):
        ctx.metric("geometry", r["export"]["path"], "STEP of the optimum (read back, volume checked)")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes")
    obj = r["objectives"][0]
    ctx.summary((f"{obj} = {r['verified'][obj]['value']:.6g} at " + ", ".join(f"{k} = {v:.4g}" for k, v in
                                                                            r["design"].items())
                 if r.get("design") else "no feasible design") + ("; fail: " + ", ".join(bad) if bad else "; verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Design optimization: geometry and parameters, re-verified on refined meshes"
    fb = FlowBuilder(title, "Geometry dimensions and parameters of a tool (a gmsh-parametrized cantilever solved "
                            "by plane-stress FEM by default) optimised under constraints; the optimum re-solved on "
                            "refined meshes (Richardson, grid convergence index), the coarse-mesh error corrected, "
                            "checks on convergence, constraints, sensitivities, robustness and the known optimum, "
                            "and the STEP of the optimum (pinneapple_design.design_optimizer.workflow).")
    fb.add("design", "design.optimization", dict(params), label="Design optimization")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("design", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="design.optimization", title="Design optimization of geometry and parameters (standard workflow)",
        domain="optimization",
        keywords={"design optimization": 4, "design optimisation": 4, "otimização de projeto": 4,
                  "otimização de design": 4, "otimizar o projeto": 3, "otimizar a geometria": 3,
                  "optimize the geometry": 3, "geometry optimization": 3, "otimização de geometria": 3,
                  "minimum mass": 2, "massa mínima": 2, "minimum weight": 2, "peso mínimo": 2,
                  "parametric cad": 2, "cad paramétrico": 2, "cantilever": 1, "viga em balanço": 1,
                  "stress limit": 1, "limite de tensão": 1},
        min_score=3, nodes=("design.optimization",), build=lambda text, det: build_flow(),
        inputs=("a tool with the geometry and parameters as keyword inputs and its mesh keyword",
                "the ranges of the design variables", "the objective and the limits"),
        uncited="the workflow cites Roache (1994) and Celik et al. (2008) for the grid convergence index, Timoshenko & "
                "Gere (1997) and Cowper (1966) for the beam reference in the library modules; they are not in the "
                "flow reference file yet",
        description="pinneapple_design.design_optimizer.workflow.design_optimization"))


_register_capabilities()
