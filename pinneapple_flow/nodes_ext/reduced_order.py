"""The standard reduced-order-modeling workflow (:mod:`pinneapple_analysis.reduced_order.workflow`, #1208):
snapshots from a library full-order model or a file, a POD basis whose rank comes from the singular-value energy,
POD-Galerkin, DMD and operator inference on one protocol, and the checks against the full model on parameters and
times never used for training (interpolation, extrapolation in time, stability, invariant, measured speed-up)."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("rom.workflow", "Reduced-order modeling: POD-Galerkin, DMD, OpInf checked against the FOM",
      category="Surrogates", outputs=[Port("metrics", "metrics")],
      params=[Param("fom", "enum", "burgers", choices=["burgers", "advection_diffusion"],
                    help="library full-order model (1-D periodic, parameter mu = viscosity / diffusivity)"),
              Param("mu_range", "list", [0.05, 0.15], help="[low, high] of the parameter"),
              Param("n_params", "int", 9, minimum=3, help="parameter values; the odd positions are held out"),
              Param("n", "int", 512, minimum=32, help="FOM grid points"),
              Param("t_end", "float", 2.0, minimum=0.01),
              Param("snapshots_file", "str", "", help="instead of the FOM: .npz with snapshots (P, T, n), times, "
                                                      "params (POD-Galerkin is then skipped: it needs the operator)"),
              Param("train_fraction", "float", 0.75, minimum=0.1, maximum=0.95,
                    help="share of the time window used for training; the rest tests extrapolation in time"),
              Param("energy", "float", 0.99999, minimum=0.5, maximum=1.0, help="singular-value energy for the rank"),
              Param("max_rank", "int", 40, minimum=1),
              Param("methods", "list", ["pod_galerkin", "dmd", "opinf"]),
              Param("invariant", "enum", "mass", choices=["mass", ""], help="invariant checked in every ROM"),
              Param("tolerance", "float", 0.02, minimum=1e-6, help="interpolation error allowed"),
              Param("extrapolation_tolerance", "float", 0.05, minimum=1e-6),
              Param("invariant_tolerance", "float", 1e-6, minimum=0.0, advanced=True),
              Param("repeats", "int", 3, minimum=1, advanced=True, help="timing repeats (best of)")],
      calls="pinneapple_analysis.reduced_order.workflow.reduced_order_workflow")
def rom_workflow(ctx, inputs, params):
    from pinneapple_analysis.reduced_order.workflow import as_table
    from pinneapple_analysis.reduced_order.workflow import reduced_order_workflow as run
    r = run(fom=str(params["fom"]), mu_range=tuple(float(v) for v in params["mu_range"]),
            n_params=int(params["n_params"]), n=int(params["n"]), t_end=float(params["t_end"]),
            snapshots_file=str(params.get("snapshots_file") or ""), train_fraction=float(params["train_fraction"]),
            energy=float(params["energy"]), max_rank=int(params["max_rank"]), methods=list(params["methods"]),
            invariant=str(params["invariant"]), tolerance=float(params["tolerance"]),
            extrapolation_tolerance=float(params["extrapolation_tolerance"]),
            invariant_tolerance=float(params["invariant_tolerance"]), repeats=int(params["repeats"]))
    b = r["basis"]
    ctx.metric("data", r["source"])
    ctx.metric("rank", b["rank"], f"POD modes {b['pod_rank']} (+ the constant mode when the invariant is the mass); "
                                  f"energy {b['energy_captured']:.8f}")
    ctx.metric("projection_error", b["projection_error_extrapolation"],
               f"floor of any ROM in this basis on the held-out runs (interpolation window "
               f"{b['projection_error_interpolation']:.2g})")
    for k, v in r["methods"].items():
        if "skipped" in v:
            ctx.metric(f"{k}.skipped", v["skipped"])
            continue
        ctx.metric(f"{k}.interpolation_error", v["interpolation_error"], f"held-out mu {r['test_params']}, "
                                                                         f"t <= {r['t_train']:g}")
        ctx.metric(f"{k}.extrapolation_error", v["extrapolation_error"], f"{r['t_train']:g} < t <= {r['t_end']:g}")
        if v.get("speedup"):
            ctx.metric(f"{k}.speedup", v["speedup"], f"online {v['online_seconds'] * 1e3:.3g} ms, measured")
    ctx.table("roms", ["method", "interpolation", "extrapolation", "growth", "mass drift", "speed-up", "verdict"],
              as_table(r))
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("recommended", r["recommended"] or "none", "the passing ROM with the lowest extrapolation error")
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Reduced-order modeling: POD-Galerkin, DMD and operator inference checked against the full model"
    fb = FlowBuilder(title, "Snapshots of a full-order model (or a file), the POD rank from the singular-value "
                            "energy, POD-Galerkin, DMD and operator inference on one protocol, checked on held-out "
                            "parameters and on extrapolation in time, with stability, the invariant and the "
                            "measured speed-up (pinneapple_analysis.reduced_order.workflow).")
    fb.add("rom", "rom.workflow", dict(params), label="Reduced-order modeling")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("rom", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="rom.workflow", title="Reduced-order modeling (standard workflow)", domain="surrogates",
        keywords={"reduced-order": 3, "reduced order model": 3, "modelo de ordem reduzida": 3, "ordem reduzida": 3,
                  "pod-galerkin": 3, "pod galerkin": 3, "operator inference": 3, "inferência de operadores": 3,
                  "dynamic mode decomposition": 3, "decomposição em modos dinâmicos": 3, "rom": 2,
                  "snapshots": 1, "proper orthogonal decomposition": 3, "decomposição ortogonal própria": 3},
        min_score=3, nodes=("rom.workflow",), build=lambda text, det: build_flow(),
        inputs=("a full-order model with a parameter, or snapshots (P, T, n)", "the parameter range",
                "the error allowed"),
        uncited="the method cites Peherstorfer & Willcox (2016) for operator inference and Schmid (2010) for DMD in "
                "the library module; they are not in the flow reference file yet",
        description="pinneapple_analysis.reduced_order.workflow.reduced_order_workflow"))


_register_capabilities()
