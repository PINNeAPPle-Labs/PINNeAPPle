"""FreeCAD -> OpenFOAM pipeline nodes: a study (spec), the FreeCAD geometry with its named patches, a full run with
its checks and references, a parameter sweep and a surrogate of the sweep (library:
:mod:`pinneapple_simulation.pipelines.freecad_openfoam`)."""
from __future__ import annotations

import json
import shutil

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact, load_artifact

_STUDIES = ["pipe_bend", "heated_pipe", "sphere"]
_TAGS = ("cfd", "openfoam", "freecad", "cad")


def _openfoam_available():
    from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available
    try:
        return (True, "") if openfoam_available() else (False, "OpenFOAM is not installed (FOAM_BASHRC, snappyHexMesh)")
    except OSError as exc:
        return False, str(exc)


def _spec(value):
    from pinneapple_simulation.pipelines.freecad_openfoam import PipelineSpec
    if not value:
        raise StepFailed("connect a study (cfd.study)")
    return PipelineSpec.from_dict(value["spec"])


@node("cfd.study", "FreeCAD -> OpenFOAM study", category="Simulation",
      outputs=[Port("study", "params")],
      params=[Param("study", "enum", "pipe_bend", choices=_STUDIES, help="a validated worked study"),
              Param("resolution", "float", 12.0, minimum=4, maximum=64,
                    help="cells across the diameter (internal) or the snappy surface level (sphere)"),
              Param("backend", "enum", "auto", choices=["auto", "freecad", "fake"],
                    help="FreeCAD headless, the NumPy stand-in, or FreeCAD when installed"),
              Param("overrides", "dict", {}, help="PipelineSpec fields to replace (advanced)", advanced=True)],
      calls="pinneapple_simulation.pipelines.freecad_openfoam.studies (PipelineSpec)", tags=_TAGS)
def study(ctx, inputs, params):
    """A parametric FreeCAD model, its patch rules, mesh, physics, sections and references, ready to run."""
    from pinneapple_simulation.pipelines.freecad_openfoam import STUDIES, PipelineSpec
    name = params["study"]
    res = params["resolution"]
    spec = STUDIES[name](int(res) if name == "sphere" else float(res), backend=params["backend"])
    if params["overrides"]:
        spec = PipelineSpec.from_dict({**spec.to_dict(), **dict(params["overrides"])})
    ctx.table("Inputs", ["input", "default", "min", "max", "unit"],
              [[k, v.get("default"), v.get("min"), v.get("max"), v.get("unit", "")]
               for k, v in {**spec.parameters, **spec.inputs}.items()])
    ctx.table("References", ["name", "measured", "reference", "tolerance", "source"],
              [[k, r["measured"], r["reference"], r.get("tol"), r.get("source", "")] for k, r in spec.references.items()])
    ctx.summary(spec.description)
    return {"study": {"spec": spec.to_dict()}}


@node("cfd.geometry", "FreeCAD geometry and patches", category="Simulation",
      inputs=[Port("study", "params")], outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("parameters", "dict", {}, help="document parameters to set (mm, degrees)")],
      calls="pinneapple_simulation.pipelines.freecad_openfoam.steps.cad_step", tags=_TAGS)
def geometry(ctx, inputs, params):
    """Set the spreadsheet parameters, recompute, export and name the patches; no solver."""
    from pinneapple_simulation.pipelines.freecad_openfoam.steps import cad_step
    spec = _spec(inputs.get("study"))
    values = {**{k: v["default"] for k, v in spec.parameters.items()}, **dict(params["parameters"] or {})}
    try:
        r = cad_step({"document": spec.document or "", "template": spec.template or "", "solid": spec.solid,
                      "parameters": values, "patches": spec.patches, "backend": spec.backend,
                      "tolerance_mm": spec.tolerance_mm, "flow": spec.flow}, ctx.path("cad"))
    except (ValueError, KeyError, RuntimeError) as exc:
        raise StepFailed(str(exc)) from exc
    o = r["outputs"]
    ctx.table("Checks", ["check", "status", "detail"], [[c[0], "PASS" if c[1] else "FAIL", c[2]] for c in r["checks"]])
    ctx.table("Patches", ["patch", "kind", "area mm²"], [[k, o["kinds"][k], a] for k, a in o["patch_areas_mm2"].items()])
    ctx.metric("volume_mm3", o["volume_mm3"], f"{o['backend']} {o['freecad_version']}")
    png = r["figures"][0][0]
    ctx.report["figures"].append({"path": str(png), "name": "geometry", "caption": r["figures"][0][2]})
    return {"metrics": {"outputs": o, "checks": [list(c[:3]) for c in r["checks"]]}, "figure": artifact(png, "png")}


@node("cfd.run", "Run FreeCAD -> OpenFOAM", category="Simulation",
      inputs=[Port("study", "params")],
      outputs=[Port("metrics", "metrics"), Port("report", "report"), Port("figure", "figure")],
      params=[Param("inputs", "dict", {}, help="document parameters and inputs (e.g. {'Rc': 30, 'Re': 150})")],
      calls="pinneapple_simulation.pipelines.freecad_openfoam.run_pipeline (pinneapple_tools.automation.Runner)",
      available=_openfoam_available, tags=_TAGS)
def run(ctx, inputs, params):
    """CAD, mesh, solve, post-process and check one case; the StudyResult and the evidence report are kept."""
    from pinneapple_simulation.pipelines.freecad_openfoam import run_pipeline
    spec = _spec(inputs.get("study"))
    folder = ctx.path("run")
    try:
        rec = run_pipeline(spec, folder, dict(params["inputs"] or {}) or None)
    except Exception as exc:                                     # a bad input is a failed node, with the reason
        raise StepFailed(f"{type(exc).__name__}: {exc}") from exc
    ctx.table("Steps", ["step", "state", "error"], [[s.id, s.state, (s.error or "")[:200]] for s in rec.steps])
    ctx.table("Criteria", ["criterion", "status", "detail"],
              [[c.name, "PASS" if c.passed else "FAIL", c.detail] for c in rec.criteria])
    m = {}
    mj = folder / "post" / "measured.json"
    if mj.is_file():
        d = json.loads(mj.read_text())
        m = d["measured"]
        ctx.table("References", ["name", "measured", "reference", "error %", "source"],
                  [[k, r["measured"], r["reference"], 100 * r["error"], r["source"]] for k, r in d["references"].items()])
        for k, r in d["references"].items():
            ctx.metric(f"{k}_error", r["error"], f"|measured - reference| / reference ({r['source']})")
        for f in ("fields", "mesh"):
            ctx.report["figures"].append({"path": str(folder / "post" / f"{f}.png"), "name": f, "caption": f})
    ctx.summary(f"{rec.state}: {rec.message}")
    fig = folder / "post" / "fields.png"
    return {"metrics": {"state": rec.state, "measured": m}, "report": artifact(folder / "study_result.json", "json"),
            "figure": artifact(fig, "png") if fig.is_file() else None}


@node("cfd.sweep", "Parameter sweep (FreeCAD -> OpenFOAM)", category="Simulation",
      inputs=[Port("study", "params")], outputs=[Port("dataset", "dataset"), Port("metrics", "metrics")],
      params=[Param("bounds", "dict", {"Rc": [20, 80]}, help="input -> [low, high]"),
              Param("n", "int", 8, minimum=2, maximum=200, help="cases (levels per input for a grid)"),
              Param("method", "enum", "lhs", choices=["lhs", "sobol", "grid"]),
              Param("fixed", "dict", {}, help="inputs held at a value"),
              Param("parallel", "int", 1, minimum=1, maximum=64, help="cases run at the same time"),
              Param("seed", "int", 0)],
      calls="pinneapple_simulation.pipelines.freecad_openfoam.sweep (doe, Sweep)", available=_openfoam_available,
      tags=_TAGS)
def sweep(ctx, inputs, params):
    """One process per case, resumable; the table of inputs and measured outputs."""
    from pinneapple_simulation.pipelines.freecad_openfoam import Sweep, doe
    spec = _spec(inputs.get("study"))
    design = doe({k: tuple(v) for k, v in params["bounds"].items()}, int(params["n"]), params["method"],
                 int(params["seed"]), fixed=dict(params["fixed"] or {}))
    try:
        rows = Sweep(spec, design, ctx.path("sweep")).run(parallel=int(params["parallel"]), log=ctx.log)
    except Exception as exc:
        raise StepFailed(f"{type(exc).__name__}: {exc}") from exc
    done = sum(r["state"] in ("passed", "rejected") for r in rows)
    ctx.metric("finished", done, f"of {len(rows)} cases")
    csv = ctx.path("sweep") / "sweep.csv"
    shutil.copy(csv, ctx.path("sweep.csv"))
    return {"dataset": artifact(ctx.path("sweep.csv"), "csv"), "metrics": {"rows": rows}}


@node("cfd.surrogate", "Surrogate of a sweep", category="Simulation",
      inputs=[Port("dataset", "dataset")], outputs=[Port("metrics", "metrics")],
      params=[Param("inputs", "list", ["Rc"], help="input columns"),
              Param("outputs", "list", ["dp_total_pa"], help="output columns"),
              Param("test_frac", "float", 0.25, minimum=0.1, maximum=0.5),
              Param("budget", "float", 5.0, minimum=0.5, maximum=600, help="seconds per training run")],
      calls="pinneapple_simulation.pipelines.freecad_openfoam.sweep.fit_surrogate (pinneapple_analysis.surrogate)",
      tags=_TAGS)
def surrogate(ctx, inputs, params):
    """Every applicable surrogate at equal budget, chosen on validation data, error measured on held-out cases."""
    import pandas as pd

    from pinneapple_simulation.pipelines.freecad_openfoam import fit_surrogate
    ds = inputs.get("dataset")
    if not ds:
        raise StepFailed("connect the dataset of a sweep")
    rows = pd.read_csv(load_artifact(ds)).to_dict("records")
    try:
        st = fit_surrogate(rows, list(params["inputs"]), list(params["outputs"]), test_frac=float(params["test_frac"]),
                           budget=float(params["budget"]))
    except Exception as exc:
        raise StepFailed(f"{type(exc).__name__}: {exc}") from exc
    rank = st.ranking("test")
    ctx.table("Held-out error", ["method", "test nRMSE"], [[r["title"], r["error"]] for r in rank])
    ctx.metric("best", st.best)
    ctx.summary(f"chosen on validation: {st.best}; trust {st.trust.get('decision')}")
    return {"metrics": {"best": st.best, "ranking": rank, "trust": st.trust.get("decision")}}
