"""Physics Design Optimizer: optimise parametric designs with explicit objectives, constraints and physical models,
gradient-aware, and compare the methods on the same problem and budget.

Run:  uvicorn physics_design.api:app --port 8106     (from apps/physics_design)

Each study is a job of the shared runtime (apps/_shared/appkit.py). The methods, the differentiable surrogates, the
verification and the design files all come from pinneapple_design.design_optimizer.

Environment (all optional; runtime variables in apps/_shared/README.md):
  PDO_USER / PDO_PASSWORD   HTTP Basic login on everything except /health
  PDO_MAX_JOBS              studies running at once (default 2)
  PDO_FREE_EVALUATIONS      free evaluations per caller and month (default 200000)
  PDO_FREE_CAMPAIGNS        free studies per caller and month (default 100)
  PDO_FREE_SOLVER_RUNS      free high-fidelity solver runs per caller and month (default 500)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_design.aero.geometry import PARAMS, outline
from pinneapple_design.design_optimizer import (
    DESIGN_FILES,
    available_methods,
    chart_points,
    compare_methods,
    feasibility_report,
    presets,
    problem_from_spec,
    reference_for_spec,
    study_result,
    verify_front,
    write_design_files,
)
from pinneapple_design.design_optimizer.spec import SpecError

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
NAME = "design"
STATIC = os.path.join(os.path.dirname(__file__), "static")
METHODS = ("gradient", "scipy-slsqp", "nsga2", "bayesian", "scipy-de", "random")
LABEL = {"gradient": "Gradient (autograd through the model)", "scipy-slsqp": "SLSQP (finite-difference gradients)",
         "nsga2": "Evolutionary (NSGA-II)", "bayesian": "Bayesian optimisation (GP)",
         "scipy-de": "Differential evolution", "random": "Latin hypercube (baseline)"}
MAX_TOTAL_EVALS = 40000
MAX_BAYES = 150
FILES = ("comparison.csv", *DESIGN_FILES, "study.json")

app = FastAPI(title="Physics Design Optimizer", version=VERSION,
              description="Optimise parametric designs (plate-fin heat sink, CST airfoil) with gradient methods "
                          "through differentiable surrogates, evolutionary and Bayesian optimisation; compare them on "
                          "the same budget; every candidate carries its verification status; STL/STEP/CSV exports.")
install(app, prefix="PDO")
rt = install_runtime(app, prefix="PDO", name=NAME, units={"evaluations": 200_000, "campaigns": 100,
                                                          "solver_runs": 500})


def scope() -> dict:
    return {
        "title": "Scope & validation", "noun": "limitation", "items_title": "What the results assume",
        "validated": [
            "Gradient path checked against the NumPy path of the same models: identical outputs (tests)",
            "Heat-sink candidates re-evaluated with the 3-D finite-volume engine the surrogate was trained on; the "
            "error is shown per candidate",
            "Method comparison on an analytic two-objective problem with a known Pareto set (tests)",
        ],
        "band": "A design is 'verified' only when the high-fidelity model ran on it and it met every limit there.",
        "tags": {"conservative": "Safe by default", "optimistic": "Review before use",
                 "check": "Assumption to confirm"},
        "items": [
            {"topic": "Gradients come from the surrogate", "effect": "check",
             "detail": "Autograd differentiates the neural surrogate (and the closed-form geometry), not the "
                       "solver. Its optimum is verified like any other.",
             "today": "Read the verification column before using a design.", "planned": ""},
            {"topic": "Integer fin count relaxed", "effect": "check",
             "detail": "Gradient methods treat the fin count as continuous during the iterations; only rounded "
                       "designs become candidates.", "today": "", "planned": ""},
            {"topic": "Airfoil: no live CFD", "effect": "optimistic",
             "detail": "The airfoil ensemble was trained on OpenFOAM k-omega SST runs at Re 4e6; no CFD runs here, "
                       "so airfoil candidates stay 'only surrogate'.",
             "today": "Verify with OpenFOAM (apps/aero_optimizer/tools/verify.py) or the Aircraft Design Optimizer.",
             "planned": "Queue CFD verification on the Workflow Runner."},
            {"topic": "Equal budget, not equal wall time", "effect": "check",
             "detail": "Every method gets the same number of model evaluations; an autograd step counts as one, a "
                       "finite-difference gradient as d+1. Wall time is reported next to it.",
             "today": "", "planned": ""},
        ]}


# ----------------------------------------------------------------------------------------------- job
def study_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    t0 = time.time()
    spec = params["spec"]
    problem, _ = problem_from_spec(spec)
    methods, budget, seeds = params["methods"], int(params["budget"]), [int(s) for s in params["seeds"]]
    ctx.log(f"{problem.name}: {methods} x seeds {seeds}, {budget} evaluations each, on {problem.evaluator.name}")
    cmp = compare_methods(problem, methods, budget=budget, seeds=seeds,
                          progress=lambda k, n: ctx.progress(0.03 + 0.8 * k / n, f"run {min(int(k) + 1, n)} of {n}"))
    ev, labels = cmp.union()
    if ev is None:
        raise SpecError("no method produced evaluations")
    front = [int(i) for i in ev.pareto()]
    reference = reference_for_spec(spec)
    if reference is not None:
        ctx.progress(0.86, f"verifying candidates with {reference.name}")
    cand, ver, status = verify_front(problem, ev, front, int(params.get("n_verify", 5)), reference)
    feas = feasibility_report(problem, ev, cand)
    ctx.progress(0.95, "writing files")
    wd = Path(ctx.workdir)
    pd.DataFrame(cmp.table()).drop(columns=["best"]).to_csv(wd / "comparison.csv", index=False)
    write_design_files(wd, problem, ev, cand, status, labels=labels)
    study_result(problem, cmp.union_result(), study_id=ctx.job_id, app=NAME, candidates=cand, verification=ver,
                 comparison=cmp, limits=[scope()["items"][0]["detail"]]).save(wd / "study.json")
    pts = [q for q in chart_points(problem, ev, front, limit=2500, labels=labels) if q["f"] and not q["r"]]
    rows = ev.rows(problem.space, cand)
    for r, f in zip(rows, feas, strict=True):
        r["found_by"] = labels[r["index"]].split("#")[0]
        r["verification"] = status.get(r["index"], "only surrogate")
        r["feasibility"] = f
        if spec.get("preset") == "airfoil":
            r["outline"] = outline(np.array([r[p] for p in PARAMS]), 60)
    return {
        "problem": problem.to_dict(), "methods": {m: LABEL.get(m, m) for m in methods},
        "table": cmp.table(), "runs": [{k: v for k, v in x.items() if k != "front"} for x in cmp.rows],
        "curves": cmp.to_dict()["curves"], "winner": cmp.winner(),
        "fronts": cmp.fronts(seeds[0]), "points": pts, "candidates": rows,
        "verification": ver.to_dict() if ver is not None else None,
        "reference": reference.info() if reference is not None else None,
        "files": list(FILES), "seconds": round(time.time() - t0, 2), "budget": budget, "seeds": seeds,
    }


rt.jobs.register("study", study_job)


# ----------------------------------------------------------------------------------------------- routes
class StudyRequest(BaseModel):
    spec: dict[str, Any] = Field(description="problem spec: preset heatsink or airfoil (pinneapple_design spec)")
    methods: list[str] = Field(default_factory=lambda: ["gradient", "nsga2", "bayesian", "random"])
    budget: int = Field(100, ge=10, le=5000, description="evaluations per method and seed")
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=5)
    n_verify: int = Field(5, ge=0, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    avail = {"heatsink": available_methods(presets.heatsink_problem("surrogate")),
             "airfoil": available_methods(presets.airfoil_problem())}
    return {"version": VERSION, "scope": scope(), "labels": LABEL,
            "methods": {k: {m: v[m] for m in METHODS} for k, v in avail.items()},
            "limits": {"max_total_evaluations": MAX_TOTAL_EVALS, "max_budget_bayesian": MAX_BAYES},
            "presets": {k: presets.preset_info(k) for k in ("heatsink", "airfoil")},
            "runtime": rt.describe()}


@app.post("/api/studies")
def create_study(request: Request, req: StudyRequest):
    """Queue a method comparison. Metered: methods x seeds x budget evaluations, one campaign, the solver runs."""
    bad = [m for m in req.methods if m not in METHODS]
    if bad or not req.methods:
        raise HTTPException(422, f"methods must be among {METHODS}")
    if "bayesian" in req.methods and req.budget > MAX_BAYES:
        raise HTTPException(422, f"with Bayesian optimisation the budget is at most {MAX_BAYES} evaluations")
    total = req.budget * len(req.methods) * len(req.seeds)
    if total > MAX_TOTAL_EVALS:
        raise HTTPException(422, f"{total} evaluations in total: at most {MAX_TOTAL_EVALS}")
    spec = {**req.spec, "source": "preset"}
    if spec.get("preset", "heatsink") not in ("heatsink", "airfoil"):
        raise HTTPException(422, "preset must be heatsink or airfoil")
    if spec.get("preset", "heatsink") == "heatsink":
        spec["fidelity"] = "surrogate"            # the differentiable model; the FVM engine verifies
    try:
        problem, _ = problem_from_spec(spec)
    except SpecError as e:
        raise HTTPException(422, str(e)) from e
    avail = available_methods(problem)
    off = [f"{m}: {avail[m]['reason']}" for m in req.methods if not avail[m]["available"]]
    if off:
        raise HTTPException(422, "; ".join(off))
    params = req.model_dump() | {"spec": spec}
    units = {"evaluations": total, "campaigns": 1}
    if reference_for_spec(spec) is not None and req.n_verify:
        units["solver_runs"] = req.n_verify
    return rt.submit(request, "study", params, units=units)


def _job_file(request: Request, job_id: str, name: str) -> Path:
    job = rt.job_for(request, job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    p = rt.jobs.workdir / job_id / name
    if not p.exists():
        raise HTTPException(404, f"{name} was not produced by this study")
    return p


@app.get("/api/studies/{job_id}/files/{name}")
def download(request: Request, job_id: str, name: str):
    if name not in FILES:
        raise HTTPException(404, f"files: {FILES}")
    return FileResponse(_job_file(request, job_id, name), filename=f"{job_id}_{name}")


@app.get("/api/studies/{job_id}/design/{k}.{fmt}")
def design_file(request: Request, job_id: str, k: int, fmt: str):
    """Candidate ``k`` (1-based, as in the table) as a binary STL or an AP214 STEP file (metres)."""
    if fmt not in ("stl", "step"):
        raise HTTPException(404, "format is stl or step")
    df = pd.read_csv(_job_file(request, job_id, "candidates.csv"))
    if not 1 <= k <= len(df):
        raise HTTPException(404, f"candidates are 1 to {len(df)}")
    spec = rt.job_for(request, job_id)["params"]["spec"]
    row = df.iloc[k - 1].to_dict()
    name = f"{spec.get('preset', 'heatsink')}_{job_id}_c{k}"
    data = presets.design_file(presets.preset_bodies(spec.get("preset", "heatsink"), row), fmt, name)
    return Response(data, media_type="model/stl" if fmt == "stl" else "application/step",
                    headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'})


@app.get("/api/presets/{preset}/baseline.{fmt}")
def baseline_file(preset: str, fmt: str):
    """The baseline design of a preset as STL or STEP (to compare in your CAD tool)."""
    if preset not in ("heatsink", "airfoil") or fmt not in ("stl", "step"):
        raise HTTPException(404, "heatsink or airfoil, stl or step")
    row = presets.heatsink_baseline() if preset == "heatsink" else presets.airfoil_baseline()
    data = presets.design_file(presets.preset_bodies(preset, row), fmt, f"{preset}_baseline")
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{preset}_baseline.{fmt}"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))

