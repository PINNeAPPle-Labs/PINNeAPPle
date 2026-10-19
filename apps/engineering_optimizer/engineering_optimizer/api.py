"""Engineering Design Optimizer: better designs within physical and manufacturing limits, black-box.

Run:  uvicorn engineering_optimizer.api:app --port 8100     (from apps/engineering_optimizer)

Every optimisation is a job of the shared runtime (apps/_shared/appkit.py): queued, metered, with progress and
logs. The numbers come from pinneapple_design.design_optimizer: the problem spec, the methods, the high-fidelity
verification and the contracts (ParameterSpace, ModelRef, StudyResult) all live in the library.

Environment (all optional; runtime variables in apps/_shared/README.md):
  EDO_USER / EDO_PASSWORD   HTTP Basic login on everything except /health
  EDO_MAX_JOBS              optimisations running at once (default 2)
  EDO_FREE_EVALUATIONS      free evaluations per caller and month (default 200000)
  EDO_FREE_CAMPAIGNS        free optimisation campaigns per caller and month (default 100)
  EDO_FREE_REFERENCE_RUNS   free high-fidelity (reference / solver) runs per caller and month (default 500)
  EDO_SOLVERS               JSON of external solvers the server operator allows as references:
                            {"name": {"command": ["prog", "{inputs}", "{outputs}"], "outputs": [...], "timeout": 600}}
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pinneapple as pp
from pinneapple_data.contracts import read_table
from pinneapple_design.design_optimizer import (
    DESIGN_FILES,
    CommandEvaluator,
    DesignProblem,
    DesignSpace,
    available_methods,
    chart_points,
    designs_frame,
    feasibility_report,
    model_ref_for_surrogate,
    presets,
    problem_from_spec,
    reference_for_spec,
    run,
    select_candidates,
    study_result,
    surrogate_assisted,
    verify_front,
    verify_uploaded_results,
    write_design_files,
)
from pinneapple_design.design_optimizer.spec import SpecError, constraints_from, objectives_from

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
NAME = "optimize"
STATIC = os.path.join(os.path.dirname(__file__), "static")
SEARCH_METHODS = ("nsga2", "pymoo-nsga2", "scipy-de", "bayesian", "random")
MAX_BUDGET = {"bayesian": 150}
MAX_BUDGET_DEFAULT = 6000
FILES = (*DESIGN_FILES, "study.json", "model_ref.json", "verification_uploaded.json")

app = FastAPI(title="Engineering Design Optimizer", version=VERSION,
              description="Find better designs within physical and manufacturing constraints: NSGA-II, pymoo, SciPy "
                          "or Bayesian optimisation on a surrogate, a library physics model or a model trained on your "
                          "data, with the chosen candidates re-evaluated by a high-fidelity reference.")
install(app, prefix="EDO")
rt = install_runtime(app, prefix="EDO", name=NAME,
                     units={"evaluations": 200_000, "campaigns": 100, "reference_runs": 500},
                     upload_suffixes={".csv", ".parquet", ".json"}, upload_max_mb=20)


def _solvers() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("EDO_SOLVERS") or ""
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return {str(k): dict(v) for k, v in d.items() if isinstance(v, dict) and v.get("command")}
    except (ValueError, AttributeError):
        return {}


def scope() -> dict:
    return {
        "title": "Scope & validation", "noun": "limitation", "items_title": "What the results assume",
        "validated": [
            "Heat-sink surrogate (pp.zoo heatsink_forced_mlp) against the FVM engine it was trained on: the "
            "candidates of each run are re-evaluated with the FVM engine and the error is shown per output",
            "Optimisers checked on an analytic two-objective problem with a known Pareto set: NSGA-II reaches 99 % of "
            "its hypervolume in 1200 evaluations (tests/test_design_studies.py)",
            "Surrogates trained on your data report their error on held-out rows, not on the training rows",
        ],
        "band": "A candidate is called verified only when the reference ran on it and it met every constraint there.",
        "tags": {"conservative": "Safe by default", "optimistic": "Review before use",
                 "check": "Assumption to confirm"},
        "items": [
            {"topic": "Surrogate optimum, then verification", "effect": "check",
             "detail": "The search runs on a fast model; only the candidates are re-evaluated with the reference. "
                       "A design the reference rejects is shown as such, not hidden.",
             "today": "Verify more candidates, or use active learning so the surrogate is refined where it matters.",
             "planned": "Queue verification runs on the Workflow Runner (cloud solvers)."},
            {"topic": "Heat sink: ducted forced air", "effect": "optimistic",
             "detail": "All air passes between the fins (a shroud); laminar/developing channel flow; no radiation.",
             "today": "Without a shroud, keep margin on temperature (bypass costs 20-50 %).",
             "planned": ""},
            {"topic": "Cost is your model", "effect": "check",
             "detail": "Unit cost = material price x mass / yield + cost per fin + fixed cost, with values you set; "
                       "the defaults are placeholders, not quotes.", "today": "Enter your supplier's numbers.",
             "planned": ""},
            {"topic": "Your data: GP surrogate", "effect": "check",
             "detail": "A Gaussian process per output, bounded to the range of your data. Its held-out error is "
                       "shown; candidates stay 'only surrogate' until your solver confirms them.",
             "today": "Download candidates_to_run.csv, run them, upload the results to verify.",
             "planned": ""},
        ]}


# ----------------------------------------------------------------------------------------------- jobs
def _reference(spec: dict, verify_with: str | None, problem: DesignProblem):
    if not verify_with or verify_with == "none":
        return None
    if verify_with == "library":
        return reference_for_spec(spec)
    s = _solvers().get(verify_with)
    if s is None:
        raise SpecError(f"unknown solver {verify_with!r}")
    return CommandEvaluator(s["command"], problem.space, s.get("outputs") or [], timeout=float(s.get("timeout", 600)),
                            name=f"solver: {verify_with}")


def optimize_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    t0 = time.time()
    spec = params["spec"]
    table = None
    if spec.get("source") == "table":
        table, _ = read_table(params["table_path"])
    ctx.log(f"source: {spec.get('source', 'preset')}")
    problem, fit = problem_from_spec(spec, table)
    if fit is not None:
        ctx.log("surrogate held-out error: " + ", ".join(f"{k} R2={m['r2']:.3f}" for k, m in fit.metrics.items()
                                                         if m.get("r2") is not None))
    ctx.progress(0.03, "problem ready")
    reference = _reference(spec, params.get("verify_with"), problem)
    method, budget, seed = params["method"], int(params["budget"]), int(params.get("seed", 0))
    n_verify = int(params.get("n_verify", 5))
    initial = None
    if spec.get("source", "preset") == "preset" and spec.get("preset", "heatsink") == "heatsink":
        try:
            initial = problem.space.from_dicts([presets.heatsink_baseline()])
        except ValueError:
            initial = None
    al = None
    if params.get("mode") == "active_learning":
        if reference is None:
            raise SpecError("active learning needs a reference (the library reference or a configured solver)")
        al = surrogate_assisted(problem, reference, method=method, n_initial=int(params.get("n_initial", 24)),
                                rounds=int(params.get("rounds", 3)), n_verify=n_verify,
                                surrogate_budget=min(budget, 1500), seed=seed,
                                progress=lambda k, n: ctx.progress(0.05 + 0.85 * k / n, f"round {k:.1f} of {n}"))
        result, ver = al.final, al.verification
        rp = result.problem
        cand = select_candidates(result.evaluations, result.pareto(), max(n_verify, 8))
        status = {i: "verified" for i in cand if result.evaluations.feasible[i]}
    else:
        result = run(problem, method, budget=budget, seed=seed, initial=initial,
                     progress=lambda u, b: ctx.progress(0.05 + 0.75 * u / b, f"{u} / {b} evaluations"))
        if result.status != "ok":
            raise SpecError(f"{method}: {result.message}")
        rp = problem
        if reference is not None:
            ctx.progress(0.82, f"verifying candidates with {reference.name}")
        cand, ver, status = verify_front(problem, result.evaluations, result.pareto(), n_verify, reference)
        if ver is not None:
            ctx.log(f"verification: {[r['status'] for r in ver.rows]} in {ver.wall_s:.1f} s")
    ev = result.evaluations
    front = [int(i) for i in result.pareto()]
    ctx.progress(0.95, "writing files")
    wd = Path(ctx.workdir)
    files = write_design_files(wd, rp, ev, cand, status)
    if fit is not None:
        model_ref_for_surrogate(fit, f"{NAME}-gp-{ctx.job_id}", f"app:{NAME}/jobs/{ctx.job_id}",
                                params.get("table_name", "")).save(wd / "model_ref.json")
        files.append("model_ref.json")
    sr = study_result(rp, result, study_id=ctx.job_id, app=NAME, candidates=cand, verification=ver,
                      active_learning=al)
    sr.save(wd / "study.json")
    files.append("study.json")
    feas = ev.F[ev.feasible & ~ev.relaxed]
    curve = result.hv_curve(feas.min(0), feas.max(0)) if len(feas) else []
    baseline = None
    if initial is not None and al is None:
        baseline = ev.rows(rp.space, [0])[0] | {"feasibility": feasibility_report(rp, ev, [0])[0]}
    return {
        "problem": rp.to_dict(), "summary": result.summary(), "mode": params.get("mode", "direct"),
        "points": chart_points(rp, ev, front), "pareto": ev.rows(rp.space, front),
        "candidates": [r | {"verification": status.get(r["index"], "only surrogate" if r["fidelity"] == "surrogate"
                                                       else "reference")}
                       for r in ev.rows(rp.space, cand)],
        "feasibility": feasibility_report(rp, ev, cand),
        "verification": ver.to_dict() if ver is not None else None,
        "reference": reference.info() if reference is not None else None,
        "rounds": al.rounds if al is not None else None, "convergence": curve, "baseline": baseline,
        "fit": fit.to_dict() if fit is not None else None,
        "study": {"passed": sr.passed, "checks": [c.model_dump() for c in sr.checks]},
        "files": files, "seconds": round(time.time() - t0, 2),
    }


rt.jobs.register("optimize", optimize_job)


# ----------------------------------------------------------------------------------------------- routes
class RunRequest(BaseModel):
    spec: dict[str, Any] = Field(description="problem spec (pinneapple_design.design_optimizer.spec)")
    method: str = Field("nsga2", description=f"one of {SEARCH_METHODS}")
    budget: int = Field(600, ge=10, le=MAX_BUDGET_DEFAULT, description="evaluations of the search model")
    seed: int = 0
    n_verify: int = Field(5, ge=0, le=20, description="candidates re-evaluated with the reference")
    verify_with: str | None = Field("library", description="'library', a configured solver name, or none")
    mode: str = Field("direct", description="'direct' or 'active_learning'")
    rounds: int = Field(3, ge=1, le=6)
    n_initial: int = Field(24, ge=8, le=80)
    upload_id: str | None = Field(None, description="table of evaluated designs (source 'table')")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "scope": scope(), "methods": {m: v for m, v in available_methods().items()
                                                              if m in SEARCH_METHODS},
            "max_budget": {**{m: MAX_BUDGET_DEFAULT for m in SEARCH_METHODS}, **MAX_BUDGET},
            "solvers": {k: {"outputs": v.get("outputs", [])} for k, v in _solvers().items()},
            "heatsink": presets.preset_info("heatsink"),
            "runtime": rt.describe()}


@app.get("/api/zoo")
def zoo_models():
    """Zoo models usable as evaluation models: inputs with validity ranges, outputs."""
    out = []
    for name in pp.zoo.list():
        try:
            m = pp.zoo.load(name)
        except Exception:  # noqa: BLE001 - a model whose weights are not available here is left out
            continue
        vr = m.card.validity_range
        out.append({"name": name, "task": m.card.task,
                    "inputs": [{"name": k, "range": vr.get(k) if isinstance(vr.get(k), (list, tuple)) else None}
                               for k in m.input_names],
                    "outputs": list(getattr(m, "output_names", []))})
    return out


@app.post("/api/table/inspect")
def inspect_table(request: Request, body: dict[str, Any]):
    """Columns of an uploaded table (numeric ones with their range) to choose inputs and outputs."""
    path = rt.upload_path(request, str(body.get("upload_id", "")))
    try:
        df, contract = read_table(path)
    except Exception as e:  # noqa: BLE001 - unreadable upload: tell the user why
        raise HTTPException(422, f"cannot read the table: {e}") from e
    cols = []
    for c in df.columns:
        x = np.asarray(df[c])
        num = np.issubdtype(x.dtype, np.number)
        cols.append({"name": str(c), "numeric": bool(num), "min": float(np.nanmin(x)) if num and len(x) else None,
                     "max": float(np.nanmax(x)) if num and len(x) else None,
                     "unique": int(len(np.unique(x[~np.isnan(x)]))) if num else None})
    hint = {}
    if contract is not None and getattr(contract, "kind", "") == "DatasetSpec":
        hint = {"inputs": contract.inputs, "outputs": contract.outputs}
    return {"rows": int(len(df)), "columns": cols, "contract_hint": hint}


@app.get("/api/examples/heatsink_doe.csv")
def example_table():
    """A worked example of 'your data': 120 heat-sink designs (Latin hypercube) evaluated with the library's closed-form
    model, as a CSV to upload."""
    p = presets.heatsink_problem("closed_form")
    X = p.space.sample(120, seed=7)
    Y = p.evaluator(X)
    df = designs_frame(p, p.record(X, Y))[p.space.names + ["mass_g", "cost", "t_source_c", "pressure_drop_pa",
                                                          "fin_stress_mpa", "fin_gap_mm"]]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="heatsink_doe.csv"'})


@app.post("/api/runs")
def create_run(request: Request, req: RunRequest):
    """Queue an optimisation. Metered: ``budget`` evaluations, one campaign and the reference runs."""
    if req.method not in SEARCH_METHODS:
        raise HTTPException(422, f"method must be one of {SEARCH_METHODS}")
    if req.budget > MAX_BUDGET.get(req.method, MAX_BUDGET_DEFAULT):
        raise HTTPException(422, f"{req.method}: budget at most {MAX_BUDGET.get(req.method)} evaluations")
    if not available_methods()[req.method]["available"]:
        raise HTTPException(422, available_methods()[req.method]["reason"])
    spec = dict(req.spec)
    if spec.get("source") == "preset" and spec.get("fidelity") == "fvm":
        raise HTTPException(422, "the FVM engine is the reference: search on 'surrogate' or 'closed_form'")
    params = req.model_dump()
    if spec.get("source") == "table":
        if not req.upload_id:
            raise HTTPException(422, "a table source needs upload_id (POST /api/uploads first)")
        params["table_path"] = str(rt.upload_path(request, req.upload_id))
        params["table_name"] = rt.uploads.get(req.upload_id).get("filename", "")
    try:
        problem, _ = problem_from_spec(spec, read_table(params["table_path"])[0] if spec.get("source") == "table"
                                       else None)
        _reference(spec, req.verify_with, problem)
    except SpecError as e:
        raise HTTPException(422, str(e)) from e
    n_ref = (req.n_initial + req.rounds * req.n_verify) if req.mode == "active_learning" else req.n_verify
    units = {"evaluations": req.budget * (req.rounds if req.mode == "active_learning" else 1), "campaigns": 1}
    if req.verify_with and req.verify_with != "none":
        units["reference_runs"] = n_ref
    return rt.submit(request, "optimize", params, units=units)


def _job_file(request: Request, job_id: str, name: str) -> Path:
    job = rt.job_for(request, job_id)
    if name not in FILES:
        raise HTTPException(404, f"files: {FILES}")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    p = rt.jobs.workdir / job_id / name
    if not p.exists():
        raise HTTPException(404, f"{name} was not produced by this run")
    return p


@app.get("/api/runs/{job_id}/files/{name}")
def download(request: Request, job_id: str, name: str):
    p = _job_file(request, job_id, name)
    return FileResponse(p, filename=f"{job_id}_{name}")


@app.post("/api/runs/{job_id}/verify")
def verify_upload(request: Request, job_id: str, body: dict[str, Any]):
    """Verify a run's candidates with solver results you ran offline: upload a table with the design variables and
    the outputs (rows matched by the variable values of candidates_to_run.csv)."""
    import pandas as pd
    job = rt.job_for(request, job_id)
    cdf = pd.read_csv(_job_file(request, job_id, "candidates.csv"))
    space = DesignSpace.from_parameter_space(json.loads(_job_file(request, job_id, "parameter_space.json")
                                                        .read_text()))
    res_path = rt.upload_path(request, str(body.get("upload_id", "")))
    try:
        rdf, _ = read_table(res_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"cannot read the results: {e}") from e
    spec = job["params"]["spec"]
    objs = objectives_from(spec.get("objectives") or [{"name": n} for n in ("mass_g", "cost")])
    cons = constraints_from(spec.get("constraints") or [])
    if not cons and spec.get("source", "preset") == "preset":
        cons = problem_from_spec(spec)[0].constraints
    try:
        rep = verify_uploaded_results(space, cdf, rdf, objs, cons)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    rt.consume(request, "reference_runs", len(cdf), ref=job_id)
    out = rep.to_dict()
    (rt.jobs.workdir / job_id / "verification_uploaded.json").write_text(json.dumps(out, indent=1, default=str))
    return out


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
