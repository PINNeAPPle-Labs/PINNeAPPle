"""Parameter Identifiability Lab: web API + single-page UI. Before estimating parameters, can these measurements
determine them at all? Fisher information rank and conditioning, local sensitivity, profile likelihood and Bayesian
posterior correlations on one model and one measurement plan, then the extra measurements that would help most
(D-optimal design).

Run:  uvicorn identifiability_lab.api:app --port 8108   (from apps/identifiability_lab)

Environment (all optional; the shared runtime adds the rest, see apps/_shared/README.md):
  IDENT_USER / IDENT_PASSWORD   HTTP Basic login on everything except /health
  IDENT_MAX_JOBS                analyses running at the same time over all workers (default 2)
  IDENT_FREE_ANALYSES           free analyses per caller and month (default 100)
  IDENT_FREE_PARAMETERS         free parameter-analyses (parameters x analyses) per month (default 600)
  IDENT_FREE_SCENARIOS          free candidate measurements evaluated by the design per month (default 5000)
  IDENT_DATA_DIR                jobs, keys, usage and uploads (default /data/identifiability)
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.inverse_problems import (
    Observations,
    Parameter,
    analyze_identifiability,
    get_preset,
    list_presets,
    model_from_spec,
)
from pinneapple_analysis.inverse_problems.presets import MODEL_TYPES

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
MAX_ROWS = 2000
MAX_PARAMS = 12

app = FastAPI(title="Parameter Identifiability Lab", version=VERSION,
              description="Before estimating parameters, check that the measurements can determine them: Fisher "
                          "information rank and conditioning, local sensitivity, profile likelihood and posterior "
                          "correlations, the parameter combinations that stay ambiguous, the value of each "
                          "measurement and the extra measurements that would help most (D-optimal design). "
                          "POST /api/analyze queues an analysis; poll GET /api/jobs/{id}.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="IDENT")
rt = install_runtime(app, prefix="IDENT", name="identifiability",
                     units={"analyses": 100, "parameters": 600, "scenarios": 5000},
                     upload_suffixes={".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json"})

PRESETS = ["fin_h_k", "fin_h_k_flux", "cooling_h_C", "cooling_h", "arrhenius_one_T", "arrhenius_two_T",
           "rod_h_field", "plate_h"]

SCOPE = {
    "title": "Scope & limits", "noun": "limit", "items_title": "What the verdicts mean",
    "validated": [
        "Known ambiguities are found: pin-fin temperatures fix only h/k, a cooling curve only h/C, one-temperature "
        "kinetics only the rate constant; the design then asks for the base heat flow or a second temperature "
        "(tests/test_inverse_core.py)",
        "Exact derivatives (the finite-volume rod's sensitivities and adjoint) agree with finite differences to 1e-6",
        "A profile that rises only because another parameter hits its bound is reported as such, not as a "
        "closed interval",
    ],
    "band": "A verdict is local to the analysis point and the bounds given; it says what these measurements can "
            "determine, not whether the model is right.",
    "tags": {"conservative": "Safe by default", "optimistic": "Can hide a problem", "check": "Assumption to confirm"},
    "items": [
        {"topic": "Local analysis", "effect": "optimistic",
         "detail": "The Fisher rank and the Cramér-Rao bounds are computed at one point (the nominal values, or the "
                   "best fit when values are given). A parameter can be identifiable here and not elsewhere.",
         "today": "Run the analysis at a few plausible points, and read the profile likelihood (not a "
                  "linearisation).",
         "planned": "Analysis over a range of nominal values."},
        {"topic": "Model form assumed right", "effect": "check",
         "detail": "Identifiability is about the parameters of this model; a missing physical effect is not "
                   "detected here.", "today": "Calibrate and validate on held-out data (Model Calibration Lab).",
         "planned": ""},
        {"topic": "Noise model", "effect": "check",
         "detail": "Independent Gaussian noise with the sigma you give per measurement. Correlated errors (a sensor "
                   "offset shared by several readings) make parameters less identifiable than reported.",
         "today": "Add the offset as a parameter, or enlarge sigma.", "planned": "Correlated noise."},
        {"topic": "Posterior and profiles", "effect": "conservative",
         "detail": "Run only for fast models with at most 6 parameters and finite bounds; the analysis says when "
                   "they were skipped.", "today": "", "planned": ""},
    ]}


def _build(body: dict):
    try:
        model = model_from_spec(body["model"])
        params = [Parameter.from_dict(p) for p in body.get("parameters") or model.describe()["parameters"]]
        obs = Observations.from_dict(body["observations"])
        cand = Observations.from_dict(body["candidates"]) if body.get("candidates") else None
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, f"invalid problem: {exc}") from None
    free = [p for p in params if not p.fixed]
    if not free:
        raise HTTPException(422, "at least one parameter must be free")
    if len(free) > MAX_PARAMS:
        raise HTTPException(422, f"at most {MAX_PARAMS} free parameters per analysis")
    if obs.n > MAX_ROWS or (cand is not None and cand.n > MAX_ROWS):
        raise HTTPException(422, f"at most {MAX_ROWS} measurements and {MAX_ROWS} candidates")
    return model, params, obs, cand


def run_analysis(params: dict, ctx) -> dict:
    """Job: the identifiability report and its StudyResult (the shared contract)."""
    model, plist, obs, cand = _build(params)
    opts = params.get("options") or {}
    ctx.progress(0.05, "Fisher information and sensitivities")
    rep = analyze_identifiability(model, obs, plist, candidates=cand, n_suggest=int(opts.get("n_suggest", 3)),
                                  profile=opts.get("profile", "auto"), posterior=opts.get("posterior", "auto"),
                                  practical_threshold=float(opts.get("practical_threshold", 0.5)),
                                  time_budget=float(opts.get("time_budget", 60)))
    ctx.progress(0.9, "writing the study")
    out = rep.to_dict()
    from pinneapple_analysis.inverse_problems.contracts import identifiability_study

    out["study"] = identifiability_study(out, plist, app="identifiability", model=model.name,
                                         study_id=getattr(ctx, "job_id", None)).to_dict()
    out["model"] = model.describe()
    out["parameters_in"] = [p.to_dict() for p in plist]
    return out


rt.jobs.register("analyze", run_analysis)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "scope": SCOPE, "runtime": rt.describe(),
            "presets": [p for p in list_presets() if p["id"] in PRESETS],
            "models": {k: f().describe() for k, f in MODEL_TYPES.items()},
            "limits": {"max_rows": MAX_ROWS, "max_free_parameters": MAX_PARAMS}}


@app.get("/api/presets/{preset_id}")
def preset(preset_id: str):
    """A ready-made problem (synthetic measurements with a known answer): model, parameters, measurements."""
    if preset_id not in PRESETS:
        raise HTTPException(404, f"unknown preset; use one of {PRESETS}")
    return get_preset(preset_id).to_dict()


EXAMPLE = {
    "model": {"type": "pin_fin", "options": {"d_mm": 5, "length_mm": 50}},
    "parameters": [{"name": "h", "value": 50, "lower": 1, "upper": 2000, "unit": "W/m²K"},
                   {"name": "k", "value": 16, "lower": 1, "upper": 500, "unit": "W/m·K"},
                   {"name": "T_base", "value": 80, "fixed": True}, {"name": "T_air", "value": 25, "fixed": True}],
    "observations": {"inputs": {"x_mm": [10, 20, 30, 40, 50]}, "sigma": [0.5] * 5, "observable": ["T"] * 5},
    "options": {"n_suggest": 3}}


@app.post("/api/analyze")
def analyze(request: Request, body: Annotated[dict, Body(examples=[EXAMPLE])]):
    """Queue an identifiability analysis. ``observations.values`` is optional: without it the analysis is done on
    the model's own predictions at the nominal values (the design stage). Returns the job; poll /api/jobs/{id}."""
    model, plist, obs, cand = _build(body)
    n_free = sum(not p.fixed for p in plist)
    n_cand = cand.n if cand is not None else 25 * len(model.outputs)
    return rt.submit(request, "analyze", body, units={"analyses": 1, "parameters": n_free, "scenarios": n_cand})


@app.post("/api/table")
async def table(file: Annotated[UploadFile, File()], sheet: Annotated[str | None, Form()] = None):
    """Read a measurement table (CSV, Excel, Parquet, JSON; sensor exports with decimal commas): its columns and the
    first rows, so the page can map columns to inputs, values, sigma and observable."""
    from pinneapple_data.data_health import load_table

    data = await file.read()
    if len(data) > rt.uploads.max_bytes:
        raise HTTPException(413, "file too large")
    try:
        df, info = load_table(data, file.filename or "table.csv", sheet=sheet, max_rows=MAX_ROWS)
    except (ValueError, ImportError) as exc:
        raise HTTPException(422, str(exc)) from None
    from pinneapple_core.jsonable import to_jsonable

    return {"columns": list(df.columns), "rows": to_jsonable(df.head(MAX_ROWS).to_dict(orient="list")),
            "n_rows": int(len(df)), "format": info.format, "notes": info.notes}


@app.post("/api/parameter-space")
def parameter_space(body: Annotated[dict, Body()]):
    """Parameters from a ParameterSpace contract (e.g. from the Campaign Planner): continuous ones free, constants
    fixed."""
    from pinneapple_analysis.inverse_problems.contracts import parameters_from_space
    from pinneapple_data.contracts import ParameterSpace

    try:
        space = ParameterSpace.from_dict(body)
        return [p.to_dict() for p in parameters_from_space(space)]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/jobs/{job_id}/download/{what}")
def download(request: Request, job_id: str, what: str):
    """``report.json`` (everything), ``study.json`` (StudyResult contract), ``parameters.csv``,
    ``suggestions.csv``."""
    job = rt.job_for(request, job_id)
    res = job.get("result")
    if job["status"] != "done" or not res:
        raise HTTPException(409, "the analysis has not finished")
    import json

    import pandas as pd

    if what == "report.json":
        return Response(json.dumps(res, indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="identifiability-{job_id}.json"'})
    if what == "study.json":
        return Response(json.dumps(res["study"], indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="study-{job_id}.json"'})
    if what == "parameters.csv":
        df = pd.DataFrame([{k: (v if not isinstance(v, list) else " .. ".join(map(str, v))) for k, v in p.items()}
                           for p in res["parameters"]])
    elif what == "suggestions.csv":
        df = pd.DataFrame([{"rank": i + 1, "measurement": s["label"], "observable": s["observable"], **s["inputs"],
                            "sigma": s["sigma"], "gain_logdet": s["gain_logdet"], "rank_after": s["rank_after"],
                            **{f"std_after_{k}": v for k, v in s["std_after"].items()}}
                           for i, s in enumerate(res["suggestions"])])
    else:
        raise HTTPException(404, "report.json, study.json, parameters.csv or suggestions.csv")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{what.replace(".csv", "")}-{job_id}.csv"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
