"""Inverse Model Studio: web API + single-page UI. One interface to formulate an inverse problem (measurements, a
physical model, unknown parameters or fields), solve it with several methods on the same data (classical least
squares, adjoint gradient, automatic differentiation, Bayesian regularisation, physics-informed networks), and
validate the answers: held-out measurements, identifiability, and recovery of a known truth on the examples.

A separate app from the Inverse Heat Lab (inverse.pinneapple.org), which stays as it is; the studio reuses its
physics (fin and plate PINNs, finite-volume references) through pinneapple_analysis.inverse_problems.

Run:  uvicorn inverse_studio.api:app --port 8104   (from apps/inverse_studio)

Environment (all optional; the shared runtime adds the rest, see apps/_shared/README.md):
  INVS_USER / INVS_PASSWORD   HTTP Basic login on everything except /health
  INVS_MAX_JOBS               inverse problems solving at the same time over all workers (default 2)
  INVS_FREE_PROBLEMS          free inverse problems per caller and month (default 50)
  INVS_FREE_METHOD_RUNS       free method runs (one per method and problem) per month (default 250)
  INVS_FREE_PINN_RUNS         free physics-informed network trainings per month (default 10)
  INVS_HEAT_LAB_URL           link to the Inverse Heat Lab (default https://inverse.pinneapple.org)
  INVS_DATA_DIR               jobs, keys, usage and uploads (default /data/inverse_studio)
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.inverse_problems import (
    METHODS,
    InverseProblem,
    Observations,
    Parameter,
    analyze_identifiability,
    available_methods,
    compare_methods,
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
MAX_ROWS = 5000
MAX_PARAMS = 16
HEAT_LAB_URL = os.environ.get("INVS_HEAT_LAB_URL", "https://inverse.pinneapple.org")

app = FastAPI(title="Inverse Model Studio", version=VERSION,
              description="Formulate an inverse problem once (measurements, a physical model, unknown parameters or "
                          "the zones of a field) and solve it with several methods on the same data: classical least "
                          "squares, adjoint gradient, automatic differentiation, Bayesian regularisation and "
                          "physics-informed networks. Estimates with uncertainty, reconstructed fields, residuals, "
                          "held-out validation, the identifiability diagnosis, and truth recovery on the examples. "
                          "POST /api/solve queues a problem; poll GET /api/jobs/{id}.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="INVS")
rt = install_runtime(app, prefix="INVS", name="inverse_studio",
                     units={"problems": 50, "method_runs": 250, "pinn_runs": 10},
                     upload_suffixes={".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json"})

PRESETS = ["rod_h_field", "fin_h", "plate_h", "rod_base_bc", "fin_h_k_flux", "fin_h_k", "cooling_h", "arrhenius_two_T"]

SCOPE = {
    "title": "Scope & validation", "noun": "limit", "items_title": "What the comparison means",
    "validated": [
        "Every example has a known answer: on the fin h(x) field, the unknown wall temperature, the fin with a "
        "heat-flux meter and the cooling curve, least squares, adjoint, autodiff and Bayesian recover the truth "
        "inside their 95 % intervals (tests/test_inverse_core.py)",
        "The adjoint gradient and the finite-volume sensitivities agree with finite differences to 1e-6; the "
        "finite-volume fin agrees with the analytic fin to 0.005 K",
        "The heat examples are the Inverse Heat Lab's (pin fin, heat-spreader plate) with the same physics-informed "
        "networks; the plate network gave h = 14.0 for a true 15 in 1500 steps, least squares on the finite-volume "
        "model 14.96: both are shown, not just the better one",
    ],
    "band": "Methods are ranked by the error on held-out measurements none of them saw; truth recovery is only "
            "checked where the truth is known (the examples).",
    "tags": {"conservative": "Safe by default", "optimistic": "Can hide a problem", "check": "Assumption to confirm"},
    "items": [
        {"topic": "Model form", "effect": "optimistic",
         "detail": "Every method fits the same model; if the model misses a physical effect, they can all agree and "
                   "all be wrong. The held-out error and the reduced chi-square are the checks.",
         "today": "Keep held-out measurements; compare the residuals of the methods.", "planned": ""},
        {"topic": "Method availability", "effect": "check",
         "detail": "The adjoint needs a discretised model with an adjoint solver (the finite-volume fin), autodiff a "
                   "PyTorch model, the PINN one of the Inverse Heat Lab formulations (h only). The page says why a "
                   "method does not apply.", "today": "", "planned": "Adjoint solvers for more models."},
        {"topic": "Uncertainty", "effect": "check",
         "detail": "Gauss-Newton intervals (least squares, adjoint, autodiff) assume local linearity; the Bayesian "
                   "interval depends on the prior; a single PINN run gives none. A non-identifiable parameter gets "
                   "no interval, except the Bayesian one, which then reflects the prior.", "today": "", "planned": ""},
        {"topic": "3D cases", "effect": "conservative",
         "detail": "The 3D block of the Inverse Heat Lab trains for about 30 minutes; it is not run here.",
         "today": "See it in the Inverse Heat Lab.", "planned": ""},
    ]}


def _obs(d):
    return Observations.from_dict(d) if d else None


def _problem(body: dict) -> InverseProblem:
    try:
        model = model_from_spec(body["model"])
        params = [Parameter.from_dict(p) for p in body.get("parameters") or model.describe()["parameters"]]
        obs = Observations.from_dict(body["observations"])
        val = _obs(body.get("validation"))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, f"invalid problem: {exc}") from None
    free = [p for p in params if not p.fixed]
    if not free:
        raise HTTPException(422, "at least one parameter must be free")
    if len(free) > MAX_PARAMS:
        raise HTTPException(422, f"at most {MAX_PARAMS} free parameters")
    if not obs.has_values:
        raise HTTPException(422, "every measurement needs a value")
    if obs.n > MAX_ROWS:
        raise HTTPException(422, f"at most {MAX_ROWS} measurements")
    opts = body.get("options") or {}
    return InverseProblem(model, obs, params, val, name=body.get("name") or "inverse problem",
                          prior_std=float(opts.get("prior_std", 1.0)), smoothness=float(opts.get("smoothness", 0.0)))


def _methods(body: dict) -> list[str]:
    ms = list(dict.fromkeys(body.get("methods") or ["least_squares"]))
    bad = [m for m in ms if m not in METHODS]
    if bad:
        raise HTTPException(422, f"unknown methods {bad}; use {list(METHODS)}")
    return ms


def run_solve(params: dict, ctx) -> dict:
    """Job: identifiability, every requested method on the same data, and the StudyResult."""
    prob = _problem(params)
    methods = _methods(params)
    opts = params.get("options") or {}
    ctx.progress(0.03, "identifiability")
    rep = analyze_identifiability(prob.model, prob.observations, prob.parameters, profile=False, posterior=False,
                                  fit_first=True)
    done = {"n": 0}

    def progress(method, step, total, value):
        frac = (done["n"] + (step / max(total, 1))) / len(methods)
        ctx.progress(0.05 + 0.9 * min(frac, 1.0), f"{method}: step {step}/{total}, h = {value:.4g}")

    out = compare_methods(prob, methods, identifiability=rep, progress=progress,
                          pinn_steps=int(min(max(opts.get("pinn_steps", 1500), 200), 3000)))
    fields = getattr(prob.model, "fields", None)
    if prob.truth is not None and callable(fields):            # examples only: the known answer, for the plots
        out["truth_fields"] = fields({**prob.params.theta(), **prob.truth})
    ctx.progress(0.97, "writing the study")
    from pinneapple_analysis.inverse_problems.contracts import inverse_study

    out["study"] = inverse_study(out, prob.parameters, app="inverse_studio",
                                 study_id=getattr(ctx, "job_id", None)).to_dict()
    return out


rt.jobs.register("solve", run_solve)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "scope": SCOPE, "runtime": rt.describe(), "methods": METHODS,
            "presets": [p for p in list_presets() if p["id"] in PRESETS],
            "models": {k: f().describe() for k, f in MODEL_TYPES.items()}, "heat_lab_url": HEAT_LAB_URL,
            "limits": {"max_rows": MAX_ROWS, "max_free_parameters": MAX_PARAMS}}


@app.get("/api/presets/{preset_id}")
def preset(preset_id: str):
    """A ready-made inverse problem with a known answer (synthetic measurements, fixed seed)."""
    if preset_id not in PRESETS:
        raise HTTPException(404, f"unknown preset; use one of {PRESETS}")
    return get_preset(preset_id).to_dict()


@app.post("/api/methods")
def methods_for(body: Annotated[dict, Body()]):
    """Which methods apply to this problem, and why not when they do not."""
    return available_methods(_problem(body))


_P = get_preset("rod_h_field")
EXAMPLE = {"model": _P.model_spec, "parameters": _P.parameters, "observations": _P.observations,
           "validation": _P.validation, "methods": ["least_squares", "adjoint", "bayesian"],
           "options": {"smoothness": 0.5}}


@app.post("/api/solve")
def solve(request: Request, body: Annotated[dict, Body(examples=[EXAMPLE])]):
    """Queue an inverse problem: ``methods`` (keys of /api/meta ``methods``) run on the same ``observations``;
    ``validation`` measurements are never fitted and rank the methods. Methods that do not apply to the model are
    reported as such. Returns the job; poll /api/jobs/{id}."""
    prob = _problem(body)
    ms = _methods(body)
    avail = available_methods(prob)
    runnable = [m for m in ms if avail[m]["available"]]
    units = {"problems": 1, "method_runs": max(1, len(runnable))}
    if "pinn" in runnable:
        units["pinn_runs"] = 1
    return rt.submit(request, "solve", body, units=units)


@app.post("/api/table")
async def table(file: Annotated[UploadFile, File()], sheet: Annotated[str | None, Form()] = None):
    """Read measurements or simulated runs (CSV, Excel, Parquet, JSON; sensor exports with decimal commas)."""
    from pinneapple_core.jsonable import to_jsonable
    from pinneapple_data.data_health import load_table

    data = await file.read()
    if len(data) > rt.uploads.max_bytes:
        raise HTTPException(413, "file too large")
    try:
        df, info = load_table(data, file.filename or "table.csv", sheet=sheet, max_rows=MAX_ROWS)
    except (ValueError, ImportError) as exc:
        raise HTTPException(422, str(exc)) from None
    return {"columns": list(df.columns), "rows": to_jsonable(df.to_dict(orient="list")), "n_rows": int(len(df)),
            "format": info.format, "notes": info.notes}


@app.post("/api/parameter-space")
def parameter_space(body: Annotated[dict, Body()]):
    """Unknowns from a ParameterSpace contract: continuous ones free (with their bounds), constants fixed."""
    from pinneapple_analysis.inverse_problems.contracts import parameters_from_space
    from pinneapple_data.contracts import ParameterSpace

    try:
        return [p.to_dict() for p in parameters_from_space(ParameterSpace.from_dict(body))]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


def _attach(content: str, name: str, media: str) -> Response:
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/jobs/{job_id}/download/{what}")
def download(request: Request, job_id: str, what: str):
    """``report.json`` (everything), ``study.json`` (StudyResult), ``estimates.csv`` (one row per method and
    parameter), ``fields.csv`` (reconstructed 1D fields per method)."""
    import pandas as pd

    job = rt.job_for(request, job_id)
    res = job.get("result")
    if job["status"] != "done" or not res:
        raise HTTPException(409, "the problem has not been solved yet")
    if what == "report.json":
        return _attach(json.dumps(res, indent=1), f"inverse-{job_id}.json", "application/json")
    if what == "study.json":
        return _attach(json.dumps(res["study"], indent=1), f"study-{job_id}.json", "application/json")
    if what == "estimates.csv":
        df = pd.DataFrame(res["study"]["table"]["columns"])
    elif what == "fields.csv":
        rows = []
        for s in res["solutions"]:
            for name, f in (s.get("fields") or {}).items():
                if f["values"] and not isinstance(f["values"][0], list):
                    rows += [{"method": s["method"], "field": name, "unit": f["unit"], f.get("axis", "x"): x, "value": v}
                             for x, v in zip(f["x"], f["values"], strict=False)]
        if not rows:
            raise HTTPException(404, "no 1D field in this problem (see report.json for 2D fields)")
        df = pd.DataFrame(rows)
    else:
        raise HTTPException(404, "report.json, study.json, estimates.csv or fields.csv")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return _attach(buf.getvalue(), f"{what.replace('.csv', '')}-{job_id}.csv", "text/csv")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
