"""Model Calibration Lab: web API + single-page UI. Fit model parameters to real measurements, with confidence
intervals only where the data determine them, before/after comparison and validation on held-out measurements.

Run:  uvicorn calibration_lab.api:app --port 8101   (from apps/calibration_lab)

Environment (all optional; the shared runtime adds the rest, see apps/_shared/README.md):
  CALIB_USER / CALIB_PASSWORD   HTTP Basic login on everything except /health
  CALIB_MAX_JOBS                calibrations running at the same time over all workers (default 2)
  CALIB_FREE_CALIBRATIONS       free calibrations per caller and month (default 50)
  CALIB_FREE_METHOD_RUNS        free extra method runs (comparison with other estimators) per month (default 200)
  CALIB_FREE_PINN_RUNS          free physics-informed network trainings per month (default 10)
  CALIB_DATA_DIR                jobs, keys, usage and uploads (default /data/calibration)
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.inverse_problems import (
    METHODS,
    InverseProblem,
    Observations,
    Parameter,
    calibrate,
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
MAX_PARAMS = 12

app = FastAPI(title="Model Calibration Lab", version=VERSION,
              description="Fit model parameters to measurements (CSV, Excel, Parquet, JSON, sensor exports) with a "
                          "built-in physical model, your formula or a table of your simulated runs. Calibrated "
                          "values with Jacobian and bootstrap confidence intervals (\"not estimable\" when the data "
                          "cannot determine a parameter), residuals, before/after comparison, validation on held-out "
                          "measurements, and the same data solved by other estimators (Bayesian, autodiff, adjoint, "
                          "PINN) ranked by held-out error. POST /api/calibrate queues a calibration.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="CALIB")
rt = install_runtime(app, prefix="CALIB", name="calibration",
                     units={"calibrations": 50, "method_runs": 200, "pinn_runs": 10},
                     upload_suffixes={".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json"})

PRESETS = ["cooling_h", "fin_h_k_flux", "fin_h_k", "plate_h", "plate_h_table", "arrhenius_two_T", "rod_base_bc",
           "fin_h"]
IDENTIFIABILITY_URL = os.environ.get("CALIB_IDENTIFIABILITY_URL", "https://identifiability.pinneapple.org")

SCOPE = {
    "title": "Scope & limits", "noun": "limit", "items_title": "What a calibrated value means here",
    "validated": [
        "On synthetic presets with a known answer the 95 % Jacobian intervals contain the truth (fin h and k with a "
        "heat-flux meter, cooling curve with an extrapolation hold-out; tests/test_inverse_core.py)",
        "Parameters the measurements cannot determine get no interval: the fin's h and k from temperatures alone, h "
        "and the heat capacity from one cooling curve (the identifiability core of the Parameter Identifiability "
        "Lab)",
        "Held-out measurements are never fitted; the before/after errors on them are computed from the same runs "
        "shown on the page",
    ],
    "band": "A calibration is as good as its model form: the intervals assume the model is right, and the "
            "reduced chi-square and the held-out error are the checks that it is.",
    "tags": {"conservative": "Safe by default", "optimistic": "Can hide a problem", "check": "Assumption to confirm"},
    "items": [
        {"topic": "Model form", "effect": "optimistic",
         "detail": "Intervals are statistical only; a missing physical effect biases every parameter. A reduced "
                   "chi-square far above 1 or a held-out error much larger than the fit error says so (warnings).",
         "today": "Hold out measurements (random, or the last ones as an extrapolation test) and read the warnings.",
         "planned": "Model-discrepancy terms."},
        {"topic": "Table of simulated runs", "effect": "optimistic",
         "detail": "Interpolating between solver runs adds an error the interval does not include (the plate example "
                   "shows it: 0.43 °C at the true h with 8 runs).",
         "today": "Use more runs near the answer, or a built-in model that runs the solver.", "planned": ""},
        {"topic": "Your own solver", "effect": "check",
         "detail": "Arbitrary code and external endpoints are not run on this server. Upload the solver's runs as a "
                   "table, or calibrate on your machine with the same library call (CallableModel).",
         "today": "See the API tab for the Python call.", "planned": "Runner integration (day 27)."},
        {"topic": "Noise", "effect": "check",
         "detail": "Independent Gaussian noise with the sigma given; robust losses (soft L1, Huber, Cauchy) for "
                   "outliers. The stated sigma is used as the real noise unless you untick it.",
         "today": "", "planned": ""},
    ]}


def _obs(d):
    return Observations.from_dict(d) if d else None


def _build(body: dict):
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
    methods = [m for m in (body.get("options") or {}).get("compare", []) if m != "least_squares"]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise HTTPException(422, f"unknown methods {bad}; use {list(METHODS)}")
    return model, params, obs, val, methods


def _curve(model, theta_before, theta_after, obs: Observations, val: Observations | None) -> dict | None:
    """Model before/after on a fine grid of the single input (1-input models), per observable."""
    if len(model.inputs) != 1:
        return None
    key = model.inputs[0]
    xs = np.concatenate([obs.inputs[key]] + ([val.inputs[key]] if val is not None and val.n else []))
    obsv = [str(o) for o in obs.observable]
    main = max(set(obsv), key=obsv.count)
    m = np.array([str(o) == main for o in obs.observable])
    lo, hi = float(np.min(xs[np.isfinite(xs)])), float(np.max(xs[np.isfinite(xs)]))
    grid = np.linspace(lo, hi, 160)
    lab = np.array([main] * len(grid), dtype=object)
    return {"input": key, "observable": main, "x": grid.tolist(),
            "before": model.predict(theta_before, {key: grid}, lab).tolist(),
            "after": model.predict(theta_after, {key: grid}, lab).tolist(), "n_main": int(m.sum())}


def run_calibration(params: dict, ctx) -> dict:
    """Job: calibration (+ optional comparison with other estimators) and its StudyResult."""
    model, plist, obs, val, methods = _build(params)
    opts = params.get("options") or {}
    hold = params.get("holdout") or {}
    mode = hold.get("mode", "validation" if val is not None else "none")
    ctx.progress(0.05, "identifiability and least squares")
    kw = dict(loss=opts.get("loss", "linear"), bootstrap=int(min(max(opts.get("bootstrap", 100), 0), 500)),
              confidence=float(opts.get("confidence", 0.95)), absolute_sigma=bool(opts.get("absolute_sigma", True)),
              time_budget=float(opts.get("time_budget", 60)))
    if mode == "validation" and val is not None:
        res = calibrate(model, obs, plist, holdout=val, **kw)
    elif mode in ("random", "last"):
        res = calibrate(model, obs, plist, holdout_fraction=float(hold.get("fraction", 0.25)),
                        holdout_by=hold.get("by") if mode == "last" else None, **kw)
    else:
        res = calibrate(model, obs, plist, **kw)
    out = res.to_dict()
    after = {p.name: p.value for p in plist}
    after.update(res.theta)
    before = {p.name: p.value for p in plist}
    rows = res.rows
    fit_mask = np.array([s == "fit" for s in rows["set"]])

    def subset(set_name):
        m = np.array([s == set_name for s in rows["set"]])
        if not m.any():
            return None
        return Observations({k: np.asarray(v, float)[m] for k, v in rows["inputs"].items()},
                            np.asarray(rows["sigma"])[m], np.asarray(rows["observed"])[m],
                            np.asarray(rows["observable"], dtype=object)[m])

    fit_o, hold_o = subset("fit"), subset("holdout")
    out["curve"] = _curve(model, before, after, fit_o, hold_o)
    fields = getattr(model, "fields", None)
    if callable(fields):
        out["fields"] = {"before": fields(before), "after": fields(after)}
    if methods:
        ctx.progress(0.5, f"comparing with {', '.join(methods)}")
        prob = InverseProblem(model, fit_o, [Parameter.from_dict(p.to_dict()) for p in plist], hold_o,
                              name="calibration", prior_std=float(opts.get("prior_std", 1.0)))
        out["comparison"] = compare_methods(prob, ["least_squares"] + methods,
                                            pinn_steps=int(min(max(opts.get("pinn_steps", 1500), 200), 3000)))
    ctx.progress(0.95, "writing the study")
    from pinneapple_analysis.inverse_problems.contracts import calibration_study

    out["study"] = calibration_study(out, plist, app="calibration", model=model.name, model_spec=params["model"],
                                     study_id=getattr(ctx, "job_id", None)).to_dict()
    out["model"] = model.describe()
    out["holdout_mode"] = mode
    out["n_fit"] = int(fit_mask.sum())
    return out


rt.jobs.register("calibrate", run_calibration)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "scope": SCOPE, "runtime": rt.describe(), "methods": METHODS,
            "presets": [p for p in list_presets() if p["id"] in PRESETS],
            "models": {k: f().describe() for k, f in MODEL_TYPES.items()},
            "identifiability_url": IDENTIFIABILITY_URL, "limits": {"max_rows": MAX_ROWS, "max_free_parameters": MAX_PARAMS}}


@app.get("/api/presets/{preset_id}")
def preset(preset_id: str):
    """A ready-made calibration (synthetic measurements from a known truth): model, parameters, measurements and
    held-out measurements."""
    if preset_id not in PRESETS:
        raise HTTPException(404, f"unknown preset; use one of {PRESETS}")
    return get_preset(preset_id).to_dict()


EXAMPLE = {
    "model": {"type": "lumped_cooling", "options": {"area_m2": 0.012}},
    "parameters": [{"name": "h", "value": 30, "lower": 0.5, "upper": 500, "unit": "W/m²K"},
                   {"name": "C", "value": 450, "lower": 1, "upper": 1e5, "unit": "J/K", "fixed": True},
                   {"name": "T0", "value": 80, "lower": -50, "upper": 1000, "unit": "°C", "log": False},
                   {"name": "T_air", "value": 30, "lower": -50, "upper": 500, "unit": "°C", "log": False}],
    "observations": {"inputs": {"t_s": [0, 300, 600, 900, 1200, 1500, 1800]}, "sigma": [0.3] * 7,
                     "values": [94.0, 84.3, 75.6, 68.1, 61.4, 55.6, 50.6]},
    "holdout": {"mode": "last", "fraction": 0.25, "by": "t_s"},
    "options": {"bootstrap": 100, "compare": ["bayesian"]}}


@app.post("/api/calibrate")
def calibrate_route(request: Request, body: Annotated[dict, Body(examples=[EXAMPLE])]):
    """Queue a calibration. ``holdout.mode``: ``"validation"`` (the ``validation`` measurements), ``"random"`` (a
    fraction of the rows), ``"last"`` (the rows with the largest ``holdout.by`` input: an extrapolation test) or
    ``"none"``. ``options.compare``: other estimators to run on the same data (see /api/meta ``methods``)."""
    model, plist, obs, val, methods = _build(body)
    units = {"calibrations": 1}
    if methods:
        units["method_runs"] = len(methods)
    if "pinn" in methods:
        units["pinn_runs"] = 1
    return rt.submit(request, "calibrate", body, units=units)


@app.post("/api/table")
async def table(file: Annotated[UploadFile, File()], sheet: Annotated[str | None, Form()] = None):
    """Read a measurement table or a table of simulated runs (CSV, Excel, Parquet, JSON; sensor exports with decimal
    commas): its columns and rows, for the page to map columns."""
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
    """Parameters from a ParameterSpace contract: continuous ones free (with their bounds), constants fixed."""
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
    """``report.json``, ``study.json`` (StudyResult), ``model_ref.json`` (ModelRef of the calibrated model, when there
    were held-out measurements), ``parameters.csv``, ``residuals.csv``."""
    import pandas as pd

    job = rt.job_for(request, job_id)
    res = job.get("result")
    if job["status"] != "done" or not res:
        raise HTTPException(409, "the calibration has not finished")
    if what == "report.json":
        return _attach(json.dumps(res, indent=1), f"calibration-{job_id}.json", "application/json")
    if what == "study.json":
        return _attach(json.dumps(res["study"], indent=1), f"study-{job_id}.json", "application/json")
    if what == "model_ref.json":
        models = res["study"].get("models") or []
        if not models:
            raise HTTPException(404, "no ModelRef: it needs held-out measurements (its metrics must be held-out)")
        return _attach(json.dumps(models[0], indent=1), f"model-ref-{job_id}.json", "application/json")
    if what == "parameters.csv":
        df = pd.DataFrame([{k: (" .. ".join(map(str, v)) if isinstance(v, list) else v) for k, v in p.items()}
                           for p in res["parameters"]])
    elif what == "residuals.csv":
        r = res["rows"]
        df = pd.DataFrame({**r["inputs"], "observable": r["observable"], "set": r["set"], "observed": r["observed"],
                           "sigma": r["sigma"], "model_before": r["before"], "model_after": r["after"]})
        df["residual_after"] = df["observed"] - df["model_after"]
    else:
        raise HTTPException(404, "report.json, study.json, model_ref.json, parameters.csv or residuals.csv")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return _attach(buf.getvalue(), f"{what.replace('.csv', '')}-{job_id}.csv", "text/csv")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
