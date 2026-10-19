"""Engineering Sensitivity Lab: web API + single-page UI. Which parameters really control the result?

Three ways in:
- a model (pp.zoo model, built-in benchmark with exact indices, or an uploaded ONNX graph): Morris screening, Sobol
  first/total/second-order indices with bootstrap intervals, effect curves (PDP + ICE), local derivatives by autodiff
  when the model is differentiable, and the "parameters you can freeze" recommendation with its direct check;
- runs done elsewhere (OpenFOAM, CalculiX, a campaign runner): download the Saltelli design, run it, upload the results;
- a dataset (experiments, past simulations): first-order indices from the data, with the input correlation reported.

Every number comes from pinneapple_analysis.sensitivity.

Run:  uvicorn sensitivity_lab.api:app --port 8099   (from apps/sensitivity_lab)

Studies run in the shared job queue (apps/_shared, pinneapple_core.service): POST /api/study, /api/runs or
/api/given-data returns a job; poll GET /api/jobs/{id} (or pass ?wait=<seconds>). Inputs and outputs use the shared
contracts: a ParameterSpace for the parameters, a ModelRef for the model, a StudyResult in every result.

Environment (all optional; the runtime variables are in apps/_shared/README.md):
  SEN_USER / SEN_PASSWORD   HTTP Basic login on everything except /health
  SEN_MAX_JOBS              studies running at once (default 2); SEN_MAX_HEAVY caps the synchronous examples
  SEN_FREE_EVALUATIONS / SEN_FREE_PARAMETERS   free model evaluations / varied parameters per caller and month
  SEN_MAX_MB                largest upload in MB (default 50)
  SEN_MAX_EVALS             model evaluations per study (default 400000)
  SEN_MAX_PARAMS            varied parameters per study (default 30)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.sensitivity import (
    first_order_given_data,
    freeze_recommendation,
    saltelli_design,
    sensitivity_study,
    sobol_from_runs,
    study_cost,
)
from pinneapple_analysis.study_contracts import (
    is_parameter_space,
    model_from_ref,
    sensitivity_result,
    space_from_contract,
    space_to_contract,
)
from pinneapple_analysis.study_models import StudyModel, builtin, columns
from pinneapple_analysis.uncertainty.distributions import as_space
from pinneapple_core.jsonable import to_jsonable
from pinneapple_core.service import UploadRejected, UploadTooLarge
from pinneapple_data.data_health import load_table

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
MAX_MB = float(os.environ.get("SEN_MAX_MB") or "50")
MAX_EVALS = int(os.environ.get("SEN_MAX_EVALS") or "400000")
MAX_PARAMS = int(os.environ.get("SEN_MAX_PARAMS") or "30")

ZOO = {"heatsink_forced_mlp": ["r_base_k_w", "pressure_drop_pa"], "airfoil_cst_rans_mlp": ["Cl", "Cd", "Cm"]}
BUILTIN = ("ishigami", "sobol_g")

app = FastAPI(title="Engineering Sensitivity Lab", version=VERSION,
              description="Which parameters really control the result: Sobol indices with confidence intervals, Morris "
                          "screening, interactions, effect curves, local derivatives and the parameters you can "
                          "freeze, for a model (POST /api/study), runs done elsewhere (/api/design + /api/runs) or a "
                          "dataset (/api/given-data).")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="SEN")
TABLES = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl"}
# metered per caller and calendar month (free defaults; SEN_FREE_<UNIT> overrides, "unlimited" lifts)
rt = install_runtime(app, prefix="SEN", name="sensitivity",
                     units={"studies": 200, "evaluations": 5_000_000, "parameters": 2000},
                     upload_suffixes=TABLES | {".onnx"}, upload_max_mb=MAX_MB)
_HEAVY = BusyLimiter("SEN_MAX_HEAVY")       # the synchronous examples


def _u(name, low, high, **kw):
    return {"name": name, "dist": "uniform", "low": low, "high": high, **kw}


HS_SPACE = [_u("base_width_mm", 90, 110), _u("base_depth_mm", 90, 110), _u("base_thickness_mm", 4, 8),
            _u("n_fins", 15, 21), _u("fin_thickness_mm", 1.0, 1.4), _u("fin_height_mm", 30, 40),
            _u("k_w_mk", 180, 210), _u("t_ambient_c", 20, 45),
            {"name": "air_velocity_m_s", "dist": "normal", "mean": 2.5, "std": 0.3}, _u("power_w", 80, 160)]

EXAMPLES = {
    "ishigami": {
        "title": "Ishigami function (exact answer known)",
        "description": "The standard benchmark: 3 inputs, a strong interaction between x1 and x3, x3 with no main "
                       "effect. The estimated indices are compared with the exact ones.",
        "config": {"model": {"source": "builtin", "name": "ishigami"}, "parameters": builtin("ishigami").metadata["space"],
                   "n_sobol": 2048, "second_order": True, "r_morris": 30, "n_grid": 15, "n_background": 64,
                   "threshold": 0.01, "nominal": {"x1": 1.0, "x2": 1.0, "x3": 1.0}}},
    "heatsink": {
        "title": "Heat-sink design: 10 parameters, 2 responses",
        "description": "Zoo model heatsink_forced_mlp around a 100 x 100 mm design: which of geometry, material, "
                       "airflow, ambient and power control the thermal resistance and the pressure drop, and which can "
                       "be frozen. Derivatives by automatic differentiation.",
        "config": {"model": {"source": "zoo", "name": "heatsink_forced_mlp"}, "parameters": HS_SPACE,
                   "fixed": {"source_width_mm": 40.0, "source_depth_mm": 40.0}, "outputs": ["r_base_k_w",
                                                                                          "pressure_drop_pa"],
                   "n_sobol": 1024, "second_order": True, "r_morris": 20, "n_grid": 15, "n_background": 48,
                   "threshold": 0.01}},
    "dataset": {
        "title": "From data only: 3,900 heat-sink simulations",
        "description": "No model: first-order indices straight from the heatsink_forced_convection training data "
                       "(12 inputs). Shows what given-data indices can and cannot say when inputs are correlated.",
        "config": {"mode": "given_data", "dataset": "heatsink_forced_convection", "split": "train",
                   "outputs": ["r_base_k_w", "pressure_drop_pa"]}},
}

VALIDATION = [
    "Sobol first-order, total and second-order estimates match the exact Ishigami indices (Sobol & Levitan 1999) "
    "within 0.02, with the exact values inside the bootstrap intervals (tests/test_sensitivity.py)",
    "Sobol' G function (a = 0, 1, 4.5, 9, 99, 99): first-order and total indices within 0.02 of the exact values; "
    "x5 and x6 recommended for freezing and the re-run check confirms it",
    "Morris on a linear model returns exactly |coefficients| with zero spread; ranks the G-function inputs correctly",
    "Local derivatives: automatic differentiation and central differences agree with the analytic Ishigami gradient "
    "(1e-5) and with each other on the heat-sink zoo model",
    "Given-data first-order indices of a random Ishigami sample within 0.02 of the exact values",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the indices mean",
            "band": "Indices describe the model over the ranges you give, not the physics everywhere.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can mislead", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Ranges and distributions", "effect": "check",
                 "detail": "Sobol indices are shares of output variance for the input distributions given; widen a "
                           "range and that input matters more.",
                 "today": "Use the real manufacturing / operating spread of each parameter.", "planned": ""},
                {"topic": "Independent inputs", "effect": "optimistic",
                 "detail": "The designed studies sample inputs independently. Given-data indices of correlated inputs "
                           "can add up to more than 1 and credit one input with another's effect.",
                 "today": "Read the input-correlation warning; prefer a designed study.",
                 "planned": "Shapley effects for correlated inputs."},
                {"topic": "Freeze recommendation", "effect": "conservative",
                 "detail": "Only when the upper bound of the total index is below the threshold for every output, then "
                           "checked by re-running with the parameter fixed.",
                 "today": "Inconclusive means more samples, not freeze.", "planned": ""},
                {"topic": "Sample size", "effect": "optimistic",
                 "detail": "At N = 256-512 the bootstrap intervals can be narrower than the real error (Ishigami at "
                           "N = 512: total index of x1 0.45, interval 0.37-0.55, exact 0.56).",
                 "today": "Use N >= 1024 for decisions; the cost is shown before running.", "planned": ""},
                {"topic": "Surrogate models", "effect": "check",
                 "detail": "The indices are those of the model; a surrogate can carry spurious small dependencies "
                           "(e.g. on power for a resistance) or miss real ones outside its data.",
                 "today": "Check the model's own validation (Uncertainty Lab) first.", "planned": ""},
            ]}


# ---------------------------------------------------------------------------------------------- helpers
def _model(spec: dict, onnx: bytes | None = None, onnx_name: str = "model.onnx") -> StudyModel:
    """A study model from ``{"source": builtin|zoo|onnx, ...}`` or a ModelRef contract (``kind: ModelRef``)."""
    if spec.get("kind") == "ModelRef":
        try:
            return model_from_ref(spec, onnx=onnx)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    src = spec.get("source", "builtin")
    if src == "builtin":
        if spec.get("name") not in BUILTIN:
            raise HTTPException(422, f"built-in model must be one of {', '.join(BUILTIN)}")
        return builtin(spec["name"])
    if src == "zoo":
        if spec.get("name") not in ZOO:
            raise HTTPException(422, f"zoo model must be one of {', '.join(ZOO)}")
        return StudyModel.from_zoo(spec["name"], outputs=ZOO[spec["name"]])
    if src == "onnx":
        if onnx is None:
            raise HTTPException(422, "Upload the ONNX file (field 'model', or model_upload from /api/uploads).")
        return StudyModel.from_onnx(onnx, spec.get("input_names") or None, spec.get("output_names") or None,
                                    name=onnx_name)
    raise HTTPException(422, "model.source must be 'builtin', 'zoo' or 'onnx' (or give a ModelRef).")


def _cost(cfg: dict, d: int) -> dict:
    return study_cost(d, int(cfg.get("n_sobol", 1024)), bool(cfg.get("second_order", False)),
                      int(cfg.get("r_morris", 20)), int(cfg.get("n_grid", 15)), int(cfg.get("n_background", 64)),
                      morris_on=cfg.get("morris", True), effects_on=cfg.get("effects", True))


def _normalize(cfg: dict) -> dict:
    """Read the shared contracts: ``parameter_space`` (ParameterSpace) -> parameters + fixed values,
    ``model_ref`` (ModelRef) -> model."""
    cfg = dict(cfg)
    ps = cfg.pop("parameter_space", None)
    if ps is not None or is_parameter_space(cfg.get("parameters")):
        try:
            params, fixed = space_from_contract(ps if ps is not None else cfg["parameters"])
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"parameter space: {e}") from e
        cfg["parameters"] = params
        cfg["fixed"] = {**fixed, **(cfg.get("fixed") or {})}
    if cfg.get("model_ref"):
        cfg["model"] = {**cfg.pop("model_ref"), "kind": "ModelRef"}
    return cfg


def _space(cfg):
    params = cfg.get("parameters") or []
    if not params:
        raise HTTPException(422, "Give at least one varied parameter (name, dist and its parameters).")
    if len(params) > MAX_PARAMS:
        raise HTTPException(413, f"{len(params)} parameters; the limit is {MAX_PARAMS}.")
    try:
        return as_space(params)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _check_cost(cfg: dict) -> tuple[list, dict]:
    space = _space(cfg)
    cost = _cost(cfg, len(space))
    if cost["total"] > MAX_EVALS:
        raise HTTPException(413, f"{cost['total']:,} model evaluations; the limit is {MAX_EVALS:,} per study "
                                 "(lower the Sobol sample size, the effect grid or the number of parameters).")
    return space, cost


class _Progress:
    """Job progress, or nothing for the synchronous examples."""

    def __init__(self, ctx=None):
        self.ctx = ctx

    def __call__(self, fraction: float, message: str):
        if self.ctx is not None:
            self.ctx.progress(fraction, message)


def run_model_study(cfg: dict, onnx: bytes | None = None, onnx_name: str = "model.onnx", ctx=None) -> dict:
    t0 = time.time()
    progress = _Progress(ctx)
    cfg = _normalize(cfg)
    space, cost = _check_cost(cfg)
    progress(0.05, f"loading the model; {cost['total']:,} evaluations planned")
    try:
        model = _model(cfg.get("model") or {}, onnx, onnx_name)
        exact = model.metadata.get("exact") if model.kind == "function" else None
        progress(0.1, "Sobol, Morris, effect curves, derivatives, freeze check")
        out = sensitivity_study(model, space, cfg.get("outputs") or None, n_sobol=int(cfg.get("n_sobol", 1024)),
                                second_order=bool(cfg.get("second_order", False)),
                                r_morris=int(cfg.get("r_morris", 20)), n_grid=int(cfg.get("n_grid", 15)),
                                n_background=int(cfg.get("n_background", 64)),
                                threshold=float(cfg.get("threshold", 0.01)), nominal=cfg.get("nominal") or None,
                                fixed=cfg.get("fixed") or None, morris_on=cfg.get("morris", True),
                                effects_on=cfg.get("effects", True), exact=exact, seed=int(cfg.get("seed", 0)))
    except HTTPException:
        raise
    except (ValueError, KeyError, ImportError) as e:
        raise HTTPException(422, str(e)) from e
    progress(0.95, "writing the StudyResult")
    out["mode"] = "model"
    out["cost"] = cost
    out["usage"] = {"parameters": len(space), "model_evaluations": out["n_evals"]["total"],
                    "seconds": round(time.time() - t0, 2)}
    out["config"] = cfg
    out["scope"] = scope()
    study_id = f"sensitivity-{uuid.uuid4().hex[:12]}"
    out["study_result"] = sensitivity_result(out, model, study_id, title=f"Sensitivity of {model.name}").to_dict()
    return to_jsonable(out)


def _design_spec(cfg: dict) -> dict:
    cfg = _normalize(cfg)
    space = _space(cfg)
    n = int(cfg.get("n_sobol", 256))
    so = bool(cfg.get("second_order", False))
    seed = int(cfg.get("seed", 0))
    runs = n * ((2 * len(space) + 2) if so else (len(space) + 2))
    if n < 1 or runs > MAX_EVALS:
        raise HTTPException(422, f"n_sobol={n} gives {runs} runs: this server designs up to {MAX_EVALS}")
    return {"parameters": [p.to_dict() for p in space], "n_sobol": n, "second_order": so, "seed": seed,
            "outputs": cfg.get("outputs") or [], "threshold": float(cfg.get("threshold", 0.01))}


def _table(name: str, data: bytes):
    try:
        df, _ = load_table(data, name)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return df


def run_runs(spec: dict, name: str, data: bytes, ctx=None) -> dict:
    t0 = time.time()
    progress = _Progress(ctx)
    df = _table(name, data)
    if "run_id" not in df.columns:
        raise HTTPException(422, "The results need a run_id column (the row number of the design).")
    params = [p["name"] for p in spec.get("parameters", [])]
    outs = spec.get("outputs") or [c for c in df.columns if c != "run_id" and c not in params]
    if not outs:
        raise HTTPException(422, "No output columns in the results.")
    progress(0.2, f"Sobol indices from {len(df)} runs")
    try:
        s = sobol_from_runs(spec["parameters"], int(spec["n_sobol"]), df["run_id"].to_numpy(),
                            columns(df, outs), second_order=bool(spec.get("second_order", False)),
                            seed=int(spec.get("seed", 0)), outputs=outs)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    out = {"mode": "runs", "outputs": outs, "parameters": spec["parameters"], "sobol": s.to_dict(),
           "threshold": float(spec.get("threshold", 0.01))}
    out["sobol"]["tables"] = {o: s.table(k) for k, o in enumerate(outs)}
    out["sobol"]["ranking"] = {o: s.ranking(k) for k, o in enumerate(outs)}
    out["freeze"] = {"threshold": out["threshold"], "rows": freeze_recommendation(s, out["threshold"]),
                     "note": "No direct re-run check: the model is not available here. Re-run your solver with the "
                             "recommended parameters frozen to confirm."}
    out["n_evals"] = {"sobol": s.n_evals, "total": s.n_evals}
    out["usage"] = {"parameters": len(params), "model_evaluations": s.n_evals, "seconds": round(time.time() - t0, 2)}
    out["scope"] = scope()
    out["study_result"] = sensitivity_result(out, None, f"sensitivity-{uuid.uuid4().hex[:12]}",
                                             title=f"Sensitivity from solver runs ({name})").to_dict()
    return to_jsonable(out)


def run_given_data(cfg: dict, name: str | None = None, data: bytes | None = None, ctx=None) -> dict:
    t0 = time.time()
    progress = _Progress(ctx)
    if data is not None:
        df = _table(name or "data.csv", data)
        src = name
    elif cfg.get("dataset"):
        from pinneapple_hub.zoo import datasets
        try:
            df = datasets.load(cfg["dataset"], split=cfg.get("split", "train")).to_pandas()
        except Exception as e:  # noqa: BLE001 - unknown dataset or split: report it
            raise HTTPException(422, str(e)) from e
        src = f"pp.datasets {cfg['dataset']} ({cfg.get('split', 'train')})"
    else:
        raise HTTPException(422, "Upload a data table (field 'data') or name a pp.datasets dataset.")
    outs = cfg.get("outputs") or []
    if not outs:
        raise HTTPException(422, "Name the output columns (outputs).")
    ins = cfg.get("inputs") or [c for c in df.columns if c not in outs and np.issubdtype(df[c].dtype, np.number)
                                and df[c].nunique() > 1 and c not in ("run_id",)]
    if cfg.get("dataset") == "heatsink_forced_convection" and not cfg.get("inputs"):
        ins = ["base_width_mm", "base_depth_mm", "base_thickness_mm", "n_fins", "fin_thickness_mm", "fin_height_mm",
               "k_w_mk", "source_width_mm", "source_depth_mm", "t_ambient_c", "air_velocity_m_s", "power_w"]
    try:
        X = columns(df, ins)
        res = {}
        for k, o in enumerate(outs):
            progress(0.1 + 0.8 * k / len(outs), f"first-order indices of {o}")
            y = columns(df, [o])[:, 0]
            ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            r = first_order_given_data(X[ok], y[ok], names=ins, n_boot=min(int(cfg.get("n_boot", 100)), 1000),
                                       seed=int(cfg.get("seed", 0)))
            r["sum_S1"] = float(np.sum(r["S1"]))
            r["dropped_rows"] = int((~ok).sum())
            res[o] = r
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    first = res[outs[0]]
    out = {"mode": "given_data", "source": src, "inputs": ins, "outputs": outs, "given_data": res,
           "n_rows": int(len(df)), "max_input_correlation": first["max_input_correlation"],
           "usage": {"parameters": len(ins), "rows": int(len(df)), "model_evaluations": 0,
                     "seconds": round(time.time() - t0, 2)},
           "config": cfg, "scope": scope()}
    out["study_result"] = _given_data_result(out)
    return to_jsonable(out)


def _given_data_result(out: dict) -> dict:
    from pinneapple_data.contracts import CheckRecord, StudyResult
    mc = out["max_input_correlation"]
    sums = {o: out["given_data"][o]["sum_S1"] for o in out["outputs"]}
    metrics = {"rows": out["n_rows"], "max_input_correlation": None if mc is None else mc["r"]}
    for o in out["outputs"]:
        g = out["given_data"][o]
        top = max(range(len(g["S1"])), key=lambda i: g["S1"][i])
        metrics[f"top_input[{o}]"] = g["names"][top]
        metrics[f"sum_first_order[{o}]"] = sums[o]
    indep = mc is None or abs(mc["r"]) <= 0.3
    checks = [CheckRecord(name="inputs close to independent (given-data indices assume it)",
                          status="PASS" if indep else "WARN", value=None if mc is None else abs(mc["r"]),
                          threshold=0.3, evidence="" if mc is None else f"largest |r|: {' x '.join(mc['pair'])}")]
    sr = StudyResult(id=f"sensitivity-{uuid.uuid4().hex[:12]}", study_type="sensitivity", app="sensitivity",
                     status="done", title=f"First-order indices from data ({out['source']})", metrics=metrics,
                     checks=checks, limits=["first-order indices only: totals and interactions need a designed study",
                                            "correlated inputs share credit; the indices are not a variance split"])
    return sr.to_dict()


def _json(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError as e:
        raise HTTPException(422, f"config is not valid JSON: {e}") from e


# ---------------------------------------------------------------------------------------------- jobs
def _upload_bytes(upload_id: str | None) -> tuple[str | None, bytes | None]:
    if not upload_id:
        return None, None
    meta = rt.uploads.get(upload_id)
    return meta.get("filename") or meta.get("name") or upload_id, rt.uploads.path(upload_id).read_bytes()


def job_study(params: dict, ctx) -> dict:
    name, data = _upload_bytes(params.get("model_upload"))
    return run_model_study(params["config"], data, name or "model.onnx", ctx)


def job_runs(params: dict, ctx) -> dict:
    name, data = _upload_bytes(params["runs_upload"])
    return run_runs(params["spec"], name, data, ctx)


def job_given_data(params: dict, ctx) -> dict:
    name, data = _upload_bytes(params.get("data_upload"))
    return run_given_data(params["config"], name, data, ctx)


rt.jobs.register("study", job_study)
rt.jobs.register("runs", job_runs)
rt.jobs.register("given_data", job_given_data)


async def _save(request: Request, f: UploadFile | None, given: str | None) -> str | None:
    """An upload id: the file of this request (stored in the shared upload store) or one uploaded before."""
    if f is not None and f.filename:
        try:
            return rt.uploads.save(f.filename, f.file, owner=principal(request))["id"]
        except UploadTooLarge as e:
            raise HTTPException(413, str(e)) from None
        except UploadRejected as e:
            raise HTTPException(415, str(e)) from None
    if given:
        rt.upload_path(request, given)            # 404 unless the caller owns it
        return given
    return None


def _submit(request: Request, kind: str, params: dict, units: dict, wait: float) -> dict:
    job = rt.submit(request, kind, params, units=units)
    if wait > 0:
        job = rt.jobs.wait(job["id"], timeout=min(float(wait), 600.0))
    return job


# ---------------------------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "examples": [{"name": k, "title": v["title"], "description": v["description"]}
                                             for k, v in EXAMPLES.items()],
            "zoo": [{"name": k, "outputs": v} for k, v in ZOO.items()], "builtin": list(BUILTIN),
            "limits": {"max_mb": MAX_MB, "max_evals": MAX_EVALS, "max_params": MAX_PARAMS},
            "runtime": rt.describe(), "scope": scope()}


@app.get("/api/model/{source}/{name}")
def model_info(source: str, name: str):
    """Inputs, outputs, units, validity range and a default parameter space (uniform on the validity range or a
    +-10 % band around the data median) for a built-in or zoo model; also as a ParameterSpace contract."""
    m = _model({"source": source, "name": name})
    info = m.info()
    if m.kind == "function":
        info["space"] = m.metadata["space"]
        info["fixed"] = {}
    else:
        X, _ = m.reference("test")
        med = np.median(X, axis=0)
        space, fixed = [], {}
        for i, n in enumerate(m.input_names):
            if n in m.bounds:
                lo, hi = m.bounds[n]
                c, w = med[i], 0.1 * abs(med[i]) if med[i] else 0.1 * (hi - lo)
                space.append(_u(n, round(max(lo, c - w), 6), round(min(hi, c + w), 6)))
            else:
                fixed[n] = float(np.round(med[i], 6))
        info["space"], info["fixed"] = space, fixed
    info["parameter_space"] = space_to_contract(info["space"], name=f"{m.name} parameters", fixed=info["fixed"],
                                                units=m.units).to_dict()
    return to_jsonable(info)


@app.post("/api/cost")
def api_cost(config: dict):
    """Model evaluations a study would take, before running it."""
    return _cost(config, len(_space(_normalize(config))))


# FastAPI keeps per-parameter state on these objects: one object per parameter, never shared
_CONFIG = Form(..., description="study configuration (JSON, see /api/example/heatsink/config)")
_GD_CONFIG = Form(..., description="{inputs, outputs} or {dataset, split, outputs} (JSON)")
_SPEC = Form(..., description="the JSON on the first line of the design CSV")
_MODEL_FILE, _DATA_FILE, _RUNS_FILE = File(None), File(None), File(None)


@app.post("/api/study")
async def api_study(request: Request, config: str = _CONFIG, model: UploadFile | None = _MODEL_FILE,
                    wait: float = 0):
    """Sensitivity study of a model, as a job. ``config``: {model: {source: builtin|zoo|onnx, name, input_names,
    output_names} or model_ref (ModelRef), parameters: [{name, dist, ...}] or parameter_space (ParameterSpace),
    fixed: {input: value}, outputs, n_sobol, second_order, r_morris, n_grid, n_background, threshold, nominal, morris,
    effects, model_upload (an /api/uploads id)}. Returns the job; ``?wait=<s>`` waits for it."""
    cfg = _normalize(_json(config))
    space, cost = _check_cost(cfg)
    upload = await _save(request, model, cfg.pop("model_upload", None))
    if (cfg.get("model") or {}).get("source") == "onnx" and not upload:
        raise HTTPException(422, "Upload the ONNX file (field 'model', or model_upload from /api/uploads).")
    return await run_in_threadpool(_submit, request, "study", {"config": cfg, "model_upload": upload},
                   {"studies": 1, "evaluations": cost["total"], "parameters": len(space)}, wait)


@app.post("/api/design")
def api_design(config: dict):
    """Saltelli design to run elsewhere (OpenFOAM, CalculiX, the Campaign Runner): CSV with run_id and one column per
    parameter. Its first line holds the spec to upload the results with to /api/runs; the parameters are also returned
    as a ParameterSpace contract (header X-Parameter-Space)."""
    spec = _design_spec(config)
    X, layout = saltelli_design(spec["parameters"], spec["n_sobol"], spec["second_order"], spec["seed"])
    names = layout["names"]
    buf = io.StringIO()
    buf.write("# pinneapple.sensitivity.design/1 " + json.dumps(spec) + "\n")
    buf.write("run_id," + ",".join(names) + "\n")
    for i, row in enumerate(X):
        buf.write(f"{i}," + ",".join(f"{v:.10g}" for v in row) + "\n")
    ps = space_to_contract(spec["parameters"], name="Sobol design").to_json(indent=None)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="sobol_design.csv"',
                             "X-Runs": str(X.shape[0]), "X-Parameter-Space": ps})


@app.post("/api/runs")
async def api_runs(request: Request, spec: str = _SPEC, runs: UploadFile | None = _RUNS_FILE, wait: float = 0):
    """Sobol indices from the runs of a design, as a job: ``spec`` is the JSON on the design's first line, ``runs`` a
    table with ``run_id`` and one column per output (extra columns are ignored), or ``spec.runs_upload``."""
    cfg = _json(spec)
    upload = await _save(request, runs, cfg.pop("runs_upload", None))
    if not upload:
        raise HTTPException(422, "Upload the results table (field 'runs').")
    for k in ("parameters", "n_sobol"):
        if k not in cfg:
            raise HTTPException(422, f"the spec has no {k!r}: use the first line of the design CSV")
    n = len(_space(cfg))
    return await run_in_threadpool(_submit, request, "runs", {"spec": cfg, "runs_upload": upload}, {"studies": 1, "parameters": n}, wait)


@app.post("/api/given-data")
async def api_given_data(request: Request, config: str = _GD_CONFIG, data: UploadFile | None = _DATA_FILE,
                         wait: float = 0):
    """First-order indices from a dataset, as a job: ``config`` {inputs: [...], outputs: [...]} (or {dataset, split}
    for a pp.datasets table, or data_upload), ``data`` a CSV / Parquet / JSON table."""
    cfg = _json(config)
    upload = await _save(request, data, cfg.pop("data_upload", None))
    if not upload and not cfg.get("dataset"):
        raise HTTPException(422, "Upload a data table (field 'data') or name a pp.datasets dataset.")
    if not cfg.get("outputs"):
        raise HTTPException(422, "Name the output columns (outputs).")
    return await run_in_threadpool(_submit, request, "given_data", {"config": cfg, "data_upload": upload},
                   {"studies": 1, "parameters": len(cfg.get("inputs") or [])}, wait)


@app.get("/api/example/{name}")
def api_example(name: str):
    """Try with example: runs at once (seconds, offline, not metered)."""
    if name not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    cfg = json.loads(json.dumps(EXAMPLES[name]["config"]))
    with _HEAVY:
        out = run_given_data(cfg) if cfg.get("mode") == "given_data" else run_model_study(cfg)
    out["example"] = {"name": name, "title": EXAMPLES[name]["title"], "description": EXAMPLES[name]["description"]}
    return out


@app.get("/api/example/{name}/config")
def api_example_config(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    return EXAMPLES[name]["config"]


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
