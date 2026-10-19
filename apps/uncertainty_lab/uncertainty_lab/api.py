"""Engineering Uncertainty Lab: web API + single-page UI. Know when a prediction is reliable and when it is not.

Pick a model (a pp.zoo model or an uploaded ONNX graph) and its reference data, and get split-conformal prediction
intervals with their coverage measured on held-out data, applicability alerts (outside the validity range, sparse
data), the screening of your cases with the rule that says which must go to the high-fidelity solver, and the
propagation of uncertain inputs around an operating point. Every number comes from pp.uncertainty.uncertainty_study.

Run:  uvicorn uncertainty_lab.api:app --port 8098   (from apps/uncertainty_lab)

Studies run in the shared job queue (apps/_shared, pinneapple_core.service): POST /api/study returns a job; poll
GET /api/jobs/{id} (or pass ?wait=<seconds>). Inputs and outputs use the shared contracts: a ModelRef for the model,
a ParameterSpace for the uncertain inputs, a StudyResult in every result.

Environment (all optional; the runtime variables are in apps/_shared/README.md):
  UQL_USER / UQL_PASSWORD   HTTP Basic login on everything except /health
  UQL_MAX_JOBS              studies running at once (default 2); UQL_MAX_HEAVY caps the synchronous examples
  UQL_FREE_STUDIES / UQL_FREE_SCENARIOS / UQL_FREE_PROPAGATION_SAMPLES / UQL_FREE_HF_RUNS   free use per month
  UQL_MAX_MB                largest upload in MB (default 50)
  UQL_MAX_CASES             cases screened per study (default 2000)
  UQL_MAX_PROPAGATION       Monte-Carlo samples per study (default 20000)
  UQL_MAX_HF                high-fidelity solver runs per study (default 40)
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

from pinneapple_analysis.study_contracts import (
    is_parameter_space,
    model_from_ref,
    model_to_ref,
    space_from_contract,
    uncertainty_result,
)
from pinneapple_analysis.study_models import StudyModel, columns
from pinneapple_analysis.uncertainty.conformal_study import split_indices
from pinneapple_analysis.uncertainty.study import SCALES, uncertainty_study
from pinneapple_core.jsonable import to_jsonable
from pinneapple_core.service import UploadRejected, UploadTooLarge
from pinneapple_data.data_health import load_table

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
MAX_MB = float(os.environ.get("UQL_MAX_MB") or "50")
MAX_CASES = int(os.environ.get("UQL_MAX_CASES") or "2000")
MAX_PROP = int(os.environ.get("UQL_MAX_PROPAGATION") or "20000")
MAX_HF = int(os.environ.get("UQL_MAX_HF") or "40")

# zoo models with a reference dataset (held-out splits) and, where one exists, their high-fidelity solver
ZOO = {
    "heatsink_forced_mlp": {"outputs": ["r_base_k_w", "pressure_drop_pa"], "cal": "cal", "test": "test",
                            "hf": "heat-sink physics engine (FVM base conduction + Teertstra convection), about 0.4 s "
                                  "per design", "hf_outputs": ["r_base_k_w", "pressure_drop_pa"]},
    "airfoil_cst_rans_mlp": {"outputs": ["Cl", "Cd", "Cm"], "cal": "naca_reference", "test": "test", "hf": None,
                             "hf_outputs": []},
}

app = FastAPI(title="Engineering Uncertainty Lab", version=VERSION,
              description="Prediction intervals with coverage measured on held-out data, applicability alerts, "
                          "uncertainty propagation and the rule that routes cases to the high-fidelity solver. "
                          "POST a study to /api/study (multipart: config JSON + optional reference data, cases and "
                          "ONNX model); it runs as a job (GET /api/jobs/{id}). Or run /api/example/{name}.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="UQL")
TABLES = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl"}
# metered per caller and calendar month (free defaults; UQL_FREE_<UNIT> overrides, "unlimited" lifts)
rt = install_runtime(app, prefix="UQL", name="uncertainty",
                     units={"studies": 200, "scenarios": 20000, "propagation_samples": 2_000_000, "hf_runs": 500},
                     upload_suffixes=TABLES | {".onnx"}, upload_max_mb=MAX_MB)
_HEAVY = BusyLimiter("UQL_MAX_HEAVY")       # the synchronous examples

_NOMINAL_HS = dict(base_width_mm=100.0, base_depth_mm=100.0, base_thickness_mm=6.0, n_fins=18.0, fin_thickness_mm=1.2,
                   fin_height_mm=35.0, k_w_mk=200.0, source_width_mm=40.0, source_depth_mm=40.0, t_ambient_c=35.0,
                   air_velocity_m_s=2.5, power_w=120.0)


def _hs(name, **kw):
    return {"name": name, **_NOMINAL_HS, **kw}


NACA2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)

EXAMPLES = {
    "heatsink": {
        "title": "Heat-sink surrogate vs its physics engine",
        "description": "Zoo model heatsink_forced_mlp: 90 % intervals calibrated on its 600-design calibration split "
                       "and measured on the 900 held-out test designs; 11 designs screened against R <= 0.40 K/W, "
                       "every one also solved by the physics engine to check the routing rule.",
        "config": {
            "model": {"source": "zoo", "name": "heatsink_forced_mlp"}, "output": "r_base_k_w", "alpha": 0.1,
            "scale": "relative", "limits": [None, 0.40], "tolerance_pct": 3.0, "high_fidelity": "all",
            "cases": [_hs("nominal design"), _hs("fan derated to 2.3 m/s", air_velocity_m_s=2.3),
                      _hs("low airflow 1 m/s", air_velocity_m_s=1.0),
                      _hs("dense fins", n_fins=30.0, fin_thickness_mm=0.8),
                      _hs("tall dense fins", fin_height_mm=75.0, n_fins=40.0, fin_thickness_mm=0.8),
                      _hs("large sink at 380 W", base_width_mm=200.0, base_depth_mm=200.0, n_fins=35.0,
                          power_w=380.0),
                      _hs("small sink", base_width_mm=35.0, base_depth_mm=35.0, n_fins=6.0, fin_thickness_mm=1.0,
                          source_width_mm=30.0, source_depth_mm=30.0, power_w=30.0),
                      _hs("8 m/s fan (beyond range)", air_velocity_m_s=8.0),
                      _hs("alloy k = 120 W/mK (beyond range)", k_w_mk=120.0),
                      _hs("thin base, tiny source, slow air", base_width_mm=190.0, base_depth_mm=190.0,
                          base_thickness_mm=2.2, source_width_mm=20.0, source_depth_mm=20.0, n_fins=60.0,
                          fin_thickness_mm=0.7, fin_height_mm=78.0, air_velocity_m_s=0.6, power_w=390.0),
                      _hs("corner of the design space", base_width_mm=195.0, base_depth_mm=32.0,
                          base_thickness_mm=14.5, n_fins=50.0, fin_thickness_mm=0.65, fin_height_mm=79.0,
                          source_width_mm=190.0, source_depth_mm=31.0, air_velocity_m_s=5.9, power_w=395.0,
                          t_ambient_c=49.0)],
            "nominal": dict(_NOMINAL_HS),
            "uncertain": [{"name": "air_velocity_m_s", "dist": "normal", "mean": 2.5, "std": 0.4},
                          {"name": "t_ambient_c", "dist": "uniform", "low": 25, "high": 45},
                          {"name": "k_w_mk", "dist": "triangular", "low": 180, "mode": 200, "high": 210},
                          {"name": "fin_thickness_mm", "dist": "normal", "mean": 1.2, "std": 0.05}],
            "n_propagation": 4096, "propagation_method": "lhs"}},
    "airfoil": {
        "title": "Airfoil lift with an ensemble spread",
        "description": "Zoo model airfoil_cst_rans_mlp (ensemble): intervals scaled by the ensemble spread, calibrated "
                       "on the 36 NACA reference runs and measured on the 60 held-out CST runs (OpenFOAM, never used "
                       "in training); NACA 2412 screened against Cl >= 0.6.",
        "config": {
            "model": {"source": "zoo", "name": "airfoil_cst_rans_mlp"}, "output": "Cl", "alpha": 0.1,
            "scale": "ensemble", "limits": [0.6, None], "tolerance": 0.05, "high_fidelity": "none",
            "cases": [{"name": f"NACA 2412 at {a:g} deg", **NACA2412, "alpha_deg": a} for a in (0.0, 4.0, 6.0, 12.0)]
            + [{"name": "NACA 2412 at 16 deg (beyond range)", **NACA2412, "alpha_deg": 16.0},
               {"name": "thick section T0 = 0.26", **{**NACA2412, "T0": 0.26}, "alpha_deg": 4.0}],
            "nominal": {**NACA2412, "alpha_deg": 4.0},
            "uncertain": [{"name": "alpha_deg", "dist": "normal", "mean": 4.0, "std": 0.5},
                          {"name": "T0", "dist": "normal", "mean": NACA2412["T0"], "std": 0.003},
                          {"name": "M1", "dist": "uniform", "low": 0.066, "high": 0.076}],
            "n_propagation": 4096, "propagation_method": "lhs"}},
}

VALIDATION = [
    "Split-conformal quantile = the finite-sample order statistic of ConformalPredictor; coverage is measured on a "
    "split never used to calibrate, with its Wilson interval (tests/test_uncertainty_study.py)",
    "Heat-sink example: calibrated on the 600-design cal split, measured on the 900 held-out test designs of "
    "heatsink_forced_convection; the screened designs are also solved by the physics engine that generated the data",
    "Propagation: Monte Carlo / Latin hypercube / Sobol reproduce the Ishigami mean and variance; polynomial chaos "
    "reports its leave-one-out Q2 and reproduces polynomials exactly",
    "Applicability gate: declared validity range + 5-nearest-neighbour distance against the 99th percentile of the "
    "reference data's own distances; flags holes inside the range in tests",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the numbers mean",
            "band": "Coverage is measured, not assumed; it holds for cases like the calibration data.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss a problem", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Conformal coverage", "effect": "optimistic",
                 "detail": "The guarantee is marginal (on average over cases like the calibration data). Coverage can "
                           "be lower in a region: see coverage by group and by gate.",
                 "today": "Use the per-group table; route sparse and outside cases to the solver.",
                 "planned": "Conditional (Mondrian) conformal by region."},
                {"topic": "Small calibration sets", "effect": "check",
                 "detail": "With few calibration points the interval is the largest residual seen and the measured "
                           "coverage has a wide confidence interval (airfoil example: 36 calibration + 60 test runs).",
                 "today": "Read the Wilson interval, not only the point estimate.", "planned": ""},
                {"topic": "Applicability gate", "effect": "optimistic",
                 "detail": "Range and k-NN density in standardised inputs; a point can be dense in inputs and still "
                           "be in a different physical regime (e.g. stall).",
                 "today": "Add the regime limits as tolerance or limits.", "planned": "Physics-regime checks."},
                {"topic": "Propagation", "effect": "check",
                 "detail": "Independent inputs; the propagated spread is of the model's prediction, the model's own "
                           "error is added only in 'with model error' (interval bounds).",
                 "today": "State correlated inputs as one input.", "planned": "Correlated inputs (copulas)."},
                {"topic": "High-fidelity checks", "effect": "conservative",
                 "detail": "Available where the library has the solver that generated the model's data (heat-sink "
                           "engine). Other models: route only.",
                 "today": "Run the routed cases in your solver.", "planned": "OpenFOAM / CalculiX via the runner."},
            ]}


# ---------------------------------------------------------------------------------------------- model + data
def _load_model(spec: dict, files: dict) -> StudyModel:
    if spec.get("kind") == "ModelRef":
        try:
            return model_from_ref(spec, onnx=files["model"][1] if "model" in files else None)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    src = spec.get("source", "zoo")
    if src == "zoo":
        name = spec.get("name", "")
        if name not in ZOO:
            raise HTTPException(422, f"Zoo model must be one of {', '.join(ZOO)} (models with held-out reference data).")
        return StudyModel.from_zoo(name, outputs=spec.get("outputs") or ZOO[name]["outputs"])
    if src == "onnx":
        if "model" not in files:
            raise HTTPException(422, "Upload the ONNX file (field 'model').")
        return StudyModel.from_onnx(files["model"][1], spec.get("input_names") or None,
                                    spec.get("output_names") or None, name=files["model"][0])
    raise HTTPException(422, "model.source must be 'zoo' or 'onnx'.")


def _table(item):
    name, data = item
    df, _ = load_table(data, name)
    return df


def _reference(model: StudyModel, cfg: dict, files: dict):
    """(X_cal, y_cal, X_test, y_test, X_ref, note) from an uploaded table or the zoo model's own splits."""
    out = cfg["output"]
    if "reference" in files:
        df = _table(files["reference"])
        X, y = columns(df, model.input_names), columns(df, [out])[:, 0]
        if "split" in df.columns:
            s = df["split"].astype(str).str.lower().to_numpy()
            cal, test, train = s == "cal", s == "test", s == "train"
            if cal.sum() == 0 or test.sum() == 0:
                raise HTTPException(422, "The 'split' column needs rows marked cal and test.")
            ref = X[train] if train.any() else X[cal | test]
            return X[cal], y[cal], X[test], y[test], ref, \
                f"uploaded table: {int(cal.sum())} cal / {int(test.sum())} test rows (split column)"
        c, t = split_indices(len(y), (0.5, 0.5), seed=int(cfg.get("seed", 0)))
        return X[c], y[c], X[t], y[t], X, f"uploaded table split at random: {len(c)} cal / {len(t)} test rows"
    if model.kind != "zoo" or model.name not in ZOO:
        raise HTTPException(422, "Upload reference data (inputs and the true output) for this model: only "
                                 f"{', '.join(ZOO)} come with held-out reference splits here.")
    z = ZOO[model.name]
    j = model.output_index(out)
    Xc, Yc = model.reference(z["cal"])
    Xt, Yt = model.reference(z["test"])
    Xr = model.reference("train")[0]
    return Xc, Yc[:, j], Xt, Yt[:, j], Xr, \
        f"{model.reference_dataset}: '{z['cal']}' split ({len(Xc)} rows) calibrates, held-out '{z['test']}' split " \
        f"({len(Xt)} rows) measures; training inputs ({len(Xr)}) define the data density"


def _cases(model: StudyModel, cfg: dict, files: dict):
    rows = list(cfg.get("cases") or [])
    X, names = [], []
    for i, r in enumerate(rows):
        missing = [n for n in model.input_names if n not in r]
        if missing:
            raise HTTPException(422, f"case {i + 1} misses {', '.join(missing)}")
        X.append([float(r[n]) for n in model.input_names])
        names.append(str(r.get("name") or f"case {i + 1}"))
    if "cases" in files:
        df = _table(files["cases"])
        Xu = columns(df, model.input_names)
        lab = df["name"].astype(str).tolist() if "name" in df.columns else \
            [f"row {i + 1}" for i in range(len(Xu))]
        X += Xu.tolist()
        names += lab
    if len(X) > MAX_CASES:
        raise HTTPException(413, f"{len(X)} cases; the limit is {MAX_CASES} per study.")
    return (np.array(X), names) if X else (None, None)


def _hf(model: StudyModel, output: str):
    if model.kind != "zoo" or not ZOO.get(model.name, {}).get("hf") or output not in ZOO[model.name]["hf_outputs"]:
        return None
    if model.name.startswith("heatsink_forced"):
        from pinneapple_design.thermal.heatsink.surrogate import reference_solve_batch
        return lambda X: reference_solve_batch(X, model.input_names, output)
    return None


class _Counter:
    """Counts model evaluations (rows) for the usage report."""

    def __init__(self, model: StudyModel):
        self.n = 0
        inner = model.predict_fn

        def counted(X):
            self.n += len(X)
            return inner(X)
        model.predict_fn = counted


def _normalize(cfg: dict) -> dict:
    """Read the shared contracts: ``model_ref`` (ModelRef) -> model, ``uncertain`` or ``parameter_space`` given as a
    ParameterSpace -> uncertain inputs (its constants join the nominal point). Checks the cheap limits."""
    cfg = dict(cfg)
    if cfg.get("model_ref"):
        cfg["model"] = {**cfg.pop("model_ref"), "kind": "ModelRef"}
    ps = cfg.pop("parameter_space", None)
    if ps is None and is_parameter_space(cfg.get("uncertain")):
        ps = cfg["uncertain"]
    if ps is not None:
        try:
            params, fixed = space_from_contract(ps)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"parameter space: {e}") from e
        cfg["uncertain"] = params
        cfg["nominal"] = {**fixed, **(cfg.get("nominal") or {})}
    if cfg.get("scale", "absolute") not in SCALES:
        raise HTTPException(422, f"scale must be one of {', '.join(SCALES)}")
    n_prop = int(cfg.get("n_propagation") or 4096)
    if n_prop > MAX_PROP:
        raise HTTPException(413, f"{n_prop} propagation samples; the limit is {MAX_PROP}.")
    if len(cfg.get("cases") or []) > MAX_CASES:
        raise HTTPException(413, f"{len(cfg['cases'])} cases; the limit is {MAX_CASES} per study.")
    return cfg


def run_study(cfg: dict, files: dict, ctx=None) -> dict:
    t0 = time.time()

    def progress(fraction, message):
        if ctx is not None:
            ctx.progress(fraction, message)

    cfg = _normalize(cfg)
    n_prop = int(cfg.get("n_propagation") or 4096)
    try:
        progress(0.05, "loading the model and its reference data")
        model = _load_model(cfg.get("model") or {}, files)
        out_name = cfg.get("output") or model.output_names[0]
        cfg["output"] = out_name
        Xc, yc, Xt, yt, Xr, note = _reference(model, cfg, files)
        cases, names = _cases(model, cfg, files)
        hf = _hf(model, out_name) if cfg.get("high_fidelity", "none") != "none" else None
        counter = _Counter(model)
        progress(0.2, "conformal calibration, applicability gate"
                 + (", high-fidelity runs" if hf is not None and cases is not None else "")
                 + (", propagation" if cfg.get("uncertain") else ""))
        limits = cfg.get("limits")
        limits = None if not limits or all(v is None or v == "" for v in limits) else \
            tuple(None if v in (None, "") else float(v) for v in limits)
        res = uncertainty_study(
            model, out_name, Xc, yc, Xt, yt, alpha=float(cfg.get("alpha", 0.1)), scale=cfg.get("scale", "absolute"),
            X_ref=Xr, bounds=model.bounds or None, cases=cases, case_names=names,
            tolerance=None if cfg.get("tolerance") in (None, "") else float(cfg["tolerance"]),
            tolerance_pct=None if cfg.get("tolerance_pct") in (None, "") else float(cfg["tolerance_pct"]),
            limits=limits, high_fidelity=hf, hf_scope="all" if cfg.get("high_fidelity") == "all" else "routed",
            max_hf=MAX_HF, nominal=cfg.get("nominal") or None, uncertain=cfg.get("uncertain") or None,
            n_propagation=n_prop, propagation_method=cfg.get("propagation_method", "lhs"),
            seed=int(cfg.get("seed", 0)))
    except HTTPException:
        raise
    except (ValueError, KeyError, ImportError) as e:
        raise HTTPException(422, str(e)) from e
    res["reference_note"] = note
    res["high_fidelity_available"] = ZOO.get(model.name, {}).get("hf") if model.kind == "zoo" else None
    hf_runs = (res.get("cases") or {}).get("high_fidelity") or {}
    res["usage"] = {"model_evaluations": counter.n, "cases": 0 if cases is None else len(cases),
                    "high_fidelity_runs": hf_runs.get("n_run", 0),
                    "propagation_samples": (res.get("propagation") or {}).get("n", 0),
                    "seconds": round(time.time() - t0, 2)}
    res["config"] = cfg
    res["scope"] = scope()
    progress(0.95, "writing the StudyResult")
    res["study_result"] = uncertainty_result(res, model, f"uncertainty-{uuid.uuid4().hex[:12]}").to_dict()
    return to_jsonable(res)


# ---------------------------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "scales": list(SCALES), "runtime": rt.describe(),
            "examples": [{"name": k, "title": v["title"], "description": v["description"]} for k, v in EXAMPLES.items()],
            "zoo": [{"name": k, "outputs": v["outputs"], "high_fidelity": v["hf"]} for k, v in ZOO.items()],
            "limits": {"max_mb": MAX_MB, "max_cases": MAX_CASES, "max_propagation": MAX_PROP, "max_hf": MAX_HF},
            "scope": scope()}


@app.get("/api/model/{name}")
def model_info(name: str):
    """Inputs, outputs, units, validity range and a nominal point (median of the held-out inputs) of a zoo model."""
    if name not in ZOO:
        raise HTTPException(404, "No such model.")
    m = StudyModel.from_zoo(name, outputs=ZOO[name]["outputs"])
    X, _ = m.reference("test")
    info = m.info()
    info["nominal"] = dict(zip(m.input_names, np.median(X, axis=0).round(6).tolist(), strict=True))
    info["high_fidelity"] = ZOO[name]["hf"]
    info["model_ref"] = model_to_ref(m).to_dict()
    info["metadata"] = {k: v for k, v in m.metadata.items() if k in ("version", "known_limitations",
                                                                     "validation_metrics")}
    return to_jsonable(info)


# ---------------------------------------------------------------------------------------------- jobs
FILE_KEYS = ("model", "reference", "cases")


def _files_from_uploads(ids: dict) -> dict:
    out = {}
    for key, upload_id in (ids or {}).items():
        if upload_id:
            meta = rt.uploads.get(upload_id)
            out[key] = (meta.get("name") or upload_id, rt.uploads.path(upload_id).read_bytes())
    return out


def job_study(params: dict, ctx) -> dict:
    return run_study(params["config"], _files_from_uploads(params.get("uploads")), ctx)


rt.jobs.register("study", job_study)


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


def _n_rows(upload_id: str | None) -> int:
    if not upload_id:
        return 0
    meta = rt.uploads.get(upload_id)
    try:
        df, _ = load_table(rt.uploads.path(upload_id).read_bytes(), meta.get("name") or "cases.csv")
    except ValueError as e:
        raise HTTPException(422, f"cases table: {e}") from e
    return len(df)


_CONFIG = Form(..., description="study configuration (JSON, see /api/example/heatsink/config)")
_MODEL_FILE, _REFERENCE_FILE, _CASES_FILE = File(None), File(None), File(None)   # one object per parameter


@app.post("/api/study")
async def api_study(request: Request, config: str = _CONFIG, model: UploadFile | None = _MODEL_FILE,
                    reference: UploadFile | None = _REFERENCE_FILE, cases: UploadFile | None = _CASES_FILE,
                    wait: float = 0):
    """Run a study as a job. ``config``: {model: {source: zoo|onnx, name, input_names, output_names} or model_ref
    (ModelRef), output, alpha, scale (absolute|relative|ensemble), limits [lo, hi], tolerance or tolerance_pct,
    high_fidelity (none|routed|all), cases [{name, input: value}], nominal {input: value}, uncertain [{name, dist, ...}]
    or a ParameterSpace (``parameter_space``), n_propagation, propagation_method (mc|lhs|sobol), and optional
    model_upload / reference_upload / cases_upload (ids from /api/uploads)}. Files: model (.onnx), reference
    (CSV/Parquet/JSON: inputs + output, optional 'split' column with cal/test/train), cases (inputs, optional 'name').
    Returns the job (GET /api/jobs/{id}); ``?wait=<s>`` waits for it."""
    try:
        cfg = _normalize(json.loads(config))
    except ValueError as e:
        raise HTTPException(422, f"config is not valid JSON: {e}") from e
    uploads = {}
    for key, f in zip(FILE_KEYS, (model, reference, cases), strict=True):
        uploads[key] = await _save(request, f, cfg.pop(f"{key}_upload", None))
    spec = cfg.get("model") or {}
    if spec.get("kind") != "ModelRef" and spec.get("source", "zoo") == "zoo":
        m = _load_model(spec, {})                          # 422 for an unknown model, before queueing
        if cfg.get("scale") == "ensemble" and m.std_fn is None:
            raise HTTPException(422, f"{m.name} has no ensemble spread: use the 'absolute' or 'relative' scale")
    if (spec.get("source") == "onnx" or str(spec.get("framework", "")).lower() == "onnx") and not uploads["model"]:
        raise HTTPException(422, "Upload the ONNX file (field 'model', or model_upload from /api/uploads).")
    for i, r in enumerate(cfg.get("cases") or []):          # a clear 422 now, not a failed job later
        bad = [k for k, v in r.items() if k != "name" and not isinstance(v, (int, float))]
        if bad:
            raise HTTPException(422, f"case {i + 1} ({r.get('name', '')}): no number for {', '.join(bad)}")
    n_cases = len(cfg.get("cases") or []) + _n_rows(uploads["cases"])
    if n_cases > MAX_CASES:
        raise HTTPException(413, f"{n_cases} cases; the limit is {MAX_CASES} per study.")
    hf = cfg.get("high_fidelity", "none")
    units = {"studies": 1, "scenarios": n_cases,
             "propagation_samples": int(cfg.get("n_propagation") or 4096) if cfg.get("uncertain") else 0,
             "hf_runs": min(n_cases, MAX_HF) if hf != "none" else 0}
    job = rt.submit(request, "study", {"config": cfg, "uploads": uploads},
                    units={k: v for k, v in units.items() if v})
    if wait > 0:
        job = await run_in_threadpool(rt.jobs.wait, job["id"], timeout=min(float(wait), 600.0))
    return job


@app.get("/api/example/{name}")
def api_example(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    with _HEAVY:
        out = run_study(json.loads(json.dumps(EXAMPLES[name]["config"])), {})
    out["example"] = {"name": name, "title": EXAMPLES[name]["title"], "description": EXAMPLES[name]["description"]}
    return out


@app.get("/api/example/{name}/config")
def api_example_config(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    return EXAMPLES[name]["config"]


@app.get("/api/template/{name}.csv")
def api_template(name: str):
    """Reference-data template for a zoo model: its held-out test rows (inputs + outputs + split column)."""
    if name not in ZOO:
        raise HTTPException(404, "No such model.")
    m = StudyModel.from_zoo(name, outputs=ZOO[name]["outputs"])
    buf = io.StringIO()
    buf.write(",".join(m.input_names + m.output_names + ["split"]) + "\n")
    for split in (ZOO[name]["cal"], ZOO[name]["test"]):
        X, Y = m.reference(split)
        for x, y in zip(X, Y, strict=True):
            buf.write(",".join(f"{v:.8g}" for v in list(x) + list(y)) + f",{'cal' if split == ZOO[name]['cal'] else 'test'}\n")
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}_reference.csv"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
