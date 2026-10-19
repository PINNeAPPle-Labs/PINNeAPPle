"""Physics Anomaly Detector (day 28): find abnormal behaviour in sensor time series with data *and* physics.

Statistical detectors, time-series models, conservation balances written as formulas over your columns, and the
residuals of a physical reference model, fused into events with severity, the variables involved and explanations
that say which balance broke and by how much. Every computation is a library call
(``pinneapple_systems.operations.anomaly``); this module is routes, uploads and the UI.

Run:  uvicorn anomaly_detector.api:app --port 8111   (from apps/anomaly_detector)

Environment (all optional):
  ANOMALY_USER / ANOMALY_PASSWORD   HTTP Basic login on everything except /health
  ANOMALY_MAX_JOBS                  analyses / benchmarks running at once (default 2)
  ANOMALY_MAX_ROWS                  largest table analysed (default 500,000 rows)
  ANOMALY_FREE_*                    monthly free units (see install_runtime)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_core.jsonable import to_jsonable
from pinneapple_data.contracts import ModelRef, read_table
from pinneapple_systems.operations.anomaly import (
    METHODS,
    BalanceRule,
    DetectorConfig,
    detect,
    intervals_from_labels,
    run_benchmark,
    score_events,
    tank_balance_rules,
    tank_constants,
)
from pinneapple_systems.operations.contracts import anomaly_study, load_model
from pinneapple_systems.operations.plant import TankParams
from pinneapple_systems.operations.twin import LumpedTankModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).parent
STATIC = HERE / "static"
EXAMPLES = HERE.parent / "examples"
MAX_ROWS = int(os.environ.get("ANOMALY_MAX_ROWS") or 500000)
OPTIONAL_BODY = Body(default={})

app = FastAPI(title="Physics Anomaly Detector", version=VERSION,
              description="Upload sensor time series (CSV, Parquet, JSON): statistical detectors, AR + CUSUM, "
                          "conservation balances over your columns and physical-model residuals, calibrated on a "
                          "window you declare normal, fused into events with severity, variables and explanations. "
                          "With a label column, precision and recall are computed.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="ANOMALY")
rt = install_runtime(app, prefix="ANOMALY", name="anomaly", max_jobs=2,
                     units={"analyses": 200, "rows": 5_000_000, "benchmarks": 20},
                     upload_suffixes={".csv", ".parquet", ".json"})
MODELS = rt.root / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def tank_preset() -> dict:
    """The example's configuration: balances and constants of the heated tank, the lumped reference model."""
    return {"sensors": ["h", "T", "q_in", "q_out", "Q"], "normal_until": 7200.0,
            "methods": list(METHODS), "window": 12, "threshold": 5.0,
            "balances": [b.__dict__ for b in tank_balance_rules()], "constants": tank_constants(),
            "model": "lumped_tank", "model_states": ["h", "T"], "model_inputs": ["q_in", "Q", "T_in", "T_amb"],
            "label_column": "fault"}


def scope() -> dict:
    return {"title": "Scope & validation", "noun": "limitation", "items_title": "What the detector assumes",
            "validated": [
                "A 0.4 L/s leak breaks the mass balance by -0.4 L/s (within 15 %), quantified in the explanation "
                "(tests/test_operations_anomaly_control.py)",
                "No physics event (balances, model residuals) on fault-free held-out runs",
                "Benchmark on held-out labelled runs (six fault kinds each): the combined detector finds at least "
                "5 of 6 faults; precision, recall and false alarms per hour are recomputed on every benchmark run",
                "Balances do not see a drain-valve fault when the outflow is measured (the physics holds); the model "
                "residual does. The benchmark reports this per fault kind instead of hiding it",
            ],
            "band": "Thresholds are calibrated only on the window you declare normal; the data after it are judged, "
                    "never used to tune.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss a problem", "check": "Know this"},
            "items": [
                {"topic": "The normal window", "effect": "optimistic",
                 "detail": "Every detector learns 'normal' from the window before normal_until. A fault already "
                           "present there becomes normal.",
                 "today": "Pick a window you have checked; at least 4x the balance window.",
                 "planned": "Automatic selection of a clean reference window."},
                {"topic": "Balances are only as good as the sensors", "effect": "check",
                 "detail": "A conservation residual mixes model and sensor errors: a broken balance says 'something "
                           "enters or leaves, or a meter is wrong', not which one.",
                 "today": "Read the variables involved and the companion events (a sensor fault often breaks one "
                          "balance and one rate test).", "planned": "Sensor-fault isolation by structured residuals."},
                {"topic": "Synthetic benchmark", "effect": "check",
                 "detail": "Precision/recall come from labelled synthetic runs of the heated tank. The public "
                           "BATADAL water-network adapter (PR #731) is not merged yet.",
                 "today": "Upload your own labelled history (a label column) to score the detector on your data.",
                 "planned": "BATADAL and other public labelled sets once the adapter lands."},
                {"topic": "Operating changes", "effect": "check",
                 "detail": "The statistical and time-series channels can fire at large set-point steps; the "
                           "benchmark counts those as false alarms (0.1-0.6 per hour in our runs).",
                 "today": "Rely on balance and model events for decisions; use the others as context.",
                 "planned": "Exogenous-input time-series models."},
            ]}


# ------------------------------------------------------------------------------------------------- models

def _reference_model(params: dict):
    kind = params.get("model") or "none"
    if kind == "none":
        return None
    if kind == "lumped_tank":
        c = params.get("constants") or {}
        p = TankParams(**{k: float(v) for k, v in (("area", c.get("A")), ("ua", c.get("UA")), ("rho", c.get("rho")),
                                                   ("cp", c.get("cp")), ("k_out", c.get("k_out"))) if v is not None})
        return LumpedTankModel(p, dt=float(params.get("model_dt") or 10.0), estimate=())
    if kind == "uploaded":
        mid = str(params.get("model_id") or "")
        d = MODELS / mid
        if not mid or "/" in mid or ".." in mid or not d.is_dir():
            raise HTTPException(404, "unknown model_id; upload a ModelRef first (POST /api/models)")
        return load_model(ModelRef.load(d / "modelref.json"), d)
    raise HTTPException(422, "model is none, lumped_tank or uploaded")


@app.post("/api/models")
async def upload_model(modelref: UploadFile, model_file: UploadFile):
    """Register a reference model: its ``ModelRef`` (JSON) and the model file (a controlled-OpInf JSON). The
    SHA-256 in the ModelRef is checked."""
    import uuid

    try:
        ref = ModelRef.model_validate(json.loads(await modelref.read()))
    except Exception as e:
        raise HTTPException(422, f"not a valid ModelRef: {e}") from e
    data = await model_file.read()
    if len(data) > 5 * 2**20:
        raise HTTPException(413, "model files up to 5 MB")
    mid = uuid.uuid4().hex[:12]
    d = MODELS / mid
    d.mkdir()
    (d / Path(ref.uri).name).write_bytes(data)
    ref = ref.model_copy(update={"uri": Path(ref.uri).name})
    ref.save(d / "modelref.json")
    try:
        m = load_model(ref, d)
    except (ValueError, KeyError, FileNotFoundError) as e:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
        raise HTTPException(422, str(e)) from e
    return {"model_id": mid, "kind": m.kind, "states": m.state_names, "inputs": m.input_names, "dt": m.dt}


# ------------------------------------------------------------------------------------------------- jobs

def _table(params: dict):
    if params.get("example"):
        df, _ = read_table(EXAMPLES / "tank_six_faults.csv")
    else:
        df, _ = read_table(params["path"])
    if len(df) > MAX_ROWS:
        raise ValueError(f"{len(df):,} rows: this server analyses up to {MAX_ROWS:,}")
    return df


def _analyze_job(params: dict, ctx) -> dict:
    ctx.progress(0.05, "reading the table")
    df = _table(params)
    if "t" not in df.columns:
        raise ValueError("the table needs a time column 't' (seconds, increasing)")
    label = params.get("label_column") or None
    cfg = DetectorConfig.from_dict({k: params[k] for k in ("normal_until", "sensors", "methods", "balances",
                                                           "constants", "window", "threshold") if k in params})
    model = _reference_model(params)
    cols = {c: df[c].to_numpy() for c in df.columns if c != label}
    ctx.progress(0.3, "detecting")
    res = detect(cols, cfg, model=model, model_states=params.get("model_states") or (),
                 model_inputs=params.get("model_inputs") or ())
    out = res.to_dict(max_points=1500)
    signals = cfg.sensors or [c for c in cols if c != "t" and np.issubdtype(np.asarray(cols[c]).dtype, np.number)]
    step = max(1, int(np.ceil(len(df) / 1500)))
    out["signals"] = {s: np.asarray(cols[s], dtype=float)[::step].tolist() for s in signals[:8]}
    out["flags"] = res.flags[::step].tolist()
    if label and label in df.columns:
        iv = intervals_from_labels(df["t"].to_numpy(), df[label].to_numpy())
        out["score"] = score_events(res, iv)
        out["truth"] = [{"kind": i.kind, "start": i.start, "end": i.end} for i in iv]
    st = anomaly_study(res, title=f"Anomaly detection on {params.get('name') or 'the example'}")
    if out.get("score"):
        sc = out["score"]
        st.metrics.update({"point_precision": sc["point_precision"], "point_recall": sc["point_recall"],
                           "event_recall": sc["event_recall"], "false_alarms_per_hour": sc["false_alarms_per_hour"]})
    (ctx.workdir / "anomaly.study.json").write_text(st.to_json())
    ev = res.events
    lines = ["start_s,end_s,severity,methods,variables,explanation"]
    lines += [f'{e.start},{e.end},{e.severity},"{" ".join(e.methods)}","{" ".join(e.variables)}","'
              + (e.explanations[0].replace('"', "'") if e.explanations else "") + '"' for e in ev]
    (ctx.workdir / "events.csv").write_text("\n".join(lines) + "\n")
    ctx.progress(1.0, "done")
    out["study_passed"] = st.passed
    return to_jsonable(out)


def _benchmark_job(params: dict, ctx) -> dict:
    ctx.progress(0.05, "simulating labelled runs and scoring each detector family")
    seeds = tuple(int(s) for s in params.get("seeds") or (101, 102, 103))
    res = run_benchmark(seeds=seeds, window=int(params.get("window") or 12),
                        threshold=float(params.get("threshold") or 5.0))
    (ctx.workdir / "benchmark.json").write_text(json.dumps(to_jsonable(res), indent=1))
    ctx.progress(1.0, "done")
    return to_jsonable(res)


rt.jobs.register("analyze", _analyze_job)
rt.jobs.register("benchmark", _benchmark_job)


@app.post("/api/analyze")
def api_analyze(request: Request, body: dict = OPTIONAL_BODY):
    """Detect anomalies. ``upload_id`` (POST /api/uploads) or ``example: true``; ``normal_until`` [s];
    ``sensors``; ``methods`` (statistical, timeseries, balance, model); ``balances`` [{name, accumulation, inflow,
    outflow, units}] with ``constants``; ``model`` none | lumped_tank | uploaded (+ ``model_id``, ``model_states``,
    ``model_inputs``); ``window``; ``threshold``; ``label_column`` for precision / recall. Returns a job."""
    params = dict(tank_preset()) if body.get("example") else {}
    params.update({k: v for k, v in body.items() if k not in ("upload_id",)})
    if not body.get("example"):
        path = rt.upload_path(request, str(body.get("upload_id") or ""))
        params.update(path=str(path), name=path.name)
        try:
            df, _ = read_table(path)
        except Exception as e:
            raise HTTPException(422, f"could not read the table: {e}") from e
        n_rows = len(df)
    else:
        n_rows = len(read_table(EXAMPLES / "tank_six_faults.csv")[0])
    if "normal_until" not in params:
        raise HTTPException(422, "normal_until (end of the normal window, same units as t) is required")
    try:
        DetectorConfig.from_dict({k: params[k] for k in ("normal_until", "sensors", "methods", "balances",
                                                         "constants", "window", "threshold") if k in params})
        for b in params.get("balances") or []:
            BalanceRule(**b)
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    _reference_model(params)
    return rt.submit(request, "analyze", params, units={"analyses": 1, "rows": n_rows})


@app.post("/api/benchmark")
def api_benchmark(request: Request, body: dict = OPTIONAL_BODY):
    """Score every detector family alone and combined on held-out labelled synthetic runs (one per seed, six
    faults each). Returns a job (about 5 s per seed)."""
    seeds = body.get("seeds") or [101, 102, 103]
    if len(seeds) > 6:
        raise HTTPException(422, "6 seeds at most")
    return rt.submit(request, "benchmark", {"seeds": seeds, "window": body.get("window"),
                                            "threshold": body.get("threshold")}, units={"benchmarks": 1})


@app.get("/api/columns/{upload_id}")
def columns(request: Request, upload_id: str):
    """Columns of an uploaded table, which are numeric, and the time range (to set up an analysis)."""
    path = rt.upload_path(request, upload_id)
    try:
        df, contract = read_table(path)
    except Exception as e:
        raise HTTPException(422, f"could not read the table: {e}") from e
    from pandas.api.types import is_numeric_dtype

    num = [c for c in df.columns if is_numeric_dtype(df[c])]
    t = df["t"].to_numpy(dtype=float) if "t" in df.columns else None
    return to_jsonable({"rows": len(df), "columns": list(map(str, df.columns)), "numeric": list(map(str, num)),
                        "t_range": None if t is None else [float(t.min()), float(t.max())],
                        "contract": contract.kind if contract is not None else None})


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(request: Request, job_id: str, name: str):
    rt.job_for(request, job_id)
    if name not in ("anomaly.study.json", "events.csv", "benchmark.json"):
        raise HTTPException(404, "no such file")
    p = rt.jobs.workdir / job_id / name
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=name)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return to_jsonable({"version": VERSION, "methods": METHODS, "max_rows": MAX_ROWS, "preset": tank_preset(),
                        "examples": ["tank_six_faults.csv"], "batadal": "not merged (PR #731): synthetic benchmark",
                        "scope": scope()})


@app.get("/api/example-file/{name}")
def example_file(name: str):
    if name != "tank_six_faults.csv":
        raise HTTPException(404, "no such file")
    return FileResponse(EXAMPLES / name, filename=name)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
