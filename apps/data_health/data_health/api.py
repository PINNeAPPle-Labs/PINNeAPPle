"""Engineering Data Health web API + single-page UI.

Run:  uvicorn data_health.api:app --port 8082   (from apps/data_health)

Environment (all optional):
  EDH_USER / EDH_PASSWORD   HTTP Basic login on everything except /health
  EDH_MAX_MB                largest upload in MB (default 50)
  EDH_MAX_ROWS              rows analysed per file (default 1,000,000)
  EDH_MAX_HEAVY             concurrent analyses per worker (default 2)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.data_health import analyze, example_chiller_plant, load_table
from pinneapple_data import process_optimizer as po

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_MB = float(os.environ.get("EDH_MAX_MB") or "50")
MAX_ROWS = int(os.environ.get("EDH_MAX_ROWS") or "1000000")
FORMATS = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".h5", ".hdf5"]

app = FastAPI(title="Engineering Data Health", version=VERSION,
              description="Physics-aware quality report for engineering datasets (PINNeAPPle Physical Data Layer). "
                          "POST a CSV, Excel, Parquet, JSON or HDF5 file to /api/analyze.")
install(app, prefix="EDH")
_HEAVY = BusyLimiter("EDH_MAX_HEAVY")

VALIDATION = [
    "Finds all 8 faults injected into a week of chilled-water plant data (gap, duplicates, frozen sensor, "
    "spikes, -999 codes, humidity above 100 %, a °C → K switch, a flow meter reading 28 % low) with no false alarm",
    "The same plant without faults scores 99.8 / 100 with no warning",
    "Identical results from CSV (comma or semicolon, decimal comma, dd/mm or mm/dd dates), Excel, Parquet, JSON and "
    "HDF5 (units read from dataset attributes)",
    "Unit conversions use exact definitions (NIST SP 811); round trips are exact to 1e-12",
]


def scope() -> dict:
    items = [
        {"topic": "Energy balance fluid", "effect": "check",
         "detail": "The heat-rate check uses water properties (ρ 997 kg/m³, cp 4186 J/(kg·K)). A glycol loop has "
                   "a lower cp, which shows as a constant ratio below 1 rather than a fault.",
         "today": "Read the ratio's changes over time, not its absolute level, for glycol loops.",
         "planned": "Fluid selection (glycol %, brines, refrigerants) with temperature-dependent properties."},
        {"topic": "Relations found from column names", "effect": "optimistic",
         "detail": "Physics checks run only when columns are named consistently (e.g. CHW_supply, CHW_return, "
                   "CHW_flow, Cooling_load). Other relations are not checked yet.",
         "today": "Use a common prefix and supply/return (or in/out) in the names of loop sensors.",
         "planned": "User-defined relations: mass balance, pump affinity laws, P = √3·V·I·PF, psychrometrics."},
        {"topic": "Generic thresholds", "effect": "check",
         "detail": "Spike, flat-line and gap thresholds are generic. A very fast real transient can be flagged as a "
                   "spike; a slow drift is not caught by any single-column check.",
         "today": "Open the flagged spans in the column chart and confirm before deleting data.",
         "planned": "Thresholds learned per quantity from your own history; drift detection against redundant "
                    "sensors and physics models."},
        {"topic": "Units come from the headers", "effect": "optimistic",
         "detail": "A column without a unit in its header (or file metadata) skips the physical-limit and unit "
                   "checks.",
         "today": "Write units in headers, e.g. 'Flow [m3/h]', or upload Parquet/HDF5 with unit metadata.",
         "planned": "Unit inference from value ranges and BACnet / OPC UA tag dictionaries."},
    ]
    return {"validated": VALIDATION, "items": items, "title": "Checks scope & validation", "noun": "limitation",
            "items_title": "What the checks assume",
            "band": "Every issue lists its evidence (rows, time spans, values) so it can be verified by hand.",
            "tags": {"conservative": "Errs on the side of flagging", "optimistic": "Can miss problems",
                     "check": "Assumption to confirm"}}


def _run(data: bytes, filename: str, sheet: Optional[str], time_column: Optional[str], units: Optional[str]) -> dict:
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_MB:g} MB.")
    if not data:
        raise HTTPException(422, "The file is empty.")
    try:
        unit_map = json.loads(units) if units else {}
        if not isinstance(unit_map, dict):
            raise ValueError("units must be a JSON object {column: unit}")
        with _HEAVY:
            df, info = load_table(data, filename, sheet=sheet or None, max_rows=MAX_ROWS)
            unit_map = {**info.units_from_file, **unit_map}
            report = analyze(df, units=unit_map, time_column=time_column or None)
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError, UnicodeError, OSError) as e:
        raise HTTPException(422, f"Could not analyse '{filename}': {e}") from e
    except Exception as e:  # pandas / pyarrow / h5py parser errors
        raise HTTPException(422, f"Could not read '{filename}': {type(e).__name__}: {e}") from e
    report["file"] = {"name": filename, "bytes": len(data), **info.__dict__}
    report["roles"] = po.suggest_roles(report["columns"])
    report["scope"] = scope()
    report["columns_all"] = list(df.columns)
    return report


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "formats": FORMATS, "max_mb": MAX_MB, "max_rows": MAX_ROWS}


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), sheet: Optional[str] = Form(None),
                      time_column: Optional[str] = Form(None), units: Optional[str] = Form(None)):
    """Upload a table; returns the quality report (JSON). `units` is an optional JSON object
    {column: unit} that overrides units read from headers."""
    data = await file.read()
    return _run(data, file.filename or "upload.csv", sheet, time_column, units)


def _example_csv() -> bytes:
    df, _ = example_chiller_plant()
    return df.to_csv(index=False).encode()


@app.get("/api/example")
def api_example():
    """Report on the built-in example (chilled-water plant with 8 injected faults) plus the answer key."""
    rep = _run(_example_csv(), "chiller_plant_week.csv", None, None, None)
    rep["answer_key"] = example_chiller_plant()[1]
    return rep


@app.get("/api/example.csv")
def api_example_csv():
    return Response(_example_csv(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="chiller_plant_week.csv"'})


# ── optimization ────────────────────────────────────────────────────────────
OPT_VALIDATION = [
    "Simulated chilled-water plant with known physics (60 days, 15-min, operators' manual setpoint habits): the "
    "model predicts plant power within 4.3 kW MAE (R² 0.99) on the last 15 days it never saw",
    "Predicted saving 8.3 % against a true 9.0 % for the example's seed. Over 8 further independent simulations the "
    "predicted saving averages 9.8 % against 9.7 % true (bias +0.1 points, worst case 1.2 points, in either direction)",
    "The CHW return-temperature limit, violated 4.0 % of the time by the operators, is violated 0.2 % of the "
    "time with the recommendations",
]


def opt_scope() -> dict:
    return {"validated": OPT_VALIDATION, "title": "Optimization scope & validation", "noun": "limitation",
            "items_title": "What the optimization assumes",
            "band": "Savings are model estimates from historical correlations; confirm with a supervised trial.",
            "tags": {"conservative": "Errs on the safe side", "optimistic": "Can overstate", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Correlation, not causation", "effect": "optimistic",
                 "detail": "The model learns how the KPI moved with the levers in the past. If operators always changed "
                           "a setpoint together with something unrecorded, part of that effect is attributed to the lever.",
                 "today": "Run the recommended schedule on alternate days (A/B) for 2-4 weeks and compare measured "
                          "consumption at equal weather and load.",
                 "planned": "Built-in trial planner and measurement & verification (IPMVP option B) report."},
                {"topic": "Model fit", "effect": "check",
                 "detail": "Training error (2.1 kW) is below the sensor-noise floor (about 2.8 kW): the model fits a little "
                           "noise. The saving estimate uses the held-out error (4.3 kW), which includes that effect. "
                           "Outside the operated range the error grows by about 30 %.",
                 "today": "Judge the model by the held-out numbers; collect data at new setpoints before trusting them.",
                 "planned": "Time-series cross-validation to tune regularization automatically."},
                {"topic": "Only inside past operation", "effect": "conservative",
                 "detail": "Recommendations are restricted to lever + context combinations similar to recorded ones, and "
                           "each move is limited. Larger savings outside that envelope are not explored.",
                 "today": "Widen the range gradually: test new setpoints deliberately so the next model can learn them.",
                 "planned": "Active learning: propose the most informative next tests."},
                {"topic": "Steady hourly decisions", "effect": "check",
                 "detail": "Each sample is optimized on its own; dynamics (thermal storage, ramp limits, equipment "
                           "starts) are not modelled.",
                 "today": "Apply the schedule as a setpoint reset table, not as minute-by-minute control.",
                 "planned": "Sequence-aware models and staging decisions."},
                {"topic": "Constraints you declare", "effect": "check",
                 "detail": "Only the constraints listed are enforced (each with its own model and a one-error margin). "
                           "Comfort, humidity or product-quality limits that are not in the data cannot be checked.",
                 "today": "Add every limit that matters as a constraint column, or narrow the lever bounds.",
                 "planned": "Constraint templates per plant type."},
            ]}


def _opt_run(df, report, cfg: dict, extra=None) -> dict:
    clean, log = po.clean_for_modeling(df, report)
    bounds = {k: tuple(v) for k, v in (cfg.get("bounds") or {}).items() if isinstance(v, (list, tuple)) and len(v) == 2}
    try:
        with _HEAVY:
            res = po.fit_and_optimize(clean, cfg.get("target"), cfg.get("levers") or [], cfg.get("context") or [],
                                      goal=cfg.get("goal", "minimize"), constraints=[dict(c) for c in cfg.get("constraints") or []],
                                      bounds=bounds, max_move=float(cfg.get("max_move", 0.5)))
    except ImportError as e:
        raise HTTPException(500, "scikit-learn is not installed on the server.") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out = po._clean_json({k: v for k, v in res.items() if not k.startswith("_")})
    if extra:
        out.update(extra(res))
    out["cleaning"] = log
    out["scope"] = opt_scope()
    return out


@app.post("/api/optimize")
async def api_optimize(file: UploadFile = File(...), config: str = Form(...), sheet: Optional[str] = Form(None),
                       time_column: Optional[str] = Form(None), units: Optional[str] = Form(None)):
    """Train a model of the KPI from levers + context and recommend lever settings. `config` (JSON):
    {target, goal: minimize|maximize, levers: [...], context: [...], constraints: [{column, op: "<="|">=", value}],
    bounds: {lever: [lo, hi]}, max_move: 0-1}."""
    data = await file.read()
    try:
        cfg = json.loads(config)
    except ValueError as e:
        raise HTTPException(422, f"config is not valid JSON: {e}") from e
    rep = _run(data, file.filename or "upload.csv", sheet, time_column, units)
    df, info = load_table(data, file.filename or "upload.csv", sheet=sheet or None, max_rows=MAX_ROWS)
    return _opt_run(df, rep, cfg)


def _opt_example_frame():
    df, info = po.example_plant_operations()
    return df, info


@app.get("/api/optimize/example")
def api_optimize_example():
    """Columns, suggested roles and health report of the optimization example (a simulated plant)."""
    df, info = _opt_example_frame()
    rep = _run(df.to_csv(index=False).encode(), "plant_operations_60d.csv", None, None, None)
    rep["roles"] = {"target": info["target"], "goal": "minimize", "levers": info["levers"], "context": info["context"],
                    "constraints": info["constraints"]}
    rep["example_info"] = info["description"]
    return rep


@app.post("/api/optimize/example")
def api_optimize_example_run(config: str = Form(...)):
    df, info = _opt_example_frame()
    rep = analyze(df)
    return _opt_run(df, rep, json.loads(config), extra=lambda res: {"answer_key": po.true_saving(res)})


@app.get("/api/optimize/example.csv")
def api_optimize_example_csv():
    df, _ = _opt_example_frame()
    return Response(df.to_csv(index=False).encode(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="plant_operations_60d.csv"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
