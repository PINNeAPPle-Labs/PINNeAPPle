"""Reduced Order Model Lab: web API + single-page UI. Build reduced models from field snapshots, measure their error on
held-out times or parameters and their speed, and export the one you choose.

Run:  uvicorn rom_lab.api:app --port 8102   (from apps/rom_lab)

Environment (all optional; the shared runtime adds ROM_DATA_DIR, ROM_ADMIN_TOKEN, ROM_REQUIRE_API_KEY, ...):
  ROM_USER / ROM_PASSWORD   HTTP Basic login on everything except /health
  ROM_MAX_JOBS              studies running at the same time (default 2)
  ROM_MAX_MB                largest upload in MB (default 200)
  ROM_MAX_VALUES            largest snapshot (values per snapshot, default 400,000)
  ROM_MAX_BUDGET_S          largest neural-surrogate budget a study may ask for, in seconds (default 30)
  ROM_FREE_STUDIES / ROM_FREE_EXPORTS / ROM_FREE_QUERIES   free use per caller and month (defaults 30 / 100 / 20,000)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_core.jsonable import to_jsonable
from pinneapple_data.snapshots import FormatError, load_snapshots
from pinneapple_neural.architectures.rom.contracts import model_ref, study_result
from pinneapple_neural.architectures.rom.study import LABELS, METHODS, ReducedModel, rom_study
from pinneapple_neural.architectures.rom.views import reconstruction_view

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
EXAMPLES_DIR = HERE.parent / "examples"
MAX_MB = float(os.environ.get("ROM_MAX_MB") or "200")
MAX_VALUES = int(os.environ.get("ROM_MAX_VALUES") or "400000")
MAX_BUDGET = float(os.environ.get("ROM_MAX_BUDGET_S") or "30")

EXAMPLES = {
    "wake": {"title": "Cylinder wake at Re = 100: forecast the shedding",
             "file": "cylinder_wake_re100.npz",
             "note": "150 velocity snapshots of PINNeAPPle's lattice-Boltzmann solver (D = 20 cells, 30,000 steps), "
                     "60 × 25 points. The last 20 % of the times are held out: every model forecasts them from the "
                     "training window.",
             "options": {"ranks": [2, 4, 8, 16], "budget_s": 3, "split": "forecast", "tolerance": 0.05}},
    "beam": {"title": "Cantilever design space: displacement and stress for unseen designs",
             "file": "cantilever_designs.npz",
             "note": "100 solid-FEM cantilevers (height, width, load angle; Latin hypercube) with nodal displacement "
                     "and von Mises stress. 20 random designs are held out; height and width are regressed in log "
                     "scale (the stiffness goes as W·H³).",
             "options": {"ranks": [4, 8, 16], "budget_s": 3, "split": "random", "tolerance": 0.05,
                         "log_params": ["H", "W"]}},
}

app = FastAPI(title="Reduced Order Model Lab", version=VERSION,
              description="Upload field snapshots (OpenFOAM case with several times, a series of VTU/VTK/FRD results, "
                          "NPZ/HDF5 arrays or a CSV/Parquet table with one row per snapshot). POST /api/studies queues a "
                          "study: POD + RBF / GPR, DMD, Operator Inference and a neural surrogate at equal wall-clock "
                          "budget, at several ranks, scored on held-out times or parameters. Each model comes with its "
                          "test error, fit and inference time and speed-up, and exports as a portable .npz, an ONNX "
                          "graph for the model gateway and a ModelRef contract.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="ROM")
rt = install_runtime(app, prefix="ROM", name="rom", units={"studies": 30, "exports": 100, "queries": 20000},
                     upload_max_mb=MAX_MB, upload_suffixes={".npz", ".npy", ".h5", ".hdf5", ".csv", ".parquet",
                                                            ".json", ".zip", ".vtu", ".vtk", ".frd"})
INPUTS = rt.root / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)

LIMITS = [
    "Snapshots must share one mesh (same points or cells in the same order); no interpolation between meshes.",
    "Time-stepping ROMs (DMD, Operator Inference, the neural stepper) need a uniform time step and predict from the "
    "latest training snapshot before the requested time; POD + RBF in time interpolates and does not forecast.",
    "Errors are measured on the held-out snapshots of this dataset only: a ROM is not validated outside the training "
    "time window or parameter box (the export records that range).",
    "The full-order cost and the speed-up are shown only when the time per snapshot of the full-order model is known "
    "(given, or stored with the example); inference times are measured on this server.",
    "ROMs are linear in the POD basis: transport-dominated or strongly nonlinear problems need many modes.",
    "No physics check is run on the reduced fields (trust decision UNCHECKED in the ModelRef); validate them in the "
    "Physics Validation Lab.",
]
VALIDATED = [
    "A travelling wave that is an exact rank-4 linear system: DMD and Operator Inference forecast it to < 1e-3 at "
    "the right rank and fail below it (tests/test_rom_study.py)",
    "No reported error is below the POD projection floor at the same rank (tested)",
    "Exported .npz models predict exactly what the fitted ones do; the ONNX graphs agree to < 1e-4 (tested)",
    "Cylinder wake (LBM, Re 100): DMD at rank 16 forecasts 30 held-out snapshots to 0.8 % (example)",
]


def scope() -> dict:
    return {"validated": VALIDATED, "title": "Scope & limits", "noun": "limit", "items_title": "What the numbers mean",
            "band": "Every number on this page comes from a run on held-out snapshots of the uploaded data.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can hide an error", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Held-out data", "effect": "check",
                 "detail": "Time series: the last 20 % of the times (forecast) or interleaved times; parametric data: "
                           "random designs. A ROM that interpolates well may still extrapolate badly.",
                 "today": "Choose the split that matches the use (forecast or interpolation).", "planned": ""},
                {"topic": "Error measure", "effect": "check",
                 "detail": "Relative L2 per test snapshot; time series relative to the fluctuation about the training "
                           "mean (the mean alone would otherwise score near zero); several fields are scaled by their "
                           "training RMS, with per-field errors shown.",
                 "today": "Read the per-field errors when the fields matter differently.", "planned": ""},
                {"topic": "Equal budget", "effect": "check",
                 "detail": "The neural surrogate trains for the stated wall-clock budget on the same training snapshots; "
                           "the closed-form ROMs report the time they actually took.",
                 "today": "Raise the budget to give the network more time.", "planned": ""},
                {"topic": "Physics", "effect": "optimistic",
                 "detail": "A small L2 error does not guarantee conservation or boundary values in the reconstruction.",
                 "today": "Check exported predictions in the Physics Validation Lab.", "planned": ""},
            ]}


# ------------------------------------------------------------------------------------------------ the job
def _options(raw: dict) -> dict:
    o = dict(raw or {})
    out: dict[str, Any] = {}
    meths = o.get("methods") or None
    if meths:
        bad = [m for m in meths if m not in LABELS]
        if bad:
            raise HTTPException(422, f"unknown methods {bad}; methods: {sorted(LABELS)}")
        out["methods"] = list(meths)
    if o.get("ranks"):
        try:
            out["ranks"] = sorted({int(r) for r in o["ranks"] if 1 <= int(r) <= 256})
        except (TypeError, ValueError) as e:
            raise HTTPException(422, "ranks must be integers between 1 and 256") from e
    out["budget_s"] = float(min(max(float(o.get("budget_s") or 3.0), 0.2), MAX_BUDGET))
    out["split"] = str(o.get("split") or "auto")
    if out["split"] not in ("auto", "forecast", "interleaved", "random"):
        raise HTTPException(422, "split is auto, forecast, interleaved or random")
    out["test_fraction"] = float(min(max(float(o.get("test_fraction") or 0.2), 0.05), 0.5))
    out["tolerance"] = float(o["tolerance"]) if o.get("tolerance") not in (None, "") else 0.05
    out["log_params"] = [str(x) for x in (o.get("log_params") or [])]
    out["amplitude_split"] = bool(o.get("amplitude_split"))
    out["hf_seconds"] = float(o["hf_seconds"]) if o.get("hf_seconds") not in (None, "") else None
    out["seed"] = int(o.get("seed") or 0)
    return out


def _model_key(method: str, rank) -> str:
    return method if rank is None else f"{method}_r{rank}"


def run_study(params: dict, ctx) -> dict:
    t0 = time.perf_counter()
    path = Path(params["input"])
    snaps = load_snapshots([(path.name, path.read_bytes())])
    o = params["options"]
    if o["log_params"]:
        missing = [n for n in o["log_params"] if n not in snaps.param_names]
        if missing:
            raise ValueError(f"log-scale parameters {missing} are not parameters of the data ({snaps.param_names})")
    ctx.progress(0.02, f"{snaps.n} snapshots × {snaps.n_values} values")
    n_steps = {"n": 0}

    def progress(msg):
        n_steps["n"] += 1
        ctx.log(msg)
        ctx.progress(min(0.95, 0.05 + 0.04 * n_steps["n"]), msg)
    res = rom_study(snaps, methods=o.get("methods"), ranks=o.get("ranks"), split=o["split"],
                    test_fraction=o["test_fraction"], budget_s=o["budget_s"], seed=o["seed"],
                    hf_seconds=o["hf_seconds"], log_params=o["log_params"], amplitude_split=o["amplitude_split"],
                    progress=progress)
    ctx.progress(0.96, "writing the models and the contracts")
    out_dir = Path(ctx.workdir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    rows_by_key = {_model_key(r["method"], r["rank"]): r for r in res.rows}
    refs, files = [], []
    units = snaps.meta.get("units") or {}
    for (meth, rank), m in res.models.items():
        key = _model_key(meth, rank)
        b = m.to_bytes()
        (out_dir / "models" / f"{key}.rom.npz").write_bytes(b)
        ref = model_ref(m, name=f"{params['name']}-{key}", uri=f"{key}.rom.npz", payload=b, row=rows_by_key.get(key),
                        split=f"test ({res.split.how}, {len(res.split.test)} snapshots)", dataset_hash=data_hash,
                        dataset_name=params["name"], study_id=ctx.job_id, units=units)
        (out_dir / "models" / f"{key}.modelref.json").write_text(ref.model_dump_json(indent=2))
        refs.append(ref)
    best = res.best(o["tolerance"])
    best_key = _model_key(best["method"], best["rank"]) if best else None
    sr = study_result(res, study_id=ctx.job_id, title=f"ROM study of {params['name']}", tolerance=o["tolerance"],
                      models=[r for r in refs if r.uri == f"{best_key}.rom.npz"] if best_key else [], limits=LIMITS)
    (out_dir / "study_result.json").write_text(sr.model_dump_json(indent=2))
    files = ["study_result.json"] + sorted(f"models/{p.name}" for p in (out_dir / "models").iterdir())
    d = res.to_dict(tolerance=o["tolerance"])
    for r in d["rows"]:
        r["key"] = _model_key(r["method"], r["rank"])
        r["onnx"] = r["method"] in ReducedModel.ONNX_METHODS and not (r["method"] == "neural" and res.mode == "time") \
            and not o["amplitude_split"]
    out = {"study": d, "best_key": best_key, "snapshots": snaps.summary(), "files": files,
           "test_inputs": (snaps.times[res.split.test].tolist() if res.mode == "time"
                           else snaps.params[res.split.test].tolist()),
           "train_inputs": (snaps.times[res.split.train].tolist() if res.mode == "time"
                            else snaps.params[res.split.train].tolist()),
           "checks": [c.model_dump() for c in sr.checks], "limits": LIMITS, "options": o,
           "wall_s": time.perf_counter() - t0, "name": params["name"]}
    (out_dir / "study.json").write_text(json.dumps(to_jsonable(out)))
    return to_jsonable(out)


rt.jobs.register("rom_study", run_study)


def _stage(files: list[tuple], param_columns: list[str] | None = None) -> tuple[Path, Any]:
    """Read the upload now (format errors answer 422 at once) and store it as one .npz for the job."""
    if sum(len(b) for _, b in files) > MAX_MB * 2 ** 20:
        raise HTTPException(413, f"uploads larger than {MAX_MB:g} MB")
    try:
        files = _with_contract(files)
        snaps = load_snapshots(files, param_columns=param_columns or None)
    except (FormatError, ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    if snaps.n_values > MAX_VALUES:
        raise HTTPException(413, f"{snaps.n_values:,} values per snapshot: this server studies up to {MAX_VALUES:,}")
    p = INPUTS / f"{uuid.uuid4().hex}.npz"
    p.write_bytes(snaps.to_npz_bytes())
    return p, snaps


def _with_contract(files):
    """A table with a DatasetSpec (sidecar ``.contract.json`` or embedded): its input columns are the parameters."""
    side = [(n, b) for n, b in files if n.endswith(".contract.json")]
    tables = [(n, b) for n, b in files if n.lower().endswith((".csv", ".parquet")) and not n.endswith(".contract.json")]
    if len(tables) != 1:
        return [f for f in files if not f[0].endswith(".contract.json")]
    import tempfile

    from pinneapple_data.contracts import DatasetSpec, read_table
    name, b = tables[0]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / Path(name).name
        p.write_bytes(b)
        for n, sb in side:
            (Path(d) / Path(n).name).write_bytes(sb)
        df, contract = read_table(p)
    if not isinstance(contract, DatasetSpec) or not contract.inputs:
        return [f for f in files if not f[0].endswith(".contract.json")]
    ins = [c for c in contract.inputs if c not in ("t", "time")]
    keep = ins + [c for c in df.columns if c in ("t", "time")] + contract.outputs
    df = df[[c for c in keep if c in df.columns]].rename(columns={c: f"param_{c}" for c in ins})
    return [(Path(name).with_suffix(".csv").name, df.to_csv(index=False).encode())]


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_mb": MAX_MB, "max_values": MAX_VALUES, "max_budget_s": MAX_BUDGET,
            "methods": {k: LABELS[k] for k in LABELS}, "methods_by_data": METHODS, "scope": scope(),
            "limits": LIMITS, "examples": [{"id": k, **{kk: v for kk, v in e.items()}} for k, e in EXAMPLES.items()
                                           if (EXAMPLES_DIR / e["file"]).exists()]}


@app.post("/api/inspect")
async def inspect(files: Annotated[list[UploadFile], File()], param_columns: Annotated[str, Form()] = ""):
    """Read an upload and describe it (snapshots, fields, times or parameters) without running anything."""
    fl = [(f.filename or f"file_{i}", await f.read()) for i, f in enumerate(files)]
    p, snaps = _stage(fl, [c.strip() for c in param_columns.split(",") if c.strip()])
    p.unlink(missing_ok=True)
    return to_jsonable(snaps.summary())


@app.post("/api/studies")
async def create_study(request: Request, files: Annotated[list[UploadFile], File()],
                       options: Annotated[str, Form()] = "{}", param_columns: Annotated[str, Form()] = "",
                       name: Annotated[str, Form()] = ""):
    """Queue a study of the uploaded snapshots. ``options`` (JSON): methods, ranks, budget_s, split
    (auto | forecast | interleaved | random), test_fraction, tolerance, log_params, amplitude_split, hf_seconds,
    seed. Poll ``GET /api/jobs/{id}``."""
    try:
        o = _options(json.loads(options or "{}"))
    except json.JSONDecodeError as e:
        raise HTTPException(422, "options must be a JSON object") from e
    fl = [(f.filename or f"file_{i}", await f.read()) for i, f in enumerate(files)]
    p, snaps = _stage(fl, [c.strip() for c in param_columns.split(",") if c.strip()])
    nm = (name or Path(fl[0][0]).stem or "snapshots")[:60]
    return rt.submit(request, "rom_study", {"input": str(p), "options": o, "name": nm}, units={"studies": 1})


@app.post("/api/examples/{eid}")
def run_example(request: Request, eid: str):
    """Queue the study of an example dataset (offline, a few seconds)."""
    if eid not in EXAMPLES:
        raise HTTPException(404, "no such example")
    e = EXAMPLES[eid]
    return rt.submit(request, "rom_study", {"input": str(EXAMPLES_DIR / e["file"]), "options": _options(e["options"]),
                                            "name": Path(e["file"]).stem}, units={"studies": 1})


@app.get("/api/examples/{eid}/file")
def example_file(eid: str):
    if eid not in EXAMPLES:
        raise HTTPException(404, "no such example")
    return FileResponse(EXAMPLES_DIR / EXAMPLES[eid]["file"], filename=EXAMPLES[eid]["file"])


def _job_dir(request: Request, job_id: str) -> tuple[Path, dict]:
    job = rt.job_for(request, job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"the study is {job['status']}")
    return Path(rt.jobs.workdir) / job_id, job


def _load_model(d: Path, key: str) -> ReducedModel:
    p = d / "models" / f"{Path(key).name}.rom.npz"
    if not p.is_file():
        raise HTTPException(404, f"no model {key}")
    return ReducedModel.from_bytes(p.read_bytes())


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(request: Request, job_id: str, path: str):
    """A study artifact: study_result.json, models/<key>.rom.npz, models/<key>.modelref.json."""
    d, _ = _job_dir(request, job_id)
    p = (d / path).resolve()
    if not str(p).startswith(str(d.resolve())) or not p.is_file():
        raise HTTPException(404, "no such file")
    if p.name.endswith(".rom.npz"):
        rt.consume(request, "exports", 1, ref=job_id)
    return FileResponse(p, filename=p.name)


@app.get("/api/jobs/{job_id}/onnx/{key}")
def job_onnx(request: Request, job_id: str, key: str):
    """The model as ONNX (``x (n, inputs) -> snapshot``) plus, in the headers, nothing else: the gateway spec is at
    ``/api/jobs/{id}/gateway/{key}``."""
    d, _ = _job_dir(request, job_id)
    m = _load_model(d, key)
    try:
        b = m.to_onnx()
    except NotImplementedError as e:
        raise HTTPException(422, str(e)) from e
    rt.consume(request, "exports", 1, ref=job_id)
    return Response(b, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{key}.onnx"'})


@app.get("/api/jobs/{job_id}/gateway/{key}")
def job_gateway_spec(request: Request, job_id: str, key: str):
    """``spec`` for ``ModelGateway.publish(name, "<key>.onnx", kind="onnx", spec=...)``, with the held-out metrics."""
    d, job = _job_dir(request, job_id)
    m = _load_model(d, key)
    if m.method not in ReducedModel.ONNX_METHODS:
        raise HTTPException(422, f"{LABELS[m.method]} has no ONNX export; use the .rom.npz file")
    spec = m.onnx_spec()
    row = next((r for r in job["result"]["study"]["rows"] if r["key"] == key), {})
    spec["metrics"] = {k: v for k, v in {"test_rel_l2_mean": row.get("error_mean"),
                                         "test_rel_l2_max": row.get("error_max")}.items() if v is not None}
    return spec


@app.get("/api/jobs/{job_id}/reconstruction")
def reconstruction(request: Request, job_id: str, model: str, snapshot: int = 0, field: str = ""):
    """Truth, prediction and error of one held-out snapshot for one model, on the locations of a field."""
    d, job = _job_dir(request, job_id)
    res = job["result"]
    m = _load_model(d, model)
    snaps = load_snapshots([("s.npz", Path(_input_of(job)).read_bytes())])
    test = res["study"]["split"]["test"]
    if not 0 <= snapshot < len(test):
        raise HTTPException(422, f"snapshot is 0 .. {len(test) - 1}")
    i = test[snapshot]
    q = {"times": [snaps.times[i]]} if m.mode == "time" else {"params": snaps.params[i:i + 1]}
    pred = m.predict(**q)[0]
    f = next((x for x in snaps.fields if x["name"] == field), snaps.fields[0])
    tv = snaps.field_values(f["name"])[i]
    pv = snaps.field_values(f["name"], pred[None])[0]
    if f["components"] > 1:
        tv, pv = np.linalg.norm(tv, axis=-1), np.linalg.norm(pv, axis=-1)
    loc = snaps.locations.get(f["where"])
    xy = None
    if loc is not None and len(loc) == len(tv):
        ext = np.ptp(loc, axis=0)
        ax = np.argsort(-ext)[:2]
        xy = loc[:, np.sort(ax)]
    return to_jsonable({"field": f["name"], "components": f["components"], "where": f["where"],
                        "magnitude": f["components"] > 1, "xy": None if xy is None else xy.round(6).tolist(),
                        "truth": tv.tolist(), "prediction": pv.tolist(), "error": (pv - tv).tolist(),
                        "rel_l2": float(np.linalg.norm(pv - tv) / max(np.linalg.norm(tv), 1e-300)),
                        "input": q, "fields": [x["name"] for x in snaps.fields],
                        "unit": (snaps.meta.get("units") or {}).get(f["name"], "")})


def _view(request: Request, job_id: str, model: str, snapshot: int) -> tuple[Path, Path]:
    """The 3-D view of one held-out snapshot (cached in the study folder): description + glTF."""
    d, job = _job_dir(request, job_id)
    res = job["result"]
    test = res["study"]["split"]["test"]
    if not 0 <= snapshot < len(test):
        raise HTTPException(422, f"snapshot is 0 .. {len(test) - 1}")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out = d / "views"
    js, glb = out / f"{safe}_{snapshot}.json", out / f"{safe}_{snapshot}.glb"
    if js.is_file() and glb.is_file():
        return js, glb
    m = _load_model(d, model)
    snaps = load_snapshots([("s.npz", Path(_input_of(job)).read_bytes())])
    row = next((r for r in res["study"]["rows"] if r["key"] == model), {})
    what = "snapshot" if res["study"]["mode"] == "time" else "design"
    errs = row.get("errors") or []
    reported = {f"study error of this {what}, all fields": errs[snapshot] if snapshot < len(errs) else None,
                f"study mean over {len(test)} held-out {what}s": row.get("error_mean")}
    try:
        fv, info = reconstruction_view(snaps, m, test[snapshot], reported=reported)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out.mkdir(exist_ok=True)
    glb.write_bytes(fv.glb())
    js.write_text(json.dumps(to_jsonable({"view": fv.meta, **info, "model": model, "snapshot": snapshot,
                                          "label": row.get("label", m.method), "rank": m.rank})))
    return js, glb


@app.get("/api/jobs/{job_id}/view3d")
def view3d(request: Request, job_id: str, model: str, snapshot: int = 0):
    """3-D view of a held-out snapshot on its mesh: the full-order field, the ROM prediction and |ROM - full-order|
    (deformed when the data has a displacement), the dataset's boundary-condition and load regions, the error
    hotspots and, per field, where the error concentrates. The glTF is at ``view3d.glb`` with the same query."""
    js, _ = _view(request, job_id, model, snapshot)
    out = json.loads(js.read_text())
    out["glb"] = f"/api/jobs/{job_id}/view3d.glb?model={quote(model)}&snapshot={snapshot}"
    return out


@app.get("/api/jobs/{job_id}/view3d.glb")
def view3d_glb(request: Request, job_id: str, model: str, snapshot: int = 0):
    _, glb = _view(request, job_id, model, snapshot)
    return FileResponse(glb, media_type="model/gltf-binary")


def _input_of(job: dict) -> str:
    p = job["params"]["input"]
    if not Path(p).is_file():
        raise HTTPException(410, "the input of this study is no longer stored")
    return p


@app.post("/api/run")
async def run_model(request: Request, model: Annotated[UploadFile, File()], inputs: Annotated[str, Form()],
                    fmt: Annotated[str, Form()] = "json"):
    """Run an exported ROM (.rom.npz). ``inputs`` (JSON): ``{"times": [...]}`` or ``{"params": [[...], ...]}`` /
    ``{"<param>": [...], ...}``. Returns the snapshots (JSON or CSV, one row per query) and whether each query is
    inside the training range."""
    b = await model.read()
    try:
        m = ReducedModel.from_bytes(b)
        q = json.loads(inputs)
    except (ValueError, KeyError, OSError) as e:
        raise HTTPException(422, f"not a ROM file or bad inputs: {e}") from e
    if m.mode == "parametric":
        names = m.config["param_names"]
        P = np.asarray(q["params"], float) if "params" in q else np.column_stack([np.atleast_1d(q[n]) for n in names])
        kw = {"params": np.atleast_2d(P)}
        n = len(kw["params"])
    else:
        kw = {"times": np.atleast_1d(np.asarray(q.get("times", q.get("t")), float))}
        n = len(kw["times"])
    if n > 1000:
        raise HTTPException(413, "at most 1000 queries per call")
    rt.consume(request, "queries", n)
    t0 = time.perf_counter()
    try:
        Y = m.predict(**kw)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    dt = time.perf_counter() - t0
    inside = m.in_range(**kw)
    if fmt == "csv":
        cols = ReducedModel.onnx_spec(m)["outputs"]
        head = (["t"] if m.mode == "time" else list(m.config["param_names"])) + ["in_range"] + cols
        X = kw["times"][:, None] if m.mode == "time" else kw["params"]
        buf = io.StringIO()
        buf.write(",".join(head) + "\n")
        for x, ok, y in zip(X, inside, Y, strict=True):
            buf.write(",".join([f"{v:.9g}" for v in x] + [str(bool(ok))] + [f"{v:.9g}" for v in y]) + "\n")
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="rom_predictions.csv"'})
    return to_jsonable({"method": m.method, "rank": m.rank, "n": n, "seconds": dt, "in_range": inside.tolist(),
                        "fields": [f["name"] for f in m.config["fields"]], "snapshots": Y.tolist()})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
