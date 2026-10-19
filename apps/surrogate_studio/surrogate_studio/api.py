"""Surrogate Studio: turn simulation results into fast predictive models, measured on data they never saw.

Upload a table of runs (CSV, Parquet, JSON, HDF5, NPZ; a DatasetSpec embedded in Parquet / HDF5 or as a
.contract.json sidecar sets the column roles) or open an example, pick the inputs and outputs, and the study trains
every applicable method (pinneapple_analysis.surrogate.run_study) on the same splits and compute budget. The chosen
model is hosted at once behind the gateway API (/v1/models/..., the shape of the Model Gateway app), with its
ModelRef, StudyResult, limitations report and downloads.

Run:  uvicorn surrogate_studio.api:app --port 8097   (from apps/surrogate_studio)

Environment (all optional):
  SUR_USER / SUR_PASSWORD     HTTP Basic login on everything except /health
  SUR_MAX_JOBS                studies training at once (default 2); SUR_JOB_EXECUTOR thread | process
  SUR_MAX_BUDGET_S            largest training budget per method and seed, seconds (default 30)
  SUR_MAX_SEEDS               most seeds per study (default 3)
  SUR_MAX_ROWS                largest table, rows (default 20000)
  SUR_FREE_TRAININGS, SUR_FREE_INFERENCES, SUR_FREE_HOSTED_MODELS   free use per caller and month
  SUR_UPLOAD_MAX_MB, SUR_ADMIN_TOKEN, SUR_REQUIRE_API_KEY, SUR_DATA_DIR   see apps/_shared/README.md
  SUR_VIEW_MAX_FACES          triangles of a 3-D view before it is decimated (default 250,000)
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_analysis import surrogate as S
from pinneapple_analysis.surrogate import contracts as SC
from pinneapple_analysis.surrogate.heldout import heldout_report
from pinneapple_core.jsonable import to_jsonable
from pinneapple_registry.gateway import ModelGateway, ood_verdict
from pinneapple_registry.gateway_api import gateway_router

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).parent
STATIC = HERE / "static"
MAX_BUDGET = float(os.environ.get("SUR_MAX_BUDGET_S") or 30)
MAX_SEEDS = int(os.environ.get("SUR_MAX_SEEDS") or 3)
MAX_ROWS = int(os.environ.get("SUR_MAX_ROWS") or 20000)
SUFFIXES = {".csv", ".txt", ".parquet", ".pq", ".json", ".h5", ".hdf5", ".npz"}
VIEW_MAX_FACES = int(os.environ.get("SUR_VIEW_MAX_FACES") or 250_000)
# the 3-D FEM cantilever designs of the ROM Lab (apps/rom_lab/examples): one dataset, two apps
CANTILEVER = HERE.parents[1] / "rom_lab" / "examples" / "cantilever_designs.npz"

EXAMPLES: dict[str, dict[str, Any]] = {
    "airfoil_rans": {
        "title": "Airfoil polars from OpenFOAM (scalars)",
        "description": "374 steady RANS runs (OpenFOAM simpleFoam, k-omega SST, Re 4e6) of 100 CST airfoils: six "
                       "shape weights and the angle of attack -> Cl, Cd. Whole airfoils are held out as the "
                       "distribution-shift split.",
        "params": {"budget_s": 2.0, "seeds": 1, "model_name": "example-airfoil-cl-cd"}},
    "burgers_fields": {
        "title": "Viscous Burgers fields u(x, t) (fields + PDE)",
        "description": "96 exact solutions of u_t + u u_x = nu u_xx (Cole-Hopf) on a 48 x 12 grid for viscosities "
                       "0.004 .. 0.1; the PDE is declared, so a PINN competes. The smallest viscosities (steepest "
                       "fronts) are held out.",
        "params": {"budget_s": 2.0, "seeds": 1, "model_name": "example-burgers-u"}},
    "cantilever_3d": {
        "title": "3-D solid FEM cantilever (displacement and stress fields)",
        "description": "100 solid-FEM cantilevers (height, width, load angle; Latin hypercube), the ROM Lab's data: "
                       "displacement and von Mises stress on the 31 x 4 x 7 node lattice. The smallest load angles are "
                       "held out as the shift. The FEM is timed here (two designs solved again) for the speed-up; "
                       "the held-out designs open in 3-D.",
        "params": {"budget_s": 3.0, "seeds": 1, "model_name": "example-cantilever-3d"}},
}

app = FastAPI(title="Surrogate Studio", version=VERSION,
              description="Train surrogate models from simulation results and serve them. A study (POST "
                          "/api/studies) runs as a job; poll /api/jobs/{id}. The chosen model is served at "
                          "/v1/models/{name}/predict with its validity range enforced.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="SUR")
rt = install_runtime(app, prefix="SUR", name="surrogate",
                     units={"trainings": 30, "inferences": 20000, "hosted_models": 10},
                     upload_suffixes=SUFFIXES, upload_max_mb=100)
GW = ModelGateway(str(rt.root / "models"))
app.include_router(gateway_router(GW, on_rows=lambda req, n: rt.consume(req, "inferences", n),
                                  caller=principal, max_rows=5000))


# ---------------------------------------------------------------------------------------------- data
def _read(path: Path):
    """``(DataFrame, column roles or None)``; a DatasetSpec in the file or its sidecar gives the roles."""
    from pinneapple_data import contracts as C
    roles = None
    df = None
    if path.suffix.lower() != ".npz":
        try:
            df, contract = C.read_table(path)
            if isinstance(contract, C.DatasetSpec):
                roles = {c.name: c.role for c in contract.columns}
        except (ValueError, KeyError, OSError):
            df = None
    if df is None:
        df = S.read_table(path)
    if len(df) > MAX_ROWS:
        raise ValueError(f"{len(df)} rows; this server takes at most {MAX_ROWS} (SUR_MAX_ROWS)")
    return df, roles


def _data(params: dict[str, Any]) -> S.SurrogateData:
    if params.get("example") == "cantilever_3d":
        return S.examples.cantilever_3d(CANTILEVER)
    if params.get("example"):
        return S.examples.EXAMPLES[params["example"]]()
    df, _ = _read(Path(params["path"]))
    coords = None
    shape = params.get("grid_shape")
    if params.get("coord_ranges") and shape:
        coords = S.grid_coords(shape, [r for r in params["coord_ranges"]])
    meta = {"source": f"upload {params.get('filename', '')}"}
    if params.get("constraints"):
        meta["constraints"] = list(params["constraints"])
    return S.from_table(df, params["inputs"], params["outputs"], group=params.get("group") or None,
                        grid_shape=shape, coords=coords, coord_names=params.get("coord_names") or (), meta=meta)


def _parity(st: S.StudyResult, data: S.SurrogateData) -> dict[str, Any]:
    """Held-out predictions of the chosen model, for the parity plots and the field view."""
    m = st.model
    out: dict[str, Any] = {}
    for split in ("test", "shift"):
        idx = getattr(st.splits, split)[:400]
        if not len(idx):
            continue
        pred = m.predict_array(data.X[idx])
        if data.kind == "scalar":
            out[split] = {n: {"true": data.Y[idx, j].tolist(), "pred": pred[:, j].tolist()}
                          for j, n in enumerate(data.output_names)}
        else:
            err = np.linalg.norm((pred - data.Y[idx]).reshape(len(idx), -1), axis=1) / np.maximum(
                np.linalg.norm(data.Y[idx].reshape(len(idx), -1), axis=1), 1e-12)
            k = int(np.argmax(err))
            out[split] = {"rel_l2": err.tolist(), "inputs": data.X[idx].tolist(),
                          "worst": {"inputs": dict(zip(data.input_names, data.X[idx[k]].tolist(), strict=True)),
                                    "true": data.Y[idx[k]].tolist(), "pred": pred[k].tolist(),
                                    "rel_l2": float(err[k])}}
    if data.kind == "field":
        out["coords"] = {"names": data.coord_names, "shape": list(data.grid_shape),
                         "axes": [np.unique(data.coords[:, i]).tolist() for i in range(data.coords.shape[1])]}
    return out


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-._")[:60]
    return s or "model"


def study_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    ctx.progress(0.01, "reading the data")
    data = _data(params)
    n_methods = sum(1 for k, v in S.applicable(data.kind, int(data.n * 0.6), data.grid_shape,
                                                bool(params.get("pde") or data.meta.get("pde"))).items()
                    if v is None and (params.get("methods") in (None, "auto") or k in params["methods"]))
    total = max(1, n_methods * int(params["seeds"]))
    done = {"n": 0}

    def log(line: str):
        ctx.log(line)
        if line.endswith(": training"):
            done["n"] += 1
            ctx.progress(0.05 + 0.85 * (done["n"] - 1) / total, line.replace(": training", ""))

    ctx.log(f"{data.n} samples, {data.kind} outputs {data.output_names}; {total} training runs of "
            f"{params['budget_s']:g} s at most")
    split_options = {}
    if params.get("shift") not in (None, "auto"):
        split_options["shift"] = params["shift"] if params["shift"] != "none" else None
    st = S.run_study(data, methods=params.get("methods") or "auto", budget=float(params["budget_s"]),
                     seeds=int(params["seeds"]), split_options=split_options, pde=params.get("pde") or None,
                     target=params.get("target"), log=log, threads=int(os.environ.get("SUR_TORCH_THREADS") or 1))
    if st.model is None:
        raise RuntimeError("no candidate could be trained: " + "; ".join(st.notes))
    ctx.progress(0.92, "packaging and hosting the model")
    wd = Path(ctx.workdir)
    st.model.save(wd / "model")
    name = _slug(params.get("model_name") or f"{data.meta.get('dataset_spec') or 'surrogate'}-{ctx.job_id[-8:]}")
    # a REJECT model is hosted in "development" (callable for review, never "production"); see ModelStore
    stage = "development" if st.trust["decision"] == "REJECT" else "production"
    version = GW.publish(name, wd / "model", ood_policy=params.get("ood_policy", "warn"), stage=stage,
                         description=st.model.card.get("title", ""), source=f"surrogate-studio job {ctx.job_id}")
    uri = f"/v1/models/{name}/versions/{version}"
    ref = SC.model_ref(st.model, name=name, version=version, uri=uri, app="surrogate-studio",
                       dataset_name=str(data.meta.get("source", "")))
    result = SC.study_result(st, id=ctx.job_id, app="surrogate-studio", ref=ref)
    spec = SC.dataset_spec(data, data.meta.get("dataset_spec") or name, app="surrogate-studio")
    files = {"study_result.json": result.to_json(), "model_ref.json": ref.to_json(), "dataset_spec.json":
             spec.to_json(), "report.md": st.to_markdown(),
             "study.json": json.dumps(to_jsonable(st.to_dict()), indent=1)}
    for k, v in files.items():
        (wd / k).write_text(v)
    with zipfile.ZipFile(wd / "model.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in (wd / "model").iterdir():
            z.write(f, f"model/{f.name}")
        z.writestr("model/model_ref.json", files["model_ref.json"])
    exports = []
    for fmt, fname in (("onnx", "model.onnx"), ("torchscript", "model.pt")):
        try:
            st.model.export(wd / fname, fmt)
            exports.append(fname)
        except Exception as e:  # noqa: BLE001 (reported, not hidden)
            ctx.log(f"{fmt} export not available: {e}")
    heldout = _heldout(params, st, data)
    ctx.progress(1.0, "done")
    return to_jsonable({"heldout3d": heldout, "study": st.to_dict(), "model": {"name": name, "version": version, "uri": uri, "stage": stage,
                                                         "ood_policy": params.get("ood_policy", "warn")},
                        "model_ref": ref.to_dict(), "study_result_passed": result.passed,
                        "parity": _parity(st, data), "files": sorted([*files, "model.zip", *exports]),
                        "example": params.get("example")})


def _snapshots(params: dict[str, Any]):
    """The snapshot set behind a lattice field dataset (its meshes and node sets), or None."""
    if params.get("example") != "cantilever_3d":
        return None
    from pinneapple_data.snapshots import load_snapshots
    return load_snapshots([(CANTILEVER.name, CANTILEVER.read_bytes())])


def _pooling_note(rep: dict[str, Any]) -> str:
    """One line on why a pooled field error differs from the per-design error of the quantity read off it."""
    for n, f in rep.get("fields", {}).items():
        q = next((x for x in rep["qois"] if x["name"].startswith(f"max |{n}|")), None)
        ps = f.get("per_sample")
        if q is None or not ps:
            continue
        return (f"the pooled {n} error ({100 * f['rel_l2']:.1f} %) weighs each design by its squared {n}; the "
                f"{ps['heaviest']} most flexible of the {rep['n']} test designs hold {100 * ps['heaviest_weight']:.0f} % "
                f"of that weight (the worst of them is off by {100 * ps['heaviest_max']:.1f} %). Per design, the {n} "
                f"error averages {100 * ps['mean']:.1f} % (median {100 * ps['median']:.1f} %), in line with the "
                f"{100 * q['rel_error_mean']:.1f} % mean error of the {q['name']}.")
    return ""


def _heldout(params: dict[str, Any], st, data) -> dict[str, Any] | None:
    """Held-out errors per field, per region and on the scalar quantities, and the measured speed-up."""
    snaps = _snapshots(params)
    if snaps is None:
        return None
    sb = next(r for r in st.methods if r["id"] == st.best)["summary"]
    fem = data.meta.get("solver_s_per_sample")
    policy = params.get("ood_policy", "warn")
    reps = {}
    for split in ("test", "shift"):
        rep = heldout_report(snaps, data, st.model, getattr(st.splits, split))
        # the hosted /v1 endpoint's rule: the model's OOD gate and the policy the version is published with
        for smp, flag in zip(rep["samples"], st.model.gate.assess(data.X[rep.get("indices", [])]) if rep["n"] else [],
                             strict=True):
            smp["gate"] = {"verdict": ood_verdict(flag, policy), "reasons": flag["reasons"],
                           "knn_distance": flag["knn_distance"]}
        rep["gate"] = {v: sum(1 for x in rep["samples"] if x["gate"]["verdict"] == v)
                       for v in ("in range", "warn", "refuse")}
        reps[split] = rep
    return {"test": reps["test"], "shift": reps["shift"], "ood_policy": policy, "pooling_note": _pooling_note(reps["test"]),
            "fem_s": fem, "fem_timed_on": data.meta.get("solver_s_timed_on"), "infer_s": sb["infer_s_per_query"],
            "speedup": fem / sb["infer_s_per_query"] if fem and sb.get("infer_s_per_query") else None,
            "channels": data.meta.get("channels")}


rt.jobs.register("study", study_job)


# ---------------------------------------------------------------------------------------------- routes
class StudyBody(BaseModel):
    example: str | None = Field(default=None, description=f"one of {list(EXAMPLES)}")
    upload_id: str | None = Field(default=None, description="from POST /api/uploads")
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    group: str | None = None
    grid_shape: list[int] | None = None
    coord_names: list[str] | None = None
    coord_ranges: list[list[float]] | None = None
    pde: dict[str, Any] | None = None
    constraints: list[str] | None = None
    methods: list[str] | None = None
    budget_s: float = 5.0
    seeds: int = 1
    shift: Any = "auto"
    target: float | None = None
    model_name: str | None = None
    ood_policy: str = "warn"


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "examples": [{"name": k, "title": v["title"], "description": v["description"],
                                              "params": v["params"]} for k, v in EXAMPLES.items()],
            "methods": [{"id": c.id, "title": c.title, "family": c.family, "iterative": c.iterative}
                        for c in S.CANDIDATES.values()],
            "pde_types": {k: {"coords": list(v[0]), "coefficients": list(v[1])} for k, v in S.PDE_TYPES.items()},
            "limits": {"max_budget_s": MAX_BUDGET, "max_seeds": MAX_SEEDS, "max_rows": MAX_ROWS},
            "default_targets": S.DEFAULT_TARGETS, "runtime": rt.describe()}


OPTIONAL_FILE = File(default=None)


@app.post("/api/inspect")
async def inspect(request: Request, file: UploadFile | None = OPTIONAL_FILE, upload_id: str | None = None):
    """Columns, fields and suggested roles of a table: send the file, or the id of an earlier upload."""
    if file is not None:
        meta = rt.uploads.save(file.filename or "table.csv", file.file, owner=principal(request))
        upload_id = meta["id"]
    if not upload_id:
        raise HTTPException(422, "send a file or an upload_id")
    path = rt.upload_path(request, upload_id)
    try:
        df, roles = _read(path)
        info = S.inspect_table(df, roles)
    except (ValueError, KeyError, OSError) as e:
        raise HTTPException(422, str(e)) from None
    info.update(upload_id=upload_id, filename=path.name, roles_from_contract=roles is not None,
                preview=to_jsonable(df.head(5).to_dict(orient="records")))
    return info


@app.post("/api/studies")
def create_study(request: Request, body: StudyBody):
    p = body.model_dump()
    if body.example:
        if body.example not in EXAMPLES:
            raise HTTPException(404, f"unknown example {body.example!r}")
        p = {**p, **EXAMPLES[body.example]["params"], "example": body.example}
    else:
        if not body.upload_id:
            raise HTTPException(422, "give an example or an upload_id")
        path = rt.upload_path(request, body.upload_id)
        if not body.inputs or not body.outputs:
            raise HTTPException(422, "choose the input and output columns")
        p.update(path=str(path), filename=path.name)
        try:                                       # validate the column choice now, not in the queue
            _data(p)
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e)) from None
    if not 0.2 <= float(p["budget_s"]) <= MAX_BUDGET:
        raise HTTPException(422, f"budget_s must be between 0.2 and {MAX_BUDGET:g} s (SUR_MAX_BUDGET_S)")
    if not 1 <= int(p["seeds"]) <= MAX_SEEDS:
        raise HTTPException(422, f"seeds must be between 1 and {MAX_SEEDS}")
    if p["ood_policy"] not in ("warn", "refuse"):
        raise HTTPException(422, "ood_policy is warn or refuse")
    if p.get("methods"):
        bad = [m for m in p["methods"] if m not in S.CANDIDATES]
        if bad:
            raise HTTPException(422, f"unknown methods {bad}")
    return rt.submit(request, "study", p, units={"trainings": 1, "hosted_models": 1})


@app.get("/api/studies/{job_id}/files/{name}")
def study_file(request: Request, job_id: str, name: str):
    job = rt.job_for(request, job_id)
    if job["status"] != "done" or name not in (job.get("result") or {}).get("files", []):
        raise HTTPException(404, "no such file for this study")
    return FileResponse(rt.jobs.workdir / job_id / name, filename=f"{job_id}-{name}")


def _view(request: Request, job_id: str, split: str, k: int):
    """3-D view of held-out sample ``k`` of ``split``: the FEM field, the surrogate's and the error on the body."""
    job = rt.job_for(request, job_id)
    res = job.get("result") or {}
    if job["status"] != "done" or not res.get("heldout3d"):
        raise HTTPException(409, "only finished studies of mesh data (the 3-D cantilever) have a 3-D view")
    if split not in ("test", "shift"):
        raise HTTPException(422, "split is test or shift")
    rep = res["heldout3d"][split]
    if not 0 <= k < rep["n"]:
        raise HTTPException(422, f"k is 0 .. {rep['n'] - 1}")
    wd = rt.jobs.workdir / job_id
    js, glb = wd / "views" / f"{split}_{k}.json", wd / "views" / f"{split}_{k}.glb"
    if not (js.is_file() and glb.is_file()):
        from pinneapple_neural.architectures.rom.views import prediction_view
        snaps = _snapshots(job["params"])
        data = S.from_snapshots(snaps)
        model = S.SurrogateModel.load(wd / "model")
        i = rep["indices"][k]
        t0 = time.perf_counter()
        y = model.predict_array(data.X[i:i + 1])
        dt = time.perf_counter() - t0
        pred = S.snapshot_values(snaps, data, y)[0]
        fv, info = prediction_view(
            snaps, i, pred, label="Surrogate", full_label="FEM", note=f"surrogate: {res['study']['best']}",
            max_faces=VIEW_MAX_FACES,
            reported_by_field={n: {f"pooled over the {rep['n']} {split} designs": f["rel_l2"]}
                               for n, f in rep["fields"].items()})
        info.update(index=i, params=dict(zip(data.input_names, data.X[i].tolist(), strict=True)),
                    predict_s=dt, gate=rep["samples"][k].get("gate"), ood_policy=res["heldout3d"].get("ood_policy"),
                    negative={n: rep["samples"][k][f"negative_{n}"] for n in rep["fields"]
                              if f"negative_{n}" in rep["samples"][k]})
        js.parent.mkdir(exist_ok=True)
        glb.write_bytes(fv.glb())
        js.write_text(json.dumps(to_jsonable({"view": fv.meta, **info})))
    out = json.loads(js.read_text())
    return {**out, "glb": f"/api/studies/{job_id}/view3d.glb?split={split}&k={k}"}, glb


@app.get("/api/studies/{job_id}/view3d")
def view3d(request: Request, job_id: str, split: str = "test", k: int = 0):
    """A held-out design of a mesh study in 3-D: FEM, surrogate and |surrogate - FEM| on the deformed body, the
    clamp and load sets, the error hotspots and where the error concentrates. The glTF is at ``view3d.glb``."""
    return _view(request, job_id, split, k)[0]


@app.get("/api/studies/{job_id}/view3d.glb")
def view3d_glb(request: Request, job_id: str, split: str = "test", k: int = 0):
    return FileResponse(_view(request, job_id, split, k)[1], media_type="model/gltf-binary")


@app.get("/api/examples/{name}/data")
def example_data(name: str):
    """The example as a CSV, to see what an upload looks like (the cantilever: its snapshot file)."""
    if name not in EXAMPLES:
        raise HTTPException(404, "no such example")
    if name == "cantilever_3d":
        return FileResponse(CANTILEVER, filename=CANTILEVER.name)
    d = S.examples.EXAMPLES[name]()
    import pandas as pd
    cols = {n: d.X[:, i] for i, n in enumerate(d.input_names)}
    if d.kind == "field":
        flat = d.Y.reshape(d.n, -1)
        cols.update({f"{d.output_names[0]}_{i}": flat[:, i] for i in range(flat.shape[1])})
    else:
        cols.update({n: d.Y[:, i] for i, n in enumerate(d.output_names)})
    if d.groups is not None:
        cols[d.meta.get("group", "group")] = d.groups
    buf = io.StringIO()
    pd.DataFrame(cols).to_csv(buf, index=False)
    from fastapi.responses import Response
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
