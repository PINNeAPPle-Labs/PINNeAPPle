"""Physics AI Benchmark Arena: which method works best for each physical problem, under one protocol.

Public rankings come from benchmark runs that were actually executed (results/*.json, written by
run_public_benchmarks.py with their date, hardware and dataset hash), and from published runs of the
pinneapple-benchmark repository copied unchanged with their commit (results/external/<id>/). Private benchmarks run the same protocol
(pinneapple_arena.regimes.run_regime_arena) on an uploaded dataset as a job: equal data, equal compute budget, same
stopping rule, several seeds, per regime (interpolation, few data, extrapolation, geometry change, steep fronts),
with cost, physics violations and an independent audit of the splits.

Run:  uvicorn benchmark_arena.api:app --port 8105   (from apps/benchmark_arena)

Environment (all optional):
  ARN_USER / ARN_PASSWORD     HTTP Basic login on everything except /health
  ARN_MAX_JOBS                benchmarks running at once (default 1); ARN_JOB_EXECUTOR thread | process
  ARN_MAX_BUDGET_S            largest training budget per method, regime and seed (default 20)
  ARN_MAX_SEEDS               most seeds per benchmark (default 5)
  ARN_MAX_ROWS                largest table, rows (default 20000)
  ARN_FREE_BENCHMARKS         private benchmarks per caller and month (default 5)
  ARN_UPLOAD_MAX_MB, ARN_ADMIN_TOKEN, ARN_REQUIRE_API_KEY, ARN_DATA_DIR   see apps/_shared/README.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_analysis import surrogate as S
from pinneapple_analysis.surrogate import contracts as SC
from pinneapple_arena.regimes import NOT_IN_LIBRARY, REGIMES, arena_markdown, run_regime_arena
from pinneapple_core.jsonable import to_jsonable

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).parent
STATIC = HERE / "static"
RESULTS = HERE.parent / "results"
MAX_BUDGET = float(os.environ.get("ARN_MAX_BUDGET_S") or 20)
MAX_SEEDS = int(os.environ.get("ARN_MAX_SEEDS") or 5)
MAX_ROWS = int(os.environ.get("ARN_MAX_ROWS") or 20000)
SUFFIXES = {".csv", ".txt", ".parquet", ".pq", ".json", ".h5", ".hdf5", ".npz"}

app = FastAPI(title="Physics AI Benchmark Arena", version=VERSION,
              description="Rankings of physics-AI methods per data regime under one protocol: equal data, equal "
                          "compute, same stopping rule, several seeds. Public results at /api/public; private "
                          "benchmarks on your data at POST /api/benchmarks (a job; poll /api/jobs/{id}).")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="ARN")
rt = install_runtime(app, prefix="ARN", name="arena", units={"benchmarks": 5}, max_jobs=1,
                     upload_suffixes=SUFFIXES, upload_max_mb=100)


def _public() -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(RESULTS.glob("*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out


PUBLIC = _public()


def _external() -> dict[str, dict[str, Any]]:
    """Published results of other benchmark runs, copied unchanged with their source (results/external/<id>/)."""
    import csv
    out = {}
    for d in sorted((RESULTS / "external").glob("*/source.json")):
        src = json.loads(d.read_text())
        with open(d.parent / "table.csv", newline="") as f:
            rows = [{k: (float(v) if k in ("mean", "std", "n", "n_seeds") and v not in ("", None) else v)
                     for k, v in r.items()} for r in csv.DictReader(f)]
        hw = json.loads((d.parent / "hardware.json").read_text()) if (d.parent / "hardware.json").exists() else {}
        out[src["id"]] = {**src, "hardware": hw, "rows": rows}
    return out


EXTERNAL = _external()


def _summary(r: dict[str, Any]) -> dict[str, Any]:
    return {"id": r.get("id"), "title": r.get("title"), "created": r["created"], "cpu": r["hardware"].get("cpu"),
            "logical_cpus": r["hardware"].get("logical_cpus"), "winner": (r["overall"] or [{}])[0].get("method"),
            "n_methods": sum(m["applicable"] for m in r["methods"]), "seeds": r["protocol"]["seeds"],
            "budget": r["protocol"]["budget"], "kind": r["problem"]["kind"], "n_samples": r["problem"]["n_samples"],
            "regimes": list(r["rankings"])}


# ---------------------------------------------------------------------------------------------- private runs
def _data(p: dict[str, Any]) -> S.SurrogateData:
    df, _ = SC.read_table_with_roles(Path(p["path"]), max_rows=MAX_ROWS)
    shape = p.get("grid_shape")
    coords = S.grid_coords(shape, p["coord_ranges"]) if p.get("coord_ranges") and shape else None
    meta = {"source": f"upload {p.get('filename', '')}"}
    return S.from_table(df, p["inputs"], p["outputs"], group=p.get("group") or None, grid_shape=shape,
                        coords=coords, coord_names=p.get("coord_names") or (), meta=meta)


def arena_job(p: dict[str, Any], ctx) -> dict[str, Any]:
    ctx.progress(0.01, "reading the data")
    data = _data(p)
    applicable = [k for k, v in S.applicable(data.kind, int(data.n * 0.6), data.grid_shape,
                                             bool(p.get("pde"))).items()
                  if v is None and (not p.get("methods") or k in p["methods"])]
    n_sets = 2 + (data.groups is not None)
    total = max(1, len(applicable) * int(p["seeds"]) * n_sets)
    done = {"n": 0}

    def log(line: str):
        ctx.log(line)
        done["n"] += 1
        ctx.progress(0.03 + 0.92 * min(1.0, (done["n"] - 1) / total), line)

    ctx.log(f"{data.n} samples ({data.kind}); up to {total} training runs of {p['budget_s']:g} s")
    ext = {"column": p["extrapolation_column"], "side": p.get("extrapolation_side", "high")} \
        if p.get("extrapolation_column") else "edge"
    res = run_regime_arena(data, methods=p.get("methods") or "auto", budget=float(p["budget_s"]),
                           seeds=int(p["seeds"]), regimes=p.get("regimes") or "auto",
                           few_data_fraction=float(p.get("few_data_fraction", 0.25)), extrapolation=ext,
                           pde=p.get("pde") or None, constraints=p.get("constraints") or [], log=log,
                           threads=int(os.environ.get("ARN_TORCH_THREADS") or 1))
    title = p.get("title") or f"Private benchmark: {', '.join(data.output_names)} from {p.get('filename', '')}"
    doc = {"id": ctx.job_id, "title": title, "private": True, **to_jsonable(res.to_dict())}
    wd = Path(ctx.workdir)
    (wd / "result.json").write_text(json.dumps(doc, indent=1))
    (wd / "report.md").write_text(arena_markdown(doc, title))
    (wd / "study_result.json").write_text(SC.arena_study_result(doc, id=ctx.job_id, title=title).to_json())
    ctx.progress(1.0, "done")
    return {**doc, "files": ["result.json", "report.md", "study_result.json"]}


rt.jobs.register("arena", arena_job)


# ---------------------------------------------------------------------------------------------- routes
class BenchmarkBody(BaseModel):
    upload_id: str
    inputs: list[str]
    outputs: list[str]
    group: str | None = None
    grid_shape: list[int] | None = None
    coord_names: list[str] | None = None
    coord_ranges: list[list[float]] | None = None
    pde: dict[str, Any] | None = None
    constraints: list[str] | None = None
    methods: list[str] | None = None
    regimes: list[str] | None = None
    budget_s: float = 5.0
    seeds: int = 3
    few_data_fraction: float = 0.25
    extrapolation_column: str | None = Field(default=None, description="extrapolate along this input (else: edge)")
    extrapolation_side: str = "high"
    title: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "regimes": REGIMES, "not_in_library": NOT_IN_LIBRARY,
            "methods": [{"id": c.id, "title": c.title, "family": c.family} for c in S.CANDIDATES.values()],
            "pde_types": {k: {"coords": list(v[0]), "coefficients": list(v[1])} for k, v in S.PDE_TYPES.items()},
            "limits": {"max_budget_s": MAX_BUDGET, "max_seeds": MAX_SEEDS, "max_rows": MAX_ROWS},
            "public": [_summary(r) for r in PUBLIC.values()],
            "external": [{k: v for k, v in e.items() if k != "rows"} for e in EXTERNAL.values()],
            "runtime": rt.describe()}


@app.get("/api/public")
def public():
    return {"benchmarks": [_summary(r) for r in PUBLIC.values()]}


def _pub(bid: str) -> dict[str, Any]:
    if bid not in PUBLIC:
        raise HTTPException(404, "no such public benchmark")
    return PUBLIC[bid]


@app.get("/api/public/{bid}")
def public_one(bid: str):
    return _pub(bid)


@app.get("/api/public/{bid}/report.md")
def public_report(bid: str):
    r = _pub(bid)
    return Response(arena_markdown(r, r.get("title", "")), media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="arena-{bid}.md"'})


@app.get("/api/public/{bid}/study_result.json")
def public_contract(bid: str):
    r = _pub(bid)
    return Response(SC.arena_study_result(r, id=f"arena-{bid}", title=r.get("title", "")).to_json(),
                    media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="arena-{bid}.study_result.json"'})


@app.get("/api/external/{eid}")
def external(eid: str):
    """A published benchmark (e.g. pinneapple-benchmark experiment 09) as its source table, unchanged."""
    if eid not in EXTERNAL:
        raise HTTPException(404, "no such external benchmark")
    return EXTERNAL[eid]


OPTIONAL_FILE = File(default=None)


@app.post("/api/inspect")
async def inspect(request: Request, file: UploadFile | None = OPTIONAL_FILE, upload_id: str | None = None):
    """Columns, fields and suggested roles of a table: send the file, or the id of an earlier upload."""
    if file is not None:
        upload_id = rt.uploads.save(file.filename or "table.csv", file.file, owner=principal(request))["id"]
    if not upload_id:
        raise HTTPException(422, "send a file or an upload_id")
    path = rt.upload_path(request, upload_id)
    try:
        df, roles = SC.read_table_with_roles(path, max_rows=MAX_ROWS)
        info = S.inspect_table(df, roles)
    except (ValueError, KeyError, OSError) as e:
        raise HTTPException(422, str(e)) from None
    info.update(upload_id=upload_id, filename=path.name, roles_from_contract=roles is not None)
    return info


@app.post("/api/benchmarks")
def create(request: Request, body: BenchmarkBody):
    p = body.model_dump()
    path = rt.upload_path(request, body.upload_id)
    p.update(path=str(path), filename=path.name)
    if not 0.2 <= body.budget_s <= MAX_BUDGET:
        raise HTTPException(422, f"budget_s must be between 0.2 and {MAX_BUDGET:g} s (ARN_MAX_BUDGET_S)")
    if not 1 <= body.seeds <= MAX_SEEDS:
        raise HTTPException(422, f"seeds must be between 1 and {MAX_SEEDS} (ARN_MAX_SEEDS)")
    if not 0.05 <= body.few_data_fraction <= 0.9:
        raise HTTPException(422, "few_data_fraction must be between 0.05 and 0.9")
    bad = [m for m in (body.methods or []) if m not in S.CANDIDATES] + \
          [r for r in (body.regimes or []) if r not in REGIMES]
    if bad:
        raise HTTPException(422, f"unknown methods or regimes: {bad}")
    try:
        d = _data(p)
        if body.extrapolation_column and body.extrapolation_column not in d.input_names:
            raise ValueError(f"extrapolation column {body.extrapolation_column!r} is not an input")
        if body.pde:
            S.parse_pde(body.pde, d.coord_names, d.input_names)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from None
    return rt.submit(request, "arena", p, units={"benchmarks": 1})


@app.get("/api/benchmarks/{job_id}/files/{name}")
def files(request: Request, job_id: str, name: str):
    job = rt.job_for(request, job_id)
    if job["status"] != "done" or name not in (job.get("result") or {}).get("files", []):
        raise HTTPException(404, "no such file for this benchmark")
    return FileResponse(rt.jobs.workdir / job_id / name, filename=f"{job_id}-{name}")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
