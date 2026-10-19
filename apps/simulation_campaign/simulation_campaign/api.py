"""Simulation Campaign Planner: web API + single-page UI. Plan which simulations need to run: a design of experiments
over the parameter space, its coverage, the cases that repeat runs you already have, a run order, and the case files
for OpenFOAM, CalculiX or your own Python code.

Run:  uvicorn simulation_campaign.api:app --port 8096   (from apps/simulation_campaign)

Environment (all optional; the runtime variables are in apps/_shared/README.md):
  CAMPAIGN_USER / CAMPAIGN_PASSWORD   HTTP Basic login on everything except /health
  CAMPAIGN_MAX_JOBS                   campaigns planned at once (default 2)
  CAMPAIGN_FREE_CAMPAIGNS             free campaigns per principal and month (default 50)
  CAMPAIGN_FREE_CASES_PLANNED         free planned cases per principal and month (default 5000)
  CAMPAIGN_MAX_CASES                  largest campaign in cases (default 5000)
  CAMPAIGN_MAX_CASE_DIRS              solver case directories written per campaign (default 50; tables hold all)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_data.contracts import ParameterSpace, read_table
from pinneapple_data.doe import METHODS, plan_campaign
from pinneapple_security.archives import ArchiveLimits, UnsafeArchive, safe_extract_zip
from pinneapple_simulation.external_solvers import campaign_cases as cc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_CASES = int(os.environ.get("CAMPAIGN_MAX_CASES") or 5000)
MAX_CASE_DIRS = int(os.environ.get("CAMPAIGN_MAX_CASE_DIRS") or 50)
MAX_CASE_MB = int(os.environ.get("CAMPAIGN_MAX_CASE_MB") or 200)        # an uploaded base case, unzipped
TABLE_SUFFIXES = {".csv", ".parquet", ".json", ".h5", ".hdf5"}

app = FastAPI(title="Simulation Campaign Planner", version=VERSION,
              description="Plan a simulation campaign over a parameter space (geometry, materials, boundary "
                          "conditions, constraints, budget): design of experiments, space coverage, redundancy "
                          "against runs you already have, priorities, and case files for OpenFOAM, CalculiX and "
                          "Python. Inputs and outputs are PINNeAPPle contracts (ParameterSpace, StudyResult, "
                          "DatasetSpec skeleton). POST /api/campaigns starts a job; poll /api/jobs/{id}.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="CAMPAIGN")
rt = install_runtime(app, prefix="CAMPAIGN", name="campaign", units={"campaigns": 50, "cases_planned": 5000},
                     upload_suffixes=TABLE_SUFFIXES | {".zip", ".inp", ".yaml", ".yml"}, upload_max_mb=100)

PRESETS: dict[str, dict[str, Any]] = {
    "airfoil": {
        "title": "Airfoil polars with OpenFOAM (CST shape + angle of attack)",
        "space": lambda: cc.airfoil_space(24),
        "existing": "436 OpenFOAM runs of the zoo dataset airfoil_rans_cst_re4e6",
        "outputs": ["Cl", "Cd", "Cm"], "openfoam": "airfoil", "method": "infill",
        "note": "Seven inputs: 6 CST weights and the angle of attack, at Re = 4e6. 436 runs already exist; plan "
                "the next 24 where they add the most, and get simpleFoam cases for them.",
    },
    "cantilever": {
        "title": "Cantilever stiffness with CalculiX (geometry, material, load)",
        "space": lambda: cc.cantilever_space(30),
        "existing": None, "outputs": ["tip_deflection", "max_von_mises"], "calculix": "cantilever",
        "method": "maximin",
        "note": "Length, width, height, material (3 alloys) and tip load, with the slenderness constraint "
                "length >= 8 height. Each case gets a CalculiX deck and its beam-theory deflection for checking.",
    },
}

VALIDATION = [
    "Fractional factorials match the minimum-aberration designs of the literature: 2^(7-3) and 2^(8-4) of "
    "resolution IV, 2^(5-1) of resolution V, all with orthogonal, balanced main effects (tested)",
    "Latin hypercubes have one case per stratum in every parameter; maximin and Sobol designs beat random sampling "
    "on minimum distance and discrepancy (tested)",
    "Generated CalculiX cantilever decks run with CalculiX 2.21: tip deflections within 1 % of Euler-Bernoulli "
    "beam theory on the cases checked (0.991, 0.994, 0.990 of F L^3 / 3EI)",
    "The airfoil cases are written by the same library builder that generated the 436 OpenFOAM runs of the zoo "
    "dataset airfoil_rans_cst_re4e6",
    "Redundancy: in those 436 existing runs the planner finds 2 near-duplicate pairs (distance < 1 % of the "
    "diagonal), and 18 runs (NACA 4412 / 4415 polars) outside the CST bounds",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the plan does and does not tell you",
            "band": "Coverage is measured in the unit cube of the parameters; it says where you have no data, not "
                    "where the response is hard.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can look better than it is",
                     "check": "Assumption to confirm"},
            "items": [
                {"topic": "Coverage metric", "effect": "check",
                 "detail": "Distances are taken in the unit cube (log scales and categories mapped), so every "
                           "parameter weighs the same. A design that fills that cube can still under-sample a "
                           "narrow feature of the response (a stall angle, a buckling threshold).",
                 "today": "Use adaptive infill once some outputs exist; add cases by hand where you expect "
                          "sharp changes.", "planned": "Model-based infill from a surrogate's uncertainty."},
                {"topic": "Adaptive infill", "effect": "optimistic",
                 "detail": "It weights candidates by how much the existing outputs vary between neighbouring "
                           "runs: a heuristic, not an error estimate.",
                 "today": "Read the coverage before / after and the checks.",
                 "planned": "Surrogate Studio (day 14) uncertainty as the infill criterion."},
                {"topic": "Redundancy", "effect": "check",
                 "detail": "A case repeats an existing run when it is within the tolerance (default 1 % of the "
                           "unit-cube diagonal). Runs with different solver settings or meshes are not "
                           "distinguished unless those are parameters.",
                 "today": "Put solver settings that matter in the parameter space (as constants).",
                 "planned": "Matching on simulation metadata (Simulation Metadata API)."},
                {"topic": "Case files", "effect": "check",
                 "detail": "Solver cases are written for the first cases in priority order (CAMPAIGN_MAX_CASE_DIRS); "
                           "the tables, the Python runner and the OpenFOAM campaign plan cover every case.",
                 "today": "Download the plan and generate the rest locally with the same library call.",
                 "planned": "Streaming download of large campaigns."},
            ]}


# --------------------------------------------------------------------------------------------- request schema
class Edit(BaseModel):
    file: str
    key: str
    value: str


class CampaignRequest(BaseModel):
    preset: Literal["airfoil", "cantilever", "custom"] = "custom"
    space: dict[str, Any] | None = Field(default=None, description="ParameterSpace JSON (preset's when omitted)")
    n: int | None = Field(default=None, ge=1, description="Number of cases (default: the space's budget)")
    method: str = "lhs"
    seed: int = 0
    levels: int | dict[str, int] = 3
    tol: float | None = Field(default=None, gt=0, description="Repeat tolerance in the unit cube")
    drop_repeats: bool = True
    use_preset_existing: bool = True
    existing_upload: str | None = Field(default=None, description="Upload id of a table of runs already done")
    existing_id_column: str | None = None
    outputs: list[str] = Field(default_factory=list, description="Output names (adaptive infill, DatasetSpec)")
    openfoam: Literal["airfoil", "base"] | None = None
    openfoam_base_upload: str | None = Field(default=None, description="Upload id of a zipped base case")
    openfoam_edits: list[Edit] = Field(default_factory=list)
    calculix: Literal["cantilever", "template"] | None = None
    calculix_template_upload: str | None = None
    case_limit: int | None = Field(default=None, ge=0)
    name: str | None = None


def _space_for(req: dict[str, Any]) -> ParameterSpace:
    if req.get("space"):
        return ParameterSpace.from_dict(req["space"])
    if req.get("preset") in PRESETS:
        return PRESETS[req["preset"]]["space"]()
    raise ValueError("give a parameter space (or pick a preset)")


def _read_existing(path: Path):
    df, _ = read_table(path)
    return df


def _unzip_case(zip_path: Path, dest: Path) -> Path:
    try:
        safe_extract_zip(zip_path, dest, ArchiveLimits(max_members=5000, max_total_bytes=MAX_CASE_MB * 2**20))
    except UnsafeArchive as exc:
        raise ValueError(str(exc)) from None
    for d in [dest, *sorted(p for p in dest.rglob("*") if p.is_dir())]:
        if (d / "system" / "controlDict").is_file():
            return d
    raise ValueError("the zip holds no OpenFOAM case (system/controlDict)")


# --------------------------------------------------------------------------------------------- the job
def plan_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Runs in the job queue: design, redundancy, priorities, case files, archive."""
    req = params
    ctx.progress(0.05, "reading the parameter space")
    space = _space_for(req)
    preset = PRESETS.get(req.get("preset") or "")
    existing, id_col, source = None, req.get("existing_id_column"), None
    if req.get("existing_path"):
        existing, source = _read_existing(Path(req["existing_path"])), "uploaded table"
    elif preset and req.get("use_preset_existing", True) and req.get("preset") == "airfoil":
        ctx.progress(0.1, "loading the 436 existing OpenFOAM runs")
        existing, id_col, source = cc.airfoil_existing(), "run_id", preset["existing"]
    outputs = [o for o in req.get("outputs") or [] if existing is None or o in existing.columns] \
        if req.get("method") == "infill" else []
    feasible = cc.airfoil_feasible if req.get("preset") == "airfoil" else None
    ctx.progress(0.2, f"design: {req.get('method')}")
    plan = plan_campaign(space, req.get("n"), req.get("method", "lhs"), existing=existing, existing_id_column=id_col,
                         outputs=outputs, seed=int(req.get("seed") or 0), levels=req.get("levels") or 3,
                         tol=req.get("tol"), drop_repeats=req.get("drop_repeats", True), feasible=feasible,
                         name=req.get("name") or space.name)
    if len(plan.cases) > MAX_CASES:
        raise ValueError(f"{len(plan.cases)} cases: this server plans up to {MAX_CASES}")
    ctx.log(f"{len(plan.cases)} cases planned, {len(plan.dropped)} dropped as repeats")
    ctx.progress(0.5, "writing the case files")
    out = ctx.workdir / "campaign"
    if out.exists():
        shutil.rmtree(out)
    limit = min(req["case_limit"], MAX_CASE_DIRS) if req.get("case_limit") is not None else MAX_CASE_DIRS
    base = None
    if req.get("openfoam") == "base":
        base = _unzip_case(Path(req["openfoam_base_path"]), ctx.workdir / "base_upload")
    template = Path(req["calculix_template_path"]).read_text() if req.get("calculix_template_path") else None
    outs = req.get("outputs") or (preset["outputs"] if preset else [])
    files = cc.write_campaign(plan, out, outputs=outs, openfoam=req.get("openfoam"), calculix=req.get("calculix"),
                              openfoam_base=base, openfoam_edits=req.get("openfoam_edits") or [],
                              calculix_template=template, case_limit=limit)
    ctx.progress(0.85, "packing the archive")
    archive = ctx.workdir / "campaign.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, Path("campaign") / p.relative_to(out))
    rt.meter.record(req["_owner"], "cases_planned", len(plan.cases), ref=ctx.job_id)
    return _result(plan, files, source, archive)


def _result(plan, files, source, archive: Path) -> dict[str, Any]:
    names = [p.name for p in plan.space.free]
    cols = list(plan.cases.columns)
    view: dict[str, Any] = {"names": names, "planned": plan.unit.round(5).tolist(),
                            "labels": {p.name: f"{p.name}" + (f" [{p.unit}]" if p.unit else "") for p in plan.space.free},
                            "kinds": {p.name: p.kind for p in plan.space.free},
                            "values": {p.name: p.values for p in plan.space.free if p.kind == "categorical"},
                            "bounds": {p.name: [p.low, p.high, p.log] for p in plan.space.free
                                       if p.kind in ("continuous", "integer")}}
    if plan.existing_unit is not None:
        view["existing"] = plan.existing_unit.round(5).tolist()
    if len(plan.dropped):
        view["dropped"] = plan.space.to_unit(plan.dropped).round(5).tolist()
    summary = plan.summary()
    summary["existing_source"] = source
    return {"summary": summary, "checks": [c.model_dump() for c in plan.checks()],
            "space": plan.space.to_dict(), "columns": cols,
            "cases": json.loads(plan.cases.head(500).to_json(orient="records")), "n_cases": int(len(plan.cases)),
            "dropped": json.loads(plan.dropped.head(200).to_json(orient="records")), "view": view,
            "files": {"case_dirs": max(len(files.get("openfoam", [])), len(files.get("calculix", []))),
                      "openfoam": bool(files.get("openfoam")), "calculix": bool(files.get("calculix")),
                      "openfoam_plan": bool(files.get("openfoam_plan")),
                      "archive_mb": round(archive.stat().st_size / 2**20, 2)},
            "scope": scope()}


rt.jobs.register("plan", plan_job)


# --------------------------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "methods": list(METHODS), "max_cases": MAX_CASES, "max_case_dirs": MAX_CASE_DIRS,
            "presets": {k: {"title": v["title"], "note": v["note"], "space": v["space"]().to_dict(),
                            "existing": v["existing"], "outputs": v["outputs"], "method": v["method"],
                            "openfoam": v.get("openfoam"), "calculix": v.get("calculix")}
                        for k, v in PRESETS.items()},
            "runtime": rt.describe(), "scope": scope()}


def _upload_path(request: Request, upload_id: str | None, suffixes: set[str], what: str) -> str | None:
    if not upload_id:
        return None
    p = rt.upload_path(request, upload_id)
    if p.suffix.lower() not in suffixes:
        raise HTTPException(422, f"{what} must be one of {sorted(suffixes)}")
    return str(p)


def _estimate(req: CampaignRequest, space: ParameterSpace) -> int:
    if req.method == "fractional_factorial":
        return 2 ** max(1, (req.n or space.budget or 2).bit_length())
    if req.method == "full_factorial":
        size = 1
        for p in space.free:
            lv = req.levels.get(p.name, 3) if isinstance(req.levels, dict) else req.levels
            size *= len(p.values) if p.kind == "categorical" else int(lv)
        return size
    return int(req.n or space.budget or 0)


@app.post("/api/campaigns")
def create_campaign(request: Request, req: CampaignRequest):
    """Start planning a campaign (a job). Poll ``GET /api/jobs/{id}``; downloads at
    ``/api/campaigns/{id}/download/{file}`` once done."""
    if req.method not in METHODS:
        raise HTTPException(422, f"method must be one of {list(METHODS)}")
    try:
        space = _space_for(req.model_dump())
    except Exception as exc:  # noqa: BLE001 - a readable 422 for any invalid space
        raise HTTPException(422, f"invalid parameter space: {exc}") from exc
    n_est = _estimate(req, space)
    if n_est < 1:
        raise HTTPException(422, "give the number of cases (n) or a budget in the parameter space")
    if n_est > MAX_CASES:
        raise HTTPException(422, f"about {n_est} cases: this server plans up to {MAX_CASES}")
    if req.method == "infill" and not (req.existing_upload or (req.preset == "airfoil" and req.use_preset_existing)):
        raise HTTPException(422, "infill needs existing runs: upload a table or use the airfoil preset")
    if req.openfoam == "airfoil" and req.preset != "airfoil":
        raise HTTPException(422, "OpenFOAM airfoil cases need the airfoil preset (its parameters)")
    if req.calculix == "cantilever" and req.preset != "cantilever":
        raise HTTPException(422, "CalculiX cantilever decks need the cantilever preset (its parameters)")
    params = req.model_dump()
    params["space"] = space.to_dict()
    params["existing_path"] = _upload_path(request, req.existing_upload, TABLE_SUFFIXES, "existing runs")
    params["openfoam_base_path"] = _upload_path(request, req.openfoam_base_upload, {".zip"}, "the base case")
    params["calculix_template_path"] = _upload_path(request, req.calculix_template_upload, {".inp"}, "the template")
    if req.openfoam == "base" and not params["openfoam_base_path"]:
        raise HTTPException(422, "OpenFOAM base-case mode needs a zipped case (openfoam_base_upload)")
    if req.calculix == "template" and not params["calculix_template_path"]:
        raise HTTPException(422, "CalculiX template mode needs an .inp (calculix_template_upload)")
    params["_owner"] = principal(request)
    rt.meter.check(principal(request), "cases_planned", n_est)
    return rt.submit(request, "plan", params, units={"campaigns": 1})


@app.post("/api/example/{preset}")
def example(request: Request, preset: str):
    """Try with example: the preset with its default design (seconds, offline)."""
    if preset not in PRESETS:
        raise HTTPException(404, "No such example.")
    p = PRESETS[preset]
    req = CampaignRequest(preset=preset, method=p["method"], n=24 if preset == "airfoil" else 30,
                          outputs=p["outputs"], openfoam=p.get("openfoam"), calculix=p.get("calculix"),
                          case_limit=4)
    return create_campaign(request, req)


@app.post("/api/space/from-table")
def space_from_table(request: Request, body: dict[str, Any]):
    """A ParameterSpace from an uploaded parameter table (one row per parameter: name, kind, low, high, values
    separated by |, unit, log, group)."""
    path = _upload_path(request, body.get("upload"), TABLE_SUFFIXES, "the parameter table")
    if not path:
        raise HTTPException(422, "give {'upload': <upload id>}")
    try:
        return ParameterSpace.from_table(path, name=body.get("name") or Path(path).stem).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"cannot read a parameter table: {exc}") from exc


@app.post("/api/existing/inspect")
def inspect_existing(request: Request, body: dict[str, Any]):
    """Columns and row count of an uploaded table of existing runs."""
    path = _upload_path(request, body.get("upload"), TABLE_SUFFIXES, "existing runs")
    if not path:
        raise HTTPException(422, "give {'upload': <upload id>}")
    try:
        df = _read_existing(Path(path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"cannot read the table: {exc}") from exc
    return {"rows": int(len(df)), "columns": [str(c) for c in df.columns],
            "numeric": [str(c) for c in df.columns if df[c].dtype.kind in "if"]}


DOWNLOADS = {"campaign.zip": "campaign.zip", "cases.csv": "campaign/cases.csv",
             "cases.parquet": "campaign/cases.parquet", "cases.json": "campaign/cases.json",
             "study.json": "campaign/study.json", "dataset.skeleton.json": "campaign/dataset.skeleton.json",
             "space.json": "campaign/space.json", "run_cases.py": "campaign/run_cases.py",
             "plan.json": "campaign/plan.json", "dropped_repeats.csv": "campaign/dropped_repeats.csv"}


@app.get("/api/campaigns/{job_id}/download/{name}")
def download(request: Request, job_id: str, name: str):
    job = rt.job_for(request, job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"campaign is {job['status']}")
    if name not in DOWNLOADS:
        raise HTTPException(404, "No such file.")
    path = rt.jobs.workdir / job_id / DOWNLOADS[name]
    if not path.is_file():
        raise HTTPException(404, f"{name} was not produced by this campaign")
    return FileResponse(path, filename=name if name != "campaign.zip" else f"{job_id}.zip")


@app.get("/api/example-file/{name}")
def example_file(name: str):
    files = {"parameters_cantilever.csv", "existing_airfoil_runs.csv"}
    if name not in files:
        raise HTTPException(404, "No such file.")
    with open(os.path.join(EXAMPLES, name), "rb") as f:
        return Response(f.read(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))

