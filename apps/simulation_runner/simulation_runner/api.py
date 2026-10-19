"""Simulation Workflow Runner: web API + single-page UI. Run a campaign of simulation cases as a DAG of steps,
with traceable runs, logs, states, retries, resume and skip-done, and turn finished runs into a training dataset.

Run:  uvicorn simulation_runner.api:app --port 8110   (from apps/simulation_runner)

Environment (all optional; runtime variables in apps/_shared/README.md):
  RUNNER_USER / RUNNER_PASSWORD   HTTP Basic login on everything except /health
  RUNNER_SELF_HOSTED              1: also accept any workflow (commands, scripts, Docker, Slurm submission).
                                  Never on a public server: it runs what the caller sends.
  RUNNER_MAX_CASES                cases per run (default 60)
  RUNNER_MAX_JOBS                 runs executing at once (default 2)
  RUNNER_FREE_JOBS_RUN, RUNNER_FREE_COMPUTE_SECONDS, RUNNER_FREE_STORAGE_MB   free use per month
  RUNNER_VIEW_MAX_FACES           triangles of a step's 3-D result view before it is decimated (default 250,000)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_core.jsonable import to_jsonable
from pinneapple_core.workflows import (
    EXECUTORS,
    TEMPLATES,
    Workflow,
    WorkflowError,
    WorkflowRunner,
    dataset_from_run,
    get_executor,
    id_column_of,
    read_cases,
)
from pinneapple_tools.visualization.studio.solver_view import solver_result_view

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
SELF_HOSTED = (os.environ.get("RUNNER_SELF_HOSTED") or "").lower() in ("1", "true", "yes")
MAX_CASES = int(os.environ.get("RUNNER_MAX_CASES") or 60)
VIEW_MAX_FACES = int(os.environ.get("RUNNER_VIEW_MAX_FACES") or 250_000)
VIEW_KINDS = ("calculix", "openfoam")                # steps whose result opens in the 3-D view
DATASET_FORMATS = {"csv": "text/csv", "parquet": "application/octet-stream", "json": "application/json",
                   "h5": "application/x-hdf5"}

app = FastAPI(title="Simulation Workflow Runner", version=VERSION,
              description="Run simulation campaigns as a DAG of steps (OpenFOAM, CalculiX, Python, commands; local, "
                          "Docker or Slurm executors): traceable runs, per-step logs and states, retries, resume of "
                          "interrupted campaigns, skip-done by content hash, and a DatasetSpec of the finished runs "
                          "for training. Public servers run only the built-in templates.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="RUNNER")
rt = install_runtime(app, prefix="RUNNER", name="runner",
                     units={"jobs_run": 200, "compute_seconds": 3600, "storage_mb": 500},
                     upload_suffixes={".csv", ".parquet", ".json", ".h5", ".hdf5"}, upload_max_mb=20)
RUNNER = WorkflowRunner(rt.root / "workflows")

try:    # the Engineering GUI Agent as a step kind ("guiagent"): FreeCAD/CfdOF/OpenFOAM/ParaView workflows as one step
    from pinneapple_tools.automation.runner_step import register as _register_guiagent
    _register_guiagent()
except ImportError:                                  # pragma: no cover - library without the automation package
    pass


def _executor_options(name: str, options: dict | None) -> dict:
    if SELF_HOSTED:
        return dict(options or {})
    return {"dry_run": True, **{k: v for k, v in (options or {}).items()
                                if k in ("partition", "time", "ntasks", "cpus_per_task", "mem")}} \
        if name == "slurm" else {}


def run_workflow_job(params: dict, ctx) -> dict:
    """Queue job: run (or resume) one workflow run, report progress, meter compute time and storage."""
    last = {"t": 0.0, "stop": False}

    def should_stop() -> bool:
        now = time.time()
        if now - last["t"] > 0.5:
            last["t"], last["stop"] = now, ctx.cancelled
        return last["stop"]

    def progress(counts: dict[str, int]) -> None:
        total = sum(counts.values()) or 1
        finished = total - counts["pending"] - counts["running"]
        ctx.progress(finished / total, f"{finished}/{total} steps finished ({counts['done']} ran, "
                                       f"{counts['skipped']} reused, {counts['failed']} failed)")

    run_id = params["run_id"]
    ctx.log(f"{'resuming' if params.get('resume') else 'starting'} run {run_id}")
    try:
        summary = RUNNER.run(run_id=run_id, should_stop=should_stop, progress=progress)
    except Exception:
        if ctx.cancelled:
            return {"run_id": run_id, "status": "cancelled"}
        raise
    seconds = RUNNER.compute_seconds(run_id)
    mb = RUNNER.storage_bytes(run_id) / 2**20
    owner = summary["metadata"].get("owner", "")
    rt.meter.record(owner, "compute_seconds", seconds, ref=run_id)
    rt.meter.record(owner, "storage_mb", mb, ref=run_id)
    ctx.log(f"run {run_id}: {summary['status']}, {seconds:.2f} s of compute, {mb:.2f} MB")
    return {"run_id": run_id, "status": summary["status"], "counts": summary["counts"],
            "compute_seconds": seconds, "storage_mb": mb}


rt.jobs.register("run", run_workflow_job)


def scope() -> dict:
    return {"title": "Scope & limits", "noun": "limit", "items_title": "What the runner does and does not do",
            "band": "It orchestrates existing solvers and schedulers; it does not replace them.",
            "validated": [
                "Skip-done: a rerun with unchanged parameters, inputs and upstream results reuses every step "
                "(tested: content hash of kind, parameters, input files and upstream hashes)",
                "Resume: an interrupted run continues with the steps that had not finished; finished steps are "
                "kept (tested)",
                "Slurm: sbatch scripts are generated per step and checked line by line (dry run, tested); "
                "submission needs a cluster",
                "The slab template's numbers come from the exact Fourier-series solution evaluated in each run "
                "(pinneapple_physics.closed_form.heat.slab_transient)"],
            "tags": {"conservative": "Safe by default", "optimistic": "Can hide a problem",
                     "check": "Assumption to confirm"},
            "items": [
                {"topic": "Public server", "effect": "conservative",
                 "detail": "Only the built-in templates run here, with case values checked against each template's "
                           "parameter ranges (numbers only), so nothing from the request reaches a shell or a solver "
                           "dictionary as text. Arbitrary commands, scripts and Docker need a self-hosted runner.",
                 "today": "Self-host (RUNNER_SELF_HOSTED=1) for your own workflows, behind your own login.",
                 "planned": "Signed workflow bundles from the Campaign Planner."},
                {"topic": "Solvers", "effect": "check",
                 "detail": "OpenFOAM and CalculiX templates run only where the solver is installed; otherwise they are "
                           "listed as 'needs solver' and nothing is faked.",
                 "today": "Use the self-hosted Docker image with OpenFOAM / CalculiX, or a Slurm cluster.",
                 "planned": "Solver containers selectable per step."},
                {"topic": "Skip-done", "effect": "optimistic",
                 "detail": "A step is reused when its definition, parameters, declared input files and upstream "
                           "results are unchanged. Files a step reads without declaring them are not in the hash, "
                           "and a new solver version with the same command is not noticed.",
                 "today": "Declare every input file in the step's inputs; put the solver version in its params.",
                 "planned": "Solver version capture per executor."},
                {"topic": "Scheduling", "effect": "check",
                 "detail": "Steps of a run execute in order of dependencies with a parallel limit per run; the "
                           "queue limits how many runs execute at once. It is not a cluster scheduler.",
                 "today": "Use the Slurm executor to hand steps to your cluster's scheduler.", "planned": ""},
            ]}


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


def _template_info(t) -> dict:
    ok, why = t.availability()
    return {"id": t.id, "title": t.title, "description": t.description, "solver": t.solver, "available": ok,
            "reason": why, "outputs": t.outputs, "output_step": t.output_step,
            "steps": [{"id": s.id, "kind": s.kind, "needs": s.needs, "final": s in t.final} for s in t.steps + t.final],
            "parameters": [p.model_dump(exclude_none=True) for p in t.space.parameters],
            "example_cases": t.example_cases(6).to_dict("records")}


@app.get("/api/meta")
def meta():
    execs = []
    for name in EXECUTORS:
        e = get_executor(name, **({"dry_run": True} if name == "slurm" else {}))
        ok, why = e.available()
        allowed = SELF_HOSTED or name in ("local", "slurm")
        execs.append({"name": name, "available": ok and allowed,
                      "reason": why if allowed else "self-hosted runners only",
                      "note": "writes sbatch scripts, does not submit" if name == "slurm" and not SELF_HOSTED else ""})
    from pinneapple_core.workflows import ADAPTERS
    kinds = [{"kind": k, "available": a.available()[0], "reason": a.available()[1]} for k, a in sorted(ADAPTERS.items())]
    return {"version": VERSION, "self_hosted": SELF_HOSTED, "max_cases": MAX_CASES,
            "templates": [_template_info(t) for t in TEMPLATES.values()], "executors": execs, "step_kinds": kinds,
            "runtime": rt.describe(), "scope": scope()}


def _owns(request: Request, run: dict) -> None:
    if run["metadata"].get("owner") != principal(request) and not rt.is_admin(request):
        raise HTTPException(404, "unknown run")


def _run(request: Request, run_id: str) -> dict:
    try:
        run = RUNNER.get_run(run_id)
    except KeyError:
        raise HTTPException(404, "unknown run") from None
    _owns(request, run)
    return run


@app.post("/api/runs")
def create_run(request: Request, body: dict[str, Any]):
    return _start(request, body)


@app.post("/api/runs/{run_id}/rerun")
def rerun(request: Request, run_id: str):
    """A new run with the same template, cases and executor: unchanged steps are reused (skip-done)."""
    m = _run(request, run_id)["metadata"]
    if not m.get("template"):
        raise HTTPException(409, "only template runs can be started again from here")
    wf = RUNNER.workflow(run_id)
    return _start(request, {"template": m["template"], "cases": m["cases"], "id_column": m["id_column"],
                            "executor": m["executor"], "max_parallel": wf.max_parallel,
                            "executor_options": next((s.executor_options for s in wf.steps if s.executor_options), {})})


def _start(request: Request, body: dict[str, Any]):
    """Start a run. Public: ``{"template": id, "cases": [{...}, ...], "executor": "local" | "slurm",
    "max_parallel": 2}`` (``cases`` omitted: the template's example). Self-hosted also: ``{"workflow": {...}}``."""
    executor = body.get("executor") or "local"
    if executor not in EXECUTORS:
        raise HTTPException(422, f"unknown executor {executor!r}")
    if not SELF_HOSTED and executor not in ("local", "slurm"):
        raise HTTPException(403, f"the {executor} executor runs on self-hosted runners only")
    max_parallel = max(1, min(int(body.get("max_parallel") or 2), 4 if not SELF_HOSTED else 64))
    meta_: dict[str, Any] = {"owner": principal(request), "executor": executor}
    if body.get("workflow") is not None:
        if not SELF_HOSTED:
            raise HTTPException(403, "custom workflows run on self-hosted runners only (RUNNER_SELF_HOSTED=1); "
                                     "this server runs the built-in templates")
        try:
            wf = Workflow.from_dict(body["workflow"])
        except (WorkflowError, TypeError, KeyError) as exc:
            raise HTTPException(422, f"invalid workflow: {exc}") from None
        meta_["n_cases"] = len({s.group for s in wf.steps if s.group})
        wf.max_parallel = max_parallel
    else:
        t = TEMPLATES.get(body.get("template") or "")
        if t is None:
            raise HTTPException(422, f"template must be one of {sorted(TEMPLATES)}")
        ok, why = t.availability()
        if not ok and executor == "local":
            raise HTTPException(409, f"{t.title}: {why}. Use the Slurm executor to generate the job scripts, or a "
                                     "self-hosted runner with the solver installed.")
        import pandas as pd
        cases = pd.DataFrame(body["cases"]) if body.get("cases") else t.example_cases(6)
        if len(cases) > MAX_CASES:
            raise HTTPException(413, f"{len(cases)} cases: this server runs up to {MAX_CASES} per run")
        idc = body.get("id_column") or id_column_of(cases) or "case_id"
        if idc not in cases.columns:
            cases.insert(0, idc, [f"c{i + 1:02d}" for i in range(len(cases))])
        try:
            wf = t.workflow(cases, id_column=idc, max_parallel=max_parallel, executor=executor,
                            executor_options=_executor_options(executor, body.get("executor_options")),
                            name=body.get("name") or t.id)
        except (ValueError, WorkflowError) as exc:
            raise HTTPException(422, str(exc)) from None
        meta_.update(template=t.id, output_step=t.output_step, id_column=idc, n_cases=len(cases),
                     cases=to_jsonable(cases.to_dict("records")))
    rt.meter.check(principal(request), "jobs_run", 1)
    for unit in ("compute_seconds", "storage_mb"):
        rt.meter.check(principal(request), unit, 0)
    run_id = RUNNER.create(wf, metadata=meta_)
    job = rt.submit(request, "run", {"run_id": run_id}, units={"jobs_run": 1})
    RUNNER.set_metadata(run_id, {**meta_, "job_id": job["id"]})
    return {"run_id": run_id, "job_id": job["id"], "steps": len(wf.steps)}


@app.post("/api/runs/{run_id}/resume")
def resume_run(request: Request, run_id: str):
    run = _run(request, run_id)
    job_id = run["metadata"].get("job_id")
    if job_id:
        try:
            if rt.jobs.get(job_id)["status"] in ("queued", "running"):
                raise HTTPException(409, "this run is still executing")
        except KeyError:
            pass
    job = rt.submit(request, "run", {"run_id": run_id, "resume": True}, units={"jobs_run": 1})
    RUNNER.set_metadata(run_id, {**run["metadata"], "job_id": job["id"]})
    return {"run_id": run_id, "job_id": job["id"]}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(request: Request, run_id: str):
    run = _run(request, run_id)
    job_id = run["metadata"].get("job_id")
    if not job_id:
        raise HTTPException(409, "this run has no job")
    return rt.jobs.cancel(job_id)


@app.get("/api/runs")
def list_runs(request: Request, limit: int = 30):
    out = []
    for r in RUNNER.runs(limit=500):
        full = RUNNER.get_run(r["id"])
        if full["metadata"].get("owner") != principal(request) and not rt.is_admin(request):
            continue
        out.append({**r, "counts": full["counts"], "template": full["metadata"].get("template"),
                    "n_cases": full["metadata"].get("n_cases"), "executor": full["metadata"].get("executor")})
        if len(out) >= limit:
            break
    return out


@app.get("/api/runs/{run_id}")
def get_run(request: Request, run_id: str):
    run = _run(request, run_id)
    meta_ = {k: v for k, v in run["metadata"].items() if k != "cases"}
    job = None
    if meta_.get("job_id"):
        try:
            job = rt.jobs.get(meta_["job_id"])
        except KeyError:
            job = None
    steps = RUNNER.steps(run_id)
    kinds = {st["id"]: st["kind"] for st in run["workflow"]["steps"]}
    for s in steps:
        s["kind"] = kinds.get(s["step_id"])
        s["viewable"] = s["kind"] in VIEW_KINDS and s["status"] in ("done", "skipped") \
            and not (s["outputs"] or {}).get("dry_run")
    return {"id": run_id, "name": run["name"], "status": run["status"], "created": run["created"],
            "started": run["started"], "finished": run["finished"], "counts": run["counts"], "metadata": meta_,
            "job": job and {k: job[k] for k in ("id", "status", "progress", "message", "error", "attempts")},
            "steps": [{**{k: v for k, v in s.items() if k not in ("hash", "reused_from")},
                       "hash": (s["hash"] or "")[:12]} for s in steps],
            "edges": [[d, st["id"]] for st in run["workflow"]["steps"] for d in st["needs"]],
            "compute_seconds": RUNNER.compute_seconds(run_id),
            "storage_mb": RUNNER.storage_bytes(run_id) / 2**20}


@app.get("/api/runs/{run_id}/steps/{step_id}/log")
def step_log(request: Request, run_id: str, step_id: str):
    _run(request, run_id)
    return Response(RUNNER.log(run_id, step_id), media_type="text/plain")


def _step_view(request: Request, run_id: str, step_id: str) -> tuple[dict, Any]:
    """The 3-D view of a finished CalculiX or OpenFOAM step (cached until the step runs again)."""
    run = _run(request, run_id)
    st = next((s for s in RUNNER.steps(run_id) if s["step_id"] == step_id), None)
    kind = next((x["kind"] for x in run["workflow"]["steps"] if x["id"] == step_id), None)
    if st is None:
        raise HTTPException(404, "unknown step")
    if kind not in VIEW_KINDS or st["status"] not in ("done", "skipped") or (st["outputs"] or {}).get("dry_run"):
        raise HTTPException(409, "only finished CalculiX and OpenFOAM steps (not dry runs) have a result to view")
    key = hashlib.sha256(f"{run_id}/{step_id}/{st['hash']}/{st['finished']}/{VIEW_MAX_FACES}".encode()).hexdigest()[:24]
    d = RUNNER.root / "views"
    js, glb = d / f"{key}.json", d / f"{key}.glb"
    if not (js.is_file() and glb.is_file()):
        try:
            fv, info = solver_result_view(RUNNER.step_workdir(run_id, step_id), max_faces=VIEW_MAX_FACES)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        d.mkdir(parents=True, exist_ok=True)
        glb.write_bytes(fv.glb())
        js.write_text(json.dumps(to_jsonable({"view": fv.meta, "info": info})))
    return json.loads(js.read_text()), glb


@app.get("/api/runs/{run_id}/steps/{step_id}/view3d")
def step_view(request: Request, run_id: str, step_id: str):
    """The result of a CalculiX or OpenFOAM step in 3-D, as a check before its numbers are used: the mesh, its
    fields, the deformed shape (CalculiX) and the boundary-condition and load regions the case defines. The glTF is at
    ``view3d.glb``; meshes above RUNNER_VIEW_MAX_FACES triangles are decimated."""
    out, _ = _step_view(request, run_id, step_id)
    return {**out, "glb": f"/api/runs/{run_id}/steps/{step_id}/view3d.glb"}


@app.get("/api/runs/{run_id}/steps/{step_id}/view3d.glb")
def step_view_glb(request: Request, run_id: str, step_id: str):
    _, glb = _step_view(request, run_id, step_id)
    return FileResponse(glb, media_type="model/gltf-binary")


@app.get("/api/runs/{run_id}/steps/{step_id}/files/{name}")
def step_file(request: Request, run_id: str, step_id: str, name: str):
    _run(request, run_id)
    base = RUNNER.step_dir(run_id, step_id).resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=p.name)


@app.get("/api/runs/{run_id}/files")
def run_files(request: Request, run_id: str):
    _run(request, run_id)
    base = RUNNER.root / "runs" / run_id
    return [{"step": p.parent.name, "name": p.name, "size": p.stat().st_size}
            for p in sorted(base.glob("*/*")) if p.is_file()]


@app.get("/api/runs/{run_id}/workflow")
def run_workflow(request: Request, run_id: str):
    return _run(request, run_id)["workflow"]


def _dataset(request: Request, run_id: str, fmt: str):
    run = _run(request, run_id)
    m = run["metadata"]
    if not m.get("template") or not m.get("cases"):
        raise HTTPException(409, "datasets are assembled for template runs (cases + output step)")
    if fmt not in DATASET_FORMATS:
        raise HTTPException(422, f"format is one of {sorted(DATASET_FORMATS)}")
    import pandas as pd
    t = TEMPLATES[m["template"]]
    out = RUNNER.root / "runs" / run_id / "_dataset"
    out.mkdir(parents=True, exist_ok=True)
    spec, path = dataset_from_run(RUNNER, run_id, pd.DataFrame(m["cases"]), output_step=m["output_step"],
                                  path=out / f"{run_id}.{fmt}", id_column=m.get("id_column"), skeleton=t.space,
                                  name=f"{t.id}-{run_id}")
    spec.columns = [c.model_copy(update={"unit": t.outputs.get(c.name, c.unit)}) for c in spec.columns]
    return spec, path


@app.get("/api/runs/{run_id}/dataset")
def dataset(request: Request, run_id: str, format: str = "csv"):
    spec, path = _dataset(request, run_id, format)
    return FileResponse(path, media_type=DATASET_FORMATS[format], filename=path.name)


@app.get("/api/runs/{run_id}/dataset-spec")
def dataset_spec(request: Request, run_id: str, format: str = "parquet"):
    spec, path = _dataset(request, run_id, format)
    df, _ = read_cases(path)
    return JSONResponse({"spec": spec.to_dict(), "preview": to_jsonable(df.head(50).to_dict("records"))})


@app.post("/api/cases/parse")
async def parse_cases(request: Request, file: UploadFile, template: str = ""):
    """Read a case table (CSV / Parquet / HDF5 / JSON, e.g. a Campaign Planner export with its .contract.json
    sidecar contents embedded) and check it against a template."""
    rec = rt.uploads.save(file.filename or "cases.csv", file.file, owner=principal(request))
    try:
        df, contract = read_cases(rt.uploads.path(rec["id"]))
    except Exception as exc:  # noqa: BLE001 - any reader error is the file's problem
        raise HTTPException(422, f"cannot read {rec['name']}: {exc}") from None
    finally:
        rt.uploads.delete(rec["id"])
    problems = TEMPLATES[template].check_cases(df) if template in TEMPLATES else []
    return {"name": rec["name"], "rows": len(df), "columns": [str(c) for c in df.columns],
            "id_column": id_column_of(df, contract), "contract": contract.kind if contract else None,
            "cases": to_jsonable(df.head(MAX_CASES).to_dict("records")), "problems": problems}


@app.get("/api/templates/{template}/example.csv")
def example_csv(template: str):
    if template not in TEMPLATES:
        raise HTTPException(404, "unknown template")
    buf = io.StringIO()
    TEMPLATES[template].example_cases(6).to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{template}_cases.csv"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
