"""Engineering GUI Agent: web API + single-page UI. Give it a task (a sentence or a structured workflow); it runs the
work in FreeCAD, CfdOF, OpenFOAM and ParaView inside this container and hands back the files plus the evidence of
every step: screenshots, logs, commands, produced files with SHA-256, checks and the acceptance criteria.

The work is done by ``pinneapple_tools.automation``; runs execute in the shared job queue of ``apps/_shared``
(``install_runtime``: job queue, API keys, usage metering). This module only connects them over HTTP.

Run:  uvicorn guiagent.api:app --port 8115   (from apps/guiagent)

Environment (all optional; the runtime's own variables are listed in apps/_shared/README.md):
  GUIAGENT_USER / GUIAGENT_PASSWORD   HTTP Basic login on everything except /health
  GUIAGENT_MODE                       public (default: only the committed workflows) or selfhosted (also uploads)
  GUIAGENT_MAX_JOBS                   runs at the same time (default 1); further runs wait in the queue
  GUIAGENT_KEEP_RUNS                  runs kept on disk, oldest deleted first (default 40)
  GUIAGENT_FREE_RUNS / _COMPUTE_SECONDS / _STORAGE_MB   free use per caller and month (20 / 1800 / 500)
  GUIAGENT_OLLAMA_URL / GUIAGENT_OLLAMA_MODEL   optional local LLM for reading sentences (rule-based without it)
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import sys
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_security.archives import ArchiveLimits, UnsafeArchive, safe_extract_zip
from pinneapple_tools.automation import (
    Runner,
    RunRecord,
    Toolchain,
    Workflow,
    WorkflowError,
    catalog,
    interpret,
    known_actions,
    plan,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.1.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
MODE = (os.environ.get("GUIAGENT_MODE") or "public").lower()
KEEP_RUNS = int(os.environ.get("GUIAGENT_KEEP_RUNS") or 40)
RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

app = FastAPI(title="Engineering GUI Agent", version=VERSION,
              description="Run a task in engineering software (FreeCAD, CfdOF, OpenFOAM, ParaView) and get the files "
                          "plus the evidence of every step. POST /api/runs queues a run of a committed workflow; it "
                          "stops before expensive steps until POST /api/runs/{id}/approve. GET /api/runs/{id} follows "
                          "it; /report, /study_result and /bundle deliver the evidence.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="GUIAGENT")
rt = install_runtime(app, prefix="GUIAGENT", name="guiagent", max_jobs=1,
                     units={"runs": 20, "compute_seconds": 1800, "storage_mb": 500})
RUNS = rt.root / "runs"
UPLOADED = rt.root / "workflows"
_toolchain = Toolchain.discover()


def _llm():
    model = os.environ.get("GUIAGENT_OLLAMA_MODEL")
    if not model:
        return None
    from pinneapple_data.formfill.llm import OllamaClient
    return OllamaClient(os.environ.get("GUIAGENT_OLLAMA_URL") or "http://ollama:11434", model, timeout=120)


def _catalog() -> dict[str, Workflow]:
    cat = catalog()
    if MODE == "selfhosted" and UPLOADED.is_dir():
        cat.update(catalog(UPLOADED))
    return cat


def _workflow(wid: str) -> Workflow:
    cat = _catalog()
    if wid not in cat:
        raise HTTPException(404, f"No workflow {wid!r}. Available: {', '.join(sorted(cat))}.")
    return cat[wid]


def _meta(folder: Path) -> dict:
    p = folder / "owner.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _active(meta: dict) -> bool:
    try:
        return rt.jobs.get(meta["job_id"])["status"] in ("queued", "running")
    except (KeyError, TypeError):
        return False


def _run_dir(request: Request, run_id: str) -> Path:
    """The run's folder if it exists and the caller owns it (or is admin); 404 otherwise."""
    folder = RUNS / run_id
    if not RUN_ID.match(run_id) or not (folder / "run.json").is_file():
        raise HTTPException(404, "No such run.")
    if _meta(folder).get("owner") != principal(request) and not rt.is_admin(request):
        raise HTTPException(404, "No such run.")
    return folder


def _prune() -> None:
    runs = sorted((p for p in RUNS.iterdir() if p.is_dir() and RUN_ID.match(p.name)), key=lambda p: p.name)
    for p in runs[:-KEEP_RUNS] if len(runs) > KEEP_RUNS else []:
        if not _active(_meta(p)):
            shutil.rmtree(p, ignore_errors=True)


def agent_job(params: dict, ctx) -> dict:
    """Queue job: run (or resume) one automation run, report progress, meter compute time and storage."""
    folder = RUNS / params["run_id"]
    rec = RunRecord.load(folder)
    wf = _workflow(rec.workflow["id"])
    ids = [s.id for s in wf.steps]
    t0 = time.time()

    def on_event(e: dict) -> None:
        if e["event"] == "step_started":
            ctx.progress(ids.index(e["step"]) / len(ids), f"{e['step']}")      # also stops here on cancel
        elif e["event"] in ("progress", "confirmation_needed", "step_blocked", "step_finished"):
            ctx.log(f"{e.get('step', '')}: {e.get('message') or e.get('state') or e.get('error') or e['event']}")

    rec = Runner(wf, folder, toolchain=_toolchain, on_event=on_event).run(
        approve=params.get("approve") or [], approver=params.get("approver") or "")
    seconds = sum(s.duration_s or 0.0 for s in rec.steps if s.started and s.started >= t0)
    mb = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 2**20
    owner = _meta(folder).get("owner", "")
    rt.meter.record(owner, "compute_seconds", seconds, ref=rec.run_id)
    rt.meter.record(owner, "storage_mb", mb, ref=rec.run_id)
    ctx.log(f"run {rec.run_id}: {rec.state}, {seconds:.1f} s of compute, {mb:.1f} MB")
    return {"run_id": rec.run_id, "state": rec.state, "message": rec.message, "compute_seconds": seconds,
            "storage_mb": mb}


rt.jobs.register("agent", agent_job)


def _submit(request: Request, folder: Path, params: dict, units: dict) -> dict:
    who = principal(request)
    for unit in ("compute_seconds", "storage_mb"):
        rt.meter.check(who, unit, 0)                     # already past the month's compute or storage: 429
    job = rt.submit(request, "agent", params, units=units)
    (folder / "owner.json").write_text(json.dumps({**_meta(folder), "owner": who, "job_id": job["id"]}))
    return job


# ----------------------------------------------------------------------------------------------------- schemas
class InterpretIn(BaseModel):
    task: str = Field(..., max_length=4000, description="A sentence, or a structured request as JSON/YAML text")


class RunIn(BaseModel):
    workflow: str = Field("channel_obstacle", description="Id of a committed workflow (GET /api/workflows)")
    inputs: dict = Field(default_factory=dict, description="Input values; the workflow's bounds apply")
    task: str | None = Field(None, max_length=4000, description="Optional sentence, read instead of workflow+inputs")


class Decision(BaseModel):
    step: str
    by: str = Field("", max_length=80, description="Who decides (kept in the evidence)")
    reason: str = Field("", max_length=400)


# ------------------------------------------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    missing = _toolchain.missing(["freecad", "cfdof", "openfoam", "gmsh", "paraview", "xvfb", "xdotool"])
    stats = rt.jobs.stats()
    return {"version": VERSION, "mode": MODE, "max_jobs": rt.jobs.max_concurrent, "queue": stats,
            "llm": bool(os.environ.get("GUIAGENT_OLLAMA_MODEL")), "tools": _toolchain.versions(),
            "missing_tools": missing, "uploads": MODE == "selfhosted"}


@app.get("/api/workflows")
def workflows():
    """Every workflow this server runs, with its inputs, steps (and whether their application is installed),
    acceptance criteria and limits."""
    out = []
    for wf in _catalog().values():
        d = wf.to_dict()
        d["plan"] = plan(wf, _toolchain)
        out.append(d)
    return out


@app.post("/api/interpret")
def api_interpret(body: InterpretIn):
    """Read a task: which committed workflow it asks for and with which inputs. Nothing is run."""
    try:
        it = interpret(body.task, _catalog(), llm=_llm())
    except (WorkflowError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    if it.workflow.id not in _catalog():
        raise HTTPException(403, "Only committed workflows run on this server; a structured workflow is read but not "
                                 "run here (self-hosted mode can add workflows).")
    return {**it.to_dict(), "resolved_inputs": it.workflow.resolve_inputs(it.inputs)}


@app.post("/api/runs")
def start_run(request: Request, body: RunIn):
    """Queue a run. It goes on in the background until it finishes, fails, or reaches a step that needs a person's
    approval (the solver step of the example); follow it with GET /api/runs/{id}. Counts one run."""
    try:
        if body.task:
            it = interpret(body.task, _catalog(), llm=_llm())
            wf, inputs = it.workflow, it.inputs
            if wf.id not in _catalog():
                raise HTTPException(403, "Only committed workflows run on this server.")
        else:
            wf, inputs = _workflow(body.workflow), body.inputs
        runner_inputs = wf.resolve_inputs(inputs)
    except WorkflowError as e:
        raise HTTPException(422, str(e)) from e
    rt.meter.check(principal(request), "runs", 1)
    RUNS.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
    folder = RUNS / run_id
    Runner(wf, folder, inputs=runner_inputs, toolchain=_toolchain)        # writes run.json (pending)
    job = _submit(request, folder, {"run_id": run_id}, units={"runs": 1})
    _prune()
    return {"run_id": run_id, "job_id": job["id"], "workflow": wf.id, "inputs": runner_inputs}


@app.get("/api/runs")
def list_runs(request: Request, limit: int = 20):
    """The caller's runs (the admin sees every run), newest first."""
    if not RUNS.is_dir():
        return []
    admin, who, out = rt.is_admin(request), principal(request), []
    for p in sorted((p for p in RUNS.iterdir() if RUN_ID.match(p.name)), reverse=True):
        m = _meta(p)
        if not admin and m.get("owner") != who:
            continue
        try:
            r = RunRecord.load(p)
        except (OSError, ValueError):
            continue
        out.append({"run_id": r.run_id, "workflow": r.workflow.get("id"), "title": r.workflow.get("title"),
                    "inputs": r.inputs, "state": r.state, "message": r.message, "created": r.created,
                    "finished": r.finished, "active": _active(m), "job_id": m.get("job_id")})
        if len(out) >= max(1, min(limit, 100)):
            break
    return out


@app.get("/api/runs/{run_id}")
def get_run(request: Request, run_id: str):
    """The live record: state, every step (state, duration, commands, logs, artifacts, screenshots, checks,
    outputs, confirmation), the acceptance criteria and the tools used; plus the queue job."""
    folder = _run_dir(request, run_id)
    d = RunRecord.load(folder).to_dict()
    m = _meta(folder)
    job = None
    if m.get("job_id"):
        try:
            job = rt.jobs.get(m["job_id"])
        except KeyError:
            job = None
    d["active"] = bool(job and job["status"] in ("queued", "running"))
    d["job"] = {k: job.get(k) for k in ("id", "status", "progress", "message", "error")} if job else None
    d["current_step"] = next((s["id"] for s in d["steps"] if s["state"] == "running"), None)
    d.pop("folder", None)
    return d


@app.post("/api/runs/{run_id}/approve")
def approve(request: Request, run_id: str, body: Decision):
    """Approve the step the run is waiting on; the run continues in the queue."""
    folder = _run_dir(request, run_id)
    if _active(_meta(folder)):
        raise HTTPException(409, "This run is already queued or running.")
    step = next((s for s in RunRecord.load(folder).steps if s.id == body.step), None)
    if step is None or step.state != "awaiting_confirmation":
        raise HTTPException(409, f"Step {body.step!r} is not waiting for a confirmation.")
    job = _submit(request, folder, {"run_id": run_id, "approve": [body.step],
                                    "approver": body.by.strip() or "web user"}, units={})
    return {"run_id": run_id, "approved": body.step, "job_id": job["id"]}


@app.post("/api/runs/{run_id}/decline")
def decline(request: Request, run_id: str, body: Decision):
    """Refuse the confirmation: the run ends there, and the evidence says who declined and why."""
    folder = _run_dir(request, run_id)
    if _active(_meta(folder)):
        raise HTTPException(409, "This run is already queued or running.")
    rec = RunRecord.load(folder)
    try:
        rec = Runner(_workflow(rec.workflow["id"]), folder, toolchain=_toolchain).decline(
            body.step, by=body.by.strip() or "web user", reason=body.reason)
    except WorkflowError as e:
        raise HTTPException(409, str(e)) from e
    return {"run_id": run_id, "state": rec.state}


_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json",
          ".csv": "text/csv", ".html": "text/html", ".txt": "text/plain", ".log": "text/plain"}


@app.get("/api/runs/{run_id}/files/{path:path}")
def run_file(request: Request, run_id: str, path: str):
    """A file of the run (screenshots, logs, exported fields, the FreeCAD documents)."""
    root = _run_dir(request, run_id).resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root) or not p.is_file() or p.name == "owner.json":
        raise HTTPException(404, "No such file in this run.")
    media = _MEDIA.get(p.suffix.lower()) or ("text/plain" if p.name.startswith("log.") else "application/octet-stream")
    return FileResponse(p, media_type=media)


@app.get("/api/runs/{run_id}/report")
def report(request: Request, run_id: str):
    """The evidence report: one self-contained HTML file (print it to PDF from the browser)."""
    p = _run_dir(request, run_id) / "report.html"
    if not p.is_file():
        raise HTTPException(404, "The report is written when the run stops.")
    return FileResponse(p, media_type="text/html")


@app.get("/api/runs/{run_id}/study_result")
def study_result(request: Request, run_id: str):
    """The run as a pinneapple_data.contracts.StudyResult, for the Evidence Report app."""
    p = _run_dir(request, run_id) / "study_result.json"
    if not p.is_file():
        raise HTTPException(404, "The StudyResult is written when the run stops.")
    return FileResponse(p, media_type="application/json")


@app.get("/api/runs/{run_id}/bundle")
def bundle(request: Request, run_id: str):
    """Everything the run produced as a zip: record, report, StudyResult, FreeCAD documents, the OpenFOAM case at
    its last time, logs, screenshots, exported fields."""
    root = _run_dir(request, run_id)
    if _active(_meta(root)):
        raise HTTPException(409, "The run is still going; download it when it stops.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.name != "owner.json":
                z.write(p, f"{run_id}/{p.relative_to(root).as_posix()}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="guiagent-{run_id}.zip"'})


@app.post("/api/workflows")
async def upload_workflow(file: UploadFile = File(...)):  # noqa: B008  (FastAPI idiom)
    """Self-hosted mode only: add a workflow (a zip with workflow.yaml and the files it uses, e.g. your FreeCAD
    project and macro). A public server refuses: macros are code."""
    if MODE != "selfhosted":
        raise HTTPException(403, "This is the public demo: only the committed workflows run here. Uploading "
                                 "workflows and macros is available in self-hosted mode (GUIAGENT_MODE=selfhosted).")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Workflow archives are limited to 50 MB.")
    UPLOADED.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADED / (".upload-" + secrets.token_hex(4))
    try:
        try:
            safe_extract_zip(data, tmp, ArchiveLimits(max_members=2000, max_total_bytes=500 * 2**20))
        except UnsafeArchive as e:
            raise HTTPException(422, f"Unsafe archive: {e}") from None
        found = sorted(tmp.rglob("workflow.y*ml")) + sorted(tmp.rglob("workflow.json"))
        if not found:
            raise HTTPException(422, "The archive has no workflow.yaml.")
        wf = Workflow.load(found[0])
        errors = wf.validate(known_actions())
        if errors:
            raise HTTPException(422, "; ".join(errors))
        if wf.id in catalog():
            raise HTTPException(409, f"{wf.id!r} is a committed workflow; choose another id.")
        dest = UPLOADED / wf.id
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(found[0].parent), dest)
    except zipfile.BadZipFile as e:
        raise HTTPException(422, "Upload a zip archive.") from e
    except WorkflowError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"workflow": wf.id, "title": wf.title}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
