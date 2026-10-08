"""Engineering Model Lineage: web API + single-page UI. How was this result produced? Upload a project (files or a
zip) or a lineage JSON and get the digital thread: every artifact (CAD, mesh, simulation, dataset, model, prediction)
with file, version, software, parameters, timestamp, hash, owner and origin, the links between them, the checks
that make it trustworthy, and W3C PROV export.

Run:  uvicorn model_lineage.api:app --port 8090   (from apps/model_lineage)

Environment (all optional):
  LIN_USER / LIN_PASSWORD   HTTP Basic login on everything except /health
  LIN_MAX_MB                largest upload in MB (default 300)
  LIN_MAX_HEAVY             concurrent analyses per worker (default 2)
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.cae.upload import expand
from pinneapple_data.lineage import Lineage, detect_project

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_MB = float(os.environ.get("LIN_MAX_MB", "300"))

EXAMPLE_INFO = {
    "openfoam_pitzDaily_study.zip": ("Auto-detected: a pitzDaily study folder (3 OpenFOAM runs, a dataset exported by "
                                     "the Interoperability Hub, a Comparator report) plus the team's lineage.json"),
    "pinneapple_pinn_plate.zip": ("Declared and verified: PINNeAPPle's PINN plate example, every hash checked "
                                  "against the repository files"),
    "illustrative_digital_thread.json": ("Illustrative digital thread (fictitious IDs): CAD → mesh → OpenFOAM → "
                                         "post-processing → dataset → FNO → prediction"),
    "simple_chain.json": "The minimal input: one JSON line per stage",
}

app = FastAPI(title="Engineering Model Lineage", version=VERSION,
              description="Upload a project (files or zip) or a lineage JSON to /api/lineage and get the digital "
                          "thread: artifacts, links, checks (broken references, cycles, files changed since recorded, "
                          "stale derivations, mixed revisions), and exports (lineage JSON, W3C PROV-JSON, Markdown).")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="LIN")
_HEAVY = BusyLimiter("LIN_MAX_HEAVY")

VALIDATION = [
    "Auto-detection on real files: three OpenFOAM v1912 runs (blockMeshDict → polyMesh → simpleFoam, with solver "
    "version, run date, cell counts, turbulence model and convergence read from the files), a Parquet dataset linked to "
    "its run by the sha256 of the source files, and a Comparator report linked to the two runs it compares",
    "Content check: the fine case's blockMeshDict differs from the hash recorded in lineage.json (it was refined "
    "after recording) and is flagged",
    "PINNeAPPle's PINN plate example: seven artifacts declared with their sha256 and commit dates, all verified "
    "against the repository files",
    "Questions answered by graph traversal: upstream (how was this produced), downstream (impact of a change), sources "
    "of a kind (which simulations trained this model); exports to W3C PROV-JSON",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the lineage knows",
            "band": "Links are either declared or detected; each artifact says which.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss a link", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Auto-detection", "effect": "optimistic",
                 "detail": "Recognises OpenFOAM cases, Gmsh .geo/.msh, CalculiX decks and results, exports of the "
                           "PINNeAPPle apps (by the source hashes they carry) and model cards; other files are listed "
                           "unlinked.", "today": "Declare the missing links in a lineage.json next to the files.",
                 "planned": "Abaqus/ANSYS job files, MLflow and DVC metadata, git history."},
                {"topic": "Hashes", "effect": "conservative",
                 "detail": "A file is verified when it is uploaded with the lineage that records its hash; an "
                           "artifact without an uploaded file is trusted as declared.",
                 "today": "Upload the files to verify them.", "planned": "Remote verification against storage URLs."},
                {"topic": "Timestamps", "effect": "check",
                 "detail": "Stale-derivation checks compare declared or detected times (solver log headers, export "
                           "times); copied files keep no reliable modification time.",
                 "today": "Record the time of each artifact when it is produced.", "planned": ""},
                {"topic": "Revisions", "effect": "check",
                 "detail": "Mixed-revision checks need the 'item' field (which revisions belong to the same CAD, "
                           "mesh, dataset).", "today": "Set item for revisions of the same thing.", "planned": ""},
            ]}


def _graph_payload(g: Lineage) -> dict:
    lay = g.layers()
    checks = g.checks()
    flag = {}
    for c in checks:
        if c.get("artifact") and c["status"] in ("fail", "warn"):
            cur = flag.get(c["artifact"])
            if cur != "fail":
                flag[c["artifact"]] = c["status"]
    nodes = []
    for a in g.nodes.values():
        d = a.to_dict()
        d.update(layer=lay.get(a.id, 0), completeness=round(a.completeness(), 2), outputs=g.children(a.id),
                 flag=flag.get(a.id), verified=bool(a.hash and a.actual_hash and (a.actual_hash.startswith(a.hash.lower()[:16])
                                                                                  or a.hash.lower().startswith(a.actual_hash[:16]))))
        nodes.append(d)
    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    return {"project": g.project, "nodes": nodes, "edges": g.edges(), "checks": checks,
            "status": {"fail": "FAIL", "warn": "WARNING", "pass": "PASS"}[worst],
            "summary": {"artifacts": len(g.nodes), "links": len(g.edges()),
                        "by_kind": {k: sum(1 for a in g.nodes.values() if a.kind == k) for k in {a.kind for a in g.nodes.values()}},
                        "verified": sum(1 for n in nodes if n["verified"]),
                        "completeness": round(sum(a.completeness() for a in g.nodes.values()) / max(len(g.nodes), 1), 2)},
            "lineage": g.to_json(), "prov": g.to_prov(), "markdown": g.to_markdown(), "scope": scope()}


def _run(files: List[Tuple[str, bytes]]) -> dict:
    if not files:
        raise HTTPException(422, "No files uploaded.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload larger than {MAX_MB:g} MB.")
    try:
        with _HEAVY:
            if len(files) == 1 and files[0][0].lower().endswith(".json") and not files[0][0].lower().endswith("lineage.json") \
                    and _is_lineage_json(files[0][1]):
                g = Lineage.from_json(json.loads(files[0][1]))
                mode = "declared"
            else:
                fs = expand(files)
                g = detect_project(fs, os.path.splitext(files[0][0])[0] if len(files) == 1 else "")
                mode = "detected"
            out = _graph_payload(g)
    except HTTPException:
        raise
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from e
    out["mode"] = mode
    out["usage"] = {"artifacts": out["summary"]["artifacts"]}
    return out


def _is_lineage_json(b: bytes) -> bool:
    try:
        d = json.loads(b)
    except ValueError:
        return False
    return isinstance(d, dict) and ("artifacts" in d or any(k in d for k in ("geometry", "mesh", "simulation", "model")))


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "examples": [{"name": k, "description": v} for k, v in EXAMPLE_INFO.items()
                                             if os.path.exists(os.path.join(EXAMPLES, k))]}


@app.post("/api/lineage")
async def api_lineage(files: List[UploadFile] = File(...)):
    """A lineage JSON ({"artifacts": [...]} or the simple chain), or a project's files / zip for auto-detection (a
    lineage.json among them annotates and is verified against the files)."""
    return _run([(f.filename or f"upload_{i}", await f.read()) for i, f in enumerate(files)])


@app.get("/api/example/{name}")
def api_example(name: str):
    if name not in EXAMPLE_INFO:
        raise HTTPException(404, "No such example.")
    with open(os.path.join(EXAMPLES, name), "rb") as f:
        out = _run([(name, f.read())])
    out["example"] = {"name": name, "description": EXAMPLE_INFO[name]}
    return out


@app.get("/api/example/{name}/file")
def api_example_file(name: str):
    if name not in EXAMPLE_INFO:
        raise HTTPException(404, "No such example.")
    with open(os.path.join(EXAMPLES, name), "rb") as f:
        return Response(f.read(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
