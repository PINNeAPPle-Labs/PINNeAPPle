"""PINNeAPPle Hub: the ready-made tools, trained models, geometries, datasets, flows and examples of the library in
one searchable catalogue -- see each one, copy its code, download its files, open it in Flow Studio, run a zoo
model, and compose your own tool from existing blocks.

Run:  uvicorn hub.api:app --port 8116     (from apps/hub)

Everything shown comes from the generated index of :mod:`pinneapple_hub.library` (``data/web/hub/index.json``);
composing, validating and saving a tool are :mod:`pinneapple_hub.library.compose`, ``.sandbox`` and ``.myhub``.
This file is only routes.

Environment (all optional; runtime and security variables in apps/_shared/README.md):
  HUB_USER / HUB_PASSWORD_HASH  HTTP Basic login on everything except /health
  HUB_FLOW_STUDIO_URL           where "Open in Flow Studio" goes (default http://127.0.0.1:8765)
  HUB_SANDBOX_S                 time limit of a validation run of a composed tool (default 60 s)
  HUB_SANDBOX_MEMORY_MB         address-space limit of that run (default 6144; PyTorch reserves much of it)
  HUB_MAX_JOBS                  validation runs at once (default 2)
  HUB_FREE_VALIDATIONS          free validation runs per caller and month (default 200)
  HUB_FREE_INFERENCES           free zoo inferences per caller and month (default 5000)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_hub import library as hub
from pinneapple_hub.library.compose import ComposeError, blocks, compose
from pinneapple_hub.library.myhub import MyHub, MyHubError
from pinneapple_hub.library.sandbox import run_snippet

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
NAME = "hub"
STATIC = os.path.join(os.path.dirname(__file__), "static")
ROOT = hub.ROOT
#: repository folders the app serves files from, and the file types it serves from them
SERVED = ("data/web/", "data/geometries/", "docs/org/reference/figures/", "docs/assets/", "lab/runs/", "examples/",
          "pinneapple_hub/zoo/assets/", "pinneapple_flow/flows/")
SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".webm", ".glb", ".stl", ".obj", ".usda",
            ".json", ".csv", ".npz", ".onnx", ".pt", ".md"}
TYPES = {".glb": "model/gltf-binary", ".stl": "model/stl", ".onnx": "application/octet-stream",
         ".npz": "application/octet-stream", ".pt": "application/octet-stream", ".usda": "text/plain"}

app = FastAPI(title="PINNeAPPle Hub", version=VERSION,
              description="Ready-made tools, trained models, geometries, datasets, flows and examples of PINNeAPPle: "
                          "search, copy the code, download, open in Flow Studio, run a zoo model, compose your own.")
install(app, prefix="HUB")
rt = install_runtime(app, prefix="HUB", name=NAME, units={"validations": 200, "inferences": 5000})

_INDEX: dict[str, Any] = {}


def index() -> dict[str, Any]:
    if not _INDEX:
        _INDEX.update(hub.load_index(os.environ.get("HUB_INDEX") or hub.INDEX))
    return _INDEX


def card(it: dict[str, Any]) -> dict[str, Any]:
    """What a result card shows."""
    q = it.get("quality") or {}
    return {"id": it["id"], "kind": it["kind"], "subkind": it.get("subkind", ""), "title": it["title"],
            "summary": it.get("summary", ""), "domain": it.get("domain", ""), "tags": it.get("tags", [])[:5],
            "quality": hub.quality_key(it), "quality_label": q.get("label", ""),
            "thumbnail": it.get("thumbnail") or next(iter((it.get("results") or {}).get("figures") or []), "")
            or next(iter((it.get("run") or {}).get("figures") or []), ""),
            "run_ok": bool((it.get("run") or {}).get("ok"))}


def mine(request: Request) -> MyHub:
    who = hashlib.sha256(principal(request).encode()).hexdigest()[:24]
    return MyHub(rt.root / "myhub" / who)


# ----------------------------------------------------------------------------------------------------- catalogue
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "items": len(index()["items"])}


@app.get("/api/config")
def config():
    from pinneapple_lab.seal import SEALABLE, TIER_COLOURS, TIER_INK, TIER_TEXT
    idx = index()
    return {"tiers": {t: {"colour": TIER_COLOURS[t], "ink": TIER_INK[t], "en": TIER_TEXT["en"][t]} for t in SEALABLE},
            "version": VERSION, "flow_studio": os.environ.get("HUB_FLOW_STUDIO_URL", "http://127.0.0.1:8765"),
            "counts": idx["counts"], "kinds": idx["kinds"], "facets": idx["facets"],
            "sandbox_s": float(os.environ.get("HUB_SANDBOX_S") or 60),
            "github": "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main"}


SUBKIND_RANK = {"function": 0, "recipe": 1, "flow node": 2, "agent tool (MCP)": 3}


def sample_rank(it: dict[str, Any]):
    """Category samples: passing runs first, graded before not graded, with a picture, the main APIs first."""
    c = card(it)
    return (not c["run_ok"], c["quality"] == "not graded", not c["thumbnail"], SUBKIND_RANK.get(c["subkind"], 0),
            c["quality"])


@app.get("/api/home")
def home():
    idx = index()
    feat = [card(hub.get(idx, i)) for i in idx["featured"]]
    cats = []
    for k, title in idx["kinds"].items():
        rows = [i for i in idx["items"] if i["kind"] == k]
        cats.append({"kind": k, "title": title, "count": len(rows),
                     "sample": [card(i) for i in sorted(rows, key=sample_rank)[:4]]})
    return {"featured": feat, "categories": cats, "domains": idx["facets"]["domain"]}


@app.get("/api/search")
def search(q: str = "", kind: str | None = None, domain: str | None = None, quality: str | None = None,
           subkind: str | None = None, limit: int = Query(60, ge=1, le=500), offset: int = Query(0, ge=0)):
    rows = hub.search(index(), q, kind=kind or None, domain=domain or None, quality=quality or None,
                      subkind=subkind or None)
    return {"total": len(rows), "items": [card(i) for i in rows[offset:offset + limit]]}


@app.get("/api/items/{item_id:path}")
def item(item_id: str, request: Request):
    if item_id.startswith("mine:"):
        try:
            return mine(request).get(item_id)
        except KeyError:
            raise HTTPException(404, f"unknown item {item_id}") from None
    try:
        it = hub.get(index(), item_id)
    except KeyError:
        raise HTTPException(404, f"unknown item {item_id}") from None
    related = []
    for rid in it.get("examples", []):
        try:
            related.append(card(hub.get(index(), rid)))
        except KeyError:
            continue
    return {**it, "related": related}


@app.get("/files/{path:path}")
def files(path: str):
    """A repository file the index points to (only under :data:`SERVED`, only :data:`SUFFIXES`)."""
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT) or not any(p.relative_to(ROOT).as_posix().startswith(s) for s in SERVED) \
            or p.suffix.lower() not in SUFFIXES or not p.is_file():
        raise HTTPException(404, "not found")
    download = p.suffix.lower() in {".stl", ".obj", ".npz", ".onnx", ".pt", ".usda"}
    return FileResponse(p, media_type=TYPES.get(p.suffix.lower()),
                        filename=p.name if download else None)


# ---------------------------------------------------------------------------------------------------- inference
class PredictBody(BaseModel):
    inputs: dict[str, float] = Field(..., description="one value per model input")


@app.post("/api/models/{name}/predict")
def predict(name: str, body: PredictBody, request: Request):
    """Server-side zoo inference with the per-query trust checks (the same as pp.zoo.load(name).predict)."""
    import warnings

    import numpy as np

    from pinneapple_hub import zoo
    if name not in zoo.list():
        raise HTTPException(404, f"unknown model {name}")
    rt.consume(request, "inferences", 1, ref=name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = zoo.load(name, offline=True)
        missing = [k for k in m.input_names if k not in body.inputs]
        if missing:
            raise HTTPException(422, f"missing inputs: {', '.join(missing)}")
        row = {k: [float(body.inputs[k])] for k in m.input_names}
        out = m.predict(row, warn=False)
        checks = zoo.query_checks(m, row)[0]
        in_range = bool(np.asarray(m.in_range(m.as_array(row))).all())
    return {"model": name, "outputs": {k: float(np.asarray(v, float).ravel()[0]) for k, v in out.items()},
            "in_range": in_range, "verdict": checks["verdict"],
            "checks": [{k: c.get(k) for k in ("name", "status", "value", "limit", "detail_text")}
                       for c in checks["checks"]]}


# ------------------------------------------------------------------------------------------------- create your own
class ComposeBody(BaseModel):
    name: str = Field("My tool", max_length=80)
    description: str = Field("", max_length=500)
    blocks: list[dict[str, Any]] = Field(default_factory=list, max_length=24)
    start: str | None = None
    geometry: str | None = None


def _compose(body: ComposeBody) -> dict[str, Any]:
    try:
        return compose(body.model_dump())
    except ComposeError as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/compose/blocks")
def compose_blocks():
    return {"blocks": blocks(), "flows": [{"name": i["name"], "title": i["title"]} for i in index()["items"]
                                          if i["kind"] == "flow"],
            "geometries": [{"name": i["name"], "title": i["title"]} for i in index()["items"]
                           if i["kind"] == "geometry"]}


@app.post("/api/compose")
def compose_route(body: ComposeBody):
    return _compose(body)


def validate_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Run a composed tool's code (regenerated here from the blocks, never visitor code) in the sandbox."""
    ctx.progress(0.05, "composing")
    out = compose(params)
    ctx.progress(0.15, "running the code in a sandbox")
    r = run_snippet(out["code"], timeout=float(os.environ.get("HUB_SANDBOX_S") or 60),
                    memory_mb=int(os.environ.get("HUB_SANDBOX_MEMORY_MB") or 6144), max_figures=2)
    figs = [base64.b64encode(f).decode() for f in r.pop("figures")]
    ctx.progress(1.0, "done")
    return {**r, "figures_png_b64": figs, "issues": out["issues"]}


rt.jobs.register("validate", validate_job)


@app.post("/api/compose/validate")
def compose_validate(body: ComposeBody, request: Request):
    out = _compose(body)
    if not out["ok"]:
        raise HTTPException(422, {"message": "fix the flow first", "issues": out["issues"]})
    return rt.submit(request, "validate", body.model_dump(), units={"validations": 1})


@app.get("/api/compose/download/{what}")
def compose_download(what: str, spec: str):
    """The composed tool as a file: ``what`` = py or json; ``spec`` = the ComposeBody as base64url JSON."""
    try:
        body = ComposeBody(**json.loads(base64.urlsafe_b64decode(spec + "=" * (-len(spec) % 4))))
    except Exception:  # noqa: BLE001 - any malformed spec is the caller's error
        raise HTTPException(422, "spec: base64url JSON of the composer body") from None
    out = _compose(body)
    if what == "py":
        return Response(out["code"], media_type="text/x-python",
                        headers={"Content-Disposition": f'attachment; filename="{out["file_stem"]}.py"'})
    if what == "json":
        return Response(json.dumps(out["flow"], indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{out["file_stem"]}.flow.json"'})
    raise HTTPException(404, "py or json")


# ------------------------------------------------------------------------------------------------------- my hub
class SaveBody(ComposeBody):
    job_id: str | None = None


@app.get("/api/mine")
def mine_list(request: Request):
    return {"items": [card(i) | {"saved": i.get("saved")} for i in mine(request).list()]}


@app.post("/api/mine")
def mine_save(body: SaveBody, request: Request):
    out = _compose(body)
    run = None
    if body.job_id:
        job = rt.job_for(request, body.job_id)
        run = job.get("result") if job.get("status") == "done" else None
    try:
        return mine(request).save(out, name=body.name, description=body.description, run=run)
    except MyHubError as exc:
        raise HTTPException(422, str(exc)) from None


@app.delete("/api/mine/{item_id}")
def mine_delete(item_id: str, request: Request):
    try:
        mine(request).delete(item_id)
    except KeyError:
        raise HTTPException(404, f"unknown item {item_id}") from None
    return {"deleted": item_id}


# ------------------------------------------------------------------------------------------------------ pages
app.mount("/static", StaticFiles(directory=STATIC), name="static")
from pinneapple_hub.zoo.web import runtime_dir  # noqa: E402

app.mount("/zoo-runtime", StaticFiles(directory=str(runtime_dir())), name="zoo-runtime")


@app.get("/")
@app.get("/item/{rest:path}")
@app.get("/create")
@app.get("/mine")
def page(rest: str = ""):
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.exception_handler(404)
async def not_found(request: Request, exc):
    if request.url.path.startswith(("/api", "/files")):
        return JSONResponse({"detail": getattr(exc, "detail", "not found")}, status_code=404)
    return FileResponse(os.path.join(STATIC, "index.html"), status_code=404)

