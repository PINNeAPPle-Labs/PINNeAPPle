"""Physics AI Model Gateway: publish physical models as versioned endpoints and run them through one interface.

Models served (pinneapple_registry.ModelGateway): zoo models (published at start-up), ONNX graphs with declared inputs,
outputs and validity bounds, surrogates from the Surrogate Studio (a model.zip, or by reference to the studio's own
endpoint), and any model behind the same API on another server (a ModelRef with an http(s) URI). Every version has
an OOD policy (warn or refuse) that callers can tighten, never relax; every call is logged and metered.

Run:  uvicorn model_gateway.api:app --port 8114   (from apps/model_gateway)

Environment (all optional):
  MDL_USER / MDL_PASSWORD     HTTP Basic login on everything except /health (the login is also admin)
  MDL_ADMIN_TOKEN             X-Admin-Token for publishing, promotion and key management without a login
  MDL_REQUIRE_API_KEY         1: every /api and /v1 call needs an X-API-Key (or the login)
  MDL_FREE_INFERENCES, MDL_FREE_HOSTED_MODELS   free use per caller and month
  MDL_MAX_ROWS                rows per prediction call (default 5000)
  MDL_UPLOAD_MAX_MB, MDL_DATA_DIR   see apps/_shared/README.md
  MDL_ALLOW_PICKLE_UPLOAD     1: accept Surrogate Studio model.zip uploads (default off: loading one unpickles it,
                              which runs code; the hash in its card comes from the same zip and proves nothing)
  MDL_REMOTE_ALLOW_HOSTS      hosts a remote ModelRef may point at although they are not public (e.g. surrogate)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pinneapple_registry.gateway import KINDS, OOD_POLICIES, ModelGateway
from pinneapple_registry.gateway_api import gateway_router
from pinneapple_security.archives import ArchiveLimits, safe_extract_zip
from pinneapple_security.model_security import scan_checkpoint
from pinneapple_security.web import UnsafeURL, check_outbound_url

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).parent
STATIC = HERE / "static"
MAX_ROWS = int(os.environ.get("MDL_MAX_ROWS") or 5000)

#: zoo models published when the gateway starts (name, zoo id, OOD policy)
SEED = [("zoo-airfoil-cst-rans", "airfoil_cst_rans_mlp", "warn"),
        ("zoo-heatsink-forced", "heatsink_forced_mlp", "refuse"),
        ("zoo-heatsink-natural", "heatsink_natural_mlp", "warn")]

EXAMPLES = [
    {"model": "zoo-airfoil-cst-rans", "title": "NACA 2412 at 4° (OpenFOAM: Cl 0.6625)",
     "inputs": {"T0": 0.169591, "T1": 0.144959, "T2": 0.131433, "M0": 0.024922, "M1": 0.071088, "M2": 0.058960,
                "alpha_deg": 4.0}},
    {"model": "zoo-airfoil-cst-rans", "title": "NACA 2412 at 20°: outside the trained angles (warned)",
     "inputs": {"T0": 0.169591, "T1": 0.144959, "T2": 0.131433, "M0": 0.024922, "M1": 0.071088, "M2": 0.058960,
                "alpha_deg": 20.0}},
]

app = FastAPI(title="Physics AI Model Gateway", version=VERSION,
              description="Versioned endpoints for physical models: GET /v1/models, POST /v1/models/{name}/predict "
                          "(validity range enforced per the model's OOD policy), run log, metrics, exports. "
                          "Publishing (POST /api/models/...) needs the login, the admin token or an API key.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="MDL")
rt = install_runtime(app, prefix="MDL", name="models", units={"inferences": 50000, "hosted_models": 5},
                     upload_suffixes={".onnx", ".zip", ".json"}, upload_max_mb=100)
GW = ModelGateway(str(rt.root / "registry"))
app.include_router(gateway_router(GW, on_rows=lambda req, n: rt.consume(req, "inferences", n), caller=principal,
                                  max_rows=MAX_ROWS))


def _seed() -> list[str]:
    """Publish the zoo models once (a fresh volume); reported, never fatal."""
    notes = []
    have = set(GW.models())
    for name, zoo, policy in SEED:
        if name in have:
            continue
        try:
            GW.publish(name, zoo, kind="pinneapple.zoo", ood_policy=policy, source="seeded at start-up from pp.zoo",
                       description=f"PINNeAPPle zoo model {zoo}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"{name}: {type(e).__name__}: {e}")
    return notes


SEED_NOTES = _seed()


def _publisher(request: Request, *, admin_only: bool = False) -> str:
    """Who may publish: the login or admin token; API-key holders for ONNX (no code runs on load)."""
    who = principal(request)
    if rt.is_admin(request):
        return who
    if not admin_only and who.startswith("key:"):
        return who
    raise HTTPException(403, "publishing needs the app login, X-Admin-Token" +
                        ("" if admin_only else " or an X-API-Key"))


def _name(s: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", s or ""):
        raise HTTPException(422, "model names: letters, digits, '.', '_', '-', at most 64 characters")
    return s


def _err(e: Exception) -> HTTPException:
    if isinstance(e, PermissionError):
        return HTTPException(403, str(e))
    if isinstance(e, KeyError):
        return HTTPException(404, str(e).strip("'\""))
    return HTTPException(422, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta(request: Request):
    return {"version": VERSION, "kinds": KINDS, "ood_policies": OOD_POLICIES, "examples": EXAMPLES,
            "seed_notes": SEED_NOTES, "max_rows": MAX_ROWS, "runtime": rt.describe(),
            "you": {"principal": principal(request), "admin": rt.is_admin(request)}}


@app.post("/api/models/onnx")
async def publish_onnx(request: Request, file: UploadFile = File(...), name: str = Form(...),  # noqa: B008
                       spec: str = Form(...), ood_policy: str = Form("warn"), description: str = Form("")):
    """An ONNX graph ``inputs (n, d) -> outputs (n, p)``; ``spec`` JSON: ``{"inputs": [...], "outputs": [...],
    "validity_bounds": {input: [lo, hi]}, "method"?, "metrics"?}``."""
    who = _publisher(request)
    rt.consume(request, "hosted_models", 1)
    try:
        sp = json.loads(spec)
    except ValueError as e:
        raise HTTPException(422, f"spec is not JSON: {e}") from None
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.onnx"
        path.write_bytes(await file.read())
        try:
            v = GW.publish(_name(name), str(path), kind="onnx", spec=sp, ood_policy=ood_policy,
                           description=description, source=f"ONNX upload by {who}")
        except Exception as e:
            raise _err(e) from None
    return GW.describe(name, v)


@app.post("/api/models/surrogate")
async def publish_surrogate(request: Request, file: UploadFile = File(...), name: str = Form(...),  # noqa: B008
                            ood_policy: str = Form("warn"), override_reason: str = Form("")):
    """A Surrogate Studio ``model.zip``. Admin only: loading it unpickles the model (sha256-checked against its
    card), which runs code — upload only models you produced."""
    who = _publisher(request, admin_only=True)
    if (os.environ.get("MDL_ALLOW_PICKLE_UPLOAD") or "").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "Pickled surrogate uploads are off on this server (loading one runs code). Publish "
                                 "an ONNX export, or set MDL_ALLOW_PICKLE_UPLOAD=1 for models you produced yourself.")
    rt.consume(request, "hosted_models", 1)
    raw = await file.read()
    with tempfile.TemporaryDirectory() as d:
        try:
            safe_extract_zip(raw, d, ArchiveLimits(max_members=200, max_total_bytes=500 * 2**20))
            card = next(Path(d).rglob("card.json"), None)
            if card is None:
                raise ValueError("no card.json in the zip (export model.zip from the Surrogate Studio)")
            for pkl in Path(d).rglob("*.pkl"):                # defence in depth: refuse known code-running globals
                rep = scan_checkpoint(str(pkl))
                if not rep.safe:
                    raise ValueError(f"{pkl.name} would import {rep.dangerous or rep.errors}: refused")
            v = GW.publish(_name(name), str(card.parent), kind="pinneapple.surrogate", ood_policy=ood_policy,
                           source=f"surrogate model.zip by {who}", override_reason=override_reason or None)
        except HTTPException:
            raise
        except Exception as e:
            raise _err(e) from None
    return GW.describe(name, v)


class RefBody(BaseModel):
    model_ref: dict[str, Any]
    name: str | None = None
    ood_policy: str = "warn"
    stage: str = "production"
    override_reason: str | None = None


@app.post("/api/models/ref")
def publish_ref(request: Request, body: RefBody):
    """A ModelRef (pinneapple_data.contracts): ``zoo:<name>``, or an http(s) URI of a model behind the same API
    (Surrogate Studio, another gateway). Admin only: the gateway will call that URL."""
    _publisher(request, admin_only=True)
    uri = str(body.model_ref.get("uri", ""))
    if uri.startswith(("http://", "https://")):
        allow = (os.environ.get("MDL_REMOTE_ALLOW_HOSTS") or "").split(",")
        try:
            check_outbound_url(uri, allow_hosts=allow)
        except UnsafeURL as e:
            raise HTTPException(422, f"{e} (MDL_REMOTE_ALLOW_HOSTS)") from None
    rt.consume(request, "hosted_models", 1)
    try:
        v = GW.publish_ref(body.model_ref, name=_name(body.name or body.model_ref.get("name", "")),
                           ood_policy=body.ood_policy, stage=body.stage, override_reason=body.override_reason)
    except Exception as e:
        raise _err(e) from None
    return GW.describe(body.name or body.model_ref["name"], v)


class PromoteBody(BaseModel):
    stage: str
    override_reason: str | None = None


@app.post("/api/models/{name}/versions/{version}/promote")
def promote(request: Request, name: str, version: str, body: PromoteBody):
    _publisher(request, admin_only=True)
    try:
        GW.promote(name, GW.resolve(name, version), body.stage, override_reason=body.override_reason)
    except Exception as e:
        raise _err(e) from None
    return GW.describe(name, version)


@app.get("/api/models/{name}/model_ref.json")
def model_ref(name: str, version: str | None = None):
    """The version as a contracts ModelRef (for the uncertainty, sensitivity and optimisation apps)."""
    from pinneapple_data import contracts as C
    try:
        d = GW.describe(name, version)
    except Exception as e:
        raise _err(e) from None
    b = d.get("validity_bounds") or {}
    metrics = {}
    for split in ("test", "shift"):
        x = (d.get("metrics") or {}).get(split)
        if isinstance(x, dict) and isinstance(x.get("error"), (int, float)):
            metrics[f"{split}_error"] = float(x["error"])
    ref = C.ModelRef(name=name, version=d["version"], model_type="surrogate", uri=f"/v1/models/{name}/versions/"
                     f"{d['version']}", sha256=d.get("model_sha256"), framework=d["kind"],
                     inputs=[C.Variable(name=k, low=b.get(k, [None, None])[0], high=b.get(k, [None, None])[1])
                             for k in d["inputs"]], outputs=[C.Variable(name=k) for k in d["outputs"]],
                     metrics=metrics, trust_decision=d.get("trust") if d.get("trust") in
                     ("APPROVED", "REVIEW", "REJECT") else "UNCHECKED",
                     provenance=C.Provenance(app="model-gateway", source=d.get("source", "")),
                     metadata={"kind": d["kind"], "stage": d.get("stage"), "ood_policy": d.get("ood_policy"),
                               "method": d.get("method")})
    return json.loads(ref.to_json())


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
