"""Physics Validation Lab: web API + single-page UI. Is a model or a result physically and numerically consistent?
Residuals, conservation, boundary conditions, constraints, units, reference error and out-of-range use, reported in
three separate groups (numerical verification, validation against data, physical consistency) with an evidence
record per check and a reproducible report.

Run:  uvicorn validation_lab.api:app --port 8107   (from apps/validation_lab)

Environment (all optional):
  VAL_USER / VAL_PASSWORD   HTTP Basic login on everything except /health
  VAL_MAX_MB                largest upload in MB (default 200)
  VAL_MAX_HEAVY             validations running at the same time per worker (default 2; the excess gets HTTP 429)
  VAL_DATA_DIR              where the 3-D views of recent validations are kept (default: the shared app data dir)
  VAL_VIEW_MAX_FACES        triangles of a 3-D view before it is decimated (default 250,000)
  VAL_VIEWS_KEPT            3-D views kept on disk, newest first (default 300)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_check import ConfigError
from pinneapple_check.rules import RULES
from pinneapple_core.service import data_dir
from pinneapple_data.cae.upload import expand
from pinneapple_veriphysics.mesh_views import mesh_views
from pinneapple_veriphysics.validation_lab import (
    CATEGORIES,
    CONFIG_NAME,
    DISCLAIMER,
    CodeNotAllowed,
    category_of,
    config_from_spec,
    render_html,
    run_validation,
    tables_to_csv,
    write_inputs,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
APP_EXAMPLES = HERE.parent / "examples"
ROOT = HERE.parents[2]
CHECK_EXAMPLES = ROOT / "examples" / "physics_check"
MAX_MB = float(os.environ.get("VAL_MAX_MB") or "200")
VIEW_MAX_FACES = int(os.environ.get("VAL_VIEW_MAX_FACES") or "250000")
VIEWS_KEPT = int(os.environ.get("VAL_VIEWS_KEPT") or "300")
VIEWS = data_dir("validate", "VAL") / "views"
VIEWS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Physics Validation Lab", version=VERSION,
              description="Upload predictions or simulation results (OpenFOAM case zip, VTU/VTK, NPZ, HDF5, CSV, "
                          "Parquet, JSON tables) with a `pinneapple check` configuration, or describe the equations, "
                          "boundary conditions, units and reference in a form, or name a zoo model. POST /api/validate "
                          "returns the report: every check grouped into numerical verification, validation against "
                          "data and physical consistency, with an evidence record (inputs SHA-256, outputs, time). "
                          "3-D results with a reference also get `views3d`: prediction, reference and pointwise error on "
                          "the mesh with the regions of largest error (glTF at /api/views/{id}.glb).")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="VAL")
_HEAVY = BusyLimiter("VAL_MAX_HEAVY")

EXAMPLES = {
    "pass": {"title": "Fifteen results and models that hold",
             "note": "OpenFOAM cavity and channel, conduction, elasticity, a 1-D conservation law, a PINN, a surrogate, "
                     "a Darcy dataset and signed artefacts: the fixtures of `pinneapple check` where every check holds.",
             "dir": CHECK_EXAMPLES / "pass"},
    "fail": {"title": "The same targets, one defect each",
             "note": "A leaking outlet, a wrong boundary value, a PDE residual, a unit mismatch, an out-of-range query, a "
                     "regression against the reference, a tampered artefact...: each defect is caught in its group.",
             "dir": CHECK_EXAMPLES / "fail"},
    "plate": {"title": "Temperature predictions from a form",
              "note": "A steady temperature field (heat_plate.npz) checked against k ∇²T + q = 0, the boundary value on "
                      "x = max, its units and bounds: the configuration is written from the form.",
              "files": [CHECK_EXAMPLES / "pass" / "fields" / "heat_plate.npz"],
              "spec": {"results": [{"path": "heat_plate.npz", "physics": "heat",
                                    "parameters": {"k": "10 W/(m*K)", "q": "1000 W/m3"}, "units": {"T": "K"},
                                    "bounds": {"T": [250, 400]}, "bc": {"x=max": {"T": "300 + 25*(1 - y**2)"}}}]}},
    "cantilever3d": {"title": "A reduced model at its worst design, against the FEM, in 3-D",
                     "note": "Chosen to fail: the ROM Lab's POD + GPR model (4.4 % mean error over its 20 held-out "
                             "designs) at the one held-out design where it is worst, near the edge of its training box "
                             "(H 0.289 m, W 0.173 m, load 4.8° from vertical), against a fresh FEM solve. The clamp "
                             "holds; the reference check fails (21 % displacement, 28 % stress). This is not the ROM's "
                             "typical quality; the 3-D view shows where it breaks down.",
                     "files": [APP_EXAMPLES / "cantilever_rom.vtu", APP_EXAMPLES / "cantilever_fem.vtu"],
                     "spec": {"results": [{"path": "cantilever_rom.vtu", "units": {"D": "m", "von_mises": "MPa"},
                                           "bounds": {"von_mises": [0, 500]}, "bc": {"x=min": {"D": [0, 0, 0]}},
                                           "reference": {"path": "cantilever_fem.vtu", "rel-l2": 0.05}}]}},
    "zoo": {"title": "Zoo heat-sink model at three operating points",
            "note": "heatsink_forced_mlp (a ModelRef to the PINNeAPPle zoo) queried at three designs; the third uses "
                    "20 m/s air, outside the 0.5–6 m/s the model was trained and validated on.",
            "files": [APP_EXAMPLES / "heatsink_queries.csv"],
            "spec": {"models": [{"zoo": "heatsink_forced_mlp", "queries": "heatsink_queries.csv",
                                 "bounds": {"r_base_k_w": [0, 10], "pressure_drop_pa": [0, 2000]}}]}},
}

SCOPE = {
    "title": "Scope & limits", "noun": "limit", "items_title": "What a pass means here",
    "validated": [
        "Every rule fails exactly where expected on the `pinneapple check` fixtures (one defect per target) and passes "
        "on the clean ones (tests/test_physics_check.py): conservation, BCs, residuals, units, OOD, regression, "
        "integrity",
        "Each check that ran has an evidence record: inputs SHA-256, outputs, start/finish time; a record edited "
        "afterwards no longer counts (tests/pinneapple_veriphysics/test_evidence_ledger.py)",
        "The configuration and the input hashes in the report reproduce the run with `pinneapple check`",
    ],
    "band": DISCLAIMER,
    "tags": {"conservative": "Safe by default", "optimistic": "Can hide a problem", "check": "Assumption to confirm"},
    "items": [
        {"topic": "Coverage", "effect": "optimistic",
         "detail": "A check runs only when its inputs exist: no reference, no validation against data; no declared "
                   "physics, no residual or conservation check. Groups where nothing ran are listed per target.",
         "today": "Declare physics, boundary conditions, units and a reference for each target.",
         "planned": "Credibility levels per factor once pp.validate (#734) is on main."},
        {"topic": "Uploaded models", "effect": "conservative",
         "detail": "Python callables and checkpoints are not run on this server (they would execute uploaded code). "
                   "Zoo models can be referenced by name; your own model is validated through its predictions.",
         "today": "Upload predictions on the points, or run `pinneapple check` in your CI (GitHub Action).",
         "planned": "ONNX models in a sandbox."},
        {"topic": "Tolerances", "effect": "check",
         "detail": "Defaults of `pinneapple check` (e.g. mass imbalance 0.1 %, relative residual 5 %, relative L2 1 %); "
                   "they decide pass or fail and are written in the report.",
         "today": "Set `tolerances` per target for your application.", "planned": ""},
        {"topic": "Report format", "effect": "check",
         "detail": "The report is `pinneapple-validation-lab/v1` JSON (and HTML); the shared `validation-report/v1` "
                   "artifact of pp.validate is not emitted yet.",
         "today": "Store the JSON: it carries the configuration, input hashes and the evidence ledger.",
         "planned": "Export to validation-report/v1 when #738 is on main."},
    ]}


def _run(files: dict[str, bytes], config: str, spec: dict | None, allow_code: bool = False) -> dict:
    if not files and not (spec and spec.get("models")):
        raise HTTPException(422, "Upload the predictions or results to check, or reference a zoo model.")
    if sum(len(b) for b in files.values()) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Uploads larger than {MAX_MB:g} MB.")
    with tempfile.TemporaryDirectory(prefix="val-") as d:
        try:
            write_inputs(d, files)
            if spec:
                config = config_from_spec(spec)
            elif not config.strip():
                cf = Path(d) / CONFIG_NAME
                if not cf.is_file():
                    raise HTTPException(422, f"Give a {CONFIG_NAME} (upload it or paste it) or fill in the form.")
                config = cf.read_text(encoding="utf-8")
            config, _ = tables_to_csv(d, config)
            (Path(d) / CONFIG_NAME).write_text(config, encoding="utf-8")
            with _HEAVY:
                lab = run_validation(d, allow_code=allow_code).to_dict()
                lab["views3d"] = _views(d, lab)
        except HTTPException:
            raise
        except CodeNotAllowed as e:
            raise HTTPException(422, str(e)) from e
        except (ConfigError, ValueError, FileNotFoundError, KeyError) as e:
            raise HTTPException(422, str(e.args[0] if e.args else e)) from e
    lab["scope"] = SCOPE
    return lab


def _views(workdir: str, lab: dict) -> list[dict]:
    """3-D views of the 3-D results that have a reference (prediction, reference, error on the mesh). They are not
    part of the hashed report: the glTF files are kept on disk for a while, newest first."""
    try:
        views = mesh_views(workdir, report=lab, max_faces=VIEW_MAX_FACES)
    except Exception as e:                                   # a view must never cost the validation itself
        lab["notes"] = [*lab.get("notes", []), f"no 3-D view: {e}"]
        return []
    out = []
    for v in views:
        vid = uuid.uuid4().hex
        (VIEWS / f"{vid}.glb").write_bytes(v["view"].glb())
        out.append({"id": vid, "target": v["target"], "reference": v["reference"], "meta": v["view"].meta,
                    "errors": v["errors"], "notes": v["notes"], "glb": f"/api/views/{vid}.glb"})
    old = sorted(VIEWS.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)[VIEWS_KEPT:]
    for p in old:
        p.unlink(missing_ok=True)
    return out


@app.get("/api/views/{vid}.glb")
def api_view(vid: str):
    """The glTF of a 3-D view returned with a report (``views3d[].glb``)."""
    p = VIEWS / f"{vid}.glb"
    if not re.fullmatch(r"[0-9a-f]{32}", vid) or not p.is_file():
        raise HTTPException(404, "This view is no longer kept; run the validation again.")
    return FileResponse(p, media_type="model/gltf-binary")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_mb": MAX_MB, "disclaimer": DISCLAIMER, "categories": CATEGORIES,
            "rules": [{"id": r.id, "name": r.name, "title": r.title, "severity": r.severity,
                       "category": category_of(r.id)} for r in RULES.values()],
            "examples": [{"id": k, "title": v["title"], "note": v["note"]} for k, v in EXAMPLES.items()],
            "zoo": _zoo_models(), "scope": SCOPE}


def _zoo_models() -> list[str]:
    try:
        import pinneapple as pp
        return list(pp.zoo.list())
    except Exception:                                          # the zoo is optional for the rest of the app
        return []


@app.post("/api/validate")
async def api_validate(files: Annotated[list[UploadFile] | None, File()] = None, config: Annotated[str, Form()] = "",
                       spec: Annotated[str, Form()] = ""):
    """files: predictions/results/references (zips are expanded; Parquet and JSON tables become CSV).
    config: a `pinneapple-check.toml` (or upload it among the files). spec: instead of a configuration, JSON
    {"results": [{"path", "physics", "parameters", "units", "bounds", "bc", "ic", "reference", "tolerances"}],
    "models": [{"zoo": name, "queries": file}]}."""
    raw = [(f.filename or f"file_{i}", await f.read()) for i, f in enumerate(files or [])]
    try:
        # .npz files are zip archives too: expand only what is named .zip
        zips = [f for f in raw if f[0].lower().endswith(".zip")]
        fs = {**(expand(zips) if zips else {}), **{n.replace("\\", "/").lstrip("/"): b for n, b in raw
                                                    if not n.lower().endswith(".zip")}}
        sp = json.loads(spec) if spec.strip() else None
    except ValueError as e:
        raise HTTPException(422, f"spec: {e}") from e
    return _run(fs, config, sp)


@app.get("/api/example/{eid}")
def api_example(eid: str):
    if eid not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    ex = EXAMPLES[eid]
    if "dir" in ex:                                              # the server's own fixtures (their models included)
        root = ex["dir"]
        fs = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
              and p.name != "expected.json"}
        out = _run(fs, "", None, allow_code=True)
    else:
        out = _run({p.name: p.read_bytes() for p in ex["files"]}, "", ex["spec"])
    out["example"] = {"id": eid, "title": ex["title"], "note": ex["note"], "spec": ex.get("spec")}
    return out


@app.post("/api/report.html", response_class=HTMLResponse)
def api_report_html(report: Annotated[dict, Body()]):
    """The self-contained HTML of a report returned by /api/validate."""
    if report.get("schema") != "pinneapple-validation-lab/v1":
        raise HTTPException(422, "Not a Validation Lab report.")
    report = {k: v for k, v in report.items() if k != "views3d"}
    return HTMLResponse(render_html(report), headers={"Content-Disposition": 'attachment; filename="validation.html"'})


@app.post("/api/config")
def api_config(spec: Annotated[dict, Body()]):
    """The `pinneapple-check.toml` a spec produces (to run the same checks locally or in CI)."""
    try:
        return Response(config_from_spec(spec), media_type="application/toml")
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from e


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
