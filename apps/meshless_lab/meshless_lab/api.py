"""Meshless Method Lab: web API + single-page UI. Solve the same 2-D problem with meshless methods (Kansa, RBF-FD, a
PINN) and mesh-based ones (finite differences, finite elements) and see where each wins: error, cost, conditioning,
stability and convergence under refinement.

Run:  uvicorn meshless_lab.api:app --port 8109   (from apps/meshless_lab)

Environment (all optional; the shared runtime adds MESHLESS_DATA_DIR, MESHLESS_ADMIN_TOKEN, ...):
  MESHLESS_USER / MESHLESS_PASSWORD   HTTP Basic login on everything except /health
  MESHLESS_MAX_JOBS                   studies running at the same time (default 2)
  MESHLESS_MAX_NODES                  largest node set of one refinement level (default 20,000)
  MESHLESS_MAX_PINN_S                 largest PINN budget a study may ask for, in seconds (default 30)
  MESHLESS_FREE_RUNS / MESHLESS_FREE_UNKNOWNS / MESHLESS_FREE_REPORTS   free use per caller and month
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_core.jsonable import to_jsonable
from pinneapple_simulation.meshless import (
    DOMAINS,
    METHODS,
    Problem,
    domain,
    meshless_study,
    point_cloud_domain,
    polygon_domain,
)
from pinneapple_simulation.meshless.contracts import study_result

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
MAX_NODES = int(os.environ.get("MESHLESS_MAX_NODES") or "20000")
MAX_PINN = float(os.environ.get("MESHLESS_MAX_PINN_S") or "30")
SINGULAR = ("((x-0.5)**2+(y-0.5)**2)**(1/3)*cos(2*atan2((x-y)/sqrt(2), -(x+y-1)/sqrt(2))/3)")

EXAMPLES = {
    "lshape": {"title": "Smooth Poisson problem on an L-shape",
               "note": "u = sin(πx)·cos(2y) + xy, three spacings. High-order meshless (RBF-FD, Kansa) against "
                       "second-order FD and FEM, and a PINN on the finest nodes.",
               "spec": {"problem": {"kind": "poisson", "exact": "sin(pi*x)*cos(2*y) + x*y"},
                        "domain": {"preset": "lshape"}, "h": [0.1, 0.05, 0.025], "pinn_budget_s": 3}},
    "corner": {"title": "Re-entrant corner singularity: high order does not help",
               "note": "Laplace on the L-shape with the corner solution r^(2/3) cos(2ψ/3): every method drops to "
                       "order ≈ 4/3 and the cheapest robust one wins.",
               "spec": {"problem": {"kind": "poisson", "exact": SINGULAR}, "domain": {"preset": "lshape"},
                        "h": [0.1, 0.05, 0.025], "pinn_budget_s": 3}},
    "heat": {"title": "Heat equation on a plate with a hole",
             "note": "u = e^(-t)·sin(πx)·sin(πy) + x, Crank-Nicolson to t = 0.1. Time-stepping the Kansa nodal "
                     "operator is a known instability: watch it fail where RBF-FD does not.",
             "spec": {"problem": {"kind": "heat", "exact": "exp(-t)*sin(pi*x)*sin(pi*y) + x", "t_end": 0.1,
                                  "n_steps": 50},
                      "domain": {"preset": "plate_hole"}, "h": [0.1, 0.05, 0.025], "pinn_budget_s": 3}},
    "helmholtz": {"title": "Helmholtz on the disk (k = 6)",
                  "note": "u = sin(3x)·cos(2y), -Δu - 36u = f on the unit disk: a curved boundary and an "
                          "indefinite operator.",
                  "spec": {"problem": {"kind": "helmholtz", "exact": "sin(3*x)*cos(2*y)", "k": 6.0},
                           "domain": {"preset": "disk"}, "h": [0.2, 0.1, 0.05], "pinn_budget_s": 3}},
}

app = FastAPI(title="Meshless Method Lab", version=VERSION,
              description="Describe a 2-D problem (Poisson, Helmholtz or heat with Dirichlet conditions, a "
                          "manufactured solution or a source and boundary value) on a domain (preset, polygon with "
                          "holes, or an uploaded point cloud). POST /api/studies queues a refinement study of Kansa, "
                          "RBF-FD, embedded finite differences, linear finite elements and a PINN: error against the "
                          "exact or a fine FEM reference, cost, condition number, noise amplification, observed order, "
                          "and the winner of each criterion.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="MESHLESS")
rt = install_runtime(app, prefix="MESHLESS", name="meshless", units={"runs": 40, "unknowns": 2_000_000, "reports": 100},
                     upload_max_mb=20, upload_suffixes={".csv", ".parquet", ".json", ".txt"})

LIMITS = [
    "2-D, scalar, linear problems with Dirichlet conditions only (Poisson, Helmholtz, heat); linear elasticity and "
    "Neumann or Robin conditions are not implemented yet, and R-OAMS is not in the library.",
    "Errors are relative L2 at independent evaluation points inside the domain, each method with its own interpolant "
    "(global RBF, local RBF-FD, P1 for FD and FEM, the network): FD is second order at its nodes but its P1 "
    "interpolant decides the error between them.",
    "Without an exact solution the reference is linear FEM at a third of the finest spacing; errors below its "
    "estimated error are not resolved.",
    "Cost is wall time on this server (setup + solve, or the PINN's training budget); Kansa uses dense matrices and "
    "stops at 4,000 nodes.",
    "Node sets are a lattice plus boundary nodes at the same spacing; scattered or adapted nodes change the numbers.",
    "Heat problems march with Crank-Nicolson at a fixed step: the time error is part of every method's error.",
]
VALIDATED = [
    "Observed orders on the L-shape with a smooth manufactured solution: FD 2.0, FEM 2.0, RBF-FD (degree 4) 4.4 "
    "(tests/test_meshless.py)",
    "Kansa reaches 1e-6 on the same problem with a condition number above 1e12; FEM stays below 1e4 (tested)",
    "Fine-FEM reference mode checked against the exact solution (1 - r²)/4 on the disk (tested)",
    "Point clouds: evaluation points are held out from the nodes; FD is refused without the geometry (tested)",
]


def scope() -> dict:
    return {"validated": VALIDATED, "title": "Scope & limits", "noun": "limit", "items_title": "How to read the numbers",
            "band": "No family is assumed to be better: each criterion names its winner from the measured numbers.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can hide an error", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Manufactured solutions", "effect": "optimistic",
                 "detail": "A smooth exact solution favours high-order methods; real problems have corners, "
                           "material jumps and data noise (try the corner example).",
                 "today": "Use a solution with the features of your problem, or no exact solution (FEM reference).",
                 "planned": ""},
                {"topic": "Equal cost", "effect": "check",
                 "detail": "Methods are compared at equal node spacing, and the PINN at a stated wall-clock budget; "
                           "the 'cheapest to reach the tolerance' criterion compares them at equal accuracy.",
                 "today": "Read that criterion for decisions.", "planned": ""},
                {"topic": "Condition numbers", "effect": "check",
                 "detail": "1-norm estimates of the assembled system (onenormest); rows are not equilibrated, so "
                           "Shortley-Weller rows near the boundary raise the FD estimate. The noise amplification "
                           "measures the effect on the solution directly.",
                 "today": "Use both columns.", "planned": ""},
            ]}


# ------------------------------------------------------------------------------------------------ parsing
def _problem(p: dict) -> Problem:
    try:
        return Problem(kind=str(p.get("kind", "poisson")), exact=p.get("exact") or None, source=p.get("source") or None,
                       boundary=p.get("boundary") or None, initial=p.get("initial") or None,
                       k=float(p.get("k", 4.0)), alpha=float(p.get("alpha", 1.0)), t_end=float(p.get("t_end", 0.1)),
                       n_steps=int(min(max(int(p.get("n_steps", 100)), 5), 1000)))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"problem: {e}") from e


def _domain(d: dict, cloud: bytes | None, cloud_name: str = ""):
    if cloud is not None:
        import pandas as pd
        try:
            df = pd.read_parquet(io.BytesIO(cloud)) if cloud_name.lower().endswith(".parquet") else \
                pd.read_csv(io.BytesIO(cloud), sep=None, engine="python")
            cols = {c.lower(): c for c in df.columns}
            bcol = next((cols[c] for c in ("boundary", "is_boundary", "b", "on_boundary") if c in cols), None)
            if "x" not in cols or "y" not in cols or bcol is None:
                raise ValueError("the point cloud needs columns x, y and boundary (1 on the boundary, 0 inside)")
            return point_cloud_domain(df[[cols["x"], cols["y"]]].to_numpy(float), df[bcol].to_numpy(float) > 0.5,
                                      name=cloud_name or "point cloud")
        except (ValueError, KeyError) as e:
            raise HTTPException(422, f"point cloud: {e}") from e
    try:
        if d.get("polygon"):
            holes = [((float(h[0]), float(h[1])), float(h[2])) for h in d.get("holes", [])]
            return polygon_domain(d["polygon"], name="polygon", holes=holes)
        return domain(str(d.get("preset", "square")))
    except (ValueError, TypeError, IndexError) as e:
        raise HTTPException(422, f"domain: {e}") from e


def _spec(spec: dict, cloud: bytes | None = None, cloud_name: str = "") -> dict:
    """Validate a study request now (errors answer 422 at once); the job gets plain JSON."""
    _problem(spec.get("problem") or {})                         # parses the expressions: 422 on a bad one
    dom = _domain(spec.get("domain") or {}, cloud, cloud_name)
    methods = [m for m in (spec.get("methods") or list(METHODS)) if m in METHODS]
    if not methods:
        raise HTTPException(422, f"methods: choose among {list(METHODS)}")
    if dom.cloud is None:
        hs = sorted({float(h) for h in (spec.get("h") or [dom.diameter / 10 / 2 ** k for k in range(3)])}, reverse=True)
        if len(hs) > 6 or hs[-1] <= 0:
            raise HTTPException(422, "h: up to six positive spacings")
        area = dom.area()
        est = [area / h ** 2 + _perimeter(dom) / h for h in hs]
        if max(est) > MAX_NODES:
            raise HTTPException(413, f"the finest spacing h = {hs[-1]:g} gives about {int(max(est)):,} nodes; this "
                                     f"server stops at {MAX_NODES:,} per level")
        n_total = int(sum(est))
    else:
        hs = None
        n_total = dom.cloud.n
        if n_total > MAX_NODES:
            raise HTTPException(413, f"{n_total:,} points: this server stops at {MAX_NODES:,}")
    budget = float(min(max(float(spec.get("pinn_budget_s") or 3.0), 0.2), MAX_PINN))
    opts = {"pinn": {"budget_s": budget}}
    kz = spec.get("kansa") or {}
    if kz.get("eps"):
        opts["kansa"] = {"eps": float(kz["eps"])}
    rf = spec.get("rbf_fd") or {}
    if rf.get("degree"):
        opts["rbf_fd"] = {"degree": int(min(max(int(rf["degree"]), 2), 6))}
    return {"problem": {k: v for k, v in (spec.get("problem") or {}).items()}, "domain": spec.get("domain") or {},
            "h": hs, "methods": methods, "options": opts, "tolerance": float(spec.get("tolerance") or 1e-3),
            "pinn_levels": "all" if spec.get("pinn_all_levels") else "finest", "n_unknowns": n_total,
            "cloud": cloud.decode("utf-8", "replace") if cloud is not None else None, "cloud_name": cloud_name}


def _perimeter(dom) -> float:
    return float(sum(np.linalg.norm(np.asarray(p[2]) - np.asarray(p[1])) if p[0] == "seg" else abs(p[4] - p[3]) * p[2]
                     for p in dom.pieces))


# ------------------------------------------------------------------------------------------------ the job
def run_study(params: dict, ctx) -> dict:
    t0 = time.perf_counter()
    pr = _problem(params["problem"])
    cloud = params.get("cloud")
    dom = _domain(params["domain"], cloud.encode() if cloud else None, params.get("cloud_name", ""))
    k = {"n": 0}
    total = len(params["methods"]) * (len(params["h"]) if params["h"] else 1) + 3

    def progress(msg):
        k["n"] += 1
        ctx.log(msg)
        ctx.progress(min(0.95, k["n"] / total), msg)
    res = meshless_study(pr, dom, params["h"], methods=params["methods"], options=params["options"],
                         pinn_levels=params["pinn_levels"], tolerance=params["tolerance"], progress=progress)
    d = res.to_dict()
    out_dir = Path(ctx.workdir)
    sr = study_result(res, study_id=ctx.job_id, title=f"Meshless study: {pr.kind} on {dom.name}", limits=LIMITS)
    (out_dir / "study_result.json").write_text(sr.model_dump_json(indent=2))
    f = d["fields"]
    names = list(f["methods"])
    lines = ["x,y,reference," + ",".join(names)]
    for i, (x, y) in enumerate(f["points"]):
        lines.append(f"{x:.9g},{y:.9g},{f['truth'][i]:.9g}," + ",".join(f"{f['methods'][m]['values'][i]:.9g}" for m in names))
    (out_dir / "solutions.csv").write_text("\n".join(lines) + "\n")
    out = {"study": d, "checks": [c.model_dump() for c in sr.checks], "limits": LIMITS, "wall_s": time.perf_counter() - t0,
           "files": ["study_result.json", "solutions.csv", "study.json"], "request": {k_: params[k_] for k_ in
                                                                                      ("problem", "domain", "h", "methods", "tolerance")}}
    js = to_jsonable(out)
    (out_dir / "study.json").write_text(json.dumps(js))
    return js


rt.jobs.register("meshless_study", run_study)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_nodes": MAX_NODES, "max_pinn_s": MAX_PINN, "methods": METHODS,
            "domains": {k: domain(k).describe() for k in DOMAINS}, "examples": [{"id": k, **v} for k, v in EXAMPLES.items()],
            "scope": scope(), "limits": LIMITS}


@app.post("/api/domain")
async def preview_domain(spec: Annotated[str, Form()] = "{}", cloud: Annotated[UploadFile | None, File()] = None):
    """Outline and the node set at a spacing ``h`` (for the preview)."""
    try:
        s = json.loads(spec or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(422, "spec must be JSON") from e
    b = await cloud.read() if cloud is not None else None
    dom = _domain(s.get("domain") or {}, b, cloud.filename if cloud is not None else "")
    h = float(s.get("h") or dom.diameter / 20)
    if dom.cloud is None and dom.area() / h ** 2 > MAX_NODES:
        raise HTTPException(413, "spacing too small for a preview")
    ns = dom.nodes(h)
    return to_jsonable({**dom.describe(), "nodes": ns.X.round(6).tolist(), "boundary": ns.boundary.tolist(),
                        "h": ns.h, "n": ns.n})


@app.post("/api/studies")
async def create_study(request: Request, spec: Annotated[str, Form()],
                       cloud: Annotated[UploadFile | None, File()] = None):
    """Queue a study. ``spec`` (JSON): ``problem`` {kind, exact | source + boundary (+ initial), k, alpha, t_end,
    n_steps}, ``domain`` {preset | polygon [[x, y], ...] + holes [[cx, cy, r], ...]}, ``h`` (spacings), ``methods``,
    ``pinn_budget_s``, ``kansa`` {eps}, ``rbf_fd`` {degree}, ``tolerance``. ``cloud``: CSV with x, y, boundary."""
    try:
        s = json.loads(spec)
    except json.JSONDecodeError as e:
        raise HTTPException(422, "spec must be JSON") from e
    b = await cloud.read() if cloud is not None else None
    p = _spec(s, b, cloud.filename if cloud is not None else "")
    return rt.submit(request, "meshless_study", p, units={"runs": 1, "unknowns": p["n_unknowns"]})


@app.post("/api/examples/{eid}")
def run_example(request: Request, eid: str):
    if eid not in EXAMPLES:
        raise HTTPException(404, "no such example")
    p = _spec(EXAMPLES[eid]["spec"])
    return rt.submit(request, "meshless_study", p, units={"runs": 1, "unknowns": p["n_unknowns"]})


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(request: Request, job_id: str, name: str):
    """study_result.json (StudyResult contract), solutions.csv (every method at the evaluation points), study.json."""
    job = rt.job_for(request, job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"the study is {job['status']}")
    if name not in ("study_result.json", "solutions.csv", "study.json"):
        raise HTTPException(404, "no such file")
    rt.consume(request, "reports", 1, ref=job_id)
    return FileResponse(Path(rt.jobs.workdir) / job_id / name, filename=name)


@app.post("/api/expression")
def check_expression(body: dict):
    """Parse a problem and return its source and boundary expressions (LaTeX), or the error."""
    pr = _problem(body)
    return {"ok": True, **pr.describe()}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

