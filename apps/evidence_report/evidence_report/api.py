"""Engineering Evidence Report: web API + single-page UI. Turn experiments, simulations and model tests into
reproducible technical evidence: a report (HTML, PDF) with methodology, conditions, baseline comparison, tests,
limitations and the SHA-256 of every original artifact, and an evidence package (zip) per run or benchmark.

A test is reported as passed only when an evidence record shows that it ran (inputs SHA-256, outputs, time);
otherwise the report prints "not run" (pinneapple_veriphysics.evidence_package).

Run:  uvicorn evidence_report.api:app --port 8113   (from apps/evidence_report)

Environment (all optional; the shared runtime's variables are in apps/_shared/README.md):
  EVR_USER / EVR_PASSWORD   HTTP Basic login on everything except /health (X-API-Key works alongside it)
  EVR_MAX_MB                largest upload in MB (default 200)
  EVR_MAX_HEAVY             reports built at the same time per worker (default 2; the excess gets HTTP 429)
  EVR_FREE_REPORTS          reports per caller and month before HTTP 429 (default 30; "unlimited" removes it)
  EVR_FREE_DOWNLOADS        PDF and package downloads per caller and month (default 30)
  EVR_ADMIN_TOKEN           admin token for /api/keys and the usage totals
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_veriphysics.evidence_package import (
    build_package,
    build_report,
    load_source,
    make_figures,
    render_html,
    render_pdf,
    to_study_result,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install, install_runtime, principal  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ROOT = HERE.parents[2]
PLATE = ROOT / "examples" / "physics_check" / "pass" / "fields" / "heat_plate.npz"
MAX_MB = float(os.environ.get("EVR_MAX_MB") or "200")

app = FastAPI(title="Engineering Evidence Report", version=VERSION,
              description="Upload the outputs of a study (Validation Lab reports, evidence ledgers, `pinneapple check` "
                          "JSON, Simulation Comparator results, StudyResult / ModelRef / DatasetSpec JSON, metrics "
                          "tables in CSV/Parquet/JSON, any other file) with the study's metadata. /api/report returns "
                          "the report as JSON and HTML, /api/report.pdf the PDF, /api/package the evidence package "
                          "(zip with a SHA-256 manifest). A test is printed as passed only when its own evidence "
                          "record shows it ran.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="EVR")
# API keys and usage metering of the shared runtime; reports build in seconds, so they run in the request
rt = install_runtime(app, prefix="EVR", name="evidence", units={"reports": 30, "downloads": 30})
_HEAVY = BusyLimiter("EVR_MAX_HEAVY")

PRINCIPLE = ("A test is reported as passed or failed only when an evidence record shows that it ran: the SHA-256 of "
             "its inputs, its outputs and its start and finish time. A source that only says \"passed\" (a badge, a "
             "summary, a JSON without the record) is printed as not run, with its claim shown next to it.")

SCOPE = {
    "title": "Scope & limits", "noun": "limit", "items_title": "What the report can and cannot say",
    "validated": [
        "A claimed pass without its own record prints \"not run\" in the report, the HTML and the PDF; a record edited "
        "after it was sealed, or another run's record under the same check id, does not count "
        "(tests/pinneapple_veriphysics/test_evidence_ledger.py)",
        "The package manifest holds the SHA-256 of every file; verify_package detects a changed source file",
        "Baseline and candidate numbers are copied from the sources, never recomputed or estimated",
    ],
    "band": PRINCIPLE,
    "tags": {"conservative": "Safe by default", "optimistic": "Can overstate", "check": "Assumption to confirm"},
    "items": [
        {"topic": "Evidence comes from the sources", "effect": "check",
         "detail": "The report checks that a record exists, is complete and intact; it cannot re-run a test or know "
                   "whether the inputs hashed were the right ones.",
         "today": "Produce records where the test runs (Validation Lab, pinneapple_veriphysics.ledger).",
         "planned": "Signed records (pinneapple_security.signing)."},
        {"topic": "Baseline matching", "effect": "check",
         "detail": "Baseline and candidate are matched by name in the run labels of the metrics; metrics with the "
                   "same name are compared as given (units are not converted).",
         "today": "Name runs clearly and use one unit per metric.", "planned": ""},
        {"topic": "Shared contracts", "effect": "check",
         "detail": "StudyResult / ModelRef / DatasetSpec are read from JSON by their type key until "
                   "pinneapple_data.contracts is on main.",
         "today": "Give each JSON a \"type\" field.", "planned": "Import the shared contract classes."},
        {"topic": "Storage", "effect": "conservative",
         "detail": "Nothing is stored: files are processed for the request; the package is the record to keep.",
         "today": "Keep the zip with the original artifacts.", "planned": "Private projects with stored packages."},
    ]}

_EXAMPLE_LOCK = threading.Lock()
_EXAMPLE: dict = {}


def _example_sources() -> tuple[dict, list[tuple[str, bytes]]]:
    """A real study, run on the first request (about 5 s): FEM baseline and PINN candidate on the same Poisson
    problem with a trust card each (recorded in an evidence ledger), plus a Validation Lab run on a temperature
    field. The StudyResult also claims one test that never ran, to show how the report prints it."""
    with _EXAMPLE_LOCK:
        if _EXAMPLE:
            return _EXAMPLE["meta"], _EXAMPLE["files"]
        import numpy as np
        import torch

        import pinneapple as pp
        from pinneapple_physics.pde_environment.builder import ProblemBuilder
        from pinneapple_veriphysics.ledger import (
            EvidenceLedger,
            record_checks,
            sha256_bytes,
            utc_now,
        )
        from pinneapple_veriphysics.validation_lab import (
            config_from_spec,
            run_validation,
            write_inputs,
        )

        spec = (ProblemBuilder("poisson").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
                .bc("dirichlet", field="u", value=0.0, on="x_boundary")
                .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
        problem = pp.PhysicalProblem.from_pde_spec(spec)

        def src(X, ctx=None):
            return -2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

        def exact(X):
            return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

        budget = {"epochs": 200, "hidden_dim": 16, "n_layers": 2, "n_collocation": 256, "seed": 0}
        fem = pp.solve(problem, "fem", ctx={"source_fn": src}, n=24)
        pinn = pp.solve(problem, "pinn", ctx={"source_fn": src}, **budget)
        table = pp.compare(problem, [fem, pinn], reference="exact",
                           options={"exact": {"fn": lambda X: exact(X)[:, None]}}).to_dict()
        labels = {"fem": "FEM P1 24x24 cells", "pinn": "PINN 200 epochs 2x16"}       # no commas: CSV
        lines = ["run,rel_l2_u,rmse_u,max_abs_u,train_s,infer_us_per_point"]
        for r in table["rows"]:
            m = r["metrics"]
            lines.append(f"{labels[r['method']]},{m['relative_l2']['u']:.6g},{m['rmse']['u']:.6g},"
                         f"{m['max_abs']['u']:.6g},{r['wall_time_s']:.4g},{r['infer_us_per_point']:.4g}")
        ledger = EvidenceLedger()
        x = y = np.linspace(0, 1, 25)                         # the FEM nodes
        X, Y = np.meshgrid(x, y, indexing="ij")
        P = np.c_[X.ravel(), Y.ravel()]
        fields, tests = {}, []
        for sol, label in ((fem, "FEM"), (pinn, "PINN")):
            t0 = utc_now()
            u = np.asarray(sol.predict(P), dtype=float).reshape(25, 25)
            card = pp.trust_card({"x": x, "y": y}, u, pde="poisson", source=src, reference=exact, name=label,
                                 dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0})
            entries = record_checks(ledger, card.checks, prefix=f"trust card {label}",
                                    inputs_sha256=sha256_bytes(u.tobytes()), started_at=t0, finished_at=utc_now(),
                                    tool="pp.trust_card", tool_version=pp.__version__)
            tests += [{"id": e.check_id, "name": e.name, "passed": e.passed} for e in entries]
            fields[f"u_{label.lower()}"] = u
        tests.append({"id": "grid convergence (GCI, 3 meshes)", "name": "Grid convergence index on three meshes",
                      "passed": True, "category": "verification"})
        buf = io.BytesIO()
        np.savez_compressed(buf, x=x, y=y, u_exact=exact(P).reshape(25, 25), **fields)
        wbuf = io.BytesIO()
        torch.save(pinn.model.state_dict(), wbuf)
        model_ref = {"type": "ModelRef", "name": "pinn-poisson-200ep", "source": "pp.solve", "framework": "torch",
                     "sha256": sha256_bytes(wbuf.getvalue()), "parameters": budget}
        study = {"type": "StudyResult", "name": "Poisson on the unit square: FEM baseline vs PINN candidate",
                 "parameters": {"pde": "laplace(u) = -2 pi^2 sin(pi x) sin(pi y)", "domain": "[0,1] x [0,1]",
                                "bc": "u = 0 on the boundary", "exact": "sin(pi x) sin(pi y)",
                                "comparison points": table["n_points"], "trust-card grid": "25 x 25 (FEM nodes)",
                                **{f"PINN {k}": v for k, v in budget.items()}},
                 "metrics": {"rel_l2_u": table["rows"][1]["metrics"]["relative_l2"]["u"]},
                 "tests": tests, "evidence": ledger.to_dict()}
        with tempfile.TemporaryDirectory() as d:
            write_inputs(d, {"heat_plate.npz": PLATE.read_bytes()})
            (Path(d) / "pinneapple-check.toml").write_text(config_from_spec({"results": [
                {"path": "heat_plate.npz", "physics": "heat", "parameters": {"k": "10 W/(m*K)", "q": "1000 W/m3"},
                 "units": {"T": "K"}, "bounds": {"T": [250, 400]}, "bc": {"x=max": {"T": "300 + 25*(1 - y**2)"}}}]}))
            lab = run_validation(d, title="Heat spreader temperature field").to_dict()
        files = [("study_result.json", json.dumps(study, indent=1).encode()),
                 ("metrics.csv", ("\n".join(lines) + "\n").encode()),
                 ("model_ref.json", json.dumps(model_ref, indent=1).encode()),
                 ("fields.npz", buf.getvalue()),
                 ("validation_lab_heat_plate.json", json.dumps(lab).encode())]
        meta = {"title": "Poisson benchmark: FEM baseline vs PINN candidate", "project": "PINNeAPPle example study",
                "author": "Engineering Evidence Report (example)", "baseline": "FEM", "candidate": "PINN",
                "purpose": "Is a small PINN an acceptable replacement for linear finite elements on this problem, and "
                           "does the heat-spreader temperature field pass its physics checks?",
                "methodology": "Both methods solve the same Poisson problem (pp.solve); pp.compare scores them "
                               "against the exact solution on 4,096 points. A trust card (PDE residual, boundary "
                               "conditions, conservation, reference error) is computed for each on the 25 x 25 FEM "
                               "nodes and every check is recorded in an evidence ledger. The temperature field is "
                               "checked by the Physics Validation Lab (pinneapple check).",
                "conditions": {"hardware": "this server, CPU", "PINN seeds": 1},
                "limitations": ["One seed and a 200-epoch budget: the PINN result says nothing about larger budgets.",
                                "The grid-convergence claim in study_result.json was planted without a record, to "
                                "show that the report prints it as not run."],
                "references": ["pp.solve / pp.compare / pp.trust_card: https://github.com/PINNeAPPle-Labs/PINNeAPPle",
                               "Validation Lab rules: pinneapple check (docs/guides/physics_checks_in_ci.md)"]}
        _EXAMPLE.update(meta=meta, files=files)
        return meta, files


def _sources(files: list[tuple[str, bytes]], urls: dict[str, str]):
    if not files:
        raise HTTPException(422, "Upload at least one result, report or metrics file.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Uploads larger than {MAX_MB:g} MB.")
    return [load_source(n, b, url=urls.get(n, "")) for n, b in files]


def _build(meta: dict, files: list[tuple[str, bytes]], urls: dict[str, str], what: str):
    srcs = _sources(files, urls)
    with _HEAVY, tempfile.TemporaryDirectory(prefix="evr-") as d:
        rep = build_report(meta, srcs)
        figs = make_figures(rep, d)
        if what == "report":
            return {"report": rep, "html": render_html(rep, figs), "study_result": to_study_result(rep),
                    "scope": SCOPE}
        if what == "pdf":
            return render_pdf(rep, figs)
        html = render_html(rep, figs)
        pdf = render_pdf(rep, figs)
        return build_package(rep, srcs, html=html, pdf=pdf, figures=figs)


def _parse(meta: str, urls: str) -> tuple[dict, dict]:
    try:
        m = json.loads(meta) if meta.strip() else {}
        u = json.loads(urls) if urls.strip() else {}
        if not isinstance(m, dict) or not isinstance(u, dict):
            raise ValueError("meta and urls must be JSON objects")
    except ValueError as e:
        raise HTTPException(422, f"meta/urls: {e}") from e
    return m, u


async def _read(files: list[UploadFile] | None) -> list[tuple[str, bytes]]:
    return [(Path(f.filename or f"file_{i}").name, await f.read()) for i, f in enumerate(files or [])]


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_mb": MAX_MB, "principle": PRINCIPLE, "scope": SCOPE, "runtime": rt.describe(),
            "kinds": {"validation-lab": "Physics Validation Lab report (with its evidence ledger)",
                      "ledger": "evidence ledger (pinneapple-evidence-ledger/v1)",
                      "pinneapple-check": "pinneapple check --json (no records: its checks read not run)",
                      "comparator": "Simulation Comparator result",
                      "contract": "StudyResult / ModelRef / DatasetSpec / ParameterSpace JSON",
                      "metrics": "metrics table (CSV, Parquet, JSON rows): one row per run",
                      "metadata": "other JSON (simulation metadata, configuration)",
                      "file": "any other file, referenced by SHA-256"}}


Files = Annotated[list[UploadFile] | None, File()]
Text = Annotated[str, Form()]


def _metered(request: Request, unit: str, build):
    """Check the caller's free limit for ``unit`` (HTTP 429 past it), build, and count it only if it succeeded."""
    who = principal(request)
    rt.meter.check(who, unit, 1)
    out = build()
    rt.meter.record(who, unit, 1)
    return out


@app.post("/api/report")
async def api_report(request: Request, files: Files = None, meta: Text = "", urls: Text = ""):
    """files: the study's outputs. meta: JSON {title, project, author, purpose, methodology, conditions {},
    limitations [], baseline, candidate, references []}. urls: JSON {file name: URL of the original}. Returns the
    report, its HTML and the report as a StudyResult contract. Metered as one report."""
    m, u = _parse(meta, urls)
    fs = await _read(files)
    return _metered(request, "reports", lambda: _build(m, fs, u, "report"))


@app.post("/api/report.pdf")
async def api_pdf(request: Request, files: Files = None, meta: Text = "", urls: Text = ""):
    """The report as PDF. Metered as one download."""
    m, u = _parse(meta, urls)
    fs = await _read(files)
    pdf = _metered(request, "downloads", lambda: _build(m, fs, u, "pdf"))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="evidence-report.pdf"'})


@app.post("/api/package")
async def api_package(request: Request, files: Files = None, meta: Text = "", urls: Text = ""):
    """The evidence package: report.json/html/pdf, study_result.json, evidence-ledger.json, figures/, sources/ and
    MANIFEST.json. Metered as one download."""
    m, u = _parse(meta, urls)
    fs = await _read(files)
    z = _metered(request, "downloads", lambda: _build(m, fs, u, "zip"))
    return Response(z, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="evidence-package.zip"'})


@app.get("/api/example")
def api_example():
    m, files = _example_sources()
    out = _build(m, files, {}, "report")
    out["example"] = {"meta": m, "files": [{"name": n, "bytes": len(b)} for n, b in files]}
    return out


@app.get("/api/example/{what}")
def api_example_file(what: str):
    m, files = _example_sources()
    if what == "pdf":
        return Response(_build(m, files, {}, "pdf"), media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="evidence-report-example.pdf"'})
    if what == "package":
        return Response(_build(m, files, {}, "zip"), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="evidence-package-example.zip"'})
    if what == "sources":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in files:
                z.writestr(n, b)
            z.writestr("meta.json", json.dumps(m, indent=1))
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="example-sources.zip"'})
    raise HTTPException(404, "pdf, package or sources")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
