"""VerifiedPhysics: the ``pinneapple_analysis.verification`` modules over HTTP.

* ``POST /v1/analyze`` (job): ``pipeline.analyze`` -- dimensional analysis, an LLM-drafted registered preset, PINN
  training per candidate architecture, ``PhysicsGuardrail``, physics-aware selection, provenance; optionally
  ensemble uncertainty (``uncertainty_inverse.quantify_uncertainty``) and a model-discrepancy correction from
  reference data (``causal_discrepancy.fit_discrepancy_model``).
* ``POST /v1/intervention`` (job): ``causal_discrepancy.run_intervention`` ("what if parameter X changes").
* ``POST /v1/inverse_problem`` (job): ``uncertainty_inverse.solve_inverse_problem`` (steady 1-D heat conduction only).
* ``GET /v1/solver_families``, ``POST /v1/recommend_solver``: ``solver_orchestration``.
* ``GET /v1/knowledge/...``: ``knowledge_graph`` built from the live preset registry.
* ``POST /v1/classify_geometry``, ``POST /v1/assess_mesh_quality`` (mesh upload): ``geometry_intelligence``,
  ``mesh_intelligence``.

How it relates to the other apps: ``apps/validation_lab`` checks results you bring (fields, models) and keeps an
evidence ledger; ``apps/evidence_report`` packages evidence into reports; ``apps/mesh_quality`` is the full
volume-mesh quality service. This app is the LLM-to-trained-PINN pipeline and the analysis modules it came with.

Run:  uvicorn verified_physics.api:app --port 8143   (from apps/verified_physics)

Environment (all optional; the rest in apps/_shared/README.md):
  VPHYS_USER / VPHYS_PASSWORD(_HASH)        login on everything except /health
  VPHYS_LLM_PROVIDER / VPHYS_LLM_MODEL      ollama (default) + llama3.2:3b; hosted providers read their key from
                                            ANTHROPIC_API_KEY / OPENAI_API_KEY
  VPHYS_MAX_EPOCHS                          largest training budget per job (default 3000)
  VPHYS_FREE_RUNS, VPHYS_FREE_CHECKS        free training jobs (20) and quick checks (500) per caller and month
"""
from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from pinneapple_core.jsonable import to_jsonable

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
PROVIDER = os.environ.get("VPHYS_LLM_PROVIDER", "ollama")
MODEL = os.environ.get("VPHYS_LLM_MODEL") or ("llama3.2:3b" if PROVIDER == "ollama" else None)
MAX_EPOCHS = int(os.environ.get("VPHYS_MAX_EPOCHS") or 3000)

app = FastAPI(title="VerifiedPhysics", version=VERSION,
              description="LLM-drafted physics, executed and mechanically re-checked, with the analysis modules.")
install(app, prefix="VPHYS")
rt = install_runtime(app, prefix="VPHYS", name="verified_physics", units={"runs": 20, "checks": 500}, max_jobs=1,
                     upload_suffixes={".stl", ".obj", ".ply", ".step", ".stp"})


def _plain(obj: Any) -> Any:
    return to_jsonable(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj)


# ------------------------------------------------------------------ jobs
def analyze_job(p: dict, ctx) -> dict:
    import torch

    import pinneapple_physics as pp
    from pinneapple_analysis.verification.pipeline import analyze

    ctx.progress(0.05, f"drafting with {PROVIDER}:{MODEL}")
    res = analyze(p["description"], physical_parameters=p.get("physical_parameters"), flow_geometry=p["flow_geometry"],
                  provider=PROVIDER, model_name=MODEL, candidate_architectures=p.get("candidate_architectures"),
                  epochs=p["epochs"], n_collocation=p["n_collocation"], residual_threshold=p["residual_threshold"],
                  reference_benchmark=p.get("reference_benchmark"))
    out: dict[str, Any] = {"rejected": res.rejected, "reason": res.rejection_reason,
                           "provenance": _plain(res.provenance),
                           "not_checked": "whether the drafted preset is the right physics for the description"}
    model = res.best_candidate.model if res.best_candidate else None
    if model is not None and (p.get("quantify_uncertainty") or p.get("reference_x")):
        spec = pp.get_preset(res.provenance.drafted_preset, **res.provenance.drafted_preset_kwargs)
        if p.get("quantify_uncertainty"):
            from pinneapple_analysis.verification.uncertainty_inverse import quantify_uncertainty

            ctx.progress(0.7, "ensemble uncertainty")
            try:
                out["uncertainty"] = _plain(quantify_uncertainty(spec, model, method="ensemble",
                                                                 n_samples=p["uq_n_samples"]))
            except Exception as e:  # noqa: BLE001 -- reported next to the main result
                out["uncertainty_error"] = f"{type(e).__name__}: {e}"
        if p.get("reference_x") and p.get("reference_y"):
            from pinneapple_analysis.verification.causal_discrepancy import fit_discrepancy_model

            ctx.progress(0.85, "model discrepancy")
            try:
                d = fit_discrepancy_model(spec, model, torch.tensor(p["reference_x"], dtype=torch.float32),
                                          torch.tensor(p["reference_y"], dtype=torch.float32), epochs=200)
                out["model_discrepancy"] = {"variance_explained_test": d.variance_explained_test,
                                            "test_rmse_physics_only": d.test_rmse_physics_only,
                                            "test_rmse_hybrid": d.test_rmse_hybrid}
            except Exception as e:  # noqa: BLE001
                out["model_discrepancy_error"] = f"{type(e).__name__}: {e}"
    ctx.progress(1.0, "rejected" if res.rejected else "done")
    return to_jsonable(out)


def intervention_job(p: dict, ctx) -> dict:
    from pinneapple_analysis.verification.causal_discrepancy import run_intervention

    ctx.progress(0.05, "baseline")
    rep = run_intervention(p["base_description"], p["param_name"], p["param_values"],
                           physical_parameters=p.get("physical_parameters"), provider=PROVIDER, model_name=MODEL,
                           epochs=p["epochs"], n_collocation=p["n_collocation"])
    return _plain(rep)


def inverse_job(p: dict, ctx) -> dict:
    import torch

    import pinneapple_physics as pp
    from pinneapple_analysis.verification.uncertainty_inverse import solve_inverse_problem

    spec = pp.get_preset(p["preset_name"], **(p.get("preset_kwargs") or {}))
    obs = (torch.tensor(p["observed_x"], dtype=torch.float32), torch.tensor(p["observed_y"], dtype=torch.float32))
    try:
        return _plain(solve_inverse_problem(spec, obs, p["unknown_param_names"], initial_guess=p.get("initial_guess")))
    except NotImplementedError as e:
        return {"error": f"not available for this preset: {e}"}


rt.jobs.register("analyze", analyze_job)
rt.jobs.register("intervention", intervention_job)
rt.jobs.register("inverse", inverse_job)


class AnalyzeRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=4000)
    physical_parameters: dict[str, float] | None = None
    flow_geometry: str = "internal_pipe"
    candidate_architectures: list[str] | None = Field(None, max_length=5)
    epochs: int = Field(1500, ge=10)
    n_collocation: int = Field(1024, ge=64, le=8192)
    residual_threshold: float = Field(1e-2, gt=0)
    reference_benchmark: str | None = None
    quantify_uncertainty: bool = False
    uq_n_samples: int = Field(5, ge=2, le=10)
    reference_x: list[list[float]] | None = None
    reference_y: list[list[float]] | None = None


class InterventionRequest(BaseModel):
    base_description: str = Field(..., min_length=3, max_length=4000)
    param_name: str
    param_values: list[float] = Field(..., min_length=1, max_length=6)
    physical_parameters: dict[str, float] | None = None
    epochs: int = Field(200, ge=10)
    n_collocation: int = Field(256, ge=64, le=8192)


class InverseRequest(BaseModel):
    preset_name: str
    preset_kwargs: dict[str, Any] | None = None
    observed_x: list[list[float]]
    observed_y: list[list[float]]
    unknown_param_names: list[str]
    initial_guess: dict[str, float] | None = None


class PresetRequest(BaseModel):
    preset_name: str
    preset_kwargs: dict[str, Any] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "llm": {"provider": PROVIDER, "model": MODEL}, "max_epochs": MAX_EPOCHS}


@app.post("/v1/analyze")
def analyze_route(request: Request, req: AnalyzeRequest):
    p = req.model_dump()
    p["epochs"] = min(p["epochs"], MAX_EPOCHS)
    n = 1 + (p["uq_n_samples"] if p["quantify_uncertainty"] else 0)
    return rt.submit(request, "analyze", p, units={"runs": n})


@app.post("/v1/intervention")
def intervention_route(request: Request, req: InterventionRequest):
    p = req.model_dump()
    p["epochs"] = min(p["epochs"], MAX_EPOCHS)
    return rt.submit(request, "intervention", p, units={"runs": 1 + len(p["param_values"])})


@app.post("/v1/inverse_problem")
def inverse_route(request: Request, req: InverseRequest):
    return rt.submit(request, "inverse", req.model_dump(), units={"runs": 1})


@app.get("/v1/runs/{job_id}")
def run(request: Request, job_id: str):
    return rt.job_for(request, job_id)


@app.get("/v1/solver_families")
def solver_families(request: Request):
    from pinneapple_analysis.verification.solver_orchestration import list_available_solver_families

    rt.consume(request, "checks")
    return _plain(list_available_solver_families())


def _spec(req: PresetRequest):
    import pinneapple_physics as pp

    try:
        return pp.get_preset(req.preset_name, **(req.preset_kwargs or {}))
    except (KeyError, TypeError) as e:
        raise HTTPException(422, str(e)) from None


@app.post("/v1/recommend_solver")
def recommend_solver(request: Request, req: PresetRequest):
    from pinneapple_analysis.verification.solver_orchestration import select_solver_family

    rt.consume(request, "checks")
    return _plain(select_solver_family(_spec(req)))


_GRAPH: dict[str, Any] = {}


def _graph():
    if "g" not in _GRAPH:
        from pinneapple_analysis.verification.knowledge_graph import build_knowledge_graph

        _GRAPH["g"] = build_knowledge_graph()
    return _GRAPH["g"]


@app.get("/v1/knowledge/stats")
def knowledge_stats():
    from pinneapple_analysis.verification.knowledge_graph import graph_stats

    return _plain(graph_stats(_graph()))


@app.get("/v1/knowledge/preset/{name}")
def knowledge_preset(name: str):
    from pinneapple_analysis.verification.knowledge_graph import explain_preset

    try:
        return _plain(explain_preset(_graph(), name))
    except KeyError:
        raise HTTPException(404, f"unknown preset {name!r}") from None


@app.get("/v1/knowledge/phenomenon/{name}")
def knowledge_phenomenon(name: str):
    from pinneapple_analysis.verification.knowledge_graph import find_presets_for_phenomenon

    return {"phenomenon": name, "presets": find_presets_for_phenomenon(_graph(), name)}


@app.get("/v1/knowledge/verified_equations")
def knowledge_verified():
    from pinneapple_analysis.verification.knowledge_graph import find_verified_equations

    return {"verified_equations": find_verified_equations(_graph())}


async def _load_mesh(file: UploadFile):
    from pinneapple_design.geometry.io.trimesh_bridge import TrimeshBridge

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".stl", ".obj", ".ply"}:
        raise HTTPException(415, "upload an .stl, .obj or .ply surface mesh")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
    try:
        return TrimeshBridge().load(tmp.name)
    except Exception as e:  # noqa: BLE001 -- unreadable upload
        raise HTTPException(422, f"could not read the mesh: {e}") from None
    finally:
        os.unlink(tmp.name)


@app.post("/v1/classify_geometry")
async def classify_geometry(request: Request, file: UploadFile = File(...), problem_description: str = Form(""),  # noqa: B008
                            use_llm: bool = Form(False)):
    from pinneapple_analysis.verification.geometry_intelligence import classify_geometry as classify

    rt.consume(request, "checks")
    mesh = await _load_mesh(file)
    return _plain(classify(mesh, problem_description, provider=PROVIDER if use_llm else None,
                           model=MODEL if use_llm else None))


@app.post("/v1/assess_mesh_quality")
async def assess_mesh_quality(request: Request, file: UploadFile = File(...), target_reynolds: float | None = Form(None),  # noqa: B008
                              target_yplus: float = Form(1.0)):
    from pinneapple_analysis.verification.mesh_intelligence import assess_mesh_quality as assess

    rt.consume(request, "checks")
    mesh = await _load_mesh(file)
    return _plain(assess(mesh, target_reynolds=target_reynolds, target_yplus=target_yplus))
