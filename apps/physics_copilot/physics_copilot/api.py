"""PhysicsCopilot: a physics problem in plain language to a trained PINN and the guardrail's re-computed report.

One job runs ``pinneapple_analysis.verification.pipeline.analyze``: dimensional analysis of the given parameters, an
LLM picks and parametrises a registered preset (a hallucinated name is rejected), a PINN trains on it, and
``PhysicsGuardrail`` re-evaluates the residual on fresh points. The answer is the report, every check with its
value, not a single verdict. What it does not check: that the preset the LLM chose is the right physics for the
description (see the README). ``apps/verified_physics`` runs the same pipeline with more options.

Run:  uvicorn physics_copilot.api:app --port 8142   (from apps/physics_copilot; needs a local Ollama server)

Environment (all optional; the rest in apps/_shared/README.md):
  COPILOT_USER / COPILOT_PASSWORD(_HASH)   login on everything except /health
  COPILOT_LLM_PROVIDER / COPILOT_LLM_MODEL  ollama (default) + llama3.2:3b; hosted providers read their key from
                                            ANTHROPIC_API_KEY / OPENAI_API_KEY
  COPILOT_MAX_EPOCHS                        largest training budget per job (default 3000)
  COPILOT_FREE_SOLVES, COPILOT_MAX_JOBS     free solves per caller and month (default 20), jobs at once (default 1)
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
PROVIDER = os.environ.get("COPILOT_LLM_PROVIDER", "ollama")
MODEL = os.environ.get("COPILOT_LLM_MODEL") or ("llama3.2:3b" if PROVIDER == "ollama" else None)
MAX_EPOCHS = int(os.environ.get("COPILOT_MAX_EPOCHS") or 3000)

app = FastAPI(title="PhysicsCopilot", version=VERSION,
              description="Plain-language physics problem to a trained PINN with a re-computed verification report.")
install(app, prefix="COPILOT")
rt = install_runtime(app, prefix="COPILOT", name="physics_copilot", units={"solves": 20}, max_jobs=1)


class SolveRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=4000)
    physical_parameters: dict[str, float] | None = None
    epochs: int = Field(2000, ge=10)
    architecture: str = "modified_mlp"
    residual_threshold: float = Field(1e-2, gt=0)


def solve_job(params: dict, ctx) -> dict:
    from pinneapple_analysis.verification.pipeline import analyze

    ctx.progress(0.05, f"drafting with {PROVIDER}:{MODEL}")
    result = analyze(params["description"], physical_parameters=params.get("physical_parameters"),
                     provider=PROVIDER, model_name=MODEL, candidate_architectures=[params["architecture"]],
                     epochs=params["epochs"], residual_threshold=params["residual_threshold"])
    p = result.provenance
    out = {"status": "rejected" if result.rejected else "done", "reason": result.rejection_reason,
           "preset": p.drafted_preset, "preset_kwargs": p.drafted_preset_kwargs, "reasoning": p.draft_reasoning,
           "dimensionless_numbers": p.dimensionless_numbers, "flow_regime": p.flow_regime,
           "architecture": p.architecture, "final_training_loss": p.final_training_loss,
           "trustworthy": p.guardrail_trustworthy, "checks": p.guardrail_checks, "summary": p.guardrail_summary,
           "not_checked": "whether the drafted preset is the right physics for the description"}
    ctx.progress(1.0, out["status"])
    return out


rt.jobs.register("solve", solve_job)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "llm": {"provider": PROVIDER, "model": MODEL}, "max_epochs": MAX_EPOCHS}


@app.post("/v1/solve")
def solve(request: Request, req: SolveRequest):
    params = req.model_dump()
    params["epochs"] = min(params["epochs"], MAX_EPOCHS)
    return rt.submit(request, "solve", params, units={"solves": 1})


@app.get("/v1/jobs/{job_id}")
def job(request: Request, job_id: str):
    return rt.job_for(request, job_id)
