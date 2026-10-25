"""Text2Part: describe a part in plain language, get a mesh (STL) or a parametric CAD file (STEP) back.

The LLM never writes CAD code: ``pinneapple_llm.draft_part`` asks it for a recipe from a fixed menu (primitives
combined by boolean CSG, or a registered CadQuery template), every name is checked against the real registry, and a
refused draft is retried. The part is then built and measured (``pinneapple_llm.measure_part``: volume, extents,
watertightness) so the caller can compare it with what was asked for; the dimensions are not checked against the
description.

Run:  uvicorn text2part.api:app --port 8140   (from apps/text2part; needs a local Ollama server)

Environment (all optional; the rest in apps/_shared/README.md):
  T2P_USER / T2P_PASSWORD(_HASH)   login on everything except /health
  T2P_LLM_PROVIDER                 ollama (default, local server at 127.0.0.1:11434), anthropic or openai
  T2P_LLM_MODEL                    model name (default llama3.2:3b for ollama); hosted providers read their key
                                   from ANTHROPIC_API_KEY / OPENAI_API_KEY, never from a request
  T2P_FREE_GENERATIONS             free generations per caller and month (default 20)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import pinneapple_llm as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
PROVIDER = os.environ.get("T2P_LLM_PROVIDER", "ollama")
MODEL = os.environ.get("T2P_LLM_MODEL") or ("llama3.2:3b" if PROVIDER == "ollama" else None)

app = FastAPI(title="Text2Part", version=VERSION,
              description="Plain-language part description to a checked CAD recipe, built and measured.")
install(app, prefix="T2P")
rt = install_runtime(app, prefix="T2P", name="text2part", units={"generations": 20})


class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=2000)
    tier: Literal["quick", "engineering"] = "quick"      # quick: CSG mesh recipe; engineering: CadQuery template
    fmt: Literal["stl", "step"] = "stl"
    max_attempts: int = Field(4, ge=1, le=8)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    from pinneapple_design.geometry.gen.cadquery_gen import cadquery_available
    from pinneapple_design.geometry.gen.primitives import _boolean_engines_available

    return {"version": VERSION, "llm": {"provider": PROVIDER, "model": MODEL}, "tiers": ["quick", "engineering"],
            "formats": ["stl", "step"], "cadquery": bool(cadquery_available()),
            "boolean_engines": list(_boolean_engines_available()),
            "limits": ["the recipe is checked, the dimensions are not: compare 'measurements' with your request",
                       "without a boolean engine (manifold3d) cuts and unions are concatenations",
                       "STEP needs cadquery; a STEP of a quick-tier recipe is a triangulated shell, not a B-rep"]}


@app.post("/v1/generate")
def generate(request: Request, req: GenerateRequest):
    rt.consume(request, "generations")
    try:
        result, attempts = pl.draft_part(req.description, tier=req.tier, provider=PROVIDER, model=MODEL,
                                         max_attempts=req.max_attempts)
    except ConnectionError as e:
        raise HTTPException(503, f"the LLM is not reachable ({PROVIDER}): {e}") from None
    except ValueError as e:
        raise HTTPException(502, str(e)) from None
    measurements = pl.measure_part(result) if result.kind == "mesh_recipe" else None
    out = os.path.join(tempfile.mkdtemp(prefix="text2part_"), f"part.{req.fmt}")
    try:
        pl.export_recipe(result, out, fmt=req.fmt)
    except ImportError as e:
        raise HTTPException(422, f"{req.fmt} output needs cadquery on this server: {e}") from None
    recipe = result.recipe if result.kind == "mesh_recipe" else {"template": result.name, "params": result.params}
    # JSON-encoded (ASCII) header values: an LLM's reasoning can hold any character
    headers = {"X-Attempts-Used": str(attempts), "X-Reasoning": json.dumps(" ".join(result.reasoning.split())[:500]),
               "X-Recipe": json.dumps(recipe)[:4000]}
    if measurements is not None:
        headers["X-Measurements"] = json.dumps(measurements)
    return FileResponse(out, filename=f"part.{req.fmt}", media_type="application/octet-stream", headers=headers)
