"""Tests for pinneapple_veriphysics.recommend.formulate_and_recommend --
the "Solution Recommendation" step factored out of pipeline.analyze() so
it can run standalone (no training). Gated on a real local Ollama
server, matching the rest of this repo's convention (see
test_orchestrator_pipeline.py's identical pattern).
"""
from __future__ import annotations

import pytest

from pinneapple_veriphysics.recommend import formulate_and_recommend, RecommendationResult

_OLLAMA_MODEL = "llama3.2:3b"


def _ollama_reachable() -> bool:
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        have = {m["name"] for m in r.json().get("models", [])}
        return _OLLAMA_MODEL in have or f"{_OLLAMA_MODEL}:latest" in have
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama server with llama3.2:3b reachable")
def test_formulate_and_recommend_burgers():
    rec = formulate_and_recommend(
        "1D viscous Burgers equation with nu=0.01",
        physical_parameters={"velocity": 1.0, "length": 1.0, "kinematic_viscosity": 0.01},
        provider="ollama", model_name=_OLLAMA_MODEL,
    )
    assert isinstance(rec, RecommendationResult)
    assert not rec.rejected, rec.rejection_reason
    assert rec.drafted_preset is not None
    assert rec.dimensionless_numbers["reynolds"] == pytest.approx(100.0)
    assert rec.recommended_solver_family is not None
    assert len(rec.recommended_architectures) >= 1
    # Every recommended architecture must be a real registry_key (not a
    # bare ARCHITECTURE_CATALOG category like "pinn").
    for arch in rec.recommended_architectures:
        assert arch in ("vanilla_pinn", "xpinn", "fno3d", "deeponet", "mesh_graph_net", "pino", "inverse_pinn")

    # The Solution Recommendation screen's core ask: a real pros/cons
    # comparison across every candidate, not just the winner's name.
    assert rec.architecture_details, "must expose at least the recommended architecture's real detail"
    assert any(d.is_recommended for d in rec.architecture_details)
    recommended_arch_detail = next(d for d in rec.architecture_details if d.is_recommended)
    assert recommended_arch_detail.strengths, "recommended architecture must carry its real cited strengths"
    assert recommended_arch_detail.weaknesses, "recommended architecture must carry its real cited weaknesses"
    for d in rec.architecture_details:
        assert d.source, f"{d.key} detail must cite a real source"

    if rec.tool_details:
        for d in rec.tool_details:
            assert d.license in ("open-source", "commercial")
            assert d.source, f"{d.key} tool detail must cite a real source"

    assert rec.recommended_pipeline, "must synthesize a real end-to-end pipeline string"
    assert rec.recommended_architectures[0] in rec.recommended_pipeline


# (The test that checks `formulate_and_recommend` and the orchestrator's `analyze()` agree lives with the orchestrator, in Veriphysics.)
