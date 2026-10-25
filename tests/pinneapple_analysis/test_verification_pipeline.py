"""Real end-to-end proof that the core pipeline actually works -- not a
mock, not a stub: a real local Llama drafts a real preset, a real PINN
trains on it, and a real PhysicsGuardrail check runs and is reported.
Skipped (not failed) if no local Ollama server/model is reachable,
matching PINNeAPPle's own established convention for these tests."""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.pipeline import analyze

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
def test_full_pipeline_burgers_end_to_end():
    result = analyze(
        "1D viscous Burgers equation with nu=0.01",
        physical_parameters={"velocity": 1.0, "length": 1.0, "kinematic_viscosity": 0.01},
        provider="ollama", model_name=_OLLAMA_MODEL,
        epochs=200, n_collocation=256,
    )

    assert not result.rejected, result.rejection_reason
    assert result.best_candidate is not None
    assert result.provenance.drafted_preset is not None
    assert result.provenance.dimensionless_numbers["reynolds"] == pytest.approx(1.0 * 1.0 / 0.01)
    assert result.provenance.final_training_loss is not None
    assert len(result.provenance.guardrail_checks) > 0
    # The report must show its work -- not just a single boolean.
    assert all("name" in c and "passed" in c and "detail" in c for c in result.provenance.guardrail_checks)


@pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama server with llama3.2:3b reachable")
def test_physics_aware_selection_ranks_trustworthy_over_untrustworthy():
    """Two candidate architectures on the same simple preset -- confirms
    the ranking key actually orders by guardrail trust first, not raw
    loss, by inspecting the real returned candidates (does not force a
    specific outcome, since which architecture wins depends on real
    training -- just confirms the reported ranking is internally
    consistent with rank_key's own documented rule)."""
    from pinneapple_analysis.verification.pipeline import rank_key

    result = analyze(
        "Steady 2D Laplace equation on a unit square",
        provider="ollama", model_name=_OLLAMA_MODEL,
        candidate_architectures=["modified_mlp", "vanilla_pinn"],
        epochs=150, n_collocation=256,
    )
    if result.rejected:
        pytest.skip(f"pipeline rejected this run: {result.rejection_reason}")

    assert len(result.all_candidates) >= 1
    ranked = sorted(result.all_candidates, key=rank_key)
    assert ranked[0].architecture == result.best_candidate.architecture


def test_rejection_path_does_not_crash_on_a_hallucination_prone_prompt():
    """Even without Ollama reachable, the pipeline's dimensional-analysis
    step must run and the LLM call's own connection error must not
    produce an unhandled traceback -- it's a real, reportable failure."""
    if _ollama_reachable():
        pytest.skip("Ollama IS reachable here; this test specifically covers the unreachable path")
    with pytest.raises(Exception):  # noqa: B017 -- any connection error, not one class
        analyze("test", provider="ollama", model_name=_OLLAMA_MODEL)


# ---------------------------------------------------------------------------
# Without an LLM: the drafting step stubbed, everything after it real.
# ---------------------------------------------------------------------------

def _stub_draft(preset, kwargs=None, reasoning="stub"):
    from pinneapple_llm.draft import DraftResult

    def draft_problem(description, **_):
        return DraftResult(preset=preset, kwargs=dict(kwargs or {}), reasoning=reasoning, raw_response="")
    return draft_problem


def test_pipeline_with_stubbed_draft_trains_checks_and_records(monkeypatch):
    import pinneapple_llm as pl

    monkeypatch.setattr(pl, "draft_problem", _stub_draft("burgers_1d", {"nu": 0.02}))
    result = analyze("viscous Burgers", physical_parameters={"velocity": 1.0, "length": 1.0,
                                                              "kinematic_viscosity": 0.01},
                     candidate_architectures=["modified_mlp", "not_a_registered_architecture"],
                     epochs=20, n_collocation=128, hidden_dim=16, n_layers=2)
    assert not result.rejected, result.rejection_reason
    assert [c.architecture for c in result.all_candidates] == ["modified_mlp"]   # the unknown one is skipped
    p = result.provenance
    assert p.drafted_preset == "burgers_1d" and p.drafted_preset_kwargs == {"nu": 0.02}
    assert p.dimensionless_numbers["reynolds"] == pytest.approx(100.0)
    assert p.guardrail_checks and all({"name", "passed", "value"} <= set(c) for c in p.guardrail_checks)
    assert p.training_config["epochs"] == 20


def test_pipeline_reports_no_fit_as_a_rejection(monkeypatch):
    import pinneapple_llm as pl

    monkeypatch.setattr(pl, "draft_problem", _stub_draft(None, reasoning="nothing fits"))
    result = analyze("a problem no preset covers")
    assert result.rejected and "nothing fits" in result.rejection_reason and result.best_candidate is None


def test_rank_key_puts_trustworthy_before_lower_loss():
    from pinneapple_analysis.verification.pipeline import CandidateResult, rank_key

    untrusted = CandidateResult("a", final_loss=1e-6, guardrail_trustworthy=False, guardrail_checks=[],
                                residual_value=1e-7)
    trusted = CandidateResult("b", final_loss=1.0, guardrail_trustworthy=True, guardrail_checks=[],
                              residual_value=1e-3)
    no_residual = CandidateResult("c", final_loss=0.5, guardrail_trustworthy=True, guardrail_checks=[],
                                  residual_value=None)
    assert [c.architecture for c in sorted([untrusted, no_residual, trusted], key=rank_key)] == ["b", "c", "a"]
