"""Planner: a problem description -> one or more proposals, each a flow of catalog nodes.

    from pinneapple_flow.planner import plan
    res = plan("transient heat in a 2D plate with fixed edges")             # heuristic, offline
    res = plan(text, backend="llm", provider="ollama", model="llama3.1")   # LLM, validated, falls back on failure
    res["planner"], res["proposals"][0]["flow"]

Every proposal says which planner produced it (``heuristic``, ``llm:<provider>/<model>``, ``+repair`` when the
LLM output needed repairs) and the result keeps what the heuristics detected in the text.
"""
from __future__ import annotations

from typing import Any

from .heuristic import detect, plan_heuristic
from .llm import plan_llm, repair

__all__ = ["plan", "detect", "plan_heuristic", "plan_llm", "repair"]


def plan(text: str, *, backend: str = "auto", provider: str = "", model: str = "", base_url: str = "",
         api_key: str = "", n: int = 3, call=None) -> dict[str, Any]:
    """``backend``: ``heuristic``, ``llm`` or ``auto`` (LLM when a provider is configured, else heuristic). The LLM
    path validates every proposal; when it fails or nothing valid comes back, the heuristic plan is returned with
    the reason in ``warnings`` and ``fallback_from``."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("describe the problem in a sentence or more")
    text = text.strip()[:6000]
    if backend == "auto":
        from ..config import Settings
        s = Settings.from_env()
        if s.llm_provider not in ("", "none"):
            backend, provider = "llm", provider or s.llm_provider
            model, base_url, api_key = model or s.llm_model, base_url or s.llm_base_url, api_key or s.llm_api_key
        else:
            backend = "heuristic"
    heur = plan_heuristic(text, max_proposals=n)
    for p in heur["proposals"]:
        p["planner"] = "heuristic"
        p["flow"]["meta"]["planner"] = "heuristic"
        p["flow"]["meta"]["prompt"] = text
    if backend == "heuristic":
        return heur
    if backend != "llm":
        raise ValueError(f"unknown planner backend {backend!r}")
    try:
        res = plan_llm(text, provider=provider, model=model, base_url=base_url, api_key=api_key, n=n, call=call)
    except Exception as exc:  # noqa: BLE001 - any provider or parsing failure falls back to the heuristic plan
        heur["warnings"].insert(0, f"LLM planner failed ({type(exc).__name__}: {str(exc)[:300]}); heuristic plan "
                                   f"shown instead")
        heur["fallback_from"] = f"llm:{provider}/{model or 'default'}"
        return heur
    for p in res["proposals"]:
        p["flow"]["meta"]["prompt"] = text
    res["detected"] = {"llm": res.get("detected"), "heuristic": heur["detected"]}
    return res
