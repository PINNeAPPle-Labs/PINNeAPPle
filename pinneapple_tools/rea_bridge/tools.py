"""Expose the REA bridge as tools of :class:`PhysicsToolRegistry`, so
:func:`pinneapple_llm.agent_loop.run_agent_loop` can study a reference
implementation the same way it calls a solver: one real tool call, a real
result fed back to the next turn.

The tools are registered only when they can actually run: the license tool
always (pure Python), the REA tools only when an REA executable resolves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["register_rea_tools"]


def _license_policy(path: str) -> dict[str, Any]:
    from .licenses import detect_licenses, policy_for
    ids = detect_licenses(path)
    pol = policy_for(ids)
    return {"licenses": list(pol.licenses), "policy": pol.policy,
            "attribution_required": pol.attribution_required, "reason": pol.reason}


def _analyze_js(path: str, save_evidence_to: str = "") -> dict[str, Any]:
    from .client import ReaClient
    from .evidence import summarize_js_evidence
    res = ReaClient().analyze_javascript_application(path)
    if save_evidence_to:
        Path(save_evidence_to).write_text(json.dumps(res.payload))
    return summarize_js_evidence(res.payload).to_dict()


def _locate_js(path: str, keywords: list[str], max_hits: int = 10) -> list[dict[str, Any]]:
    from .client import ReaClient
    from .evidence import locate, summarize_js_evidence
    fmap = summarize_js_evidence(ReaClient().analyze_javascript_application(path).payload)
    return [{"function": h.function.label, "matched": h.matched, "snippet": h.snippet,
             "callees": h.callees, "callers": h.callers}
            for h in locate(fmap, keywords, max_hits=max_hits)]


def register_rea_tools(registry: Any) -> list[str]:
    """Register the bridge's tools on ``registry``; returns the names added."""
    from pinneapple_worldmodel.physics_tools import PhysicsTool
    from .client import ReaUnavailable, resolve_rea_command

    added = []

    def reg(tool: PhysicsTool) -> None:
        if tool.name not in getattr(registry, "_tools", {}):
            registry.register(tool)
            added.append(tool.name)

    reg(PhysicsTool(
        name="reference_license_policy",
        category="research",
        description="Detect a local project's license and the reuse policy PINNeAPPle applies "
                    "to it (adapt / clean_room / reference_only)",
        input_schema={"path": "str"},
        output_schema={"licenses": "list[str]", "policy": "str", "reason": "str"},
        module_path="pinneapple_tools.rea_bridge.licenses",
        fn=_license_policy,
        tags=["rea", "license", "provenance"],
    ))
    try:
        resolve_rea_command()
    except ReaUnavailable:
        return added
    reg(PhysicsTool(
        name="rea_analyze_javascript",
        category="research",
        description="Statically reconstruct a local JavaScript/Electron app with REA (nothing is "
                    "executed) and summarise its functions, call graph, timers and storage",
        input_schema={"path": "str", "save_evidence_to": "str (optional)"},
        output_schema={"functions": "list[str]", "call_edges": "int", "limitations": "list[str]"},
        module_path="pinneapple_tools.rea_bridge.evidence",
        fn=_analyze_js,
        tags=["rea", "reverse_engineering", "javascript"],
    ))
    reg(PhysicsTool(
        name="rea_locate_feature",
        category="research",
        description="Find where a feature is implemented in a local JavaScript app: functions "
                    "whose name/body match the keywords, with snippet, callers and callees",
        input_schema={"path": "str", "keywords": "list[str]", "max_hits": "int"},
        output_schema={"hits": "list[dict]"},
        module_path="pinneapple_tools.rea_bridge.evidence",
        fn=_locate_js,
        tags=["rea", "reverse_engineering", "feature"],
    ))
    return added
