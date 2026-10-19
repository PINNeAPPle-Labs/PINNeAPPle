"""The guided wizard of "New flow": problem -> physics / equations -> geometry / data -> methods -> validation ->
outputs. Each step is a choice from :data:`STEPS` (or free text); :func:`plan_from_answers` turns the answers into
a description the heuristic planner reads, and when the problem step picked a registered capability
(:mod:`pinneapple_flow.planner.extensions`) its flow is the first proposal. The result is the planner's own
(proposals with decision records), so a wizard flow is as editable as any other.

    plan_from_answers({"problem": "heat in a plate", "physics": "heat", "geometry": "2d", "methods": ["pinn", "fd"],
                       "validation": ["exact"], "outputs": ["3d"]})
"""
from __future__ import annotations

from typing import Any

STEPS: list[dict[str, Any]] = [
    {"id": "problem", "title": "Problem", "kind": "text+capability",
     "help": "what you want to know, in a sentence; or pick a domain study (aircraft, motor, ENSO, ...)"},
    {"id": "physics", "title": "Physics / equations", "kind": "choice",
     "choices": {"auto": "from the description", "heat": "heat / diffusion", "wave": "waves / vibration",
                 "poisson": "steady field with a source", "laplace": "steady, no source", "burgers": "Burgers",
                 "advection_diffusion": "advection-diffusion", "reaction_diffusion": "reaction-diffusion",
                 "domain": "the domain study's own model"}},
    {"id": "geometry", "title": "Geometry / data", "kind": "choice",
     "choices": {"auto": "from the description", "1d": "1-D (rod, wire)", "2d": "2-D (plate)", "3d": "3-D (block)",
                 "measurements": "from measurements", "stream": "a live sensor stream"}},
    {"id": "methods", "title": "Methods", "kind": "multi",
     "choices": {"classical": "classical solver (finite differences / elements)", "pinn": "PINN",
                 "operator": "neural operator (FNO / DeepONet)", "surrogate": "fast surrogate",
                 "twin": "digital twin"}},
    {"id": "validation", "title": "Validation", "kind": "multi",
     "choices": {"exact": "against an exact / reference solution", "compare": "compare the methods",
                 "uncertainty": "uncertainty / sensitivity", "example": "same numbers as the example"}},
    {"id": "outputs", "title": "Outputs", "kind": "multi",
     "choices": {"dashboard": "dashboard / report", "3d": "3-D view", "blender": "Blender render"}},
]

_PHRASE = {
    "physics": {"heat": "transient heat conduction", "wave": "wave propagation", "poisson": "Poisson equation with a "
                "source", "laplace": "steady Laplace equation", "burgers": "Burgers equation",
                "advection_diffusion": "advection-diffusion transport", "reaction_diffusion": "reaction-diffusion"},
    "geometry": {"1d": "in a 1D rod", "2d": "in a 2D plate", "3d": "in a 3D block", "measurements":
                 "from measurements", "stream": "digital twin on a live sensor stream"},
    "methods": {"classical": "finite differences", "pinn": "a PINN", "operator": "a neural operator (FNO)",
                "surrogate": "a fast surrogate", "twin": "a digital twin"},
    "validation": {"exact": "against the exact solution", "compare": "compare", "uncertainty": "with uncertainty "
                   "and sensitivity"},
    "outputs": {"dashboard": "with a report", "3d": "show it in 3D", "blender": "render with Blender"},
}


def describe(answers: dict[str, Any]) -> str:
    """The description the planner reads for these answers."""
    parts = [str(answers.get("problem") or "").strip().rstrip(".")]
    for step in ("physics", "geometry"):
        v = answers.get(step)
        if v and v in _PHRASE[step]:
            parts.append(_PHRASE[step][v])
    methods = [_PHRASE["methods"][m] for m in answers.get("methods") or [] if m in _PHRASE["methods"]]
    val = [v for v in answers.get("validation") or [] if v in _PHRASE["validation"]]
    if len(methods) > 1 or "compare" in val:
        parts.append("compare " + " with ".join(methods) if methods else "compare the methods")
    elif methods:
        parts.append("solve with " + methods[0])
    parts += [_PHRASE["validation"][v] for v in val if v != "compare"]
    parts += [_PHRASE["outputs"][o] for o in answers.get("outputs") or [] if o in _PHRASE["outputs"]]
    return ". ".join(p for p in parts if p) or "heat in a 2D plate"


def plan_from_answers(answers: dict[str, Any], *, max_proposals: int = 3) -> dict[str, Any]:
    """The planner's result for the wizard's answers (``answers["capability"]``: a capability id to put first)."""
    from . import extensions as X
    from .heuristic import detect, plan_heuristic
    text = describe(answers)
    res = plan_heuristic(text, max_proposals=max_proposals)
    cap_id = answers.get("capability")
    if cap_id:
        caps = {c.id: c for c in X.capabilities()}
        if cap_id not in caps:
            raise ValueError(f"unknown capability {cap_id!r}")
        cap = caps[cap_id]
        s, hits = X.score(cap, text)
        p = X.proposal(cap, text, detect(text), hits or ["chosen in the wizard"], s)
        res["proposals"] = [p] + [q for q in res["proposals"] if q["flow"].get("meta", {}).get("capability") != cap_id]
        res["proposals"] = res["proposals"][:max_proposals]
    res["wizard"] = {"answers": answers, "description": text}
    return res


__all__ = ["STEPS", "describe", "plan_from_answers"]
