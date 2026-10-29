"""The planner's public extension hook: another package (the example-domain nodes of
:mod:`pinneapple_flow.nodes_ext`, PINNeAPPle-CFD, a user's own nodes) declares *capabilities* and the heuristic
planner proposes them for a matching description, with a decision record per choice.

A :class:`Capability` names what it does (``title``, ``domain``), how to recognise it in a description
(``keywords``: phrase -> weight, matched at word starts like the planner's own keywords; ``min_score``), what it needs
(``inputs``: the things a user must supply, shown in the proposal), which node types it uses (``nodes``) and which
usually follow (``downstream``), the references that back it (``references``: keys of
:mod:`pinneapple_flow.references`, or ``uncited`` saying why there is none; the decision is then recorded as
"no reference", with a lower confidence) and how to build its flow: ``template`` (an example flow,
``<domain dir>/<stem>`` under ``pinneapple_flow/flows/examples``, optionally in its quick configuration) and / or
``build(text, detected) -> FlowBuilder`` for flows that depend on the text. ``adapt(doc, text, detected)`` may edit
the template's document (for instance keep only the aircraft the text names)::

    from pinneapple_flow.planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="cfd.external_aero", title="External aerodynamics with OpenFOAM", domain="CFD",
        keywords={"drag": 2, "external flow": 3, "ahmed": 3}, nodes=("cfd.external_flow",),
        inputs=("a body geometry",), references=("menter1994sst",), template="cfd/ahmed_body"))

Registering a capability also registers its decision rule (``capability.<id>``) through
:func:`pinneapple_flow.rules.register_rule`. :func:`match` scores a text against every capability;
:func:`proposals` builds the flows of the best matches; the heuristic planner puts them first.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .build import FlowBuilder


@dataclass
class Capability:
    id: str
    title: str
    domain: str
    keywords: dict[str, float]
    nodes: tuple[str, ...]
    description: str = ""
    inputs: tuple[str, ...] = ()
    downstream: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    uncited: str = ""
    template: str = ""
    quick: bool = False
    min_score: float = 2.0
    requires_all: tuple[tuple[str, ...], ...] = ()   # every group needs one hit (e.g. a body and an analysis)
    build: Callable[[str, dict[str, Any]], FlowBuilder] | None = None
    adapt: Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any]] | None = None
    example: str = ""                                # the examples/* folder it reproduces (if any)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rule_id(self) -> str:
        return f"capability.{self.id}"


CAPABILITIES: dict[str, Capability] = {}


def register_capability(cap: Capability) -> Capability:
    """Add (or replace, same id) a capability and its decision rule."""
    from ..rules import RULES, Rule, register_rule
    if not re.match(r"^[a-z][a-z0-9_.]{1,60}$", cap.id):
        raise ValueError(f"capability id {cap.id!r}: lower-case letters, digits, '.' and '_'")
    if not cap.keywords or not cap.nodes:
        raise ValueError(f"capability {cap.id!r} needs keywords and nodes")
    if not (cap.template or cap.build):
        raise ValueError(f"capability {cap.id!r} needs a template flow or a build function")
    rule = Rule(cap.rule_id, f"{cap.title}: proposed when the description matches its keywords",
                cites=tuple(cap.references), uncited="" if cap.references else
                (cap.uncited or "no literature reference registered for this capability"), topic="capability")
    if cap.rule_id in RULES and RULES[cap.rule_id] != rule:
        RULES.pop(cap.rule_id)                     # a re-registration with new references replaces the rule
    register_rule(rule)
    CAPABILITIES[cap.id] = cap
    return cap


def capabilities() -> list[Capability]:
    _load()
    return [CAPABILITIES[k] for k in sorted(CAPABILITIES)]


def _load() -> None:
    from ..registry import REGISTRY
    REGISTRY.types()                               # importing the node catalog registers the built-in capabilities


def _hit(t: str, kw: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(kw.lower()), t) is not None


def score(cap: Capability, text: str) -> tuple[float, list[str]]:
    t = text.lower()
    hits = [k for k in cap.keywords if _hit(t, k)]
    if any(not any(_hit(t, k) for k in group) for group in cap.requires_all):
        return 0.0, hits
    return float(sum(cap.keywords[k] for k in hits)), hits


def match(text: str) -> list[tuple[float, Capability, list[str]]]:
    """Every capability whose score reaches its ``min_score``, best first."""
    out = []
    for cap in capabilities():
        s, hits = score(cap, text)
        if s >= cap.min_score:
            out.append((s, cap, hits))
    return sorted(out, key=lambda r: (-r[0], r[1].id))


def template_doc(name: str, quick: bool = False) -> dict[str, Any]:
    from ..examples import flow_path, with_quick
    doc = json.loads(flow_path(name).read_text())
    return with_quick(doc) if quick else doc


def _from_template(cap: Capability, text: str, det: dict[str, Any]) -> FlowBuilder:
    doc = template_doc(cap.template, cap.quick)
    if cap.adapt is not None:
        doc = cap.adapt(copy.deepcopy(doc), text, det)
    fb = FlowBuilder(doc.get("name", cap.title), doc.get("description", ""))
    for n in doc["nodes"]:
        fb.add(n["id"], n["type"], n.get("params") or {}, label=n.get("label", ""))
    keep = set(fb.nodes)
    for e in doc["edges"]:
        if e["source"] in keep and e["target"] in keep:
            fb.connect(e["source"], e["source_port"], e["target"], e["target_port"])
    fb.meta = {k: v for k, v in (doc.get("meta") or {}).items() if k in ("example", "domain", "scripts", "quick")}
    return fb


def proposal(cap: Capability, text: str, det: dict[str, Any], hits: list[str], s: float) -> dict[str, Any]:
    fb = cap.build(text, det) if cap.build else _from_template(cap, text, det)
    first = next(iter(fb.nodes), "")
    fb.decide(cap.rule_id, f"{cap.title} ({cap.domain})", node=first, value=cap.id,
              rationale=f"keywords {', '.join(hits)} (score {s:g} >= {cap.min_score:g})"
              + (f"; from the example flow {cap.template}" if cap.template else ""))
    if cap.inputs:
        fb.decide(cap.rule_id, "inputs to supply: " + "; ".join(cap.inputs), node=first, value=list(cap.inputs))
    meta = {"family": cap.domain, "capability": cap.id, "nodes_used": list(cap.nodes),
            "downstream": list(cap.downstream), "inputs": list(cap.inputs), **getattr(fb, "meta", {})}
    why = cap.description or f"{cap.title}: matched {', '.join(hits)}."
    return fb.proposal(cap.title, why, **meta)


def proposals(text: str, det: dict[str, Any] | None = None, *, limit: int = 2) -> list[dict[str, Any]]:
    """The flows of the best-matching capabilities (at most ``limit``)."""
    det = det or {}
    return [proposal(cap, text, det, hits, s) for s, cap, hits in match(text)[:limit]]


def describe() -> list[dict[str, Any]]:
    """Every capability as JSON (for the GUI's guided wizard and the LLM planner's prompt)."""
    return [{"id": c.id, "title": c.title, "domain": c.domain, "description": c.description,
             "keywords": sorted(c.keywords), "nodes": list(c.nodes), "inputs": list(c.inputs),
             "downstream": list(c.downstream), "references": list(c.references), "uncited": c.uncited,
             "template": c.template, "example": c.example} for c in capabilities()]


__all__ = ["Capability", "CAPABILITIES", "register_capability", "capabilities", "match", "proposals", "describe",
           "score", "template_doc"]
