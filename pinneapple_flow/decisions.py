"""Decision records: what a flow decided, who decided it, why, on which references and evidence, how confident.

Every automatic choice (identified equations, dimension, conditions and coefficients, method, architecture,
hyperparameters and HPO ranges, twin design, validation thresholds, the planner's alternative routes) and every
user edit becomes a :class:`Decision`. A decision with neither a reference nor library evidence is marked
``no reference``, gets a lower confidence and says how to raise it.

    d = rule_decision("method.fd", node="fd", what="solve with finite differences", value={"method": "fd"})
    d.confidence, d.label, [r["key"] for r in d.references]     # 0.85, 'high', ['leveque2007fd', 'hairer1996odes2']
    u = user_decision("eq", "coefficients", "alpha = 2e-5", value={"alpha": 2e-5}, notes="handbook value")

Records travel as JSON in ``flow.meta["decisions"]`` (planner / user) and in each node's report
(``report["decisions"]``, decisions taken while running); the store keeps them per flow version and run.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import references as refs
from .rules import rule

TOPICS = ("equations", "dimension", "bcs", "coefficients", "method", "architecture", "hyperparameters", "hpo",
          "twin", "validation", "route", "data", "other")
RAISE_HINT = ("no reference and no evidence in the library: connect a validation node (validate.compare against an "
              "exact solution, a reference solver or measurements) or run a benchmark to raise the confidence")

# node types that would back a weak decision with evidence, by topic (all registered; checked by the tests). The
# GUI offers them as "add node" buttons that insert them wired into the flow (pinneapple_flow.lab_status).
RAISE_NODES: dict[str, tuple[str, ...]] = {
    "equations": ("validate.compare", "inverse.discover"),
    "dimension": ("validate.compare",),
    "bcs": ("validate.compare",),
    "coefficients": ("inverse.calibrate", "validate.compare"),
    "method": ("validate.compare", "lib.compare.preset"),
    "architecture": ("surrogate.hpo", "validate.compare"),
    "hyperparameters": ("surrogate.hpo",),
    "hpo": ("validate.compare", "uq.sensitivity"),
    "twin": ("validate.compare",),
    "validation": ("validate.lab",),
    "route": ("validate.compare",),
    "data": ("data.observe",),
    "other": ("validate.compare",),
}


@dataclass
class Decision:
    topic: str
    what: str
    decided_by: dict[str, str]                       # {"kind": "rule" | "llm" | "user", "id": ...}
    rationale: str = ""
    value: Any = None
    references: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)
    node: str = ""
    notes: str = ""
    uncited_reason: str = ""
    confidence: float = 0.0
    label: str = ""                                  # high | medium | low
    no_reference: bool = False
    suggestion: str = ""
    raise_with: list[str] = field(default_factory=list)      # node types that would raise a low confidence
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.topic not in TOPICS:
            self.topic = "other"
        score(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Decision:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def score(d: Decision) -> Decision:
    """Confidence from who decided, the references (registry > verified > unverified) and the evidence. With
    neither a usable reference nor evidence the decision is ``no reference``: confidence at most 0.35."""
    kind = d.decided_by.get("kind")
    good = [r for r in d.references if r.get("status") in ("registry", "verified")]
    base = {"rule": 0.6, "llm": 0.45, "user": 0.7}.get(kind, 0.4)
    c = base + (0.15 if good else 0.0) + (0.1 if d.evidence else 0.0)
    if kind == "llm" and d.references and not good:
        c -= 0.1                                     # it cited something we could not check
    d.no_reference = not good and not d.evidence
    if d.no_reference:
        c = min(c, 0.35 if kind != "user" else 0.5)
        d.suggestion = RAISE_HINT
    d.confidence = round(max(0.05, min(c, 0.95)), 2)
    d.label = "high" if d.confidence >= 0.75 else "medium" if d.confidence >= 0.5 else "low"
    d.raise_with = list(RAISE_NODES.get(d.topic, ())) if d.no_reference or d.label == "low" else []
    return d


def _evidence(rid: str) -> list[dict[str, str]]:
    return [{"kind": k, "ref": p, "detail": det} for k, p, det in rule(rid).evidence]


def rule_decision(rule_id: str, *, what: str, node: str = "", value: Any = None, rationale: str = "",
                  topic: str | None = None, extra: tuple[str, ...] = ()) -> Decision:
    """A decision taken by a heuristic rule: it carries the rule's references and evidence."""
    r = rule(rule_id)
    cited = refs.cite(*r.cites, *extra)
    return Decision(topic=topic or r.topic, what=what, decided_by={"kind": "rule", "id": rule_id},
                    rationale=rationale or r.description, value=value, references=cited,
                    evidence=_evidence(rule_id), node=node, uncited_reason=r.uncited if not cited else "")


def user_decision(node: str, topic: str, what: str, value: Any = None, *, notes: str = "",
                  references: list[dict[str, Any] | str] = (), user: str = "user") -> Decision:
    """A user's edit, with optional notes and references (checked against the registry, or their DOI / URL)."""
    return Decision(topic=topic, what=what, decided_by={"kind": "user", "id": user}, rationale=notes or "user edit",
                    value=value, references=[refs.check(r) for r in references], node=node, notes=notes)


def llm_decision(provider_model: str, *, topic: str, what: str, node: str = "", value: Any = None,
                 rationale: str = "", references: list[dict[str, Any] | str] = ()) -> Decision:
    """A decision proposed by an LLM: each reference is a registry key or is checked by DOI / URL, else
    ``unverified``; nothing it cites is trusted on its word."""
    return Decision(topic=topic, what=what, decided_by={"kind": "llm", "id": provider_model}, rationale=rationale,
                    value=value, references=[refs.check(r) for r in references], node=node)


def edits_to_decisions(edits: list[dict[str, Any]]) -> list[Decision]:
    """``flow.meta["user_edits"]`` entries (``{node, param, value, notes?, references?}``) as user decisions."""
    topic_of = {"pdes": "equations", "conditions": "bcs", "coefficients": "coefficients", "unknowns": "equations",
                "dimension": "dimension", "method": "method", "architecture": "architecture", "kind": "architecture",
                "space": "hpo", "tolerance": "validation", "reference": "validation"}
    out = []
    for e in edits or []:
        if not isinstance(e, dict) or not e.get("node") or not e.get("param"):
            continue
        p = str(e["param"])
        d = user_decision(str(e["node"]), topic_of.get(p, "other"), f"{p} set by the user",
                          value=e.get("value"), notes=str(e.get("notes") or ""), references=e.get("references") or [])
        d.id = f"edit:{e['node']}:{p}"
        out.append(d)
    return out


def collect(flow, reports: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every decision of a flow: planner / LLM decisions in ``meta["decisions"]``, the user's edits, and the
    decisions nodes took while running (``reports[node]["decisions"]``)."""
    meta = getattr(flow, "meta", None) or (flow.get("meta") if isinstance(flow, dict) else {}) or {}
    out = [Decision.from_dict(d).to_dict() for d in meta.get("decisions") or [] if isinstance(d, dict)]
    out += [d.to_dict() for d in edits_to_decisions(meta.get("user_edits") or [])]
    for nid, rep in (reports or {}).items():
        for d in (rep or {}).get("decisions") or []:
            out.append(Decision.from_dict({**d, "node": d.get("node") or nid}).to_dict())
    return out


def by_node(decisions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for d in decisions:
        out.setdefault(d.get("node") or "", []).append(d)
    return out


def summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(decisions)
    return {"decisions": n, "no_reference": sum(1 for d in decisions if d.get("no_reference")),
            "by_label": {lab: sum(1 for d in decisions if d.get("label") == lab) for lab in ("high", "medium", "low")},
            "references": sorted({r.get("key") or r.get("link") or r.get("title", "")
                                  for d in decisions for r in d.get("references", [])} - {""})}


def report_markdown(decisions: list[dict[str, Any]], title: str = "Decisions and references") -> str:
    """The flow-level decisions and references section of a report."""
    lines = [f"## {title}", ""]
    if not decisions:
        return "\n".join(lines + ["No decision recorded."])
    s = summary(decisions)
    lines.append(f"{s['decisions']} decisions; {s['no_reference']} without a reference "
                 f"(high {s['by_label']['high']}, medium {s['by_label']['medium']}, low {s['by_label']['low']}).")
    lines.append("")
    for d in decisions:
        who = d["decided_by"].get("kind") + ":" + d["decided_by"].get("id", "")
        flag = " **[no reference]**" if d.get("no_reference") else ""
        lines.append(f"- **{d.get('node') or 'flow'}** / {d['topic']}: {d['what']} ({who}; confidence "
                     f"{d['confidence']} {d['label']}){flag}")
        if d.get("rationale"):
            lines.append(f"  - why: {d['rationale']}")
        for r in d.get("references", []):
            lines.append(f"  - ref: {refs.format_reference(r)}")
        for e in d.get("evidence", []):
            lines.append(f"  - evidence ({e['kind']}): {e['ref']} - {e['detail']}")
        if d.get("uncited_reason"):
            lines.append(f"  - no literature reference: {d['uncited_reason']}")
        if d.get("suggestion"):
            lines.append(f"  - to raise it: {d['suggestion']}"
                         + (f" (nodes: {', '.join(d['raise_with'])})" if d.get("raise_with") else ""))

    used = {}
    for d in decisions:
        for r in d.get("references", []):
            used[r.get("key") or r.get("link") or r.get("title")] = r
    if used:
        lines += ["", "### References", ""]
        lines += [f"- {refs.format_reference(r)}" for r in used.values()]
    return "\n".join(lines)


def report_html(decisions: list[dict[str, Any]], title: str = "Decisions and references") -> str:
    """The same section as :func:`report_markdown`, as HTML with clickable references."""
    import html as _h
    e = _h.escape
    if not decisions:
        return f"<h2>{e(title)}</h2><p>No decision recorded.</p>"
    s = summary(decisions)
    out = [f"<h2>{e(title)}</h2>", f"<p>{s['decisions']} decisions; <b>{s['no_reference']}</b> without a reference "
           f"(high {s['by_label']['high']}, medium {s['by_label']['medium']}, low {s['by_label']['low']}).</p>",
           "<table border=1 cellpadding=4 style='border-collapse:collapse;font-size:13px'><tr><th>node</th>"
           "<th>topic</th><th>decision</th><th>by</th><th>confidence</th><th>references / evidence</th></tr>"]
    for d in decisions:
        refs_html = "<br>".join(
            (f"<a href='{e(r['link'])}'>{e(refs.format_reference({**r, 'link': ''}))}</a>" if r.get("link")
             else e(refs.format_reference(r))) for r in d.get("references", []))
        ev = "<br>".join(e(f"{x['kind']}: {x['ref']}") for x in d.get("evidence", []))
        flag = "<b style='color:#b42318'>no reference</b><br>" + e(d.get("suggestion", "")) \
            if d.get("no_reference") else ""
        unc = f"<br><i>{e(d['uncited_reason'])}</i>" if d.get("uncited_reason") else ""
        out.append(f"<tr><td>{e(d.get('node') or 'flow')}</td><td>{e(d['topic'])}</td><td>{e(d['what'])}"
                   f"<br><small>{e(d.get('rationale', ''))}</small></td><td>{e(d['decided_by'].get('kind', ''))}: "
                   f"{e(d['decided_by'].get('id', ''))}</td><td>{d['confidence']} {e(d['label'])}</td>"
                   f"<td>{refs_html}{'<br>' if refs_html and ev else ''}{ev}{unc}{flag}</td></tr>")
    out.append("</table>")
    return "\n".join(out)
