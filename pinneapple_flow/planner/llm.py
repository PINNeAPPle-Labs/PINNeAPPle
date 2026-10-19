"""The LLM planner: the node catalog and the flow schema go in the prompt, the model answers JSON, and every
proposal is validated against the catalog. Invalid output is repaired (unknown types and parameters dropped, bad
edges removed, missing required inputs wired from a compatible upstream output, equations re-checked by the safe
parser); what cannot be repaired falls back to the heuristic planner, and the result says which planner produced
each proposal.

Providers go through :func:`pinneapple_llm.call_llm`: a local Ollama, any OpenAI-compatible endpoint
(``base_url``: vLLM, LM Studio, llama.cpp, a user-hosted model), OpenAI or Anthropic with a key.
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from ..registry import REGISTRY
from ..schema import SCHEMA_ID, Flow, FlowError, validate_flow
from ..types import compatible
from .build import layout

SYSTEM = ("You plan engineering simulation workflows for the PINNeAPPle library. You answer with one JSON object "
          "and nothing else. You only use node types from the catalog, with their exact port names and parameter "
          "names. Equations use the text grammar: u_t, u_xx (derivative shorthand), lap(u), d(f, x), sin, cos, exp, "
          "log, sqrt, pi; one '=' per equation; conditions are {kind: dirichlet|neumann|robin|initial|periodic, on: "
          "x_min|x_max|y_min|...|boundary|t_min, equation}.")


def catalog_prompt() -> str:
    lines = []
    for s in REGISTRY.specs():
        ins = ", ".join(f"{p.name}:{p.type}{'*' if p.multiple else ''}{'' if p.required else '?'}" for p in s.inputs)
        outs = ", ".join(f"{p.name}:{p.type}" for p in s.outputs)
        pars = ", ".join(f"{p.name}={json.dumps(p.default)}" for p in s.params)
        lines.append(f"- {s.type} [{s.category}] in({ins}) out({outs}) params({pars}): "
                     f"{s.description.splitlines()[0] if s.description else s.title}")
    return "\n".join(lines)


def build_prompt(text: str, n: int = 2) -> str:
    example = {"proposals": [{"title": "Classical solver + PINN, validated", "rationale": "why this way",
                              "flow": {"schema": SCHEMA_ID, "name": "heat 2D", "nodes": [
                                  {"id": "geo", "type": "geometry.domain", "params": {"dimension": 2}},
                                  {"id": "eq", "type": "equations.system", "params": {
                                      "template": "heat", "pdes": ["u_t = alpha*lap(u)"],
                                      "conditions": [{"kind": "initial", "on": "t_min",
                                                      "equation": "u = sin(pi*x)*sin(pi*y)"},
                                                     {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                                      "coefficients": {"alpha": 0.1}}},
                                  {"id": "fd", "type": "solver.classical", "params": {"method": "fd"}}],
                                  "edges": [{"source": "geo", "source_port": "geometry", "target": "eq",
                                             "target_port": "geometry"},
                                            {"source": "eq", "source_port": "problem", "target": "fd",
                                             "target_port": "problem"}]}}],
               "detected": {"family": "heat", "dimension": 2}}
    from ..references import registry
    keys = "\n".join(f"- {k}: {v['title'][:90]} ({v['year']})" for k, v in registry().items())
    return (f"Node catalog (port name:type, * = several edges, ? = optional):\n{catalog_prompt()}\n\n"
            f"Reference registry (cite these keys; cite anything else only with a DOI or URL you are sure of, "
            f"never invent one; leave references empty when you have none):\n{keys}\n\n"
            f"Problem description:\n<<<\n{text}\n>>>\n\n"
            f"Propose {n} different ways to solve it with these nodes (e.g. classical + PINN with validation, a "
            f"surrogate, an inverse problem). For every proposal list its decisions (the identified equations, the "
            f"dimension, conditions and coefficients, the method, the architecture and hyperparameters, validation "
            f"thresholds) with their references: \"decisions\": [{{\"node\": \"eq\", \"topic\": \"equations\", "
            f"\"what\": \"...\", \"rationale\": \"...\", \"references\": [\"evans2010pde\"]}}]. "
            f"Answer with JSON shaped like:\n{json.dumps(example)}")


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if m:
        raw = m.group(1)
    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        raw = raw[i:j + 1] if i >= 0 and j > i else raw
    return json.loads(raw)


def repair(flow_doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A flow that validates, built from what can be kept of ``flow_doc``; and the list of repairs made."""
    notes: list[str] = []
    nodes, seen = [], set()
    for n in (flow_doc.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        spec = REGISTRY.get(str(n.get("type")))
        nid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(n.get("id") or ""))[:64] or f"n{len(nodes)}"
        if nid[0].isdigit() or nid[0] == "-":
            nid = "n" + nid
        if spec is None:
            notes.append(f"dropped node {nid!r}: unknown type {n.get('type')!r}")
            continue
        if nid in seen:
            notes.append(f"dropped duplicate node {nid!r}")
            continue
        seen.add(nid)
        params = {}
        for k, v in (n.get("params") or {}).items() if isinstance(n.get("params"), dict) else []:
            p = spec.param(k)
            if p is None:
                notes.append(f"{nid}: dropped unknown parameter {k!r}")
            elif v is not None and p.check(v):
                notes.append(f"{nid}: dropped invalid {k!r} ({p.check(v)})")
            else:
                params[k] = v
        if spec.type == "equations.system":
            params, why = _check_equations(params)
            if why:
                notes.append(f"{nid}: {why}")
        nodes.append({"id": nid, "type": spec.type, "params": params, "label": str(n.get("label") or "")[:80]})
    ids = {n["id"]: n for n in nodes}
    edges = []
    for e in flow_doc.get("edges") or []:
        if not isinstance(e, dict):
            continue
        s, t = ids.get(str(e.get("source"))), ids.get(str(e.get("target")))
        if not s or not t:
            notes.append(f"dropped edge {e.get('source')}->{e.get('target')}: unknown node")
            continue
        sp, tp = REGISTRY[s["type"]].output(str(e.get("source_port"))), REGISTRY[t["type"]].input(
            str(e.get("target_port")))
        if sp is None or tp is None or not compatible(sp.type, tp.type):
            notes.append(f"dropped edge {s['id']}.{e.get('source_port')}->{t['id']}.{e.get('target_port')}: "
                         "unknown port or incompatible types")
            continue
        if not tp.multiple and any(x["target"] == t["id"] and x["target_port"] == tp.name for x in edges):
            notes.append(f"dropped a second edge into {t['id']}.{tp.name}")
            continue
        edges.append({"source": s["id"], "source_port": sp.name, "target": t["id"], "target_port": tp.name})
    # wire missing required inputs from the nearest earlier compatible output
    for i, n in enumerate(nodes):
        for p in REGISTRY[n["type"]].inputs:
            if not p.required or any(x["target"] == n["id"] and x["target_port"] == p.name for x in edges):
                continue
            for m in reversed(nodes[:i]):
                out = next((o for o in REGISTRY[m["type"]].outputs if compatible(o.type, p.type)), None)
                if out is not None:
                    edges.append({"source": m["id"], "source_port": out.name, "target": n["id"],
                                  "target_port": p.name})
                    notes.append(f"wired {m['id']}.{out.name} -> {n['id']}.{p.name}")
                    break
    layout(nodes, edges)
    doc = {"schema": SCHEMA_ID, "name": str(flow_doc.get("name") or "flow")[:120],
           "description": str(flow_doc.get("description") or "")[:2000], "nodes": nodes, "edges": edges,
           "meta": {}}
    return doc, notes


def _check_equations(params: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Parse the equations with the safe parser; on failure, fall back to the template the planner named."""
    from pinneapple_physics.symbolic_pde.text import EquationSyntaxError, parse_system
    pdes = params.get("pdes") or []
    if not pdes:
        return params, ""
    fields = params.get("fields") or ["u"]
    coords = ["x", "y", "z", "t"]
    try:
        parse_system(pdes, params.get("conditions") or [], coords=coords, fields=fields,
                     coefficients=params.get("coefficients") or {})
        return params, ""
    except EquationSyntaxError as exc:
        tpl = params.get("template")
        keep = {k: v for k, v in params.items() if k in ("template", "coefficients")}
        if tpl and tpl != "custom":
            return {**keep, "follow_template": True}, f"equations rejected by the parser ({exc}); using the " \
                                                      f"{tpl} template"
        return {}, f"equations rejected by the parser ({exc}); cleared"


def _provider_kwargs(provider: str, base_url: str) -> dict[str, Any]:
    if provider == "ollama":
        u = urlparse(base_url or "http://127.0.0.1:11434")
        return {"host": u.hostname or "127.0.0.1", "port": u.port or 11434, "timeout": 600}
    if provider in ("openai", "openai_compatible") and base_url:
        return {"base_url": base_url}
    return {}


def plan_llm(text: str, *, provider: str, model: str = "", base_url: str = "", api_key: str = "", n: int = 2,
             call=None) -> dict[str, Any]:
    """Proposals from an LLM, each validated (and repaired) against the catalog. Raises on a provider failure or
    unusable output; :func:`pinneapple_flow.planner.plan` turns that into the heuristic fallback."""
    from pinneapple_llm import call_llm
    call = call or call_llm
    raw = call(build_prompt(text, n), provider=provider, model=model or None, api_key=api_key or None, system=SYSTEM,
               json_mode=True, module="pinneapple_flow.planner", max_tokens=6000,
               **_provider_kwargs(provider, base_url))
    data = _extract_json(raw)
    props = []
    label = f"llm:{provider}/{model or 'default'}"
    for p in (data.get("proposals") or [])[:4]:
        if not isinstance(p, dict) or not isinstance(p.get("flow"), dict):
            continue
        doc, notes = repair(p["flow"])
        if not doc["nodes"]:
            continue
        errs = [i for i in validate_flow(doc) if i.level == "error"]
        if errs:
            continue
        try:
            f = Flow.from_dict(doc)
        except FlowError:
            continue
        from ..decisions import TOPICS, llm_decision
        node_ids = {n.id for n in f.nodes}
        decs = []
        for d in (p.get("decisions") or [])[:60]:
            if not isinstance(d, dict) or not d.get("what"):
                continue
            rs = [r if isinstance(r, (dict, str)) else str(r) for r in (d.get("references") or [])][:8]
            decs.append(llm_decision(f"{provider}/{model or 'default'}", topic=str(d.get("topic") or "other")
                                     if str(d.get("topic")) in TOPICS else "other", what=str(d["what"])[:300],
                                     node=str(d.get("node")) if str(d.get("node")) in node_ids else "",
                                     rationale=str(d.get("rationale") or "")[:600], references=rs).to_dict())
        decs.append(llm_decision(f"{provider}/{model or 'default'}", topic="route",
                                 what=f"route: {str(p.get('title') or f.name)[:120]}",
                                 rationale=str(p.get("rationale") or "")[:600]).to_dict())
        f.meta = {"title": str(p.get("title") or f.name)[:120], "planner": label + ("+repair" if notes else ""),
                  "repairs": notes, "decisions": decs}
        f.description = str(p.get("rationale") or "")[:2000]
        props.append({"title": f.meta["title"], "rationale": f.description, "flow": f.to_dict(),
                      "planner": f.meta["planner"], "repairs": notes})
    if not props:
        raise ValueError("the LLM answer had no valid proposal")
    return {"planner": label, "detected": data.get("detected") or {}, "proposals": props, "warnings": []}
