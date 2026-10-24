"""The Hub and Flow Studio on one registry: a ready flow around any node type, and links between the two apps.

    from pinneapple_hub.library.flowlink import flow_for_node, flow_link, add_link
    out = flow_for_node("geometry.mesh", {"n": 41})     # {"flow", "issues", "ok", "source", "node"}
    flow_link(out["flow"], base="http://127.0.0.1:8765")   # Flow Studio opens it (?flow=<base64url JSON>)
    add_link("geometry.mesh", {"n": 41}, base=...)          # Flow Studio adds the node to a flow the user picks (?add=)

``flow_for_node`` builds *inputs -> the tool -> a report*:

1. when a bundled flow (``pinneapple_flow/flows``) uses the node type, its upstream nodes with their wiring and
   parameters, as committed and validated there;
2. otherwise a producer for each required input, chosen among node types without required inputs of their own (or,
   up to three levels, with inputs that can be produced the same way) whose output feeds that input type;
3. then ``export.report`` on the node's outputs (``items: any``), unless the node already ends in a report.

Every node is a registered type and every parameter is checked against its schema (``NodeSpec.check_params``); the
wiring is ``pinneapple_flow.lab_status.insert_nodes``, the one the Flow Studio GUI uses.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

REPORT = "export.report"
MAX_DEPTH = 3
#: preferred producers per port type (the first registered, available one wins; the rest are found by search)
PREFERRED = {"geometry": "geometry.domain", "mesh": "geometry.mesh", "equations": "equations.system",
             "problem": "problem.generic", "dataset": "data.ingest", "series": "data.ingest",
             "vehicle": "vehicle.spec", "soil": "terramechanics.soil", "site": "scenario.site",
             "concept": "reason.concept", "requirement": "reason.requirement"}


class FlowLinkError(ValueError):
    pass


def _flows_dir() -> Path:
    from pinneapple_flow.service import FLOWS_DIR
    return Path(FLOWS_DIR)


def bundled_flows_using(type_: str) -> list[str]:
    """Names of the bundled flows that contain a node of ``type_``."""
    out = []
    for p in sorted(_flows_dir().glob("*.json")):
        if any(n.get("type") == type_ for n in json.loads(p.read_text()).get("nodes", [])):
            out.append(p.stem)
    return out


def _example_flows_using(type_: str) -> list[tuple[str, Path]]:
    """The example flows (``pinneapple_flow/flows/examples``, one per ``examples/*`` study) that contain ``type_``."""
    try:
        from pinneapple_flow.examples import flow_path, flows
    except ImportError:
        return []
    return [(f"examples/{f['name']}", flow_path(f["name"])) for f in flows()
            if any(n.get("type") == type_ for n in json.loads(flow_path(f["name"]).read_text()).get("nodes", []))]


def _from_bundled(type_: str) -> tuple[dict[str, Any], str, str] | None:
    """The node and its ancestors (nodes, edges, parameters) from the first bundled flow that uses it, else from the
    first example flow that does (the example-domain nodes are wired there)."""
    from pinneapple_flow.schema import as_flow
    found = [(name, _flows_dir() / f"{name}.json") for name in bundled_flows_using(type_)]
    for name, path in found + _example_flows_using(type_):
        flow = as_flow(json.loads(path.read_text()))
        node = next(n for n in flow.nodes if n.type == type_)
        keep = flow.ancestors(node.id) | {node.id}
        doc = flow.to_dict()
        doc["nodes"] = [n for n in doc["nodes"] if n["id"] in keep]
        doc["edges"] = [e for e in doc["edges"] if e["source"] in keep and e["target"] in keep]
        return doc, node.id, name
    return None


def _producer(port_type: str, depth: int, seen: set[str]) -> list[str] | None:
    """Node types (upstream first) that end in one producing ``port_type``."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.palette import is_auto
    from pinneapple_flow.types import compatible
    if depth > MAX_DEPTH:
        return None
    cands = [s for s in REGISTRY.specs() if not is_auto(s) and s.type not in seen and s.available()[0]
             and any(compatible(o.type, port_type) for o in s.outputs)]
    pref = PREFERRED.get(port_type)
    cands.sort(key=lambda s: (s.type != pref, sum(p.required for p in s.inputs), s.type))
    for s in cands:
        chain: list[str] = []
        ok = True
        for p in (p for p in s.inputs if p.required):
            sub = _producer(p.type, depth + 1, seen | {s.type})
            if sub is None:
                ok = False
                break
            chain += [t for t in sub if t not in chain]
        if ok:
            return chain + [s.type]
    return None


def flow_for_node(type_: str, params: dict[str, Any] | None = None, *, report: bool = True) -> dict[str, Any]:
    """See the module doc. Raises :class:`FlowLinkError` for an unknown type; parameter problems and unwired inputs
    come back in ``issues``."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.lab_status import insert_nodes
    from pinneapple_flow.schema import SCHEMA_ID, validate_flow
    spec = REGISTRY.get(type_)
    if spec is None or type_.startswith("custom."):
        raise FlowLinkError(f"unknown node type {type_!r}")
    params = dict(params or {})
    issues = [{"level": "error", "message": f"{spec.title}: {msg}", "param": p} for p, msg in spec.check_params(params)]
    found = _from_bundled(type_)
    if found:
        doc, node_id, src = found
        source = f"upstream of {type_} in the bundled flow {src}"
    else:
        doc = {"schema": SCHEMA_ID, "name": spec.title, "nodes": [], "edges": []}
        anchor = None
        chain: list[str] = []
        for p in (p for p in spec.inputs if p.required):
            sub = _producer(p.type, 1, {type_})
            if sub is None:
                issues.append({"level": "warning", "message": f"no node produces {p.type} for input {p.name}"})
                continue
            chain += [t for t in sub if t not in chain]
        for t in chain:
            before = {n["id"] for n in doc["nodes"]}
            doc, _ = insert_nodes(doc, anchor, [{"type": t}])
            anchor = next(n["id"] for n in doc["nodes"] if n["id"] not in before)
        before = {n["id"] for n in doc["nodes"]}
        doc, missing = insert_nodes(doc, anchor, [{"type": type_}])
        node_id = next(n["id"] for n in doc["nodes"] if n["id"] not in before)
        issues += [{"level": "warning", "message": f"{spec.title}: input {m} is not connected"} for m in missing]
        source = "composed: " + " -> ".join([*chain, type_]) if chain else f"{type_} on its own"
    for n in doc["nodes"]:
        if n["id"] == node_id:
            n["params"] = {**n.get("params", {}), **params}
    if report and type_ != REPORT and spec.outputs and REPORT in REGISTRY:
        doc, _ = insert_nodes(doc, node_id, [{"type": REPORT}])
    doc["name"] = f"{spec.title} (from the PINNeAPPle Hub)"
    doc["description"] = f"Inputs -> {spec.title} -> report. {source}."
    doc.setdefault("meta", {})["created_with"] = "PINNeAPPle Hub"
    issues += [i.to_dict() for i in validate_flow(doc)]
    return {"flow": doc, "node": node_id, "source": source, "issues": issues,
            "ok": not any(i["level"] == "error" for i in issues)}


def b64url(obj: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")


def from_b64url(text: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))


def flow_link(flow: dict[str, Any], *, base: str) -> str:
    """Flow Studio URL that opens ``flow`` (``?flow=``)."""
    return f"{base.rstrip('/')}/?flow={b64url(flow)}"


def add_link(type_: str, params: dict[str, Any] | None = None, *, base: str) -> str:
    """Flow Studio URL that adds a node of ``type_`` (with ``params``) to a flow the user picks (``?add=``)."""
    q = f"?add={type_}"
    if params:
        q += f"&params={b64url(params)}"
    return f"{base.rstrip('/')}/{q}"


__all__ = ["flow_for_node", "flow_link", "add_link", "bundled_flows_using", "b64url", "from_b64url", "FlowLinkError"]
