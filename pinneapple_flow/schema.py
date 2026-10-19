"""The flow document (``pinneapple-flow/1``): nodes, edges, metadata; load, save, validate, JSON Schema.

A flow is plain JSON, so any app (Flow Studio, the PINNeAPPle-CFD app, a script) can write and read it::

    {
      "schema": "pinneapple-flow/1",
      "id": "heat-rod", "name": "Heat in a rod", "description": "...",
      "nodes": [{"id": "geo", "type": "geometry.domain", "params": {"shape": "interval"},
                 "position": {"x": 0, "y": 0}, "label": "Rod"}],
      "edges": [{"id": "e1", "source": "geo", "source_port": "geometry",
                 "target": "eq", "target_port": "geometry"}],
      "meta": {"planner": "heuristic", "prompt": "heat in a rod ..."}
    }

Only node types registered in the catalog run (:mod:`pinneapple_flow.nodes`); the document never carries code.
:func:`validate_flow` checks the structure, the parameters against each node's schema, the port types of every edge
(:func:`pinneapple_flow.types.compatible`), required inputs and cycles, and returns a list of :class:`Issue`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_ID = "pinneapple-flow/1"
SCHEMA_VERSION = 1

_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")
MAX_NODES = 200


class FlowError(ValueError):
    """The flow document is malformed or does not validate."""

    def __init__(self, message: str, issues: list[Issue] | None = None):
        super().__init__(message)
        self.issues = issues or []


@dataclass
class Node:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    label: str = ""


@dataclass
class Edge:
    source: str
    source_port: str
    target: str
    target_port: str
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.source}.{self.source_port}->{self.target}.{self.target_port}"


@dataclass
class Flow:
    name: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    id: str = ""
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA_ID

    def __post_init__(self):
        self.nodes = [n if isinstance(n, Node) else Node(**n) for n in self.nodes]
        self.edges = [e if isinstance(e, Edge) else Edge(**e) for e in self.edges]
        if not self.id:
            self.id = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")[:48] or "flow"

    # -- graph helpers ---------------------------------------------------------------------------------------------
    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def upstream_ids(self, node_id: str) -> list[str]:
        return list(dict.fromkeys(e.source for e in self.incoming(node_id)))

    def downstream(self, node_id: str) -> set[str]:
        out: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for e in self.edges:
                if e.source == cur and e.target not in out:
                    out.add(e.target)
                    frontier.append(e.target)
        return out

    def ancestors(self, node_id: str) -> set[str]:
        out: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for e in self.edges:
                if e.target == cur and e.source not in out:
                    out.add(e.source)
                    frontier.append(e.source)
        return out

    def order(self) -> list[str]:
        """Topological order, stable in node order; FlowError on a cycle."""
        indeg = {n.id: 0 for n in self.nodes}
        for e in self.edges:
            if e.target in indeg:
                indeg[e.target] += 1
        ready = [n.id for n in self.nodes if indeg[n.id] == 0]
        out = []
        while ready:
            cur = ready.pop(0)
            out.append(cur)
            for e in self.edges:
                if e.source == cur and e.target in indeg:
                    indeg[e.target] -= 1
                    if indeg[e.target] == 0:
                        ready.append(e.target)
        if len(out) != len(self.nodes):
            raise FlowError(f"the flow has a cycle through {sorted(k for k, v in indeg.items() if v > 0)}")
        return out

    # -- (de)serialisation -----------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "id": self.id, "name": self.name, "description": self.description,
                "nodes": [asdict(n) for n in self.nodes], "edges": [asdict(e) for e in self.edges],
                "meta": self.meta}

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Flow:
        if not isinstance(d, dict):
            raise FlowError("a flow is a JSON object")
        schema = d.get("schema", SCHEMA_ID)
        if schema != SCHEMA_ID:
            raise FlowError(f"unsupported flow schema {schema!r} (this library reads {SCHEMA_ID!r})")
        extra = set(d) - {"schema", "id", "name", "description", "nodes", "edges", "meta"}
        if extra:
            raise FlowError(f"unknown flow keys {sorted(extra)}")
        nodes, edges = d.get("nodes") or [], d.get("edges") or []
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise FlowError("nodes and edges are lists")
        if len(nodes) > MAX_NODES:
            raise FlowError(f"at most {MAX_NODES} nodes")
        try:
            ns = [Node(**_only(n, Node, "node")) for n in nodes]
            es = [Edge(**_only(e, Edge, "edge")) for e in edges]
        except TypeError as exc:
            raise FlowError(f"malformed node or edge: {exc}") from None
        return cls(name=str(d.get("name") or "flow"), nodes=ns, edges=es, id=str(d.get("id") or ""),
                   description=str(d.get("description") or ""), meta=dict(d.get("meta") or {}))

    @classmethod
    def load(cls, path: str | Path) -> Flow:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path


def _only(d: Any, cls, what: str) -> dict:
    if not isinstance(d, dict):
        raise FlowError(f"each {what} is a JSON object")
    extra = set(d) - set(cls.__dataclass_fields__)
    if extra:
        raise FlowError(f"{what} {d.get('id', '')!r}: unknown keys {sorted(extra)}")
    return d


def as_flow(flow: Flow | dict | str | Path) -> Flow:
    """A :class:`Flow` from a Flow, a dict, a JSON string or a path to a ``.json`` file."""
    if isinstance(flow, Flow):
        return flow
    if isinstance(flow, dict):
        return Flow.from_dict(flow)
    if isinstance(flow, Path) or (isinstance(flow, str) and not flow.lstrip().startswith("{")):
        return Flow.load(flow)
    return Flow.from_dict(json.loads(flow))


@dataclass
class Issue:
    level: str                  # "error" | "warning"
    message: str
    node: str = ""
    edge: str = ""
    param: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_flow(flow: Flow | dict, registry=None) -> list[Issue]:
    """Every problem of a flow: unknown node types or parameters, bad values, edges between unknown ports or
    incompatible types, missing required inputs, too many edges into a single input, cycles. Empty list = valid."""
    from .registry import REGISTRY
    from .types import compatible

    reg = registry or REGISTRY
    issues: list[Issue] = []
    try:
        flow = as_flow(flow)
    except FlowError as exc:
        return [Issue("error", str(exc))]
    ids = [n.id for n in flow.nodes]
    for i in sorted({i for i in ids if ids.count(i) > 1}):
        issues.append(Issue("error", f"duplicate node id {i!r}", node=i))
    specs = {}
    for n in flow.nodes:
        if not _ID.match(n.id):
            issues.append(Issue("error", "node ids: a letter or _, then letters, digits, _ or - (max 64)", node=n.id))
        spec = reg.get(n.type)
        if spec is None:
            issues.append(Issue("error", f"unknown node type {n.type!r}", node=n.id))
            continue
        specs[n.id] = spec
        if not isinstance(n.params, dict):
            issues.append(Issue("error", "params must be an object", node=n.id))
            continue
        for name, msg in spec.check_params(n.params):
            issues.append(Issue("error", msg, node=n.id, param=name))
    known = set(ids)
    fan_in: dict[tuple[str, str], int] = {}
    for e in flow.edges:
        if e.source not in known or e.target not in known:
            issues.append(Issue("error", "edge between unknown nodes", edge=e.id))
            continue
        s, t = specs.get(e.source), specs.get(e.target)
        if s is None or t is None:
            continue
        sp, tp = s.output(e.source_port), t.input(e.target_port)
        if sp is None:
            issues.append(Issue("error", f"{s.type} has no output {e.source_port!r}", edge=e.id, node=e.source))
            continue
        if tp is None:
            issues.append(Issue("error", f"{t.type} has no input {e.target_port!r}", edge=e.id, node=e.target))
            continue
        if not compatible(sp.type, tp.type):
            issues.append(Issue("error", f"type mismatch: {e.source}.{e.source_port} is {sp.type!r}, "
                                         f"{e.target}.{e.target_port} expects {tp.type!r}", edge=e.id,
                                node=e.target))
        fan_in[(e.target, e.target_port)] = fan_in.get((e.target, e.target_port), 0) + 1
    for (nid, port), k in fan_in.items():
        p = specs[nid].input(port)
        if k > 1 and p is not None and not p.multiple:
            issues.append(Issue("error", f"input {port!r} takes one edge, has {k}", node=nid))
    for nid, spec in specs.items():
        for p in spec.inputs:
            if p.required and (nid, p.name) not in fan_in:
                issues.append(Issue("error", f"required input {p.name!r} ({p.type}) is not connected", node=nid))
    try:
        flow.order()
    except FlowError as exc:
        issues.append(Issue("error", str(exc)))
    return issues


def check_flow(flow: Flow | dict, registry=None) -> Flow:
    """The flow, or FlowError listing every error (warnings pass)."""
    flow = as_flow(flow)
    errs = [i for i in validate_flow(flow, registry) if i.level == "error"]
    if errs:
        raise FlowError("; ".join(f"{i.node or i.edge or 'flow'}: {i.message}" for i in errs[:8]), errs)
    return flow


def flow_json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of the flow document; node ``params`` are checked per type by the catalog
    (``REGISTRY.get(type).params_schema()``)."""
    pos = {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://pinneapple.dev/schemas/pinneapple-flow-1.json",
        "title": "PINNeAPPle flow", "type": "object", "required": ["schema", "name", "nodes", "edges"],
        "additionalProperties": False,
        "properties": {
            "schema": {"const": SCHEMA_ID}, "id": {"type": "string"}, "name": {"type": "string"},
            "description": {"type": "string"}, "meta": {"type": "object"},
            "nodes": {"type": "array", "maxItems": MAX_NODES, "items": {
                "type": "object", "required": ["id", "type"], "additionalProperties": False,
                "properties": {"id": {"type": "string", "pattern": _ID.pattern}, "type": {"type": "string"},
                               "params": {"type": "object"}, "position": pos, "label": {"type": "string"}}}},
            "edges": {"type": "array", "items": {
                "type": "object", "required": ["source", "source_port", "target", "target_port"],
                "additionalProperties": False,
                "properties": {k: {"type": "string"} for k in ("id", "source", "source_port", "target",
                                                                "target_port")}}},
        },
    }
