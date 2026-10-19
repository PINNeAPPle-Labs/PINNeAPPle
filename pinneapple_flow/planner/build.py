"""Building flows in code (planners, examples): nodes, edges, an automatic left-to-right layout."""
from __future__ import annotations

from typing import Any

from ..schema import Flow


class FlowBuilder:
    def __init__(self, name: str, description: str = ""):
        self.name, self.description = name, description
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self.decisions: list[dict[str, Any]] = []

    def decide(self, rule_id: str, what: str, *, node: str = "", value: Any = None, rationale: str = "",
               extra: tuple[str, ...] = ()) -> None:
        """Record why a node or parameter was chosen (a heuristic rule with its references)."""
        from ..decisions import rule_decision
        self.decisions.append(rule_decision(rule_id, what=what, node=node, value=value, rationale=rationale,
                                            extra=extra).to_dict())

    def add(self, node_id: str, type_: str, params: dict[str, Any] | None = None, *, label: str = "") -> str:
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id!r}")
        self.nodes[node_id] = {"id": node_id, "type": type_, "params": dict(params or {}), "label": label}
        return node_id

    def connect(self, source: str, source_port: str, target: str, target_port: str) -> None:
        self.edges.append({"source": source, "source_port": source_port, "target": target,
                           "target_port": target_port})

    def flow(self, meta: dict[str, Any] | None = None) -> Flow:
        nodes = [dict(n) for n in self.nodes.values()]
        layout(nodes, self.edges)
        meta = dict(meta or {})
        if self.decisions:
            meta["decisions"] = list(self.decisions)
        return Flow(name=self.name, description=self.description, nodes=nodes, edges=list(self.edges), meta=meta)

    def proposal(self, title: str, rationale: str, **meta) -> dict[str, Any]:
        f = self.flow({"title": title, **meta})
        f.description = rationale
        return {"title": title, "rationale": rationale, "flow": f.to_dict()}


def layout(nodes: list[dict[str, Any]], edges: list[dict[str, str]], dx: float = 270.0, dy: float = 150.0):
    """Columns by longest path from a source, rows in order of appearance (in place)."""
    ids = [n["id"] for n in nodes]
    depth = {i: 0 for i in ids}
    for _ in range(len(ids)):
        changed = False
        for e in edges:
            if e["source"] in depth and e["target"] in depth and depth[e["target"]] < depth[e["source"]] + 1:
                depth[e["target"]] = depth[e["source"]] + 1
                changed = True
        if not changed:
            break
    rows: dict[int, int] = {}
    for n in nodes:
        d = depth[n["id"]]
        r = rows.get(d, 0)
        rows[d] = r + 1
        n["position"] = {"x": 40.0 + d * dx, "y": 40.0 + r * dy}
    return nodes
