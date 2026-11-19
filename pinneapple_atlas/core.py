"""Evidence-first Physics AI knowledge graph and coverage analysis.

Unknown is never interpreted as absent, and plans are never treated as working code.
"""
from dataclasses import dataclass, asdict
from typing import Literal

Status = Literal["validated", "implemented", "partial", "planned", "not_identified", "unknown"]
Origin = Literal["public", "pinneapple"]

@dataclass(frozen=True)
class Evidence:
    url: str
    kind: str
    title: str
    origin: Origin
    revision: str = ""

    def __post_init__(self):
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("Evidence requires a public HTTP(S) reference")

@dataclass(frozen=True)
class Capability:
    key: str
    name: str
    origin: Origin
    status: Status
    evidence: tuple[Evidence, ...] = ()
    dimensions: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self):
        if self.status in ("validated", "implemented", "partial") and not self.evidence:
            raise ValueError("Implemented or validated capabilities require evidence")
        if any(e.origin != self.origin for e in self.evidence):
            raise ValueError("Evidence origin must match capability origin")

@dataclass(frozen=True)
class Coverage:
    key: str
    public: tuple[Capability, ...]
    pinneapple: tuple[Capability, ...]
    classification: str


def compare_capabilities(items: list[Capability]) -> list[Coverage]:
    grouped: dict[str, dict[str, list[Capability]]] = {}
    for item in items:
        grouped.setdefault(item.key, {"public": [], "pinneapple": []})[item.origin].append(item)
    result = []
    for key, group in sorted(grouped.items()):
        local = group["pinneapple"]
        if not local:
            state = "not_assessed"  # no inventory is not proof of absence
        elif any(c.status == "validated" for c in local):
            state = "validated"
        elif any(c.status == "implemented" for c in local):
            state = "implemented"
        elif any(c.status == "partial" for c in local):
            state = "partial"
        elif any(c.status == "planned" for c in local):
            state = "planned"
        else:
            state = "unconfirmed"
        result.append(Coverage(key, tuple(group["public"]), tuple(local), state))
    return result


def build_graph(items: list[Capability]) -> dict:
    nodes, edges = [], []
    for item in items:
        node_id = f"{item.origin}:{item.key}"
        nodes.append({"id": node_id, "type": "capability", "name": item.name,
                      "status": item.status, "dimensions": list(item.dimensions)})
        for index, evidence in enumerate(item.evidence):
            ref_id = f"{node_id}:evidence:{index}"
            nodes.append({"id": ref_id, "type": "evidence", **asdict(evidence)})
            edges.append({"source": node_id, "target": ref_id, "relation": "supported_by"})
    for coverage in compare_capabilities(items):
        for public in coverage.public:
            for local in coverage.pinneapple:
                edges.append({"source": f"public:{public.key}",
                              "target": f"pinneapple:{local.key}", "relation": "compared_with"})
    return {"nodes": nodes, "edges": edges}
