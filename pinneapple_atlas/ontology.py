"""Small versioned ontology and explicit evidence-backed relations."""
from dataclasses import dataclass

NODE_TYPES = frozenset({"equation", "method", "paper", "implementation", "dataset", "benchmark", "capability"})
RELATIONS = frozenset({"solves", "describes", "implements", "evaluated_on", "uses_dataset", "supported_by", "compared_with"})
ONTOLOGY_VERSION = "0.1"

@dataclass(frozen=True)
class ScientificRelation:
    source_id: str
    target_id: str
    relation: str
    evidence_url: str
    review_state: str = "proposed"

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ValueError("unknown scientific relation")
        if self.review_state not in ("proposed", "reviewed"):
            raise ValueError("invalid review state")
        if not self.evidence_url.startswith(("https://", "http://")):
            raise ValueError("scientific relation requires source evidence")
