"""RankedItem, DiscoveryResult data models for researcher output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ItemType = Literal["paper", "repo"]


@dataclass
class RankedItem:
    type: ItemType
    id: str
    title: str
    url: str
    score: float
    why: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    topic: str
    papers: list[RankedItem]
    repos: list[RankedItem]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class KBArtifact:
    """
    A persisted artifact (paper/repo) stored inside a KB run folder.
    """
    type: ItemType
    id: str
    title: str
    url: str
    path: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class KBIndex:
    """
    A light-weight index pointing to chunked documents (JSONL).
    """
    topic: str
    run_dir: str
    chunks_path: str
    manifest_path: str


@dataclass
class ExtractedProblemSolution:
    """
    Output of ExtractorAgent. Keep it generic but structured enough
    to map into pinneapple_problemdesign later.
    """
    source_type: ItemType
    source_id: str
    title: str

    problem: str
    solution: str

    # optional structured fields
    equations: str | None = None
    data_requirements: str | None = None
    training_recipe: str | None = None
    metrics: str | None = None
    limitations: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)
