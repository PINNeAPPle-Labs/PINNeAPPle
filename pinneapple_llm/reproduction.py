"""Evidence-gated reproduction plans from papers or engineering descriptions.

This module deliberately NEVER executes generated code or fabricates missing
parameters. A language model may propose an extraction; every quoted evidence
span is checked against the supplied source. A plan is blocked when mandatory
experimental inputs are unavailable. Execution belongs to separately audited
capability-registry workers, not to the LLM.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

SCHEMA_VERSION = "1.0"
STAGES = ("source", "extraction", "formulation", "geometry", "mesh", "solver",
          "dataset", "model", "training", "evaluation", "verification", "report")
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "domain", "quantities", "methods", "missing_information"],
    "properties": {
        "title": {"type": "string"},
        "domain": {"type": "string"},
        "quantities": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "value", "unit", "evidence"],
            "properties": {"name": {"type": "string"},
                           "value": {"type": "number"}, "unit": {"type": "string"},
                           "evidence": {"type": "string"}}}},
        "methods": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "evidence"],
            "properties": {"name": {"type": "string"},
                           "evidence": {"type": "string"}}}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
}
REQUIRED = {
    "geometry": ("geometry_source", "geometry_dimensions"),
    "solver": ("governing_equations", "material_parameters", "boundary_conditions",
               "mesh_definition"),
    "dataset": ("case_sampling", "split_protocol"),
    "model": ("model_architecture", "feature_encoding"),
    "training": ("optimizer", "loss", "training_schedule"),
    "evaluation": ("metrics",),
}


def _evidence(source: str, quote: str) -> bool:
    return bool(quote.strip()) and quote.strip() in source


def compile_reproduction(source: str, extraction: dict[str, Any],
                         *, source_type: str = "text",
                         supplied_artifacts: dict[str, str] | None = None) -> dict:
    if source_type not in {"text", "paper", "pdf_text"}:
        raise ValueError("Unsupported source type")
    if not source.strip() or len(source) > 200_000:
        raise ValueError("Source must contain 1..200000 characters")
    if not isinstance(extraction, dict):
        raise ValueError("Extraction must be an object")
    artifacts = supplied_artifacts or {}
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in artifacts.items()):
        raise ValueError("Artifact references must be string identifiers")
    accepted, rejected = [], []
    for q in extraction.get("quantities", []):
        if not isinstance(q, dict) or not isinstance(q.get("value"), (int, float)) or isinstance(q.get("value"), bool):
            rejected.append({"item": q, "reason": "invalid_quantity"})
            continue
        quote = q.get("evidence", "")
        if not isinstance(quote, str) or not _evidence(source, quote):
            rejected.append({"item": q, "reason": "missing_verbatim_source_evidence"})
            continue
        accepted.append({**q, "origin": "source_quote", "verified": "quote_present_only"})
    methods = []
    for item in extraction.get("methods", []):
        if isinstance(item, dict) and isinstance(item.get("evidence"), str) and _evidence(source, item["evidence"]):
            methods.append({**item, "origin": "source_quote"})
        else:
            rejected.append({"item": item, "reason": "missing_verbatim_source_evidence"})
    # Artifact references are metadata only, never trusted paths or shell commands.
    gates = {stage: [key for key in fields if not artifacts.get(key)]
             for stage, fields in REQUIRED.items()}
    missing = sorted(set(str(x) for x in extraction.get("missing_information", [])) |
                     {field for fields in gates.values() for field in fields})
    tasks = []
    for stage in STAGES:
        blockers = gates.get(stage, [])
        tasks.append({"stage": stage,
                      "status": "blocked" if blockers else "specified_not_executed",
                      "missing_artifacts": blockers,
                      "execution": "requires_registry_capability_and_human_approval"})
    return {
        "schema_version": SCHEMA_VERSION,
        "title": str(extraction.get("title", "Untitled study"))[:240],
        "domain": str(extraction.get("domain", "unknown"))[:120],
        "source_type": source_type, "source_sha256": sha256(source.encode()).hexdigest(),
        "quantities": accepted, "methods": methods, "rejected_claims": rejected,
        "missing_information": missing, "stages": tasks,
        "reproduction_level": "source_extracted_not_executed",
        "exact_reproduction_ready": False,  # requires independent runtime and scientific validation
        "warning": "Source quote presence does not verify physical meaning; no simulation or training was run.",
    }
