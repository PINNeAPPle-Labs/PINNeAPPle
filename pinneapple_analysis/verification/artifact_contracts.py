"""Cross-stage semantic contracts for scientific artifacts.

These checks validate *declared metadata*, not the underlying CAD/mesh/solver.
They are preflight gates and must not be described as physical validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

from .pipeline_assurance import Artifact, Check, Status, STAGES


@dataclass(frozen=True)
class FieldContract:
    name: str
    unit: str
    components: int = 1
    frame: str = "global"
    location: str = "node"


def check_field_contracts(
    stage: str,
    expected: Sequence[FieldContract],
    actual: Sequence[FieldContract],
) -> list[Check]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    actual_by_name = {f.name: f for f in actual}
    if len(actual_by_name) != len(actual):
        return [Check("field_schema_unique", stage, Status.FAIL,
                      reason="Duplicate field names")]
    checks = []
    for field in expected:
        observed = actual_by_name.get(field.name)
        if observed is None:
            checks.append(Check(f"field:{field.name}", stage, Status.FAIL,
                                reason="Required field missing",
                                suggested_action="Restore field from the original source"))
            continue
        mismatch = {key: {"expected": getattr(field, key), "actual": getattr(observed, key)}
                    for key in ("unit", "components", "frame", "location")
                    if getattr(field, key) != getattr(observed, key)}
        checks.append(Check(f"field:{field.name}", stage,
                            Status.FAIL if mismatch else Status.PASS,
                            evidence={"mismatches": mismatch},
                            reason="Field metadata mismatch" if mismatch else "",
                            suggested_action="Verify unit/frame mapping against raw source" if mismatch else None))
    return checks


def check_artifact_metadata(artifact: Artifact) -> list[Check]:
    """Reject known-invalid declarations; do not claim uninspected geometry is valid."""
    stage, meta = artifact.stage, artifact.metadata
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    checks = []
    if stage in ("geometry", "mesh"):
        units = meta.get("length_unit")
        checks.append(Check("length_unit_declared", stage,
                            Status.PASS if isinstance(units, str) and units.strip() else Status.FAIL,
                            evidence={"declared_unit": units},
                            reason="" if units else "Geometry/mesh length unit not declared"))
    if stage == "mesh":
        nodes, cells = meta.get("node_count"), meta.get("cell_count")
        valid = (type(nodes) is int and nodes > 0 and type(cells) is int and cells > 0)
        checks.append(Check("mesh_counts", stage, Status.PASS if valid else Status.FAIL,
                            evidence={"nodes": nodes, "cells": cells}))
    if stage == "boundary_conditions":
        required = meta.get("required_regions")
        assigned = meta.get("assigned_regions")
        if not isinstance(required, list) or not isinstance(assigned, list):
            checks.append(Check("bc_region_coverage", stage, Status.NOT_RUN,
                                reason="Region inventory missing"))
        else:
            missing = sorted(set(required) - set(assigned))
            checks.append(Check("bc_region_coverage", stage,
                                Status.FAIL if missing else Status.PASS,
                                evidence={"missing_regions": missing},
                                assumptions=("Required regions list is complete and physically correct",)))
    if stage == "dataset":
        train, test = meta.get("train_groups"), meta.get("evaluation_groups")
        if isinstance(train, list) and isinstance(test, list):
            from .physics_ai_checks import holdout_leakage
            checks.append(holdout_leakage("group_holdout", train, test))
        else:
            checks.append(Check("group_holdout", stage, Status.NOT_RUN,
                                reason="Split group identifiers not provided"))
    if not checks:
        checks.append(Check("stage_metadata_contract", stage, Status.NOT_RUN,
                            reason="No stage-specific metadata contract configured"))
    return checks


def check_lineage_compatibility(parent: Artifact, child: Artifact) -> list[Check]:
    """Cross-stage declarations only: units and coordinate reference."""
    checks = []
    for key in ("length_unit", "coordinate_frame"):
        a, b = parent.metadata.get(key), child.metadata.get(key)
        if a is None or b is None:
            checks.append(Check(f"lineage:{key}", child.stage, Status.NOT_RUN,
                                reason=f"Cannot compare {key}: missing declaration"))
        else:
            checks.append(Check(f"lineage:{key}", child.stage,
                                Status.PASS if a == b else Status.FAIL,
                                evidence={"parent": a, "child": b},
                                reason="Undocumented representation change" if a != b else "",
                                suggested_action="Record an explicit conversion and verify raw/converted parity" if a != b else None))
    return checks
