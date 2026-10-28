"""Scientific and industrial quality assurance for ``pinneapple_llm`` artifacts.

This package is the pipeline described in the module docstrings of its
parts, in execution order::

    spec.py        stage A  typed, serializable physical specification
    static.py      stage C  AST + real API-introspection validation
    sandbox.py     stage D  controlled execution (and its honest limits)
    validators/    stage E  units, numeric, physics, geometry, temporal, sensors
    references.py  stage F  comparison against independent reference data
    repair.py      stage G  diagnosis + bounded automatic repair
    levels.py       Q0-Q4  what the run's evidence actually earns
    report.py              machine-readable, self-contained run record
    pipeline.py            wires A-G; never generates the artifact itself

The one rule that holds everywhere here: no step contributes a ``PASS`` it
did not observe. ``NOT_RUN``, ``BLOCKED`` and ``NOT_APPLICABLE`` stay
distinct all the way into the JSON report and into the quality level.
"""
from __future__ import annotations

from .generation import (GENERATION_SCHEMA, GeneratedArtifact, build_prompt, check_claims,
                         generate_artifact, parse_generation_response, validate_generated)
from .levels import LEVELS, blocking_reasons, evaluate_level, level_description
from .pipeline import PipelineConfig, QualityPipeline
from .references import ReferenceData, compare, split_mask
from .repair import Diagnosis, RepairAttempt, RepairLoop, RepairOutcome, diagnose
from .report import (ArtifactRecord, Finding, QualityReport, Status, StepOutcome,
                     environment_snapshot)
from .sandbox import ISOLATION_LIMITATIONS, IsolationLevel, SandboxExecutor, SandboxLimits, SandboxResult
from .spec import (GeometrySpec, Invariant, Observation, PhysicalSpecification, Provenance,
                   Quantity, SpecError, quantity_map)
from .static import StaticPolicy, validate_json_artifact, validate_python_artifact
from .validators import SensorReading, ValidationContext, ValidatorResult, available

__all__ = [
    "GENERATION_SCHEMA", "GeneratedArtifact", "build_prompt", "check_claims",
    "generate_artifact", "parse_generation_response", "validate_generated",
    "LEVELS", "blocking_reasons", "evaluate_level", "level_description",
    "PipelineConfig", "QualityPipeline",
    "ReferenceData", "compare", "split_mask",
    "Diagnosis", "RepairAttempt", "RepairLoop", "RepairOutcome", "diagnose",
    "ArtifactRecord", "Finding", "QualityReport", "Status", "StepOutcome",
    "environment_snapshot",
    "ISOLATION_LIMITATIONS", "IsolationLevel", "SandboxExecutor", "SandboxLimits", "SandboxResult",
    "GeometrySpec", "Invariant", "Observation", "PhysicalSpecification", "Provenance",
    "Quantity", "SpecError", "quantity_map",
    "StaticPolicy", "validate_json_artifact", "validate_python_artifact",
    "SensorReading", "ValidationContext", "ValidatorResult", "available",
]
