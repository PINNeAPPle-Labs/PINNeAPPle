"""Quality levels Q0-Q4, awarded from what actually ran -- never from an LLM.

The levels are cumulative and each one names the *evidence* that grants
it. Two rules keep this from decaying into a badge:

* A level is granted only when every step it requires is present and
  ``PASS``. ``NOT_RUN``, ``BLOCKED`` and (for a validation the spec itself
  required) ``NOT_APPLICABLE`` all fail to grant it.
* ``Q4`` cannot be reached from inside this repository at all. It requires
  operational evidence supplied by a human, and :func:`evaluate_level`
  only accepts it when the evidence names a reviewer, an environment, a
  date and monitoring data. Test-suite output is not operational
  evidence, no matter how many tests pass.

What each level means in this pipeline:

===  ==========================================================================
Q0   artifact generated; nothing verified
Q1   syntax, schema, imports, real-API use and security checks passed
Q2   the scientific/numeric validators the spec required passed
Q3   comparison against independent reference data passed the declared metrics
Q4   operational evidence supplied by a human reviewer (cannot be automated)
===  ==========================================================================
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .report import QualityReport, Status

LEVELS = ("Q0", "Q1", "Q2", "Q3", "Q4")

LEVEL_DESCRIPTIONS = {
    "Q0": "artifact generated, not validated",
    "Q1": "syntax, structure, imports and API contracts verified",
    "Q2": "applicable numeric and physical checks passed",
    "Q3": "independent reference data compared and accepted",
    "Q4": "operational evidence and human review supplied",
}

#: Steps that must exist and pass for Q1 when the artifact is generated
#: *source code*. ``static_schema``/``static_security`` are required only
#: when an artifact of that kind was submitted; the pipeline simply does not
#: record them otherwise.
Q1_REQUIRED = ("static_syntax", "static_imports", "static_api_contracts")

#: For a data/scene artifact (no source), "structure verified" is carried by
#: the structural validators instead -- at least one of these must pass.
Q1_STRUCTURAL = ("static_schema", "validator:numeric", "validator:geometry",
                 "validator:temporal")

#: Prefix identifying the scientific/numeric validators for Q2.
Q2_PREFIX = "validator:"

#: Keys required in ``operational_evidence`` for Q4.
Q4_REQUIRED_EVIDENCE = ("environment", "reviewer", "reviewed_at", "monitoring")


def evaluate_level(report: QualityReport, spec=None) -> Tuple[str, List[str]]:
    """Return ``(level, reasons)`` -- the reasons explain why it stopped there."""
    reasons: List[str] = []

    if not report.artifacts:
        return "Q0", ["no artifact recorded in the report"]

    q1_ok, q1_reasons = _check_q1(report)
    reasons.extend(q1_reasons)
    if not q1_ok:
        return "Q0", reasons
    reasons.append("Q1: static/syntax/API checks passed")

    required = list(getattr(spec, "required_validations", []) or [])
    q2_ok, q2_reasons = _check_q2(report, required)
    reasons.extend(q2_reasons)
    if not q2_ok:
        return "Q1", reasons
    reasons.append("Q2: all applicable and all spec-required validators passed")

    q3_ok, q3_reasons = _check_q3(report)
    reasons.extend(q3_reasons)
    if not q3_ok:
        return "Q2", reasons
    reasons.append("Q3: independent reference comparison passed the declared metrics")

    q4_ok, q4_reasons = _check_q4(report)
    reasons.extend(q4_reasons)
    if not q4_ok:
        return "Q3", reasons
    reasons.append("Q4: operational evidence and human review supplied")

    return "Q4", reasons


def _check_q1(report: QualityReport) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    has_source = any(a.kind == "python_source" for a in report.artifacts)

    if has_source:
        for name in Q1_REQUIRED:
            status = report.status_of(name)
            if status is not Status.PASS:
                reasons.append(f"Q1 blocked: '{name}' is {status.value}")
    else:
        structural = [name for name in Q1_STRUCTURAL if report.status_of(name) is Status.PASS]
        if not structural:
            reasons.append(
                "Q1 blocked: no source artifact to parse and no structural validator passed "
                f"(looked for one of {list(Q1_STRUCTURAL)})")
        for name in Q1_STRUCTURAL:
            step = report.step(name)
            if step is not None and step.status in (Status.FAIL, Status.BLOCKED, Status.NOT_RUN):
                reasons.append(f"Q1 blocked: structural check '{name}' is {step.status.value}")

    for step in report.steps:
        if step.step.startswith("static_") and step.step not in Q1_REQUIRED:
            if step.status in (Status.FAIL, Status.BLOCKED, Status.NOT_RUN):
                reasons.append(f"Q1 blocked: '{step.step}' is {step.status.value}")
    return not reasons, reasons


def _check_q2(report: QualityReport, required: List[str]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not required:
        reasons.append(
            "Q2 blocked: the specification named no required_validations, so no scientific "
            "validator was demanded -- a level must rest on required evidence, not on whatever "
            "happened to run"
        )
    for name in required:
        status = report.status_of(f"{Q2_PREFIX}{name}")
        if status is not Status.PASS:
            reasons.append(f"Q2 blocked: required validator '{name}' is {status.value}")
    for step in report.steps:
        if not step.step.startswith(Q2_PREFIX):
            continue
        if step.status in (Status.FAIL, Status.BLOCKED, Status.NOT_RUN):
            reasons.append(f"Q2 blocked: applicable validator '{step.step[len(Q2_PREFIX):]}' is {step.status.value}")
    return not reasons, reasons


def _check_q3(report: QualityReport) -> Tuple[bool, List[str]]:
    step = report.step("reference_comparison")
    if step is None or step.status is Status.NOT_RUN:
        return False, ["Q3 blocked: no independent reference comparison was run "
                       "(internal consistency is not external evidence)"]
    if step.status is Status.NOT_APPLICABLE:
        return False, ["Q3 blocked: reference comparison was not applicable to this spec"]
    if step.status is not Status.PASS:
        return False, [f"Q3 blocked: reference comparison is {step.status.value}"]
    origin = step.metadata.get("reference_origin")
    if not origin:
        return False, ["Q3 blocked: reference comparison did not record the reference's origin/version"]
    return True, []


def _check_q4(report: QualityReport) -> Tuple[bool, List[str]]:
    ev = report.operational_evidence
    if not ev:
        return False, ["Q4 blocked: operational evidence absent -- synthetic tests and simulations "
                       "cannot establish operational validity"]
    missing = [k for k in Q4_REQUIRED_EVIDENCE if not ev.get(k)]
    if missing:
        return False, [f"Q4 blocked: operational evidence is missing {missing}"]
    return True, []


def blocking_reasons(report: QualityReport) -> List[str]:
    """Human-readable reasons the report cannot be treated as approved."""
    out: List[str] = []
    for step in report.steps:
        if step.status is Status.FAIL:
            out.append(f"step '{step.step}' failed")
        elif step.status is Status.BLOCKED:
            out.append(f"step '{step.step}' was blocked")
    for finding in report.blocking_findings:
        out.append(f"{finding.id}: {finding.message}")
    return out


def level_description(level: str) -> str:
    return LEVEL_DESCRIPTIONS.get(level, "unknown level")
