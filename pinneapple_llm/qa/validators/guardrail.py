"""Bridge to the existing :class:`pinneapple_llm.guardrail.PhysicsGuardrail`.

This exists so the pipeline reuses the repo's real anti-hallucination
machinery instead of re-implementing a weaker copy. It applies only when
the caller supplies a trained model *and* the ``ProblemSpec`` it was
trained on; otherwise it is ``NOT_APPLICABLE``. Every guardrail check
becomes one finding, so a pipeline report shows exactly which physical
signals were recomputed and which were absent from the guardrail's own
coverage.
"""
from __future__ import annotations

from ..report import Finding, Status
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "physics_guardrail"


@register(_NAME)
def check_guardrail(ctx: ValidationContext) -> ValidatorResult:
    model = ctx.model
    problem_spec = ctx.extra.get("problem_spec")
    if model is None or problem_spec is None:
        return not_applicable(
            _NAME,
            "no trained model + ProblemSpec supplied; the guardrail's residual, conservation and "
            "reference checks were not run",
        )

    try:
        from ..guardrail import PhysicsGuardrail

        guardrail = PhysicsGuardrail(problem_spec, **ctx.extra.get("guardrail_kwargs", {}))
        kwargs = dict(ctx.extra.get("guardrail_check_kwargs", {}))
        report = guardrail.check(model, **kwargs)
    except Exception as exc:  # a missing torch, a bad model, a mismatched spec
        return ValidatorResult(
            name=_NAME, status=Status.BLOCKED,
            findings=[Finding(
                id="validator.physics_guardrail.blocked", status=Status.BLOCKED, severity="error",
                message=f"PhysicsGuardrail could not run: {type(exc).__name__}: {exc}",
                evidence={"error": str(exc)},
                justification="a guardrail that did not run cannot support any claim of verification")],
        )

    findings = []
    for check in report.checks:
        findings.append(Finding(
            id=f"guardrail.{check.name}",
            status=Status.PASS if check.passed else Status.FAIL,
            severity="info" if check.passed else "error",
            message=check.detail,
            evidence={"check": check.name, "value": check.value, "threshold": check.threshold},
        ))
    not_checked = report.skipped
    if not_checked:
        findings.append(Finding(
            id="validator.physics_guardrail.coverage", status=Status.NOT_RUN, severity="warning",
            message=f"guardrail checks not applicable to this pde_kind/spec: {not_checked}",
            evidence={"skipped": not_checked, "ran": report.checked_names},
        ))
    status = Status.PASS if report.trustworthy else Status.FAIL
    return ValidatorResult(
        name=_NAME, status=status, findings=findings,
        evidence={"checks": report.checked_names, "skipped": report.skipped,
                  "trustworthy": report.trustworthy},
        limitations=["PhysicsGuardrail checks a numerical residual and a few conservation laws; it "
                     "does not establish agreement with the real system"],
    )
