"""Findings: each signal a validation run produced, explained without an LLM.

A number alone does not tell an engineer what to do. Every :class:`Finding` answers the same four questions, from
facts the run already produced and nothing else:

* **what happened** -- the observed value against the criterion that was configured, quoted from the record;
* **why it matters** -- what the criterion is for, in terms of the declared use;
* **what is still unknown** -- the causes this evidence cannot separate, stated rather than guessed;
* **what we recommend doing** -- next tests, in the order that would settle the doubt.

The text per rule lives in :data:`RULES` (a versioned table, not generated prose): the same input always gives the
same sentence, and every sentence points at the rule, the metric and the evidence that fired it. Nothing here decides
a status: :mod:`pinneapple_veriphysics.validation_case` turns findings into one, with a documented precedence.
Findings are never "pass" because a check was absent -- an absent check is a ``"not run"`` finding.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .usecase import ValidationProfile

__all__ = ["PASS", "FAIL", "WARNING", "NOT_RUN", "SEVERITY", "Rule", "RULES", "Finding", "build_findings",
           "sort_findings"]

PASS, FAIL, WARNING, NOT_RUN = "pass", "fail", "warning", "not run"
#: Render / precedence order: a failure first, a pass last.
SEVERITY = {FAIL: 0, WARNING: 1, NOT_RUN: 2, PASS: 3}


@dataclass(frozen=True)
class Rule:
    """The fixed wording of one kind of finding. ``why`` and ``unknown`` are about the *rule*, not about a run: the
    numbers of a particular run are quoted in ``Finding.what_happened``."""
    key: str
    category: str
    title: str
    why: str
    unknown: str
    actions: tuple[str, ...]


#: Every explanation this package can produce, keyed by rule id. A rule that is not here cannot fire: an unrecognised
#: signal is reported as an unrecognised signal, never dressed in borrowed words.
RULES: dict[str, Rule] = {
    "VP-ERR-OVER": Rule(
        "VP-ERR-OVER", "validation", "Error above the declared tolerance",
        "The measured error of the primary quantity is larger than the tolerance the declared use accepts, so this "
        "result does not satisfy the criterion that was set for it.",
        "This comparison does not tell whether the excess comes from the surrogate, from the reference itself "
        "(resolution, convergence, measurement), or from both. It also says nothing about the region the decision "
        "depends on: the figure reported is the one in the record.",
        ("Check the convergence of the reference before touching the model: a reference that is not converged cannot "
         "condemn or exonerate a surrogate.",
         "Break the error down by region and by quantity, and look at the one the decision actually uses.",
         "Re-run the comparison with an independent second reference if one is available.")),
    "VP-ERR-OK": Rule(
        "VP-ERR-OK", "validation", "Error within the declared tolerance",
        "The measured error is inside the tolerance configured for the declared use, on the quantities and region "
        "this comparison covered.",
        "Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can "
        "hide a region where the sign of the error changes. Passing here is a precondition of validation, not "
        "validation.",
        ("Confirm the comparison covers the critical quantities and regions declared in the profile.",
         "Add a case at the edge of the intended operating range if none was run.")),
    "VP-CHECK-FAIL": Rule(
        "VP-CHECK-FAIL", "verification", "A mandatory check failed",
        "A check that the configured criteria treated as mandatory failed for this result, so the result cannot be "
        "used as evidence for the declared use.",
        "A failed check establishes that the property does not hold for the evaluated points; it does not by itself "
        "identify whether the cause is the model, the discretisation of the check, or the inputs.",
        ("Read the recorded value and threshold of the check and reproduce it, so the failure is not a harness "
         "artefact.",
         "Fix the cause; do not relax the threshold to make it pass.",
         "Record the new run as a new revision: the failure stays in the history.")),
    "VP-CHECK-NOT-RUN": Rule(
        "VP-CHECK-NOT-RUN", "verification", "A check the profile requires did not run",
        "The criterion refers to a check that produced no verdict in this case. A check that did not run is not a "
        "pass: it is missing evidence, and it stays missing until a record exists.",
        "Nothing is known about that property for this result -- neither that it holds nor that it fails.",
        ("Run the check (the tool named in the finding reports how) and attach its evidence.",
         "If the check genuinely does not apply to this model, change the profile and say why, instead of leaving it "
         "silently empty.")),
    "VP-REF-MISSING": Rule(
        "VP-REF-MISSING", "validation", "No independent reference for the declared use",
        "The profile declares that this use needs evidence independent of the model, and the case carries no "
        "comparison against one. Internal consistency checks can establish that the equations are satisfied; they "
        "cannot establish that the answer is right.",
        "Whether the prediction agrees with reality in this condition is unknown. Nothing about the model's error "
        "can be stated, in either direction.",
        ("Obtain a reference: a converged numerical solution on the same problem, experimental data, an analytic "
         "solution, or a published benchmark.",
         "Until then, report the result as internally consistent and unvalidated against reality -- never as "
         "validated.")),
    "VP-EXTRAPOLATION": Rule(
        "VP-EXTRAPOLATION", "applicability", "Condition of use outside the verified domain",
        "The condition the use asks for falls outside the domain the evidence covers. Being outside the domain does "
        "not make the prediction wrong, and being inside it does not make it right; it makes it unsupported.",
        "The behaviour of the model in this condition is unknown: the residual and the error measured inside the "
        "domain say nothing about how the surrogate behaves where nothing was checked.",
        ("Run a reference computation or an experiment at this condition and compare the fields and the quantities "
         "of interest.",
         "If the result holds, extend the validated envelope to cover it and record the evidence.",
         "If it does not, restrict the declared scope or retrain with samples in this condition, then repeat the "
         "checks.")),
    "VP-UQ-UNCALIBRATED": Rule(
        "VP-UQ-UNCALIBRATED", "uncertainty", "Uncertainty estimated but not calibrated",
        "The case carries an uncertainty estimate (for instance the disagreement between members of an ensemble) "
        "that was never checked against reference data. It is a disagreement bar, not a confidence interval.",
        "Whether the bars cover the true error at the stated confidence is unknown, so they cannot be used as a "
        "safety margin.",
        ("Compare the estimate against reference data and report coverage and calibration error.",
         "Until then, use the estimate only to detect that the model is unstable, not to bound the error.")),
    "VP-UQ-MISSING": Rule(
        "VP-UQ-MISSING", "uncertainty", "No uncertainty estimate",
        "The prediction is a point estimate: the case carries no uncertainty, so the risk of the declared decision "
        "cannot be characterised.",
        "How far the true value can be from the prediction is unknown, which is exactly what a decision with "
        "consequences needs.",
        ("Estimate epistemic uncertainty (ensemble of independent trainings, conformal prediction, Bayesian "
         "posterior) and report the method with its limitations.",
         "Calibrate it against reference data before using it as a bound.")),
    "VP-CRITICAL-NOT-ASSESSED": Rule(
        "VP-CRITICAL-NOT-ASSESSED", "validation", "The declared critical quantity was not assessed",
        "The profile names a quantity or region the decision depends on, and no evidence in this case measured it. A "
        "global error figure does not substitute for it.",
        "Whether the model is right where the decision is made is unknown, even if it is right on average.",
        ("Measure the critical quantity against the reference, or assess the critical region locally.",
         "If it cannot be measured, say so in the declared scope instead of averaging it away.")),
    "VP-PROVENANCE": Rule(
        "VP-PROVENANCE", "provenance", "Provenance incomplete",
        "Part of what produced this evidence is not recorded, so the comparison cannot be reproduced from the case "
        "alone.",
        "Whether a later run of the same checks would give the same numbers is unknown -- inputs, code, seeds or "
        "tolerances are missing from the record.",
        ("Record the missing items (input hashes, tool versions, seeds, tolerances, compute configuration).",
         "Re-run the checks once the record is complete: an evidence that cannot be reproduced cannot be audited.")),
    "VP-PROFILE": Rule(
        "VP-PROFILE", "process", "The use case is not fully specified",
        "The declared profile leaves out something the assessment needs to interpret the evidence, so criteria cannot "
        "be applied to it.",
        "What an acceptable result would be for this use is unknown, so nothing can be declared adequate.",
        ("Complete the profile (decision, consequence, reference kinds) and reassess. Nothing else changes: the "
         "evidence already recorded stays valid.")),
    "VP-REVIEW-PENDING": Rule(
        "VP-REVIEW-PENDING", "process", "Expert review pending",
        "The automated checks do not authorize use on their own. This case has no expert decision recorded, so it "
        "cannot be reported as validated whatever the checks say.",
        "Whether the evidence is sufficient for the intended use, and inside which conditions, is a judgement this "
        "case has not received yet.",
        ("Have a qualified engineer review the evidence, the findings and the limitations, and record a decision with "
         "its justification and the conditions it applies to.",
         "If the reviewer needs more evidence, record that request: it becomes a task, not a change of status.")),
    "VP-REVIEW-APPROVED": Rule(
        "VP-REVIEW-APPROVED", "process", "Expert decision recorded",
        "A reviewer recorded a decision on this case, with its justification and the scope it applies to.",
        "An approval is a judgement about the evidence the reviewer saw, for the scope the reviewer stated. It is "
        "not a proof of validity beyond that scope, and it does not survive a change of model version.",
        ("Apply the result only inside the approved scope.",
         "Re-open the case when the model version, the inputs or the intended use change.")),
    "VP-REVIEW-REJECTED": Rule(
        "VP-REVIEW-REJECTED", "process", "Use rejected by a reviewer",
        "A reviewer examined the evidence and refused this use of the result.",
        "The technical findings stand on their own; the refusal records that the evidence was judged insufficient "
        "for the declared use.",
        ("Address the reason recorded in the review, then reassess as a new revision.",
         "Do not treat the rejection as a technical failure of the model unless a check actually failed.")),
    "VP-REVIEW-MORE": Rule(
        "VP-REVIEW-MORE", "process", "Reviewer asked for more evidence",
        "The reviewer did not decide: the case asks for additional evidence before any decision.",
        "Whether the result is adequate for the declared use is open until the requested evidence exists.",
        ("Produce the requested evidence and attach it: the case re-assesses and keeps the earlier revision.",
         "Record the new decision once the evidence has been reviewed.")),
    "VP-CHECKS-RAN": Rule(
        "VP-CHECKS-RAN", "verification", "Automated checks recorded",
        "The case carries verdicts of automated checks, each traced to the entry that produced it.",
        "Which checks are missing is more important than how many ran: the coverage of the declared criteria is "
        "reported separately, and a check that did not run is not a pass.",
        ("Read the not-run findings next to this one; they are the actual gaps.",
         )),
}


@dataclass(frozen=True)
class Finding:
    """One explained signal. ``status`` is the outcome of the rule (fail / warning / not run / pass); the case, not
    the finding, decides the status of the validation."""
    rule: str
    status: str
    title: str
    what_happened: str
    why_it_matters: str
    what_is_unknown: str
    actions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    observed: float | None = None
    threshold: float | None = None
    unit: str = ""
    category: str = ""
    finding_id: str = ""             # assigned by build_findings, VP-001...

    def __post_init__(self) -> None:
        if self.status not in SEVERITY:
            raise ValueError(f"status must be one of {tuple(SEVERITY)}, got {self.status!r}")

    def render(self) -> str:
        """The four questions, in the order an engineer reads them."""
        head = f"{self.finding_id or 'VP-???'} [{self.status.upper()}] {self.title}"
        lines = [head, f"  What happened: {self.what_happened}", f"  Why it matters: {self.why_it_matters}",
                 f"  What is still unknown: {self.what_is_unknown}"]
        if self.actions:
            lines.append("  Recommended next steps:")
            lines += [f"    {i}. {a}" for i, a in enumerate(self.actions, 1)]
        if self.evidence:
            lines.append(f"  Evidence: {', '.join(self.evidence)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id, "rule": self.rule, "status": self.status, "category": self.category,
            "title": self.title, "what_happened": self.what_happened, "why_it_matters": self.why_it_matters,
            "what_is_unknown": self.what_is_unknown, "actions": list(self.actions), "evidence": list(self.evidence),
            "observed": self.observed, "threshold": self.threshold, "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Finding:
        return cls(rule=str(d["rule"]), status=str(d["status"]), title=str(d["title"]),
                   what_happened=str(d["what_happened"]), why_it_matters=str(d["why_it_matters"]),
                   what_is_unknown=str(d["what_is_unknown"]), actions=tuple(d.get("actions") or ()),
                   evidence=tuple(d.get("evidence") or ()), observed=d.get("observed"),
                   threshold=d.get("threshold"), unit=str(d.get("unit") or ""), category=str(d.get("category") or ""),
                   finding_id=str(d.get("finding_id") or ""))


def _rule(rule_id: str) -> Rule:
    if rule_id not in RULES:
        raise KeyError(f"unknown finding rule {rule_id!r}; known: {sorted(RULES)}")
    return RULES[rule_id]


def _make(rule_id: str, status: str, what_happened: str, *, evidence: Iterable[str] = (),
          observed: float | None = None, threshold: float | None = None, unit: str = "") -> Finding:
    r = _rule(rule_id)
    return Finding(rule=r.key, status=status, title=r.title, what_happened=what_happened, why_it_matters=r.why,
                   what_is_unknown=r.unknown, actions=r.actions, evidence=tuple(evidence), observed=observed,
                   threshold=threshold, unit=unit, category=r.category)


def sort_findings(findings: Sequence[Finding]) -> list[Finding]:
    """Failures first, then warnings, then not-run, then passes; ties keep the generation order (stable), so the same
    case always numbers its findings the same way."""
    return sorted(findings, key=lambda f: SEVERITY[f.status])


def build_findings(*, profile: ValidationProfile, checks: Sequence[Any] = (), references: Sequence[Any] = (),
                   uncertainty: Any = None, applicability: Any = None, provenance_problems: Sequence[str] = (),
                   reviews: Sequence[Any] = (), critical_not_assessed: Sequence[str] = ()) -> list[Finding]:
    """Every finding the case's own contents justify, numbered ``VP-001``... in severity order.

    The arguments are the plain records of a case (see :mod:`pinneapple_veriphysics.validation_case`); this function
    reads them and quotes them, and computes nothing but comparisons of the numbers they carry.
    """
    out: list[Finding] = []
    ran = [c for c in checks if c.passed is not None]
    if ran:
        passed = sum(1 for c in ran if c.passed)
        out.append(_make("VP-CHECKS-RAN", PASS,
                         f"{len(ran)} check(s) produced a verdict: {passed} passed, {len(ran) - passed} failed; "
                         f"{len(checks) - len(ran)} recorded check(s) did not run.",
                         evidence=[c.check_id for c in ran]))

    # ---- automated checks -------------------------------------------------
    for c in checks:
        if c.passed is None:
            if c.critical:
                out.append(_make("VP-CHECK-NOT-RUN", NOT_RUN,
                                 f"{c.check_id}: no verdict recorded"
                                 + (f" ({c.message})" if c.message else "")
                                 + "; the profile treats this check as mandatory.",
                                 evidence=[c.check_id]))
        elif not c.passed:
            val = "" if c.value is None else f" value {c.value:.6g}"
            thr = "" if c.threshold is None else f" against a threshold of {c.threshold:.6g}"
            out.append(_make("VP-CHECK-FAIL", FAIL,
                             f"{c.check_id}:{val}{thr}{(' -- ' + c.message) if c.message else ''}",
                             evidence=[c.check_id], observed=c.value, threshold=c.threshold))
    # a mandatory check that is simply missing from the record (never recorded at all)
    recorded = {c.check_id for c in checks}
    for req in profile.required_checks:
        if req not in recorded:
            out.append(_make("VP-CHECK-NOT-RUN", NOT_RUN,
                             f"{req}: the profile requires this check and the case has no record of it.",
                             evidence=[req]))

    # ---- reference comparisons (the declared acceptance criterion) --------
    declared = [r for r in references if r.independent]
    for r in references:
        tol = r.tolerance_pct if r.tolerance_pct is not None else float(profile.target_error_pct)
        ok = r.value_pct <= tol
        base = (f"{r.metric} against {r.kind} '{r.evidence_id}': {r.value_pct:g} % error, tolerance {tol:g} % "
                f"(declared for this use), source: {r.source or 'not stated'}")
        if ok:
            out.append(_make("VP-ERR-OK", PASS, base, evidence=[r.evidence_id], observed=r.value_pct, threshold=tol,
                             unit="%"))
        else:
            out.append(_make("VP-ERR-OVER", FAIL, base, evidence=[r.evidence_id], observed=r.value_pct, threshold=tol,
                             unit="%"))
    if profile.needs_reference_at_all and not declared:
        out.append(_make("VP-REF-MISSING", NOT_RUN,
                         f"no comparison against an independent reference is recorded, and the profile "
                         f"({profile.objective}) requires one.", evidence=[]))

    # ---- declared critical quantities / regions --------------------------
    for q in critical_not_assessed:
        out.append(_make("VP-CRITICAL-NOT-ASSESSED", NOT_RUN,
                         f"'{q}' is declared critical in the profile and no evidence in this case measures it.",
                         evidence=[q]))

    # ---- applicability / extrapolation -----------------------------------
    if applicability is not None and applicability.checked and applicability.outside_variables:
        margin = ("" if applicability.supported_margin is None
                  else f" evidence supports use up to +{applicability.supported_margin:.0%} beyond the verified range")
        out.append(_make("VP-EXTRAPOLATION", WARNING,
                         f"requested condition '{applicability.requested_condition or 'not named'}' is outside the "
                         f"verified range in: {', '.join(applicability.outside_variables)}.{margin}",
                         evidence=list(applicability.outside_variables)))

    # ---- uncertainty ------------------------------------------------------
    if uncertainty is None:
        out.append(_make("VP-UQ-MISSING", NOT_RUN, "the case carries no uncertainty estimate."))
    elif not uncertainty.calibrated:
        spread = "" if uncertainty.spread_pct is None else f" spread {uncertainty.spread_pct:g} % of field scale;"
        out.append(_make("VP-UQ-UNCALIBRATED", WARNING,
                         f"method '{uncertainty.method}':{spread} not calibrated against reference data"
                         + (f" ({uncertainty.limitation})" if uncertainty.limitation else "."),
                         evidence=[uncertainty.evidence] if uncertainty.evidence else []))

    # ---- provenance -------------------------------------------------------
    for p in provenance_problems:
        out.append(_make("VP-PROVENANCE", WARNING, f"provenance: {p}."))

    # ---- profile ----------------------------------------------------------
    for p in profile.problems():
        out.append(_make("VP-PROFILE", WARNING, p))

    # ---- expert review ----------------------------------------------------
    if not reviews:
        out.append(_make("VP-REVIEW-PENDING", NOT_RUN,
                         "no expert decision is recorded; automated checks alone do not authorize use."))
    else:
        last = reviews[-1]
        scope = (", ".join(f"{k}={v}" for k, v in dict(getattr(last, "scope", {}) or {}).items())
                 or "no scope stated")
        tail = f" by {last.reviewer or 'an unnamed reviewer'} at {last.decided_at}: {last.justification}"
        if last.decision in ("approved", "approved with conditions"):
            out.append(_make("VP-REVIEW-APPROVED", PASS, f"decision '{last.decision}'{tail}; scope: {scope}.",
                             evidence=[last.review_id] if getattr(last, "review_id", "") else []))
        elif last.decision == "rejected":
            out.append(_make("VP-REVIEW-REJECTED", FAIL, f"decision 'rejected'{tail}; scope: {scope}.",
                             evidence=[last.review_id] if getattr(last, "review_id", "") else []))
        elif last.decision == "more evidence requested":
            out.append(_make("VP-REVIEW-MORE", WARNING, f"decision 'more evidence requested'{tail}; scope: {scope}.",
                             evidence=[last.review_id] if getattr(last, "review_id", "") else []))
        else:                                        # suspended
            out.append(_make("VP-REVIEW-MORE", WARNING, f"decision '{last.decision}'{tail}; scope: {scope}.",
                             evidence=[last.review_id] if getattr(last, "review_id", "") else []))

    ordered = sort_findings(out)
    return [replace(f, finding_id=f"VP-{i:03d}") for i, f in enumerate(ordered, 1)]
