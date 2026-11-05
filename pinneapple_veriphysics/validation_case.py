"""Validation case: one prediction, the evidence behind it, the decision about it, and the history of that decision.

The unit of the validation loop is not a score but a *case*: a model version, a reproducible physical problem, the
checks that actually ran, the reference comparisons, the measured domain of applicability, the uncertainty, an expert
decision with its scope, and an append-only history of all of it. A case is created by :meth:`ValidationCase.open` and
only ever evolves by appending a revision -- reviewing, attaching evidence or re-running checks produces a new
revision, never an edit of the old one. Every revision stores the snapshot it sealed, so tampering is detectable
(:meth:`ValidationCase.history_problems`).

The status of a case is one of :data:`STATUSES`, decided by the documented precedence in :meth:`ValidationCase.decide`
from the findings (:mod:`pinneapple_veriphysics.findings`) of the evidence actually present:

``failed``         a mandatory check failed, a reference comparison exceeded the declared tolerance, or an expert
                   refused the use;
``unvalidated``    the evidence the declared profile requires has not been produced (nothing ran, a required check
                   or an independent reference is missing);
``review required`` evidence exists but carries flags (extrapolation, uncalibrated uncertainty, an incomplete
                   requirement) or the expert decision that authorizes use is still missing;
``validated``      the requirements hold *and* an authorized expert approved the use inside a documented scope.

Absence of alerts is never a pass: the checks are not run by this module, and a case with nothing recorded is
``unvalidated`` by construction.

::

    from pinneapple_veriphysics.validation_case import ValidationCase, ModelRef, ProblemRef, CheckRecord

    case = ValidationCase.open(profile=profile, model=ModelRef("pinn", "v3", checkpoint_sha256=sha),
                               problem=ProblemRef("heat 2-D", pde="poisson"), requested_use="screening at 95 C",
                               actor="analyst")
    case = case.with_checks([CheckRecord("PDE residual", "verification", True, 3e-3, 1e-2)], actor="analyst")
    print(case.assess().render())            # findings, maturity on both axes, status and why
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .findings import FAIL, NOT_RUN, WARNING, Finding, build_findings
from .ledger import EvidenceLedger, canonical_json, sha256_json, utc_now
from .maturity import MaturityAssessment, assess_maturity
from .usecase import REFERENCE_KINDS, ValidationProfile

__all__ = [
    "STATUSES", "VALIDATED", "REVIEW_REQUIRED", "FAILED", "UNVALIDATED", "CHECK_CATEGORIES", "REVIEW_DECISIONS",
    "ModelRef", "ProblemRef", "Provenance", "ProcessControls", "CheckRecord", "ReferenceComparison",
    "UncertaintyReport", "ApplicabilityFacts", "ExpertReview", "Revision", "Assessment", "ValidationCase",
    "ValidationRegistry", "checks_from_lab_report", "checks_from_ledger", "checks_from_trust_card",
    "applicability_facts_from_map", "CASE_SCHEMA",
]

VALIDATED, REVIEW_REQUIRED, FAILED, UNVALIDATED = "validated", "review required", "failed", "unvalidated"
STATUSES = (VALIDATED, REVIEW_REQUIRED, FAILED, UNVALIDATED)
CASE_SCHEMA = "pinneapple-validation-case/v1"
REGISTRY_SCHEMA = "pinneapple-validation-registry/v1"

#: What a check establishes. ``verification``: the computation was done right; ``physics``: the solution obeys the
#: physical constraints; ``validation``: it agrees with independent data; ``applicability``: it is used where it holds;
#: ``uncertainty``/``robustness``/``reproducibility``: the statements about it.
CHECK_CATEGORIES = ("verification", "physics", "validation", "applicability", "uncertainty", "robustness",
                    "reproducibility", "other")

#: What an expert can decide. Every decision needs a justification; an approval needs the scope it applies to.
REVIEW_DECISIONS = ("approved", "approved with conditions", "rejected", "more evidence requested", "suspended")
_APPROVALS = ("approved", "approved with conditions")

_KIND_TO_CATEGORY = {"sanity": "verification", "physics": "physics", "reference": "validation",
                     "generalization": "applicability", "coverage": "other", "numerics": "verification",
                     "verification": "verification", "validation": "validation"}


def _frozen(pairs: Iterable[tuple[str, Any]]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(k), v) for k, v in pairs))


def _dict(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    return {k: v for k, v in pairs}


# ==================================================================================================== references
@dataclass(frozen=True)
class ModelRef:
    """What exactly produced the prediction: a name, a version and an identifiable artefact (checkpoint hash) or the
    configuration that rebuilds it. Without one of the two the case cannot say which model it validated."""
    name: str
    version: str = ""
    checkpoint_sha256: str = ""
    preset: str = ""
    configuration: tuple[tuple[str, Any], ...] = ()

    @property
    def configuration_sha256(self) -> str:
        return sha256_json(_dict(self.configuration)) if self.configuration else ""

    def problems(self) -> list[str]:
        out = []
        if not self.version:
            out.append(f"model '{self.name}' has no version: a validated model whose version is not pinned is not "
                       "identifiable")
        if not self.checkpoint_sha256 and not self.configuration:
            out.append(f"model '{self.name}' carries no checkpoint hash and no configuration digest")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "checkpoint_sha256": self.checkpoint_sha256,
                "preset": self.preset, "configuration": _dict(self.configuration)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModelRef:
        return cls(str(d["name"]), str(d.get("version") or ""), str(d.get("checkpoint_sha256") or ""),
                   str(d.get("preset") or ""), _frozen((d.get("configuration") or {}).items()))


@dataclass(frozen=True)
class ProblemRef:
    """The physical problem the prediction answers, described well enough to rebuild it."""
    description: str
    pde: str = ""
    geometry: str = ""
    parameters: tuple[tuple[str, float], ...] = ()
    conditions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "pde": self.pde, "geometry": self.geometry,
                "parameters": _dict(self.parameters), "conditions": list(self.conditions)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ProblemRef:
        return cls(str(d["description"]), str(d.get("pde") or ""), str(d.get("geometry") or ""),
                   tuple((str(k), float(v)) for k, v in (d.get("parameters") or {}).items()),
                   tuple(str(c) for c in (d.get("conditions") or ())))


@dataclass(frozen=True)
class Provenance:
    """What reproducing these numbers requires: which tool ran, on which version, over which inputs, with which seeds
    and tolerances. ``problems()`` lists what is missing -- an incomplete record is a finding, never a silent gap."""
    tool: str = ""
    tool_version: str = ""
    python: str = ""
    platform: str = ""
    code_sha256: str = ""
    data_sha256: str = ""
    seeds: tuple[int, ...] = ()
    tolerances: tuple[tuple[str, float], ...] = ()
    compute: tuple[tuple[str, str], ...] = ()

    def problems(self) -> list[str]:
        out = []
        if not self.tool or not self.tool_version:
            out.append("the tool that ran the checks is not identified with a version")
        if not self.data_sha256:
            out.append("no SHA-256 of the inputs")
        if not self.code_sha256:
            out.append("no SHA-256 of the code that produced the evidence")
        if not self.seeds:
            out.append("no seed recorded, so a stochastic step cannot be repeated")
        if not self.tolerances:
            out.append("the tolerances used are not recorded with the results")
        return out

    @property
    def complete(self) -> bool:
        return not self.problems()

    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "tool_version": self.tool_version, "python": self.python, "platform": self.platform,
                "code_sha256": self.code_sha256, "data_sha256": self.data_sha256, "seeds": list(self.seeds),
                "tolerances": _dict(self.tolerances), "compute": _dict(self.compute)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        return cls(str(d.get("tool") or ""), str(d.get("tool_version") or ""), str(d.get("python") or ""),
                   str(d.get("platform") or ""), str(d.get("code_sha256") or ""), str(d.get("data_sha256") or ""),
                   tuple(int(s) for s in (d.get("seeds") or ())),
                   tuple((str(k), float(v)) for k, v in (d.get("tolerances") or {}).items()),
                   _frozen((d.get("compute") or {}).items()))


@dataclass(frozen=True)
class ProcessControls:
    """The controls an operational use needs: who owns the result, what is monitored, what happens when the model
    leaves its scope, and how a change of model re-opens the validation."""
    owner: str = ""
    monitoring: str = ""
    fallback: str = ""
    change_management: str = ""

    @property
    def documented(self) -> bool:
        return bool(self.monitoring.strip() and self.fallback.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "monitoring": self.monitoring, "fallback": self.fallback,
                "change_management": self.change_management}


# ==================================================================================================== evidence
@dataclass(frozen=True)
class CheckRecord:
    """The verdict of one check, as the tool that ran it recorded it. ``passed is None`` means the check did not run,
    and it stays "not run" -- never a neutral pass."""
    check_id: str
    category: str
    passed: bool | None
    value: float | None = None
    threshold: float | None = None
    message: str = ""
    critical: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.category not in CHECK_CATEGORIES:
            raise ValueError(f"category must be one of {CHECK_CATEGORIES}, got {self.category!r}")

    @property
    def ran(self) -> bool:
        return self.passed is not None

    def to_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "category": self.category, "passed": self.passed, "value": self.value,
                "threshold": self.threshold, "message": self.message, "critical": self.critical,
                "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CheckRecord:
        return cls(str(d["check_id"]), str(d["category"]), d.get("passed"), d.get("value"), d.get("threshold"),
                   str(d.get("message") or ""), bool(d.get("critical")), str(d.get("evidence") or ""))


@dataclass(frozen=True)
class ReferenceComparison:
    """A comparison of the prediction against evidence independent of it: the acceptance criterion of the declared
    use, expressed as ``value_pct`` (the measured error, percent) against ``tolerance_pct`` (the accepted one)."""
    evidence_id: str
    kind: str
    metric: str
    value_pct: float
    tolerance_pct: float | None = None          # None: the profile's declared tolerance applies
    independent: bool = True
    description: str = ""
    source: str = ""
    artefact_sha256: str = ""
    covers: tuple[str, ...] = ()                # quantities / regions this comparison covers
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.kind not in REFERENCE_KINDS or self.kind == "none":
            raise ValueError(f"kind must be one of {REFERENCE_KINDS[:-1]}, got {self.kind!r}")
        if not (isinstance(self.value_pct, (int, float)) and self.value_pct >= 0):
            raise ValueError(f"value_pct must be a non-negative error in percent, got {self.value_pct!r}")

    def within(self, declared_tolerance_pct: float) -> bool:
        tol = declared_tolerance_pct if self.tolerance_pct is None else self.tolerance_pct
        return self.value_pct <= tol

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "kind": self.kind, "metric": self.metric,
                "value_pct": float(self.value_pct),
                "tolerance_pct": None if self.tolerance_pct is None else float(self.tolerance_pct),
                "independent": self.independent, "description": self.description, "source": self.source,
                "artefact_sha256": self.artefact_sha256, "covers": list(self.covers), "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReferenceComparison:
        return cls(str(d["evidence_id"]), str(d["kind"]), str(d["metric"]), float(d["value_pct"]),
                   None if d.get("tolerance_pct") is None else float(d["tolerance_pct"]),
                   bool(d.get("independent", True)), str(d.get("description") or ""), str(d.get("source") or ""),
                   str(d.get("artefact_sha256") or ""), tuple(d.get("covers") or ()), str(d.get("created_at") or ""))


@dataclass(frozen=True)
class UncertaintyReport:
    """A measured uncertainty and, explicitly, whether it was calibrated. An uncalibrated spread is a disagreement
    bar (it detects instability); it is not a confidence interval and cannot bound the error."""
    method: str
    calibrated: bool = False
    spread_pct: float | None = None             # of field scale, when the method yields a spread
    coverage: float | None = None               # empirical interval coverage, when calibrated
    target_coverage: float | None = None
    limitation: str = ""
    evidence: str = ""

    @property
    def quantified(self) -> bool:
        return self.calibrated or self.spread_pct is not None or self.coverage is not None

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "calibrated": self.calibrated, "spread_pct": self.spread_pct,
                "coverage": self.coverage, "target_coverage": self.target_coverage, "limitation": self.limitation,
                "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> UncertaintyReport:
        return cls(str(d["method"]), bool(d.get("calibrated")), d.get("spread_pct"), d.get("coverage"),
                   d.get("target_coverage"), str(d.get("limitation") or ""), str(d.get("evidence") or ""))


@dataclass(frozen=True)
class ApplicabilityFacts:
    """What the applicability evidence says about the condition the use asks for: the ranges that were verified and
    which of the requested inputs fall outside them."""
    checked: bool = False
    requested_condition: str = ""
    outside_variables: tuple[str, ...] = ()
    supported_margin: float | None = None       # measured, as a fraction of the verified range
    verified_ranges: tuple[tuple[str, float, float], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def within_domain(self) -> bool:
        return self.checked and not self.outside_variables

    def to_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "requested_condition": self.requested_condition,
                "outside_variables": list(self.outside_variables), "supported_margin": self.supported_margin,
                "verified_ranges": [{"name": n, "lower": lo, "upper": hi} for n, lo, hi in self.verified_ranges],
                "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ApplicabilityFacts:
        return cls(bool(d.get("checked")), str(d.get("requested_condition") or ""),
                   tuple(d.get("outside_variables") or ()), d.get("supported_margin"),
                   tuple((str(r["name"]), float(r["lower"]), float(r["upper"]))
                         for r in (d.get("verified_ranges") or ())), tuple(d.get("notes") or ()))


# ==================================================================================================== review
@dataclass(frozen=True)
class ExpertReview:
    """A decision taken by a person about the evidence in the case. It never overwrites the technical results: it is
    recorded next to them, with its justification and -- for an approval -- the scope it applies to."""
    reviewer: str
    decision: str
    justification: str
    scope: tuple[tuple[str, Any], ...] = ()
    role: str = ""
    decided_at: str = ""
    evidence_ids: tuple[str, ...] = ()
    review_id: str = ""

    def __post_init__(self) -> None:
        if self.decision not in REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of {REVIEW_DECISIONS}, got {self.decision!r}")
        if not str(self.justification).strip():
            raise ValueError("an expert decision needs a justification: an approval without a stated reason cannot "
                             "be audited")
        if not str(self.reviewer).strip():
            raise ValueError("an expert decision needs a reviewer")

    @property
    def approved(self) -> bool:
        return self.decision in _APPROVALS

    def covers(self, requested_use: str) -> bool:
        """Whether this decision applies to ``requested_use``. A scope that names no use covers the case as assessed;
        a scope that names uses covers only those (a validated range is not a validated use outside it)."""
        raw = _dict(self.scope).get("uses", _dict(self.scope).get("use"))
        if raw is None or (isinstance(raw, (list, tuple)) and not raw):
            return True
        uses = [raw] if isinstance(raw, str) else [str(u) for u in raw]
        return requested_use in uses

    def to_dict(self) -> dict[str, Any]:
        return {"reviewer": self.reviewer, "role": self.role, "decision": self.decision,
                "justification": self.justification, "scope": _dict(self.scope), "decided_at": self.decided_at,
                "evidence_ids": list(self.evidence_ids), "review_id": self.review_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExpertReview:
        return cls(str(d["reviewer"]), str(d["decision"]), str(d.get("justification") or ""),
                   _frozen((d.get("scope") or {}).items()), str(d.get("role") or ""), str(d.get("decided_at") or ""),
                   tuple(d.get("evidence_ids") or ()), str(d.get("review_id") or ""))

    @classmethod
    def record(cls, reviewer: str, decision: str, justification: str, *, role: str = "", scope: Mapping[str, Any] | None
               = None, evidence_ids: Sequence[str] = ()) -> ExpertReview:
        """A review stamped with the current UTC time and a stable id (``reviewer`` + ``decided_at`` + decision)."""
        at = utc_now()
        body = {"reviewer": reviewer, "decision": decision, "justification": justification, "scope": dict(scope or {}),
                "decided_at": at, "evidence_ids": list(evidence_ids)}
        return cls(reviewer=reviewer, decision=decision, justification=justification, role=role,
                   scope=_frozen((scope or {}).items()), decided_at=at, evidence_ids=tuple(evidence_ids),
                   review_id="RV-" + sha256_json(body)[:10])


# ==================================================================================================== record of revisions
@dataclass(frozen=True)
class Revision:
    """One sealed state of a case. ``snapshot`` is what the case looked like after the change; ``snapshot_sha256``
    seals it, so an edited history no longer verifies."""
    revision: int
    created_at: str
    action: str
    actor: str
    note: str
    status: str
    snapshot_sha256: str
    snapshot: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "created_at": self.created_at, "action": self.action, "actor": self.actor,
                "note": self.note, "status": self.status, "snapshot_sha256": self.snapshot_sha256,
                "snapshot": dict(self.snapshot)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Revision:
        return cls(int(d["revision"]), str(d["created_at"]), str(d["action"]), str(d.get("actor") or ""),
                   str(d.get("note") or ""), str(d.get("status") or ""), str(d.get("snapshot_sha256") or ""),
                   dict(d.get("snapshot") or {}))


# ==================================================================================================== assessment
@dataclass(frozen=True)
class Assessment:
    """The state of a case, computed from its own contents: findings, maturity on both axes, status and the reason."""
    status: str
    reason: str
    findings: tuple[Finding, ...]
    maturity: MaturityAssessment
    status_source: str = "evidence"          # "technical" | "authorization" | "evidence"
    approved_scope: Mapping[str, Any] | None = None
    notes: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.status == FAIL)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.status == WARNING)

    @property
    def not_run(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.status == NOT_RUN)

    def render(self) -> str:
        lines = [f"Status: {self.status.upper()} (from {self.status_source})", f"  {self.reason}", "",
                 "Findings:"]
        lines += [f.render() for f in self.findings] or ["  (none)"]
        lines += ["", self.maturity.render(), ""]
        if self.notes:
            lines += [f"note: {n}" for n in self.notes]
        if self.approved_scope:
            lines.append("approved scope: " + ", ".join(f"{k}={v}" for k, v in dict(self.approved_scope).items()))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "status_source": self.status_source,
                "findings": [f.to_dict() for f in self.findings], "maturity": self.maturity.to_dict(),
                "approved_scope": None if self.approved_scope is None else dict(self.approved_scope),
                "notes": list(self.notes)}


# ==================================================================================================== the case
@dataclass(frozen=True)
class ValidationCase:
    """One validation case. Immutable: every change returns a new case with a new :class:`Revision` appended."""
    case_id: str
    profile: ValidationProfile
    model: ModelRef
    problem: ProblemRef
    requested_use: str = ""
    checks: tuple[CheckRecord, ...] = ()
    references: tuple[ReferenceComparison, ...] = ()
    uncertainty: UncertaintyReport | None = None
    applicability: ApplicabilityFacts | None = None
    provenance: Provenance = field(default_factory=Provenance)
    controls: ProcessControls = field(default_factory=ProcessControls)
    limitations: tuple[str, ...] = ()
    reviews: tuple[ExpertReview, ...] = ()
    opened_at: str = ""
    history: tuple[Revision, ...] = ()
    title: str = ""

    # ------------------------------------------------------------------ creation / evolution
    @classmethod
    def open(cls, *, profile: ValidationProfile, model: ModelRef, problem: ProblemRef, requested_use: str = "",
             actor: str = "", title: str = "") -> ValidationCase:
        """A new case, sealed as revision 1. ``case_id`` derives from the profile, the model and the problem, so the
        same case re-opened with the same inputs is recognisable."""
        body = {"profile": profile.to_dict(), "model": model.to_dict(), "problem": problem.to_dict(),
                "requested_use": requested_use}
        case = cls(case_id="VC-" + sha256_json(body)[:10], profile=profile, model=model, problem=problem,
                   requested_use=requested_use, title=title or f"{model.name} on {problem.description}",
                   opened_at=utc_now())
        return case._sealed("case opened", actor, "")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": CASE_SCHEMA, "case_id": self.case_id, "title": self.title, "opened_at": self.opened_at,
            "requested_use": self.requested_use, "profile": self.profile.to_dict(), "model": self.model.to_dict(),
            "problem": self.problem.to_dict(), "provenance": self.provenance.to_dict(),
            "controls": self.controls.to_dict(), "checks": [c.to_dict() for c in self.checks],
            "references": [r.to_dict() for r in self.references],
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_dict(),
            "applicability": None if self.applicability is None else self.applicability.to_dict(),
            "limitations": list(self.limitations), "reviews": [r.to_dict() for r in self.reviews],
        }

    def _sealed(self, action: str, actor: str, note: str) -> ValidationCase:
        snapshot = self._body()
        snapshot["status"] = self.assess().status
        rev = Revision(revision=len(self.history) + 1, created_at=utc_now(), action=action, actor=actor, note=note,
                       status=snapshot["status"], snapshot_sha256=sha256_json(snapshot), snapshot=snapshot)
        return replace(self, history=self.history + (rev,))

    def _evolve(self, action: str, actor: str = "", note: str = "", **changes: Any) -> ValidationCase:
        return replace(self, **changes)._sealed(action, actor, note)

    def with_checks(self, checks: Sequence[CheckRecord], *, action: str = "checks recorded", actor: str = "",
                    note: str = "") -> ValidationCase:
        """Attach the verdicts of a verification run (they replace the previous set, which stays in the history)."""
        return self._evolve(action, actor, note, checks=tuple(checks))

    def with_reference(self, reference: ReferenceComparison, *, actor: str = "", note: str = "") -> ValidationCase:
        if any(r.evidence_id == reference.evidence_id for r in self.references):
            raise ValueError(f"evidence {reference.evidence_id!r} is already in this case: evidence is never "
                             "overwritten, use a new evidence_id for a new comparison")
        ref = replace(reference, created_at=reference.created_at or utc_now())
        return self._evolve(f"evidence added: {ref.evidence_id}", actor, note, references=self.references + (ref,))

    def with_uncertainty(self, uncertainty: UncertaintyReport, *, actor: str = "") -> ValidationCase:
        return self._evolve("uncertainty recorded", actor, "", uncertainty=uncertainty)

    def with_applicability(self, applicability: ApplicabilityFacts, *, actor: str = "") -> ValidationCase:
        return self._evolve("applicability recorded", actor, "", applicability=applicability)

    def with_provenance(self, provenance: Provenance, *, actor: str = "") -> ValidationCase:
        return self._evolve("provenance recorded", actor, "", provenance=provenance)

    def with_controls(self, controls: ProcessControls, *, actor: str = "") -> ValidationCase:
        return self._evolve("process controls recorded", actor, "", controls=controls)

    def with_limitations(self, limitations: Sequence[str], *, actor: str = "") -> ValidationCase:
        return self._evolve("limitations recorded", actor, "", limitations=tuple(limitations))

    def with_review(self, review: ExpertReview, *, actor: str = "") -> ValidationCase:
        """Record an expert decision as a new revision (previous decisions stay in the history)."""
        return self._evolve(f"expert review: {review.decision}", actor or review.reviewer, review.justification[:200],
                            reviews=self.reviews + (review,))

    # ------------------------------------------------------------------ evidence helpers
    def evidence_ids(self) -> tuple[str, ...]:
        ids = [r.evidence_id for r in self.references] + [c.check_id for c in self.checks if c.ran]
        return tuple(dict.fromkeys(ids))

    def critical_not_assessed(self) -> tuple[str, ...]:
        """Declared critical quantities/regions that no evidence in the case covers (a reference comparison whose
        measured quantity is the critical one counts as covering it)."""
        declared = list(self.profile.critical_quantities) + list(self.profile.critical_regions)
        if not declared:
            return ()
        covered = {c.lower() for r in self.references for c in (*r.covers, r.metric)}
        checked = " ".join(c.check_id.lower() for c in self.checks if c.ran)
        return tuple(q for q in declared if q.lower() not in covered and q.lower() not in checked)

    def independent_references(self) -> tuple[ReferenceComparison, ...]:
        return tuple(r for r in self.references if r.independent)

    # ------------------------------------------------------------------ facts and assessment
    def facts(self) -> dict[str, bool]:
        """Every fact the maturity assessment reads, with the value this case actually supports."""
        ran = [c for c in self.checks if c.ran]
        residual = any(c.ran and ("residual" in c.check_id.lower() or "pde" in c.check_id.lower()) for c in self.checks)
        conditions = any(c.ran and any(w in c.check_id.lower() for w in ("boundary", "condition", "dirichlet",
                                                                         "initial")) for c in self.checks)
        indep = self.independent_references()
        tol = float(self.profile.target_error_pct)
        primary_ok = any(r.within(tol) for r in indep) and bool(indep)
        uq = self.uncertainty
        app = self.applicability
        approving = [r for r in self.reviews if r.approved]
        facts = {
            "usecase.profile_defined": not self.profile.problems(),
            "model.prediction_recorded": bool(ran) or bool(self.references),
            "verification.residual_executed": residual,
            "verification.conditions_executed": conditions,
            "validation.independent_reference_recorded": bool(indep),
            "validation.error_reported": bool(self.references),
            "validation.primary_error_within_tolerance": primary_ok,
            "validation.critical_regions_addressed": not self.critical_not_assessed(),
            "validation.critical_quantities_assessed": not self.critical_not_assessed(),
            "reproducibility.provenance_complete": self.provenance.complete,
            "generalization.domain_measured": bool(app is not None and app.checked),
            "uncertainty.quantified": bool(uq is not None and uq.quantified),
            "robustness.sensitivity_measured": any(c.ran and "sensitiv" in c.check_id.lower() for c in self.checks),
            "limitations.documented": bool(self.limitations),
            "process.second_expert_review": len({r.reviewer for r in approving}) >= 2,
            "applicability.within_domain": bool(app is not None and app.within_domain),
            "applicability.operating_condition_assessed": bool(app is not None and app.requested_condition),
            "uncertainty.risk_characterized": bool(uq is not None and uq.quantified and app is not None and app.checked),
            "process.version_recorded": bool(self.model.version and (self.model.checkpoint_sha256
                                                                     or self.model.configuration)),
            "process.expert_approval": bool(approving),
            "process.controls_documented": self.controls.documented,
            "process.approved_scope": any(r.approved and r.scope for r in self.reviews),
        }
        return facts

    def assess(self) -> Assessment:
        """Findings, maturity and status of this case, computed from its contents only."""
        findings = build_findings(
            profile=self.profile, checks=self.checks, references=self.references, uncertainty=self.uncertainty,
            applicability=self.applicability, provenance_problems=self.provenance.problems(), reviews=self.reviews,
            critical_not_assessed=self.critical_not_assessed())
        maturity = assess_maturity(self.facts())
        status, reason, source, scope, notes = self.decide(findings)
        return Assessment(status=status, reason=reason, findings=tuple(findings), maturity=maturity,
                          status_source=source, approved_scope=scope, notes=tuple(notes))

    def decide(self, findings: Sequence[Finding]) -> tuple[str, str, str, Mapping[str, Any] | None, list[str]]:
        """The status policy, in documented precedence order. A reviewer cannot overwrite a technical failure."""
        notes: list[str] = []
        technical = [f for f in findings if f.status == FAIL and f.category in ("verification", "physics", "validation",
                                                                               "applicability")]
        if technical:
            return (FAILED, "mandatory criterion not met: " + "; ".join(f"{f.finding_id} ({f.title})" for f in technical),
                    "technical", None, notes)

        last = self.reviews[-1] if self.reviews else None
        if last is not None and last.decision == "rejected":
            return (FAILED, f"use refused by {last.reviewer} at {last.decided_at}: {last.justification} -- an "
                            "authorization decision, not a technical failure", "authorization", None, notes)

        indep = self.independent_references()
        missing_mandatory = [c.check_id for c in self.checks if c.critical and not c.ran]
        missing_required = [k for k in self.profile.required_checks if k not in {c.check_id for c in self.checks}]
        nothing_ran = not any(c.ran for c in self.checks) and not self.references
        if nothing_ran:
            return (UNVALIDATED, "no check and no reference comparison is recorded for this case: with no evidence "
                                 "there is nothing to validate", "evidence", None, notes)
        if (self.profile.needs_reference_at_all and not indep) or missing_mandatory or missing_required:
            why = []
            if self.profile.needs_reference_at_all and not indep:
                why.append(f"the profile ({self.profile.objective}) requires an independent reference and none is "
                           "recorded")
            if missing_mandatory:
                why.append(f"mandatory check(s) did not run: {', '.join(missing_mandatory)}")
            if missing_required:
                why.append(f"check(s) required by the profile have no record: {', '.join(missing_required)}")
            return (UNVALIDATED, "evidence required by the declared use is absent -- " + "; ".join(why), "evidence",
                    None, notes)

        flagged = [f for f in findings if f.status in (WARNING, NOT_RUN)]
        if last is None:
            notes += [f"{f.finding_id} {f.title}: {f.what_happened}" for f in flagged][:8]
            return (REVIEW_REQUIRED, "automated requirements hold, but no expert decision is recorded: checks alone do "
                                     "not authorize use", "evidence", None, notes)
        if last.decision in ("more evidence requested", "suspended"):
            return (REVIEW_REQUIRED, f"{last.reviewer} recorded '{last.decision}' at {last.decided_at}: the decision "
                                     "is open until the requested evidence exists", "authorization", None, notes)
        if not last.covers(self.requested_use):
            return (REVIEW_REQUIRED, f"the approval by {last.reviewer} does not cover the requested use "
                                     f"'{self.requested_use}' (scope: {_dict(last.scope)}): a validated scope is not a "
                                     "validated use outside it", "authorization", None, notes)
        notes += [f"{f.finding_id} {f.title}: {f.what_happened}" for f in flagged][:8]
        return (VALIDATED, f"approved by {last.reviewer} at {last.decided_at} for the scope recorded in the review: "
                           f"{_dict(last.scope) or 'the case as assessed'}", "authorization", _dict(last.scope), notes)

    # ------------------------------------------------------------------ audit
    def history_problems(self) -> list[str]:
        """Why the stored history cannot be trusted (empty when every revision still verifies)."""
        out = []
        for i, rev in enumerate(self.history, 1):
            if rev.revision != i:
                out.append(f"revision {rev.revision} is out of sequence at position {i}")
            if sha256_json(rev.snapshot) != rev.snapshot_sha256:
                out.append(f"revision {rev.revision} ({rev.action}) was altered after it was sealed")
            if rev.snapshot.get("case_id") not in (None, self.case_id):
                out.append(f"revision {rev.revision} belongs to another case")
        return out

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        body["history"] = [r.to_dict() for r in self.history]
        body["assessment"] = self.assess().to_dict()
        body["case_sha256"] = sha256_json(body)
        return body

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ValidationCase:
        case = cls(case_id=str(d["case_id"]), profile=ValidationProfile.from_dict(d["profile"]),
                   model=ModelRef.from_dict(d["model"]), problem=ProblemRef.from_dict(d["problem"]),
                   requested_use=str(d.get("requested_use") or ""), title=str(d.get("title") or ""),
                   opened_at=str(d.get("opened_at") or ""),
                   checks=tuple(CheckRecord.from_dict(c) for c in d.get("checks") or ()),
                   references=tuple(ReferenceComparison.from_dict(r) for r in d.get("references") or ()),
                   uncertainty=None if not d.get("uncertainty") else
                   UncertaintyReport.from_dict(d["uncertainty"]),
                   applicability=None if not d.get("applicability") else
                   ApplicabilityFacts.from_dict(d["applicability"]),
                   provenance=Provenance.from_dict(d.get("provenance") or {}),
                   controls=ProcessControls(**{k: str(v) for k, v in (d.get("controls") or {}).items()
                                               if k in ProcessControls.__dataclass_fields__}),
                   limitations=tuple(d.get("limitations") or ()),
                   reviews=tuple(ExpertReview.from_dict(r) for r in d.get("reviews") or ()),
                   history=tuple(Revision.from_dict(r) for r in d.get("history") or ()))
        return case


# ==================================================================================================== registry
@dataclass
class ValidationRegistry:
    """The evidence registry of a team: every case (latest revision), queried by model, by status and by use, plus the
    improvement tasks derived from the gaps -- so a confirmed failure or a missing evidence becomes an action rather
    than a footnote."""
    cases: dict[str, ValidationCase] = field(default_factory=dict)

    def add(self, case: ValidationCase) -> ValidationCase:
        """Store a case. A new revision of a known case replaces its previous state; the case itself carries the
        history of both."""
        self.cases[case.case_id] = case
        return case

    def get(self, case_id: str) -> ValidationCase:
        if case_id not in self.cases:
            raise KeyError(f"no case {case_id!r}; known: {sorted(self.cases)}")
        return self.cases[case_id]

    def by_status(self, status: str) -> list[ValidationCase]:
        return [c for c in self.cases.values() if c.assess().status == status]

    def by_model(self, name: str) -> list[ValidationCase]:
        return [c for c in self.cases.values() if c.model.name == name]

    def evidence_rows(self) -> list[dict[str, Any]]:
        """Flat view of every piece of evidence, with the case and verdict it belongs to (the auditable index)."""
        rows = []
        for c in self.cases.values():
            a = c.assess()
            for r in c.references:
                rows.append({"case_id": c.case_id, "model": c.model.name, "model_version": c.model.version,
                             "evidence_id": r.evidence_id, "kind": r.kind, "metric": r.metric,
                             "value_pct": r.value_pct, "tolerance_pct": r.tolerance_pct,
                             "within_tolerance": r.within(float(c.profile.target_error_pct)),
                             "artefact_sha256": r.artefact_sha256, "case_status": a.status,
                             "decided_use": c.requested_use})
            for ck in c.checks:
                rows.append({"case_id": c.case_id, "model": c.model.name, "model_version": c.model.version,
                             "evidence_id": ck.check_id, "kind": f"check ({ck.category})",
                             "metric": ck.check_id, "value_pct": None, "tolerance_pct": ck.threshold,
                             "within_tolerance": ck.passed, "artefact_sha256": ck.evidence,
                             "case_status": a.status, "decided_use": c.requested_use})
        return rows

    def tasks(self) -> list[dict[str, Any]]:
        """Improvement tasks, derived deterministically from what each case lacks (never invented): the open maturity
        requirements, a refused use, and the evidence a reviewer asked for."""
        out: list[dict[str, Any]] = []
        for c in self.cases.values():
            a = c.assess()
            for row in a.maturity.improvement_actions():
                level = int(row["level"])
                out.append({"task_id": f"{c.case_id}/{row['requirement']}", "case_id": c.case_id,
                            "source": f"maturity requirement {row['requirement']}",
                            "action": row["action"], "statement": row["statement"],
                            "priority": "high" if level <= 2 else "medium" if level == 3 else "low",
                            "status": "open"})
            for f in a.failures:
                out.append({"task_id": f"{c.case_id}/{f.rule}", "case_id": c.case_id,
                            "source": "finding " + f.finding_id,
                            "action": f.actions[0] if f.actions else "investigate and record the outcome",
                            "statement": f.title,
                            "priority": "high", "status": "open"})
            last = c.reviews[-1] if c.reviews else None
            if last is not None and last.decision == "more evidence requested":
                out.append({"task_id": f"{c.case_id}/review-{last.review_id}", "case_id": c.case_id,
                            "source": f"review {last.review_id}", "action": last.justification,
                            "statement": f"{last.reviewer} requested more evidence",
                            "priority": "high", "status": "open"})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"schema": REGISTRY_SCHEMA, "cases": [c.to_dict() for c in self.cases.values()]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ValidationRegistry:
        reg = cls()
        for row in d.get("cases") or ():
            reg.add(ValidationCase.from_dict(row))
        return reg

    def save(self, path: str) -> str:
        """Write the registry as JSON (cases keep their full history) and return the file's SHA-256."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = canonical_json(self.to_dict())
        p.write_text(text, encoding="utf-8")
        return sha256_json(self.to_dict())

    @classmethod
    def load(cls, path: str) -> ValidationRegistry:
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ==================================================================================================== converters
def checks_from_trust_card(card: Any) -> tuple[CheckRecord, ...]:
    """The checks of ``pinneapple_analysis.trust.trust_card`` (a ``FieldTrustCard``, its ``.checks`` list, or their
    ``to_dict()`` form) as :class:`CheckRecord`."""
    checks = getattr(card, "checks", card)
    out = []
    for c in checks:
        d = c.to_dict() if hasattr(c, "to_dict") else dict(c)
        detail = d.get("detail") or d.get("reason") or ""
        if d.get("passed") is None and d.get("status") in ("supports", "contradicts", "not_run"):
            passed = {"supports": True, "contradicts": False}.get(d["status"])
        else:
            passed = d.get("passed")
        value = d.get("value")
        threshold = d.get("threshold", d.get("limit"))
        out.append(CheckRecord(check_id=str(d.get("name") or d.get("check_id") or "check"),
                               category=_KIND_TO_CATEGORY.get(str(d.get("kind") or ""), "other"), passed=passed,
                               value=None if value is None else float(value),
                               threshold=None if threshold is None else float(threshold), message=str(detail),
                               critical=bool(d.get("critical", False))))
    return tuple(out)


def checks_from_lab_report(report: Mapping[str, Any] | Any) -> tuple[CheckRecord, ...]:
    """The outcomes of a Validation Lab run (``LabReport.to_dict()``, or a ``LabReport``) as :class:`CheckRecord`."""
    d = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    out = []
    for o in d.get("outcomes", ()):
        cat = str(o.get("category") or "other")
        out.append(CheckRecord(check_id=str(o.get("check_id") or f"{o.get('rule')}/{o.get('subject')}"),
                               category=_KIND_TO_CATEGORY.get(cat, cat if cat in CHECK_CATEGORIES else "other"),
                               passed=bool(o.get("passed")) if o.get("passed") is not None else None,
                               value=_as_float(o.get("value")), threshold=_as_float(o.get("threshold")),
                               message=str(o.get("message") or o.get("title") or ""),
                               critical=str(o.get("severity")) == "error"))
    return tuple(out)


def checks_from_ledger(ledger: EvidenceLedger) -> tuple[CheckRecord, ...]:
    """The latest valid entry of every check in an :class:`~pinneapple_veriphysics.ledger.EvidenceLedger`. A check with
    no valid entry is reported once as "not run" -- never dropped."""
    out, seen = [], set()
    for e in ledger.entries:
        latest = ledger.latest(e.check_id)
        if latest is None:
            if e.check_id not in seen:
                seen.add(e.check_id)
                out.append(CheckRecord(check_id=e.check_id, category=e.category or "other", passed=None,
                                       message="record invalid: " + "; ".join(e.problems())))
            continue
        if latest.check_id in seen:
            continue
        seen.add(latest.check_id)
        outputs = latest.outputs or {}
        out.append(CheckRecord(check_id=latest.check_id, category=latest.category or "other", passed=latest.passed,
                               value=_as_float(outputs.get("value")), threshold=_as_float(outputs.get("threshold")),
                               message=str(outputs.get("detail") or outputs.get("message") or ""),
                               evidence=latest.entry_sha256))
    return tuple(out)


def _as_float(v: Any) -> float | None:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def applicability_facts_from_map(amap: Mapping[str, Any] | Any, query: Mapping[str, float] | None = None, *,
                                 requested_condition: str = "") -> ApplicabilityFacts:
    """Extrapolation detection from an ``applicability.build_applicability_map(...).to_dict()`` and the inputs the use
    actually asks for: every queried variable outside its verified range lands in ``outside_variables``.

    The range is widened by the *measured* extrapolation margin when the study established one; without a measured
    margin nothing outside the verified range is supported.
    """
    d = amap.to_dict() if hasattr(amap, "to_dict") else dict(amap)
    envelope = d.get("envelope") or ()
    supported = None
    for v in envelope:                                    # the margin recorded by an extrapolation study, if any
        note = str(v.get("note") or "")
        if "up to +" in note:
            try:
                supported = max(supported or 0.0, float(note.split("up to +", 1)[1].split("%", 1)[0]) / 100.0)
            except ValueError:
                continue
    ranges, outside, notes = [], [], []
    by_name = {str(v.get("name")): v for v in envelope}
    for name, q in (query or {}).items():
        env = by_name.get(str(name))
        if env is None:
            notes.append(f"{name}: no verified range recorded, so the query cannot be shown to be inside it")
            continue
        rng, point, extrap = env.get("verified_range"), env.get("verified_point"), bool(env.get("extrapolation_supported"))
        if rng:
            lo, hi = float(rng[0]), float(rng[1])
            ranges.append((str(name), lo, hi))
            margin = (supported or 0.0) * (hi - lo)
            if not (lo - margin <= float(q) <= hi + margin):
                outside.append(str(name))
        elif point is not None:
            if abs(float(q) - float(point)) > 1e-12 and not extrap:
                outside.append(str(name))
    if outside:
        notes.append("outside the verified range: " + ", ".join(outside)
                     + ("" if supported else "; no measured extrapolation margin is recorded, so nothing beyond the "
                                                 "verified range is supported"))
    return ApplicabilityFacts(checked=bool(envelope), requested_condition=requested_condition,
                              outside_variables=tuple(outside), supported_margin=supported,
                              verified_ranges=tuple(ranges), notes=tuple(notes))
