"""Use-case specification: what "success" means, before any verification runs.

A model is not good or bad in the abstract. A surrogate whose error is 8 % is useless for a paper that reports 0.5 % and
may be perfectly adequate to triage industrial designs -- and a low *mean* error can still be unacceptable when it
sits in the region the decision depends on. So the first object of a validation case is not a test but a declaration:
the intended use, the quantity that decides it, the tolerance that use needs, which independent reference exists, and
what happens if the decision is wrong.

:class:`ValidationProfile` carries that declaration. Everything here is *configured by the caller*: the tolerance is
a requirement, not a measurement, and choosing one never proves that a model meets it. :data:`OBJECTIVES` records
starting points that practitioners commonly quote (a fraction of a percent for a rigorous research comparison, single
digits to low teens of a percent for engineering triage) and says so -- they are illustrative, not norms, and not a
substitute for the tolerance the intended use actually requires.

::

    from pinneapple_veriphysics.usecase import PROFILE_PRESETS

    profile = PROFILE_PRESETS["industrial_screening"]
    print(profile.describe())            # what this profile requires, and what it cannot require
    print(profile.problems())            # ["no independent reference declared: ..."] when incomplete
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

__all__ = ["OBJECTIVES", "REFERENCE_KINDS", "CONSEQUENCES", "ValidationProfile", "PROFILE_PRESETS", "profile_presets"]

#: What the result is going to be used for. ``tolerance_pct`` is an *illustrative starting point* (read from common
#: practice, never from a standard) and ``needs_reference`` says whether a claim of validation is even meaningful.
#: ``note`` is rendered to the user next to the value: the number is a placeholder until the use sets its own.
OBJECTIVES: dict[str, dict[str, Any]] = {
    "academic": {
        "title": "Rigorous academic research",
        "tolerance_pct": 0.5,
        "needs_reference": True,
        "note": ("A research comparison is stated against an independent reference and a stated error; fractions of a "
                 "percent are common in benchmark papers, but the number follows from the quantity, the reference "
                 "and the discretisation actually used -- not from this table."),
    },
    "industrial": {
        "title": "Industrial application",
        "tolerance_pct": 10.0,
        "needs_reference": True,
        "note": ("Industrial acceptance depends on the decision the number feeds, the consequence of being wrong and "
                 "the design margin: single-digit to low-teens percent is a common starting range, never a rule. A "
                 "high-consequence use needs its tolerance argued, not inherited from here."),
    },
    "exploratory": {
        "title": "Exploration and prototyping",
        "tolerance_pct": 15.0,
        "needs_reference": False,
        "note": ("For triage and prototyping the comparison may be against a coarser reference or a previous "
                 "campaign. Nothing validated for exploration may be reported as validated for a design decision."),
    },
}

#: The independent evidence a comparison can be made against, strongest first. ``"none"`` is allowed and honest: it
#: means no independent reference exists, so no error against reality can be measured (only internal consistency).
REFERENCE_KINDS = ("numerical reference", "experimental data", "analytic solution", "benchmark", "none")

#: Consequence of a wrong decision taken on this result. It drives how much uncertainty has to be characterised.
CONSEQUENCES = ("low", "medium", "high", "safety")


@dataclass(frozen=True)
class ValidationProfile:
    """The declared use of a result. Immutable: changing the use means a new profile, and a new case.

    ``target_error_pct``: the largest error the intended use accepts, in percent, for ``primary_metric``. It is a
    requirement. ``reference_evidence``: kinds of independent evidence the use needs (see :data:`REFERENCE_KINDS`).
    ``critical_quantities`` / ``critical_regions``: the quantities and places the decision depends on; a global mean
    error says nothing about them. ``consequence``: one of :data:`CONSEQUENCES`.
    """
    name: str
    objective: str
    primary_metric: str
    target_error_pct: float
    reference_evidence: tuple[str, ...] = ("none",)
    critical_quantities: tuple[str, ...] = ()
    critical_regions: tuple[str, ...] = ()
    required_checks: tuple[str, ...] = ()    # check ids this use treats as mandatory
    consequence: str = "medium"
    decision: str = ""                       # the decision the result feeds, in the user's words
    notes: str = ""

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {tuple(OBJECTIVES)}, got {self.objective!r}")
        if not (isinstance(self.target_error_pct, (int, float)) and 0 < float(self.target_error_pct) < 100):
            raise ValueError(f"target_error_pct must be a number in (0, 100), got {self.target_error_pct!r}")
        if not str(self.primary_metric).strip():
            raise ValueError("primary_metric is required: an error tolerance without a quantity is not a criterion")
        bad = [k for k in self.reference_evidence if k not in REFERENCE_KINDS]
        if bad:
            raise ValueError(f"unknown reference kind(s) {bad}; use {REFERENCE_KINDS}")
        if self.consequence not in CONSEQUENCES:
            raise ValueError(f"consequence must be one of {CONSEQUENCES}, got {self.consequence!r}")

    # ------------------------------------------------------------------ derived requirements
    @property
    def needs_independent_reference(self) -> bool:
        """An objective that claims validation has to be backed by something independent of the model's own loss."""
        return bool(OBJECTIVES[self.objective]["needs_reference"]) and "none" not in self.reference_evidence

    @property
    def needs_reference_at_all(self) -> bool:
        return bool(OBJECTIVES[self.objective]["needs_reference"])

    @property
    def declares_critical_scope(self) -> bool:
        return bool(self.critical_quantities or self.critical_regions)

    def problems(self) -> list[str]:
        """Why this profile cannot yet support a validation claim (empty when it can). Never a score."""
        out: list[str] = []
        if self.objective != "exploratory" and "none" in self.reference_evidence:
            out.append("no independent reference declared: with no numerical, experimental, analytic or benchmark "
                       "reference, an error against reality cannot be measured -- only internal consistency can be")
        if not self.decision.strip():
            out.append("the decision this result feeds is not stated, so the consequence of being wrong "
                       "cannot be weighed")
        if self.consequence in ("high", "safety") and not self.declares_critical_scope:
            out.append(f"consequence '{self.consequence}' but no critical quantity or region is declared: a global "
                       "metric cannot show that the region driving the decision is right")
        return out

    def describe(self) -> str:
        """Deterministic explanation of what this profile requires and what it does not claim."""
        obj = OBJECTIVES[self.objective]
        refs = ", ".join(self.reference_evidence) or "none"
        lines = [
            f"Validation profile -- {self.name}",
            f"Objective: {obj['title']} ({self.objective}); consequence of a wrong decision: {self.consequence}",
            f"Primary metric: {self.primary_metric}",
            f"Acceptance tolerance: {self.target_error_pct:g} % (declared by the caller; "
            f"{obj['tolerance_pct']:g} % is the illustrative starting point of '{self.objective}')",
            f"Reference evidence: {refs}",
        ]
        if self.declares_critical_scope:
            lines.append("Critical quantities: " + (", ".join(self.critical_quantities) or "none declared"))
            lines.append("Critical regions: " + (", ".join(self.critical_regions) or "none declared"))
        if self.decision:
            lines.append(f"Decision it feeds: {self.decision}")
        lines += ["", obj["note"]]
        if "none" in self.reference_evidence and self.objective != "exploratory":
            lines.append("Without an independent reference this case can establish internal consistency and a scope "
                         "of applicability, but it cannot be reported as validated against reality.")
        for p in self.problems():
            lines.append(f"warning: {p}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "objective": self.objective, "primary_metric": self.primary_metric,
            "target_error_pct": float(self.target_error_pct), "reference_evidence": list(self.reference_evidence),
            "critical_quantities": list(self.critical_quantities), "critical_regions": list(self.critical_regions),
            "required_checks": list(self.required_checks), "consequence": self.consequence, "decision": self.decision,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ValidationProfile:
        return cls(name=str(d["name"]), objective=str(d["objective"]), primary_metric=str(d["primary_metric"]),
                   target_error_pct=float(d["target_error_pct"]),
                   reference_evidence=tuple(d.get("reference_evidence") or ("none",)),
                   critical_quantities=tuple(d.get("critical_quantities") or ()),
                   critical_regions=tuple(d.get("critical_regions") or ()),
                   required_checks=tuple(d.get("required_checks") or ()),
                   consequence=str(d.get("consequence") or "medium"), decision=str(d.get("decision") or ""),
                   notes=str(d.get("notes") or ""))

    def with_changes(self, **changes: Any) -> ValidationProfile:
        """A new profile with ``changes`` applied (the case history then shows two profiles, not one edited one)."""
        return replace(self, **changes)


#: Ready-made starting points. They set the structure of a profile for a common situation; the tolerance and the
#: reference kinds still have to be confirmed for the actual use.
PROFILE_PRESETS: dict[str, ValidationProfile] = {
    "academic_cfd": ValidationProfile(
        name="Academic CFD comparison", objective="academic", primary_metric="relative L2 error of the field",
        target_error_pct=0.5, reference_evidence=("numerical reference", "analytic solution"), consequence="low",
        decision="report the result as a comparison against the reference in a publication"),
    "industrial_screening": ValidationProfile(
        name="Industrial screening", objective="industrial", primary_metric="relative error of the quantity of interest",
        target_error_pct=10.0, reference_evidence=("numerical reference",), consequence="medium",
        decision="use the prediction to screen candidate designs before a full simulation"),
    "exploration": ValidationProfile(
        name="Exploration", objective="exploratory", primary_metric="relative error of the field",
        target_error_pct=15.0, reference_evidence=("none",), consequence="low",
        decision="decide whether the approach is worth pursuing"),
}


def profile_presets() -> list[str]:
    """Names of the ready-made profiles (each still has to be confirmed for the actual use)."""
    return list(PROFILE_PRESETS)
