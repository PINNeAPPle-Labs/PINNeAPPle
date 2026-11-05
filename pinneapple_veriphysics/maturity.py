"""Maturity and gap assessment: how far the evidence reaches, on two independent axes.

A single score hides what matters. A model can be excellent numerically and impossible to reproduce, or match a
benchmark and never have been evaluated in the conditions it will be operated in. So the assessment is a
multidimensional profile on two axes that are not forced to agree:

* **scientific maturity** -- quality of the evidence, methodological rigour, reproducibility and physical validation;
* **industrial readiness** -- fitness for the declared use, operational risk, controls and human authorization.

Five levels per axis (:data:`LEVELS`), each defined by *verifiable* requirements (:data:`REQUIREMENTS`) that facts of
the case either show or do not. A level is reached only when that level and every level below it are met: a good
average metric cannot lift a case past a missing reference, a missing reproducibility record or a missing review. The
scale is this package's own starting scale -- not an academic or industrial standard, and not a certification.

::

    from pinneapple_veriphysics.maturity import assess_maturity

    a = assess_maturity({"verification.residual_executed": True, "usecase.profile_defined": True, ...})
    print(a.render())                     # per axis: level reached, next level, the gaps with how to close each
    print(a.level("scientific"), a.gaps("scientific"))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = ["DIMENSIONS", "LEVELS", "MaturityRequirement", "REQUIREMENTS", "AxisAssessment", "MaturityAssessment",
           "assess_maturity", "requirements_for", "FACT_DESCRIPTIONS"]

#: The two axes. They are assessed separately on purpose.
DIMENSIONS = ("scientific", "industrial")

LEVELS: dict[str, dict[int, tuple[str, str]]] = {
    "scientific": {
        1: ("Exploratory", "A model runs and produces results; coverage of tests and evidence is limited."),
        2: ("Verified", "The computational checks, the boundary conditions and the applicable physical properties "
                        "were evaluated."),
        3: ("Validated for research", "An independent reference, an error analysis, reproducibility and evidence "
                                      "sustain the specific conclusions of the study."),
        4: ("Generalisation characterised", "The limits of the validated domain, the uncertainty and the sensitivity "
                                            "to modelling choices were measured on representative scenarios."),
        5: ("Rigorously evidenced", "Two independent kinds of evidence or a second expert review, and the known "
                                    "limitations stated, support the conclusions."),
    },
    "industrial": {
        1: ("Exploratory", "The intended use, the quantity that decides it, its tolerance and the consequence of "
                           "being wrong are declared."),
        2: ("Verified", "The computational checks, the boundary conditions and the applicable physical properties "
                        "were evaluated."),
        3: ("Application-qualified", "The declared acceptance criterion is met on the decision quantity, the critical "
                                     "region is assessed and the use sits inside the verified domain."),
        4: ("Operationally assessed", "The operating condition itself was assessed, the uncertainty and the "
                                      "consequence of a wrong decision are characterised, and the model version is "
                                      "recorded."),
        5: ("Ready for controlled operation", "An authorised expert approved a documented scope, and monitoring, "
                                              "fallback and change control are recorded."),
    },
}


@dataclass(frozen=True)
class MaturityRequirement:
    """One verifiable requirement of one level of one axis. ``met_by`` is the key of the fact that shows it (see
    :func:`assess_maturity`); ``how_to_meet`` is what produces that fact."""
    key: str
    dimension: str
    level: int
    statement: str
    met_by: str
    how_to_meet: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "dimension": self.dimension, "level": self.level, "statement": self.statement,
                "met_by": self.met_by, "how_to_meet": self.how_to_meet, "met": None}


_R = MaturityRequirement
REQUIREMENTS: tuple[MaturityRequirement, ...] = (
    # ---------------------------------------------------------------- scientific
    _R("sci.model_runs", "scientific", 1, "A prediction was produced and recorded with the model version it came from.",
       "model.prediction_recorded", "record the run, its model version and its outputs"),
    _R("sci.residual", "scientific", 2, "The governing-equation residual was re-computed on the prediction.",
       "verification.residual_executed", "run the residual check on fresh points"),
    _R("sci.conditions", "scientific", 2, "Boundary and/or initial conditions were checked against the prediction.",
       "verification.conditions_executed", "add the boundary/initial condition check"),
    _R("sci.reference", "scientific", 3, "A comparison against an independent reference is recorded.",
       "validation.independent_reference_recorded", "compute or obtain an independent reference on the same case"),
    _R("sci.error", "scientific", 3, "The error of the primary quantity against that reference is reported.",
       "validation.error_reported", "report the metric declared in the profile"),
    _R("sci.provenance", "scientific", 3, "Inputs, code, versions, seeds and tolerances are recorded so the run is "
                                          "reproducible.", "reproducibility.provenance_complete",
       "fill the provenance record (hashes, versions, seeds, tolerances)"),
    _R("sci.critical", "scientific", 3, "The critical quantity or region declared in the profile is assessed, or the "
                                        "profile declares none.", "validation.critical_regions_addressed",
       "measure the critical quantity, or assess the critical region locally"),
    _R("sci.domain", "scientific", 4, "The domain where the evidence holds was measured (applicability envelope).",
       "generalization.domain_measured", "run an extrapolation / held-out study and record the envelope"),
    _R("sci.uncertainty", "scientific", 4, "An uncertainty estimate exists (calibrated, or declared uncalibrated).",
       "uncertainty.quantified", "estimate epistemic uncertainty and state the method"),
    _R("sci.sensitivity", "scientific", 4, "Sensitivity to a modelling choice (weights, architecture, parameters) was "
                                           "measured.", "robustness.sensitivity_measured",
       "re-train with a perturbed choice and measure how much the prediction moves"),
    _R("sci.limitations", "scientific", 5, "The known limitations and what remains unknown are stated in the case.",
       "limitations.documented", "list the unsupported regions, the untested physics and the open questions"),
    _R("sci.second_review", "scientific", 5, "A second, distinct expert reviewed the evidence.",
       "process.second_expert_review", "have another qualified reviewer record a decision"),
    # ---------------------------------------------------------------- industrial
    _R("ind.usecase", "industrial", 1, "The intended use is declared: decision, decision quantity, tolerance and "
                                        "consequence.", "usecase.profile_defined",
       "complete the validation profile"),
    _R("ind.residual", "industrial", 2, "The governing-equation residual was re-computed on the prediction.",
       "verification.residual_executed", "run the residual check on fresh points"),
    _R("ind.conditions", "industrial", 2, "Boundary and/or initial conditions were checked against the prediction.",
       "verification.conditions_executed", "add the boundary/initial condition check"),
    _R("ind.tolerance", "industrial", 3, "The error of the decision quantity is inside the tolerance the use declares.",
       "validation.primary_error_within_tolerance", "compare the decision quantity against a reference and report it"),
    _R("ind.critical", "industrial", 3, "The quantities and regions the decision depends on are assessed, or the "
                                        "profile declares none.", "validation.critical_quantities_assessed",
       "measure the decision quantity where the decision is made"),
    _R("ind.domain", "industrial", 3, "The requested condition of use is inside the verified domain.",
       "applicability.within_domain", "restrict the scope, or produce evidence covering the requested condition"),
    _R("ind.operating", "industrial", 4, "The operating condition of the intended use was itself assessed.",
       "applicability.operating_condition_assessed", "evaluate the case at the conditions it will be used in"),
    _R("ind.risk", "industrial", 4, "The uncertainty and the consequence of a wrong decision are characterised.",
       "uncertainty.risk_characterized", "quantify uncertainty and relate it to the consequence declared"),
    _R("ind.version", "industrial", 4, "The model version and its checkpoint are recorded with the evidence.",
       "process.version_recorded", "record the version and the checkpoint hash"),
    _R("ind.approval", "industrial", 5, "An authorised expert approved the use, with justification and scope.",
       "process.expert_approval", "record an expert decision"),
    _R("ind.controls", "industrial", 5, "Monitoring, the fallback procedure and change control are documented.",
       "process.controls_documented", "document what is monitored, the fallback and the change-control rule"),
    _R("ind.scope", "industrial", 5, "The approved scope is documented with the conditions it applies to.",
       "process.approved_scope", "record the conditions and quantities the approval covers"),
)

#: One line per fact, so a report can say what a missing fact means.
FACT_DESCRIPTIONS: dict[str, str] = {
    "usecase.profile_defined": "the validation profile is complete",
    "model.prediction_recorded": "a prediction with its model version is recorded",
    "verification.residual_executed": "the governing-equation residual check ran",
    "verification.conditions_executed": "the boundary/initial condition check ran",
    "validation.independent_reference_recorded": "an independent reference comparison is recorded",
    "validation.error_reported": "the error of the primary quantity is reported",
    "validation.primary_error_within_tolerance": "the error of the decision quantity is inside the declared tolerance",
    "validation.critical_regions_addressed": "the critical region declared in the profile is assessed",
    "validation.critical_quantities_assessed": "the critical quantities declared in the profile are assessed",
    "reproducibility.provenance_complete": "the provenance record is complete",
    "generalization.domain_measured": "the domain where the evidence holds was measured",
    "uncertainty.quantified": "an uncertainty estimate exists",
    "robustness.sensitivity_measured": "sensitivity to a modelling choice was measured",
    "limitations.documented": "the known limitations are stated",
    "process.second_expert_review": "a second expert reviewed the evidence",
    "applicability.within_domain": "the requested condition is inside the verified domain",
    "applicability.operating_condition_assessed": "the operating condition of use was assessed",
    "uncertainty.risk_characterized": "the uncertainty and the consequence of failure are characterised",
    "process.version_recorded": "the model version and checkpoint are recorded",
    "process.expert_approval": "an expert approved the use",
    "process.controls_documented": "monitoring, fallback and change control are documented",
    "process.approved_scope": "the approved scope is recorded",
}


def requirements_for(dimension: str, level: int | None = None) -> tuple[MaturityRequirement, ...]:
    """The requirements of one axis (all levels, or one)."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}, got {dimension!r}")
    return tuple(r for r in REQUIREMENTS if r.dimension == dimension and (level is None or r.level == level))


@dataclass(frozen=True)
class AxisAssessment:
    """Where one axis stands: the level reached, the one that is next, and what is missing for it."""
    dimension: str
    level: int                                    # highest level whose requirements (and all below) are met
    next_level: int | None                        # None once level 5 is reached
    met: tuple[str, ...] = ()                     # requirement keys met
    gaps: tuple[MaturityRequirement, ...] = ()            # unmet requirements of the next level
    total_open: tuple[MaturityRequirement, ...] = ()      # every unmet requirement of the axis

    @property
    def title(self) -> str:
        return LEVELS[self.dimension][self.level][0] if self.level >= 1 else "Not assessed"

    @property
    def next_title(self) -> str | None:
        return None if self.next_level is None else LEVELS[self.dimension][self.next_level][0]

    def render(self) -> str:
        head = f"{self.dimension.capitalize()} maturity: level {self.level} ({self.title})"
        if self.next_level is None:
            return head + " -- the highest level of this scale, within the scope assessed."
        lines = [head, f"  To reach level {self.next_level} ({self.next_title}):"]
        lines += [f"    - {g.statement} -> {g.how_to_meet}" for g in self.gaps] or ["    - (nothing missing)"]
        later = [r for r in self.total_open if r.level > self.next_level]
        if later:
            lines.append(f"  Beyond that, {len(later)} more requirement(s) up to level "
                         f"{max(r.level for r in later)} remain open.")
        return "\n".join(lines)


@dataclass(frozen=True)
class MaturityAssessment:
    """The two axes, assessed independently, plus the facts they were read from."""
    axes: dict[str, AxisAssessment]
    facts: dict[str, bool] = field(default_factory=dict)

    def level(self, dimension: str) -> int:
        return self.axes[dimension].level

    def gaps(self, dimension: str) -> tuple[MaturityRequirement, ...]:
        return self.axes[dimension].gaps

    def open_requirements(self) -> tuple[MaturityRequirement, ...]:
        return tuple(r for a in self.axes.values() for r in a.total_open)

    def improvement_actions(self) -> list[dict[str, Any]]:
        """The open requirements as concrete, ordered actions (a lower level first)."""
        rows = sorted(self.open_requirements(), key=lambda r: (r.level, r.dimension, r.key))
        return [{"requirement": r.key, "dimension": r.dimension, "level": r.level, "action": r.how_to_meet,
                 "statement": r.statement,
                 "priority": "high" if r.level <= 2 else "medium" if r.level == 3 else "low"} for r in rows]

    def render(self) -> str:
        return "\n".join(a.render() for a in self.axes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": {k: {"dimension": a.dimension, "level": a.level, "title": a.title, "next_level": a.next_level,
                         "next_title": a.next_title, "met": list(a.met),
                         "gaps": [g.to_dict() for g in a.gaps],
                         "open": [{"key": r.key, "level": r.level, "statement": r.statement,
                                   "how_to_meet": r.how_to_meet} for r in a.total_open]}
                     for k, a in self.axes.items()},
            "facts": dict(self.facts),
            "levels": {d: {lvl: {"title": t, "definition": txt} for lvl, (t, txt) in LEVELS[d].items()}
                       for d in DIMENSIONS},
        }


def _levels_for(dimension: str, facts: Mapping[str, bool]) -> AxisAssessment:
    reqs = requirements_for(dimension)
    met_keys = {r.key for r in reqs if bool(facts.get(r.met_by, False))}
    level = 0
    for lvl in sorted(LEVELS[dimension]):
        for_lvl = [r for r in reqs if r.level == lvl]
        if all(r.key in met_keys for r in for_lvl):
            level = lvl
        else:
            break
    next_level = level + 1 if level < max(LEVELS[dimension]) else None
    gaps = tuple(r for r in reqs if r.level == next_level and r.key not in met_keys) if next_level else ()
    open_all = tuple(r for r in reqs if r.key not in met_keys)
    return AxisAssessment(dimension=dimension, level=level, next_level=next_level,
                          met=tuple(r.key for r in reqs if r.key in met_keys), gaps=gaps, total_open=open_all)


def assess_maturity(facts: Mapping[str, bool] | Iterable[str]) -> MaturityAssessment:
    """Assess both axes from the facts of a case (``{fact key: True/False}``, or the keys that hold).

    A fact that is absent is false: nothing is assumed to have happened because it usually does.
    """
    f = {k: True for k in facts} if not isinstance(facts, Mapping) else {str(k): bool(v) for k, v in facts.items()}
    return MaturityAssessment(axes={d: _levels_for(d, f) for d in DIMENSIONS}, facts=f)
