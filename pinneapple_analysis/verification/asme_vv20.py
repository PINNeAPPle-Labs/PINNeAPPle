"""ASME V&V 20 validation, and the credibility matrix that says how much is needed (#106).

The library already verifies: manufactured solutions, Richardson extrapolation and the grid convergence index give
the **numerical** uncertainty of one number. What it did not do is the step ASME V&V 20-2009 defines for validation,
which is a comparison against data with every uncertainty in the same budget:

    E        = S - D                                      the comparison error (simulation minus data)
    u_val    = sqrt(u_num**2 + u_input**2 + u_ref**2)     the validation uncertainty
    achieved: |E| <= u_val                                the model is validated at that uncertainty level

Reporting ``E`` without ``u_val`` is the mistake the standard exists to prevent: a simulation 5 % away from the data
looks wrong until you learn that the numerical and input uncertainties are 8 % together.

    >>> from pinneapple_analysis.verification.asme_vv20 import validation
    >>> r = validation(simulation=0.98, data=1.0, u_numerical=0.02, u_input=0.01, u_reference=0.01)
    >>> r.achieved, round(r.e, 3), round(r.u_val, 3)
    (True, -0.02, 0.024)

What it does **not** do: it does not invent the input uncertainty from a sensitivity the caller did not run, and it
does not call a model credible because a box was ticked — the matrix says which evidence is still missing.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CredibilityMatrix", "CredibilityResult", "UncertaintyBudget", "ValidationResult", "validation",
           "validation_report"]

#: ASME V&V 20-2009, the standard this module implements
CITATION = ("ASME V&V 20-2009, Standard for Verification and Validation in Computational Fluid Dynamics and Heat "
            "Transfer, American Society of Mechanical Engineers, 2009")

#: ASME V&V 40-2018, the influence/consequence table the credibility block follows
CREDIBILITY_CITATION = ("ASME V&V 40-2018, Assessing Credibility of Computational Modeling through Verification and "
                        "Validation: Application to Medical Devices, American Society of Mechanical Engineers, 2018")

#: needed credibility for (model influence, decision consequence); ASME V&V 40's Table 2
_NEEDED = {("high", "high"): "high", ("high", "medium"): "high", ("high", "low"): "medium",
           ("medium", "high"): "high", ("medium", "medium"): "medium", ("medium", "low"): "low",
           ("low", "high"): "medium", ("low", "medium"): "low", ("low", "low"): "low"}

_LEVELS = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class UncertaintyBudget:
    """Every uncertainty that enters the comparison, in the same units as the quantity, plus each share.

    ``u_input`` is the uncertainty of the input parameters propagated to the quantity of interest (a sensitivity or
    a Monte Carlo run gives it; the caller has to have done one). ``u_reference`` is the uncertainty of the data
    itself — an experiment's own error bars, which are never zero.
    """

    u_numerical: float
    u_input: float = 0.0
    u_reference: float = 0.0
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("u_numerical", "u_input", "u_reference"):
            value = float(getattr(self, name))
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative; got {value!r}")

    @property
    def u_val(self) -> float:
        """The validation uncertainty: the three contributions in quadrature."""
        return float(np.sqrt(self.u_numerical ** 2 + self.u_input ** 2 + self.u_reference ** 2))

    def shares(self) -> dict[str, float]:
        """Fraction of the variance each contribution carries — where to spend the next effort."""
        total = self.u_val ** 2
        if total == 0:
            return {k: 0.0 for k in ("numerical", "input", "reference")}
        return {"numerical": self.u_numerical ** 2 / total, "input": self.u_input ** 2 / total,
                "reference": self.u_reference ** 2 / total}

    def dominant(self) -> str:
        """The contribution with the largest share: ``"numerical"``, ``"input"`` or ``"reference"``."""
        return max(self.shares(), key=lambda k: self.shares()[k])

    def to_dict(self) -> dict[str, Any]:
        return {"u_numerical": self.u_numerical, "u_input": self.u_input, "u_reference": self.u_reference,
                "u_val": self.u_val, "shares": self.shares(), "dominant": self.dominant(),
                "notes": dict(self.notes)}


@dataclass
class ValidationResult:
    """The V&V 20 comparison: the error, the uncertainty it has to be judged against, and the verdict."""

    simulation: float
    data: float
    budget: UncertaintyBudget
    quantity: str = ""
    citation: str = CITATION

    @property
    def e(self) -> float:
        """Comparison error ``S - D``."""
        return float(self.simulation) - float(self.data)

    @property
    def relative_error(self) -> float | None:
        """``E / D``, or ``None`` when the data is zero (a ratio to zero is not a number worth reporting)."""
        return None if float(self.data) == 0.0 else self.e / float(self.data)

    @property
    def u_val(self) -> float:
        return self.budget.u_val

    @property
    def ratio(self) -> float | None:
        """``|E| / u_val``: below 1 the comparison is inside the uncertainty, and how far below is how much room."""
        return None if self.u_val == 0 else abs(self.e) / self.u_val

    @property
    def achieved(self) -> bool:
        """Validated at ``u_val``: the standard's condition ``|E| <= u_val``."""
        return abs(self.e) <= self.u_val

    @property
    def verdict(self) -> str:
        """What to do about it, not only whether the box is ticked."""
        if self.achieved:
            return (f"validated at u_val = {self.u_val:.4g}: the comparison error {self.e:+.4g} is inside the "
                    f"uncertainty")
        return (f"not validated: the comparison error {self.e:+.4g} is {self.ratio:.2f} times u_val = "
                f"{self.u_val:.4g}; the difference is bigger than what numerical, input and reference uncertainty "
                f"explain, so the model or an assumption is wrong")

    def to_dict(self) -> dict[str, Any]:
        return {"quantity": self.quantity, "simulation": self.simulation, "data": self.data, "e": self.e,
                "relative_error": self.relative_error, "u_val": self.u_val, "ratio": self.ratio,
                "achieved": self.achieved, "verdict": self.verdict, "budget": self.budget.to_dict(),
                "citation": self.citation}

    def to_markdown(self) -> str:
        b = self.budget
        rel = "—" if self.relative_error is None else f"{self.relative_error:+.2%}"
        ratio = "—" if self.ratio is None else f"{self.ratio:.2f}"
        return "\n".join([
            f"# ASME V&V 20 validation{' — ' + self.quantity if self.quantity else ''}", "",
            "| Quantity | Value |", "|---|---|",
            f"| Simulation S | {self.simulation:.6g} |", f"| Data D | {self.data:.6g} |",
            f"| Comparison error E = S − D | {self.e:+.6g} ({rel}) |",
            f"| Numerical uncertainty u_num | {b.u_numerical:.4g} |",
            f"| Input uncertainty u_input | {b.u_input:.4g} |",
            f"| Reference uncertainty u_ref | {b.u_reference:.4g} |",
            f"| Validation uncertainty u_val | {self.u_val:.4g} |",
            f"| |E| / u_val | {ratio} |",
            f"| Achieved at u_val | {'yes' if self.achieved else 'no'} |", "",
            f"**{self.verdict}.** The uncertainty is {b.dominant()}-dominated "
            f"({b.shares()[b.dominant()]:.0%} of the variance).", "", f"_{self.citation}_", "",
        ])

    def __str__(self) -> str:
        return f"{'achieved' if self.achieved else 'not achieved'}: E = {self.e:+.4g}, u_val = {self.u_val:.4g}"


def validation(simulation: float, data: float, *, u_numerical: float, u_input: float = 0.0,
               u_reference: float = 0.0, quantity: str = "",
               notes: Mapping[str, str] | None = None) -> ValidationResult:
    """The V&V 20 comparison of one number against one datum.

    ``u_numerical`` is required and has no default: a validation without the numerical uncertainty is the report the
    standard exists to replace. When a mesh study was run, ``gci`` gives it (``u_numerical = gci_fine / 100 * |S|``
    for a GCI in percent).
    """
    return ValidationResult(float(simulation), float(data),
                            UncertaintyBudget(float(u_numerical), float(u_input), float(u_reference),
                                              dict(notes or {})), quantity=quantity)


def validation_from_gci(simulation: float, data: float, study: Any, *, u_input: float = 0.0,
                        u_reference: float = 0.0, quantity: str = "") -> ValidationResult:
    """The same comparison with ``u_numerical`` taken from a mesh-independence study's fine-grid GCI.

    ``study`` is a :class:`~pinneapple_analysis.verification.convergence.MeshIndependenceStudyResult` (or its
    ``ConvergenceResult``): the GCI is a percentage of the fine-grid value, so the numerical uncertainty is that
    fraction of the simulation's own number.
    """
    conv = getattr(study, "convergence", study)
    gci = float(conv.gci_fine)
    if not np.isfinite(gci) or gci < 0:
        raise ValueError(f"the study reports a non-finite GCI ({gci!r}); a validation cannot be built on it")
    return validation(simulation, data, u_numerical=abs(float(simulation)) * gci / 100.0, u_input=u_input,
                      u_reference=u_reference, quantity=quantity,
                      notes={"numerical": f"GCI_fine = {gci:.3g} % of the fine-grid value"})


@dataclass
class CredibilityResult:
    """Whether the evidence gathered is the amount this decision needs."""

    influence: str
    consequence: str
    needed: str
    achieved: str
    evidence: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return _LEVELS.index(self.achieved) >= _LEVELS.index(self.needed) and not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {"influence": self.influence, "consequence": self.consequence, "needed": self.needed,
                "achieved": self.achieved, "sufficient": self.sufficient, "evidence": dict(self.evidence),
                "missing": list(self.missing)}


class CredibilityMatrix:
    """The credibility block: model influence × decision consequence → the level of evidence required.

    The model's influence is how much the decision depends on this answer (a screening number or the number the
    design is signed off on), the consequence is what a wrong answer costs. Together they say whether *low*,
    *medium* or *high* credibility is needed; each piece of evidence is recorded at a level, and the block reports
    the gaps instead of a single score. This is ASME V&V 40's table applied to a simulation result.
    """

    #: the evidence categories the standard expects, in the order a reviewer reads them
    CATEGORIES = ("verification", "validation", "uncertainty", "applicability")

    def __init__(self, *, influence: str = "medium", consequence: str = "medium") -> None:
        for name, value in (("influence", influence), ("consequence", consequence)):
            if str(value).lower() not in _LEVELS[1:]:
                raise ValueError(f"{name} must be one of {_LEVELS[1:]}; got {value!r}")
        self.influence = str(influence).lower()
        self.consequence = str(consequence).lower()
        self.required = {c: "low" for c in self.CATEGORIES}
        self._evidence: dict[str, tuple[str, str]] = {}

    @property
    def needed(self) -> str:
        """The credibility level this decision requires."""
        return _NEEDED[(self.influence, self.consequence)]

    def require(self, category: str, level: str) -> CredibilityMatrix:
        """Raise the level needed in one category above the decision's baseline (a code-verification requirement,
        say, for a new solver)."""
        if category not in self.CATEGORIES:
            raise ValueError(f"unknown category {category!r}; one of {self.CATEGORIES}")
        if level not in _LEVELS[1:]:
            raise ValueError(f"level must be one of {_LEVELS[1:]}; got {level!r}")
        self.required[category] = level
        return self

    def add(self, category: str, level: str, evidence: str) -> CredibilityMatrix:
        """Record evidence: which category, at which level, and what backs it (a file, a test, a report)."""
        if category not in self.CATEGORIES:
            raise ValueError(f"unknown category {category!r}; one of {self.CATEGORIES}")
        if level not in _LEVELS[1:]:
            raise ValueError(f"level must be one of {_LEVELS[1:]}; got {level!r}")
        if not evidence:
            raise ValueError("evidence needs something to point at: a path, a test name or a report section")
        self._evidence[category] = (level, evidence)
        return self

    def evaluate(self) -> CredibilityResult:
        """The block as a verdict: the level achieved per category (the lowest recorded), and what is missing."""
        achieved, missing = {}, []
        for category in self.CATEGORIES:
            recorded = self._evidence.get(category)
            want = max(self.needed, self.required[category], key=_LEVELS.index)
            if recorded is None:
                achieved[category] = "none"
                missing.append(f"{category}: no evidence recorded, {want} needed")
                continue
            level, _ = recorded
            achieved[category] = level
            if _LEVELS.index(level) < _LEVELS.index(want):
                missing.append(f"{category}: {level} recorded, {want} needed")
        # the weakest level actually recorded; an empty category shows up in ``missing`` and razes ``sufficient``
        weakest = min((v for v in achieved.values() if v != "none"), key=_LEVELS.index, default="none")
        return CredibilityResult(self.influence, self.consequence, self.needed, weakest,
                                 {c: f"{lvl} — {self._evidence[c][1]}" for c, lvl in achieved.items() if c in self._evidence},
                                 missing)

    def to_markdown(self) -> str:
        result = self.evaluate()
        rows = ["| Category | Needed | Recorded | Evidence |", "|---|---|---|---|"]
        for category in self.CATEGORIES:
            want = max(self.needed, self.required[category], key=_LEVELS.index)
            recorded = self._evidence.get(category)
            rows.append(f"| {category} | {want} | {recorded[0] if recorded else 'none'} | "
                        f"{recorded[1] if recorded else '—'} |")
        status = ("sufficient" if result.sufficient else "NOT sufficient")
        head = (f"# Credibility matrix\n\nModel influence **{self.influence}**, decision consequence "
                f"**{self.consequence}** → {self.needed} credibility needed; achieved **{result.achieved}** "
                f"({status}).")
        gaps = "" if not result.missing else "\n\n**Missing:**\n\n" + "\n".join(f"- {m}" for m in result.missing)
        return "\n".join([head, "", *rows]) + gaps + f"\n\n_{CREDIBILITY_CITATION}_\n"

    def to_dict(self) -> dict[str, Any]:
        result = self.evaluate()
        return {"influence": self.influence, "consequence": self.consequence, "needed": self.needed,
                "required": dict(self.required),
                "evidence": {c: {"level": lvl, "ref": ref} for c, (lvl, ref) in self._evidence.items()},
                "achieved": result.achieved, "sufficient": result.sufficient, "missing": list(result.missing)}


def validation_report(results: Iterable[ValidationResult], *, credibility: CredibilityMatrix | None = None,
                      title: str = "Validation report") -> str:
    """A report a reviewer can read: one row per validated quantity, then the credibility block."""
    rows = list(results)
    lines = [f"# {title}", "", "## Quantities (ASME V&V 20)", "",
             "| Quantity | S | D | E | E/D | u_val | \\|E\\|/u_val | Achieved |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rel = "—" if r.relative_error is None else f"{r.relative_error:+.2%}"
        ratio = "—" if r.ratio is None else f"{r.ratio:.2f}"
        lines.append(f"| {r.quantity or '—'} | {r.simulation:.6g} | {r.data:.6g} | {r.e:+.4g} | {rel} | "
                     f"{r.u_val:.4g} | {ratio} | {'yes' if r.achieved else 'no'} |")
    achieved = sum(1 for r in rows if r.achieved)
    lines += ["", f"**{achieved} of {len(rows)} quantities validated at their own u_val.**", ""]
    if credibility is not None:
        lines += [credibility.to_markdown()]
    else:
        lines += [f"_{CITATION}_", ""]
    return "\n".join(lines)
