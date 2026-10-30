"""The Physics Plausibility Framework: what a result supports, and what it does not.

A simulation shows the behaviour of the equations that were adopted. It does not show that a technology will work,
that a material can be made, or that a measurement agrees. Those are different claims, and the Lab keeps them
apart instead of averaging them into one number.

Six axes, each scored only from evidence that the run recorded (a check tagged with its ``source``):

=================================  ===================================================================
``mathematical_consistency``       the discretised problem solves the stated equation (a residual, an
                                   analytic solution on the same grid, a bound)
``numerical_stability``            the answer does not depend on the mesh, the step or the seed (a
                                   refinement ladder with its observed order)
``physical_plausibility``          the answer obeys the physics it must (conservation, a bound, a symmetry,
                                   an asymptotic limit)
``experimental_validation``        a measurement -- a bench or published data -- agrees
``technological_feasibility``      the thing can be built, made or powered with known means
``uncertainty_sensitivity``        how much the answer moves when the inputs move
=================================  ===================================================================

An axis with no evidence is **not assessed**, never zero and never one: a run cannot earn credit for a check it
did not perform. ``technological_feasibility`` and ``experimental_validation`` therefore stay not assessed in a
pure simulation, and the report says so in as many words -- this is the caution the Lab is built on.

Two verdicts come out, and they are independent on purpose:

* **physics** -- ``SUPPORTED`` / ``CONDITIONAL`` / ``CONTRADICTED`` / ``NOT ASSESSED`` from the first three axes;
* **engineering** -- ``FEASIBLE`` / ``CONDITIONAL`` / ``NOT FEASIBLE`` / ``NOT ASSESSED`` from
  ``technological_feasibility`` alone.

The epistemic class of the proposal (:mod:`pinneapple_futures.schema`) is reported next to them and is never
upgraded by a passing simulation: a passing simulation of hypothetical physics is still hypothetical physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AXES = ("mathematical_consistency", "numerical_stability", "physical_plausibility", "experimental_validation",
        "technological_feasibility", "uncertainty_sensitivity")

PHYSICS_AXES = ("mathematical_consistency", "numerical_stability", "physical_plausibility")

#: the ``source`` a check carries, and the axis it counts for
SOURCE_TO_AXIS = {
    "analytic": "mathematical_consistency",
    "residual": "mathematical_consistency",
    "bound": "mathematical_consistency",
    "convergence": "numerical_stability",
    "stability": "numerical_stability",
    "conservation": "physical_plausibility",
    "limit": "physical_plausibility",
    "symmetry": "physical_plausibility",
    "bench": "experimental_validation",
    "engineering": "technological_feasibility",
    "sensitivity": "uncertainty_sensitivity",
}

PHYSICS_VERDICTS = ("SUPPORTED", "CONDITIONAL", "CONTRADICTED", "NOT ASSESSED")
ENGINEERING_VERDICTS = ("FEASIBLE", "CONDITIONAL", "NOT FEASIBLE", "NOT ASSESSED")

SUPPORTED_AT = 0.8
FLOOR_AT = 0.5

CAVEAT = ("a model that is mathematically consistent and numerically stable shows the behaviour of the equations "
          "that were adopted; it does not show that the technology can be built or that the world behaves so")

__all__ = ["AXES", "PHYSICS_AXES", "SOURCE_TO_AXIS", "PHYSICS_VERDICTS", "ENGINEERING_VERDICTS", "CAVEAT",
           "AxisScore", "Assessment", "assess"]


@dataclass(frozen=True)
class AxisScore:
    """The evidence on one axis: how many checks passed, which ones, and what they were."""

    axis: str
    passed: int = 0
    total: int = 0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"axis {self.axis!r} not in {AXES}")
        if self.passed < 0 or self.total < 0 or self.passed > self.total:
            raise ValueError(f"{self.axis}: {self.passed} passed of {self.total} is not a tally")

    @property
    def assessed(self) -> bool:
        return self.total > 0

    @property
    def score(self) -> float | None:
        """Fraction of the axis' checks that passed, or ``None`` when nothing was assessed."""
        return self.passed / self.total if self.total else None

    def __str__(self) -> str:
        if not self.assessed:
            return f"{self.axis}: not assessed"
        return f"{self.axis}: {self.passed}/{self.total} ({self.score:.2f})"

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "passed": self.passed, "total": self.total,
                "score": self.score, "evidence": list(self.evidence)}


@dataclass
class Assessment:
    """The six axes, the epistemic class and the two verdicts, with the reason for each."""

    epistemic: str = "fictional"
    axes: dict[str, AxisScore] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = [a for a in AXES if a not in self.axes]
        for a in missing:
            self.axes[a] = AxisScore(a)

    # ------------------------------------------------------------------ reading the axes
    def axis(self, name: str) -> AxisScore:
        return self.axes[name]

    def missing(self) -> list[str]:
        """The axes with no evidence at all."""
        return [a for a in AXES if not self.axes[a].assessed]

    def blockers(self, floor: float = FLOOR_AT) -> list[str]:
        """The assessed axes below ``floor``: what stands in the way, weakest link first."""
        bad = [(a, self.axes[a].score) for a in AXES if self.axes[a].assessed and self.axes[a].score < floor]
        return [a for a, _ in sorted(bad, key=lambda kv: kv[1])]

    def score(self, names: tuple[str, ...]) -> float | None:
        """The pooled fraction of passed checks over ``names`` (``None`` if none of them was assessed)."""
        p = sum(self.axes[a].passed for a in names)
        t = sum(self.axes[a].total for a in names)
        return p / t if t else None

    # ------------------------------------------------------------------ verdicts
    @property
    def physics_verdict(self) -> str:
        s = self.score(PHYSICS_AXES)
        if s is None:
            return "NOT ASSESSED"
        if any(a in PHYSICS_AXES for a in self.blockers()):
            return "CONTRADICTED"
        return "SUPPORTED" if s >= SUPPORTED_AT else "CONDITIONAL"

    @property
    def engineering_verdict(self) -> str:
        a = self.axes["technological_feasibility"]
        if not a.assessed:
            return "NOT ASSESSED"
        if a.score >= SUPPORTED_AT:
            return "FEASIBLE"
        return "CONDITIONAL" if a.score >= FLOOR_AT else "NOT FEASIBLE"

    @property
    def caveat(self) -> str:
        return CAVEAT

    def headline(self) -> str:
        """One line: what the run supports, what it does not, and how strong the physics was to begin with."""
        return (f"physics {self.physics_verdict} | engineering {self.engineering_verdict} | "
                f"class {self.epistemic}")

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        return {"epistemic": self.epistemic, "physics_verdict": self.physics_verdict,
                "engineering_verdict": self.engineering_verdict, "caveat": self.caveat,
                "axes": {a: self.axes[a].to_dict() for a in AXES}, "blockers": self.blockers(),
                "not_assessed": self.missing(), "notes": list(self.notes)}

    def __str__(self) -> str:
        lines = [self.headline()]
        for a in AXES:
            ax = self.axes[a]
            mark = "  -  " if not ax.assessed else ("ok   " if ax.score >= SUPPORTED_AT else
                                                   ("weak " if ax.score >= FLOOR_AT else "BAD  "))
            lines.append(f"  {mark}{a:26s} {ax.score if ax.assessed else 'not assessed'}"
                         f"{'' if not ax.evidence else '  <- ' + ', '.join(ax.evidence)}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        lines.append(f"  {self.caveat}")
        return "\n".join(lines)


def assess(result: Any, *, epistemic: str = "fictional", notes: tuple[str, ...] = ()) -> Assessment:
    """Score the six axes from what the result recorded: every check counts for the axis its ``source`` names.

    A check with ``passed`` ``None`` was not run and does not count; a check whose ``source`` names no axis is
    ignored (it is a note, not evidence). Comparisons against a reference are always mathematical consistency --
    they say the computed number matches the number it should match, not that the physics of the world was
    reproduced.
    """
    tally: dict[str, list[tuple[bool, str]]] = {a: [] for a in AXES}
    for c in getattr(result, "checks", []):
        axis = SOURCE_TO_AXIS.get(str(c.get("source", "")))
        if axis is None or c.get("passed") is None:
            continue
        tally[axis].append((bool(c["passed"]), str(c.get("name", ""))))
    for cmp in getattr(result, "comparisons", []):
        tally["mathematical_consistency"].append((bool(cmp.passed), f"{cmp.quantity} vs {cmp.kind}"))

    axes = {}
    for a in AXES:
        rows = tally[a]
        axes[a] = AxisScore(a, sum(1 for ok, _ in rows if ok), len(rows),
                            tuple(n for _, n in rows if n))

    out_notes = list(notes)
    if not axes["experimental_validation"].assessed:
        out_notes.append("no measurement was compared: the run validates the model, not the world")
    if not axes["technological_feasibility"].assessed:
        out_notes.append("technological feasibility was not assessed: a simulation cannot establish it")
    if not axes["uncertainty_sensitivity"].assessed:
        out_notes.append("no sensitivity study was run: the spread of the answer is unknown")
    return Assessment(epistemic=epistemic, axes=axes, notes=tuple(out_notes))
