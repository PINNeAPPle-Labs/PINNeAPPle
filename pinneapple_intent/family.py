"""What a study family is: the physics knowledge that turns a filled contract into runs, checks and prose.

A family is the unit a new domain plugs in with. It declares

* the words that select it (``keywords``) and the values it can read from a sentence (``slots``) — the same
  :class:`~pinneapple_design.intent.spec.Slot` ``pp.design`` uses, so one sentence reads the same way in both;
* the governing equations, the assumptions behind them, what the study does not cover, and the references the
  analytic rung is compared against — all of them printed in the report, because a result without its assumptions is
  not a result;
* the **fidelity ladder**: an analytic rung (closed form or a cited reference), a numerical rung (a real solver in the
  library), a convergence rung (the same solver refined, with the measured change) and, for ``physics_ai`` mode, a
  surrogate rung trained on the solver and scored on held-out designs;
* the quantity of interest (``qoi``) each rung reports, so the ladder can be compared rung against rung.

The contract the family reads is written by :mod:`pinneapple_intent.compile`; the ladder is driven by
:mod:`pinneapple_intent.execute`. A family never invents a value: everything it computes comes from a closed form, a
solver call, or a number the contract carries with its provenance.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pinneapple_design.intent.spec import Slot

from .contract import (
    MODE_PHYSICS_AI,
    Rung,
    StudyContract,
)

# The four rungs, in the order they run: cheap and exact first, expensive and approximate last, with the refinement
# that makes the solver's own error visible between them.
ANALYTIC, NUMERICAL, CONVERGENCE, SURROGATE = "analytic", "numerical", "convergence", "surrogate"
LADDER = (ANALYTIC, NUMERICAL, CONVERGENCE, SURROGATE)


@dataclass
class Measurement:
    """What one rung produced: the quantity of interest, the checklines, and the evidence it left in ``workdir``.

    ``checks`` follow the shape the rest of the library uses (``name``, ``passed``, ``value``, ``limit``,
    ``detail``), so a study's checks can be read next to a trust card's without translation.
    """
    rung: str
    outputs: dict[str, float] = field(default_factory=dict)      # quantity name -> value, SI
    units: dict[str, str] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)      # name -> path or a small JSON-able payload
    wall_s: float = 0.0
    ran: bool = True
    reason: str = ""                                             # why it did not run

    def to_dict(self) -> dict[str, Any]:
        return {"rung": self.rung, "ran": self.ran, "reason": self.reason, "outputs": self.outputs,
                "units": self.units, "checks": self.checks, "notes": self.notes, "artifacts": self.artifacts,
                "wall_s": self.wall_s}

    @property
    def passed(self) -> bool:
        return self.ran and all(c.get("passed", True) for c in self.checks)


def check(name: str, passed: bool, *, value: Any = None, limit: Any = None, detail: str = "") -> dict[str, Any]:
    """One checkline. ``limit`` is the number it was judged against, so a reader can re-judge it."""
    return {"name": name, "passed": bool(passed), "value": value, "limit": limit, "detail": detail}


def relative_error(measured: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if measured != 0 else 0.0
    return abs(measured - reference) / abs(reference)


class StudyFamily:
    """Base class. Subclasses set the class attributes and implement :meth:`evaluate`."""

    name: str = ""
    title: str = ""
    domain: str = ""
    objective: str = ""                     # the quantity of interest, in words ("steady temperature field")
    qoi: str = ""                           # the output key every rung reports (``outputs[qoi]``)
    qoi_unit: str = ""
    qoi_label: str = ""
    keywords: dict[str, float] = {}
    equations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    slots: tuple[Slot, ...] = ()
    objectives: tuple = ()          # kept empty: a study predicts, it does not optimise; the parser still asks
    # Phrases the family knows better than the generic parser does, as ``(slot, regex)``. The regex must name its
    # two groups ``v`` (the number) and ``u`` (the unit), and a hint wins over the generic reading: "a 20 mm pipe"
    # is a diameter, while the words alone would let the object's name pull the number to whichever slot sits
    # closest to it.
    hints: tuple[tuple[str, str], ...] = ()
    validation: tuple[str, ...] = ()
    # relative tolerance of the solver against the analytic rung, and the change allowed under refinement. A study
    # that misses either is reported as REVIEW, never as passed.
    tolerance: float = 0.05
    convergence_tolerance: float = 0.02
    cells_through_scale: int = 8        # how many grid cells the smallest length scale of the case must carry

    # ------------------------------------------------------------------ declaration helpers
    def slot_map(self) -> dict[str, Slot]:
        return {s.name: s for s in self.slots}

    def required(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots if s.critical)

    def rungs(self, mode: str) -> tuple[Rung, ...]:
        """The ladder this family offers, filtered by mode: ``surrogate`` only when the study asks for Physics AI."""
        wanted = (ANALYTIC, NUMERICAL, CONVERGENCE) if mode != MODE_PHYSICS_AI else LADDER
        return tuple(r for r in self.ladder() if r.name in wanted)

    def ladder(self) -> tuple[Rung, ...]:
        return (
            Rung(ANALYTIC, self.analytic_label(), self.analytic_method(), modules=self.analytic_modules()),
            Rung(NUMERICAL, self.numerical_label(), self.numerical_method(),
                 modules=("pinneapple_simulation.immersed",)),
            Rung(CONVERGENCE, "the same solver on two grids",
                 "the numerical rung repeated at a coarser and a finer spacing, and the change of the quantity "
                 "of interest between them", modules=("pinneapple_simulation.immersed",)),
            Rung(SURROGATE, "a surrogate trained on the solver",
                 "a sweep over one input run through the numerical rung, a regression fitted to it, and its "
                 "relative error on designs held out of the fit", modules=("pinneapple_simulation.immersed",)),
        )

    def analytic_label(self) -> str:
        return "closed form"

    def analytic_method(self) -> str:
        return ""

    def analytic_modules(self) -> tuple[str, ...]:
        return ()

    def numerical_label(self) -> str:
        return "finite cell on a background grid"

    def numerical_method(self) -> str:
        return "the immersed solver of pinneapple_simulation"

    def length_scale(self, contract: StudyContract) -> float:
        """The smallest length the grid has to resolve; a family overrides it with the case's own dimension."""
        return 0.04

    def spacing(self, contract: StudyContract, rung: str) -> float | None:
        """The grid spacing of a rung, derived from the case rather than fixed: a grid that does not resolve the
        section is the usual reason a solver "disagrees" with a closed form it should reproduce."""
        if rung not in (NUMERICAL, CONVERGENCE):
            return None
        return self.length_scale(contract) / self.cells_through_scale

    def check_scope(self, contract: StudyContract) -> list[str]:
        """Parts of the request this family cannot address (empty = in scope)."""
        return []

    def default_objectives(self) -> list:
        """A study has no objective to optimise; the attribute exists so the shared parser can read this family."""
        return list(self.objectives)

    def extra_questions(self, contract: StudyContract) -> list:
        """Questions beyond the critical slots."""
        return []

    # ------------------------------------------------------------------ the physics
    def evaluate(self, rung: str, contract: StudyContract, *, workdir: str | None = None,
                 options: dict[str, Any] | None = None) -> Measurement:
        raise NotImplementedError

    def compare(self, measured: dict[str, Measurement]) -> list[dict[str, Any]]:
        """The cross-rung checks: does the solver reproduce the closed form, and does it survive refinement?"""
        checks = []
        analytic, numerical = measured.get(ANALYTIC), measured.get(NUMERICAL)
        if analytic is not None and numerical is not None and analytic.ran and numerical.ran:
            got = numerical.outputs.get(self.qoi)
            ref = analytic.outputs.get(self.qoi)
            if got is not None and ref is not None:
                err = relative_error(got, ref)
                checks.append(check(
                    f"{self.qoi}: the solver against the {self.analytic_label()}",
                    err <= self.tolerance, value=round(err, 6), limit=self.tolerance,
                    detail=f"{self.qoi} = {got:.6g} {self.qoi_unit} from the solver, {ref:.6g} {self.qoi_unit} "
                           f"from the {self.analytic_label()}; relative error {err:.2%}"))
        convergence = measured.get(CONVERGENCE)
        if convergence is not None and convergence.ran:
            fine_value = convergence.outputs.get(self.qoi)
            coarse_value = convergence.outputs.get(f"coarse_{self.qoi}")
            if fine_value is not None and coarse_value is not None:
                change = relative_error(fine_value, coarse_value)
                checks.append(check(
                    f"{self.qoi}: change under refinement",
                    change <= self.convergence_tolerance, value=round(change, 6),
                    limit=self.convergence_tolerance,
                    detail=f"{self.qoi} = {coarse_value:.6g} {self.qoi_unit} on the coarse grid and "
                           f"{fine_value:.6g} {self.qoi_unit} on the fine one; change {change:.2%} "
                           f"(the limit is the discretisation error accepted for this study, not a physical one)"))
        surrogate = measured.get(SURROGATE)
        if surrogate is not None and surrogate.ran:
            held_out = surrogate.outputs.get("held_out_relative_error")
            if held_out is not None:
                checks.append(check(
                    "the surrogate on designs held out of its fit",
                    held_out <= self.tolerance, value=round(held_out, 6), limit=self.tolerance,
                    detail=surrogate.notes[0] if surrogate.notes else ""))
        return checks

    # ------------------------------------------------------------------ helpers shared by the families
    @staticmethod
    def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
        start = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - start

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "domain": self.domain, "objective": self.objective,
                "qoi": self.qoi, "qoi_unit": self.qoi_unit, "equations": list(self.equations),
                "assumptions": list(self.assumptions), "limitations": list(self.limitations),
                "references": list(self.references),
                "slots": [{"name": s.name, "label": s.label, "unit": s.shown_unit(), "critical": s.critical,
                           "default": s.default, "default_source": s.default_source} for s in self.slots],
                "ladder": [r.to_dict() for r in self.ladder()]}


# ------------------------------------------------------------------------------------------------ registry
_REGISTRY: dict[str, StudyFamily] = {}


def register(family: StudyFamily) -> StudyFamily:
    if not family.name:
        raise ValueError("a study family needs a name")
    _REGISTRY[family.name] = family
    return family


def families() -> dict[str, StudyFamily]:
    _load_builtin()
    return dict(_REGISTRY)


def get_family(name: str) -> StudyFamily:
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown study family {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def _load_builtin() -> None:
    if _REGISTRY:
        return
    from .families import builtin
    for family in builtin():
        register(family)


MIN_SCORE = 0.6


def _keyword_hit(text: str, keyword: str) -> bool:
    """Whole-word match, tolerant of a plural — the same rule ``pp.design`` selects its templates with."""
    import re
    return re.search(rf"(?<![a-z]){re.escape(keyword)}(?:s|es)?(?![a-z])", text) is not None


def select_family(text: str) -> tuple[StudyFamily | None, list[tuple[str, float]]]:
    """The family whose keywords are closest in ``text``, and the ranked alternatives (name, score).

    A request whose best score is below :data:`MIN_SCORE` selects nothing: the compiler then refuses with the words
    it could not place, instead of running the nearest study on it.
    """
    low = text.lower()
    scored: list[tuple[str, float]] = []
    for family in families().values():
        score = sum(weight for word, weight in family.keywords.items() if _keyword_hit(low, word))
        if score:
            scored.append((family.name, float(score)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    if not scored or scored[0][1] < MIN_SCORE:
        return None, scored
    return get_family(scored[0][0]), scored


def numeric(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


__all__ = [
    "ANALYTIC", "NUMERICAL", "CONVERGENCE", "SURROGATE", "LADDER",
    "Measurement", "StudyFamily", "check", "relative_error",
    "register", "families", "get_family", "select_family", "numeric",
]
