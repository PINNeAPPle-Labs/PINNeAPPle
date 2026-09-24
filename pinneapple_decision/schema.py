"""Core types of the decision layer: choices, decisions, evidence.

Vocabulary rule (deliberate): a :class:`Decision` carries ``probabilities``
and ``probability`` -- the backend's debiased distribution over the feasible
options. It is **not** called "confidence". ``calibrated_confidence`` exists
only at :attr:`DecisionLevel.L1`, i.e. after calibration on labelled
outcomes, and :class:`Decision` refuses to be built with one at any other
level.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class DecisionLevel(str, Enum):
    """What was done to the backend's scores before reading them.

    ``RAW``  one ordering of the options, softmax of the scores. A ranking, not a probability.
    ``L0``   debiased without labels: cyclic-shift averaging over option order
             (position bias) + prior correction (label prior). See ``debias.py``.
    ``L1``   L0 + calibration fit on labelled outcomes. **Not implemented yet**;
             the level exists so ``calibrated_confidence`` has a contract.
    ``FACT`` no scoring at all: the answer was read from a fact of the problem
             (e.g. "is there an analytical solution?" when the problem says so).
             Used by the decision tree; the 1.0 is the fact, not a model output.
    """

    RAW = "raw"
    L0 = "L0"
    L1 = "L1"
    FACT = "fact"


@dataclass(frozen=True)
class OptionInfo:
    """Metadata of one option, used by rules and constraints (never by the LLM to decide).

    ``implementation`` is a dotted path to the real PINNeAPPle symbol that runs this
    option, or ``None`` when the library has no implementation (the
    ``requires_implementation`` constraint excludes it, with that reason).
    """

    name: str
    description: str = ""
    implementation: Optional[str] = None
    representations: Tuple[str, ...] = ()
    supports_varying_geometry: bool = False
    relative_cost: int = 1  # 1 cheap .. 3 expensive (coarse, order-of-magnitude)
    source: str = ""
    notes: str = ""


@dataclass
class PhysicsChoice:
    """A typed question with a closed set of options.

    ``constraints`` holds constraint names (resolved in ``constraints.py``, e.g.
    ``"must_support_geometry"``) or :class:`~pinneapple_decision.constraints.Constraint`
    objects. They run *before* scoring (an excluded option is never shown to the
    backend) and again right before execution.
    """

    name: str
    question: str
    options: List[str]
    constraints: List[Any] = field(default_factory=list)
    option_info: Dict[str, OptionInfo] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError(f"PhysicsChoice {self.name!r} needs at least 2 options")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"PhysicsChoice {self.name!r} has duplicate options")

    def info(self, option: str) -> OptionInfo:
        return self.option_info.get(option, OptionInfo(name=option))


@dataclass
class ExecutionResult:
    """What an executor returns. ``diagnostics`` are short tags a rule or LLM can
    read on the next decision (e.g. ``"residual_localized"``, ``"late_time_error"``)."""

    status: str  # "ok" | "failed"
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    decision_id: Optional[str] = None


@dataclass
class Verification:
    """Outcome of verifying one :class:`ExecutionResult`.

    ``checks`` maps check name -> ``True``/``False``; a check that did not run is
    absent, never a fabricated pass. ``tags`` feed the next decision.
    """

    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source: str = ""
    notes: str = ""


@dataclass
class Decision:
    choice: str
    question: str
    selected: str
    probabilities: Dict[str, float]
    level: DecisionLevel = DecisionLevel.L0
    requires_validation: bool = True
    calibrated_confidence: Optional[float] = None
    excluded: Dict[str, str] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    backend: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.level = DecisionLevel(self.level)
        if self.calibrated_confidence is not None and self.level is not DecisionLevel.L1:
            raise ValueError(
                "calibrated_confidence is only defined at level L1 (calibrated on labelled "
                f"outcomes); got level {self.level.value}. Use `probability` instead."
            )
        if self.selected not in self.probabilities:
            raise ValueError(f"selected option {self.selected!r} has no probability")

    @property
    def probability(self) -> float:
        """Probability of ``selected`` under the level's distribution (not a confidence)."""
        return self.probabilities[self.selected]

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Decision":
        return cls(**dict(d))


@dataclass
class Evidence:
    """One OBSERVE -> DECIDE -> EXECUTE -> VALIDATE record."""

    decision: Decision
    result: Optional[ExecutionResult] = None
    verification: Optional[Verification] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "result": asdict(self.result) if self.result else None,
            "verification": asdict(self.verification) if self.verification else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Evidence":
        return cls(
            decision=Decision.from_dict(d["decision"]),
            result=ExecutionResult(**d["result"]) if d.get("result") else None,
            verification=Verification(**d["verification"]) if d.get("verification") else None,
        )


@dataclass
class DecisionState:
    """Everything a backend may look at: the problem plus the evidence so far."""

    problem: Dict[str, Any] = field(default_factory=dict)
    previous_result: Optional[ExecutionResult] = None
    verification: Optional[Verification] = None
    history: Tuple[Evidence, ...] = ()
    #: provenance of the problem facts (``adapter.AdaptedProblem.report()``): given / inferred / unknown.
    facts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_problem(cls, problem: Any, **kwargs: Any) -> "DecisionState":
        """Adapt ``problem`` (``ProblemSpec``, free-form dict, or ``None``) and keep its provenance."""
        from .adapter import adapt_problem

        ad = adapt_problem(problem)
        return cls(problem=ad.problem, facts=ad.report(), **kwargs)

    @classmethod
    def coerce(cls, obj: Any) -> "DecisionState":
        """Accept a ``DecisionState``, a ``{"problem": ..., "previous_result": ...,
        "verification": ...}`` dict, a bare problem dict, or a
        ``pinneapple_problemdesign.ProblemSpec`` (duck-typed).

        Problems go through :func:`pinneapple_decision.adapter.adapt_problem`:
        synonyms (``Re``, ``steady``, a named geometry, ...) become the canonical
        keys the rules and constraints read, and facts that cannot be determined
        are listed as unknown in ``facts`` instead of being filled in."""
        if isinstance(obj, cls):
            return obj
        if obj is None:
            return cls.from_problem(None)
        if hasattr(obj, "physics") and hasattr(obj, "geometry") and hasattr(obj, "task_type"):
            return cls.from_problem(obj)
        if isinstance(obj, Mapping):
            if "problem" in obj:
                return cls.from_problem(
                    obj["problem"],
                    previous_result=obj.get("previous_result"),
                    verification=obj.get("verification"),
                    history=tuple(obj.get("history", ())),
                )
            return cls.from_problem(obj)
        raise TypeError(f"cannot build a DecisionState from {type(obj).__name__}")

    def tags(self) -> List[str]:
        """Diagnostic tags from the previous result and its verification."""
        out: List[str] = []
        if self.previous_result is not None:
            out += list(self.previous_result.diagnostics)
        if self.verification is not None:
            out += list(self.verification.tags)
        return out

    def failed_options(self, choice: str) -> List[str]:
        """Options of ``choice`` already executed whose verification failed."""
        return [
            ev.decision.selected
            for ev in self.history
            if ev.decision.choice == choice and ev.verification is not None and not ev.verification.passed
        ]

    def summary(self) -> Dict[str, Any]:
        """JSON-able view used in LLM prompts."""
        return {
            "problem": self.problem,
            "unknown_facts": sorted(self.facts.get("unknown", {})),
            "previous_result": asdict(self.previous_result) if self.previous_result else None,
            "verification": asdict(self.verification) if self.verification else None,
            "tried": [
                {"choice": ev.decision.choice, "selected": ev.decision.selected,
                 "passed": None if ev.verification is None else ev.verification.passed}
                for ev in self.history
            ],
        }


def problem_from_spec(spec: Any) -> Dict[str, Any]:
    """Map a ``pinneapple_problemdesign.ProblemSpec`` to this module's problem dict.

    Delegates to :func:`pinneapple_decision.adapter.adapt_problem`. Only fields
    the spec actually carries are mapped; nothing numeric is guessed (same
    non-invention policy as ``method_selection.recommend_method_from_spec``).
    Use ``adapt_problem(spec).report()`` to see what stayed unknown.
    """
    from .adapter import adapt_problem

    return adapt_problem(spec).problem


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


__all__ = [
    "Decision",
    "DecisionLevel",
    "DecisionState",
    "Evidence",
    "ExecutionResult",
    "OptionInfo",
    "PhysicsChoice",
    "Verification",
    "problem_from_spec",
]
