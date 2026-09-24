"""Hard constraints on options: physics/geometry, resources, implementation.

A constraint never re-ranks options. It either keeps an option or excludes it
with a human-readable reason, and every exclusion is recorded on the
:class:`~pinneapple_decision.schema.Decision` (``decision.excluded``).

Constraints run twice: before scoring (an excluded option is never shown to the
backend, so an LLM cannot pick it) and right before execution (the problem may
have changed between ``decide`` and ``execute``).

Named constraints read their parameters from the problem dict:

``must_support_geometry``
    ``problem["representation"]`` (``"structured_grid"``, ``"unstructured_mesh"``,
    ``"point_cloud"``, ``"analytic"``) must be in the option's
    ``OptionInfo.representations``; if ``problem["geometry_varies"]`` the option
    must declare ``supports_varying_geometry``. Options with no declared
    representations are not filtered (nothing to check against).
``requires_implementation``
    ``OptionInfo.implementation`` must be set (a real PINNeAPPle symbol exists).
``within_budget``
    ``OptionInfo.relative_cost <= problem["max_relative_cost"]`` when that key exists.
``not_failed_before``
    the option was not already executed for this choice with a failed verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import DecisionState, OptionInfo, PhysicsChoice

# check(option_info, state, choice) -> None (keep) or a reason string (exclude)
CheckFn = Callable[[OptionInfo, DecisionState, PhysicsChoice], Optional[str]]


@dataclass(frozen=True)
class Constraint:
    name: str
    check: CheckFn


def _must_support_geometry(info: OptionInfo, state: DecisionState, choice: PhysicsChoice) -> Optional[str]:
    rep = state.problem.get("representation")
    if rep and info.representations and rep not in info.representations:
        return (f"must_support_geometry: {info.name} supports {list(info.representations)}, "
                f"problem representation is {rep!r}")
    if state.problem.get("geometry_varies") and info.representations and not info.supports_varying_geometry:
        return f"must_support_geometry: {info.name} does not generalize across varying geometry"
    return None


def _requires_implementation(info: OptionInfo, state: DecisionState, choice: PhysicsChoice) -> Optional[str]:
    if not info.implementation:
        extra = f" ({info.notes})" if info.notes else ""
        return f"requires_implementation: no PINNeAPPle implementation for {info.name}{extra}"
    return None


def _within_budget(info: OptionInfo, state: DecisionState, choice: PhysicsChoice) -> Optional[str]:
    budget = state.problem.get("max_relative_cost")
    if budget is not None and info.relative_cost > budget:
        return f"within_budget: {info.name} relative_cost={info.relative_cost} > max_relative_cost={budget}"
    return None


def _not_failed_before(info: OptionInfo, state: DecisionState, choice: PhysicsChoice) -> Optional[str]:
    if info.name in state.failed_options(choice.name):
        return f"not_failed_before: {info.name} was already executed for {choice.name!r} and failed verification"
    return None


NAMED_CONSTRAINTS: Dict[str, Constraint] = {
    "must_support_geometry": Constraint("must_support_geometry", _must_support_geometry),
    "requires_implementation": Constraint("requires_implementation", _requires_implementation),
    "within_budget": Constraint("within_budget", _within_budget),
    "not_failed_before": Constraint("not_failed_before", _not_failed_before),
}


def resolve(constraint: Any) -> Constraint:
    if isinstance(constraint, Constraint):
        return constraint
    if isinstance(constraint, str):
        try:
            return NAMED_CONSTRAINTS[constraint]
        except KeyError:
            raise KeyError(f"unknown constraint {constraint!r}; known: {sorted(NAMED_CONSTRAINTS)}") from None
    raise TypeError(f"constraint must be a name or a Constraint, got {type(constraint).__name__}")


def apply_constraints(choice: PhysicsChoice, state: DecisionState) -> Tuple[List[str], Dict[str, str]]:
    """Return ``(feasible_options, {excluded_option: reason})`` in the choice's option order."""
    constraints = [resolve(c) for c in choice.constraints]
    feasible: List[str] = []
    excluded: Dict[str, str] = {}
    for opt in choice.options:
        info = choice.info(opt)
        reasons = [r for r in (c.check(info, state, choice) for c in constraints) if r]
        if reasons:
            excluded[opt] = "; ".join(reasons)
        else:
            feasible.append(opt)
    return feasible, excluded


class NoFeasibleOptionError(RuntimeError):
    def __init__(self, choice: str, excluded: Dict[str, str]):
        self.choice = choice
        self.excluded = excluded
        lines = "\n".join(f"  - {k}: {v}" for k, v in excluded.items())
        super().__init__(f"every option of {choice!r} was excluded by constraints:\n{lines}")


class ConstraintViolationError(RuntimeError):
    """Raised when a decision is about to be executed but its option no longer passes."""


__all__ = [
    "Constraint",
    "ConstraintViolationError",
    "NAMED_CONSTRAINTS",
    "NoFeasibleOptionError",
    "apply_constraints",
    "resolve",
]
