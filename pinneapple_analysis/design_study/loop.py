"""The budgeted design loop of #79: fit, optimise, measure, and a **stop decided on a number**.

:func:`pinneapple_analysis.design_study.run_study` runs the cycle once by default — baseline, DOE, surrogate,
NSGA-II, verification. That answers "what does one surrogate, fitted on one DOE, recommend", which is the right
answer when the reference budget is exactly one DOE. When the budget is larger than that, running the cycle once
spends the DOE and leaves the rest of the budget unspent: nothing new is measured, so nothing new is learned.

This module is the policy of the loop that spends it. Each round refits the surrogate on **everything measured so
far**, optimises on it, measures the design it recommends on the reference evaluator, and then the rule decides
whether another round buys anything. Every round costs reference evaluations, which is the cost this loop budgets;
the rule stops on whichever of four numbers is reached first, and the reason it stopped is written into
:attr:`~.study.StudyRun.loop` so a reader can check the decision instead of trusting the runner:

``budget``
    no reference evaluation left for another design (the plan in :meth:`~.spec.Budget.planned` split them all);
``nothing_new``
    the round could not measure anything: every design on the surrogate's front was within the exclusion radius of a
    design the reference had already measured, so another round would spend the budget re-measuring the same design;
``no_improvement``
    the primary objective's best measured value has not improved by ``min_improvement`` (relative) for ``patience``
    rounds: the loop is measuring, not learning;
``decision``
    the surrogate picks the same design the reference does, within ``decision_tol`` of the *regret* measured by
    :func:`pinneapple_analysis.decision.regret` over the designs the loop recommended and the reference measured —
    and has done so for ``patience`` rounds. More measurements cannot change the design the study recommends, so
    there is nothing left to buy;
``max_rounds``
    the hard cap, reached with budget still left (which the run reports as unspent).
``single_pass``
    not a decision of this rule: the run that was asked for one cycle, and says so in the same field.

Progress is measured on the **primary objective** (the first one declared), as the relative gain of its best
*measured* value over the previous round's, and every objective's best measured value is recorded each round so a
multi-objective reader can see the whole picture. Averages of objectives would be a weighting nobody asked for.

Why not keep going until the budget is gone: an optimum sits on a limit and the last rounds of a converged loop
re-measure designs within the surrogate's own error of each other, which is how a budget buys a false sense of
precision. The rule's job is to stop at the round after the answer stopped moving, and to say which number stopped
it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["StopRule", "decide"]

#: The reasons, in the order they are checked: cost first (a budget that is gone decides itself), then a round that
#: measured nothing, then the objective that stopped moving, then the decision-grade surrogate, then the cap.
REASONS = ("budget", "nothing_new", "no_improvement", "decision", "max_rounds")


@dataclass(frozen=True)
class StopRule:
    """How many rounds the loop may run and which number ends it.

    ``per_round``: reference designs measured per round (each one is a design the surrogate recommended that the
    reference then confirms). ``patience``: how many consecutive rounds a criterion has to hold before it stops the
    loop, so a single quiet round does not end it. ``min_improvement``: the relative gain of the primary objective's
    best measured value that counts as progress. ``decision_tol``: the relative regret under which the surrogate
    picks the design the reference would, i.e. is decision-grade. ``max_rounds``: the hard cap.
    """

    min_improvement: float = 0.01
    patience: int = 2
    max_rounds: int = 6
    per_round: int = 1
    decision_tol: float = 0.02

    def __post_init__(self):
        mi = float(self.min_improvement)
        if not mi >= 0.0 or mi != mi:
            raise ValueError(f"min_improvement is a relative gain and cannot be negative, got {self.min_improvement!r}")
        object.__setattr__(self, "min_improvement", mi)
        for name, low in (("patience", 1), ("max_rounds", 1), ("per_round", 1)):
            v = int(getattr(self, name))
            if v < low:
                raise ValueError(f"{name} must be at least {low}, got {v}")
            object.__setattr__(self, name, v)
        dt = float(self.decision_tol)
        if not 0.0 <= dt < 1.0:
            raise ValueError(f"decision_tol is a relative regret and must be in [0, 1), got {self.decision_tol!r}")
        object.__setattr__(self, "decision_tol", dt)

    def to_dict(self) -> dict[str, Any]:
        return {"min_improvement": self.min_improvement, "patience": self.patience, "max_rounds": self.max_rounds,
                "per_round": self.per_round, "decision_tol": self.decision_tol}

    @classmethod
    def from_dict(cls, d: Mapping) -> StopRule:
        extra = sorted(set(d) - set(cls.__dataclass_fields__))
        if extra:
            raise ValueError(f"stop rule: unknown field(s) {extra}; known: {list(cls.__dataclass_fields__)}")
        return cls(**dict(d))


def decide(rule: StopRule, rounds: Sequence[Mapping[str, Any]], *, budget_left: int) -> dict[str, Any]:
    """Whether the loop stops after the rounds recorded so far, and which number decided it.

    ``rounds`` are the round records the runner builds (``round``, ``best``, ``improvement``, ``regret``,
    ``evaluations``); ``budget_left`` is how many reference evaluations remain. The returned dict is what the run
    keeps: ``stop``, ``reason``, ``round``, the ``value`` that decided it and the ``limit`` it was compared with.
    """
    n = len(rounds)
    last = rounds[-1] if rounds else {}
    evals = int(last.get("evaluations", 0))

    def verdict(stop: bool, reason: str, value: float | None, limit: float | None, text: str) -> dict[str, Any]:
        return {"stop": bool(stop), "reason": reason, "round": n, "evaluations": evals,
                "budget_left": int(budget_left), "value": value, "limit": limit, "text": text}

    if budget_left <= 0:
        return verdict(True, "budget", float(evals), None,
                       "the reference budget is spent: no design can be measured again")
    if int(last.get("measured", 0) or 0) == 0:
        return verdict(True, "nothing_new", 0.0, None,
                       "no design the surrogate recommended was far enough from what was already measured: another "
                       "round would re-measure the same designs")
    tail = list(rounds[-rule.patience:])
    if len(tail) >= rule.patience:
        gains = [r.get("improvement") for r in tail]
        if all(v is not None and float(v) < rule.min_improvement for v in gains):
            best = max(float(v) for v in gains)
            return verdict(True, "no_improvement", best, rule.min_improvement,
                           f"the primary objective gained at most {100 * best:.2f} % a round for {rule.patience} "
                           "rounds: the loop is measuring, not learning")
        regrets = [r.get("regret") for r in tail]
        if all(v is not None and float(v) <= rule.decision_tol for v in regrets):
            worst = max(float(v) for v in regrets)
            return verdict(True, "decision", worst, rule.decision_tol,
                           f"the surrogate picked the reference's design within {100 * worst:.2f} % regret for "
                           f"{rule.patience} rounds: no measurement left can change the recommendation")
    if n >= rule.max_rounds:
        return verdict(True, "max_rounds", float(n), float(rule.max_rounds),
                       f"the hard cap of {rule.max_rounds} round(s) was reached")
    return verdict(False, "continue", None, None, f"round {n} of at most {rule.max_rounds}")
