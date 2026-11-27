"""Survivor selection for the NSGA-II of :mod:`.optimizer`: Deb's constrained domination, and the trust-aware variant.

:class:`~pinneapple_design.design_optimizer.optimizer.EvolutionaryDesignOptimizer` merges the parents with the
offspring, ranks that pool into non-dominated fronts and keeps the best ``mu``. Two different things can carry a
second signal into that decision, and they are not interchangeable:

``penalties``
    the *measured* constraint violation: ``> 0`` means the design breaks a constraint the solver answered, and Deb's
    rule (feasible first, infeasible ordered by violation) is how NSGA-II has always handled that.

:class:`TrustVerdict`
    the *unmeasured* part — whether the prediction of those constraints can be vouched for at all. A design that
    lies outside the surrogate's applicability domain has not violated anything; what it would score is simply not
    known. Its violation is therefore not comparable with a measured one, which is why it is kept in a separate
    channel and ordered lexicographically in front of (never summed into) the measured penalties.

The decision behind this module — why a constraint and not a third objective, why a separate front rather than
removal or a sum, and what is done with a candidate too uncertain to judge — is written, with the measurements it
rests on, in :mod:`pinneapple_design.design_optimizer.trust_survivors`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .pareto import crowding_distance, fast_non_dominated_sort

__all__ = ["TRUST_LEVELS", "TrustVerdict", "constrained_select", "trust_aware_survivors"]

#: The three levels of the trust channel, best first. ``undetermined`` is a prediction the surrogate cannot judge
#: (its interval is wider than the accuracy the decision needs, or it straddles a limit): not vouched for, but not
#: contradicted either.
TRUST_LEVELS = ("trusted", "undetermined", "untrusted")


@dataclass(frozen=True)
class TrustVerdict:
    """One candidate's trust channel: how a survivor rule may use the surrogate's own admission.

    ``level`` is one of :data:`TRUST_LEVELS`; ``violation`` is how far the design sits outside what the model can
    vouch for (``0`` when trusted, larger is worse, in no common unit with the measured penalties on purpose);
    ``codes``/``reasons`` are the router's own (``outside_range``, ``sparse_data``, ``too_uncertain``,
    ``straddles_limit``) so a report can say why.

    The default is ``trusted`` with no violation: a caller that knows nothing about a design's trust says nothing,
    and the survivor rule reduces to plain NSGA-II.
    """
    level: str = "trusted"
    violation: float = 0.0
    codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in TRUST_LEVELS:
            raise ValueError(f"TrustVerdict({self.level!r}): level must be one of {TRUST_LEVELS}")
        if not (self.violation >= 0.0 and np.isfinite(self.violation)):
            raise ValueError(f"TrustVerdict({self.level!r}): violation must be finite and >= 0; got {self.violation}")

    @property
    def vouched(self) -> bool:
        """The prediction may be used as a measurement: no code fired."""
        return self.level == "trusted"

    def to_dict(self) -> dict:
        return {"level": self.level, "violation": float(self.violation), "codes": list(self.codes),
                "reasons": list(self.reasons)}


def constrained_select(objectives: np.ndarray, penalties: np.ndarray, mu: int) -> np.ndarray:
    """NSGA-II survivor selection with Deb's constrained-domination rule; the index of the ``mu`` survivors.

    Feasible solutions (``penalty <= 0``) are ranked by non-dominated front and then crowding distance; infeasible
    ones fill the remaining slots, ordered by ascending violation. Returns fewer than ``mu`` indices when there are
    fewer candidates than slots.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    penalties = np.asarray(penalties, dtype=np.float64)
    if len(objectives) != len(penalties):
        raise ValueError(f"{len(objectives)} objective rows but {len(penalties)} penalties")
    mu = int(mu)
    feasible = penalties <= 0.0
    feas_idx = np.where(feasible)[0]
    infeas_idx = np.where(~feasible)[0]

    selected: list[int] = []
    if len(feas_idx) > 0:
        fronts = fast_non_dominated_sort(objectives[feas_idx])
        for front in fronts:
            front_global = feas_idx[front]
            if len(selected) + len(front_global) <= mu:
                selected.extend(front_global.tolist())
            else:
                remaining = mu - len(selected)
                if remaining > 0:
                    cd = crowding_distance(objectives[front_global])
                    order = np.argsort(-cd)          # larger crowding distance preferred
                    selected.extend(front_global[order[:remaining]].tolist())
                break

    if len(selected) < mu and len(infeas_idx) > 0:
        remaining = mu - len(selected)
        order = infeas_idx[np.argsort(penalties[infeas_idx])]
        selected.extend(order[:remaining].tolist())

    return np.array(selected[:mu], dtype=int)


def _verdict_arrays(verdicts: Sequence[TrustVerdict]) -> tuple[np.ndarray, np.ndarray]:
    level = np.array([v.level for v in verdicts], dtype=object)
    violation = np.array([float(v.violation) for v in verdicts], dtype=float)
    return level, violation


def trust_aware_survivors(objectives: np.ndarray, penalties: np.ndarray, verdicts: Sequence[TrustVerdict],
                          mu: int, rule: str = "front") -> np.ndarray:
    """The same survivors, with the trust channel applied; ``rule`` is ``"front"``, ``"penalty"`` or ``"remove"``.

    ``"front"`` (the default, and the decision of #297) fills the vouched band first, by Deb's rule among the
    designs that pass the trust gate, whatever their measured violations; the undetermined band comes next and the
    untrusted band last, both ordered with their own severity added to the violation. Every band contributes only
    the slots the bands before it left, so the population keeps its size ``mu`` and a design the model cannot vouch
    for can still be kept — at the bottom of the ranking, where the next verification run would look.

    ``"penalty"`` folds the severity into the measured violation and leaves the rest to Deb's rule: a single
    ordering in which a *vouched* design that breaks a constraint can lose its slot to an *unvouched* one that does
    not. The two are not comparable — the second one's "does not break it" is the surrogate's guess — which is why
    this is offered as a measurement and not as the default.

    ``"remove"`` keeps the vouched band alone, by Deb's rule. The population then shrinks below ``mu`` whenever the
    applicability domain cannot fill it, and a population of three designs breeds three designs; when *nothing* is
    vouched the filter has nothing to keep, and it falls back to the plain rule rather than to an empty population.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    penalties = np.asarray(penalties, dtype=np.float64)
    if len(objectives) != len(verdicts):
        raise ValueError(f"{len(objectives)} objective rows but {len(verdicts)} trust verdicts")
    level, violation = _verdict_arrays(verdicts)

    if rule == "penalty":
        return constrained_select(objectives, penalties + violation, mu)

    if rule == "remove":
        sel = np.flatnonzero(level == "trusted")
        if not len(sel):
            return constrained_select(objectives, penalties, mu)
        return sel[constrained_select(objectives[sel], penalties[sel], mu)]

    if rule == "front":
        order: list[int] = []
        for band in TRUST_LEVELS:
            sel = np.flatnonzero(level == band)
            if not len(sel):
                continue
            left = int(mu) - len(order)
            if left <= 0:
                break
            take = constrained_select(objectives[sel], penalties[sel] + violation[sel], left)
            order.extend(sel[take].tolist())
        return np.array(order[:int(mu)], dtype=int)

    raise ValueError(f"trust_rule must be 'front', 'penalty' or 'remove'; got {rule!r}")
