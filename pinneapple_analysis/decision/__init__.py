"""pinneapple_analysis.decision -- decision-level validation of a surrogate; see ``metrics`` and ``card``.

A field error says how wrong a surrogate is, not whether it changes the design. This package holds the numbers
that answer the second question, measured against the reference solver on the customer's own design set:

``flip_rate``, ``rank_preservation``, ``regret``, ``failure_map``
    the fraction of design pairs whose order the surrogate reverses, the rank correlations and the fraction of
    the true top-k kept, the objective value given up by the design it picks, and the map of where the error sits
    in the design space.
``decision_card``, ``DecisionCard``, ``DecisionThresholds``
    the same numbers as a one page card with a verdict, printable with ``to_markdown()``.
``decision_report``, ``DecisionReport``
    the ensemble report (#219): the distribution of the quantity of interest, the probability of violating each hard
    constraint, the inputs that drive the risk and a recommendation with its validity domain and trust status.

The reference values must come from a test split disjoint from the split the surrogate was fitted on, and the
flip rate is only a measurement when it is quoted with the tolerance that produced it.

.. code-block:: python

    from pinneapple_analysis.decision import decision_card
    card = decision_card(reference, surrogate, names=cases, margin=0.02)
    print(card.summary())
"""
from __future__ import annotations

from .card import DecisionCard, DecisionThresholds, decision_card
from .metrics import (
    FailureMap,
    FlipRate,
    RankPreservation,
    Regret,
    failure_map,
    flip_rate,
    rank_preservation,
    regret,
)
from .report import Combination, DecisionReport, EnsembleMember, InputDriver, decision_report

__all__ = [
    "FlipRate",
    "RankPreservation",
    "Regret",
    "FailureMap",
    "flip_rate",
    "rank_preservation",
    "regret",
    "failure_map",
    "DecisionThresholds",
    "DecisionCard",
    "decision_card",
    "Combination",
    "EnsembleMember",
    "InputDriver",
    "DecisionReport",
    "decision_report",
]
