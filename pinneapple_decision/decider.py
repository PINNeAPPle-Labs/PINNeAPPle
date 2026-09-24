"""PhysicsDecider: constraints -> scoring -> L0 debiasing -> Decision."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .backends import RuleBasedBackend, ScoringBackend
from .constraints import NoFeasibleOptionError, apply_constraints
from .debias import BatchPrior, apply_prior, permutation_marginalized, raw_distribution
from .schema import Decision, DecisionLevel, DecisionState, PhysicsChoice

#: Neutral states for the content-free prior (contextual calibration).
NEUTRAL_STATES = (DecisionState(problem={}), DecisionState(problem={"description": "N/A"}))


class PhysicsDecider:
    """Decide which option of a :class:`PhysicsChoice` to run next.

    Parameters
    ----------
    backend : ScoringBackend, default :class:`RuleBasedBackend`
    debias : bool
        ``True`` -> level ``L0`` (cyclic-shift marginalization + prior correction);
        ``False`` -> level ``raw`` (one ordering, plain softmax).
    prior : ``"content_free"`` | ``"batch"`` | ``"none"``
    prior_strength : exponent on the prior (1.0 = full correction).
    combine : ``"geometric"`` | ``"mean"`` (see ``debias.py``).
    max_permutations : cap on the number of cyclic shifts.
    """

    def __init__(
        self,
        backend: Optional[ScoringBackend] = None,
        *,
        debias: bool = True,
        prior: str = "content_free",
        prior_strength: float = 1.0,
        combine: str = "geometric",
        max_permutations: Optional[int] = None,
        neutral_states: Sequence[DecisionState] = NEUTRAL_STATES,
        min_prior_n: int = 8,
    ):
        if prior not in ("content_free", "batch", "none"):
            raise ValueError("prior must be 'content_free', 'batch' or 'none'")
        self.backend = backend or RuleBasedBackend()
        self.debias = debias
        self.prior = prior
        self.prior_strength = prior_strength
        self.combine = combine
        self.max_permutations = max_permutations
        self.neutral_states = tuple(neutral_states)
        self.batch_prior = BatchPrior(min_prior_n)

    def _score_fn(self, state: DecisionState, choice: PhysicsChoice):
        return lambda order: self.backend.score(state, choice, order)

    def distribution(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]):
        diag: Dict[str, Any] = {}
        if not self.debias:
            return raw_distribution(self._score_fn(state, choice), options), DecisionLevel.RAW, diag

        dist, perm_diag = permutation_marginalized(
            self._score_fn(state, choice), options, combine=self.combine, max_permutations=self.max_permutations)
        diag.update(perm_diag)

        if not getattr(self.backend, "has_label_prior", True):
            diag["prior_method"] = "none (backend declares no label prior)"
        elif self.prior == "content_free":
            priors = [
                permutation_marginalized(self._score_fn(ns, choice), options, combine=self.combine,
                                         max_permutations=self.max_permutations)[0]
                for ns in self.neutral_states
            ]
            prior = {o: sum(p[o] for p in priors) / len(priors) for o in options}
            dist = apply_prior(dist, prior, self.prior_strength)
            diag.update(prior_method="content_free", prior=prior)
        elif self.prior == "batch":
            prior = self.batch_prior.get(choice.name, options)
            self.batch_prior.update(choice.name, dist)
            if prior is None:
                diag["prior_method"] = (f"none (batch prior has {self.batch_prior.count(choice.name, options) - 1}"
                                        f" < {self.batch_prior.min_prior_n} states)")
            else:
                dist = apply_prior(dist, prior, self.prior_strength)
                diag.update(prior_method="batch", prior=prior)
        else:
            diag["prior_method"] = "none"
        diag["prior_strength"] = self.prior_strength
        return dist, DecisionLevel.L0, diag

    def decide(self, problem_or_state: Any, choice: PhysicsChoice) -> Decision:
        state = DecisionState.coerce(problem_or_state)
        feasible, excluded = apply_constraints(choice, state)
        if not feasible:
            raise NoFeasibleOptionError(choice.name, excluded)
        if len(feasible) == 1:
            dist, level, diag = {feasible[0]: 1.0}, DecisionLevel.L0 if self.debias else DecisionLevel.RAW, {
                "note": "single feasible option after constraints"}
        else:
            dist, level, diag = self.distribution(state, choice, feasible)
        if state.facts:
            diag["problem_facts"] = state.facts
        # Ties (up to float noise) break alphabetically, so the listed order never matters.
        selected = max(sorted(feasible), key=lambda o: round(dist[o], 12))
        return Decision(
            choice=choice.name,
            question=choice.question,
            selected=selected,
            probabilities=dist,
            level=level,
            requires_validation=True,
            excluded=excluded,
            rationale=self.backend.rationale(state, choice, feasible),
            backend=self.backend.name,
            diagnostics=diag,
        )


__all__ = ["NEUTRAL_STATES", "PhysicsDecider"]
