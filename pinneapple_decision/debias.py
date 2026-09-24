"""L0 debiasing without labels (AnyJev-like; own implementation, see README).

Exactly what is implemented here:

1. **Cyclic-shift marginalization (position bias).** For K feasible options the
   backend scores the list in K cyclic rotations, so every option appears once
   at every position (``max_permutations`` caps this with evenly spaced shifts).
   Each rotation's scores are turned into log-probabilities (log-softmax) and
   mapped back to option names. Combination:

   * ``combine="geometric"`` (default): mean of the log-probabilities per option,
     then softmax. If the position bias is additive in logit space
     (``score = s(option) + b(position)``) this removes it *exactly* with the
     full K rotations, and the result no longer depends on the listed order.
   * ``combine="mean"``: arithmetic mean of the per-rotation probabilities
     (the form of Zheng et al., 2024).

2. **Prior correction (label prior).** ``p(option) ∝ p(option) / prior(option)**strength``.

   * ``prior="content_free"`` (default): contextual calibration (Zhao et al.,
     2021). The prior is the rotation-averaged distribution the backend gives on
     *neutral* states (by default an empty problem and ``{"description": "N/A"}``),
     averaged over those states. Computed per call; no labels involved.
   * ``prior="batch"``: batch calibration (Zhou et al., 2024). The prior is the
     running mean of the distributions produced on real states for the same
     choice (same name + same feasible option set), used only once
     ``min_prior_n`` states were seen; before that no correction is applied and
     the diagnostics say so. Keep one :class:`BatchPrior` per decider.
   * ``prior="none"``.

   Backends that declare ``has_label_prior = False`` (the deterministic rule
   backend: its scores are a function of the problem only, and an empty problem
   is not a neutral input for it) skip step 2; the diagnostics record why.

What L0 does not do: it does not make the distribution calibrated. A backend
that is overconfident stays overconfident. That is L1 and needs labelled
outcomes.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ScoreFn = Callable[[Sequence[str]], Sequence[float]]  # options in shown order -> scores in same order


def log_softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    z = m + math.log(sum(math.exp(x - m) for x in xs))
    return [x - z for x in xs]


def softmax_dict(logits: Mapping[str, float]) -> Dict[str, float]:
    keys = list(logits)
    lp = log_softmax([logits[k] for k in keys])
    return {k: math.exp(v) for k, v in zip(keys, lp)}


def normalize(p: Mapping[str, float]) -> Dict[str, float]:
    s = sum(p.values())
    if s <= 0 or not math.isfinite(s):
        raise ValueError(f"cannot normalize distribution {dict(p)}")
    return {k: v / s for k, v in p.items()}


def rotations(options: Sequence[str], max_permutations: Optional[int] = None) -> List[List[str]]:
    k = len(options)
    n = k if not max_permutations or max_permutations >= k else max(1, max_permutations)
    shifts = sorted({round(i * k / n) % k for i in range(n)})
    return [list(options[s:]) + list(options[:s]) for s in shifts]


def raw_distribution(score_fn: ScoreFn, options: Sequence[str]) -> Dict[str, float]:
    scores = list(score_fn(list(options)))
    _check_scores(scores, options)
    return softmax_dict(dict(zip(options, scores)))


def permutation_marginalized(
    score_fn: ScoreFn,
    options: Sequence[str],
    *,
    combine: str = "geometric",
    max_permutations: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Distribution averaged over cyclic rotations of the option order."""
    if combine not in ("geometric", "mean"):
        raise ValueError("combine must be 'geometric' or 'mean'")
    rots = rotations(options, max_permutations)
    per_rot: List[Dict[str, float]] = []
    for order in rots:
        scores = list(score_fn(order))
        _check_scores(scores, order)
        per_rot.append(dict(zip(order, log_softmax(scores))))
    if combine == "geometric":
        avg = {o: sum(r[o] for r in per_rot) / len(per_rot) for o in options}
        dist = softmax_dict(avg)
    else:
        dist = normalize({o: sum(math.exp(r[o]) for r in per_rot) / len(per_rot) for o in options})
    argmaxes = [max(r, key=r.get) for r in per_rot]
    diag = {
        "n_rotations": len(rots),
        "combine": combine,
        "rotation_argmaxes": argmaxes,
        # fraction of rotations whose raw argmax differs from the debiased argmax
        "order_flip_rate": sum(a != max(dist, key=dist.get) for a in argmaxes) / len(argmaxes),
    }
    return dist, diag


def apply_prior(dist: Mapping[str, float], prior: Mapping[str, float], strength: float = 1.0) -> Dict[str, float]:
    eps = 1e-12
    return normalize({o: dist[o] / (max(prior[o], eps) ** strength) for o in dist})


class BatchPrior:
    """Running mean of L0 distributions per (choice name, feasible option set)."""

    def __init__(self, min_prior_n: int = 8):
        self.min_prior_n = min_prior_n
        self._sum: Dict[Tuple[str, frozenset], Dict[str, float]] = {}
        self._n: Dict[Tuple[str, frozenset], int] = {}

    def key(self, choice: str, options: Sequence[str]) -> Tuple[str, frozenset]:
        return (choice, frozenset(options))

    def get(self, choice: str, options: Sequence[str]) -> Optional[Dict[str, float]]:
        k = self.key(choice, options)
        n = self._n.get(k, 0)
        if n < self.min_prior_n:
            return None
        return {o: v / n for o, v in self._sum[k].items()}

    def update(self, choice: str, dist: Mapping[str, float]) -> None:
        k = self.key(choice, list(dist))
        acc = self._sum.setdefault(k, {o: 0.0 for o in dist})
        for o, v in dist.items():
            acc[o] += v
        self._n[k] = self._n.get(k, 0) + 1

    def count(self, choice: str, options: Sequence[str]) -> int:
        return self._n.get(self.key(choice, options), 0)


def _check_scores(scores: Sequence[float], options: Sequence[str]) -> None:
    if len(scores) != len(options):
        raise ValueError(f"backend returned {len(scores)} scores for {len(options)} options")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError(f"backend returned non-finite scores: {list(scores)}")


__all__ = [
    "BatchPrior",
    "apply_prior",
    "log_softmax",
    "permutation_marginalized",
    "raw_distribution",
    "rotations",
    "softmax_dict",
]
