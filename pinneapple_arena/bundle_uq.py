"""Uncertainty of a surrogate, and a design search over it: INT-010.

The uncertainty that matters for a surrogate is **epistemic**: it was trained on this much data and stops varying
where the data stops. Measuring it needs several models, not one number, so :func:`seed_ensemble` trains the same
architecture on the same data with different seeds and reports what the disagreement between them looks like --
mean and standard deviation of the held-out relative L2, and the pointwise spread of the predictions themselves,
which is what a confidence band on a field is made of.

The design half is the same models used the other way: :func:`design_optimize` searches a box for the input that
minimises an objective evaluated through the surrogate, reports the best design found and ranks the inputs by rank
correlation with the objective -- the sensitivity a designer actually asks for ("does the fin height or the
thickness matter here?").

Neither is a promise: the ensemble's spread is a measure of disagreement between three models, not a posterior, and
the search is over the box it is given, not over the physics. The docstrings say so, and the report says so.

    >>> from pinneapple_arena.bundle_uq import seed_ensemble, design_optimize
    >>> uq = seed_ensemble(bundle, BenchConfig(field="density", epochs=80), seeds=(0, 1, 2))
    >>> uq.summary()
    >>> best = design_optimize(lambda d: float((d["thickness"] - 0.004) ** 2), {"thickness": (0.001, 0.01)})
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from typing import Any

import numpy as np

__all__ = ["EnsembleReport", "seed_ensemble", "DesignResult", "design_optimize", "spearman"]


@dataclass
class EnsembleReport:
    """What several seeds of the same model say about each other on the held-out part of a trajectory."""

    model: str = ""
    field: str = ""
    seeds: tuple[int, ...] = ()
    rel_l2: list[float] = _dc_field(default_factory=list)
    spread: float = float("nan")          # mean pointwise std across the ensemble, relative to the data's std
    predictions: np.ndarray | None = None  # (n_seeds, n_test, n_x) when `keep_predictions`
    note: str = ""

    @property
    def mean(self) -> float:
        return float(np.mean(self.rel_l2)) if self.rel_l2 else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.rel_l2, ddof=1)) if len(self.rel_l2) > 1 else 0.0

    @property
    def coefficient_of_variation(self) -> float:
        return float(self.std / self.mean) if self.mean else float("nan")

    def summary(self) -> str:
        seeds = ", ".join(str(s) for s in self.seeds)
        lines = [f"{self.model or 'the model'} on '{self.field}', seeds {seeds}:",
                 f"  rel-L2 mean {self.mean:.3e} +- {self.std:.1e} (cv {self.coefficient_of_variation:.2f})",
                 f"  pointwise spread {self.spread:.3e} of the field's own scale"]
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("predictions", None)
        data["seeds"] = list(self.seeds)
        data["mean"] = self.mean
        data["std"] = self.std
        data["coefficient_of_variation"] = self.coefficient_of_variation
        return data


def seed_ensemble(bundle: Any, config: Any, *, model: str = "fno", seeds: Iterable[int] = (0, 1, 2),
                  keep_predictions: bool = True) -> EnsembleReport:
    """Train ``model`` once per seed on the same split and report how much the answers disagree.

    The spread is measured on the held-out time steps and normalised by the standard deviation of the data there, so
    it reads as "the models disagree by this fraction of the signal" -- comparable across fields and units."""
    from .bundle_bench import run_model

    seeds = tuple(int(s) for s in seeds)
    if len(seeds) < 2:
        raise ValueError("an ensemble needs at least two seeds: one model has no spread to report")
    predictions, scores, reference = [], [], None
    for seed in seeds:
        config_seed = config.__class__(**{**config.to_dict(), "seed": seed, "hidden": tuple(config.hidden),
                                          "inputs": tuple(config.inputs)})
        result, pred, ref = run_model(bundle, config_seed, model)
        scores.append(float(result.rel_l2))
        predictions.append(np.asarray(pred, dtype=np.float64))
        reference = ref
    stack = np.stack(predictions)
    scale = float(np.std(np.asarray(reference, dtype=np.float64))) or 1.0
    spread = float(np.mean(np.std(stack, axis=0))) / scale
    note = ("the spread is the disagreement between seeds, not a posterior: it says how much this training "
            "protocol, this data and this architecture leave undetermined")
    return EnsembleReport(model=model, field=config.field, seeds=seeds, rel_l2=scores, spread=spread,
                          predictions=stack if keep_predictions else None, note=note)


@dataclass
class DesignResult:
    """The best design found, what it scored, and which inputs moved the objective."""

    best: dict[str, float] = _dc_field(default_factory=dict)
    value: float = float("nan")
    evaluations: list[tuple[dict[str, float], float]] = _dc_field(default_factory=list)
    sensitivity: dict[str, float] = _dc_field(default_factory=dict)
    note: str = ""

    def summary(self) -> str:
        best = ", ".join(f"{k}={v:.4g}" for k, v in self.best.items())
        ranks = ", ".join(f"{k}:{v:+.2f}" for k, v in sorted(self.sensitivity.items(), key=lambda kv: -abs(kv[1])))
        return (f"best design {best} -> {self.value:.4g} after {len(self.evaluations)} evaluations\n"
                f"  rank correlation with the objective: {ranks or 'n/a'}")


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, the dependence measure that does not assume the objective is linear in the input."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def design_optimize(objective: Callable[[Mapping[str, float]], float], bounds: Mapping[str, tuple[float, float]],
                    *, n: int = 64, seed: int = 0, maximize: bool = False) -> DesignResult:
    """Sample ``n`` designs by Latin hypercube, evaluate ``objective`` and keep the best.

    ``objective`` is whatever the caller wants to minimise -- a surrogate's prediction, a closed-form estimate, a
    cycle of the solver. A budget of 64 evaluations is a search, not an optimiser: it is honest about what a cheap
    surrogate makes possible without a gradient, and the sensitivity it reports is a rank correlation over the
    sample, which is why it is given with the number of evaluations that produced it."""
    names = list(bounds)
    if not names:
        raise ValueError("bounds is empty: nothing to search")
    lows = np.array([bounds[k][0] for k in names], dtype=np.float64)
    highs = np.array([bounds[k][1] for k in names], dtype=np.float64)
    if np.any(highs < lows):
        raise ValueError("each bound is (low, high) with high >= low")
    rng = np.random.default_rng(seed)
    # Latin hypercube: one sample per stratum of every input, with a *separate* permutation per column -- sharing
    # one permutation across the columns lines the strata up and makes the inputs correlated by construction
    perm = np.stack([rng.permutation(n) for _ in names], axis=1)
    strata = (perm + rng.random((n, len(names)))) / n
    designs = lows + strata * (highs - lows)
    evaluations: list[tuple[dict[str, float], float]] = []
    for row in designs:
        point = {k: float(v) for k, v in zip(names, row, strict=True)}
        evaluations.append((point, float(objective(point))))
    sign = -1.0 if maximize else 1.0
    best_point, best_value = min(evaluations, key=lambda item: sign * item[1])
    sensitivity = {k: spearman([p[k] for p, _ in evaluations], [v for _, v in evaluations]) for k in names}
    note = (f"Latin hypercube of {n} designs in {len(names)} input(s); a rank correlation over that sample is a "
            "sensitivity indicator, not a global analysis")
    return DesignResult(best=best_point, value=best_value, evaluations=evaluations, sensitivity=sensitivity,
                        note=note)
