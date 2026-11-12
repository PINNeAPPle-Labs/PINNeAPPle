"""The published FluidsBench reductions, computed here so a participant can see the number before submitting it.

These are the equations of the benchmark's metric reference, implemented from the definitions (not copied from their
code): the weighted MAE / MSE / RMSE / R2, the relative L1 and L2 in percent, the chunk-safe sufficient statistics for
relative L2, the per-geometry macro-average, and the composite the leaderboard ranks on. Two rules are enforced rather
than assumed, because they are where a comparison stops being fair:

* a chunk-level L2 is never averaged — the additive numerator and denominator are summed first
  (:func:`relative_l2_from_statistics`);
* a case metric is computed per entity set and the cases are averaged afterwards, so a fine mesh does not outvote a
  coarse one (:func:`aggregate`).

The number that goes into a submission is the one the official evaluator produces; :meth:`Checkout.validate_submission`
runs it. This module says what any other implementation must reproduce, and is what the adapter uses to check a package
before it leaves the machine.

    from pinneapple_arena.fluidsbench import relative_l2_statistics, relative_l2_from_statistics

    first = relative_l2_statistics(t1, p1, w1, weighting="support_weights")
    second = relative_l2_statistics(t2, p2, w2, weighting="support_weights")
    value = relative_l2_from_statistics([first, second])          # one square root over both chunks
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np

from .support import CaseSupport, MetricBinding

__all__ = ["REDUCTIONS", "Statistics", "SupportScore", "MetricError", "relative_l1", "relative_l2",
           "relative_l2_statistics", "relative_l2_from_statistics", "weighted_mae", "weighted_mse", "weighted_rmse",
           "r2", "reduction_value", "macro_average", "aggregate", "competition_rank", "score_support",
           "score_release"]

REL_L2 = "relative_l2_percent"
REDUCTIONS = frozenset({"relative_l1_percent", REL_L2, "mae", "mse", "rmse", "r2"})
_MAX_MACRO_REL_TOL = 1e-9


class MetricError(ValueError):
    """An input the published definitions reject: empty, non-finite, a non-positive weight total, a zero denominator."""


@dataclass(frozen=True)
class Statistics:
    """The additive sufficient statistics of one relative-L2 metric on one case support."""
    reduction: str
    weighting: str
    dataset_weighting: str
    numerator: float
    denominator: float
    entity_count: int
    total_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {"reduction": self.reduction, "weighting": self.weighting,
                "dataset_weighting": self.dataset_weighting, "numerator": self.numerator,
                "denominator": self.denominator, "entity_count": self.entity_count,
                "total_weight": self.total_weight}


def _prepare(truth: Any, prediction: Any, weights: Any, *, weighting: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one field and return the flattened truth, prediction, spatial weights and repeated weights.

    A spatial weight multiplies the complete squared vector magnitude, so it is repeated once per component."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim == 1:
        truth = truth[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if truth.shape != prediction.shape:
        raise MetricError(f"truth {truth.shape} and prediction {prediction.shape} must have the same shape")
    if truth.size == 0:
        raise MetricError("metric arrays cannot be empty")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise MetricError("metric arrays must contain only finite values")
    spatial = np.ones(truth.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, np.float64)
    if spatial.shape != (truth.shape[0],):
        raise MetricError(f"weights must hold one value per entity ({truth.shape[0]}), got {spatial.shape}")
    if not np.all(np.isfinite(spatial)):
        raise MetricError("weights must contain only finite values")
    if np.any(spatial < 0):
        raise MetricError("weights cannot be negative")
    if weighting == "uniform":
        spatial = np.ones_like(spatial)
    elif weighting != "support_weights":
        raise MetricError(f"unknown weighting {weighting!r}; use uniform or support_weights")
    if not float(spatial.sum()) > 0:
        raise MetricError("weights must have a positive sum")
    return truth.reshape(-1), prediction.reshape(-1), spatial, np.repeat(spatial, truth.shape[1])


def weighted_mae(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights") -> float:
    t, p, _, w = _prepare(truth, prediction, weights, weighting=weighting)
    return float(np.sum(w * np.abs(p - t)) / np.sum(w))


def weighted_mse(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights") -> float:
    t, p, _, w = _prepare(truth, prediction, weights, weighting=weighting)
    return float(np.sum(w * np.square(p - t)) / np.sum(w))


def weighted_rmse(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights") -> float:
    return float(math.sqrt(weighted_mse(truth, prediction, weights, weighting=weighting)))


def relative_l1(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights") -> float:
    """``100 * sum(w |p - t|) / sum(w |t|)``, in percent."""
    t, p, _, w = _prepare(truth, prediction, weights, weighting=weighting)
    denominator = float(np.sum(w * np.abs(t)))
    if denominator == 0:
        raise MetricError("relative L1 is undefined when the weighted ground-truth norm is zero")
    return float(100.0 * np.sum(w * np.abs(p - t)) / denominator)


def r2(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights") -> float:
    """Weighted R² with the weighted ground-truth mean; constant ground truth is undefined, not zero."""
    t, p, spatial, w = _prepare(truth, prediction, weights, weighting=weighting)
    del spatial
    mean = float(np.sum(w * t) / np.sum(w))
    denominator = float(np.sum(w * np.square(t - mean)))
    if denominator == 0:
        raise MetricError("R² is undefined for constant ground truth")
    return float(1.0 - np.sum(w * np.square(p - t)) / denominator)


def relative_l2_statistics(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights",
                           dataset_weighting: str = "") -> Statistics:
    """Accumulate ``N = Σ w‖p−t‖²``, ``D = Σ w‖t‖²``, the entity count and the total spatial weight."""
    t, p, spatial, w = _prepare(truth, prediction, weights, weighting=weighting)
    error = p - t
    return Statistics(reduction=REL_L2, weighting=weighting, dataset_weighting=dataset_weighting,
                      numerator=float(np.sum(w * np.square(error))),
                      denominator=float(np.sum(w * np.square(t))),
                      entity_count=int(len(spatial)), total_weight=float(np.sum(spatial)))


def relative_l2(truth: Any, prediction: Any, weights: Any = None, *, weighting: str = "support_weights") -> float:
    return relative_l2_from_statistics([relative_l2_statistics(truth, prediction, weights, weighting=weighting)])


def relative_l2_from_statistics(statistics: Sequence[Statistics | Mapping[str, Any]]) -> float:
    """``100 sqrt(Σ_c N_c / Σ_c D_c)`` over one or more chunks — never the average of chunk L2 values."""
    if not statistics:
        raise MetricError("relative L2 needs at least one chunk of statistics")
    numerator = denominator = total_weight = 0.0
    entities = 0
    for chunk in statistics:
        values = chunk.as_dict() if isinstance(chunk, Statistics) else dict(chunk)
        n, d = float(values["numerator"]), float(values["denominator"])
        count, weight = int(values["entity_count"]), float(values["total_weight"])
        if not (math.isfinite(n) and n >= 0) or not (math.isfinite(d) and d >= 0):
            raise MetricError("relative L2 statistics must be finite and non-negative")
        if count < 1 or not math.isfinite(weight) or weight < 0:
            raise MetricError("relative L2 statistics describe an empty support")
        numerator, denominator, entities, total_weight = (numerator + n, denominator + d, entities + count,
                                                          total_weight + weight)
    if entities < 1 or not total_weight > 0:
        raise MetricError("relative L2 statistics describe an empty support")
    if not denominator > 0:
        raise MetricError("relative L2 ground-truth denominator is zero")
    value = 100.0 * math.sqrt(numerator / denominator)
    if not math.isfinite(value):
        raise MetricError("relative L2 result is non-finite")
    return value


def reduction_value(reduction: str, truth: Any, prediction: Any, weights: Any = None, *,
                    weighting: str = "support_weights", dataset_weighting: str = ""
                    ) -> tuple[float, Statistics | None]:
    """One metric binding applied to one field; returns the value and, for relative L2, its sufficient statistics."""
    if reduction == REL_L2:
        statistics = relative_l2_statistics(truth, prediction, weights, weighting=weighting,
                                            dataset_weighting=dataset_weighting)
        return relative_l2_from_statistics([statistics]), statistics
    if reduction == "relative_l1_percent":
        return relative_l1(truth, prediction, weights, weighting=weighting), None
    if reduction == "mae":
        return weighted_mae(truth, prediction, weights, weighting=weighting), None
    if reduction == "mse":
        return weighted_mse(truth, prediction, weights, weighting=weighting), None
    if reduction == "rmse":
        return weighted_rmse(truth, prediction, weights, weighting=weighting), None
    if reduction == "r2":
        return r2(truth, prediction, weights, weighting=weighting), None
    raise MetricError(f"unknown reduction {reduction!r}; the contract publishes {sorted(REDUCTIONS)}")


def macro_average(values: Iterable[float]) -> float:
    """Equal weight per geometry, whatever its resolution."""
    collected = [float(v) for v in values]
    if not collected:
        raise MetricError("a macro-average needs at least one case")
    if not all(math.isfinite(v) for v in collected):
        raise MetricError("a macro-average needs finite case values")
    return float(np.mean(collected))


@dataclass(frozen=True)
class SupportScore:
    """Every metric the contract binds to one case support, with the coverage it was scored on."""
    case_id: str
    support_id: str
    metric_values: Mapping[str, float]
    statistics: Mapping[str, Statistics] = field(default_factory=dict)
    coverage: Mapping[str, Any] = field(default_factory=dict)


def score_support(case: CaseSupport, predictions: Mapping[str, np.ndarray], *,
                  bindings: Sequence[MetricBinding] | None = None,
                  coverage: Mapping[str, Any] | None = None) -> SupportScore:
    """Apply every metric binding of the support's definition to one case's predictions.

    ``predictions`` maps each component's ``prediction_field`` to an array over the support's entities, in the support's
    ``support_id_ascending`` order — which is what :func:`~pinneapple_arena.fluidsbench.mapping.map_to_support`
    returns."""
    bindings = tuple(bindings if bindings is not None else case.definition.bindings)
    values: dict[str, float] = {}
    statistics: dict[str, Statistics] = {}
    for binding in bindings:
        quantity = case.definition.quantity(binding.quantity_id)
        truth = case.target(quantity.id)
        columns = []
        for component in quantity.components:
            if component.prediction_field not in predictions:
                raise MetricError(f"case {case.case_id!r} has no prediction for {component.prediction_field!r}")
            column = np.asarray(predictions[component.prediction_field], dtype=np.float64)
            column = column[:, None] if column.ndim == 1 else column
            if column.shape[0] != truth.shape[0]:
                raise MetricError(f"{component.prediction_field!r} has {column.shape[0]} entities, the support has "
                                  f"{truth.shape[0]}")
            columns.append(column)
        prediction = np.concatenate(columns, axis=1)
        weights = case.weights if binding.weighting == "support_weights" else None
        value, stats = reduction_value(binding.reduction, truth, prediction, weights,
                                       weighting=binding.weighting, dataset_weighting=binding.dataset_weighting)
        values[binding.metric_id] = value
        if stats is not None:
            statistics[binding.metric_id] = stats
    return SupportScore(case_id=case.case_id, support_id=case.support_id, metric_values=values,
                        statistics=statistics, coverage=dict(coverage or {}))


def score_release(release, predictions: Mapping[tuple[str, str], Mapping[str, np.ndarray]], *,
                  coverage: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> list[SupportScore]:
    """Score a whole support release: ``predictions[(case_id, support_id)]`` per case and support."""
    out = []
    for case_id in release.case_ids:
        for support_id in release.support_ids(case_id):
            key = (case_id, support_id)
            if key not in predictions:
                raise MetricError(f"no predictions for case {case_id!r} support {support_id!r}")
            case = release.support(case_id, support_id)
            out.append(score_support(case, predictions[key], coverage=(coverage or {}).get(key)))
    return out


def aggregate(scores: Sequence[SupportScore], *, aggregation: str = "per_geometry_then_macro_average",
              reductions: Mapping[str, str] | None = None) -> dict[str, float]:
    """Reduce the per-case values of every metric the way the contract says.

    ``per_geometry_then_macro_average`` (the default for field metrics) averages the case values. For a relative L2,
    ``flatten_all_aligned_field_values`` instead sums the additive statistics of every case before the one square root,
    which is the only reduction under which the cases' resolutions cancel."""
    if not scores:
        raise MetricError("at least one case is required")
    metric_ids = sorted({metric for score in scores for metric in score.metric_values})
    out: dict[str, float] = {}
    for metric_id in metric_ids:
        if aggregation == "flatten_all_aligned_field_values":
            chunks = [s.statistics[metric_id] for s in scores if metric_id in s.statistics]
            if chunks and len(chunks) == len([s for s in scores if metric_id in s.metric_values]):
                out[metric_id] = relative_l2_from_statistics(chunks)
                continue
        out[metric_id] = macro_average([s.metric_values[metric_id] for s in scores if metric_id in s.metric_values])
    if reductions and aggregation == "flatten_all_aligned_field_values":
        unresolved = [m for m in reductions if m in out and reductions[m] != REL_L2]
        if unresolved:
            raise MetricError(f"flatten_all_aligned_field_values is defined for relative L2 only; it cannot reduce "
                              f"{unresolved}")
    return out


def competition_rank(values: Sequence[float], *, direction: str = "higher", decimal_places: int = 1) -> list[int]:
    """Rank rows the way the leaderboard does: round half-up, then equal values share a competition rank (1, 2, 2, 4)."""
    if direction not in ("higher", "lower"):
        raise MetricError(f"direction must be higher or lower, got {direction!r}")
    rounded = [Decimal(str(float(v))).quantize(Decimal(1).scaleb(-decimal_places), rounding=ROUND_HALF_UP)
               for v in values]
    ordered = sorted(set(rounded), reverse=(direction == "higher"))
    position = {value: index + 1 for index, value in enumerate(ordered)}
    return [position[value] for value in rounded]


def check_macro_identity(case_values: Sequence[float], aggregate_value: float, *, tolerance: float = _MAX_MACRO_REL_TOL
                         ) -> None:
    """A package's aggregate value must be the macro-average of its own case values; the official validator refuses
    otherwise, and so does this."""
    expected = macro_average(case_values)
    scale = max(abs(expected), 1e-12)
    if abs(expected - float(aggregate_value)) > tolerance * scale:
        raise MetricError(f"aggregate {aggregate_value!r} is not the macro-average {expected!r} of its case values")
