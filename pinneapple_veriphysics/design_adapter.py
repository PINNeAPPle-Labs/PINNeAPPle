"""Solver/model-neutral adapters for physics-validated design optimization.

Wraps an existing trained PINN, FNO, DeepONet or numerical surrogate; does not
train a new model or imply that predictions have been physically validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .design_harness import DesignVerificationHarness


@dataclass(frozen=True)
class SurrogateAdapter:
    predict: Callable
    name: str = "external"
    error_scale: Callable | None = None

    def evaluate(self, candidates):
        x = np.asarray(candidates, dtype=float)
        if x.ndim != 2 or not np.all(np.isfinite(x)):
            raise ValueError("candidates must be a finite 2D array")
        pred = np.asarray(self.predict(x), dtype=float).reshape(-1)
        if pred.shape != (len(x),) or not np.all(np.isfinite(pred)):
            raise ValueError("surrogate must return one finite scalar per candidate")
        if self.error_scale is None:
            raise ValueError("explicit calibration/error_scale callback required")
        error = np.asarray(self.error_scale(x), dtype=float)
        if error.ndim == 0:
            error = np.full(len(x), float(error))
        error = error.reshape(-1)
        if error.shape != pred.shape or not np.all(np.isfinite(error)) or np.any(error < 0):
            raise ValueError("invalid error scale")
        return pred, error


def freeze_design_selection(adapter, candidates, train_points, bounds, *,
                            baseline_prediction, threshold, top_k=16,
                            physics_check=None, ood_distance=.42,
                            uncertainty_multiplier=.15, ledger=None):
    """Select candidates without querying any independent reference solver.

    Returns frozen indices and screening decisions. Caller must perform
    independent reference adjudication AFTER this function returns.
    """
    x = np.asarray(candidates, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(bounds) or not len(x):
        raise ValueError("invalid candidate matrix")
    if not 1 <= top_k <= len(x):
        raise ValueError("invalid top_k")
    predictions, errors = adapter.evaluate(x)
    harness = DesignVerificationHarness(
        bounds, train_points, baseline_prediction, threshold=threshold,
        ood_distance=ood_distance, uncertainty_multiplier=uncertainty_multiplier,
        ledger=ledger,
    )
    selected = []
    decisions = {}
    for i in np.argsort(-predictions, kind="stable"):
        i = int(i)
        if len(selected) >= top_k:
            break
        decision = harness.check(
            x[i], float(predictions[i]), float(errors[i]),
            physics_check=physics_check, check_id=f"selection/{i}",
        )
        decisions[i] = decision.to_dict()
        if decision.status == "provisional":
            selected.append(i)
    return {
        "model": adapter.name,
        "selected_indices": selected,
        "decisions": decisions,
        "predictions": predictions.tolist(),
        "error_scales": errors.tolist(),
        "candidate_budget": len(x),
        "reference_calls_selection": 0,
        "selection_frozen": True,
        "warning": "Provisional selection is not physical verification",
    }
