"""Stage 6: what the model is worth — the held-out error, a calibrated confidence, and the gates that ran.

Two different questions, and the pipeline answers both because the first one alone has never been enough to decide
anything:

*how far is the model from what it saw?* — the OOD score and its calibrated confidence, so a number that reads 0.87
means something to somebody who is not the person who built the model;

*is the answer any good?* — the held-out error **against the constant-field baseline**, and the gates: the checks that
ask whether the prediction obeys the physics rather than merely matching the reference.

The gates come from :mod:`pinneapple_veriphysics.surrogate_gates`, which needs a structured bundle (a field on a grid
with quadrature weights) for conservation, boundary and residual. A design space here is a point cloud on a body's
surface, so those three are reported as **not applicable with the reason** rather than quietly skipped, and a family
that can state a check of its own declares it in ``identity_checks``. What always runs is the accuracy gate and the
per-design comparison against the baseline, which together answer "is this model better than nothing on this body".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["Calibration", "calibrate", "held_out_report", "point_gates", "confidence"]

CALIBRATION_SCHEMA = "pinneapple.lpm.calibration/1"


@dataclass
class Calibration:
    """A monotone map from the raw OOD score to a confidence in [0, 1], with the knots that define it.

    The raw score is ``distance to the nearest training design / the 0.95 quantile of the training nearest-neighbour
    distances`` (:meth:`pinneapple_neural.lpm.LargePhysicsModel.ood`), so it is around 1 inside the training set and
    grows outside it. The knots come from the training designs themselves: the median reads 1.0, the tail of the
    training set reads lower, and the decay beyond it is exponential. The knots are stored so the table can be
    published, which is the part that makes the number usable by somebody else.
    """

    knots: list[tuple[float, float]]
    schema: str = CALIBRATION_SCHEMA

    def __call__(self, raw: float) -> float:
        x = [k[0] for k in self.knots]
        y = [k[1] for k in self.knots]
        return float(np.clip(np.interp(float(raw), x, y), 0.0, 1.0))

    def table(self) -> str:
        head = "raw OOD score -> confidence"
        rows = [f"{x:8.2f} -> {y:5.2f}" for x, y in self.knots]
        return "\n".join([head] + rows)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "knots": [list(k) for k in self.knots]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        return cls(knots=[(float(a), float(b)) for a, b in d["knots"]], schema=d.get("schema", CALIBRATION_SCHEMA))


def calibrate(model: Any, designs: list[Any]) -> Calibration:
    """Fit the confidence map on designs the model was trained on.

    In-distribution inputs must read high — that is the calibration's whole job — so the knots are quantiles of the
    score on the training set: the median reads 0.99, the 95th percentile 0.90, the maximum 0.65, and beyond the
    training set the confidence decays (1.5x the worst training score reads 0.40, 3x reads 0.10). Those numbers are a
    choice, not a measurement, and they are stated here so they can be replaced by one: with held-out bodies whose
    true error is known, the same function can be fitted to the error instead of to the quantiles.
    """
    if len(designs) < 2:
        raise ValueError("a calibration needs at least two training designs")
    model.eval()
    scores = np.array([float(model.ood(d)["ood_score"]) for d in designs], dtype=float)
    q50, q95, top = (float(np.quantile(scores, 0.5)), float(np.quantile(scores, 0.95)), float(scores.max()))
    top = max(top, q95, 1e-6)
    knots = [(0.0, 1.0), (q50, 0.99), (q95, 0.90), (top, 0.65),
             (1.5 * top, 0.40), (3.0 * top, 0.10), (10.0 * top, 0.01)]
    # the knots must be strictly increasing in x, or the interpolation is undefined
    merged: list[tuple[float, float]] = []
    for x, y in knots:
        if merged and x <= merged[-1][0]:
            continue
        merged.append((x, y))
    return Calibration(knots=merged)


def confidence(model: Any, design: Any, calibration: Calibration | None = None) -> dict[str, Any]:
    """One design's confidence, with the reasons behind it."""
    ood = model.ood(design)
    raw = float(ood["ood_score"])
    conf = 1.0 if calibration is None and not ood["ood"] else (calibration(raw) if calibration else 0.0)
    return {"ood_score": raw, "confidence": float(conf), "ood": bool(ood["ood"]),
            "reasons": list(ood.get("reasons") or [])}


def held_out_report(model: Any, family: Any, hold_designs: list[Any], train_designs: list[Any], *,
                    calibration: Calibration | None = None) -> dict[str, Any]:
    """What a card needs: per-design error, the baseline beside it, the p90, and the confidence of each body.

    The baseline is the constant field — the mean of the training targets — scored on the same held-out designs. It is
    the number that turns "0.36 relative L2" into a decision: against a baseline of 0.95 the model learnt something,
    against a baseline of 0.30 it did not.

    ``relative_l2`` is the **global** relative L2 (``sqrt(sum ||err||^2 / sum ||target||^2)``) — the number the
    campaign's cards report and the one the tolerance is applied to. ``relative_l2_p90`` and the per-design rows are
    the spread beside it, and the two are not interchangeable: a design whose field is nearly constant has a small
    target norm, so its own relative error is large while contributing almost nothing to the global number. On the
    airfoil family the mean of the per-design errors is 1.31 where the global one is 0.71.
    """
    model.eval()
    stacked = torch.cat([d.target for d in train_designs if d.target is not None])
    baseline_value = stacked.mean(0)

    rows, errors, baseline_errors = [], [], []
    squared_error = squared_baseline = squared_norm = 0.0
    for d in hold_designs:
        if d.target is None:
            continue
        with torch.no_grad():
            pred = model.predict(d)["mean"]
        ref = d.target
        norm = float(ref.norm())
        err_abs = float((pred - ref).norm())
        base_abs = float((baseline_value.expand_as(ref) - ref).norm())
        errors.append(err_abs / max(norm, 1e-12))
        baseline_errors.append(base_abs / max(norm, 1e-12))
        squared_error += err_abs ** 2
        squared_baseline += base_abs ** 2
        squared_norm += norm ** 2
        rows.append({"label": getattr(d, "label", ""), "relative_l2": err_abs / max(norm, 1e-12),
                     "baseline_relative_l2": base_abs / max(norm, 1e-12),
                     "beats_baseline": bool(err_abs < base_abs),
                     **confidence(model, d, calibration)})

    if not rows:
        raise ValueError("no held-out design carries a target: there is nothing to report")
    errors_a = np.asarray(errors)
    tolerance = float(getattr(family, "tolerance", 0.05))
    global_error = float(np.sqrt(squared_error / squared_norm)) if squared_norm > 0 else float("nan")
    global_baseline = float(np.sqrt(squared_baseline / squared_norm)) if squared_norm > 0 else float("nan")
    return {
        "n_holdout_designs": len(rows),
        "n_train_designs": len(train_designs),
        "relative_l2": global_error,
        "relative_l2_mean_per_design": float(errors_a.mean()),
        "relative_l2_median": float(np.median(errors_a)),
        "relative_l2_p90": float(np.quantile(errors_a, 0.9)),
        "relative_l2_max": float(errors_a.max()),
        "constant_field_baseline_relative_l2": global_baseline,
        "within_tolerance": bool(global_error <= tolerance),
        "tolerance": tolerance,
        "beats_baseline": bool(global_error < global_baseline),
        "designs_above_baseline": int(sum(1 for r in rows if not r["beats_baseline"])),
        "metric": "global relative L2 over every held-out point: sqrt(sum of squared error / sum of squared target). "
                  "The per-design rows and the p90 are the spread beside it, not a substitute for it",
        "calibration": calibration.to_dict() if calibration else None,
        "per_design": rows,
    }


def point_gates(report: dict[str, Any], family: Any, *, designs: list[Any] | None = None,
                predictions: list[Any] | None = None) -> list[dict[str, Any]]:
    """The gate table for a point-cloud design space: what ran, what did not, and why.

    ``accuracy`` and ``baseline`` always run. ``conservation``, ``boundary`` and ``residual`` need a structured bundle
    with quadrature weights (they are the :mod:`pinneapple_veriphysics.surrogate_gates` checks, which is where they
    live) — here each is reported as not applicable with that reason, unless the family declares its own check in
    ``identity_checks``, in which case the family's check is what runs and its own tolerance decides.
    """
    tol = float(getattr(family, "tolerance", 0.05))
    gates: list[dict[str, Any]] = [
        {"check": "accuracy", "status": "passed" if report["within_tolerance"] else "failed",
         "value": report["relative_l2"], "tolerance": tol,
         "detail": f"mean relative L2 over {report['n_holdout_designs']} held-out designs"},
        {"check": "baseline", "status": "passed" if report["beats_baseline"] else "failed",
         "value": report["constant_field_baseline_relative_l2"], "tolerance": report["relative_l2"],
         "detail": "the constant field this model has to beat, scored on the same designs"},
    ]
    hook = getattr(family, "identity_checks", None)
    for name in ("conservation", "boundary", "residual"):
        if callable(hook) and designs is not None and predictions is not None:
            outcome = hook(name, designs, predictions)
            if outcome is not None:
                gates.append(outcome)
                continue
        gates.append({"check": name, "status": "not applicable", "value": None, "tolerance": None,
                      "detail": "a point cloud on a body's surface: no quadrature weights and no declared field "
                                "equation, so this gate belongs to the structured-bundle path "
                                "(pinneapple_veriphysics.surrogate_gates)"})
    return gates
