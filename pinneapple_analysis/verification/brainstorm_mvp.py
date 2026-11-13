"""MVP scientific assurance coverage for the complete Physics AI brainstorm.

Explicitly bounded, deterministic implementations. Unavailable measurements
return NOT_RUN/INCONCLUSIVE rather than invented physics or confidence scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import math

from .pipeline_assurance import Check, Status, numeric_close


def falsify_scalar_property(name: str, stage: str, inputs: Sequence[float],
                            model: Callable[[float], float], predicate: Callable[[float, float], bool],
                            *, assumption: str) -> Check:
    """Search supplied admissible points for a property counterexample."""
    if not assumption or not inputs:
        return Check(name, stage, Status.NOT_RUN, reason="Applicability and test domain required")
    violations = []
    for x in inputs:
        try:
            y = float(model(float(x)))
            if not math.isfinite(y) or not predicate(float(x), y):
                violations.append({"x": float(x), "y": y if math.isfinite(y) else None})
        except (ValueError, ArithmeticError, OverflowError) as exc:
            violations.append({"x": float(x), "error": type(exc).__name__})
    return Check(name, stage, Status.FAIL if violations else Status.PASS,
                 evidence={"tested": len(inputs), "counterexamples": violations[:20],
                           "counterexample_count": len(violations)},
                 assumptions=(assumption, "Supplied points satisfy the physical domain constraints"))


def calibration_coverage(name: str, predictions: Sequence[float], lower: Sequence[float],
                         upper: Sequence[float], targets: Sequence[float], *,
                         nominal: float, tolerance: float = 0.1) -> Check:
    """Empirical interval coverage; not a full calibration certificate."""
    n = len(targets)
    if not 0 < nominal < 1 or tolerance < 0:
        raise ValueError("Invalid coverage target/tolerance")
    if not n or len(predictions) != n or len(lower) != n or len(upper) != n:
        return Check(name, "training", Status.NOT_RUN, reason="Aligned holdout intervals required")
    if any(not all(math.isfinite(float(v)) for v in values)
           for values in zip(predictions, lower, upper, targets)):
        return Check(name, "training", Status.INCONCLUSIVE, reason="Nonfinite interval data")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        return Check(name, "training", Status.FAIL, reason="Invalid prediction interval")
    covered = sum(lo <= y <= hi for lo, hi, y in zip(lower, upper, targets))
    empirical = covered / n
    return Check(name, "training",
                 Status.PASS if abs(empirical - nominal) <= tolerance else Status.FAIL,
                 evidence={"samples": n, "nominal": nominal, "empirical": empirical,
                           "absolute_coverage_error": abs(empirical - nominal)},
                 assumptions=("Independent representative evaluation holdout",))


def ood_envelope(name: str, samples: Sequence[Sequence[float]],
                 lower: Sequence[float], upper: Sequence[float]) -> Check:
    """Axis-aligned envelope warning, not a learned OOD probability."""
    if not lower or len(lower) != len(upper) or not samples:
        return Check(name, "serving", Status.NOT_RUN, reason="Training envelope and samples required")
    if any(lo > hi or not math.isfinite(lo) or not math.isfinite(hi) for lo, hi in zip(lower, upper)):
        return Check(name, "serving", Status.INCONCLUSIVE, reason="Invalid envelope")
    if any(len(row) != len(lower) or any(not math.isfinite(v) for v in row) for row in samples):
        return Check(name, "serving", Status.INCONCLUSIVE, reason="Invalid sample dimensions/values")
    outside = [i for i, row in enumerate(samples)
               if any(v < lo or v > hi for v, lo, hi in zip(row, lower, upper))]
    return Check(name, "serving", Status.FAIL if outside else Status.PASS,
                 evidence={"outside_indices": outside[:100], "outside_count": len(outside),
                           "sample_count": len(samples)},
                 assumptions=("Envelope reflects declared training support",))


def independent_design_check(name: str, predicted: float, independent_solver: float,
                             *, atol: float, rtol: float) -> Check:
    return numeric_close(name, "engineering_decision", independent_solver, predicted,
                         atol=atol, rtol=rtol,
                         assumptions=("Independent solver is adequately converged",
                                      "Same design, loading, units and quantity of interest"))


def sensor_drift(name: str, residuals: Sequence[float], *,
                 baseline_mean: float, baseline_std: float, sigma_limit: float = 3.0) -> Check:
    if not residuals or baseline_std <= 0 or sigma_limit <= 0:
        return Check(name, "digital_twin", Status.NOT_RUN, reason="Valid baseline and residuals required")
    if any(not math.isfinite(float(v)) for v in (*residuals, baseline_mean, baseline_std)):
        return Check(name, "digital_twin", Status.INCONCLUSIVE, reason="Nonfinite residual")
    anomalies = [i for i, value in enumerate(residuals)
                 if abs(value - baseline_mean) > sigma_limit * baseline_std]
    return Check(name, "digital_twin", Status.FAIL if anomalies else Status.PASS,
                 evidence={"anomaly_indices": anomalies[:100], "anomaly_count": len(anomalies)},
                 assumptions=("Stationary baseline noise model and synchronized sensors",))


def evidence_dossier(checks: Sequence[Check], *, context_of_use: str,
                     limitations: Sequence[str] = ()) -> dict:
    """Decision-facing report without synthetic confidence percentages."""
    counts = {status.value: sum(c.status == status for c in checks) for status in Status}
    return {"context_of_use": context_of_use, "checks": [
        {"name": c.name, "stage": c.stage, "status": c.status.value,
         "reason": c.reason, "assumptions": list(c.assumptions), "evidence": c.evidence}
        for c in checks], "counts": counts,
        "limitations": list(limitations) + [
            "Internal consistency does not establish empirical validity",
            "Human engineering review and context-specific acceptance criteria required"],
        "decision": "blocked" if not checks or counts["fail"] or counts["not_run"] or counts["inconclusive"]
                    else "eligible_for_review"}
