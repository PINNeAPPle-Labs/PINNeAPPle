"""Stage F: comparison against independent reference data.

Without reference data a pipeline can only show *internal consistency*:
the artifact respects its own constraints. That is genuinely useful and it
is not evidence of fidelity to the physical system. This module makes the
difference explicit rather than rhetorical:

* it refuses to invent or default a reference (:class:`ReferenceData`
  requires an origin, a version and its known limitations);
* it refuses to compute an "acceptable" threshold -- the criteria come
  from the caller/specification;
* when predictions are compared on data that was also used for
  calibration, that is reported, because the resulting metric is an
  in-sample fit, not a validation;
* every metric is recorded with the subset it was computed on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .report import Finding, Status
from .validators.base import ValidatorResult

_ORIGINS = ("experiment", "analytical", "solver", "benchmark", "field_measurement")


@dataclass(frozen=True)
class ReferenceData:
    """A reference the artifact is compared against -- never invented here."""

    name: str
    origin: str
    version: str
    values: np.ndarray
    unit: str = ""
    limitations: Sequence[str] = ()
    #: True when this data was also used to calibrate the artifact. A metric
    #: computed on it is in-sample and is reported as such.
    used_for_calibration: bool = False

    def __post_init__(self):
        if self.origin not in _ORIGINS:
            raise ValueError(f"reference origin must be one of {_ORIGINS}, got '{self.origin}'")
        if not self.version:
            raise ValueError(
                "ReferenceData requires a version -- an unversioned reference cannot be reproduced "
                "or audited later")
        if not np.asarray(self.values).size:
            raise ValueError("reference has no values")


METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": lambda p, r: float(np.sqrt(np.mean((p - r) ** 2))),
    "mae": lambda p, r: float(np.mean(np.abs(p - r))),
    "max_abs": lambda p, r: float(np.max(np.abs(p - r))),
    "mape": lambda p, r: float(np.mean(np.abs((p - r) / np.where(r == 0, np.nan, r))) * 100.0),
    "bias": lambda p, r: float(np.mean(p - r)),
}


def compare(predictions: np.ndarray, reference: ReferenceData,
            criteria: Dict[str, float], *,
            subset: str = "validation",
            groups: Optional[Dict[str, np.ndarray]] = None,
            baseline: Optional[np.ndarray] = None) -> ValidatorResult:
    """Compare ``predictions`` with ``reference.values`` under ``criteria``.

    ``criteria`` maps a metric name to its maximum acceptable value. Only
    the metrics named there are evaluated; a metric that is not required is
    still *reported* but does not gate. An empty ``criteria`` means the
    caller declared nothing acceptable, and is a failure of the check, not
    a free pass.
    """
    findings: List[Finding] = []
    pred = np.asarray(predictions, dtype=float).ravel()
    ref = np.asarray(reference.values, dtype=float).ravel()

    if pred.size != ref.size:
        return ValidatorResult(
            name="reference_comparison", status=Status.BLOCKED,
            findings=[Finding(
                id="reference.shape", status=Status.BLOCKED, severity="error",
                message=f"predictions have {pred.size} value(s) but the reference has {ref.size}",
                evidence={"predictions": int(pred.size), "reference": int(ref.size)})],
            evidence={"reference_origin": reference.origin, "reference_name": reference.name},
        )

    finite = np.isfinite(pred) & np.isfinite(ref)
    if not finite.any():
        return ValidatorResult(
            name="reference_comparison", status=Status.FAIL,
            findings=[Finding(id="reference.non_finite", status=Status.FAIL, severity="error",
                              message="predictions and reference share no finite element")],
            evidence={"reference_origin": reference.origin, "reference_name": reference.name},
        )
    pred_f, ref_f = pred[finite], ref[finite]

    values = {name: fn(pred_f, ref_f) for name, fn in METRICS.items()}
    unknown = sorted(set(criteria) - set(METRICS))
    if unknown:
        findings.append(Finding(
            id="reference.unknown_metric", status=Status.BLOCKED, severity="error",
            message=f"criteria name unknown metric(s) {unknown}; available: {sorted(METRICS)}",
            evidence={"unknown": unknown}))

    if not criteria:
        findings.append(Finding(
            id="reference.no_criteria", status=Status.FAIL, severity="error",
            message="no acceptance criteria were declared, so agreement cannot be judged",
            justification="a comparison without a threshold is a number, not a validation"))

    evaluated: Dict[str, dict] = {}
    for name, limit in criteria.items():
        if name not in METRICS:
            continue
        value = values[name]
        ok = bool(np.isfinite(value) and value <= float(limit))
        evaluated[name] = {"value": value, "limit": float(limit), "passed": ok}
        findings.append(Finding(
            id=f"reference.metric.{name}",
            status=Status.PASS if ok else Status.FAIL,
            severity="info" if ok else "error",
            message=f"{name} = {value:.6g} ({'<=' if ok else '>'} {float(limit):g})",
            evidence={"metric": name, "value": value, "limit": float(limit)}))

    evidence: Dict[str, object] = {
        "reference_origin": reference.origin,
        "reference_name": reference.name,
        "reference_version": reference.version,
        "reference_limitations": list(reference.limitations),
        "subset": subset,
        "in_sample": bool(reference.used_for_calibration),
        "n_compared": int(finite.sum()),
        "metrics": values,
        "criteria": evaluated,
        "subset_definition": "all elements unless a mask was applied by the caller",
    }
    if reference.used_for_calibration:
        findings.append(Finding(
            id="reference.in_sample", status=Status.FAIL, severity="warning",
            message="reference data was also used for calibration: this metric is an in-sample fit, "
                    "not an independent validation",
            evidence={"reference": reference.name}))

    if baseline is not None:
        base = np.asarray(baseline, dtype=float).ravel()[finite]
        if base.size == pred_f.size:
            base_rmse = METRICS["rmse"](base, ref_f)
            evidence["baseline_rmse"] = base_rmse
            if base_rmse < values["rmse"]:
                findings.append(Finding(
                    id="reference.worse_than_baseline", status=Status.FAIL, severity="error",
                    message=f"the artifact's RMSE ({values['rmse']:.6g}) is worse than the baseline "
                            f"({base_rmse:.6g})",
                    evidence={"rmse": values["rmse"], "baseline_rmse": base_rmse}))
            else:
                findings.append(Finding(
                    id="reference.vs_baseline", status=Status.PASS, severity="info",
                    message=f"RMSE {values['rmse']:.6g} better than baseline {base_rmse:.6g}"))

    if groups:
        per_group = {}
        for group_name, mask in groups.items():
            m = np.asarray(mask, dtype=bool).ravel()
            if m.size != pred.size:
                continue
            mm = m[finite]
            if not mm.any():
                continue
            per_group[group_name] = {k: fn(pred_f[mm], ref_f[mm]) for k, fn in METRICS.items()}
        evidence["per_group"] = per_group

    failed = [f for f in findings if f.blocks]
    status = Status.FAIL if failed else Status.PASS
    if not failed:
        findings.append(Finding(
            id="validator.reference_comparison", status=Status.PASS, severity="info",
            message=f"all {len(evaluated)} declared metric(s) within their limits on '{subset}'"))
    return ValidatorResult(
        name="reference_comparison", status=status, findings=findings, evidence=evidence,
        limitations=[
            "agreement with one reference over one operating range does not generalise beyond it",
            "reference data carries its own uncertainty, which this comparison does not propagate",
            "no reference was generated, altered or selected by this module",
        ] + list(reference.limitations),
    )


def split_mask(n: int, holdout_fraction: float = 0.3, seed: int = 0) -> np.ndarray:
    """A deterministic boolean mask marking the validation (holdout) subset."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_hold = max(1, int(round(n * holdout_fraction)))
    mask = np.zeros(n, dtype=bool)
    mask[idx[:n_hold]] = True
    return mask
