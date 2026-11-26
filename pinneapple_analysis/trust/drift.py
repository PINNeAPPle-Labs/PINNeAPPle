"""pinneapple_analysis.trust.drift — is the model still fitted to the data it is being served (#114).

The trust gate asks *can I trust this prediction?* before the prediction is used. This module asks the same
question over a window of live traffic, after deployment, and answers it in the two situations a served model
actually meets:

- **with labels** (the ground truth arrived, late but it arrived): the *error* drifted -- the residuals of this
  window are not the residuals of the validation split;
- **without labels** (the usual case in production): the *predictions* and their *predictive uncertainty* drifted
  away from the validation baseline. No labels are needed for that, only the outputs and the spread the model
  reports for them.

It also rings the envelope alarm: the fraction of live inputs that fall outside the distribution the model was
fitted on, which is what ``TrustGate`` already scores.

Three choices make the alarms mean something, and they are the reason this is not a set of ad-hoc thresholds:

1. **The statistic is a two-sample distance, not a difference of means.** Each quantity (input features, outputs,
   predictive uncertainty, residuals) is compared with the baseline using the Kolmogorov--Smirnov statistic, the
   largest distance between the two empirical CDFs, per output column, and the maximum over columns is reported. A
   shift that moves the mean shows up; a variance squeeze that keeps the mean does too.
2. **The threshold is calibrated on held-out data.** ``calibrate`` draws windows of the deployment size from a
   *second* held-out split -- not the baseline, not the live window -- recomputes the statistic of each against the
   baseline, and takes a high quantile of that null distribution (0.99 by default). So "drift" means "further from
   the baseline than 99 % of genuinely in-distribution windows were", and the false-alarm rate is a number the
   caller chose rather than an accident of a hand-picked cutoff.
3. **The decision is written down.** ``DriftReport`` carries the checks, the statistics, the thresholds they were
   compared against and a ``RetrainDecision`` with its reason, and ``write_record`` appends it as one JSON line: the
   retrain decision is part of the evidence, not a boolean somebody remembers.

The decision rule, documented rather than implied:

- a **critical** check that contradicts (error drift, envelope) means the model is answering outside what was
  validated: ``RETRAIN``;
- two or more label-free drifts at the same time (input, prediction, uncertainty) are enough to ``RETRAIN`` without
  any label: one of them alone is a signal to look, so it is ``REVIEW``;
- otherwise ``STABLE``.

Usage
-----
    monitor = DriftMonitor.from_validation(model, x_val, y_val)          # the baseline, from held-out data
    monitor.calibrate(model, x_holdout, y_holdout)                       # the null distribution, from a second split
    report = monitor.check(model, x_live, labels=y_live)                 # with labels: error drift
    report = monitor.check(model, x_live)                                # without: prediction and uncertainty drift
    report.decision.verdict, report.decision.reason                      # RETRAIN / REVIEW / STABLE, and why
    write_record(report, "monitoring/drift.jsonl")                       # one line per window, for the audit trail
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

from .trust_gate import TrustGate
from .trust_report import CONTRADICTS, NOT_RUN, SUPPORTS, Check

STABLE = "STABLE"
REVIEW = "REVIEW"
RETRAIN = "RETRAIN"

# the quantity names, used as the Check names and as the keys of the record
INPUT = "input_drift"
PREDICTION = "prediction_drift"
UNCERTAINTY = "uncertainty_drift"
ERROR = "error_drift"
ENVELOPE = "envelope"


def _as_2d(values: Any, what: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{what} must be 1-D or 2-D, got shape {a.shape}")
    return a


def ks_distance(reference: Any, window: Any) -> float:
    """The two-sample Kolmogorov--Smirnov statistic, maximised over the columns of a 2-D sample.

    ``0`` means the two empirical distributions are identical where they were sampled; ``1`` means they do not
    overlap at all. Column-wise then maximum, because an output drifting in one component is drift.
    """
    ref, cur = _as_2d(reference, "the reference sample"), _as_2d(window, "the window")
    if ref.shape[1] != cur.shape[1]:
        raise ValueError(f"the samples have {ref.shape[1]} and {cur.shape[1]} columns; they must match")
    if len(ref) < 2 or len(cur) < 2:
        raise ValueError("a two-sample statistic needs at least two points on each side")
    return float(max(ks_2samp(ref[:, j], cur[:, j]).statistic for j in range(ref.shape[1])))


def _call_model(model: Callable[..., Any], x: np.ndarray) -> Any:
    """Call the model on plain points: a torch module gets a tensor of its own dtype, anything else gets NumPy."""
    if callable(model) and hasattr(model, "parameters"):
        dtype = next(model.parameters()).dtype if list(model.parameters()) else torch.float32
        return model(torch.as_tensor(x, dtype=dtype))
    return model(x)


def _predict(model: Callable[..., Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(predictions, per-sample predictive uncertainty or ``None``), from a callable or a torch module.

    A model returning ``(mean, std)`` -- what ``DeepEnsemble`` and the surrogate models do -- gives both. Anything
    else gives predictions only and the uncertainty check is reported as not run, never as passed.
    """
    out = _call_model(model, x)
    if isinstance(out, tuple) and len(out) == 2:
        mean, std = out
        mean = np.asarray(mean.detach().cpu() if hasattr(mean, "detach") else mean, dtype=np.float64)
        std = np.asarray(std.detach().cpu() if hasattr(std, "detach") else std, dtype=np.float64)
        mean = mean.reshape(len(mean), -1) if mean.ndim > 1 else mean.reshape(-1, 1)
        std = std.reshape(len(mean), -1) if std.ndim > 1 else std.reshape(-1, 1)
        return mean, np.abs(std).mean(axis=1)              # one spread per sample, over its outputs
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    arr = np.asarray(out, dtype=np.float64)
    return (arr.reshape(-1, 1) if arr.ndim == 1 else arr), None


@dataclass(frozen=True)
class DriftThresholds:
    """The calibrated limits: a statistic above one of these is drift at the chosen false-alarm rate."""

    input: float = 1.0
    prediction: float = 1.0
    uncertainty: float | None = None
    error: float | None = None
    envelope: float | None = None
    calibrated_window: int = 0
    quantile: float = 0.99
    windows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RetrainDecision:
    """What the alarms add up to, and why. ``reason`` is the sentence a report or an on-call reads."""

    verdict: str
    retrain: bool
    reason: str
    triggered: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "retrain": self.retrain, "reason": self.reason,
                "triggered": list(self.triggered)}


@dataclass
class DriftReport:
    """One window, measured: the checks, the statistics, the thresholds and the decision."""

    checks: list[Check]
    decision: RetrainDecision
    statistics: dict[str, float]
    thresholds: DriftThresholds
    window: int
    baseline_window: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            self.at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    @property
    def alarms(self) -> list[str]:
        """The checks that contradicted, in the order they were evaluated."""
        return [c.name for c in self.checks if c.status == CONTRADICTS]

    @property
    def has_alarm(self) -> bool:
        return bool(self.alarms)

    def to_dict(self) -> dict[str, Any]:
        return {"at": self.at, "window": self.window, "baseline_window": self.baseline_window,
                "statistics": self.statistics, "thresholds": self.thresholds.to_dict(),
                "checks": [{"name": c.name, "status": c.status, "critical": c.critical,
                            "value": None if c.value is None or np.isnan(c.value) else float(c.value),
                            "limit": c.limit, "reason": c.reason} for c in self.checks],
                "decision": self.decision.to_dict(), "details": self.details}

    def __str__(self) -> str:                                    # a line a log can carry
        return (f"drift {self.decision.verdict}: {len(self.alarms)} alarm(s) on a window of {self.window}"
                + (f" ({', '.join(self.alarms)})" if self.alarms else "") + f" -- {self.decision.reason}")


@dataclass
class DriftBaseline:
    """The validation split, kept: what the live window is compared against."""

    inputs: np.ndarray
    predictions: np.ndarray
    uncertainties: np.ndarray | None = None
    errors: np.ndarray | None = None
    labels: np.ndarray | None = None
    gate: TrustGate | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.inputs)


class DriftMonitor:
    """Compares a window of live traffic with a validation baseline, with and without labels."""

    def __init__(self, baseline: DriftBaseline, thresholds: DriftThresholds | None = None, *,
                 envelope_level: float = 0.01, quantile: float = 0.99) -> None:
        if baseline.size < 10:
            raise ValueError(f"a baseline of {baseline.size} points is too small to compare a window against; "
                             "keep the validation split (at least a few hundred points)")
        if not 0.5 <= quantile < 1.0:
            raise ValueError(f"quantile must be in [0.5, 1), got {quantile}")
        if not 0.0 < envelope_level < 1.0:
            raise ValueError(f"envelope_level must be in (0, 1), got {envelope_level}")
        self.baseline = baseline
        self.thresholds = thresholds
        self.envelope_level = float(envelope_level)
        self.quantile = float(quantile)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_validation(cls, model: Callable[..., Any], inputs: Any, labels: Any = None, *,
                        envelope_level: float = 0.01, quantile: float = 0.99,
                        source: dict[str, Any] | None = None) -> DriftMonitor:
        """The baseline from a held-out split the model was *not* trained on.

        The envelope reference (``TrustGate``) is fitted on the same points: where the model was fitted is the
        domain it may be asked about, and the validation split is the honest stand-in for it when the training set
        is not kept.
        """
        x = _as_2d(inputs, "the validation inputs")
        predictions, uncertainty = _predict(model, x)
        y = None if labels is None else _as_2d(labels, "the validation labels")
        errors = None if y is None else np.abs(y - predictions)
        gate = TrustGate(ood_level=envelope_level).fit(torch.as_tensor(x, dtype=torch.float64))
        baseline = DriftBaseline(inputs=x, predictions=predictions, uncertainties=uncertainty, errors=errors,
                                 labels=y, gate=gate,
                                 source=source or {"split": "validation", "n": int(len(x))})
        return cls(baseline, envelope_level=envelope_level, quantile=quantile)

    def calibrate(self, model: Callable[..., Any], inputs: Any, labels: Any = None, *,
                  window: int | None = None, windows: int = 200, seed: int = 0) -> DriftThresholds:
        """The null distribution of every statistic, from windows of *held-out* data.

        These points must not be the baseline's: comparing a split with itself would produce a null distribution of
        near-zero statistics and a threshold that alarms on any difference at all. ``windows`` random windows of
        ``window`` points are drawn, each measured against the baseline, and the ``quantile`` of that sample is the
        limit. Raises when the split is too small to hold ten windows, and when it holds fewer than that per
        quantity the limit is ``None`` and the check reports itself as not run.
        """
        x = _as_2d(inputs, "the calibration inputs")
        n = int(window or min(500, max(50, len(x) // 4)))
        if len(x) < 10 * n:
            raise ValueError(f"calibrating on {len(x)} points with a window of {n} gives fewer than 10 windows; "
                             "pass more held-out data or window= a smaller size (the deployment window)")
        y = None if labels is None else _as_2d(labels, "the calibration labels")
        predictions, uncertainty = _predict(model, x)
        errors = None if y is None else np.abs(y - predictions)
        rng = np.random.default_rng(seed)
        stats: dict[str, list[float]] = {k: [] for k in (INPUT, PREDICTION, UNCERTAINTY, ERROR, ENVELOPE)}
        for _ in range(windows):
            idx = rng.choice(len(x), size=n, replace=False)
            stats[INPUT].append(ks_distance(self.baseline.inputs, x[idx]))
            stats[PREDICTION].append(ks_distance(self.baseline.predictions, predictions[idx]))
            if self.baseline.uncertainties is not None and uncertainty is not None:
                stats[UNCERTAINTY].append(ks_distance(self.baseline.uncertainties, uncertainty[idx]))
            if self.baseline.errors is not None and errors is not None:
                stats[ERROR].append(ks_distance(self.baseline.errors, errors[idx]))
            if self.baseline.gate is not None:
                stats[ENVELOPE].append(self._envelope_fraction(x[idx]))

        def limit(key: str) -> float | None:
            return float(np.quantile(stats[key], self.quantile)) if stats[key] else None

        input_limit, prediction_limit = limit(INPUT), limit(PREDICTION)
        self.thresholds = DriftThresholds(
            input=1.0 if input_limit is None else input_limit,
            prediction=1.0 if prediction_limit is None else prediction_limit,
            uncertainty=limit(UNCERTAINTY), error=limit(ERROR), envelope=limit(ENVELOPE),
            calibrated_window=n, quantile=self.quantile, windows=windows)
        return self.thresholds

    # -- one window --------------------------------------------------------
    def check(self, model: Callable[..., Any], inputs: Any, labels: Any = None) -> DriftReport:
        """Measure one window: the model is called here, so this is the served model, not a copy of it."""
        x = _as_2d(inputs, "the live inputs")
        predictions, uncertainty = _predict(model, x)
        y = None if labels is None else _as_2d(labels, "the live labels")
        return self.evaluate(x, predictions, uncertainty=uncertainty,
                             errors=None if y is None else np.abs(y - predictions))

    def evaluate(self, inputs: Any, predictions: Any, *, uncertainty: Any = None,
                 errors: Any = None) -> DriftReport:
        """The measurement itself, for a pipeline that already has the outputs (a batch job, a replay)."""
        if self.thresholds is None:
            raise RuntimeError("this monitor has no calibrated thresholds: call calibrate() with a second held-out "
                               "split first, or decide the limits yourself and pass DriftThresholds")
        x = _as_2d(inputs, "the window inputs")
        pred = _as_2d(predictions, "the window predictions")
        unc = None if uncertainty is None else np.asarray(uncertainty, dtype=np.float64).reshape(-1, 1)
        err = None if errors is None else _as_2d(errors, "the window errors")
        if len(x) != len(pred):
            raise ValueError(f"{len(x)} inputs but {len(pred)} predictions: one window, one row per point")
        if len(x) < 10:
            raise ValueError(f"a window of {len(x)} points cannot be compared: the statistic would be noise")

        stats: dict[str, float] = {}
        checks: list[Check] = []
        details: dict[str, Any] = {}

        stats[INPUT] = ks_distance(self.baseline.inputs, x)
        checks.append(self._check(INPUT, stats[INPUT], self.thresholds.input, critical=False, weight=0.5,
                                  description="the live inputs against the validation inputs"))

        stats[PREDICTION] = ks_distance(self.baseline.predictions, pred)
        checks.append(self._check(PREDICTION, stats[PREDICTION], self.thresholds.prediction, critical=False,
                                  weight=1.0, description="the live predictions against the validation ones"))

        if unc is not None and self.baseline.uncertainties is not None and self.thresholds.uncertainty is not None:
            stats[UNCERTAINTY] = ks_distance(self.baseline.uncertainties, unc)
            checks.append(self._check(UNCERTAINTY, stats[UNCERTAINTY], self.thresholds.uncertainty, critical=False,
                                      weight=1.0, description="the predictive uncertainty the model reports, live "
                                                              "against validation"))
        else:
            checks.append(Check(name=UNCERTAINTY, status=NOT_RUN, weight=1.0, source="drift",
                                reason="the model reports no predictive uncertainty (it returns predictions only), "
                                       "so this window has nothing to compare: pass an ensemble or a model that "
                                       "returns (mean, std)"))

        if err is not None and self.baseline.errors is not None and self.thresholds.error is not None:
            stats[ERROR] = ks_distance(self.baseline.errors, err)
            baseline_rmse = float(np.sqrt(np.mean(np.square(self.baseline.errors))))
            live_rmse = float(np.sqrt(np.mean(np.square(err))))
            ratio = None if baseline_rmse == 0 else live_rmse / baseline_rmse
            details["rmse_baseline"], details["rmse_window"], details["rmse_ratio"] = \
                baseline_rmse, live_rmse, ratio
            checks.append(self._check(ERROR, stats[ERROR], self.thresholds.error, critical=True, weight=2.0,
                                      description=f"the residuals against the validation residuals (RMSE "
                                                  f"{live_rmse:.4g} vs {baseline_rmse:.4g}, ratio "
                                                  f"{ratio if ratio is None else round(ratio, 4)})"))
        else:
            checks.append(Check(name=ERROR, status=NOT_RUN, weight=2.0, source="drift",
                                reason="no labels for this window, and the error drift needs them: the label-free "
                                       "checks are what can be said without ground truth"))

        if self.baseline.gate is not None and self.thresholds.envelope is not None:
            stats[ENVELOPE] = self._envelope_fraction(x)
            checks.append(self._check(ENVELOPE, stats[ENVELOPE], self.thresholds.envelope, critical=True,
                                      weight=2.0,
                                      description="the fraction of inputs outside the fitted distribution "
                                                  f"(per-point p-value below {self.envelope_level}): the model "
                                                  "has left its envelope"))

        return DriftReport(checks=checks, decision=self._decide(checks), statistics=stats,
                           thresholds=self.thresholds, window=len(x), baseline_window=self.baseline.size,
                           details=details)

    # -- internals ---------------------------------------------------------
    def _envelope_fraction(self, x: np.ndarray) -> float:
        p = self.baseline.gate.ood_p_values(x)
        return float(np.mean(p < self.envelope_level))

    def _check(self, name: str, value: float, limit: float | None, *, critical: bool, weight: float,
               description: str) -> Check:
        if limit is None:
            return Check(name=name, status=NOT_RUN, critical=critical, weight=weight, source="drift",
                         reason=f"no calibrated limit for {name}: the calibration split did not produce this "
                                "quantity")
        return Check(name=name, status=SUPPORTS if value <= limit else CONTRADICTS, critical=critical,
                     weight=weight, value=float(value), limit=float(limit), source="drift",
                     reason=f"{description}: {value:.4g} against a limit of {limit:.4g} (calibrated at the "
                            f"{self.thresholds.quantile:.0%} quantile of {self.thresholds.windows} held-out "
                            f"windows of {self.thresholds.calibrated_window})")

    @staticmethod
    def _decide(checks: Sequence[Check]) -> RetrainDecision:
        """The documented rule: a critical alarm retrains, two label-free alarms retrain, one asks for a look."""
        critical = [c.name for c in checks if c.status == CONTRADICTS and c.critical]
        label_free = [c.name for c in checks if c.status == CONTRADICTS and not c.critical]
        triggered = tuple(critical + label_free)
        if critical:
            return RetrainDecision(verdict=RETRAIN, retrain=True, triggered=triggered,
                                   reason=f"{', '.join(critical)} contradicted on validated ground (error or "
                                          "envelope): the model is being asked outside what it was validated on, "
                                          "so retrain before serving it further")
        if len(label_free) >= 2:
            return RetrainDecision(verdict=RETRAIN, retrain=True, triggered=triggered,
                                   reason=f"{', '.join(label_free)} drifted together without any label: the "
                                          "outputs and the uncertainty left the validation baseline, which is "
                                          "enough to retrain without waiting for ground truth")
        if label_free:
            return RetrainDecision(verdict=REVIEW, retrain=False, triggered=triggered,
                                   reason=f"{label_free[0]} alone is above its limit: one label-free signal is "
                                          "worth looking at, not yet a retrain -- watch the next window, or wait "
                                          "for labels to check the error")
        return RetrainDecision(verdict=STABLE, retrain=False,
                               reason="every check is inside its calibrated limit: the model is still being served "
                                      "the data it was validated on")


def write_record(report: DriftReport, path: str | Path) -> Path:
    """Append one JSON line: the record of a window, its decision and the reason. The audit trail of retraining."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report.to_dict(), default=float) + "\n")
    return target
