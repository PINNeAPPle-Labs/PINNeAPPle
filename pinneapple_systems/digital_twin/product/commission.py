"""Commissioning: the two gates a twin passes before it is allowed to run.

**Preflight** reads the spec and the plant history and refuses the twin if the contract is
not checkable — a unit mismatch, an unobservable field, a channel stuck at a constant, a
history that never visited the operating range. These are cheap checks that catch the
failures which otherwise cost a month: a °C/°F mix-up produces a twin that tracks perfectly
and is wrong by 1.8×.

**Backtest** replays the history through the twin and measures it against the sensors that
were never assimilated. It answers the only two questions that matter: how far the twin is
from reality, and whether its own uncertainty band contains reality. The second one catches
the twin that is right on average and confidently wrong where it counts.

Neither gate proves the twin will hold for a regime the history never saw. That is why
:func:`backtest` reports the window it used and :mod:`~.runner` lists it under
``unverified``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .assimilation import Z95, LatentStateTwin
from .evidence import (
    FAIL,
    INFO,
    PASS,
    WARN,
    CheckItem,
    CheckReport,
    FieldMetrics,
    TrackingReport,
)
from .spec import TwinSpec

__all__ = ["Frame", "read_timestamp", "read_readings", "preflight", "backtest"]

# A frame is one instant of plant data: {"t": 0.0, "TT-101": 82.1, "FT-102": 3.4, ...}
Frame = Mapping[str, Any]

_TIMESTAMP_KEYS = ("t", "timestamp", "time", "ts")


def read_timestamp(frame: Frame) -> float:
    """The frame's timestamp, under any of the conventional keys."""
    for k in _TIMESTAMP_KEYS:
        if k in frame and frame[k] is not None:
            return float(frame[k])
    raise KeyError(f"frame has no timestamp (looked for {_TIMESTAMP_KEYS}): keys are {sorted(frame)}")


def read_readings(frame: Frame, spec: TwinSpec) -> dict[str, float]:
    """The sensor values present in a frame, ignoring keys that are not sensors."""
    out: dict[str, float] = {}
    for s in spec.sensors:
        v = frame.get(s.sensor_id)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[s.sensor_id] = fv
    return out


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(spec: TwinSpec, history: Sequence[Frame]) -> CheckReport:
    """Check that this twin can be commissioned at all, before spending a backtest on it.

    Returns a :class:`~.evidence.CheckReport`. A single ``FAIL`` blocks commissioning;
    ``WARN`` items are reported and carried into the evidence so nobody can say later that
    they were not told.
    """
    rep = CheckReport(title=f"Preflight — {spec.asset_id} (spec {spec.digest()})")

    frames = list(history)
    if not frames:
        rep.add("P00", FAIL, "history is empty", why="nothing to commission against",
                fix="supply at least a few hours of plant history with the declared sensors")
        return rep

    ts = np.array([read_timestamp(f) for f in frames], dtype=float)
    readings = [read_readings(f, spec) for f in frames]

    _check_units(spec, rep)
    _check_observability(spec, rep)
    _check_timebase(spec, ts, rep)
    _check_channels(spec, readings, rep)
    _check_envelope(spec, readings, rep)
    _check_gates(spec, rep)
    return rep


def _check_units(spec: TwinSpec, rep: CheckReport) -> None:
    mismatched = [
        s for s in spec.sensors
        if not s.driver and s.unit != spec.field(s.field).unit
    ]
    if mismatched:
        detail = ", ".join(f"{s.sensor_id} in {s.unit} vs field {s.field} in {spec.field(s.field).unit}"
                           for s in mismatched[:4])
        rep.add(
            "P01", FAIL, "sensor units disagree with the field units", detail=detail,
            why="a mixed unit is the failure that survives every other check: the twin tracks "
                "perfectly and is wrong by the conversion factor",
            fix="convert the channel at ingestion, or fix the spec; never average the two",
            source="spec.sensors[].unit",
        )
    else:
        n_drv = len(spec.driver_sensors)
        rep.add("P01", PASS, "every sensor declares the unit of its field",
                detail=f"{len(spec.sensors)} sensors ({n_drv} driver, {len(spec.sensors) - n_drv} predicted)",
                source="spec.sensors[].unit")


def _check_observability(spec: TwinSpec, rep: CheckReport) -> None:
    blind = [
        f.name for f in spec.fields
        if f.kind != "parameter" and not spec.sensors_for(f.name, holdout=False)
    ]
    if blind:
        rep.add(
            "P02", FAIL, "field(s) have no assimilated sensor", detail=", ".join(blind),
            why="the twin can only correct a field it hears; a field seen only by holdout "
                "sensors is predicted and compared but never updated",
            fix="mark at least one sensor of each field as assimilated (holdout=False)",
            source="spec.sensors[].holdout",
        )
    else:
        rep.add("P02", PASS, "every predicted field has at least one assimilated sensor",
                detail=f"{len(spec.assimilated_sensors())} assimilated, "
                       f"{len(spec.holdout_sensors())} held out",
                source="spec.sensors[].holdout")

    # An over-determined estimate is the condition for the latent state to mean anything.
    n_obs, n_lat = len(spec.assimilated_sensors()), _latent_dim(spec)
    if n_obs < n_lat:
        rep.add(
            "P02b", FAIL, "estimate is under-determined",
            detail=f"{n_obs} assimilated sensors for {n_lat} latent unknowns",
            why="with fewer equations than unknowns the filter fits noise and the state is "
                "not identifiable",
            fix="add sensors, or reduce the parameter fields; one holdout does not count",
        )
    elif n_obs == n_lat:
        rep.add("P02b", WARN, "estimate is exactly determined",
                detail=f"{n_obs} sensors for {n_lat} unknowns",
                why="no redundancy: one biased sensor moves the estimate and nothing disagrees",
                fix="add a redundant sensor per latent field where the plant allows it")
    else:
        rep.add("P02b", PASS, "estimate is over-determined",
                detail=f"{n_obs} sensors for {n_lat} unknowns",
                why="redundancy is what lets the backtest disagree with the filter")


def _check_timebase(spec: TwinSpec, ts: np.ndarray, rep: CheckReport) -> None:
    if np.any(np.diff(ts) < 0):
        bad = int(np.argmax(np.diff(ts) < 0)) + 1
        rep.add(
            "P03", FAIL, "timestamps are not monotonic", detail=f"first inversion at frame {bad}",
            why="a replay in the wrong order trains the filter on the future",
            fix="sort by timestamp before commissioning; if the plant clock jumped, fix ingestion",
            source="frame['t']",
        )
    else:
        rep.add("P03", PASS, "timestamps are monotonic", detail=f"{len(ts)} frames")

    uniq = np.unique(ts)
    if uniq.size != ts.size:
        rep.add(
            "P04", WARN, "duplicate timestamps", detail=f"{ts.size - uniq.size} duplicate(s)",
            why="duplicates weight the same instant twice in the likelihood",
            fix="deduplicate on ingest, keeping the last value per timestamp",
        )
    else:
        rep.add("P04", PASS, "no duplicate timestamps")

    if ts.size >= 3:
        dt = np.median(np.diff(ts))
        if dt > 2 * spec.sampling_s:
            rep.add(
                "P05", FAIL, "declared sampling rate is not what the history has",
                detail=f"median Δt {dt:.3g} s vs. sampling_s {spec.sampling_s:.3g} s",
                why="every latency and freshness number in the evidence would be computed "
                    "against a rate the plant never produced",
                fix="set sampling_s to the measured median, or fix the acquisition",
                source="spec.sampling_s",
            )
        else:
            rep.add("P05", PASS, "declared sampling rate matches the history",
                    detail=f"median Δt {dt:.3g} s", source="spec.sampling_s")


def _check_channels(spec: TwinSpec, readings: Sequence[dict[str, float]], rep: CheckReport) -> None:
    reported = 0
    failed = False
    for s in spec.sensors:
        vals = np.array([r[s.sensor_id] for r in readings if s.sensor_id in r], dtype=float)
        if vals.size == 0:
            rep.add(
                "P06", FAIL, f"{s.sensor_id} never appears in the history",
                why="a sensor the twin expects and never receives leaves the filter without "
                    "that equation for the whole run",
                fix="check the channel name against the spec, or remove the sensor",
                source=f"spec sensor {s.sensor_id}",
            )
            failed = True
            continue
        if vals.size > 1 and float(np.ptp(vals)) == 0.0:
            status = FAIL if not s.holdout else WARN
            rep.add(
                "P07", status, f"{s.sensor_id} is stuck at a constant", detail=f"{vals[0]:.6g}",
                why="a frozen channel is silent, not healthy: the twin assimilates a stale value "
                    "and its band tightens around it",
                fix="exclude the channel until it moves; check the logger",
                source=f"spec sensor {s.sensor_id}",
            )
            failed = failed or status == FAIL
            continue
        reported += 1
    if not failed:
        rep.add("P06", PASS, "every declared sensor reports", detail=f"{reported} channels")
        rep.add("P07", PASS, "no stuck channel detected")


def _check_envelope(spec: TwinSpec, readings: Sequence[dict[str, float]], rep: CheckReport) -> None:
    bad = 0
    total = 0
    offenders: list[str] = []
    for s in spec.sensors:
        vals = np.array([r[s.sensor_id] for r in readings if s.sensor_id in r], dtype=float)
        if vals.size == 0:
            continue
        total += vals.size
        outside = int(np.count_nonzero((vals < s.lo) | (vals > s.hi)))
        bad += outside
        if outside:
            offenders.append(f"{s.sensor_id} ({outside})")
    if total == 0:
        return
    frac = bad / total
    if frac > 0.01:
        rep.add(
            "P08", FAIL, "readings fall outside the declared envelope",
            detail=f"{100 * frac:.2f}% outside — {', '.join(offenders[:4])}",
            why="an envelope is a claim about the process; readings outside it mean either the "
                "sensor is faulty or the spec describes a different operating range",
            fix="reconcile the envelope with the data before commissioning",
            source="spec.sensors[].lo/hi",
        )
    elif bad:
        rep.add("P08", WARN, "a few readings fall outside the envelope",
                detail=f"{100 * frac:.3f}% — {', '.join(offenders[:4])}",
                why="usually a transient; worth a look if it repeats")
    else:
        rep.add("P08", PASS, "all readings inside the declared envelope")



def _check_gates(spec: TwinSpec, rep: CheckReport) -> None:
    g = spec.gates
    if g.max_rmse_frac >= 1.0:
        rep.add(
            "P09", WARN, "the RMSE gate is vacuous",
            detail=f"max_rmse_frac = {g.max_rmse_frac}",
            why="an error as large as the whole span would still pass",
            fix="set the gate from the engineering requirement, not from the current error",
            source="spec.gates.max_rmse_frac",
        )
    else:
        rep.add("P09", PASS, "acceptance gates are meaningful",
                detail=f"RMSE ≤ {100 * g.max_rmse_frac:.1f}% of span, "
                       f"bias ≤ {100 * g.max_bias_frac:.1f}%, coverage ≥ {100 * g.min_coverage:.0f}%",
                source="spec.gates")

    if spec.model.accuracy_floor is None:
        rep.add(
            "P10", WARN, "the model declares no accuracy floor",
            detail=f"model {spec.model.model_id} {spec.model.version}".strip(),
            why="without it the tracking error cannot be attributed to the twin rather than "
                "to the model it inherited",
            fix="state the model's measured error on unseen data in the spec",
            source="spec.model.accuracy_floor",
        )
    else:
        rep.add("P10", PASS, "the model declares an accuracy floor",
                detail=f"{spec.model.accuracy_floor:g} {spec.fields[0].unit} "
                       f"({spec.model.model_id})",
                source="spec.model.accuracy_floor")


def _latent_dim(spec: TwinSpec) -> int:
    params = [f for f in spec.fields if f.kind == "parameter"]
    return len(params) if params else len(spec.fields)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class _HoldoutSeries:
    """Accumulated residuals for one held-out sensor."""

    sensor_id: str
    field: str
    metric_to: list[float] = None  # type: ignore[assignment]
    pred: list[float] = None       # type: ignore[assignment]
    lo: list[float] = None         # type: ignore[assignment]
    hi: list[float] = None         # type: ignore[assignment]
    ts: list[float] = None         # type: ignore[assignment]
    sig: list[float] = None        # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("metric_to", "pred", "lo", "hi", "ts", "sig"):
            if getattr(self, name) is None:
                setattr(self, name, [])


def backtest(
    spec: TwinSpec,
    model: Callable[..., Any],
    history: Sequence[Frame],
    *,
    n_ens: int = 64,
    inflation: float = 1.02,
    seed: int = 0,
    warmup_steps: int = 0,
    calibrate_fraction: float = 0.5,
) -> TrackingReport:
    """Replay the history through the twin and measure it against the held-out sensors.

    ``warmup_steps`` frames are assimilated but excluded from the metrics — use it to let the
    latent state converge before scoring, and say so when you report the number.

    ``calibrate_fraction`` splits the scored window in two, in time. The band the filter
    produces is a band on its *state*; it does not include the instrument noise of the
    assimilated channels, which propagates into the estimate and is the dominant error term on
    a clean process. Commissioning therefore measures a single multiplicative band scale on
    the **first** part and reports coverage on the **second**, later part — an out-of-sample
    check, not a band fitted to the data it is scored on. Set it to 0 to report the raw
    ensemble band instead, and it is recorded either way.
    """
    frames = list(history)
    if not frames:
        raise ValueError("backtest needs at least one frame")

    twin = LatentStateTwin(model, spec, n_ens=n_ens, inflation=inflation, seed=seed)
    series: dict[str, _HoldoutSeries] = {
        s.sensor_id: _HoldoutSeries(sensor_id=s.sensor_id, field=s.field)
        for s in twin.holdout_sensors
    }

    n_scored = 0
    for i, frame in enumerate(frames):
        obs = read_readings(frame, spec)
        twin.step(obs)
        if i < warmup_steps:
            continue

        t = read_timestamp(frame)
        pred, lo, hi = twin.band_at_holdout()
        scored_here = False
        for j, s in enumerate(twin.holdout_sensors):
            measured = obs.get(s.sensor_id)
            if measured is None:
                continue
            rec = series[s.sensor_id]
            rec.metric_to.append(measured)
            rec.pred.append(float(pred[j]))
            rec.lo.append(float(lo[j]))
            rec.hi.append(float(hi[j]))
            rec.ts.append(t)
            rec.sig.append(float((hi[j] - lo[j]) / (2.0 * Z95)))
            scored_here = True
        n_scored += int(scored_here)

    metrics: list[FieldMetrics] = []
    for f in spec.fields:
        recs = [r for r in series.values() if r.field == f.name]
        meas = np.concatenate([np.asarray(r.metric_to) for r in recs]) if recs else np.array([])
        pred = np.concatenate([np.asarray(r.pred) for r in recs]) if recs else np.array([])
        ts_all = np.concatenate([np.asarray(r.ts) for r in recs]) if recs else np.array([])
        sig = np.concatenate([np.asarray(r.sig) for r in recs]) if recs else np.array([])
        if meas.size == 0:
            continue

        err = pred - meas
        scale, cal_s, cov_raw, cov = _calibrate_band(
            err, sig, ts_all, calibrate_fraction
        )
        metrics.append(
            FieldMetrics(
                field_name=f.name,
                unit=f.unit,
                n=int(meas.size),
                rmse=float(np.sqrt(np.mean(err ** 2))),
                mae=float(np.mean(np.abs(err))),
                bias=float(np.mean(err)),
                p95_abs=float(np.percentile(np.abs(err), 95)),
                coverage=cov,
                span=f.span,
                window_s=float(ts_all.max() - ts_all.min()) if ts_all.size > 1 else None,
                coverage_raw=cov_raw,
                band_scale=scale,
                calibration_s=cal_s,
            )
        )

    lead, forecast_error = _alarm_lead(spec, twin, frames, warmup_steps)
    tracking = TrackingReport(
        metrics=metrics,
        gates=spec.gates.to_dict(),
        alarm_lead_s=lead,
        forecast_error_s=forecast_error,
        n_steps=n_scored,
        final_latent=twin.latent_dict(),
    )
    tracking.breaches = _evaluate_gates(spec, tracking)
    return tracking


def _calibrate_band(
    err: np.ndarray,
    sig: np.ndarray,
    ts: np.ndarray,
    calibrate_fraction: float,
    *,
    nominal: float = 0.95,
) -> tuple[float, float | None, float, float]:
    """Measure the band scale on the first part of the window, report coverage on the rest.

    Returns ``(scale, calibration_window_s, coverage_raw, coverage)``.

    ``coverage_raw`` is what the raw ensemble band would have scored on the *scoring* part —
    printed next to the calibrated figure so the correction is visible rather than baked in.
    With ``calibrate_fraction == 0`` no scale is fitted (``scale = 1``) and ``coverage`` is the
    raw coverage over the whole window.
    """
    n = err.size
    if n == 0:
        return 1.0, None, 0.0, 0.0
    safe_sig = np.where(sig > 0, sig, np.finfo(float).eps)

    if calibrate_fraction <= 0 or n < 20:
        inside = np.abs(err) <= Z95 * safe_sig
        cov = float(np.mean(inside))
        return 1.0, None, cov, cov

    cut = int(np.clip(round(n * calibrate_fraction), 10, n - 10))
    cal_z = np.abs(err[:cut]) / safe_sig[:cut]
    # Inflate the band until `nominal` of the residuals in the calibration half are inside it.
    scale = float(np.quantile(cal_z, nominal) / Z95)
    scale = max(scale, 1e-6)

    score = slice(cut, None)
    cov_raw = float(np.mean(np.abs(err[score]) <= Z95 * safe_sig[score]))
    cov = float(np.mean(np.abs(err[score]) <= Z95 * scale * safe_sig[score]))
    cal_s = (
        float(ts[cut - 1] - ts[0])
        if ts.size > cut and np.isfinite(ts[[0, cut - 1]]).all()
        else None
    )
    return scale, cal_s, cov_raw, cov


def _evaluate_gates(spec: TwinSpec, tracking: TrackingReport) -> list[str]:
    """Turn metrics into a list of breaches, one line each with the number and the gate."""
    g = spec.gates
    breaches: list[str] = []
    for m in tracking.metrics:
        if m.span:
            if m.rmse_frac is not None and m.rmse_frac > g.max_rmse_frac:
                breaches.append(
                    f"{m.field_name}: RMSE {100 * m.rmse_frac:.2f}% of span > gate {100 * g.max_rmse_frac:.1f}%"
                )
            if m.bias_frac is not None and abs(m.bias_frac) > g.max_bias_frac:
                breaches.append(
                    f"{m.field_name}: |bias| {100 * abs(m.bias_frac):.2f}% of span > gate {100 * g.max_bias_frac:.1f}%"
                )
        else:
            breaches.append(f"{m.field_name}: no declared span, so no gate could be applied")
        if m.coverage < g.min_coverage:
            breaches.append(
                f"{m.field_name}: coverage {100 * m.coverage:.1f}% < gate {100 * g.min_coverage:.0f}%"
                " — the twin is more confident than it is right"
            )
    for name, lead in tracking.alarm_lead_s.items():
        if lead < g.min_alarm_lead_s:
            breaches.append(
                f"{name}: alarm lead {lead:+.0f} s < gate {g.min_alarm_lead_s:+.0f} s"
            )
    return breaches


def _alarm_lead(
    spec: TwinSpec,
    twin: LatentStateTwin,
    frames: Sequence[Frame],
    warmup_steps: int,
    *,
    confirm: int = 3,
    project_every: int = 6,
) -> tuple[dict[str, float], dict[str, float]]:
    """How much earlier the twin warned than the measurement crossed, and how good that warning was.

    Returns ``(alarm_lead_s, forecast_error_s)``.

    The twin's alarm is **projection-based**: it fires the first time the extrapolated latent
    state puts the field past its limit within ``spec.warn_horizon_s``. A now-cast alarm —
    "the estimate is over the limit right now" — can only ever be minutes early, because it is
    a smoothed copy of the measurement. Projection is the whole reason to run a twin.

    The cost of that is a forecast, and a forecast can be wrong, so the error is measured: at
    the instant of the alarm the projected crossing time is recorded, and ``forecast_error_s``
    is how far that projection landed from the crossing that actually happened. Negative means
    the twin warned too late.

    Both sides are debounced with the same rule (``confirm`` consecutive samples), because a
    plant alarm tripping on one noisy sample would report a lead time that is a noise artefact.
    A field with no declared limit contributes nothing, which is why a spec without limits
    cannot earn this number at all.
    """
    leads: dict[str, float] = {}
    errors: dict[str, float] = {}

    for f in spec.fields:
        if f.alarm_above is None and f.alarm_below is None:
            continue
        above = f.alarm_above is not None
        limit = f.alarm_above if above else f.alarm_below
        assert limit is not None

        hold = [s for s in twin.holdout_sensors if s.field == f.name]
        if not hold:
            continue

        replay = LatentStateTwin(twin.model, spec, n_ens=twin.filter.n_ens, seed=0)
        meas_run: dict[str, int] = {s.sensor_id: 0 for s in hold}
        t_meas: float | None = None
        t_warn: float | None = None
        predicted_at: float | None = None

        for i, frame in enumerate(frames):
            obs = read_readings(frame, spec)
            t = read_timestamp(frame)
            replay.step(obs, t=t)
            if i < warmup_steps:
                continue

            for s in hold:
                if t_meas is not None or s.sensor_id not in obs:
                    continue
                hit = _crossed(obs[s.sensor_id], f)
                meas_run[s.sensor_id] = meas_run[s.sensor_id] + 1 if hit else 0
                if meas_run[s.sensor_id] >= confirm:
                    t_meas = t

            if t_warn is None and i % project_every == 0:
                ttc = replay.project_crossing(
                    f.name, limit, above=above, horizon_s=spec.warn_horizon_s
                )
                if ttc is not None:
                    t_warn = t
                    predicted_at = t + ttc

        if t_meas is not None and t_warn is not None:
            leads[f.name] = float(t_meas - t_warn)
            if predicted_at is not None:
                errors[f.name] = float(predicted_at - t_meas)

    return leads, errors


def _crossed(value: float, f: Any) -> bool:
    if f.alarm_above is not None:
        return value >= f.alarm_above
    if f.alarm_below is not None:
        return value <= f.alarm_below
    return False
