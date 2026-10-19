"""Physics anomaly detection on sensor time series: statistics, time-series models, conservation balances and
physical-model residuals, fused into events with severities and verifiable explanations.

Four families of detectors, all calibrated on a window the user declares normal (never on the data being judged):

``statistical``   per sensor, robust z-score of the value and of its rate of change against the normal window
                  (median / MAD; Rousseeuw & Croux 1993);
``timeseries``    per sensor, an autoregressive model AR(p) of the first difference fitted on the normal window;
                  one-step residuals feed a two-sided CUSUM (Page 1954) that catches slow drifts;
``balance``       conservation laws written as formulas over the columns (:class:`BalanceRule`): over a trailing
                  window ``W``, ``[Acc(t) - Acc(t - W)] / W - mean(inflow - outflow)``; a broken balance says *which*
                  law failed and *by how much* in its own units;
``model``         a physical reference model with ``step(x, v)`` (e.g. the lumped tank or a reduced model) predicts
                  each state one step ahead from the measured state and inputs; the windowed prediction error is the
                  residual.

A sample is flagged when a detector's score (residual / its normal-window spread) exceeds ``threshold``; flags
closer than ``merge_gap`` samples are merged into events, events shorter than ``min_samples`` are dropped.

:func:`fault_benchmark` builds labelled runs of the heated tank with known faults and :func:`score_events`
reports point-wise and event-wise precision / recall, false alarms and detection delays.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from pinneapple_systems.process_components.explicit_equation_system import _names_in_expr, safe_eval

from .plant import Fault, SensorNoise, TankParams, demo_scenario, simulate

METHODS = ("statistical", "timeseries", "balance", "model")
SEVERITIES = ("low", "medium", "high", "critical")


# --------------------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------------------


@dataclass
class BalanceRule:
    """A conservation law over the columns: ``d(accumulation)/dt = inflow - outflow``.

    Each field is a formula over column names and ``constants`` (``+ - * / **``, ``sqrt``, ``exp``, ``log``, ...,
    evaluated safely, no Python ``eval``). ``units`` names the units of the rate (e.g. ``m3/s``, ``W``)."""

    name: str
    accumulation: str
    inflow: str
    outflow: str = "0"
    units: str = ""
    description: str = ""

    def variables(self, constants: Mapping[str, float]) -> list[str]:
        names = set()
        for expr in (self.accumulation, self.inflow, self.outflow):
            names |= _names_in_expr(expr)
        return sorted(n for n in names if n not in constants)


def tank_balance_rules() -> list[BalanceRule]:
    """Mass and energy balances of the heated tank (:mod:`pinneapple_systems.operations.plant`)."""
    return [
        BalanceRule("mass", "A * h", "q_in", "q_out", "m3/s",
                    "volume in the tank changes by what the pump brings minus what the drain takes"),
        BalanceRule("energy", "rho * cp * A * h * T", "rho * cp * q_in * T_in + Q",
                    "rho * cp * q_out * T + UA * (T - T_amb)", "W",
                    "enthalpy in the tank changes by inlet enthalpy + heater power - outlet enthalpy - wall loss"),
    ]


def tank_constants(p: TankParams | None = None) -> dict[str, float]:
    p = p or TankParams()
    return {"A": p.area, "rho": p.rho, "cp": p.cp, "UA": p.ua}


@dataclass
class DetectorConfig:
    """What to run and how strict to be. ``normal_until`` (time, same units as ``t``) ends the normal window."""

    normal_until: float
    sensors: list[str] | None = None              # default: every numeric column except t
    methods: tuple[str, ...] = METHODS
    balances: list[BalanceRule] = field(default_factory=list)
    constants: dict[str, float] = field(default_factory=dict)
    window: int = 12                                  # samples in the balance / model residual window
    threshold: float = 5.0                            # flag when |score| > threshold
    ar_order: int = 3
    cusum_k: float = 0.5                              # CUSUM allowance, in residual sigmas
    cusum_h: float = 12.0                             # CUSUM decision interval, in residual sigmas
    merge_gap: int = 6
    min_samples: int = 3

    def to_dict(self) -> dict:
        d = asdict(self)
        d["methods"] = list(self.methods)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DetectorConfig:
        """Build from a JSON object (balances as a list of objects); unknown keys are an error."""
        bad = set(d) - set(cls.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown detector fields {sorted(bad)}")
        kw = dict(d)
        kw["balances"] = [b if isinstance(b, BalanceRule) else BalanceRule(**b) for b in kw.get("balances", [])]
        if "methods" in kw:
            kw["methods"] = tuple(kw["methods"])
        return cls(**kw)


# --------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------


def _columns(data) -> dict[str, np.ndarray]:
    if hasattr(data, "to_dict") and hasattr(data, "columns"):          # pandas DataFrame
        return {str(c): np.asarray(data[c]) for c in data.columns}
    return {str(k): np.asarray(v) for k, v in dict(data).items()}


def _robust(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 5:
        raise ValueError("the normal window needs at least 5 finite samples per signal")
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    if mad <= 0:
        mad = float(np.std(x)) or 1e-12
    return med, mad


def _trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(x.shape, np.nan)
    out[w:] = (c[w + 1:] - c[1:-w]) / w
    return out


def windowed_balance(t, acc, net_in, w: int) -> np.ndarray:
    """``[acc(t_k) - acc(t_{k-w})] / (t_k - t_{k-w}) - (trapezoid mean of net_in over the same interval)``."""
    t, acc, net_in = (np.asarray(a, dtype=float) for a in (t, acc, net_in))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (net_in[1:] + net_in[:-1]) * np.diff(t))])
    out = np.full(t.shape, np.nan)
    dt = t[w:] - t[:-w]
    out[w:] = (acc[w:] - acc[:-w] - (cum[w:] - cum[:-w])) / dt
    return out


def _ar_fit(x: np.ndarray, p: int) -> np.ndarray:
    X = np.column_stack([np.ones(x.size - p)] + [x[p - j - 1:x.size - j - 1] for j in range(p)])
    coef, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
    return coef


def _ar_residual(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    p = coef.size - 1
    r = np.full(x.shape, np.nan)
    X = np.column_stack([np.ones(x.size - p)] + [x[p - j - 1:x.size - j - 1] for j in range(p)])
    r[p:] = x[p:] - X @ coef
    return r


def _cusum(z: np.ndarray, k: float, clip: float = 4.0) -> np.ndarray:
    """Two-sided CUSUM statistic of standardised residuals ``z`` (signed: + for upward, - for downward). Each
    residual is clipped to ``+-clip`` first (a robust CUSUM): one spike, such as an operator's set-point step,
    cannot cross the decision interval alone; a persistent shift still does."""
    hi = lo = 0.0
    out = np.zeros(z.shape)
    for i, v in enumerate(np.clip(np.nan_to_num(z), -clip, clip)):
        hi = max(0.0, hi + v - k)
        lo = min(0.0, lo + v + k)
        out[i] = hi if hi >= -lo else lo
    return out


def severity_of(ratio: float) -> str:
    """Severity from the peak score / threshold ratio: < 2 low, < 4 medium, < 8 high, else critical."""
    return SEVERITIES[int(np.searchsorted([2.0, 4.0, 8.0], ratio, side="right"))]


# --------------------------------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Channel:
    """One detector output: a residual series, its normal-window centre/spread and score = (r - centre) / spread."""

    method: str
    name: str
    variables: list[str]
    residual: np.ndarray
    centre: float
    spread: float
    units: str = ""
    description: str = ""
    score: np.ndarray = field(default=None)
    threshold: float = 5.0

    def __post_init__(self):
        if self.score is None:
            self.score = (self.residual - self.centre) / self.spread


@dataclass
class Event:
    start: float
    end: float
    i0: int
    i1: int
    severity: str
    peak_ratio: float
    methods: list[str]
    channels: list[str]
    variables: list[str]
    explanations: list[str]
    evidence: list[dict]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


@dataclass
class DetectionResult:
    t: np.ndarray
    channels: list[Channel]
    flags: np.ndarray                  # (n,) bool: any channel above threshold
    events: list[Event]
    config: DetectorConfig
    normal_samples: int

    def to_dict(self, max_points: int = 2000) -> dict:
        step = max(1, int(np.ceil(self.t.size / max_points)))
        idx = slice(None, None, step)

        def clean(a):
            return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)[idx]]

        return {"n": int(self.t.size), "normal_samples": self.normal_samples, "t": clean(self.t),
                "events": [e.to_dict() for e in self.events], "config": self.config.to_dict(),
                "channels": [{"method": c.method, "name": c.name, "variables": c.variables, "units": c.units,
                              "description": c.description, "centre": c.centre, "spread": c.spread,
                              "threshold": c.threshold, "residual": clean(c.residual), "score": clean(c.score),
                              "n_flagged": int(np.sum(np.abs(np.nan_to_num(c.score)) > c.threshold))}
                             for c in self.channels],
                "flagged_fraction": float(self.flags.mean())}


def _model_channels(cols, t, normal, model, state_names, input_names, cfg) -> list[Channel]:
    X = np.column_stack([cols[n].astype(float) for n in state_names])
    V = np.column_stack([cols[n].astype(float) for n in input_names])
    pred = np.asarray(model.step(X[:-1], V[:-1]))[:, :len(state_names)]
    out = []
    for j, name in enumerate(state_names):
        r = np.full(t.shape, np.nan)
        r[1:] = X[1:, j] - pred[:, j]
        rw = _trailing_mean(r, cfg.window)
        c, s = _robust(rw[normal])
        out.append(Channel("model", f"model:{name}", [name] + list(input_names), rw, c, s,
                           description=f"one-step prediction error of {name} by the reference model, averaged over "
                                       f"{cfg.window} samples", threshold=cfg.threshold))
    return out


def detect(data, config: DetectorConfig, model=None, model_states: Sequence[str] = (),
           model_inputs: Sequence[str] = ()) -> DetectionResult:
    """Run the configured detectors on ``data`` (DataFrame or dict of columns with ``t``)."""
    cols = _columns(data)
    if "t" not in cols:
        raise ValueError("data needs a time column 't'")
    t = cols["t"].astype(float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    normal = t <= config.normal_until
    n_norm = int(normal.sum())
    if n_norm < max(30, 4 * config.window):
        raise ValueError(f"the normal window has {n_norm} samples; need at least {max(30, 4 * config.window)}")
    unknown = set(config.methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
    sensors = config.sensors or [k for k, v in cols.items() if k != "t" and np.issubdtype(v.dtype, np.number)]
    missing = [s for s in sensors if s not in cols]
    if missing:
        raise ValueError(f"columns not found: {missing}")
    channels: list[Channel] = []
    th = config.threshold
    if "statistical" in config.methods:
        for s in sensors:
            x = cols[s].astype(float)
            c, sp = _robust(x[normal])
            channels.append(Channel("statistical", f"value:{s}", [s], x, c, sp, description=f"{s} against its "
                                    "normal-window median and spread", threshold=th))
            rate = np.concatenate([[np.nan], np.diff(x) / np.diff(t)])
            c, sp = _robust(rate[normal])
            channels.append(Channel("statistical", f"rate:{s}", [s], rate, c, sp,
                                    description=f"rate of change of {s}", threshold=th))
    if "timeseries" in config.methods:
        for s in sensors:
            x = cols[s].astype(float)
            dx = np.concatenate([[0.0], np.diff(x)])
            coef = _ar_fit(dx[normal][1:], config.ar_order)
            r = _ar_residual(dx, coef)
            c, sp = _robust(r[normal])
            z = (r - c) / sp
            cs = _cusum(z, config.cusum_k)
            channels.append(Channel("timeseries", f"cusum:{s}", [s], r, c, sp, score=cs * th / config.cusum_h,
                                    description=f"CUSUM of the AR({config.ar_order}) one-step residuals of the "
                                                f"increments of {s}", threshold=th))
    if "balance" in config.methods:
        for rule in config.balances:
            env = dict(config.constants)
            env.update({k: v.astype(float) for k, v in cols.items() if np.issubdtype(v.dtype, np.number)})
            try:
                acc = np.broadcast_to(safe_eval(rule.accumulation, env), t.shape)
                net = np.broadcast_to(safe_eval(rule.inflow, env), t.shape) - \
                    np.broadcast_to(safe_eval(rule.outflow, env), t.shape)
            except Exception as e:
                raise ValueError(f"balance {rule.name!r}: {e}") from e
            r = windowed_balance(t, acc, net, config.window)
            c, sp = _robust(r[normal])
            ch = Channel("balance", f"balance:{rule.name}", rule.variables(config.constants), r, c, sp, rule.units,
                         rule.description, threshold=th)
            ch.inflow_mean = float(np.nanmean(np.abs(np.asarray(safe_eval(rule.inflow, env), dtype=float)
                                                     * np.ones(t.shape))[normal]))
            channels.append(ch)
    if "model" in config.methods and model is not None:
        channels += _model_channels(cols, t, normal, model, list(model_states), list(model_inputs), config)

    F = np.column_stack([np.abs(np.nan_to_num(c.score)) > c.threshold for c in channels]) if channels else \
        np.zeros((t.size, 0), bool)
    flags = F.any(axis=1)
    events = _events(t, channels, F, config)
    return DetectionResult(t, channels, flags, events, config, n_norm)


def _explain(c: Channel, i0: int, i1: int, t: np.ndarray) -> tuple[str, dict]:
    seg = c.residual[i0:i1 + 1]
    sc = c.score[i0:i1 + 1]
    k = int(np.nanargmax(np.abs(np.nan_to_num(sc))))
    excess = float(np.nanmean(seg) - c.centre)
    peak = float(seg[k] - c.centre) if np.isfinite(seg[k]) else excess
    dur = float(t[i1] - t[i0])
    ev = {"channel": c.name, "method": c.method, "mean_excess": excess, "peak_excess": peak,
          "peak_score": float(sc[k]), "threshold": c.threshold, "units": c.units, "normal_spread": c.spread}
    if c.method == "balance":
        name = c.name.split(":", 1)[1]
        ref = getattr(c, "inflow_mean", 0.0)
        rel = f" ({abs(excess) / ref:.1%} of the mean inflow term)" if ref > 0 else ""
        direction = ("accumulation is lower than in - out: something leaves (or an inflow is over-read / an outflow "
                     "under-read)" if excess < 0 else
                     "accumulation is higher than in - out: something enters (or an inflow is under-read / an outflow "
                     "over-read)")
        msg = (f"{name} balance broken by {excess:+.4g} {c.units}{rel} on average over {dur:.0f} s "
               f"(normal spread {c.spread:.3g} {c.units}, {abs(excess) / c.spread:.1f} sigma): {direction}")
        ev["relative_to_inflow"] = abs(excess) / ref if ref > 0 else None
    elif c.method == "model":
        state = c.name.split(":", 1)[1]
        msg = (f"reference model mispredicts {state} by {excess:+.4g} per step on average over {dur:.0f} s "
               f"({abs(excess) / c.spread:.1f} x its normal error)")
    elif c.method == "timeseries":
        s = c.variables[0]
        msg = (f"{s} drifts {'up' if sc[k] > 0 else 'down'} against its own dynamics (CUSUM of AR residuals "
               f"{abs(sc[k]) / c.threshold:.1f} x the decision interval)")
    else:
        what, s = c.name.split(":", 1)
        lo, hi = c.centre - c.threshold * c.spread, c.centre + c.threshold * c.spread
        msg = (f"{s}{' rate' if what == 'rate' else ''} reached {c.residual[i0 + k]:.4g}, outside the normal band "
               f"[{lo:.4g}, {hi:.4g}]")
    return msg, ev


def _events(t: np.ndarray, channels: list[Channel], F: np.ndarray, cfg: DetectorConfig) -> list[Event]:
    flags = F.any(axis=1) if F.size else np.zeros(t.size, bool)
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > cfg.merge_gap:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))
    events = []
    for i0, i1 in groups:
        if i1 - i0 + 1 < cfg.min_samples:
            continue
        active = [j for j in range(len(channels)) if F[i0:i1 + 1, j].any()]
        ratios = {j: float(np.nanmax(np.abs(channels[j].score[i0:i1 + 1])) / channels[j].threshold) for j in active}
        order = sorted(active, key=lambda j: ({"balance": 0, "model": 1, "timeseries": 2, "statistical": 3}
                                              [channels[j].method], -ratios[j]))
        expl, evid = [], []
        for j in order:
            m, e = _explain(channels[j], i0, i1, t)
            expl.append(m)
            evid.append(e)
        peak = max(ratios.values())
        variables = sorted({v for j in active for v in channels[j].variables})
        events.append(Event(float(t[i0]), float(t[i1]), int(i0), int(i1), severity_of(peak), peak,
                            sorted({channels[j].method for j in active}), [channels[j].name for j in order],
                            variables, expl, evid))
    return events


# --------------------------------------------------------------------------------------------------------------
# Benchmark with known faults
# --------------------------------------------------------------------------------------------------------------


def benchmark_faults(t0: float = 7200.0) -> list[Fault]:
    """The faults of :func:`fault_benchmark`, after a fault-free first ``t0`` seconds (the calibration window).
    Magnitudes are small on purpose (a few percent of the nominal flows and powers)."""
    nominal_q, _nominal_Q = 6.6e-3, 2.8e5
    return [
        Fault("leak", t0 + 1800, t0 + 4200, 0.06 * nominal_q),                    # 6 % of the inflow
        Fault("heater", t0 + 6000, t0 + 8400, 0.10),                              # 10 % power loss
        Fault("sensor_bias", t0 + 10200, t0 + 12000, 0.5, "T"),                   # +0.5 degC on T
        Fault("sensor_drift", t0 + 13800, t0 + 17400, 0.06, "h"),                 # +6 cm/h on h
        Fault("stuck_sensor", t0 + 19200, t0 + 21000, target="q_in"),             # flow meter frozen
        Fault("valve", t0 + 22800, t0 + 25200, 0.08),                             # drain 8 % more closed
    ]


def fault_benchmark(seed: int = 0, t0: float = 7200.0, dt: float = 10.0) -> tuple[dict[str, np.ndarray],
                                                                                  list[Fault]]:
    """One labelled run: ``t0`` s of normal operation, then the six faults of :func:`benchmark_faults` separated by
    normal periods. Different ``seed`` = different operating profile and noise (a held-out run)."""
    faults = benchmark_faults(t0)
    duration = faults[-1].end + 1800
    data = simulate(demo_scenario(duration, dt, seed=seed), faults=faults, noise=SensorNoise(), seed=seed)
    return data, faults


@dataclass
class LabelledInterval:
    """A labelled abnormal interval ``[start, end)`` read from data (same interface as a plant ``Fault``)."""

    kind: str
    start: float
    end: float
    target: str = ""

    def active(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (t >= self.start) & (t < self.end)


def intervals_from_labels(t, labels) -> list[LabelledInterval]:
    """Contiguous runs of a label column: empty, ``0``, ``normal`` or ``false`` mean normal, anything else names
    the anomaly. Each run ends at the next sample's time."""
    t = np.asarray(t, dtype=float)
    lab = ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip() for v in labels]
    lab = ["" if v.lower() in ("", "0", "0.0", "normal", "false", "none", "nan") else v for v in lab]
    out, i = [], 0
    while i < len(lab):
        if not lab[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        end = t[j + 1] if j + 1 < len(t) else t[j] + (t[j] - t[j - 1] if j > 0 else 1.0)
        out.append(LabelledInterval(lab[i], float(t[i]), float(end)))
        i = j + 1
    return out


def score_events(result: DetectionResult, faults: Sequence[Fault], grace: float = 600.0,
                 evaluate_from: float | None = None) -> dict:
    """Point-wise and event-wise scores against the true fault intervals.

    Point-wise: a sample is positive when a fault is active; predictions are the flagged samples. Event-wise: a
    fault is detected when an event overlaps ``[start, end + grace]`` (the grace covers windowed detectors that
    lag); an event overlapping no fault (with grace) is a false alarm. Only samples after ``evaluate_from``
    (default: the end of the calibration window) count."""
    t = result.t
    t_from = result.config.normal_until if evaluate_from is None else evaluate_from
    ev_mask = t > t_from
    truth = np.zeros(t.size, bool)
    truth_grace = np.zeros(t.size, bool)
    for f in faults:
        truth |= f.active(t)
        truth_grace |= (t >= f.start) & (t < f.end + grace)
    pred = result.flags & ev_mask
    tp = int(np.sum(pred & truth_grace))
    fp = int(np.sum(pred & ~truth_grace))
    fn = int(np.sum(~pred & truth & ev_mask))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = float(np.sum(pred & truth)) / max(1, int(np.sum(truth & ev_mask)))
    events = [e for e in result.events if e.end > t_from]
    per_fault, matched = [], set()
    for f in faults:
        hits = [k for k, e in enumerate(events) if e.end >= f.start and e.start < f.end + grace]
        matched |= set(hits)
        delay = min((max(0.0, events[k].start - f.start) for k in hits), default=None)
        per_fault.append({"kind": f.kind, "target": f.target, "start": f.start, "end": f.end, "detected": bool(hits),
                          "delay_s": delay, "events": len(hits),
                          "methods": sorted({m for k in hits for m in events[k].methods})})
    false_alarms = [events[k].to_dict() for k in range(len(events)) if k not in matched]
    hours = float(t[ev_mask][-1] - t[ev_mask][0]) / 3600.0 if ev_mask.sum() > 1 else 0.0
    det = [p for p in per_fault if p["detected"]]
    return {"point_precision": precision, "point_recall": recall,
            "point_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "event_recall": len(det) / len(faults) if faults else None,
            "event_precision": (len(events) - len(false_alarms)) / len(events) if events else None,
            "false_alarm_events": len(false_alarms), "false_alarms_per_hour": len(false_alarms) / hours if hours else None,
            "mean_delay_s": float(np.mean([p["delay_s"] for p in det])) if det else None,
            "per_fault": per_fault, "false_alarms": false_alarms, "evaluated_hours": hours, "grace_s": grace}


def run_benchmark(seeds: Sequence[int] = (101, 102, 103), methods_sets: dict[str, tuple[str, ...]] | None = None,
                  window: int = 12, threshold: float = 5.0) -> dict:
    """Score each detector family alone and all together on held-out labelled runs (one per seed). The
    calibration window of every run is its own fault-free first two hours; thresholds are never tuned on the
    faulty part. Returns per-method aggregate scores (sums of TP/FP/FN over runs) and per-run details."""
    from .twin import LumpedTankModel

    methods_sets = methods_sets or {"statistical": ("statistical",), "timeseries": ("timeseries",),
                                    "balance": ("balance",), "model": ("model",), "combined": METHODS}
    p = TankParams()
    model = LumpedTankModel(p, dt=10.0, estimate=())
    rows: dict[str, dict] = {}
    for label, methods in methods_sets.items():
        agg = {"tp": 0, "fp": 0, "fn": 0, "detected": 0, "faults": 0, "false_alarm_events": 0, "hours": 0.0,
               "delays": [], "per_kind": {}}
        runs = []
        for seed in seeds:
            data, faults = fault_benchmark(seed)
            cfg = DetectorConfig(normal_until=7200.0, sensors=["h", "T", "q_in", "q_out", "Q"], methods=methods,
                                 balances=tank_balance_rules(), constants=tank_constants(p), window=window,
                                 threshold=threshold)
            res = detect(data, cfg, model=model, model_states=["h", "T"], model_inputs=["q_in", "Q", "T_in", "T_amb"])
            sc = score_events(res, faults)
            runs.append({"seed": seed, **{k: v for k, v in sc.items() if k not in ("false_alarms",)}})
            for k in ("tp", "fp", "fn", "false_alarm_events"):
                agg[k] += sc[k]
            agg["hours"] += sc["evaluated_hours"]
            agg["faults"] += len(faults)
            for pf in sc["per_fault"]:
                agg["detected"] += pf["detected"]
                key = pf["kind"] + (f"({pf['target']})" if pf["target"] else "")
                agg["per_kind"].setdefault(key, [0, 0])
                agg["per_kind"][key][0] += pf["detected"]
                agg["per_kind"][key][1] += 1
                if pf["detected"]:
                    agg["delays"].append(pf["delay_s"])
        prec = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
        rec = float(np.mean([r["point_recall"] for r in runs]))
        rows[label] = {"methods": list(methods), "point_precision": prec, "point_recall": rec,
                       "point_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                       "event_recall": agg["detected"] / agg["faults"],
                       "false_alarms_per_hour": agg["false_alarm_events"] / agg["hours"] if agg["hours"] else None,
                       "mean_delay_s": float(np.mean(agg["delays"])) if agg["delays"] else None,
                       "per_fault_kind": {k: f"{v[0]}/{v[1]}" for k, v in agg["per_kind"].items()},
                       "runs": runs}
    return {"seeds": list(seeds), "window": window, "threshold": threshold, "results": rows,
            "protocol": "synthetic heated-tank runs with six injected faults each; detectors calibrated on the "
                        "fault-free first 2 h of each run; scored on the remaining 7.5 h; seeds not used during "
                        "development of the detectors"}
