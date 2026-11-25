"""The twin benchmark (#57): the same rigour for twins that a surrogate gets, measured on one case.

A surrogate benchmark reports an error. A twin is judged on six other things as well, and this module measures them
on the same case, with the protocol written down and a results table at the end:

**state error** -- how far the estimate is from the plant, against the plant's own truth: RMSE of the outlet
temperature, the slow parameter's error at the end of the run, NIS against its chi-square band, NEES over independent
plants, and the *persistence* baseline (estimate the next reading with the previous one). A twin that cannot beat
persistence has not earned the filter.

**latency** -- the hard constraint of a real-time twin, ``T_inference <= T_data_input``
(:mod:`pinneapple_systems.digital_twin.realtime_pipeline` states it and enforces it): the per-reading cost of
forecast + innovation + analysis at the case's ensemble size, its 95th percentile, and the share of readings that
would overrun the data period. The *recovery* path is timed separately, because a recalibration runs an inverse
workflow over the alarm window: what matters is whether it fits inside the window it collects
(``recalibration_window * dt``) or arrives after the fact.

**robustness to noise** -- the plant's readings are made 0.5x .. 4x noisier than the twin was told
(``sensor_std``), which is the commissioning error that actually happens: RMSE ratio against the nominal case, mean
NIS and whether it stays inside its band.

**robustness to sensor failure** -- four failure modes on the two-sensor channel, each injected into a simulated
record: *dropout* (a fraction of readings missing at random), *outage* (a contiguous block missing on one sensor),
*stuck* (a sensor repeating its last value) and *bias* (a step offset). For each: the state RMSE during the failure
against the healthy baseline, whether the twin raised anything, and how many readings it took; a twin that keeps a
finite, sane estimate through a dead sensor and *says* it lost it is the deliverable -- one that silently
interpolates is not.

**anomaly detection** -- spikes injected at known steps: detection rate, detection delay and false alarms per 100
readings, over several seeds.

**remaining useful life** -- a wear scenario with a known failure time (the wall conductance ramping to a stated
threshold): the twin's estimate of the parameter is the health indicator, ``estimate_rul``
(:mod:`pinneapple_systems.digital_twin.prognostics`) extrapolates it, and the RUL error is reported at 40 %, 60 % and
80 % of life, with the share inside a stated +-30 % band (the PHM alpha-lambda idea, at the checkpoints stated rather
than as a cone over every lambda).

**decision** -- not only the error: the action the twin's estimate would drive. Each reading is classified into
*safe* / *warn* / *stop* by the outlet temperature against the case's limits, once from the twin's estimate and once
from the truth, and the disagreement is priced by a documented cost matrix (a missed stop is 100, a false stop 1, a
missed warning 10, a false warning 1). The result is the agreement rate, the cost of the twin's actions, the cost of
a perfect-information operator (0) and the cost of never acting -- the number an operations manager actually asks
for. :mod:`pinneapple_analysis.decision` holds the same idea for design surrogates (``flip_rate``, ``regret``): it
compares *designs* pairwise, and a twin has no pairwise designs, so the decision here is the action confusion at a
stated limit, priced.

Protocol
--------
Fixed seeds per scenario, the plant's true parameters stated (a twin is never commissioned exactly), the sensor noise
and the thresholds the twin is told stated in the protocol block of the result, and every number reproducible from
``seed``. ``quick=True`` (the default) keeps each run to a couple of hundred readings and 80 members -- the table is
then a few seconds; ``quick=False`` multiplies the readings, the members and the seeds, which is what a published
number should be measured at.

The case defaults to :func:`~pinneapple_systems.digital_twin.workflow.heated_channel` (two temperature sensors around
a fouling wall conductance), because sensor failure needs more than one output to be interesting; any built-in model
or ``module:function`` can be passed instead, and the sections that need two sensors say so when it has one.

Usage
-----
    from pinneapple_systems.digital_twin.benchmark import list_benchmark, twin_benchmark

    r = twin_benchmark()                       # the heated channel, quick sizes
    print(list_benchmark(r))                   # the results table, as markdown
    r["sections"]["decision"]["cost"]["twin"]   # one section's numbers, as measured

Scope: nothing here is a truth it does not have. Every scenario is a simulated plant, so every number is scored
against a known truth and the result says ``synthetic``; a twin on a real record is scored by
:func:`pinneapple_systems.digital_twin.end_to_end.end_to_end_twin` and its drift verdict instead.
"""
from __future__ import annotations

import html
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .prognostics import estimate_rul
from .workflow import (
    StateSpaceModel,
    TwinFilter,
    chi2_band,
    digital_twin_workflow,
    resolve_model,
    simulate_plant,
)

__all__ = ["CASES", "benchmark_case", "cases", "list_benchmark", "twin_benchmark"]

# the cases the benchmark has been calibrated on, with the decision limits that come from their own physics
CASES: dict[str, dict[str, Any]] = {
    "heated_channel": {
        "model_params": {},
        "warn_fraction": 0.70,      # of the steady outlet rise: the level at which an operator wants to look
        "stop_fraction": 0.90,      # of it: the level at which the plant is tripped
        "failure_drop": 0.70,       # the wall conductance at end of life, as a fraction of nominal
        "plant_spread": 0.05,       # how far the commissioned plant is from nominal (a twin is never fitted exactly)
    },
    "thermal_plant": {"model_params": {}, "warn_fraction": 0.70, "stop_fraction": 0.90, "failure_drop": 0.70,
                      "plant_spread": 0.05},
}

ACTIONS = ("safe", "warn", "stop")
# the price of each (truth, twin) pair: missing a warning costs 10, missing a stop costs 100, a false alarm costs 1
DECISION_COST: dict[tuple[str, str], float] = {
    ("safe", "safe"): 0.0, ("safe", "warn"): 1.0, ("safe", "stop"): 1.0,
    ("warn", "safe"): 10.0, ("warn", "warn"): 0.0, ("warn", "stop"): 1.0,
    ("stop", "safe"): 100.0, ("stop", "warn"): 100.0, ("stop", "stop"): 0.0,
}


def cases() -> list[str]:
    """The benchmark cases that carry decision limits."""
    return sorted(CASES)


def benchmark_case(name: str = "heated_channel") -> dict[str, Any]:
    """The case's protocol: its model, the plant's true parameters, the decision limits and the failure level."""
    if name not in CASES:
        raise ValueError(f"case: one of {', '.join(CASES)} (or pass model= to twin_benchmark)")
    spec = CASES[name]
    m = resolve_model(name, spec["model_params"])
    out_name = m.outputs[-1]
    steady = _steady_output(m)
    t0 = _initial_output(m)
    rise = steady - t0
    return {"name": name, "model": m, "output": out_name, "plant_params": None,
            "limits": {"warn": t0 + spec["warn_fraction"] * rise, "stop": t0 + spec["stop_fraction"] * rise},
            "failure_param": m.params[-1],
            "failure_value": spec["failure_drop"] * m.nominal[m.params[-1]],
            "plant_spread": spec["plant_spread"], "steady": steady}


def _initial_output(m: StateSpaceModel) -> float:
    th = np.array([[m.nominal[p] for p in m.params]])
    return float(m.observe(np.asarray(m.x0, float)[None, :], th)[0, -1])


def _steady_output(m: StateSpaceModel, u: float | Sequence[float] | None = None, steps: int = 6000,
                   dt: float = 1.0) -> float:
    """The outlet the plant reaches with the input held, by running it: the limits have to sit where the plant is."""
    x = np.asarray(m.x0, float)[None, :]
    th = np.array([[m.nominal[p] for p in m.params]])
    uu = np.asarray([float(u)] * len(m.inputs) if isinstance(u, (int, float)) else
                    (np.asarray(u, float) if u is not None else
                     np.asarray([2000.0 if i == 0 else float(x[0, 0]) for i in range(len(m.inputs))], float)))
    for _ in range(steps):
        x = m.step(x, uu, th, dt)
    return float(m.observe(x, th)[0, -1])


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def _plant(case: Mapping[str, Any], *, n_steps: int, dt: float, plant_params: Mapping[str, float] | None = None,
           sensor_scale: float = 1.0, drift: Mapping[str, Any] | None = None,
           spike: Mapping[str, Any] | None = None, horizon: int = 0, seed: int = 0) -> dict[str, Any]:
    """A simulated plant at the case's parameters, with its readings scaled and its truth kept."""
    m: StateSpaceModel = case["model"]
    spread = float(case["plant_spread"])
    pp = dict(plant_params or {p: (1.0 + spread) * v for p, v in m.nominal.items()})
    scaled = m
    if sensor_scale != 1.0:                      # the plant's readings are noisier than the twin was told
        scaled = StateSpaceModel(**{**m.__dict__, "sensor_std": [s * sensor_scale for s in m.sensor_std]})
    sim = simulate_plant(scaled, n_steps=n_steps, dt=dt, plant_params=pp, drift=drift, spike=spike,
                         horizon=horizon, seed=seed)
    sim["plant_params"] = pp
    return sim


# a recalibration window longer than any record: the twin runs with its alarms but never runs the inverse workflow
NO_RECALIBRATION = 10 ** 6


def _run(case: Mapping[str, Any], sim: Mapping[str, Any], *, members: int, dt: float = 1.0,
         recalibration_window: int | None = 50, horizon: int = 0, anomaly_z: float = 4.0,
         seed: int = 0) -> dict[str, Any]:
    """The twin over a simulated record, always told the case's own sensor noise (that is the commissioning).

    ``recalibration_window=None`` disables the recalibration (the window never closes): the sections that score the
    parameter's own trajectory -- the wear scenario in particular -- must not have it reset mid-run by an inverse
    workflow, and the sections that only measure the estimate do not need it. The latency section times one.
    """
    m: StateSpaceModel = case["model"]
    rec = {k: np.asarray(v) for k, v in sim["record"].items() if k != "dt"}
    return digital_twin_workflow(model=m, source="arrays", data=rec, members=members, horizon=horizon,
                                 anomaly_z=anomaly_z,
                                 recalibration_window=(NO_RECALIBRATION if recalibration_window is None
                                                       else recalibration_window),
                                 seed=seed)


def _state_error(case: Mapping[str, Any], r: Mapping[str, Any], truth: Any,
                 output_index: int = -1) -> dict[str, float]:
    """The estimate against the truth the plant kept (``truth`` is the state array, or a mapping holding ``states``),
    and against persistence."""
    m: StateSpaceModel = case["model"]
    x = np.asarray(r["trace"]["x"], float)
    y = np.asarray(r["measured"], float)
    T = np.asarray(truth["states"] if isinstance(truth, Mapping) else truth, float)
    n = min(len(x), len(T))
    est = m.observe(x[:n], np.array([[r["final"]["theta"][p] for p in m.params]] * n))[:, output_index]
    true = T[:n, -1]
    return {"rmse": _rmse(est, true), "persistence_rmse": _rmse(y[1:n - 1], true[2:n]),
            "n": int(n)}


def _latency(case: Mapping[str, Any], sim: Mapping[str, Any], *, members: int, dt: float,
             budget_s: float, seed: int = 0) -> dict[str, Any]:
    """The online path timed reading by reading, and the recovery path timed once."""
    m: StateSpaceModel = case["model"]
    rec = sim["record"]
    U = np.column_stack([np.asarray(rec[k], float) for k in m.inputs])
    Y = np.column_stack([np.asarray(rec[k], float) for k in m.outputs])
    f = TwinFilter(m, members=members, x0=Y[0, :len(m.states)], seed=seed)
    times = []
    for k in range(len(Y)):
        t0 = time.perf_counter()
        if k:
            f.forecast(U[k], dt)
        inn = f.innovation(Y[k])
        f.analyse(Y[k], inn["hx"])
        times.append(time.perf_counter() - t0)
    t = np.asarray(times)
    from .workflow import recalibrate
    t1 = time.perf_counter()
    recalibrate(m, U[50:100], Y[50:100], dt, x_start=Y[49, :len(m.states)], theta_start=f.summary()["theta"],
                bootstrap=20, n_recovery=6, seed=seed)
    t_recal = time.perf_counter() - t1
    return {"readings": int(len(t)), "members": int(members), "mean_ms": float(t.mean() * 1e3),
            "p95_ms": float(np.quantile(t, 0.95) * 1e3), "max_ms": float(t.max() * 1e3),
            "budget_ms": float(budget_s * 1e3), "over_budget": float(np.mean(t > budget_s)),
            "throughput_hz": float(1.0 / t.mean()) if t.mean() else float("inf"),
            "recalibration_s": float(t_recal),
            "recalibration_window_s": float(50 * dt),
            "recalibration_fits_the_window": bool(t_recal <= 50 * dt)}


def _noise(case: Mapping[str, Any], *, scales: Sequence[float], n_steps: int, members: int, dt: float,
           seed: int = 0) -> list[dict[str, Any]]:
    """The plant's readings scaled around the sigma the twin was told; ratios are against the 1.0 x row."""
    rows = []
    for s in scales:
        sim = _plant(case, n_steps=n_steps, dt=dt, sensor_scale=s, seed=seed)
        r = _run(case, sim, members=members, dt=dt, recalibration_window=None, seed=seed)
        se = _state_error(case, r, sim["truth"])
        nis_trace = np.asarray(r["trace"]["nis"], float)[20:]
        nis = float(np.mean(nis_trace)) if nis_trace.size else float("nan")
        ny = len(case["model"].outputs)
        lo, hi = chi2_band(ny, max(10, int(nis_trace.size)))
        rows.append({"sensor_scale": float(s), "rmse": se["rmse"], "nis": nis, "nis_band": [lo, hi],
                     "consistent": bool(lo <= nis <= hi)})
    nominal = next((row["rmse"] for row in rows if row["sensor_scale"] == 1.0), None)
    for row in rows:
        row["rmse_ratio"] = row["rmse"] / nominal if nominal else float("nan")
    return rows


def _fail_sensor(sim: dict[str, Any], mode: str, *, name: str, onset: int, length: int,
                 fraction: float = 0.2, size_sigmas: float = 8.0, seed: int = 0,
                 sensor_std: float = 0.15) -> dict[str, Any]:
    """Inject a sensor failure into the named reading of a simulated record; returns the record and what was done."""
    rec = {k: np.asarray(v, float).copy() for k, v in sim["record"].items()}
    if name not in rec:
        raise ValueError(f"no reading called {name!r} in the record")
    y = rec[name].copy()
    n = len(y)
    stop = min(n, onset + length)
    rng = np.random.default_rng(seed)
    if mode == "dropout":
        idx = rng.choice(np.arange(onset, n), size=int(fraction * (n - onset)), replace=False)
        y[idx] = np.nan
        detail = f"{len(idx)} of {n - onset} readings dropped at random from {name}"
    elif mode == "outage":
        y[onset:stop] = np.nan
        detail = f"{stop - onset} readings lost in a block from {name}"
    elif mode == "stuck":
        y[onset:stop] = y[onset]
        detail = f"{name} repeats {y[onset]:.4g} for {stop - onset} readings"
    elif mode == "bias":
        y[onset:] = y[onset:] + size_sigmas * sensor_std
        detail = f"{name} offset by {size_sigmas:g} sigma from reading {onset}"
    else:
        raise ValueError("mode: dropout, outage, stuck or bias")
    rec[name] = y
    return {"record": rec, "detail": detail, "sensor": name, "onset": int(onset), "end": int(stop)}


def _failure(case: Mapping[str, Any], *, modes: Sequence[str], n_steps: int, members: int, dt: float,
             seed: int = 0) -> list[dict[str, Any]]:
    """Each failure mode injected at the same place of the same plant, scored against the healthy run."""
    onset, length = int(0.4 * n_steps), int(0.2 * n_steps)
    rows = []
    healthy = _plant(case, n_steps=n_steps, dt=dt, seed=seed)
    r0 = _run(case, healthy, members=members, dt=dt, seed=seed)
    base = _state_error(case, r0, healthy["truth"])["rmse"]
    for mode in modes:
        if len(case["model"].outputs) < 2 and mode in ("outage", "stuck", "bias"):
            rows.append({"mode": mode, "status": "not_run",
                         "reason": "one output: a failure of the only sensor leaves nothing to assimilate"})
            continue
        sim = _plant(case, n_steps=n_steps, dt=dt, seed=seed)
        bad = _fail_sensor(sim, mode, name=case["model"].outputs[-1], onset=onset, length=length, seed=seed,
                           sensor_std=float(case["model"].sensor_std[-1]))
        sim["record"] = bad["record"]
        r = _run(case, sim, members=members, dt=dt, recalibration_window=None, seed=seed)
        se = _state_error(case, r, sim["truth"])
        events = [e for e in r["events"] if e["step"] >= onset]
        gaps = [g for g in r["gaps"] if g["step"] >= onset]
        first = events[0]["step"] - onset if events else None
        state = np.asarray(r["trace"]["x"], float)
        # a reading that did not arrive is *recorded* (a gap); a reading that lies (stuck, biased) has to be
        # *detected*, and only an alarm says so. Either way the estimate must survive it.
        rows.append({"mode": mode, "detail": bad["detail"], "rmse": se["rmse"],
                     "rmse_ratio": se["rmse"] / base if base else float("nan"),
                     "finite_state": bool(np.all(np.isfinite(state))),
                     "alarmed": bool(events), "first_event": events[0]["kind"] if events else None,
                     "delay_readings": first, "gaps": len(gaps),
                     "handled": bool((gaps or events) and np.all(np.isfinite(state))),
                     "lies": mode in ("stuck", "bias"),
                     "checks": {c["code"]: c["status"] for c in r["checks"] if c["code"] == "twin.missing_readings"}})
    return rows


def _anomaly(case: Mapping[str, Any], *, n_steps: int, members: int, dt: float, n_spikes: int = 8,
             seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """Spikes at known steps: detection rate, delay, and false alarms per 100 readings."""
    hit = total = 0
    delays = []
    false = readings = 0
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        steps = sorted(int(s) for s in rng.choice(np.arange(30, n_steps - 5), size=n_spikes, replace=False))
        spikes = [{"at": s, "size": 12.0} for s in steps]
        sim = _plant(case, n_steps=n_steps, dt=dt, seed=seed)
        for s in spikes:                       # each spike into the *reading*, which is what a sensor does
            out = case["model"].outputs[0]
            sim["record"][out][s["at"]] += 12.0 * float(case["model"].sensor_std[0])
        r = _run(case, sim, members=members, dt=dt, seed=seed)
        flagged = [e["step"] for e in r["events"] if e["kind"] == "anomaly"]
        for s in spikes:
            total += 1
            near = [f for f in flagged if s["at"] <= f <= s["at"] + 3]
            if near:
                hit += 1
                delays.append(near[0] - s["at"])
            false += sum(1 for f in flagged if min(abs(f - s2["at"]) for s2 in spikes) > 3)
        readings += n_steps
    return {"spikes": int(total), "detected": int(hit), "detection_rate": hit / total if total else float("nan"),
            "delay_readings_mean": float(np.mean(delays)) if delays else float("nan"),
            "false_alarms": int(false), "false_per_100": 100.0 * false / readings if readings else float("nan"),
            "seeds": list(seeds)}


def _rul(case: Mapping[str, Any], *, n_steps: int, members: int, dt: float, checkpoints: Sequence[float],
         tolerance: float = 0.30, at_fraction: float = 0.15, multiple: float = 2.0, smooth: int = 10,
         wear_drop: float = 0.85, seed: int = 0) -> dict[str, Any]:
    """Remaining useful life on a wear scenario, with the health indicator a twin on a real plant could build.

    The plant's slow parameter ramps linearly from its nominal value (starting at ``at_fraction`` of the record) to
    the case's failure value. The twin is run **without** recalibration, which is the honest configuration for
    prognostics: a recalibration would reset the very estimate the degradation lives in, and the filter's own
    parameter random walk is deliberately too slow to follow a wear ramp (that is why the workflow recalibrates in the
    first place). So the health indicator is the one
    :mod:`pinneapple_systems.digital_twin.prognostics` documents: the magnitude of the twin's normalised innovations,
    smoothed over ``smooth`` readings -- higher is worse, and it needs no truth to build.

    The end of life is then defined **observationally**: the first reading at which the indicator reaches
    ``multiple`` times its median over the healthy prefix (before the ramp starts). That crossing is computed from the
    whole record and then withheld: the estimator sees only the readings up to each checkpoint and has to extrapolate
    the crossing that the full record shows. If the indicator never crosses, the section says so instead of inventing
    an end of life.

    The wear is ``wear_drop`` of the parameter's nominal value over the record -- 8 % by default, and stated: a 30 %
    loss is caught within a few readings and leaves no trend to extrapolate, so the scenario has to be gentle for the
    question (how much life is left) to have an answer.
    """
    m: StateSpaceModel = case["model"]
    p = case["failure_param"]
    fail_at = float(wear_drop) * float(m.nominal[p])
    at = int(at_fraction * n_steps)
    drift = {"param": p, "at": at, "to": fail_at, "ramp": n_steps - at}
    sim = _plant(case, n_steps=n_steps, dt=dt, drift=drift, seed=seed)
    r = _run(case, sim, members=members, dt=dt, recalibration_window=None, seed=seed)
    z = np.abs(np.asarray(r["trace"]["z"], float))
    t = np.asarray(r["t"], float)
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    if z.size < 4 * smooth + at:
        return {"status": "not_run", "reason": f"{z.size} usable readings: too few to smooth a trend up to the ramp"}
    h = np.convolve(z, np.ones(smooth) / smooth, mode="same")[smooth:-smooth]
    t = t[smooth:-smooth]
    healthy = h[:max(smooth, int(0.9 * at) - smooth)]
    z_fail = float(multiple * np.median(healthy)) if healthy.size else float("nan")
    crossing = np.nonzero(h >= z_fail)[0]
    theta = np.asarray(r["trace"]["theta"], float)[:, m.params.index(p)]
    if not crossing.size or crossing[0] <= at + smooth + 4:
        # no end of life to predict, and *why* is the result: the two indicators are measured, not assumed
        facts = {"healthy_level": float(np.median(healthy)) if healthy.size else float("nan"),
                 "threshold": z_fail, "multiple": float(multiple), "max_indicator": float(h.max()),
                 "readings": int(len(h)), "wear_from_reading": int(at),
                 "parameter_end": float(theta[-1]), "parameter_nominal": float(m.nominal[p]),
                 "parameter_wear_to": fail_at}
        if crossing.size:
            facts["end_of_life_reading"] = int(crossing[0])
            reason = (f"the indicator reached its threshold at reading {int(crossing[0])}, "
                      f"{int(crossing[0]) - at} readings after the wear started: the fault is found at once and "
                      f"there is no trend left to extrapolate")
        else:
            reason = (f"the indicator never reached {multiple:g}x its healthy level ({z_fail:.3g}) over this ramp "
                      f"(it peaked at {h.max():.3g}): the filter follows the wear by adapting its parameter "
                      f"(from {m.nominal[p]:.4g} to {theta[-1]:.4g} against a plant that went to {fail_at:.4g}), so "
                      f"the residual is not a degradation signal on this case, and there is no end of life to "
                      f"predict from one")
        return {"status": "not_run", "verdict": "not_measurable", "reason": reason, "facts": facts}
    t_fail = int(crossing[0])
    rows = []
    for frac in checkpoints:                       # fractions of the wear itself, from its start to the crossing
        k = max(smooth + 4, min(int(at + frac * (t_fail - at)), t_fail - 2))
        # the trend is fitted on the recent window and not on the whole history: before the wear starts the
        # indicator is flat by definition, and a fit that spends most of its points on the flat part extrapolates a
        # slope of nothing (that is what the first version of this section measured, and why it changed)
        w0 = max(smooth, min(k - 20, int(k - max(20, 0.5 * (k - at)))))
        try:
            est = estimate_rul(h[w0:k + 1], timestamps=t[w0:k + 1], failure_threshold=z_fail, model="linear")
            rul, r2 = float(est.estimated_rul), float(est.fit_quality)
        except (ValueError, np.linalg.LinAlgError):
            rul, r2 = float("nan"), 0.0
        true_left = max(0.0, (t_fail - k)) * (float(t[1] - t[0]) if len(t) > 1 else dt) / dt
        rel = abs(rul - true_left) / true_left if (true_left > 0 and np.isfinite(rul)) else float("nan")
        rows.append({"fraction_of_wear": float(frac), "from_reading": int(w0), "reading": int(k),
                     "indicator": float(h[k]),
                     "rul_estimate": rul, "rul_true": float(true_left), "relative_error": rel,
                     "fit_quality": r2, "reliable": bool(r2 >= 0.7),      # the module's own reliability limit
                     "inside_tolerance": bool(np.isfinite(rel) and rel <= tolerance)})
    reliable = [row for row in rows if row["reliable"]]
    inside = sum(1 for row in reliable if row["inside_tolerance"])
    if not reliable:
        return {"status": "not_run", "verdict": "not_measurable", "rows": rows, "threshold": z_fail,
                "healthy_level": float(np.median(healthy)), "end_of_life_reading": int(t_fail), "of": len(rows),
                "reason": f"the indicator has no usable trend at any checkpoint (R^2 "
                          f"{min(row['fit_quality'] for row in rows):.2f} to "
                          f"{max(row['fit_quality'] for row in rows):.2f}, against the 0.7 the prognostics module "
                          f"calls reliable): a smoothed residual does not degrade in a straight line while the "
                          f"filter follows the wear, so a linear extrapolation of it predicts nothing. The end of "
                          f"life the record shows is at reading {int(t_fail)}, of {int(len(h))}."}
    return {"rows": reliable, "rows_all": rows, "checkpoints": list(checkpoints), "tolerance": tolerance,
            "healthy_level": float(np.median(healthy)), "threshold": z_fail, "multiple": float(multiple),
            "smoothing": int(smooth), "end_of_life_reading": int(t_fail), "readings": int(len(h)),
            "wear_to": float(fail_at), "wear_from_reading": int(at),
            "inside": int(inside), "of": len(reliable),
            "verdict": "pass" if inside == len(reliable) else ("partial" if inside else "fail")}


def _actions(case: Mapping[str, Any], r: Mapping[str, Any], truth: Any,
             limits: Mapping[str, float] | None = None) -> dict[str, Any]:
    """What the twin's estimate would do against what the truth would do, priced by DECISION_COST.

    The three action classes have to occur for the comparison to mean anything, so when the case declares no limits
    (``limits=None``) they are placed on the case's own operating range: the 60th and 85th percentile of the true
    outlet over the record. That is a design choice for the exercise, stated in the result, not a measurement; pass
    ``limits={"warn": .., "stop": ..}`` to use the plant's own --- which is what a real installation has, and what
    the ``CASES`` entry carries as the physical limits of the case.
    """
    m: StateSpaceModel = case["model"]
    x = np.asarray(r["trace"]["x"], float)
    T = np.asarray(truth["states"] if isinstance(truth, Mapping) else truth, float)
    n = min(len(x), len(T))
    th = np.array([[r["final"]["theta"][p] for p in m.params]] * n)
    est = m.observe(x[:n], th)[:, -1]
    true = T[:n, -1]
    if limits is None:
        lim = {"warn": float(np.quantile(true, 0.60)), "stop": float(np.quantile(true, 0.85)),
               "basis": "the 60th and 85th percentile of the true outlet over the record (the exercise needs all "
                        "three action classes to occur)"}
    else:
        lim = {"warn": float(limits["warn"]), "stop": float(limits["stop"]),
               "basis": "the limits the case declares"}

    def act(v: np.ndarray) -> np.ndarray:
        a = np.full(len(v), "safe", dtype=object)
        a[v >= lim["warn"]] = "warn"
        a[v >= lim["stop"]] = "stop"
        return a
    a_twin, a_true = act(est), act(true)
    cost = float(np.sum([DECISION_COST[(str(t), str(s))] for t, s in zip(a_true, a_twin, strict=True)]))
    cost_never = float(np.sum([DECISION_COST[(str(t), "safe")] for t in a_true]))
    agree = float(np.mean(a_twin == a_true)) if n else float("nan")
    return {"agreement": agree, "n": int(n), "cost": {"twin": cost, "perfect": 0.0, "never_act": cost_never},
            "cost_per_reading": cost / n if n else float("nan"),
            "counts": {f"{t}->{s}": int(np.sum((a_true == t) & (a_twin == s))) for t in ACTIONS for s in ACTIONS},
            "limits": {k: float(v) for k, v in lim.items() if k != "basis"}, "basis": lim["basis"],
            "case_limits": {k: float(v) for k, v in (case.get("limits") or {}).items()},
            "matrix": {f"{t}->{s}": float(DECISION_COST[(t, s)]) for t in ACTIONS for s in ACTIONS}}


def twin_benchmark(*, case: str = "heated_channel", model: Any = None, model_params: Mapping[str, Any] | None = None,
                   quick: bool = True, seed: int = 0, dt: float = 1.0, budget_s: float | None = None,
                   limits: Mapping[str, float] | None = None, n_steps: int | None = None,
                   members: int | None = None) -> dict[str, Any]:
    """Run the protocol on one case and return every section, the protocol that produced it and the table.

    ``quick`` sizes: 200 readings, 80 members, 4 noise levels, 4 failure modes, 3 anomaly seeds; the full sizes are
    600 readings, 200 members, 6 levels, 4 modes and 5 seeds. ``budget_s`` is the data period the latency section
    measures against (default ``dt``: one reading per second is what the channel was built for). ``n_steps`` and
    ``members`` override the sizes outright, for a caller (a test) that needs the whole protocol faster.
    """
    spec = (benchmark_case(case) if model is None else
            {"name": "custom", "model": resolve_model(model, model_params), "output": None,
             "limits": {}, "failure_param": None, "failure_value": None, "plant_spread": 0.05, "steady": None})
    m: StateSpaceModel = spec["model"]
    if model is not None:                        # a caller's model has no case physics: the limits are the model's own
        steady = _steady_output(m)
        t_0 = _initial_output(m)
        spec = {**spec, "output": m.outputs[-1],
                "limits": {"warn": t_0 + 0.70 * (steady - t_0), "stop": t_0 + 0.90 * (steady - t_0)},
                "failure_param": m.params[-1],
                "failure_value": 0.70 * m.nominal[m.params[-1]], "steady": steady}
    n_steps = int(n_steps or (200 if quick else 600))
    members = int(members or (80 if quick else 200))
    budget = float(dt if budget_s is None else budget_s)
    t_start = time.perf_counter()

    # ---- the reference run: a steady plant, every check the workflow makes scored against a known truth
    pp = {p: (1.0 + float(spec["plant_spread"])) * v for p, v in m.nominal.items()}
    r_ref = digital_twin_workflow(model=m, source="simulated", n_steps=n_steps, dt=dt, plant_params=pp,
                                  drift={}, spike={}, members=members, horizon=40,
                                  nees_runs=20 if quick else 40, recalibration_window=50, seed=seed)
    T_ref = np.asarray(r_ref["truth"]["states"], float)
    se = _state_error(spec, r_ref, T_ref)
    ny = len(m.outputs)
    nis_trace = np.asarray(r_ref["trace"]["nis"], float)[20:]
    nis = float(np.mean(nis_trace)) if nis_trace.size else float("nan")
    lo, hi = chi2_band(ny, max(10, int(nis_trace.size)))
    d_checks = {c["code"]: c["status"] for c in r_ref["checks"]}
    pidx = m.params.index(spec["failure_param"]) if spec["failure_param"] in m.params else -1
    truth_p = float(np.asarray(r_ref["truth"]["params"], float)[-1, pidx]) if pidx >= 0 else float("nan")
    est_p = float(r_ref["final"]["theta"][spec["failure_param"]]) if spec["failure_param"] in m.params else float("nan")
    state_error = {"rmse": se["rmse"], "persistence_rmse": se["persistence_rmse"],
                   "beats_persistence": bool(se["rmse"] < se["persistence_rmse"]),
                   "nis": nis, "nis_band": [lo, hi], "consistent": bool(lo <= nis <= hi),
                   "parameter": spec["failure_param"], "parameter_estimate": est_p, "parameter_true": truth_p,
                   "parameter_relative_error": (abs(est_p - truth_p) / abs(truth_p)
                                                if np.isfinite(est_p) and truth_p else float("nan")),
                   "nees": d_checks.get("twin.nees_consistent"),
                   "forecast_coverage": d_checks.get("twin.forecast_coverage"),
                   "readings": se["n"], "checks": dict(d_checks)}
    ref = _plant(spec, n_steps=n_steps, dt=dt, seed=seed)      # the same plant, for the latency measurement

    # ---- latency, on the same plant
    latency = _latency(spec, ref, members=members, dt=dt, budget_s=budget, seed=seed)

    # ---- the rest of the protocol
    noise = _noise(spec, scales=(0.5, 1.0, 2.0, 4.0) if quick else (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                   n_steps=n_steps, members=members, dt=dt, seed=seed)
    failure = _failure(spec, modes=("dropout", "outage", "stuck", "bias"), n_steps=n_steps, members=members, dt=dt,
                       seed=seed)
    anomaly = _anomaly(spec, n_steps=n_steps, members=members, dt=dt, n_spikes=6 if quick else 10,
                       seeds=(0, 1, 2) if quick else (0, 1, 2, 3, 4))
    rul = _rul(spec, n_steps=int(6 * n_steps), members=members, dt=dt,
               checkpoints=(0.3, 0.6, 0.9), seed=seed) if spec["failure_param"] in m.params else {
        "status": "not_run", "reason": "the case declares no failure parameter to ramp"}
    decision = _actions(spec, r_ref, T_ref, limits)

    verdicts = {
        "state_error": "pass" if (state_error["beats_persistence"] and state_error["consistent"]) else "fail",
        "latency": "pass" if latency["over_budget"] <= 0.05 else "fail",
        "noise": "pass" if all(row["consistent"] for row in noise if row["sensor_scale"] == 1.0) else "review",
        "sensor_failure": "pass" if all(row.get("handled") for row in failure if row.get("status") != "not_run")
        else "review",
        "anomaly": "pass" if anomaly["detection_rate"] >= 0.9 and anomaly["false_per_100"] <= 1.0 else "review",
        "rul": rul.get("verdict", "not_run"),
        "decision": "pass" if decision["agreement"] >= 0.95 else "review",
    }
    protocol = {"case": spec["name"], "model": m.name, "states": m.states, "outputs": m.outputs,
                "params": m.params, "nominal": m.nominal, "plant_params": ref["plant_params"],
                "sensor_std": list(map(float, m.sensor_std)), "process_std": list(map(float, m.process_std)),
                "n_steps": n_steps, "dt": dt, "members": members, "seed": seed, "quick": bool(quick),
                "budget_s": budget, "anomaly_z": 4.0, "recalibration_window": 50,
                "limits": {k: float(v) for k, v in spec["limits"].items()},
                "failure": {"param": spec["failure_param"], "value": spec["failure_value"]},
                "truth": "simulated plant: every number is scored against a truth known by construction",
                "space": "synthetic", "seconds": round(time.perf_counter() - t_start, 2)}
    out = {"case": spec["name"], "protocol": protocol, "verdicts": verdicts,
           "sections": {"state_error": state_error, "latency": latency, "noise": noise, "sensor_failure": failure,
                        "anomaly": anomaly, "rul": rul, "decision": decision},
           "summary": (f"{spec['name']}: state RMSE {state_error['rmse']:.3g} (persistence "
                       f"{state_error['persistence_rmse']:.3g}), {latency['mean_ms']:.2f} ms per reading "
                       f"({latency['p95_ms']:.2f} ms p95) against a {latency['budget_ms']:.0f} ms budget, "
                       f"anomaly {anomaly['detection_rate']:.0%} detected, decision agreement "
                       f"{decision['agreement']:.0%}")}
    return out


# ---------------------------------------------------------------------------------------------- the results table


def list_benchmark(r: Mapping[str, Any]) -> str:
    """The results table of a benchmark run, as markdown: one row per measured number, with its verdict."""
    s = r["sections"]
    p = r["protocol"]
    v = r["verdicts"]
    lines = [f"# Twin benchmark: {p['case']} ({p['model']})", "",
             f"Protocol: {p['n_steps']} readings at dt = {p['dt']:g} s, {p['members']} members, seed {p['seed']}, "
             f"plant at {p['plant_params']}, sensor sigma {p['sensor_std']}, "
             f"limits warn/stop {p['limits']['warn']:.4g}/{p['limits']['stop']:.4g}.", "",
             "| section | measurement | value | verdict |", "|---|---|---|---|"]
    se = s["state_error"]
    lines += [
        f"| state error | RMSE of {p['outputs'][-1]} | {se['rmse']:.4g} | {'pass' if se['beats_persistence'] else 'fail'}"
        f" |",
        f"| state error | RMSE of persistence (the baseline to beat) | {se['persistence_rmse']:.4g} | — |",
        f"| state error | mean NIS against its band [{se['nis_band'][0]:.3g}, {se['nis_band'][1]:.3g}] | "
        f"{se['nis']:.3g} | {'pass' if se['consistent'] else 'fail'} |",
        f"| state error | {se['parameter']}: estimate {se['parameter_estimate']:.5g} against the plant's "
        f"{se['parameter_true']:.5g} | {se['parameter_relative_error']:.2%} | — |",
        f"| state error | NEES check / forecast coverage | {se['nees']} / {se['forecast_coverage']} | — |",
    ]
    la = s["latency"]
    lines += [
        f"| latency | per reading (mean / p95 / max) | {la['mean_ms']:.2f} / {la['p95_ms']:.2f} / "
        f"{la['max_ms']:.2f} ms | — |",
        f"| latency | readings over the {la['budget_ms']:.0f} ms data period | {la['over_budget']:.1%} | "
        f"{v['latency']} |",
        f"| latency | one recalibration (inverse workflow) | {la['recalibration_s']:.2f} s against a "
        f"{la['recalibration_window_s']:.1f} s window | "
        f"{'pass' if la['recalibration_fits_the_window'] else 'fail'} |",
    ]
    for row in s["noise"]:
        lines.append(f"| noise | plant readings at {row['sensor_scale']:g}x the sigma the twin was told: RMSE "
                     f"{row['rmse']:.4g} ({row['rmse_ratio']:.2f}x), mean NIS {row['nis']:.3g} | "
                     f"{'consistent' if row['consistent'] else 'outside the band'} | — |")
    for row in s["sensor_failure"]:
        if row.get("status") == "not_run":
            lines.append(f"| sensor failure | {row['mode']} | not run: {row['reason']} | — |")
            continue
        lines.append(f"| sensor failure | {row['mode']}: RMSE {row['rmse']:.4g} ({row['rmse_ratio']:.2f}x the "
                     f"healthy run), finite state {row['finite_state']}, alarmed {row['alarmed']}"
                     + (f" after {row['delay_readings']} readings ({row['first_event']})" if row["alarmed"] else "")
                     + r" | — |")
    an = s["anomaly"]
    lines.append(f"| anomaly | spikes detected | {an['detected']}/{an['spikes']} ({an['detection_rate']:.0%}), delay "
                 f"{an['delay_readings_mean']:.2g} readings, {an['false_per_100']:.2f} false alarms per 100 readings "
                 f"| {v['anomaly']} |")
    ru = s["rul"]
    if ru.get("status") == "not_run":
        lines.append(f"| RUL | not measurable on this case | {html.escape(ru['reason'])} | "
                     f"{ru.get('verdict', 'not_run')} |")
    else:
        for row in ru["rows"]:
            lines.append(f"| RUL | at {row['fraction_of_wear']:.0%} of the wear (reading {row['reading']}, "
                         f"fitted on {row['from_reading']}-{row['reading']}): {row['rul_estimate']:.4g} against "
                         f"{row['rul_true']:.4g} | error {row['relative_error']:.1%} | "
                         f"{'inside' if row['inside_tolerance'] else 'outside'} +-{ru['tolerance']:.0%} |")
    de = s["decision"]
    lines += [
        f"| decision | actions agreeing with the truth's | {de['agreement']:.1%} of {de['n']} readings | "
        f"{v['decision']} |",
        f"| decision | cost: twin {de['cost']['twin']:.0f}, never act {de['cost']['never_act']:.0f}, perfect "
        f"{de['cost']['perfect']:.0f} | {de['cost_per_reading']:.3g} per reading | — |",
        f"| decision | limits warn/stop {de['limits']['warn']:.4g}/{de['limits']['stop']:.4g} | {de['basis']} | — |",
    ]
    lines += ["", "Verdicts: " + ", ".join(f"{k} {val}" for k, val in v.items()) + ".",
              f"Measured in {p['seconds']} s ({'quick' if p['quick'] else 'full'} sizes) on a "
              f"{p['space']} plant."]
    return "\n".join(lines)
