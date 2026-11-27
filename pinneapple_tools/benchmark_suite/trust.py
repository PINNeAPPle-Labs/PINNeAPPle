"""The trust benchmark: four injected failure kinds, one protocol, and a leaderboard anyone can be placed on (#349).

A leaderboard answers "which model is more accurate", and answers it with a mean over in-domain data. This suite asks
the other question -- *does the model know when it stopped being valid?* -- and answers it the only way it can be
answered: by injecting the failures a deployed physics model actually meets, with the step at which each one starts
written down, and measuring what the model and its monitor did about it.

One API
-------
``run_suite(candidates)`` takes a set of candidate models and returns, per candidate and per failure kind, the
measured row; ``write`` turns the run into the record and the published page; ``check`` is the gate. A candidate is
any callable ``predict(X) -> y`` (or ``-> (mean, std)``) on the protocol's inputs, already fitted on the protocol's
in-domain sample -- so a model from anywhere can be placed on the board without touching this file.

The four kinds (each an injected failure with ground truth, not an assumed one)
------------------------------------------------------------------------------
==============  ==============================================================================  ============
kind            what changes, and what the model is fed                                       ground truth
==============  ==============================================================================  ============
regime_change   a **hidden** parameter of the plant steps (a fin array derates, fins are        the step
                blocked); the inputs stay inside the box, so no function of the inputs can
                separate the regimes -- this is the kind a label-free monitor cannot see
sensor_fault    spikes, a stuck value, a bias, and a slow ramp, on the measured **inputs** and  the step
                on the measured **response**; a fault on the response corrupts the labels the
                monitor checks against, so it measures the loop, not only the model
slow_drift      a hidden parameter ramps instead of stepping: two ramps, one fast enough to     the step
                cross the tolerance and one deliberately near the threshold                  the ramp starts
out_of_domain   readings outside the box the model was fitted on (the fan below its range,      the step
                the power above it): the right answer is **"do not trust"**, not a switch
==============  ==============================================================================  ============

The monitor is the protocol's, not the candidate's
--------------------------------------------------
Every candidate is watched by the same monitor: :class:`pinneapple_analysis.trust.drift.DriftMonitor`, built on the
held-out split and calibrated on a second one (the input, prediction, uncertainty, error and envelope checks, each
against the 99 % quantile of genuinely in-distribution windows). The candidate changes the monitor's *input*, never
its code -- that is what makes the detection columns comparable, and what makes a candidate that reports an
uncertainty a candidate that gives the monitor one more thing to check. The alarm is the drift report's own
``has_alarm``: any check that contradicted.

What is measured, per candidate per kind
----------------------------------------
* **missed** -- cases where the monitor never alarmed after the change. It is the first key of the ranking, because a
  model that never notices is the failure this benchmark exists to find, and a mean over the ones it did notice
  would hide it.
* **detect_delay** -- readings from the change to the first alarm, averaged over the kind's cases; a case that was
  missed is censored at the readings that remained, which is a stand-in and is counted as one by *missed*.
* **false_alarms_per_1000** -- alarms on the readings before the change, per 1000 of them. A monitor that alarms on
  everything detects everything; this column is what says so.
* **error_after** -- the mean absolute error after the change, divided by the spread (std) of the in-domain
  responses: 1.0 is an error the size of the whole operating range. This is the truth column, measured against the
  plant, and it is the first tie-break.
* **error_exit_delay** -- readings after the change before that error crossed the declared tolerance: the same
  absolute exit criterion the battle scoreboard uses (:func:`pinneapple_arena.battle.domain_exit`), printed next to
  the monitor's own delay so a monitor that fires long after the error did is visible as such.
* **coverage** -- for a candidate that reports ``(mean, std)``: the share of readings whose truth fell inside
  ``mean +- 2 std``, before and after the change. A stated 95 % band that covers 95 % in domain and 0 % under a
  regime change is the honest picture of a fixed band; a candidate with no uncertainty reports ``not_measured``
  rather than a made-up number.

The ranking is declared as data (``PROTOCOL["ranking"]``), total, and recomputable from the record alone:
``(missed, error_after, detect_delay, false_alarms_per_1000, id)`` -- *did it ever notice, was it wrong, how fast, how
often does it cry wolf* -- with a declared resolution on the error and false-alarm columns, so a kind where every
model misses the change by the same amount is decided by the delay instead of by the last digit. The published
placement is the mean of a candidate's rank over the four kinds, so a model that is excellent at one failure and blind
to another is placed where that average says -- and every column of the row is on the page, so a reader can re-rank
on any of them.

What is deliberately **not** measured here: the calibration of a stated *horizon* (a forecast's "good for N steps"
claim). That belongs to the trust record (:class:`pinneapple_analysis.trust.record.Horizon`, #345) and needs a
forecast, not a reading; this suite measures interval coverage, which is what a reading-level candidate can state.

Measured, not estimated
-----------------------
The published record is ``benchmarks/trust/results.json`` and the page ``docs/trust_benchmark.md``, both written by
``python scripts/trust_benchmark.py --run`` -- two runs of that declaration took 12.1 s and 15.5 s of wall time on the
publishing machine, the only number here that varies, and the record carries its own ``wall_s`` -- and both refused
by ``--check`` when either drifted from the measurements. The candidates of the published run are the one-shot fits of
``pinneapple_analysis.surrogate.methods`` (ridge, GP, gradient-boosted trees) plus a constant band around the ridge,
all fitted on the same in-domain sample, one BLAS thread, so the record reproduces. The numbers of that run are
asserted in ``tests/test_trust_benchmark_suite.py`` against the record, so neither they nor the rules can drift
silently.

What that run measured, as published:

* **the leaderboard**: ``gp`` 1.00, ``ridge`` 2.00, ``ridge_band`` 3.00, ``gbt`` 4.00 -- the GP is first on all four
  kinds, so its row carries no *worst* kind; the trees are last on all four, and the columns say why.
* **in domain**: the GP's mean error is 1.4e-5 to 2.2e-5 of the response spread, the ridge's 0.036 and the trees'
  0.118 to 0.130. A surrogate's accuracy inside the box says very little about the four kinds, which is the point.
* **hidden regime change**: every candidate is wrong by the same 0.833-0.835 once the fan derates or the fins block --
  the hidden parameter is what moved -- and the monitor catches it in **4.5 (gp), 5.5 (ridge), 5.5 (band) and 6.5
  (gbt) readings**, from the labeled error check: the inputs stay inside the box on both sides of the change (the
  tests assert a KS statistic under 0.20 on every input column), so nothing else has anything to see.
* **out of domain**: the GP is 0.125 and the trees **1.944**, a factor of 15.6, and the ranking follows that column
  (gp #1, gbt #4) rather than the false-alarm column, where the trees are the best of the four (0.0 per 1000 against
  98.8 for the GP). The `envelope` check is what fires, at 0 readings of delay, for every candidate.
* **sensor fault**: 4.0 readings to detect for the GP against 14.75 for the trees; the ambient-probe bias and the
  stuck tachometer are caught by ``input_drift``, the power spikes and the ramping thermocouple by ``error_drift``.
* **slow drift**: the GP detects in 6.5 readings, the ridge in 20.5, and the trees **miss one of the two ramps
  entirely** (``missed`` = 1: the slow fouling never alarms it) -- the column the ranking puts first.
* **false alarms**: 0.0 to 216.0 per 1000 clean readings across candidates and kinds, measured, and published next to
  every delay so a monitor that alarms on everything cannot win the detection column silently.
* **coverage of a stated band**: the ridge's fixed +-2 sigma band covers 95.5 % of the in-domain readings and **1.0 %**
  after the hidden regime change -- the price of a band that does not know it left the box.
* ``thermocouple_drift`` is the case that separates the two questions: the model's error against the plant is
  **0.0000** and its monitor still fires in 3 readings, because the sensor is what failed.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SCHEMA", "RESULTS", "PAGE", "KINDS", "PROTOCOL", "REQUIRED_PROTOCOL_FIELDS", "CASE_DECLARATIONS",
    "Candidate", "Case", "Readings", "SuiteResult", "plant_response", "in_domain_data", "build_case",
    "default_candidates", "run_suite", "leaderboard", "ranked", "markdown", "write", "check",
    "resolve_protocol",
]

SCHEMA = "pinneapple.trust_benchmark/1"
RESULTS = Path("benchmarks") / "trust" / "results.json"
PAGE = Path("docs") / "trust_benchmark.md"

#: The four failure kinds of the issue, in the order the record and the page carry them.
KINDS = ("regime_change", "sensor_fault", "slow_drift", "out_of_domain")
KIND_TITLES = {"regime_change": "hidden regime change", "sensor_fault": "sensor fault",
               "slow_drift": "slow drift", "out_of_domain": "out of domain"}

#: The in-domain box: what the candidates are fitted on and what the monitor's baseline is drawn from.
BOX = {"power_w": (10.0, 70.0), "ambient_c": (15.0, 35.0), "fan_rpm": (900.0, 2400.0)}
INPUT_NAMES = ("power_w", "ambient_c", "fan_rpm")

#: The fan curve: the convection coefficient the array sees at a fan speed. Below 900 rpm the fins are
#: convection-limited and the response is steep; the exponent is a declared engineering correlation, not a cited one,
#: and it is the only piece of the plant that is not the closed form of a textbook.
FAN_H_REF = 18.0
FAN_RPM_REF = 2000.0
FAN_EXPONENT = 0.8

PROTOCOL: dict[str, Any] = {
    "box": BOX,
    "inputs": list(INPUT_NAMES),
    "samples": {"train": 240, "validation": 200, "calibration": 400},
    "split_seed": 0,
    "stream": {"n_clean": 100, "n_after": 100, "seed": 7},
    "error": {"normalisation": "absolute error / std of the in-domain training responses", "tolerance": 0.25},
    "fits": {"budget_s": 10.0, "seed": 0, "threads": 1,
             "candidates": ("ridge", "gp", "gbt"), "band_candidate": "ridge_band"},
    "ranking": {
        "key": "(missed, error_after, detect_delay, false_alarms_per_1000, id)",
        "resolution": {"error_after": 0.05, "false_alarms_per_1000": 0.10},
        "why": "the four questions a trust benchmark asks, in this order: did the monitor ever notice (a model that "
               "never notices is the failure this suite exists to find, so a missed case outranks every other "
               "column); was it wrong after the change (the truth column -- a model that notices in four readings and "
               "is fifteen times further from the plant is not the more trustworthy one); how many readings did it "
               "take to notice (the headline metric of the issue); how often does it cry wolf (a monitor that alarms "
               "on everything detects everything, and this is the column that says so). The error column is compared "
               "with the resolution below, and so is the false-alarm column, so that a kind where every model misses "
               "the change by the same amount is decided by the delay instead of by the last digit",
        "placement": "the mean of a candidate's rank over the four kinds, lowest first; ties go to the smaller mean "
                     "error after the change, then to the candidate id",
    },
    "monitor": {
        "window": 20, "quantile": 0.99, "calibration_windows": 200, "envelope_level": 0.01,
        "alarm": "DriftReport.has_alarm: any check of pinneapple_analysis.trust.drift that contradicted",
        "labels": "the measured response, as a backtest: the monitor sees the response sensor of the same window",
    },
}
REQUIRED_PROTOCOL_FIELDS = ("box", "samples", "split_seed", "stream", "monitor", "error", "fits", "ranking")


# ----------------------------------------------------------------------------- the plant (exact, closed form)
def _h_convection(fan_rpm: np.ndarray, eta_fan: np.ndarray) -> np.ndarray:
    """The convection coefficient the fin array sees: the fan curve, times the fan's own efficiency."""
    return FAN_H_REF * (np.asarray(fan_rpm, dtype=np.float64) / FAN_RPM_REF) ** FAN_EXPONENT * np.asarray(eta_fan)


def plant_response(power_w: Any, ambient_c: Any, fan_rpm: Any, *, eta_fan: Any = 1.0,
                   n_fins: Any = 12.0, k_material: float = 205.0) -> np.ndarray:
    """The base temperature of the fin array [degC]: the closed form of
    :mod:`pinneapple_physics.closed_form.fin_array_conduction`, one call per reading.

    ``eta_fan`` (the fan derating), ``n_fins`` (how many fins the array still has) and ``k_material`` are the plant's
    *hidden* parameters: they change the response and are not among the inputs, which is what makes a regime change
    invisible to anything that only reads the inputs. Every one of them is a real parameter of the same cited model
    (Incropera & DeWitt, fin efficiency ``tanh(m Lc) / (m Lc)`` on a corrected length), not a synthetic knob.
    """
    from pinneapple_physics.closed_form.fin_array_conduction import (
        fin_array_base_temperature_c,
        fin_array_profile_params,
    )

    power = np.atleast_1d(np.asarray(power_w, dtype=np.float64))
    ambient = np.broadcast_to(np.asarray(ambient_c, dtype=np.float64), power.shape)
    rpm = np.broadcast_to(np.asarray(fan_rpm, dtype=np.float64), power.shape)
    eta = np.broadcast_to(np.asarray(eta_fan, dtype=np.float64), power.shape)
    fins = np.broadcast_to(np.asarray(n_fins, dtype=np.float64), power.shape)
    h = _h_convection(rpm, eta)
    out = np.empty(len(power))
    for i in range(len(power)):
        params = fin_array_profile_params(
            k_material_w_mk=k_material, h_conv_w_m2k=float(h[i]), n_fins=float(fins[i]),
            fin_thickness_m=2e-3, fin_length_m=0.03, transverse_extent_m=0.09, spacing_extent_m=0.09,
            base_thickness_m=4e-3, power_w=float(power[i]))
        out[i] = fin_array_base_temperature_c(params, float(ambient[i]), float(power[i]))
    return out


def in_domain_data(protocol: Mapping[str, Any] | None = None, *, seed: Any = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)`` in the box: what the candidates are fitted on, what the monitor's baseline and limit come from.

    Three splits, drawn once from the same generator and filtered by name, because the monitor needs them apart: the
    training split for the candidates, a **validation** split for the monitor's baseline, and a second **calibration**
    split for the null distribution of every check (comparing a split with itself would give a null of near-zero
    statistics and a monitor that alarms on any difference at all).
    """
    p = resolve_protocol(protocol)
    rng = np.random.default_rng(_seed_of(p["split_seed"], seed))
    n = int(p["samples"]["train"] + p["samples"]["validation"] + p["samples"]["calibration"])
    box = p["box"]
    power = rng.uniform(*box["power_w"], n)
    ambient = rng.uniform(*box["ambient_c"], n)
    rpm = rng.uniform(*box["fan_rpm"], n)
    X = np.column_stack([power, ambient, rpm])
    return X, plant_response(power, ambient, rpm)


def _seed_of(*parts: Any) -> int:
    """A generator seed from the protocol's numbers: the same declare numbers always draw the same stream."""
    import zlib
    return int(zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) % (2 ** 31 - 1))


# ----------------------------------------------------------------------------- the injected streams
@dataclass(frozen=True)
class Readings:
    """One stream of readings: what the plant really did, and what the model and the monitor are handed.

    ``x`` is the *measured* input (a sensor may be lying), ``y`` the plant's response -- the reference -- and
    ``measured`` the response sensor: what a monitor with labels reads, which is the truth unless the response sensor
    is the thing that failed. ``changed`` is the ground truth the issue asks for: the reading at which the fault, the
    regime change or the departure from the box begins.
    """
    x: np.ndarray
    y: np.ndarray
    measured: np.ndarray
    changed: int

    def __len__(self) -> int:
        return int(len(self.y))


@dataclass(frozen=True)
class Case:
    """One injected stream: its kind, the ground truth, and the function that builds it."""
    id: str
    kind: str
    title: str
    description: str
    build: Callable[[int, int, int], Readings]

    def as_dict(self, *, n_clean: int, n_after: int) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "description": self.description,
                "changed": int(n_clean), "readings": int(n_clean + n_after),
                "ground_truth": f"the failure begins at reading {n_clean} of {n_clean + n_after}"}


def _in_domain_readings(n_clean: int, n_after: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The clean stream both halves of every case start from: i.i.d. points from the box, and the plant's response."""
    rng = np.random.default_rng(seed)
    power = rng.uniform(*BOX["power_w"], n_clean + n_after)
    ambient = rng.uniform(*BOX["ambient_c"], n_clean + n_after)
    rpm = rng.uniform(*BOX["fan_rpm"], n_clean + n_after)
    X = np.column_stack([power, ambient, rpm])
    return X, plant_response(power, ambient, rpm)


def _regime_change(fan: float | None = None, fins: float | None = None) -> Callable[[int, int, int], Readings]:
    """A hidden parameter steps at the change: the inputs are the same draw from the same box either side of it."""

    def build(n_clean: int, n_after: int, seed: int) -> Readings:
        X, y = _in_domain_readings(n_clean, n_after, seed)
        after = X[n_clean:]
        y = np.asarray(y, dtype=np.float64)
        kwargs = {k: v for k, v in (("eta_fan", fan), ("n_fins", fins)) if v is not None}
        y[n_clean:] = plant_response(after[:, 0], after[:, 1], after[:, 2], **kwargs)
        return Readings(x=X, y=y, measured=y.copy(), changed=n_clean)
    return build


def _sensor_fault(fault: str, where: str) -> Callable[[int, int, int], Readings]:
    """One sensor fault, on the input the model is fed (``where="input"``) or on the response sensor (``"response"``).

    The four faults the issue names -- a spike, a stuck value, a bias and a slow ramp -- and all of them are what a
    real sensor does: the spike on 1 reading in 5, the stuck value on the fan tachometer, a bias on the ambient
    probe, a ramp on the thermocouple. A fault on the **response** leaves the plant and the model's inputs alone, so
    the model stays right and only the labels the monitor checks against are wrong -- the case that separates "the
    model stopped being valid" from "the loop is lying".
    """

    def build(n_clean: int, n_after: int, seed: int) -> Readings:
        X, y = _in_domain_readings(n_clean, n_after, seed)
        x = np.array(X, dtype=np.float64, copy=True)
        y = np.asarray(y, dtype=np.float64)
        measured = y.copy()
        after = np.arange(n_clean, len(y))
        if where == "input":
            if fault == "spike":
                hit = after[(after - n_clean) % 5 == 0]
                x[hit, 0] *= 1.4
            elif fault == "stuck":
                x[after, 2] = x[n_clean - 1, 2]
            elif fault == "bias":
                x[after, 1] += 3.0
            elif fault == "drift":
                x[after, 1] += 0.05 * (after - n_clean)
            else:                                                     # pragma: no cover - guarded by the table
                raise ValueError(f"unknown input fault {fault!r}")
        else:
            if fault == "spike":
                hit = after[(after - n_clean) % 5 == 0]
                measured[hit] += 25.0
            elif fault == "stuck":
                measured[after] = measured[n_clean - 1]
            elif fault == "bias":
                measured[after] += 4.0
            elif fault == "drift":
                measured[after] += 0.4 * (after - n_clean)
            else:                                                     # pragma: no cover
                raise ValueError(f"unknown response fault {fault!r}")
        return Readings(x=x, y=y, measured=measured, changed=n_clean)
    return build


def _slow_drift(drop: float) -> Callable[[int, int, int], Readings]:
    """A hidden parameter ramps at a constant rate from the change to the end: dust on the fan, not a step."""

    def build(n_clean: int, n_after: int, seed: int) -> Readings:
        X, y = _in_domain_readings(n_clean, n_after, seed)
        y = np.asarray(y, dtype=np.float64)
        after = X[n_clean:]
        weight = np.linspace(0.0, 1.0, len(after))
        eta = 1.0 - drop * weight
        y[n_clean:] = plant_response(after[:, 0], after[:, 1], after[:, 2], eta_fan=eta)
        return Readings(x=X, y=y, measured=y.copy(), changed=n_clean)
    return build


def _out_of_domain(fan: tuple[float, float] | None = None,
                   power: tuple[float, float] | None = None) -> Callable[[int, int, int], Readings]:
    """The readings leave the box the candidates were fitted on; the plant itself does not change at all."""

    def build(n_clean: int, n_after: int, seed: int) -> Readings:
        X, y = _in_domain_readings(n_clean, n_after, seed)
        rng = np.random.default_rng(_seed_of(seed, "ood", fan, power))
        x = np.array(X, dtype=np.float64, copy=True)
        if power is not None:
            x[n_clean:, 0] = rng.uniform(*power, n_after)
        if fan is not None:
            x[n_clean:, 2] = rng.uniform(*fan, n_after)
        x[n_clean:, 1] = rng.uniform(*BOX["ambient_c"], n_after)
        y = np.asarray(y, dtype=np.float64)
        y[n_clean:] = plant_response(x[n_clean:, 0], x[n_clean:, 1], x[n_clean:, 2])
        return Readings(x=x, y=y, measured=y.copy(), changed=n_clean)
    return build


#: The nine to ten streams the suite runs, three of them the issue's "regimes no model covers" split in two. The list
#: is data: a case is a declaration with the ground truth of its change, so a reader can check that the injection did
#: what the row says it did.
CASE_DECLARATIONS: tuple[Case, ...] = (
    Case("fan_derate", "regime_change", "the fan derates", "eta_fan 1.00 -> 0.70: the same inputs, 30 % less "
         "convection, invisible in the inputs", _regime_change(fan=0.70)),
    Case("fins_blocked", "regime_change", "fins are blocked", "n_fins 12 -> 8: a third of the array stops "
         "convecting, and the inputs are unchanged", _regime_change(fins=8.0)),
    Case("power_spike", "sensor_fault", "spikes on the power meter", "1 reading in 5 reads 40 % high, from the "
         "change onwards", _sensor_fault("spike", "input")),
    Case("fan_stuck", "sensor_fault", "the fan tachometer sticks", "the rpm input freezes at its value on the change",
         _sensor_fault("stuck", "input")),
    Case("ambient_bias", "sensor_fault", "a bias on the ambient probe", "the ambient input reads 3 K high from the "
         "change onwards", _sensor_fault("bias", "input")),
    Case("thermocouple_drift", "sensor_fault", "the thermocouple ramps", "the response sensor ramps 0.4 K per "
         "reading: the plant and the model are fine, the labels are not", _sensor_fault("drift", "response")),
    Case("dust_ramp", "slow_drift", "dust on the fan", "eta_fan falls 1.00 -> 0.68 linearly over the tail",
         _slow_drift(0.32)),
    Case("slow_fouling", "slow_drift", "slow fouling", "eta_fan falls 1.00 -> 0.90 over the tail: a ramp that "
         "may not cross the tolerance at all", _slow_drift(0.10)),
    Case("fan_below_range", "out_of_domain", "the fan below its range", "rpm drawn from 480-700, under the 900 the "
         "candidates were fitted on", _out_of_domain(fan=(480.0, 700.0))),
    Case("power_above_range", "out_of_domain", "the power above its range", "power drawn from 90-140 W, above the 70 "
         "the candidates were fitted on", _out_of_domain(power=(90.0, 140.0))),
)
BY_ID = {c.id: c for c in CASE_DECLARATIONS}


def build_case(case_id: str, *, protocol: Mapping[str, Any] | None = None) -> Readings:
    """The readings of one declared case, deterministic in the protocol's seeds."""
    if case_id not in BY_ID:
        raise KeyError(f"unknown case {case_id!r}; declared: {sorted(BY_ID)}")
    p = resolve_protocol(protocol)
    return BY_ID[case_id].build(int(p["stream"]["n_clean"]), int(p["stream"]["n_after"]),
                                _seed_of(p["stream"]["seed"], case_id))


# ----------------------------------------------------------------------------- the candidates
@dataclass(frozen=True)
class Candidate:
    """A model, as the suite sees it: an id and a callable on the protocol's inputs.

    ``predict(X)`` takes an ``(n, 3)`` array of ``(power_w, ambient_c, fan_rpm)`` and returns ``(n,)`` -- or
    ``(mean, std)``, in which case the monitor gets an uncertainty to check and the record gets a coverage column.
    Nothing else is asked: the candidate is fitted wherever it was fitted, and the suite only reads it.
    """
    id: str
    title: str
    predict: Callable[[np.ndarray], Any]
    description: str = ""
    family: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "description": self.description, "family": self.family,
                "reports_uncertainty": bool(getattr(self.predict, "reports_std", False))}


def _predict(candidate: Candidate, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """``(mean, std or None)`` from a candidate's callable: a tuple is read as (mean, std), as ``drift.py`` does."""
    out = candidate.predict(np.asarray(X, dtype=np.float64))
    if isinstance(out, tuple) and len(out) == 2:
        mean = np.asarray(out[0], dtype=np.float64).reshape(len(X), -1)
        std = np.asarray(out[1], dtype=np.float64).reshape(len(X), -1)
        return mean, np.abs(std).mean(axis=1)
    arr = np.asarray(out, dtype=np.float64)
    return (arr.reshape(len(X), -1) if arr.ndim > 1 else arr.reshape(-1, 1)), None


def _wrapped(method: str, fitted: Any) -> Callable[[np.ndarray], np.ndarray]:
    def predict(X: np.ndarray) -> np.ndarray:
        return np.asarray(fitted.predict(np.asarray(X, dtype=np.float64))).reshape(len(X), -1)
    predict.description = f"{method} from pinneapple_analysis.surrogate.methods"      # type: ignore[attr-defined]
    predict.reports_std = False                                                        # type: ignore[attr-defined]
    return predict


def default_candidates(protocol: Mapping[str, Any] | None = None, *,
                       methods: Sequence[str] | None = None) -> list[Candidate]:
    """The published candidate set: the one-shot fits of ``surrogate.methods`` on the protocol's in-domain sample.

    Every candidate sees the same training split, the same normalisation and the same seed -- equal data, as the
    regime arena does -- and the fits are one-shot (ridge, GP, gradient-boosted trees), so the record reproduces bit
    for bit: no time-based budget is involved and no number here depends on how loaded the machine was.

    ``ridge_band`` is the ridge with a **constant** band of twice the validation residuals' standard deviation. It is
    not a calibrated interval and does not claim to be one: it is there because a coverage column with no candidate
    reporting uncertainty would measure nothing, and a fixed band is the honest baseline that a calibrated interval
    has to beat.
    """
    from threadpoolctl import threadpool_limits

    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_analysis.surrogate.methods import TrainContext, fit_candidate

    p = resolve_protocol(protocol)
    methods = tuple(p["fits"]["candidates"]) if methods is None else tuple(methods)
    X, y = in_domain_data(p)
    n_train, n_val = int(p["samples"]["train"]), int(p["samples"]["validation"])
    tr, va = slice(0, n_train), slice(n_train, n_train + n_val)
    ctx = TrainContext(X[tr], y[tr].reshape(-1, 1), X[va], y[va].reshape(-1, 1),
                       ComputeBudget(wall_s=float(p["fits"]["budget_s"])), int(p["fits"]["seed"]))
    out: list[Candidate] = []
    with threadpool_limits(limits=int(p["fits"]["threads"])):
        for method in methods:
            fitted, _ = fit_candidate(method, ctx)
            out.append(Candidate(id=method, title=method, family=method, predict=_wrapped(method, fitted),
                                 description=f"fitted on {n_train} in-domain samples, {fitted.n_params} "
                                             "parameters, one shot"))
            if method == "ridge":
                band = float(np.std(np.asarray(fitted.predict(X[va])).ravel() - y[va]))
                out.append(Candidate(
                    id=p["fits"]["band_candidate"], title="ridge + constant band", family="ridge",
                    predict=_banded(_wrapped(method, fitted), band),
                    description=f"ridge with a fixed +-2 sigma band of {band:.3g} K: the standard deviation of its "
                                "validation residuals, not a calibrated interval"))
    return out


def _banded(inner: Callable[[np.ndarray], np.ndarray], band: float) -> Callable[[np.ndarray], tuple]:
    def predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = inner(np.asarray(X, dtype=np.float64))
        return mean, np.full_like(mean, float(band))
    predict.reports_std = True                                                         # type: ignore[attr-defined]
    predict.description = f"constant band {band:.3g} K"                                # type: ignore[attr-defined]
    return predict


# ----------------------------------------------------------------------------- the protocol and the measurement
def resolve_protocol(protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``PROTOCOL`` with the caller's overrides merged in, one level deep: what a run actually did, declared.

    The override exists for the tests and for a cheap look (a short stream, a smaller window). A run whose protocol
    differs from ``PROTOCOL`` is reported as such by :func:`markdown`, and ``check`` refuses to publish a record whose
    protocol is not the declared one -- so a reduced run can never replace the published numbers by accident.
    """
    if not protocol:
        return _deep_copy(PROTOCOL)
    merged = _deep_copy(PROTOCOL)
    for key, value in protocol.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key].update(_deep_copy(dict(value)))
        else:
            merged[key] = _deep_copy(value)
    for f in REQUIRED_PROTOCOL_FIELDS:
        if f not in merged:
            raise ValueError(f"the protocol does not declare {f!r}")
    if merged["monitor"]["window"] < 10:
        raise ValueError("a monitor window under 10 readings is not comparable: drift.py refuses it")
    return merged


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_deep_copy(v) for v in value)
    return value


def _callable(candidate: Candidate) -> Callable[[np.ndarray], Any]:
    """The candidate as ``DriftMonitor`` wants it: ``(mean, std)`` when it has a spread, the mean alone otherwise.

    ``drift.py`` reads *any* two-tuple as (mean, std), so handing it ``(mean, None)`` would turn "this model reports
    no uncertainty" into a one-point uncertainty column and a monitor checking a constant -- the exact silent
    substitution that module's ``NOT_RUN`` check exists to avoid.
    """
    def model(Z: np.ndarray) -> Any:
        mean, std = _predict(candidate, np.asarray(Z, dtype=np.float64))
        return (mean, std) if std is not None else mean
    return model


def _monitor(candidate: Candidate, X: np.ndarray, y: np.ndarray, p: Mapping[str, Any]) -> Any:
    """The protocol's monitor for one candidate: baseline on the validation split, limits on the calibration split."""
    from pinneapple_analysis.trust.drift import DriftMonitor

    mon = p["monitor"]
    model = _callable(candidate)
    n_train, n_val = int(p["samples"]["train"]), int(p["samples"]["validation"])
    baseline = DriftMonitor.from_validation(
        model, X[n_train:n_train + n_val], y[n_train:n_train + n_val],
        envelope_level=float(mon["envelope_level"]), quantile=float(mon["quantile"]))
    baseline.calibrate(model, X[n_train + n_val:], y[n_train + n_val:], window=int(mon["window"]),
                       windows=int(mon["calibration_windows"]))
    return baseline


def _score_case(candidate: Candidate, monitor: Any, readings: Readings, p: Mapping[str, Any],
                scale: float) -> dict[str, Any]:
    """One candidate on one stream: the error curve, the monitor's verdicts, and where each of them moved.

    The model is asked once per reading, up front, and the monitor is then fed the window's predictions through
    ``DriftMonitor.evaluate`` -- the entry point that module documents for a replay. Asking the model again inside
    every window would measure the model twenty times per reading and charge none of it to anything.
    """
    window, changed, n = int(p["monitor"]["window"]), int(readings.changed), len(readings)
    mean, std = _predict(candidate, readings.x)
    error = np.abs(mean[:, 0] - readings.y) / scale
    errors = np.abs(readings.measured - mean[:, 0]).reshape(-1, 1)
    covered = None if std is None else (np.abs(mean[:, 0] - readings.y) <= 2.0 * std)
    alarms: list[bool] = [False] * n
    checks: dict[str, int] = {}
    verdicts: list[str] = ["not_checked"] * n
    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        report = monitor.evaluate(readings.x[sl], mean[sl], uncertainty=None if std is None else std[sl],
                                  errors=errors[sl])
        verdicts[i] = report.decision.verdict
        alarms[i] = report.has_alarm
        for name in report.alarms:
            checks[name] = checks.get(name, 0) + 1
    checked = np.array([i >= window - 1 for i in range(n)])
    clean = checked & (np.arange(n) < changed)
    post = np.arange(n) >= changed
    detected_at = next((i for i in range(changed, n) if alarms[i]), None)
    exit_at = next((i for i in range(changed, n) if error[i] > float(p["error"]["tolerance"])), None)
    return {
        "n_clean_checked": int(clean.sum()),
        "false_alarms": int(sum(1 for i in range(n) if clean[i] and alarms[i])),
        "detected_at": detected_at, "detect_delay": None if detected_at is None else int(detected_at - changed),
        "error_exit_at": exit_at, "error_exit_delay": None if exit_at is None else int(exit_at - changed),
        "error_before": float(np.mean(error[~post])) if (~post).any() else None,
        "error_after": float(np.mean(error[post])) if post.any() else None,
        "error_after_max": float(np.max(error[post])) if post.any() else None,
        "coverage_before": None if covered is None else float(np.mean(covered[~post])),
        "coverage_after": None if covered is None else float(np.mean(covered[post])),
        "alarmed_checks": dict(sorted(checks.items())),
        "error_curve": [round(float(v), 6) for v in error],
        "alarm_readings": [int(i) for i in range(n) if alarms[i]],
        "verdicts_after": {v: int(sum(1 for i in range(n) if post[i] and verdicts[i] == v))
                           for v in ("STABLE", "REVIEW", "RETRAIN") if any(
                               post[i] and verdicts[i] == v for i in range(n))},
    }


def _aggregate(kind: str, rows: Mapping[str, dict[str, Any]], p: Mapping[str, Any]) -> dict[str, Any]:
    """The kind's row: the cases' measurements added up the way the ranking key needs them.

    Delays are averaged over the kind's cases with a **miss censored at the readings that remained**, and ``missed``
    says how many were censored, because the average of the cases a monitor noticed is the number that flatters it.
    False alarms are summed over the clean readings and divided once, which is a rate and not a mean of rates.
    """
    n_after = int(p["stream"]["n_after"])
    missed = [cid for cid, r in rows.items() if r["detected_at"] is None]
    delays = [n_after if r["detect_delay"] is None else r["detect_delay"] for r in rows.values()]
    exits = [n_after if r["error_exit_delay"] is None else r["error_exit_delay"] for r in rows.values()]
    fa = sum(r["false_alarms"] for r in rows.values())
    clean = sum(r["n_clean_checked"] for r in rows.values())
    cov_before = [r["coverage_before"] for r in rows.values() if r["coverage_before"] is not None]
    cov_after = [r["coverage_after"] for r in rows.values() if r["coverage_after"] is not None]
    checks: dict[str, int] = {}
    for r in rows.values():
        for name, count in r["alarmed_checks"].items():
            checks[name] = checks.get(name, 0) + count
    return {
        "kind": kind, "kind_title": KIND_TITLES[kind], "cases": len(rows), "case_ids": list(rows),
        "missed": len(missed), "missed_cases": missed, "detected": f"{len(rows) - len(missed)}/{len(rows)}",
        "detect_delay": float(np.mean(delays)), "detect_delay_max": float(np.max(delays)),
        "false_alarms": int(fa), "clean_readings": int(clean),
        "false_alarms_per_1000": float(1000.0 * fa / clean) if clean else None,
        "error_before": float(np.mean([r["error_before"] for r in rows.values()
                                       if r["error_before"] is not None])),
        "error_after": float(np.mean([r["error_after"] for r in rows.values() if r["error_after"] is not None])),
        "error_after_max": float(np.max([r["error_after_max"] for r in rows.values()
                                         if r["error_after_max"] is not None])),
        "error_exit_delay": float(np.mean(exits)),
        "coverage_before": float(np.mean(cov_before)) if cov_before else None,
        "coverage_after": float(np.mean(cov_after)) if cov_after else None,
        "alarmed_checks": dict(sorted(checks.items())),
    }


def _compare(a: Mapping[str, Any], b: Mapping[str, Any], resolution: Mapping[str, Any]) -> int:
    """The declared order, column by column, with a resolution: a difference under it is not a difference.

    Without a resolution the order is decided by whatever the last digit says. That is not a hypothetical: on the
    regime-change kind every candidate's error after the change lands within 1 % of every other's (the hidden parameter
    is what moved, and every model misses it by the same amount), so a strict order would rank a model that detected
    four readings late above one that detected in five because of 7e-3 of error -- and the published table would be
    reporting noise as a distinction. With the resolution, that kind is decided by the delay, which is what actually
    differed, and a difference as large as the resolution (the out-of-domain kind, where the errors are 0.125 and 1.944)
    is still decided by the error.
    """
    for key, res in (("missed", 0.0), ("error_after", resolution["error_after"]),
                     ("detect_delay", 0.0), ("false_alarms_per_1000", resolution["false_alarms_per_1000"])):
        left, right = a[key], b[key]
        if left is None or right is None:
            if left == right:
                continue
            return -1 if (left is None or right is not None) else 1
        if res and abs(left - right) <= res * max(abs(left), abs(right), 1e-12):
            continue
        if left != right:
            return -1 if left < right else 1
    return -1 if a["id"] < b["id"] else (0 if a["id"] == b["id"] else 1)


def ranked(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The rows in the declared order, from a fixed starting point so the order is a function of the rows alone.

    The rows are first put in id order and then sorted with the declared comparison: a sort that starts from the
    published order would only be reproducible for a comparator that is consistent everywhere, and this one compares
    with a resolution, which is a comparison a reader can recompute but not one worth relying on for stability.
    """
    from functools import cmp_to_key

    resolution = resolve_protocol(protocol)["ranking"]["resolution"]
    base = sorted(rows, key=lambda r: r["id"])
    return sorted(base, key=cmp_to_key(lambda a, b: _compare(a, b, resolution)))


def leaderboard(kinds: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The placement: the mean of a candidate's rank over the kinds, and every rank that average came from.

    Sorted by that mean, then by the mean error after the change, then by id -- a total order, so a tie is still a
    decision and a published table can be checked against the measurements instead of trusted.
    """
    ranks: dict[str, dict[str, int]] = {}
    titles: dict[str, str] = {}
    for rows in kinds.values():
        for row in rows:
            ranks.setdefault(row["id"], {})[row["kind"]] = row["rank"]
            titles[row["id"]] = row["title"]
    out = []
    for cid, by_kind in ranks.items():
        values = list(by_kind.values())
        errors = [float(next(r["error_after"] for r in rows if r["id"] == cid)) for rows in kinds.values()]
        worst = max(by_kind, key=lambda k: (by_kind[k], k))
        # a candidate that holds the same rank on every kind has no worst kind, and naming one would be reading a
        # difference into the alphabetical order of the keys
        flat = len(set(values)) == 1
        out.append({"id": cid, "title": titles[cid], "mean_rank": round(sum(values) / len(values), 6),
                    "kinds_ranked": len(values), "ranks": by_kind,
                    "worst_kind": None if flat else worst, "worst_rank": by_kind[worst],
                    "mean_error_after": round(sum(errors) / len(errors), 6),
                    "same_rank_on_every_kind": flat})
    out.sort(key=lambda r: (r["mean_rank"], r["mean_error_after"], r["id"]))
    for position, row in enumerate(out, start=1):
        row["rank"] = position
    return out


@dataclass
class SuiteResult:
    """One run of the suite: the protocol it ran under, the streams' ground truth, and the rows."""
    protocol: dict[str, Any]
    candidates: list[dict[str, Any]]
    ground_truth: dict[str, Any]
    scale_k: float
    cases: dict[str, dict[str, Any]]
    kinds: dict[str, list[dict[str, Any]]]
    leaderboard: list[dict[str, Any]]
    monitors: dict[str, Any]
    created: str
    wall_s: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The record, as a copy: a suite result is read many times and handed to ``write`` and ``check`` once each."""
        import copy
        return copy.deepcopy({"schema": SCHEMA, "protocol": self.protocol, "candidates": self.candidates,
                              "ground_truth": self.ground_truth, "scale_k": self.scale_k, "cases": self.cases,
                              "kinds": self.kinds, "leaderboard": self.leaderboard, "monitors": self.monitors,
                              "created": self.created, "wall_s": self.wall_s, "notes": self.notes})

    def to_markdown(self, title: str = "") -> str:
        return markdown(self.to_dict(), title)


def run_suite(candidates: Sequence[Candidate], *, protocol: Mapping[str, Any] | None = None,
              kinds: Sequence[str] | None = None, only: Sequence[str] | None = None,
              log: Callable[[str], None] | None = None) -> SuiteResult:
    """One API: every candidate, every failure kind, the same monitor, one record.

    ``protocol`` overrides the declared numbers one level deep (a short stream for a test); ``kinds`` selects kinds;
    ``only`` selects case ids. The in-domain sample, the streams and the monitor's limits are all deterministic in the
    protocol's seeds, so two runs of the same declaration produce the same record.
    """
    started = time.perf_counter()
    log = log or (lambda s: None)
    p = resolve_protocol(protocol)
    wanted = list(KINDS) if kinds is None else list(kinds)
    unknown = [k for k in wanted if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown failure kind(s) {unknown}; declared: {list(KINDS)}")
    case_ids = [c.id for c in CASE_DECLARATIONS if c.kind in wanted and (only is None or c.id in set(only))]
    if not case_ids:
        raise ValueError(f"no case of {wanted!r} is declared{' and selected' if only else ''}")
    X, y = in_domain_data(p)
    scale = float(np.std(y[:int(p["samples"]["train"])]))
    rows: dict[str, dict[str, dict[str, Any]]] = {cid: {} for cid in case_ids}
    monitors: dict[str, Any] = {}
    for candidate in candidates:
        print("CAND", candidate.id, flush=True)
        log(f"{candidate.id}: monitor")
        mon = _monitor(candidate, X, y, p)
        thresholds = mon.thresholds.to_dict()
        monitors[candidate.id] = {"thresholds": thresholds,
                                  "baseline": "the validation split, predicted by this candidate",
                                  "calibration": "the second held-out split, 99 % quantile of "
                                                 f"{thresholds['windows']} windows of {thresholds['calibrated_window']}"}
        for cid in case_ids:
            readings = build_case(cid, protocol=p)
            log(f"{candidate.id}: {cid}")
            rows[cid][candidate.id] = _score_case(candidate, mon, readings, p, scale)
    kinds_out: dict[str, list[dict[str, Any]]] = {}
    for kind in wanted:
        ids = [cid for cid in case_ids if BY_ID[cid].kind == kind]
        if not ids:
            continue
        table = []
        for candidate in candidates:
            row = {"id": candidate.id, "title": candidate.title, **_aggregate(kind, {c: rows[c][candidate.id]
                                                                                   for c in ids}, p)}
            table.append(row)
        table = ranked(table, p)
        for position, row in enumerate(table, start=1):
            row["rank"] = position
        kinds_out[kind] = table
    ground_truth = {cid: BY_ID[cid].as_dict(n_clean=int(p["stream"]["n_clean"]),
                                            n_after=int(p["stream"]["n_after"])) for cid in case_ids}
    notes = []
    declared = p == _deep_copy(PROTOCOL)
    if not declared:
        notes.append("this run used a protocol that is not the declared one (a reduced stream or window): its record "
                     "is a look, not the published benchmark")
    return SuiteResult(
        protocol=_declared_protocol(p), candidates=[c.as_dict() for c in candidates], ground_truth=ground_truth,
        scale_k=scale, cases=rows, kinds=kinds_out, leaderboard=leaderboard(kinds_out), monitors=monitors,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        wall_s=time.perf_counter() - started, notes=notes)


def _declared_protocol(p: Mapping[str, Any]) -> dict[str, Any]:
    """The protocol as JSON: tuples become lists, so a record round-trips through a file unchanged."""
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value
    out = clean(p)
    out["declared"] = p == _deep_copy(PROTOCOL)
    return out


# ----------------------------------------------------------------------------- the page
def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _record(results: Mapping[str, Any] | SuiteResult) -> Mapping[str, Any]:
    """The functions below take a record (a dict) or the :class:`SuiteResult` a run returns."""
    return results.to_dict() if isinstance(results, SuiteResult) else results


def _box_text(p: Mapping[str, Any]) -> str:
    return ", ".join(f"{name} {p['box'][name][0]:g}-{p['box'][name][1]:g}" for name in p["inputs"])


def table(results: Mapping[str, Any] | SuiteResult) -> str:
    """The leaderboard: the placement, and the four ranks it is the mean of."""
    results = _record(results)
    present = [k for k in KINDS if k in results["kinds"]]
    head = ("| rank | candidate | mean rank | " + " | ".join(k.replace("_", " ") for k in present) +
            " | worst kind | error after the change |")
    lines = [head, "|---" * (len(present) + 4) + "|"]
    for row in results["leaderboard"]:
        cells = [f"#{row['ranks'][k]}" if k in row["ranks"] else "—" for k in present]
        worst = ("the same rank on every kind" if row.get("worst_kind") is None else
                 f"{row['worst_kind'].replace('_', ' ')} (#{row['worst_rank']})")
        lines.append(f"| {row['rank']} | {row['title']} | {_fmt(row['mean_rank'])} | " + " | ".join(cells) +
                     f" | {worst} | {_fmt(row['mean_error_after'], 3)} |")
    return "\n".join(lines)


def _kind_table(rows: Sequence[Mapping[str, Any]]) -> str:
    head = ("| rank | candidate | detected | readings to detect | false alarms / 1000 | error before | error after "
            "| readings to the error exit | coverage before | coverage after | checks that fired |")
    lines = [head, "|---" * 11 + "|"]
    for row in rows:
        checks = ", ".join(f"{k} ({v})" for k, v in row["alarmed_checks"].items()) or "—"
        lines.append(
            f"| {row['rank']} | {row['title']} | {row['detected']} | {_fmt(row['detect_delay'], 1)} | "
            f"{_fmt(row['false_alarms_per_1000'], 1)} | {_fmt(row['error_before'], 3)} | "
            f"{_fmt(row['error_after'], 3)} | {_fmt(row['error_exit_delay'], 1)} | "
            f"{_fmt(row['coverage_before'])} | {_fmt(row['coverage_after'])} | {checks} |")
    return "\n".join(lines)


def markdown(results: Mapping[str, Any] | SuiteResult, title: str = "") -> str:
    """The published page, generated from the record so the numbers on it cannot drift from the run."""
    results = _record(results)
    p = results["protocol"]
    mon = p["monitor"]
    parts = [
        f"# {title or 'The trust benchmark: four failure kinds, one protocol, one leaderboard'} (#349)",
        "",
        "A leaderboard of in-domain accuracy hides the one failure that matters in deployment: *where* a model stops",
        "being valid. This suite injects the four failures a deployed physics model meets -- a hidden regime change,",
        "a sensor fault, a slow drift, an out-of-domain reading -- with the reading at which each one starts written",
        "down, and measures what the model and its monitor did about it.",
        "",
        "Every table on this page is generated from",
        f"`{RESULTS.as_posix()}` by `python scripts/trust_benchmark.py --docs`. Do not edit it by hand;",
        "`--check` refuses a page that drifted from the record, and a record whose order is not the one the",
        "measurements give.",
        "",
        "## The protocol",
        "",
        "* **the plant**: the base temperature of a fin array,",
        "  `pinneapple_physics.closed_form.fin_array_conduction` (fin efficiency `tanh(m Lc) / (m Lc)`, adiabatic-tip",
        "  corrected length), with a declared fan curve mapping rpm to the convection coefficient. The plant's hidden",
        "  parameters -- the fan's derating, how many fins the array still has -- are not among the inputs.",
        f"* **the inputs**: `{', '.join(p['inputs'])}`, in the box {_box_text(p)}. That box is what the candidates "
        "were",
        "  fitted on, and what the monitor's envelope was fitted on.",
        f"* **the data**: {p['samples']['train']} training, {p['samples']['validation']} validation and",
        f"  {p['samples']['calibration']} calibration readings, drawn once (split seed {p['split_seed']}); every",
        "  candidate sees the same three splits.",
        f"* **the streams**: {p['stream']['n_clean']} clean readings then {p['stream']['n_after']} under the injected",
        f"  failure, per case, seed {p['stream']['seed']}. The change is at reading {p['stream']['n_clean']} of the",
        "  stream, for every case, and the case table below says what changed.",
        "* **the monitor**: `pinneapple_analysis.trust.drift.DriftMonitor` -- input, prediction, uncertainty, error",
        "  and envelope checks -- on a trailing window of",
        f"  {mon['window']} readings, its limits calibrated at the {mon['quantile']:.0%} quantile of",
        f"  {mon['calibration_windows']} genuinely in-distribution windows. The monitor is the protocol's, not the",
        "  candidate's: the candidate changes what the monitor is given, never what the monitor checks.",
        f"* **the alarm**: {mon['alarm']}.",
        f"* **the labels**: {mon['labels']}.",
        f"* **the error**: {p['error']['normalisation']}, with a tolerance of {p['error']['tolerance']} for the exit.",
        f"* **the ranking**: `{p['ranking']['key']}` -- {p['ranking']['why']}.",
        f"* **the resolution**: a difference under {p['ranking']['resolution']['error_after']:.0%} in the error column,"
        f" or under {p['ranking']['resolution']['false_alarms_per_1000']:.0%} in the false-alarm column, is not a "
        "difference: the order falls through to the next column.",
        f"* **the placement**: {p['ranking']['placement']}.",
        "",
        "## The leaderboard",
        "",
        table(results),
        "",
        "The mean rank is the average of the candidate's rank on each of the four failure kinds. A candidate that is",
        "excellent on one kind and blind to another cannot hide the second half of that sentence: the *worst kind*",
        "column names it, and the per-kind tables below carry every column the ranking was computed from.",
        "",
    ]
    for kind in KINDS:
        rows = results["kinds"].get(kind)
        if not rows:
            continue
        first = rows[0]
        parts += [f"## {KIND_TITLES[kind]}", "",
                  ("The cases: " + "; ".join(f"`{cid}`" for cid in first["case_ids"]) + ".") if first else "",
                  "",
                  _kind_table(rows),
                  ""]
    parts += ["## The injected streams and their ground truth", "",
              "| case | kind | the change | ground truth |", "|---|---|---|---|"]
    for cid in [c.id for c in CASE_DECLARATIONS if c.id in results["ground_truth"]]:
        gt = results["ground_truth"][cid]
        parts.append(f"| `{cid}` | {gt['kind'].replace('_', ' ')} | {gt['title']}: {gt['description']} | "
                     f"{gt['ground_truth']} |")
    parts += ["", "The rows above are measured against the plant, not against the model: the response `y` of every",
              "stream is the closed form's, whatever the sensors read. That is what makes the truth column a truth",
              "column -- in `thermocouple_drift` the model is right and the sensor is wrong, and the record says so.",
              "", "## The candidates", ""]
    for c in results["candidates"]:
        parts.append(f"* **{c['id']}** ({c['family']}): {c['description']} "
                     f"{'-- reports an uncertainty' if c['reports_uncertainty'] else ''}".rstrip())
    parts += ["", "## Reproducing it", "", "```bash",
              "python scripts/trust_benchmark.py --run     # fit the candidates, run the suite, write the record and "
              "this page",
              "python scripts/trust_benchmark.py --docs    # regenerate this page from the record (no fits)",
              "python scripts/trust_benchmark.py --check   # the gate: declaration, record and page agree", "```", "",
              "The candidates are one-shot fits (ridge, Gaussian process, gradient-boosted trees) on one BLAS thread,",
              "so the record reproduces: no time-based budget is involved and the same declaration draws the same",
              "streams. Measured: two consecutive runs of the declared protocol produced a byte-identical record,",
              "apart from the timestamp and the wall time. To place a model of your own on the board, fit it on",
              "`in_domain_data()` and pass it to `run_suite` as a `Candidate` -- a callable `predict(X) -> y`,",
              "or `-> (mean, std)`; nothing else is asked of it.", ""]
    return "\n".join(parts)


def write(results: Mapping[str, Any] | SuiteResult, *, root: Path | str = ".") -> tuple[Path, Path]:
    results = _record(results)
    root = Path(root)
    path = root / RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    page = root / PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(markdown(results), encoding="utf-8")
    return path, page


def check(root: Path | str = ".") -> list[str]:
    """The gate: the declaration is complete, the record's order is the measurements', the page matches the record."""
    root = Path(root)
    problems: list[str] = []
    for f in REQUIRED_PROTOCOL_FIELDS:
        if f not in PROTOCOL:
            problems.append(f"the protocol does not declare {f!r}")
    if set(KINDS) != set(KIND_TITLES):
        problems.append(f"a kind of the issue has no title: {set(KINDS) ^ set(KIND_TITLES)}")
    missing = set(KINDS) - {c.kind for c in CASE_DECLARATIONS}
    if missing:
        problems.append(f"no injected case for the kind(s) {sorted(missing)}: the suite would measure nothing there")
    results_path = root / RESULTS
    if not results_path.is_file():
        return problems + [f"{RESULTS.as_posix()} is missing: run python scripts/trust_benchmark.py --run"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    protocol = results.get("protocol", {})
    if results.get("schema") != SCHEMA:
        problems.append(f"schema is {results.get('schema')!r}, expected {SCHEMA!r}")
    if set(results.get("kinds", {})) != set(KINDS):
        problems.append(f"the record covers {sorted(results.get('kinds', {}))}, not the four kinds {list(KINDS)}")
    for kind, rows in results.get("kinds", {}).items():
        derived = ranked(rows, protocol)
        if [r["id"] for r in rows] != [r["id"] for r in derived]:
            problems.append(f"kind {kind}: the published order is not the one the measurements give "
                            f"(published {[r['id'] for r in rows]}, derived {[r['id'] for r in derived]})")
        for position, row in enumerate(rows, start=1):
            if row.get("rank") != position:
                problems.append(f"kind {kind}: {row.get('id')!r} carries rank {row.get('rank')}, published in "
                                f"position {position}")
    if results.get("leaderboard") != leaderboard(results.get("kinds", {})):
        problems.append("the leaderboard is not what the per-kind measurements give")
    if not protocol.get("declared", False):
        problems.append("the record was produced by a reduced protocol, not the declared one")
    page_path = root / PAGE
    if not page_path.is_file():
        problems.append(f"{PAGE.as_posix()} is missing: run python scripts/trust_benchmark.py --docs")
    elif page_path.read_text(encoding="utf-8") != markdown(results):
        problems.append(f"{PAGE.as_posix()} is not what the record generates")
    return problems
