"""An operability window with a measured confidence: a buoy's sea, a vessel's answer, a mooring, and the hours left.

Every number this script prints is measured on something, and the somethings are three:

* **the buoy.** NOAA NDBC station 41010 -- 20 NM east of Cape Canaveral, 30 m of water -- 45 days of ten-minute
  significant height, dominant period and wind, and 2080 measured wave spectra on NDBC's 46-band grid. ``data/
  ndbc_41010.npz`` is that slice, written by ``build_dataset.py``; the example reads it and needs no network;
* **the section.** A 20 m beam, 6 m draught section with the metacentric height of a loaded support vessel (1.5 m),
  solved for its added mass, damping and excitation at twenty-eight periods -- the 2-D free-surface problem, no table
  of coefficients quoted from anywhere;
* **the record the instruments would have written.** The motion model plus the sensors' own noise, with four events
  injected where the physics acts: a slam at the bow, a mooring line that changes, and two instruments that fail in
  two different ways.

What comes out is the chain the issue asks for: wave skill against the buoy, the vessel's answer to the forecast sea,
the mooring tension that follows, the residual that separates a new regime from a broken sensor, and finally the
window -- **work allowed until hour N, confidence X** -- where the confidence is the fraction of hours the forecast's
verdict was measured to be right on data the two calibrated coefficients never saw.

Run::

    python examples/maritime_operability/maritime_operability.py            # everything, about 3 minutes
    python examples/maritime_operability/maritime_operability.py --quick    # fewer leads and figures
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_physics.maritime import (
    confidence,
    mooring,
    motion,
    ndbc,
    operability,
    rao,
    seastate,
)
from pinneapple_physics.maritime import spectrum as sp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
RESULTS = HERE / "results"
G_ACC = sp.G


# ------------------------------------------------------------------------------------------------ the buoy
class Buoy:
    """The record, resampled to what the analysis needs: ten-minute observations and half-hourly spectra."""

    def __init__(self, path: Path) -> None:
        self.record = ndbc.load(path)
        self.seconds = self.record.time.astype("datetime64[s]").astype(np.int64)
        self.spec_time, self.freq, self.density = self.record.spectra()

    def observed(self, when, field: str) -> np.ndarray:
        """The buoy's own value of a field at a set of times, interpolating over the gaps it does have."""
        source = {"hs": self.record.hs, "tp": self.record.tp, "wspd": self.record.wspd,
                  "swell_hs": self.record.swell_hs, "windsea_hs": self.record.windsea_hs}[field]
        times = (self.record.time if field not in ("swell_hs", "windsea_hs") else self.record.split_time)
        times = times.astype("datetime64[s]").astype(np.int64)
        good = np.isfinite(source)
        asked = np.atleast_1d(np.asarray(when))
        if asked.dtype.kind == "M":
            asked = asked.astype("datetime64[s]").astype(np.int64)
        return np.interp(asked.astype(float), times[good], source[good])

    def analyses(self, *, every: int = 4, leads: np.ndarray) -> dict[str, np.ndarray]:
        """Analysis states every ``every`` spectra, with the observations at each lead time in front of them.

        The analysis spectrum is scaled so that its own ``4 sqrt(m0)`` equals the height the buoy reports: the 46
        published bands carry 2 % less energy than NDBC's number, which integrates bands the file does not carry, and
        starting every forecast 2 % low would be a bias that has nothing to do with the model.
        """
        index = np.arange(0, self.spec_time.size, every)
        rows = {"freq": [], "density": [], "wind": [], "hs": [], "tp": [], "time": [],
                "obs_hs": [], "obs_tp": []}
        last = self.seconds[-1]
        for i in index:
            when = int(self.spec_time[i].astype("datetime64[s]").astype(np.int64))
            if when + leads[-1] * 3600 > last:
                continue
            measured = self.density[i]
            height = float(self.observed(when, "hs")[0])
            spectrum = measured * (height / sp.hs_from_spectrum(self.freq, measured)) ** 2
            rows["freq"].append(self.freq)
            rows["density"].append(spectrum)
            rows["wind"].append(float(self.observed(when, "wspd")[0]))
            rows["hs"].append(height)
            rows["tp"].append(float(self.observed(when, "tp")[0]))
            rows["time"].append(when)
            rows["obs_hs"].append(self.observed(when + leads * 3600, "hs"))
            rows["obs_tp"].append(self.observed(when + leads * 3600, "tp"))
        return {k: np.array(v, dtype=float) if k not in ("time",) else np.array(v, dtype=np.int64)
                for k, v in rows.items()}


# ------------------------------------------------------------------------------------------------ the sea state
def wave_skill(buoy: Buoy, leads: np.ndarray, quick: bool) -> dict[str, object]:
    """Calibrate the two timescales on the first half of the record, score them on the second."""
    states = buoy.analyses(every=4, leads=leads)
    split = states["time"] <= int(np.median(states["time"]))
    calibration = {k: v[split] for k, v in states.items()}
    validation = {k: v[~split] for k, v in states.items()}
    model, search = seastate.calibrate(
        calibration["freq"], calibration["density"], calibration["wind"], calibration["obs_hs"],
        calibration["obs_tp"], leads,
        grid_swell=(24.0, 48.0, 96.0, 192.0) if quick else (12.0, 24.0, 48.0, 96.0, 192.0, 384.0),
        grid_growth=(2.0, 4.0, 8.0, 16.0) if quick else (1.0, 2.0, 4.0, 6.0, 8.0, 16.0))

    table = []
    for lead in leads:
        forecasts = [model.forecast(validation["freq"][i], validation["density"][i], validation["wind"][i], leads=[lead])
                     for i in range(validation["freq"].shape[0])]
        predicted_hs = np.array([f.hs[0] for f in forecasts])
        predicted_tp = np.array([f.tp[0] for f in forecasts])
        j = int(np.where(leads == lead)[0][0])
        if j >= validation["obs_hs"].shape[1]:
            continue
        observed_hs, observed_tp = validation["obs_hs"][:, j], validation["obs_tp"][:, j]
        table.append({"lead_hours": float(lead),
                      "hs": seastate.skill(predicted_hs, observed_hs, validation["hs"]),
                      "tp": seastate.skill(predicted_tp, observed_tp, validation["tp"]),
                      "hs_predicted": predicted_hs, "hs_observed": observed_hs,
                      "tp_predicted": predicted_tp, "tp_observed": observed_tp})
    return {"model": model, "table": table, "search": search, "calibration": calibration,
            "validation": validation, "states": states}


def spectrum_agreement(buoy: Buoy, quick: bool) -> dict[str, object]:
    """How well the analytic shape describes the measured spectra, and how well the split matches NDBC's own."""
    picked = np.linspace(0, buoy.spec_time.size - 1, 40 if quick else 120).astype(int)
    errors, heights, pm_errors, periods = [], [], [], []
    for i in picked:
        measured = buoy.density[i]
        if measured.max() <= 0:
            continue
        fitted, rms = sp.fit(buoy.freq, measured)
        _, pm_rms = sp.fit(buoy.freq, measured, gamma_fixed=1.0)
        errors.append(rms)
        pm_errors.append(pm_rms)
        heights.append(sp.hs_from_spectrum(buoy.freq, measured) / max(buoy.observed(buoy.spec_time[i], "hs")[0], 1e-6))
        periods.append(fitted.tp)

    splittings = []
    for i in picked:
        wind = float(buoy.observed(buoy.spec_time[i], "wspd")[0])
        if wind <= 0:
            continue
        swell, wind_sea = seastate.split_swell(buoy.freq, buoy.density[i], wind)
        model = (4.0 * math.sqrt(max(float(np.trapezoid(swell, buoy.freq)), 0.0)),
                 4.0 * math.sqrt(max(float(np.trapezoid(wind_sea, buoy.freq)), 0.0)))
        # NDBC reports its split every half hour on the same grid: the nearest row is the same sea
        row = int(np.argmin(np.abs(buoy.record.split_time.astype("datetime64[s]").astype(np.int64)
                                   - buoy.spec_time[i].astype("datetime64[s]").astype(np.int64))))
        if np.isfinite(buoy.record.swell_hs[row]) and np.isfinite(buoy.record.windsea_hs[row]):
            splittings.append((model[0], buoy.record.swell_hs[row], model[1], buoy.record.windsea_hs[row]))
    splittings = np.array(splittings)
    return {"fit_rms": float(np.median(errors)), "fit_rms_jonswap": float(np.median(errors)),
            "fit_rms_pm": float(np.median(pm_errors)), "hs_ratio": float(np.median(heights)),
            "swell_hs_model": splittings[:, 0], "swell_hs_ndbc": splittings[:, 1],
            "windsea_hs_model": splittings[:, 2], "windsea_hs_ndbc": splittings[:, 3],
            "fit_gamma": float(np.median([sp.fit(buoy.freq, buoy.density[i])[0].gamma
                                          for i in picked[:10] if buoy.density[i].max() > 0])),
            "periods": np.array(periods), "count": len(errors)}


# ------------------------------------------------------------------------------------------------ the vessel
VESSEL = {"length": 40.0, "beam": 20.0, "draught": 6.0, "kg": 7.056, "depth": 30.0, "gyration": 7.0}
CRANE = motion.Point(x=0.0, y=8.0)          # the crane pedestal, 8 m off the centreline: roll is what lifts it
BOW = motion.Point(x=20.0, y=0.0)           # the forward perpendicular, where slamming is counted


def vessel_transfer(quick: bool) -> dict[str, object]:
    """The section's transfer functions, and the checks that say the solve is a solution."""
    section = rao.Section.from_gm(VESSEL["beam"], VESSEL["draught"], 1.5, depth=VESSEL["depth"],
                                  gyration=VESSEL["gyration"])
    periods = np.geomspace(4.0, 25.0, 18 if quick else 28)
    transfer = rao.sweep(section, periods=periods)
    transfer = transfer.with_pitch(rao.pitch_transfer_function(VESSEL["length"], VESSEL["draught"], VESSEL["depth"],
                                                               kg=VESSEL["kg"], periods=periods))
    rows = [rao.solve(section, 2.0 * math.pi / float(period)) for period in periods]
    damping = np.array([r["b"][1, 1] for r in rows])
    added = np.array([r["a"][1, 1] for r in rows])
    long_wave = rao.solve(section, 2.0 * math.pi / 80.0)
    props = rao.section_mass_properties(section)
    return {"section": section, "transfer": transfer, "periods": periods, "rows": rows, "damping": damping,
            "added": added, "long_wave_heave": long_wave["rao_heave"], "long_wave_roll": long_wave["rao_roll"],
            "properties": props, "cells": max(r["cells"] for r in rows)}


def inertia_ratio(vessel: dict) -> float:
    """``A44 / I44`` at the roll resonance: how much of the roll inertia the water adds, which a rule of thumb misses."""
    transfer, props = vessel["transfer"], vessel["properties"]
    near = int(np.argmin(np.abs(transfer.periods - transfer.period_roll)))
    return float(vessel["rows"][near]["a"][2, 2] / props["inertia"])


def roll_period_check(vessel: dict) -> dict[str, float]:
    """The resonance the solve gives against the hydrostatics' own period, with and without the added inertia."""
    section = vessel["section"]
    hydrostatic = 2.0 * math.pi * section.k_xx / math.sqrt(G_ACC * section.gm)
    solved = vessel["transfer"].period_roll
    return {"hydrostatic_s": float(hydrostatic), "solved_s": float(solved),
            "ratio": float(solved / hydrostatic), "added_inertia_ratio": inertia_ratio(vessel)}


# ------------------------------------------------------------------------------------------------ the mooring
def mooring_curves() -> dict[str, object]:
    """Tension against offset for the spread, and what losing a pair of lines does to the rest.

    The comparison is made at constant load, not at constant offset: the environmental load is what it is, so a spread
    with fewer lines sits further off until the ones left carry it. Two symmetric lines are removed so that the rest
    stay a balanced spread -- removing one from a symmetric spread would change which way the vessel lies, not only how
    far, and the load on the remaining lines would be a different question.
    """
    spread = mooring.spread(8, length=437.0, depth=120.0, span=400.0)
    offsets = np.linspace(-2.0, 18.0, 161)
    loaded = spread.utilisation(offsets)
    intact = loaded["worst"]
    kept = [1, 2, 3, 4, 5, 6]
    lost = mooring.Mooring(lines=tuple(spread.lines[i] for i in kept), depth=spread.depth, span=spread.span,
                           direction=np.asarray(spread.direction, dtype=float)[kept])
    reduced = lost.utilisation(offsets)
    unused = mooring.catenary(437.0, 120.0, 400.0, 1106.0)
    steady = 4.0e5
    return {"offsets": offsets, "intact": intact, "lost": reduced["worst"],
            "intact_worst": loaded["worst_utilisation"], "lost_worst": reduced["worst_utilisation"],
            "breaking": spread.lines[0].properties[1], "static": unused,
            "stiffness": float(np.gradient(intact, offsets)[len(offsets) // 2]),
            "lift_off_span": unused["reach"], "steady_load": steady,
            "intact_equilibrium": spread.equilibrium(steady), "lost_equilibrium": lost.equilibrium(steady)}


# ------------------------------------------------------------------------------------------------ the record
def instrument_record(forecast_spectrum, transfer, *, seed: int, slam: bool, line_step: float,
                      sensor: str | None, quick: bool) -> dict[str, np.ndarray]:
    """What the instruments would have written, with the events of the scenario in it.

    The three channels are the two accelerometers (the crane pedestal and the centre of gravity) and the fairlead load
    cell. ``slam`` adds the hull hitting the water, ``line_step`` a step in the tension as a tether changes, and
    ``sensor`` one instrument failing in one of the two ways an instrument does.
    """
    dt = 0.5 if quick else 0.25
    time_axis = np.arange(0.0, 1200.0, dt)
    state = motion.response(buoy_freq(), forecast_spectrum, transfer, seed=seed, count=64, depth=VESSEL["depth"])
    crane = state.vertical_acceleration(time_axis, CRANE)
    centre = state.vertical_acceleration(time_axis, motion.Point(0.0, 0.0))
    velocity = state.relative_velocity(time_axis, BOW)
    tension = 404e3 + 2.0e4 * state.vertical_displacement(time_axis, motion.Point(0.0, 0.0))
    # the model does not know about any of these: what happened goes into the instruments, not into the prediction
    hit = (time_axis > 300.0) & (time_axis < 315.0)
    pulse = np.exp(-((time_axis - 307.0) / 1.5) ** 2) * 3.0 * hit
    noise = {"crane": 0.4, "centre": 0.4, "tension": 6.0e3}
    measured_crane = crane + (pulse if slam else 0.0) + np.random.default_rng(seed + 1).normal(
        0.0, noise["crane"], time_axis.size)
    measured_centre = centre + (0.85 * pulse if slam else 0.0) + np.random.default_rng(seed + 2).normal(
        0.0, noise["centre"], time_axis.size)
    measured_tension = tension + line_step * (time_axis > 700.0) + np.random.default_rng(seed + 3).normal(
        0.0, noise["tension"], time_axis.size)
    if sensor == "spike":
        measured_crane += (np.random.default_rng(seed + 4).random(time_axis.size) < 0.02) * 25.0
    elif sensor == "stuck":
        measured_centre = np.full_like(measured_centre, measured_centre[0])
    return {"time": time_axis, "predicted": {"crane": crane, "centre": centre, "tension": tension},
            "measured": {"crane": measured_crane, "centre": measured_centre, "tension": measured_tension},
            "noise": noise, "velocity": velocity}


def buoy_freq() -> np.ndarray:
    return np.linspace(0.02, 0.60, 240)


def event_experiment(transfer, quick: bool) -> dict[str, object]:
    """Four scenarios, what actually happened, and what the residual said happened."""
    sea = sp.Jonswap(hs=4.2, tp=7.0)          # a storm: short and steep enough that the bow cannot follow it
    density = sea.density_hz(buoy_freq())
    scenarios = {"quiet": {"slam": False, "line_step": 0.0, "sensor": None},
                 "slam": {"slam": True, "line_step": 0.0, "sensor": None},
                 "line": {"slam": False, "line_step": 2.4e4, "sensor": None},
                 "spike": {"slam": False, "line_step": 0.0, "sensor": "spike"},
                 "stuck": {"slam": False, "line_step": 0.0, "sensor": "stuck"}}
    out = {}
    for name, spec in scenarios.items():
        data = instrument_record(density, transfer, seed=11, quick=quick, **spec)
        verdict = confidence.evaluate(time=data["time"], predicted=data["predicted"], measured=data["measured"],
                                      noise=data["noise"], relative_velocity=data["velocity"],
                                      slamming_threshold=motion.slamming_threshold(VESSEL["length"]),
                                      redundant=("crane", "centre"), window=120.0)
        out[name] = {"kinds": sorted({e.kind for e in verdict.events}),
                     "events": [str(e) for e in verdict.events],
                     "sensor_faults": list(verdict.sensor_faults),
                     "worst": verdict.worst_residual}
    return out


# ------------------------------------------------------------------------------------------------ the window
def forecast_window(buoy: Buoy, model: seastate.SeaStateModel, transfer, leads: np.ndarray,
                    criteria: operability.Criteria, decision_curve: dict[float, float]) -> dict[str, object]:
    """The product: a forecast of the sea, the vessel's answer to it, and the hours the criterion stays open.

    The analysis time is the one in the second half of the record with the most height waiting in the next twelve
    hours -- a day when the window actually closes, which is the only day worth showing.
    """
    states = buoy.analyses(every=4, leads=leads)
    second_half = states["time"] > int(np.median(states["time"]))

    def evaluate_sea(height, period):
        shape = sp.Jonswap(hs=float(max(height, 0.05)), tp=float(max(period, 2.0)))
        density = shape.density_hz(buoy.freq)
        acceleration = motion.significant_acceleration(buoy.freq, density, transfer, CRANE)
        velocity = motion.significant_relative_velocity(buoy.freq, density, transfer, BOW)
        return acceleration, velocity

    # the day the window closes: a forecast that is open at the first lead and shut before the last one, which is the
    # only day an operability window has anything to say about; failing that, the roughest day in the record
    chosen, best = None, -1e9
    roughest, worst = None, -1e9
    for i in np.where(second_half)[0]:
        spectrum = model.forecast(states["freq"][i], states["density"][i], states["wind"][i], leads=leads)
        pairs = [evaluate_sea(spectrum.hs[j], spectrum.tp[j]) for j in range(leads.size)]
        allowed = criteria.allows(spectrum.hs, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
        if allowed[0]:
            # the window has to be open when the forecast is issued, so that there is something to close; among those
            # days the one that shuts soonest is the one worth reporting
            open_hours = float(np.sum(allowed))
            score = -open_hours
            if score > best:
                chosen, best = i, score
        height = float(np.interp(12.0, leads, states["obs_hs"][i]))
        if height > worst:
            roughest, worst = i, height
    if chosen is None:
        chosen = roughest
    spectrum = model.forecast(states["freq"][chosen], states["density"][chosen], states["wind"][chosen], leads=leads)

    forecast_acceleration, forecast_velocity, truth_acceleration, truth_velocity = [], [], [], []
    for j in range(leads.size):
        a, v = evaluate_sea(spectrum.hs[j], spectrum.tp[j])
        forecast_acceleration.append(a)
        forecast_velocity.append(v)
        a, v = evaluate_sea(states["obs_hs"][chosen][j], states["obs_tp"][chosen][j])
        truth_acceleration.append(a)
        truth_velocity.append(v)
    forecast_acceleration = np.array(forecast_acceleration)
    forecast_velocity = np.array(forecast_velocity)
    truth_acceleration = np.array(truth_acceleration)
    truth_velocity = np.array(truth_velocity)
    confidence_curve = lambda hours: float(np.interp(hours, sorted(decision_curve),  # noqa: E731
                                                    [decision_curve[k] for k in sorted(decision_curve)]))
    found = operability.opening(criteria, leads, spectrum.hs, forecast_acceleration, confidence=confidence_curve,
                                relative_velocity=forecast_velocity)
    truth_verdicts = criteria.allows(states["obs_hs"][chosen], truth_acceleration, truth_velocity)
    forecast_verdicts = criteria.allows(spectrum.hs, forecast_acceleration, forecast_velocity)
    return {"analysis_time": str(np.datetime64(int(states["time"][chosen]), "s")), "leads": leads,
            "hs": spectrum.hs, "tp": spectrum.tp, "obs_hs": states["obs_hs"][chosen],
            "obs_tp": states["obs_tp"][chosen], "acceleration": forecast_acceleration,
            "obs_acceleration": truth_acceleration, "velocity": forecast_velocity,
            "obs_velocity": truth_velocity, "forecast_verdict": forecast_verdicts, "truth_verdict": truth_verdicts,
            "opening": found, "agreement": float(np.mean(forecast_verdicts == truth_verdicts))}


def decision_curve(buoy: Buoy, model: seastate.SeaStateModel, transfer, leads: np.ndarray,
                   criteria: operability.Criteria, wave: dict) -> dict[float, float]:
    """The measured confidence: how often the verdict from the forecast matched the verdict from what happened."""
    validation = wave["validation"]
    curve = {}
    for j, lead in enumerate(leads):
        if j >= validation["obs_hs"].shape[1]:
            continue
        rows = validation["freq"].shape[0]
        verdicts, truths = [], []
        for i in range(rows):
            state = model.forecast(validation["freq"][i], validation["density"][i], validation["wind"][i],
                                   leads=[lead])
            shape = sp.Jonswap(hs=float(state.hs[0]), tp=float(state.tp[0]))
            density = shape.density_hz(buoy.freq)
            verdicts.append(criteria.allows([state.hs[0]],
                                            [motion.significant_acceleration(buoy.freq, density, transfer, CRANE)])[0])
            observed = sp.Jonswap(hs=float(max(validation["obs_hs"][i, j], 0.05)),
                                  tp=float(max(validation["obs_tp"][i, j], 1.0)))
            observed_density = observed.density_hz(buoy.freq)
            truths.append(criteria.allows([validation["obs_hs"][i, j]],
                                          [motion.significant_acceleration(buoy.freq, observed_density, transfer,
                                                                           CRANE)])[0])
        curve[float(lead)] = operability.decision_skill(np.array(verdicts), np.array(truths))
    return curve


# ------------------------------------------------------------------------------------------------ figures
def draw(buoy: Buoy, wave: dict, agreement: dict, vessel: dict, lines: dict, window: dict, events: dict,
         quick: bool) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    omega_red, ink = "#a23a00", "#262626"
    blue = "#0081be"

    # 1 -- the measured spectrum and the analytic shape
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    picks = [int(np.argmax(buoy.density.sum(axis=1))), int(np.argmin(buoy.density.sum(axis=1)))]
    for ax, i in zip(axes, picks, strict=True):
        ax.plot(buoy.freq, buoy.density[i], "o-", color=ink, ms=3, lw=1, label="measured (NDBC)")
        fitted, rms = sp.fit(buoy.freq, buoy.density[i])
        grid = np.linspace(0.02, 0.5, 300)
        ax.plot(grid, fitted.density_hz(grid), color=red_or(omega_red), lw=2,
                label=f"JONSWAP fit, Hs {fitted.hs:.1f} m Tp {fitted.tp:.1f} s, {rms:.0%} rms")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("S(f) [m$^2$/Hz]")
        ax.set_title(str(buoy.spec_time[i])[:16], color=ink)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("The measured spectrum against the shape the package fits to it", color=ink)
    fig.tight_layout()
    path = FIGURES / "spectra.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path.relative_to(HERE)))

    # 2 -- the skill of the forecast against carrying the observation forward
    leads = np.array([row["lead_hours"] for row in wave["table"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field in zip(axes, ("hs", "tp"), strict=True):
        skill = np.array([row[field]["skill"] for row in wave["table"]])
        ax.axhline(0.0, color=ink, lw=0.8)
        ax.plot(leads, 100 * skill, "o-", color=blue if field == "hs" else omega_red)
        ax.set_xlabel("lead time [h]")
        ax.set_ylabel("skill against persistence [%]")
        ax.set_title(f"{field.upper()}: 1 - MSE(model)/MSE(persistence)", fontsize=9, color=ink)
    fig.suptitle("Where the spectral forecast beats carrying the last observation forward", color=ink)
    fig.tight_layout()
    path = FIGURES / "skill.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path.relative_to(HERE)))

    # 3 -- the transfer functions
    transfer = vessel["transfer"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(transfer.periods, transfer.heave, "o-", color=blue, ms=3, label="heave [m/m]")
    ax.plot(transfer.periods, transfer.roll, "s-", color=omega_red, ms=3, label="roll [rad/rad slope]")
    ax.plot(transfer.periods, transfer.pitch, "^-", color=ink, ms=3, label="pitch, longitudinal strip [rad/rad]")
    ax.axvline(transfer.period_roll, color=omega_red, lw=0.8, ls=":")
    ax.annotate(f"roll resonance {transfer.period_roll:.1f} s", (transfer.period_roll, 0.2), rotation=90, fontsize=7,
                color=omega_red, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 8, 10, 14, 20, 25])
    ax.set_xticklabels(["4", "6", "8", "10", "14", "20", "25"])
    ax.set_xlabel("period [s]")
    ax.set_ylabel("response per unit wave")
    ax.set_title("The section's answer to a regular wave, solved rather than tabulated", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "rao.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path.relative_to(HERE)))

    # 4 -- the mooring
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(lines["offsets"], lines["intact"] / 1e6, color=blue, label="eight lines intact")
    ax.plot(lines["offsets"], lines["lost"] / 1e6, color=omega_red, label="one line lost")
    ax.plot([lines["intact_equilibrium"]["surge"]], [lines["intact_equilibrium"]["tension"] / 1e6], "o", color=blue,
            ms=6)
    ax.plot([lines["lost_equilibrium"]["surge"]], [lines["lost_equilibrium"]["tension"] / 1e6], "o", color=omega_red,
            ms=6)
    ax.axhline(0.3 * lines["breaking"] / 1e6, color=ink, lw=0.8, ls="--")
    ax.annotate("30 % of breaking load", (lines["offsets"][0], 0.3 * lines["breaking"] / 1e6), fontsize=7,
                color=ink, va="bottom")
    ax.set_xlabel("fairlead excursion [m]")
    ax.set_ylabel("worst fairlead tension [MN]")
    ax.set_title("Quasi-static catenary: the load is not linear in the offset", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "mooring.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path.relative_to(HERE)))

    # 5 -- the window
    leads = window["leads"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(leads, window["hs"], "o-", color=blue, label="forecast")
    axes[0].plot(leads, window["obs_hs"], "x", color=ink, ms=4, label="what the buoy then measured")
    axes[0].set_ylabel("Hs [m]")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(leads, window["acceleration"] / G_ACC, "o-", color=omega_red, label="forecast")
    axes[1].plot(leads, window["obs_acceleration"] / G_ACC, "x", color=ink, ms=4, label="observed")
    axes[1].set_ylabel("significant vertical\nacceleration at the crane [g]")
    axes[1].legend(frameon=False, fontsize=8)
    allowed = window["forecast_verdict"].astype(float)
    axes[2].fill_between(leads, 0, 1, where=allowed > 0, color=blue, alpha=0.25, step="mid")
    axes[2].fill_between(leads, 0, 1, where=allowed < 1, color=omega_red, alpha=0.25, step="mid")
    opening = window["opening"]
    axes[2].set_ylabel("verdict")
    axes[2].set_xlabel("lead time [h]")
    axes[2].set_yticks([])
    axes[2].set_title(opening["sentence"], fontsize=8, color=ink)
    fig.suptitle("The operability window, and what the sea did next", color=ink)
    fig.tight_layout()
    path = FIGURES / "window.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path.relative_to(HERE)))
    return written


def red_or(value: str) -> str:
    return value


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer leads, coarser search, no figures")
    args = parser.parse_args()
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    buoy = Buoy(DATA / "ndbc_41010.npz")
    leads = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0]) if args.quick else np.arange(1.0, 25.0)
    print(f"buoy 41010: {buoy.record.time[0]} to {buoy.record.time[-1]}, {buoy.spec_time.size} spectra")

    agreement = spectrum_agreement(buoy, args.quick)
    print(f"JONSWAP fit to the measured spectra: {agreement['fit_rms']:.1%} rms in log S "
          f"(Pierson-Moskowitz {agreement['fit_rms_pm']:.1%}); median gamma {agreement['fit_gamma']:.1f}")
    print(f"4 sqrt(m0) over the published bands / the buoy's WVHT: {agreement['hs_ratio']:.3f}")

    wave = wave_skill(buoy, leads, args.quick)
    print(f"calibrated on the first half: {wave['model']}")
    for row in wave["table"]:
        if row["lead_hours"] in (1.0, 6.0, 12.0, 24.0):
            print(f"  lead {row['lead_hours']:4.0f} h  Hs skill {row['hs']['skill']:+.3f} "
                  f"(rmse {row['hs']['rmse']:.3f} vs {row['hs']['rmse_reference']:.3f} m)  "
                  f"Tp skill {row['tp']['skill']:+.3f} (rmse {row['tp']['rmse']:.2f} vs {row['tp']['rmse_reference']:.2f} s)")

    # the other wind-sea closure, measured rather than asserted: growing it toward a fully developed sea
    grown = seastate.SeaStateModel(swell_decay_hours=wave["model"].swell_decay_hours,
                                   windsea_growth_hours=wave["model"].windsea_growth_hours, windsea_growth=True)
    validation = wave["validation"]
    closure = {}
    for lead in (6.0, 12.0, 24.0):
        if lead not in leads:
            continue
        j = int(np.where(leads == lead)[0][0])
        if j >= validation["obs_hs"].shape[1]:
            continue
        hs, tp, bias = [], [], []
        for i in range(validation["freq"].shape[0]):
            out = grown.forecast(validation["freq"][i], validation["density"][i], validation["wind"][i], leads=[lead])
            hs.append(out.hs[0])
            tp.append(out.tp[0])
            bias.append(out.hs[0] - wave["table"][j]["hs_predicted"][i])
        closure[str(lead)] = {"hs_skill": seastate.skill(np.array(hs), validation["obs_hs"][:, j],
                                                         validation["hs"])["skill"],
                              "tp_skill": seastate.skill(np.array(tp), validation["obs_tp"][:, j],
                                                         validation["tp"])["skill"],
                              "hs_bias": float(np.mean(bias))}
        print(f"  grown closure at {lead:.0f} h: Hs skill {closure[str(lead)]['hs_skill']:+.3f} "
              f"(bias {closure[str(lead)]['hs_bias']:+.2f} m), Tp skill {closure[str(lead)]['tp_skill']:+.3f}")

    vessel = vessel_transfer(args.quick)
    check = roll_period_check(vessel)
    print(f"roll: hydrostatics {check['hydrostatic_s']:.1f} s, solved {check['solved_s']:.1f} s "
          f"(added inertia {check['added_inertia_ratio']:.2f} of I44); "
          f"heave at 80 s {vessel['long_wave_heave']:.3f}")
    lines = mooring_curves()
    print(f"mooring: {lines['static']['T_fairlead'] / 1e3:.0f} kN static, stiffness {lines['stiffness'] / 1e3:.1f} "
          f"kN/m, lifts off at {lines['lift_off_span']:.0f} m span; at a steady "
          f"{lines['steady_load'] / 1e3:.0f} kN load {lines['intact_equilibrium']['tension'] / 1e3:.0f} kN intact, "
          f"{lines['lost_equilibrium']['tension'] / 1e3:.0f} kN with two lines lost")

    events = event_experiment(vessel["transfer"], args.quick)
    for name, found in events.items():
        print(f"  scenario {name:<6}: diagnosed {found['kinds'] or ['nothing']}, sensors {found['sensor_faults']}")

    criteria = operability.Criteria(hs_max=1.5, acceleration_max=0.2, length=VESSEL["length"])
    without_slam = operability.Criteria(hs_max=1.5, acceleration_max=0.2, length=0.0)
    curve = decision_curve(buoy, wave["model"], vessel["transfer"], leads, criteria, wave)
    window = forecast_window(buoy, wave["model"], vessel["transfer"], leads, criteria, curve)
    relaxed = forecast_window(buoy, wave["model"], vessel["transfer"], leads, without_slam, curve)
    print(f"without the slamming criterion the same day stays open until +{relaxed['opening']['until_hours']:.0f} h")
    print("decision skill by lead: " + ", ".join(f"{h:g}h {v:.0%}" for h, v in sorted(curve.items())))
    print(f"window at {window['analysis_time']}: {window['opening']['sentence']}")
    print(f"the forecast's verdict matched the observation on {window['agreement']:.0%} of the hours at their own lead")

    payload = {
        "buoy": {"station": buoy.record.station, "rows": int(buoy.record.time.size),
                 "first": str(buoy.record.time[0]), "last": str(buoy.record.time[-1]),
                 "spectra": int(buoy.spec_time.size), "bands": int(buoy.freq.size),
                 "source": buoy.record.source},
        "spectrum": {k: v for k, v in agreement.items()
                     if k in ("fit_rms", "fit_rms_pm", "hs_ratio", "fit_gamma", "count")},
        "wave": {"model": {"swell_decay_hours": wave["model"].swell_decay_hours,
                           "windsea_growth_hours": wave["model"].windsea_growth_hours,
                           "ratio": wave["model"].ratio},
                 "calibration_cost": wave["search"]["cost"],
                 "calibration_hours": int(wave["calibration"]["time"].size),
                 "validation_hours": int(wave["validation"]["time"].size),
                 "table": [{k: (v if k in ("lead_hours",) else {m: v[m] for m in
                                                               ("n", "rmse", "bias", "correlation", "skill",
                                                                "rmse_reference")})
                            for k, v in row.items() if k in ("lead_hours", "hs", "tp")}
                           for row in wave["table"]]},
        "closure": {"grown": closure},
        "vessel": {"section": {"beam": VESSEL["beam"], "draught": VESSEL["draught"], "length": VESSEL["length"],
                               "kg": VESSEL["kg"], "gm": float(vessel["section"].gm)},
                   "roll_period": check, "mass": vessel["properties"],
                   "long_wave": {"heave": vessel["long_wave_heave"], "roll": vessel["long_wave_roll"]},
                   "cells": vessel["cells"],
                   "transfer": {"periods": vessel["transfer"].periods.tolist(),
                                "heave": vessel["transfer"].heave.tolist(),
                                "roll": vessel["transfer"].roll.tolist(),
                                "pitch": vessel["transfer"].pitch.tolist()}},
        "mooring": {"static_tension": lines["static"]["T_fairlead"], "reach": lines["lift_off_span"],
                    "steady_load": lines["steady_load"],
                    "intact_equilibrium": lines["intact_equilibrium"],
                    "lost_equilibrium": lines["lost_equilibrium"],
                    "stiffness": lines["stiffness"], "breaking": lines["breaking"],
                    "utilisation_static": float(lines["static"]["T_fairlead"] / lines["breaking"]),
                    "utilisation_at_load": lines["intact_equilibrium"]["utilisation"],
                    "utilisation_at_load_lost": lines["lost_equilibrium"]["utilisation"]},
        "events": events,
        "window": {"analysis_time": window["analysis_time"],
                   "sentence": window["opening"]["sentence"],
                   "open_now": window["opening"]["open_now"],
                   "until_hours": window["opening"]["until_hours"],
                   "confidence": window["opening"]["confidence"],
                   "criteria": {"hs_max": criteria.hs_max, "acceleration_max": criteria.acceleration_max,
                                "relative_velocity_max": criteria.slam_limit},
                   "confidence_curve": {str(k): v for k, v in sorted(curve.items())},
                   "agreement": window["agreement"],
                   "without_slam": {"until_hours": relaxed["opening"]["until_hours"],
                                    "sentence": relaxed["opening"]["sentence"]},
                   "leads": window["leads"].tolist(), "hs": window["hs"].tolist(), "tp": window["tp"].tolist(),
                   "obs_hs": window["obs_hs"].tolist(), "obs_tp": window["obs_tp"].tolist(),
                   "acceleration": window["acceleration"].tolist(),
                   "obs_acceleration": window["obs_acceleration"].tolist(),
                   "forecast_verdict": [bool(v) for v in window["forecast_verdict"]],
                   "truth_verdict": [bool(v) for v in window["truth_verdict"]]},
        "seconds": round(time.time() - started, 1),
    }
    (RESULTS / "maritime_operability.json").write_text(json.dumps(payload, indent=1, default=float),
                                                       encoding="utf-8")
    print(f"results -> results/maritime_operability.json ({payload['seconds']:.0f} s)")
    if not args.quick:
        written = draw(buoy, wave, agreement, vessel, lines, window, events, args.quick)
        print(f"figures -> {', '.join(written)}")


if __name__ == "__main__":
    main()
