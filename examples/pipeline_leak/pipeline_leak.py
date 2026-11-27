"""A leak in a crude-oil pipeline: when it is heard, where it is, and whether it was ever there.

    python examples/pipeline_leak/pipeline_leak.py            # the campaign, about a minute
    python examples/pipeline_leak/pipeline_leak.py --quick    # fewer seeds and holes, about 20 s

The line is 20 km of 400 mm steel carrying 850 kg/m3 crude at 0.25 m3/s (2 m/s): a pressure-wave speed of 1 140 m/s,
so a wave crosses it in 17.5 s, and 202 m of head is lost to friction, which a pump supplies and a receiving valve
disposes of. Two pressure transmitters sit 480 m inside each end -- on the line, not on the station boundary, because
a transmitter *on* a constant-head boundary sees nothing at all -- and two flow meters measure what goes in and what
comes out.

What the example measures, and why each number is the one that matters:

1. **The transient is the method of characteristics, checked against what can be checked.** At the instant the hole
   opens the head there must drop by ``a Q_L / (2 g A)`` and each side of the line must take half the leak; the wave
   must then leave the hole at ``a`` in both directions. Both are compared with the computation, not asserted.
2. **The delay.** A leak cannot be reported before its rarefaction reaches a transmitter, ``min(x_L - x_a, x_b - x_L)/a``
   -- 8.2 s for a hole in the middle, 2.2 s near the upstream end, and that floor is a property of the line, not of
   the detector. The measured delay is put next to it.
3. **The place.** ``x_L`` from the difference of the two arrival times, against the position the hole was actually
   drilled at, in metres and as a fraction of the line. The floor here is the acquisition, ``a dt_s / 2``, plus what
   the interpolation inside the sample buys back -- and a hole outside the two transmitters cannot be placed at all.
4. **The false alarms.** Sixty records of 1 400 s of normal operation -- including two valve manoeuvres the model
   knows about -- per threshold, so the number is alarms per hour and not a claim of "none".
5. **Instrument or leak.** A bias, a drift, extra noise and a latch, on one transmitter at a time, against holes of
   known size: the answer is a confusion matrix over what the reading actually was.
6. **The curve.** Detection rate and delay against hole diameter, for each threshold, so the cost of a quiet
   detector is visible in the small holes it loses, and the point where the loss stops being the detector's fault.

Two negatives are recorded rather than hidden. A hole outside the two transmitters cannot be timed by their
difference -- both fronts go past both of them and the estimate collapses onto the nearer transmitter, so the verdict
says ``outside_transmitters`` and the error is the distance to the transmitter. And a valve the model was not told
about produces exactly the pressure signature of a leak at both ends; what refuses it is the mass balance, which costs
the settle time ``2L/a``, and the verdict is then ``operation`` -- the right refusal for the wrong reason if the real
event had been a leak that had not yet settled.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_physics.pipeline_leak import (
    Fault,
    G,
    Leak,
    Pipeline,
    Sensors,
    detect,
    expected,
    minimum_detectable_leak,
    observe,
    simulate,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

PIPE = Pipeline(length=20000.0, diameter=0.4)
SENSORS = Sensors(upstream=480.0, downstream=480.0, sample=0.05)
REACHES = 500
MONITOR = (SENSORS.upstream, PIPE.length - SENSORS.downstream)
HOLES_MM = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0)
POSITIONS = (3000.0, 9800.0, 15500.0)
THRESHOLDS = (8.0, 12.0, 16.0, 20.0)
OPERATING = 12.0                                   # the decision interval the example runs at
QUIET = 1400.0                                     # seconds of normal operation per false-alarm record


def valve(t: float) -> float:
    """Normal operation: every 170 s the receiving valve closes 10 % over 5 s, holds for 25 s and reopens over 5 s.
    Manoeuvres the model *knows about*, which is the case a model-based detector is built for -- and, in the
    false-alarm count, the case where a naive threshold alarms twice every three minutes."""
    phase = t % 170.0
    if 30.0 <= phase <= 35.0:
        return 1.0 - 0.10 * (phase - 30.0) / 5.0
    if 35.0 < phase < 60.0:
        return 0.90
    if 60.0 <= phase <= 65.0:
        return 0.90 + 0.10 * (phase - 60.0) / 5.0
    return 1.0


def hole(diameter_mm: float, distance: float, t_open: float = 20.0, discharge: float = 0.62) -> Leak:
    return Leak(distance=distance, area=math.pi * (diameter_mm * 1e-3) ** 2 / 4.0, t_open=t_open,
                discharge=discharge)


def transient(name: str) -> dict:
    """The transient and what can be checked of it without a second solver: the closed form of the first step, the
    speed of the front that leaves the hole, and the amplitude the two transmitters finally see."""
    leak = hole(10.0, 9800.0)
    run = simulate(PIPE, n_reaches=REACHES, t_end=120.0, leak=leak, monitor=MONITOR, record=True)
    field, t = run["field"], run["t"]
    steady = run["steady"]
    node = int(round(9800.0 / run["dx"]))
    opened = int(np.argmax(t > 20.0))

    leak_at_hole = run["leak_flow"][opened]
    first_step = float(field[opened][node] - steady[node])
    closed = -PIPE.wave_speed * leak_at_hole / (2.0 * G * PIPE.area)
    split_up, split_down = (float(v) for v in run["leak_split"][opened])
    linear = leak.capacity() * math.sqrt(float(steady[node]) - 0.0)
    linear_dh = -PIPE.wave_speed * linear / (2.0 * G * PIPE.area)

    front = {}
    for elapsed in (2.0, 4.0, 6.0, 8.0):
        k = int(np.argmin(np.abs(t - (20.0 + elapsed))))
        moved = np.abs(field[k] - steady) > 0.02 * abs(closed)
        if moved.any():
            reach = float(run["x"][moved].max() - run["x"][moved].min())
            front[f"{elapsed:g}s"] = {"spread_m": reach, "speed_m_s": reach / (2.0 * elapsed)}

    sensor = int(round(MONITOR[0] / run["dx"]))
    fig_frame(field, t, steady, run["x"], node, name)

    return {"hole_mm": 10.0, "distance_m": 9800.0, "leak_rate_m3_s": float(leak_at_hole),
            "wave_speed_m_s": PIPE.wave_speed, "travel_time_s": PIPE.travel_time,
            "period_s": 2.0 * PIPE.travel_time, "friction_head_m": PIPE.friction_head,
            "friction_factor": PIPE.friction, "reynolds": PIPE.reynolds,
            "first_step": {"computed_dh_m": first_step, "closed_form_dh_m": closed,
                           "relative_error": abs(first_step - closed) / abs(closed),
                           "computed_split_up_m3_s": split_up, "computed_split_down_m3_s": split_down,
                           "closed_form_split_m3_s": leak_at_hole / 2.0,
                           "split_relative_error": max(abs(split_up - leak_at_hole / 2.0),
                                                       abs(-split_down - leak_at_hole / 2.0))
                           / (leak_at_hole / 2.0),
                           "estimated_before_opening_dh_m": linear_dh,
                           "estimate_error": abs(linear_dh - first_step) / abs(first_step)},
            "front": front,
            "sensor_amplitude_m": float(np.min(field[:, sensor] - steady[sensor])),
            "joukowsky_m": PIPE.wave_speed * float(leak_at_hole) / (2.0 * G * PIPE.area),
            "attenuation_over_9_3_km": 1.0 - abs(float(np.min(field[:, sensor] - steady[sensor])))
            / abs(PIPE.wave_speed * float(leak_at_hole) / (2.0 * G * PIPE.area))}


def fig_frame(field, t, steady, x, node, name: str) -> None:
    """Space-time map of the hole: two fronts leaving it at ``a``, the reflections at the two ends, and where the
    transmitters sit. A picture of the delay and of why a leak is seen at both ends before it is seen anywhere else."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), gridspec_kw={"width_ratios": [1.6, 1.0]})
    deviation = steady - field                        # positive where the line has lost head
    limit = float(np.abs(deviation).max())
    mesh = axes[0].pcolormesh(t, x / 1000.0, deviation.T, cmap="Reds", vmin=0.0, vmax=limit, shading="auto")
    axes[0].set_xlim(15.0, 72.0)
    axes[0].plot([20.0, 20.0], [0.0, 20.0], color="black", lw=0.8, ls=":")
    for position, label, offset in ((480.0, "transmitter A", 0.4), (PIPE.length - 480.0, "transmitter B", -0.9)):
        axes[0].axhline(position / 1000.0, color="black", lw=1.0, ls="--")
        axes[0].annotate(label, xy=(46.0, position / 1000.0 + offset), fontsize=8)
    axes[0].annotate("the hole opens", xy=(20.5, 9.8), xytext=(26.0, 6.5), fontsize=9,
                     arrowprops={"arrowstyle": "->", "lw": 0.8})
    axes[0].set_xlabel("time  [s]")
    axes[0].set_ylabel("distance along the line  [km]")
    axes[0].set_title("head lost, $H_0-H$: two fronts leave the hole at $a$ = 1 140 m/s", fontsize=10)
    figure.colorbar(mesh, ax=axes[0], fraction=0.046, label="H$_0$ - H  [m]")

    axes[1].plot(t, field[:, node] - steady[node], lw=0.9, label="at the hole (9.8 km)")
    axes[1].plot(t, field[:, int(round(MONITOR[0] / (PIPE.length / REACHES)))]
                 - steady[int(round(MONITOR[0] / (PIPE.length / REACHES)))], lw=0.9, label="at transmitter A")
    axes[1].axvline(20.0, color="black", lw=0.8, ls=":")
    axes[1].axvline(20.0 + (9800.0 - 480.0) / PIPE.wave_speed, color="black", lw=0.8, ls="--")
    axes[1].annotate("the front reaches A", xy=(20.0 + (9800.0 - 480.0) / PIPE.wave_speed, -0.85),
                     xytext=(55.0, -2.2), fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.8})
    axes[1].set_ylim(-2.6, 0.4)
    axes[1].set_xlabel("time  [s]")
    axes[1].set_ylabel("H - H$_0$  [m]")
    axes[1].set_title("the same event at two places", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)
    figure.suptitle("A 10 mm hole at 9.8 km in a 20 km line", fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(FIGURES / name, dpi=130)
    plt.close(figure)


def instrument(reference: dict, leak_observation: dict, nominal: dict) -> dict:
    """The ways a transmitter lies, on a record of normal operation, next to the truth on the same record: what the
    residual looks like when it is not the line that moved."""
    rows = {}
    cases = {"bias +3 m": (Fault("offset", "a", 40.0, 3.0),),
             "bias -3 m": (Fault("offset", "a", 40.0, -3.0),),
             "drift -20 mm/s": (Fault("drift", "a", 40.0, -0.02),),
             "noise x8": (Fault("noise", "a", 40.0, 8.0),),
             "latched": (Fault("stuck", "a", 40.0, 0.0),),
             "latched (B)": (Fault("stuck", "b", 40.0, 0.0),)}
    hits = {name: {"leak": 0, "instrument": 0, "operation": 0, "none": 0} for name in cases}
    for name, faults in cases.items():
        for seed in range(6):
            verdict = detect(observe(nominal, PIPE, SENSORS, seed=700 + seed, faults=faults), reference, PIPE, SENSORS)
            hits[name][verdict["source"]] += 1
    for name, counts in hits.items():
        rows[name] = {"runs": 6, **counts, "called_instrument": counts["instrument"] / 6.0}
    fig_faults(nominal, reference, leak_observation, PIPE)
    return rows


def fig_faults(nominal: dict, reference: dict, leak_observation: dict, pipe: Pipeline) -> None:
    """Four residuals side by side: a hole in the line and three instruments that are lying. Which end moved, and in
    which direction, is the whole argument."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 6.4), sharex=True)
    panels = (
        (leak_observation, reference, "a 10 mm hole at 9.8 km: both ends fall", axes[0][0]),
        (observe(nominal, pipe, SENSORS, seed=703, faults=(Fault("offset", "a", 40.0, 3.0),)), reference,
         "a bias of +3 m on transmitter A: one end rises", axes[0][1]),
        (observe(nominal, pipe, SENSORS, seed=703, faults=(Fault("drift", "a", 40.0, -0.02),)), reference,
         "a drift of -20 mm/s on A: one end walks away", axes[1][0]),
        (observe(nominal, pipe, SENSORS, seed=703, faults=(Fault("stuck", "a", 40.0, 0.0),)), reference,
         "A latched at 40 s: one end stops, the other does not", axes[1][1]),
    )
    for observation, ref, title, axis in panels:
        axis.plot(observation["t"], (observation["head_a"] - ref["head_a"]) / SENSORS.sigma, lw=0.8,
                  label="transmitter A", color="#0081be")
        axis.plot(observation["t"], (observation["head_b"] - ref["head_b"]) / SENSORS.sigma, lw=0.8,
                  label="transmitter B", color="#a23a00")
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("residual  [$\\sigma$]")
        axis.grid(alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("time  [s]")
    axes[0][0].legend(fontsize=8, loc="lower left")
    figure.suptitle("Residuals against the nominal transient: the line moves both ends, an instrument moves one",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "residuals.png", dpi=130)
    plt.close(figure)


def false_alarms(nominal: dict, reference: dict, seeds: int) -> dict:
    """Alarms per hour on normal operation, and what a hole that is *not* there does to the count."""
    hours = seeds * float(nominal["t"][-1]) / 3600.0
    counts = {}
    for threshold in THRESHOLDS:
        alarms = sum(detect(observe(nominal, PIPE, SENSORS, seed=3000 + s), reference, PIPE, SENSORS,
                            threshold=threshold)["source"] != "none" for s in range(seeds))
        counts[f"{threshold:g}"] = {"alarms": alarms, "records": seeds, "hours": hours,
                                    "per_hour": alarms / hours}
    hidden = simulate(PIPE, n_reaches=REACHES, t_end=float(nominal["t"][-1]), monitor=MONITOR)
    blind = expected(hidden, SENSORS)                    # a model that was never told about the manoeuvres
    unrecorded = {"alarms": 0, "records": seeds}
    sources = {}
    for seed in range(seeds):
        verdict = detect(observe(nominal, PIPE, SENSORS, seed=3000 + seed), blind, PIPE, SENSORS,
                         threshold=OPERATING)
        sources[verdict["source"]] = sources.get(verdict["source"], 0) + 1
        unrecorded["alarms"] += verdict["source"] != "none"
    unrecorded["sources"] = sources
    unrecorded["per_hour"] = unrecorded["alarms"] / hours
    return {"by_threshold": counts, "unrecorded_manoeuvre": unrecorded}


def leak_sweep(nominal: dict, reference: dict, seeds: int) -> dict:
    """Holes of known size at known places: is it seen, how late, how far off, and how much does it say it is losing."""
    table = {}
    for diameter in HOLES_MM:
        row = {}
        for position in POSITIONS:
            run = simulate(PIPE, n_reaches=REACHES, t_end=float(nominal["t"][-1]),
                           leak=hole(diameter, position), monitor=MONITOR)
            truth = float(run["leak_flow"][-1])
            floor = min(position - MONITOR[0], MONITOR[1] - position) / PIPE.wave_speed
            observations = [observe(run, PIPE, SENSORS, seed=500 + seed) for seed in range(seeds)]
            by_threshold = {}
            for threshold in THRESHOLDS:
                seen, delays, errors, held, rates, fronts = 0, [], [], [], [], []
                for observation in observations:
                    verdict = detect(observation, reference, PIPE, SENSORS, threshold=threshold)
                    if verdict["source"] != "leak":
                        continue
                    seen += 1
                    delays.append(verdict["alarm_time"] - 20.0)
                    evidence = verdict["evidence"]
                    if evidence["front_sigma"] is not None:
                        fronts.append(min(evidence["front_sigma"].values()))
                    if verdict["leak_distance"] is not None:          # heard and placed
                        errors.append(verdict["leak_distance"] - position)
                    elif evidence["delta_t"] is not None:             # placed, then withheld: what it would have said
                        held.append((MONITOR[0] + MONITOR[1] - PIPE.wave_speed * evidence["delta_t"]) / 2.0 - position)
                    if verdict["leak_rate"] is not None:
                        rates.append(verdict["leak_rate"] / truth - 1.0)
                by_threshold[f"{threshold:g}"] = {
                    "detected": seen, "placed": len(errors), "runs": seeds,
                    "errors_m": [round(e, 1) for e in errors],
                    "withheld_errors_m": [round(e, 1) for e in held],
                    "weakest_front_sigma": round(min(fronts), 2) if fronts else None,
                    "median_delay_s": float(np.median(delays)) if delays else None,
                    "median_error_m": float(np.median(np.abs(errors))) if errors else None,
                    "worst_error_m": float(np.max(np.abs(errors))) if errors else None,
                    "median_rate_error": float(np.median(np.abs(rates))) if rates else None}
            row[f"{position:.0f}"] = {"hole_mm": diameter, "distance_m": position, "leak_rate_m3_s": truth,
                                      "fraction_of_flow": truth / PIPE.flow, "delay_floor_s": floor,
                                      "far_leg_m": max(position - MONITOR[0], MONITOR[1] - position),
                                      "by_threshold": by_threshold}
        table[f"{diameter:g}"] = row
    return table


def by_position(sweep: dict, threshold: float = OPERATING) -> dict:
    """Accepted and withheld positions, side by side: what the gate bought and what it cost."""
    accepted, withheld = [], []
    for diameter in HOLES_MM:
        for position in POSITIONS:
            entry = sweep[f"{diameter:g}"][f"{position:.0f}"]["by_threshold"][f"{threshold:g}"]
            accepted += [abs(e) for e in entry["errors_m"]]
            withheld += [abs(e) for e in entry["withheld_errors_m"]]
    return {"threshold": threshold, "accepted": len(accepted), "withheld": len(withheld),
            "accepted_median_m": float(np.median(accepted)) if accepted else None,
            "accepted_worst_m": float(np.max(accepted)) if accepted else None,
            "withheld_median_m": float(np.median(withheld)) if withheld else None,
            "withheld_worst_m": float(np.max(withheld)) if withheld else None}


def positions_by_band(sweep: dict, threshold: float = OPERATING) -> dict:
    """Where the localiser holds and where it does not, by the only variable that turns out to matter: how far the
    hole is from the **farther** transmitter, which is the leg that decides how loud the quieter of the two fronts is."""
    out = {}
    for position in POSITIONS:
        errors, fronts = [], []
        placed = runs = 0
        for diameter in HOLES_MM:
            cell = sweep[f"{diameter:g}"][f"{position:.0f}"]
            entry = cell["by_threshold"][f"{threshold:g}"]
            placed += entry["placed"]
            runs += entry["runs"]
            errors += [abs(e) for e in entry["errors_m"]]
            if entry["weakest_front_sigma"] is not None:
                fronts.append(entry["weakest_front_sigma"])
        out[f"{position:.0f}"] = {
            "distance_m": position, "runs": runs, "placed": placed,
            "far_leg_m": max(position - MONITOR[0], MONITOR[1] - position),
            "median_error_m": float(np.median(errors)) if errors else None,
            "worst_error_m": float(np.max(errors)) if errors else None,
            "worst_weakest_front_sigma": float(np.min(fronts)) if fronts else None}
    return out


def fig_tradeoff(sweep: dict, reference: dict, pipe: Pipeline) -> None:
    """The curve: what a higher decision interval buys in false alarms and costs in holes nobody hears."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    colours = {8.0: "#a23a00", 12.0: "#0081be", 16.0: "#4a4a4a", 20.0: "#8a8a8a"}
    for threshold in THRESHOLDS:
        rates, delays = [], []
        for diameter in HOLES_MM:
            keys = [row[f"{p:.0f}"] for row in [sweep[f"{diameter:g}"]] for p in POSITIONS]
            detected = sum(k["by_threshold"][f"{threshold:g}"]["detected"] for k in keys)
            runs = sum(k["by_threshold"][f"{threshold:g}"]["runs"] for k in keys)
            kept = [k["by_threshold"][f"{threshold:g}"]["median_delay_s"] for k in keys
                    if k["by_threshold"][f"{threshold:g}"]["median_delay_s"] is not None]
            rates.append(detected / runs)
            delays.append(float(np.mean(kept)) if kept else np.nan)
        axes[0].plot(HOLES_MM, rates, "o-", color=colours[threshold], label=f"decision interval {threshold:g}$\\sigma$")
        axes[1].plot(HOLES_MM, delays, "o-", color=colours[threshold], label=f"{threshold:g}$\\sigma$")
    for position in POSITIONS:
        axes[1].axhline(min(position - MONITOR[0], MONITOR[1] - position) / pipe.wave_speed,
                        color="black", lw=0.7, ls=":")
    floor = minimum_detectable_leak(pipe, SENSORS, attenuation=0.305)["leak_rate"]
    head = float(np.mean([pipe.steady_head(p) for p in POSITIONS]))
    hole_mm = 1000.0 * math.sqrt(4.0 * floor / (math.pi * 0.62 * math.sqrt(2.0 * G * head)))
    for axis in axes:
        axis.axvline(hole_mm, color="black", lw=0.8, ls="--")
    axes[0].annotate(f"the transducer noise: {hole_mm:.1f} mm", xy=(hole_mm + 0.3, 0.55), fontsize=8)
    for axis in axes:
        axis.set_xlabel("hole diameter  [mm]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("fraction of the holes found")
    axes[0].set_title("what a quieter detector misses")
    axes[1].set_ylabel("median detection delay  [s]")
    axes[1].set_yscale("log")
    axes[1].annotate("dotted: the flight time from each of the three positions", xy=(8.0, 3.0), fontsize=8)
    axes[1].set_title("and how late the rest are")
    figure.suptitle("The trade-off: a quieter detector loses holes, and the ones it loses are the transducer's, "
                    "not the threshold's", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "tradeoff.png", dpi=130)
    plt.close(figure)


def fig_localisation(sweep: dict) -> None:
    """Estimated against true position, and the error as a fraction of the line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    style = {2.0: ":", 3.0: ":", 4.0: "--", 6.0: "--", 8.0: "-", 10.0: "-", 14.0: "-", 20.0: "-"}
    for diameter in HOLES_MM:
        xs, errs = [], []
        for position in POSITIONS:
            entry = sweep[f"{diameter:g}"][f"{position:.0f}"]["by_threshold"][f"{OPERATING:g}"]
            if entry["median_error_m"] is None:
                continue
            xs.append(position)
            errs.append(entry["median_error_m"])
        if xs:
            axes[0].plot(xs, errs, "o" + style[diameter], ms=4, label=f"{diameter:g} mm")
    resolution = PIPE.wave_speed * SENSORS.sample / 2.0
    axes[0].axhline(resolution, color="black", lw=0.9, ls="--")
    axes[0].annotate("one acquisition sample: $a\\,\\Delta t_s/2$ = 28.5 m", xy=(6400.0, resolution * 1.6),
                     fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("true distance of the hole along the line  [m]")
    axes[0].set_ylabel("median localisation error  [m]")
    axes[0].set_title("where the wave puts it")
    axes[0].legend(fontsize=8, title="hole", ncol=2)
    axes[0].grid(alpha=0.3)

    for diameter in HOLES_MM:
        delays, floors = [], []
        for position in POSITIONS:
            entry = sweep[f"{diameter:g}"][f"{position:.0f}"]
            got = entry["by_threshold"][f"{OPERATING:g}"]["median_delay_s"]
            if got is None:
                continue
            delays.append(got)
            floors.append(entry["delay_floor_s"])
        if delays:
            axes[1].scatter(floors, delays, s=18, label=f"{diameter:g} mm")
    span = [1.5, 9.5]
    axes[1].plot(span, span, "k--", lw=0.9)
    axes[1].set_xlim(*span)
    axes[1].annotate("the detector adds nothing to the flight time", xy=(5.4, 6.6), fontsize=8)
    axes[1].set_xlabel("floor: distance to the nearer transmitter / a  [s]")
    axes[1].set_ylabel("measured detection delay  [s]")
    axes[1].set_title("why it is late: the wave had to fly there")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, title="hole", ncol=2)
    axes[1].grid(alpha=0.3)
    figure.suptitle("Place and time of the hole, at a decision interval of 12$\\sigma$", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "localisation.png", dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer seeds")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    global FIGURES
    arguments = parser.parse_args()
    if arguments.figures:
        FIGURES = arguments.figures
    figures = FIGURES
    output = arguments.out or RESULTS / "pipeline_leak.json"
    figures.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = 3 if arguments.quick else 6
    hours = 20 if arguments.quick else 60

    started = time.time()
    report = {"pipeline": {"length_m": PIPE.length, "diameter_m": PIPE.diameter, "flow_m3_s": PIPE.flow,
                           "velocity_m_s": PIPE.velocity, "wave_speed_m_s": PIPE.wave_speed,
                           "friction_factor": PIPE.friction, "friction_head_m": PIPE.friction_head,
                           "valve_drop_m": PIPE.valve_drop},
              "sensors": {"upstream_m": SENSORS.upstream, "downstream_m": SENSORS.downstream,
                          "sigma_m": SENSORS.sigma, "bits": SENSORS.bits, "sample_s": SENSORS.sample,
                          "step_m": SENSORS.step, "flow_sigma_m3_s": SENSORS.flow_sigma}}
    report["transient"] = transient("transient.png")

    nominal = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR)
    reference = expected(nominal, SENSORS)
    report["steady_state"] = {"intact_max_drift_m": float(np.max(np.abs(nominal["monitor_head"] - nominal["steady"][
        [int(round(MONITOR[0] / nominal["dx"])), int(round(MONITOR[1] / nominal["dx"]))]])))}

    sweep = leak_sweep(nominal, reference, seeds)
    report["leaks"] = sweep
    report["localisation_gate"] = by_position(sweep)
    report["localisation_by_position"] = positions_by_band(sweep)
    fault_nominal = simulate(PIPE, n_reaches=REACHES, t_end=140.0, monitor=MONITOR, valve=valve)
    fault_leak = simulate(PIPE, n_reaches=REACHES, t_end=140.0, monitor=MONITOR, valve=valve,
                          leak=hole(10.0, 9800.0))
    report["instruments"] = instrument(expected(fault_nominal, SENSORS),
                                       observe(fault_leak, PIPE, SENSORS, seed=703),
                                       fault_nominal)
    long_nominal = simulate(PIPE, n_reaches=REACHES, t_end=QUIET, monitor=MONITOR, valve=valve)
    report["false_alarms"] = false_alarms(long_nominal, expected(long_nominal, SENSORS), hours)
    report["minimum_detectable"] = minimum_detectable_leak(
        PIPE, SENSORS, attenuation=report["transient"]["attenuation_over_9_3_km"])
    report["seconds"] = round(time.time() - started, 1)

    fig_tradeoff(sweep, reference, PIPE)
    fig_localisation(sweep)
    output.write_text(json.dumps(report, indent=1) + "\n")

    operating = report["leaks"]["10"]["9800"]["by_threshold"]["12"]
    print(f"{report['seconds']:.0f}s  hole 10 mm at 9.8 km: detected in {operating['median_delay_s']:.2f} s, "
          f"placed within {operating['median_error_m']:.0f} m; "
          f"false alarms {report['false_alarms']['by_threshold']['12']['per_hour']:.2f}/h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
