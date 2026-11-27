"""Flight safety with physics-verified dynamics: a drone that says what is wrong with it (#335).

    python examples/multirotor_flight/flight_safety.py            # the whole campaign (~10 min)
    python examples/multirotor_flight/flight_safety.py --quick    # one seed per fault (~2 min)

The vehicle is a 1.5 kg quad-X with four BEMT rotors (10x4.7-class), a pack that sags under load, drag, ground effect
and a cascaded controller. Eight faults are injected at t = 8 s, one per flight: a payload, a degraded motor, a chipped
blade, a gust, a GPS glitch, magnetometer interference, IMU vibration and a pack whose resistance has tripled.

What the example measures, and why each number is the one that matters:

1. **the residual** -- Newton-Euler on the *nominal* vehicle, evaluated against the accelerometer, the gyro and the pack
   telemetry, with the commanded rotor speeds. The controller is never told about the fault, so it re-trims and the
   residual is what is left over.
2. **the diagnosis** -- not a classifier trained on labels: the *direction* of the residual names the fault. A payload
   pushes along body ``-z``; a failing rotor points along its own arm; a GPS glitch moves the fix and nothing else.
3. **the comparison** -- three detectors (a fixed threshold, a Kalman innovation gate, a CUSUM), each tuned to the
   *same* false-alarm budget: the most sensitive setting that does not fire on the calibration flights. Then the
   detection delay is measured on flights the detector never saw. Anything else compares three different alarms.

Two honest negatives are recorded rather than hidden: a degraded motor and a chipped blade are **not** told apart by
these channels (the controller re-trims both into the same signature, and the power-per-newton ratio that separates them
in theory, 1.96 against 2.02, is inside the noise), and a 4 m/s gust is only a 3-sigma event on a 1.5 kg quad, which is
why the gust in the table is 10 m/s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

from pinneapple_physics.multirotor.residuals import CHANNELS

OUT = os.path.dirname(os.path.abspath(__file__))
FAULT_TIME = 8.0
DURATION = 20.0

#: The faults, at the size each one has to be to matter on a 1.5 kg quad. ``motor=2`` is the right rear.
CASES = {
    "payload": (("payload", 0.40),),
    "motor": (("motor", 0.70, 2),),
    "blade": (("blade", 0.12, 2),),
    "gust": (("gust", 10.0, 0, (1.0, 0.3, 0.0)),),
    "gps": (("gps", 3.0, 0, (1.0, 0.0, 0.0)),),
    "magnetometer": (("magnetometer", math.radians(25.0)),),
    "vibration": (("vibration", 1.2),),
    "battery": (("battery", 2.0),),
    # the two at the edge of what this vehicle can feel: a 0.1 kg payload is a quarter of the 0.4 kg one, and a 5 m/s
    # gust is a quarter of the drag. They are in the table because a fault every detector finds instantly says nothing
    # about which detector is better -- the difference lives where the signal is small.
    "payload-small": (("payload", 0.10),),
    "gust-small": (("gust", 5.0, 0, (1.0, 0.3, 0.0)),),
}


#: what a correct diagnosis of each case has to contain
EXPECTED = {"payload": "payload", "motor": "motor", "blade": "motor", "gust": "gust", "gps": "GPS",
            "magnetometer": "magnetometer", "vibration": "vibration", "battery": "battery",
            "payload-small": "payload", "gust-small": "gust"}


def _fault(spec):
    from pinneapple_physics.multirotor.faults import Fault

    spec = spec[0] if isinstance(spec[0], tuple) else spec
    kind = spec[0]
    if kind == "payload":
        return Fault(kind, FAULT_TIME, spec[1])
    if kind in ("motor", "blade"):
        return Fault(kind, FAULT_TIME, spec[1], motor=spec[2])
    if kind in ("gust", "gps"):
        return Fault(kind, FAULT_TIME, spec[1], direction=spec[3])
    if kind == "magnetometer":
        return Fault(kind, FAULT_TIME, spec[1])
    return Fault(kind, FAULT_TIME, spec[1])


def _out(*parts: str) -> str:
    path = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(path) or OUT, exist_ok=True)
    return path


def run_campaign(*, seeds=(3, 4, 5)) -> tuple[dict, dict, object]:
    from pinneapple_physics.multirotor.detect import (
        CusumDetector,
        KalmanDetector,
        ThresholdDetector,
        calibrate,
        diagnose,
        tune,
    )
    from pinneapple_physics.multirotor.sim import fly

    calibration_seeds = (0, 1, 2)
    print("calibration flights:", calibration_seeds, flush=True)
    healthy = [fly(duration=DURATION, seed=s) for s in calibration_seeds]
    calibration = calibrate(healthy)
    print("  channel sigmas:", {k: round(v["std"], 3) for k, v in calibration.to_dict().items()}, flush=True)

    print(f"test flights: fault-free {seeds} and {len(CASES)} faults x {len(seeds)} seeds", flush=True)
    clean = []
    for seed in seeds:
        clean.append(fly(duration=DURATION, seed=seed))
        print(f"  fault-free seed {seed} flown", flush=True)
    flights = {}
    for name, spec in CASES.items():
        flights[name] = [fly((_fault(spec),), duration=DURATION, seed=s) for s in seeds]
        print(f"  {name:13s} flown ({len(seeds)} seed(s))", flush=True)

    # the signature of each fault: the residual it leaves, in units of the fault-free spread. This is the table the
    # guide reproduces, and it is why the diagnosis can be a rule about direction instead of a fitted classifier.
    signatures = {}
    for name, runs in flights.items():
        shifts = []
        for f in runs:
            before = f.residual[(f.time > 1.5) & (f.time < FAULT_TIME)]
            after = f.residual[f.time > FAULT_TIME + 2.0]
            shifts.append((after.mean(axis=0) - before.mean(axis=0)) / np.maximum(calibration.std, 1e-9))
        signatures[name] = {c: round(float(v), 1) for c, v in zip(CHANNELS, np.mean(shifts, axis=0), strict=False)}
        print(f"  signature {name:13s} " +
              "  ".join(f"{c}={signatures[name][c]:7.1f}" for c in ("force_x", "force_z", "torque_x", "power",
                                                                    "gps", "heading", "vibration")), flush=True)

    table = {}
    for detector_class in (ThresholdDetector, KalmanDetector, CusumDetector):
        detector = tune(detector_class, calibration, healthy)
        false_alarms = float(np.mean([detector.run(f).detected for f in clean]))
        rows = {}
        for name, runs in flights.items():
            detections = [detector.run(f) for f in runs]
            delays = [(d.time - FAULT_TIME) if d.detected and d.time is not None else None for d in detections]
            found = [d for d in delays if d is not None]
            verdicts = [diagnose(f, d, calibration)["label"] for f, d in zip(runs, detections, strict=False)]
            rows[name] = {"detected": len(found) / len(delays),
                          "delay_s": float(np.mean(found)) if found else None,
                          "delay_max_s": float(np.max(found)) if found else None,
                          "labels": verdicts,
                          "correct": [EXPECTED[name] in v for v in verdicts]}
            print(f"  {detector_class.__name__:18s} {name:13s} "
                  f"detected {rows[name]['detected']:5.0%}  delay "
                  f"{'' if not found else f'{np.mean(found):5.2f} s'}", flush=True)
        table[detector_class.__name__] = {"sensitivity": detector.sensitivity, "false_alarm_rate": false_alarms,
                                          "cases": rows}
        print(f"  {detector_class.__name__}: false alarms on {len(clean)} healthy flights = {false_alarms:.0%}, "
              f"sensitivity {detector.sensitivity}", flush=True)

    result = {"issue": 335, "fault_time_s": FAULT_TIME, "duration_s": DURATION, "seeds": list(seeds),
              "calibration": calibration.to_dict(), "signatures": signatures, "detectors": table,
              "machine": {"platform": platform.platform(), "python": platform.python_version(),
                          "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count()}}
    result["figures"] = ["figures/flight.gif", "figures/residuals.png"]
    # the flights come back with the result so the caller can draw from them instead of flying them a second time
    return result, flights, calibration


def _flight_figure(payload_flight, motor_flight, calibration, gif_path: str, png_path: str) -> None:
    """The GIF the issue asks for: the trajectory, the residual and the check that failed, over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = []
    for flight, title in ((payload_flight, "payload +0.40 kg"), (motor_flight, "motor 2 at 70 %")):
        ax = axes[0] if flight is payload_flight else axes[2]
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("vertical force residual (m/s$^2$)")
        ax.axvline(FAULT_TIME, color="#a23a00", ls="--", lw=1.0)
        line, = ax.plot([], [], color="#0081be", lw=1.0)
        ax.set_xlim(0, payload_flight.time[-1])
        ax.set_ylim(-3.2, 1.2)
        panels.append((ax, line, flight))

    ax = axes[1]
    ax.set_title("trajectory (truth)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    trail, = ax.plot([], [], color="#262626", lw=1.0)
    mark, = ax.plot([], [], "o", color="#a23a00", ms=6)
    positions = np.concatenate([payload_flight.position, motor_flight.position])
    ax.set_xlim(positions[:, 0].min() - 0.5, positions[:, 0].max() + 0.5)
    ax.set_ylim(positions[:, 1].min() - 0.5, positions[:, 1].max() + 0.5)

    stride = max(len(payload_flight.time) // 220, 1)
    frames = list(range(0, len(payload_flight.time), stride))

    def update(k):
        for _ax, line, flight in panels:
            line.set_data(flight.time[:k + 1], flight.channel("force_z")[:k + 1])
        trail.set_data(payload_flight.position[:k + 1, 0], payload_flight.position[:k + 1, 1])
        mark.set_data([payload_flight.position[k, 0]], [payload_flight.position[k, 1]])
        return [line for _, line, _ in panels] + [trail, mark]

    animation = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=25))

    fig2, axes2 = plt.subplots(2, 1, figsize=(11.0, 6.4), constrained_layout=True, sharex=True)
    for axis, channel in zip(axes2, ("force_z", "power"), strict=False):
        index = CHANNELS.index(channel)
        for flight, label in ((payload_flight, "payload +0.40 kg"), (motor_flight, "motor 2 at 70 %")):
            axis.plot(flight.time, (flight.residual[:, index] - calibration.mean[index]) /
                      max(calibration.std[index], 1e-9), lw=0.8,
                      color="#a23a00" if "motor" in label else "#0081be", label=label)
        axis.axhline(0.0, color="#999999", lw=0.6)
        axis.axvline(FAULT_TIME, color="#262626", ls="--", lw=1.0, label="fault injected")
        axis.set_ylabel(f"{channel} residual (sigma)")
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    axes2[-1].set_xlabel("time (s)")
    fig2.savefig(png_path, dpi=130)
    plt.close(fig)
    plt.close(fig2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="one seed per fault instead of three")
    ap.add_argument("--out", default=_out("results", "flight_safety.json"))
    args = ap.parse_args(argv)

    started = time.time()
    result, flights, calibration = run_campaign(seeds=(3,) if args.quick else (3, 4, 5))
    result["seconds"] = time.time() - started
    # the numbers are written before the pictures: the campaign is ten minutes of flying, and the figures are the last
    # thing that should be able to fail it
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.out}", flush=True)
    _flight_figure(flights["payload"][0], flights["motor"][0], calibration, _out("figures", "flight.gif"),
                   _out("figures", "residuals.png"))
    print("wrote the figures", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
