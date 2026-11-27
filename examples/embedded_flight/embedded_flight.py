"""The drone in the graph, measured: the multirotor's dynamics, its Newton-Euler residual and its sensor check in one
ONNX file, with the latency, the threads, fp32 against int8, and the two flags doing their job (#350).

    python examples/embedded_flight/embedded_flight.py        # ~40 s

What it produces
----------------
1. **the graph** -- ``flight.onnx``: seven inputs, thirteen outputs, no ``If``/``Loop``/``Scan``, and the BEMT rotor
   table baked in as knot weights;
2. **parity** -- four open-loop flights of the numpy airframe (:mod:`pinneapple_physics.multirotor`), and the same
   states through the graph: the 18 numbers of the next state against ``Multirotor.step``, the seven Newton-Euler
   channels against ``ResidualModel``;
3. **the flags** -- a tolerance calibrated on one healthy flight, a false-alarm rate measured on a *different* healthy
   flight, and a detection rate on a payload and on a loose IMU;
4. **the numbers that decide whether it flies** -- latency against threads (1, 2, 4), fp32 against int8, and what
   quantisation costs the residual.

What it does not produce, and why
---------------------------------
There is **no ARM board** on this machine -- no Raspberry Pi, no Jetson -- so the latency below is the Mac's ARM core
(``arm64``, measured and printed), which is the same *kind* of number a board gives and not the same number. Nothing
here was measured on a board, nothing was measured against PX4, and the ROS 2 adapter in
:mod:`pinneapple_physics.embedded_bridge` is marked as untested for the same reason. What the measurements do settle
is the shape of the budget: whether the graph is microseconds of arithmetic or milliseconds of something else.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import deque

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
DT = 0.002                      # 500 Hz, the rate a flight controller runs its estimator at
STEPS = 400                     # 0.8 s of flight per tape: long enough for the states to move


def _out(name: str) -> str:
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path) or OUT, exist_ok=True)
    return path


def hover_command(spec, omega: float | None = None) -> np.ndarray:
    """The command that holds the vehicle up -- and ``Airframe.hover_omega`` is 60 BEMT solves, so pass one in."""
    af = spec.airframe
    return np.full(4, (af.hover_omega() if omega is None else omega) / af.omega_max)


def command_at(hover: np.ndarray, t: float) -> np.ndarray:
    """Hover, then a differential on two motors: the vehicle rolls and translates, so the tape is not a fixed point."""
    command = hover + np.array([0.02, 0.02, -0.01, -0.01]) if t >= 0.3 else hover.copy()
    return np.clip(command, 0.0, 1.0)


def tape(spec, table=None, *, faults=(), seed: int = 0, steps: int = STEPS):
    """Fly the numpy airframe open-loop and record what a flight computer would have at every step.

    The graph is given exactly what the vehicle has: the estimated state, the command it sent, the three IMU channels,
    the pack power and the accelerometer buffer. Nothing here reads the fault, and neither does the graph.
    """
    from pinneapple_physics.multirotor import (Multirotor, ResidualModel, SensorObservatory, State, apply_faults)
    from pinneapple_physics.multirotor.faults import faulty_airframe

    af = spec.airframe
    table = table if table is not None else spec.table()
    observatory = SensorObservatory(seed=seed)
    residual_model = ResidualModel(af)
    hover = af.hover_omega()                                # one call: it is a bisection over 60 BEMT solves
    hover_command_of = hover_command(spec, hover)
    state = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, hover))
    residual_model.reset()
    # the accelerometer buffer the graph's sensor check needs: a flight computer's ring buffer, pre-filled so the
    # frame is always the size the graph was exported with
    history: deque = deque(maxlen=spec.history)
    previous_gyro = np.zeros(3)
    rows = []
    for k in range(steps):
        t = k * spec.dt
        current = apply_faults(faults, t, base_airframe=af, base_resistance_ohm=af.battery.resistance_ohm,
                               rotor=af.rotor, omega_hover=hover)
        model = Multirotor(faulty_airframe(af, mass_kg=current.mass_kg,
                                           battery_resistance_ohm=current.battery_resistance_ohm),
                           table=table, thrust_gain=current.thrust_gain, torque_gain=current.torque_gain)
        command = command_at(hover_command_of, t)
        telemetry = model.measure(state)
        reading = observatory.measure(state, model, command, telemetry, current)
        # the graph is given the gyro sample from *before* this one: that is what an angular acceleration is, and on
        # the first step there is no before, which is exactly what ResidualModel.reset() means by no previous gyro
        gyro_previous = np.asarray(reading.gyro if k == 0 else previous_gyro, np.float32)
        reference = residual_model(reading, spec.dt)
        previous_gyro = np.asarray(reading.gyro, float)
        if not history:
            history.extend([np.asarray(reading.accel, np.float32)] * spec.history)
        else:
            history.append(np.asarray(reading.accel, np.float32))
        following, _ = model.step(state, command, spec.dt)
        rows.append({
            "state": np.concatenate([state.position, state.velocity, state.attitude, state.rates, state.rotors,
                                     [state.soc]]).astype(np.float32),
            "command": command.astype(np.float32),
            "accel": np.asarray(reading.accel, np.float32),
            "gyro": np.asarray(reading.gyro, np.float32),
            "gyro_previous": gyro_previous,
            "power": np.array([reading.power], np.float32),
            "history": np.asarray(list(history), np.float32),
            "next_state": np.concatenate([following.position, following.velocity, following.attitude, following.rates,
                                          following.rotors, [following.soc]]),
            "channels": reference.vector()[:7],
        })
        state = following
    return rows


def replay(runtime, rows):
    """Every step of a tape through the graph, once."""
    outs = []
    for row in rows:
        outs.append(runtime.step(row["state"], row["command"], row["accel"], row["gyro"], row["gyro_previous"],
                                 row["power"], row["history"]))
    return outs


def parity(rows, outs):
    """The two things that must agree with the library: the next state, and the seven residual channels."""
    worst_state, scale = 0.0, 0.0
    for row, out in zip(rows, outs, strict=True):
        expected = row["next_state"]
        scale = max(scale, float(np.max(np.abs(expected))))
        worst_state = max(worst_state, float(np.max(np.abs(out.next_state - expected) / np.maximum(np.abs(expected), 1.0))))
    force = max(float(np.max(np.abs(out.residual_force - row["channels"][:3])))
                for row, out in zip(rows, outs, strict=True))
    torque = max(float(np.max(np.abs(out.residual_torque - row["channels"][3:6])))
                 for row, out in zip(rows, outs, strict=True))
    power = max(abs(out.residual_power - row["channels"][6]) for row, out in zip(rows, outs, strict=True))
    return {"next_state_relative": worst_state, "next_state_scale": scale,
            "force_channel_abs": force, "torque_channel_abs": torque, "power_channel_abs": float(power)}


def false_alarms(outs) -> float:
    return float(np.mean([o.off_physics for o in outs]))


def calibrate(spec, outs, *, percentile: float = 99.5) -> dict[str, float]:
    """The limits, taken from a healthy flight instead of from a habit.

    The graph's residual is ``max(rms(force)/F, rms(torque)/T)``: one number that scales exactly with ``1/F`` and
    ``1/T``, so the calibration is *one* scalar -- the percentile of the residual the healthy flight reports -- applied
    to both tolerances. Calibrating the two channels separately would put the false-alarm rate of their *maximum* at
    twice the per-channel rate, which is the kind of arithmetic that turns "0.5 %" into "2 %" in a report.
    """
    residual = np.array([o.residual for o in outs])
    factor = max(float(np.percentile(residual, percentile)), 1e-9)
    vibration = np.array([o.vibration for o in outs])
    return {"force_tolerance": float(spec.force_limit * factor),
            "torque_tolerance": float(spec.torque_limit * factor),
            "vibration_tolerance": float(max(np.percentile(vibration, percentile), 1e-3)),
            "factor": factor}


def quantisation(fp32_path, spec, rows):
    """fp32 against int8 on both graphs that matter: the flight one, and a convolutional one for contrast."""
    import torch
    from pinneapple_physics.embedded import EmbeddedSpec, export, quantize

    from pinneapple_physics.embedded_flight import FlightRuntime, graph_summary

    int8_path = quantize(fp32_path, _out("flight_int8.onnx"))
    fp32 = FlightRuntime(fp32_path, threads=1)
    int8 = FlightRuntime(int8_path, threads=1)
    row = rows[0]
    feed = FlightRuntime.feed(row["state"], row["command"], row["accel"], row["gyro"], row["gyro_previous"],
                              row["power"], row["history"])
    reference = fp32.step(row["state"], row["command"], row["accel"], row["gyro"], row["gyro_previous"], row["power"],
                          row["history"])
    quantised = int8.step(row["state"], row["command"], row["accel"], row["gyro"], row["gyro_previous"], row["power"],
                          row["history"])

    # the contrast: a graph with matrices in it, which is what dynamic quantisation is for
    torch.manual_seed(0)
    convolutional = torch.nn.Sequential(torch.nn.Conv2d(2, 8, 3, padding=1, padding_mode="circular"),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(8, 1, 3, padding=1, padding_mode="circular")).eval()
    conv_spec = EmbeddedSpec(state_shape=(2, 32, 64), tolerance=0.05)
    conv_fp32 = export(convolutional, conv_spec, _out("conv_fp32.onnx"))
    conv_int8 = quantize(conv_fp32, _out("conv_int8.onnx"))
    conv_state = torch.randn(2, 32, 64)
    from pinneapple_physics.embedded import EmbeddedRuntime
    conv_a = EmbeddedRuntime(conv_fp32, threads=1).step(conv_state.numpy(), np.zeros(1))
    conv_b = EmbeddedRuntime(conv_int8, threads=1).step(conv_state.numpy(), np.zeros(1))
    del feed
    return {
        "flight": {"fp32_bytes": int(fp32_path.stat().st_size), "int8_bytes": int(int8_path.stat().st_size),
                   "new_ops": sorted(set(graph_summary(int8_path).op_types) - set(graph_summary(fp32_path).op_types)),
                   "next_state_max_abs": float(np.max(np.abs(quantised.next_state - reference.next_state))),
                   "residual_fp32": reference.residual, "residual_int8": quantised.residual,
                   "path": str(int8_path)},
        "convolutional": {"op_types": sorted(set(graph_summary(conv_int8).op_types)
                                             - set(graph_summary(conv_fp32).op_types)),
                          "fp32_bytes": int(conv_fp32.stat().st_size), "int8_bytes": int(conv_int8.stat().st_size),
                          "prediction_max_abs": float(np.max(np.abs(conv_b.prediction - conv_a.prediction))),
                          "residual_fp32": conv_a.residual, "residual_int8": conv_b.residual},
    }


def figure(series, tolerance, latency, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    ax = axes[0]
    colours = {"healthy (other seed)": "#0081be", "payload +0.6 kg": "#a23a00", "loose IMU": "#6b6b6b"}
    for name, values in series.items():
        ax.plot(np.arange(len(values)) * DT * 1000.0, values, lw=1.1, alpha=0.9, label=name,
                color=colours.get(name))
    ax.axhline(1.0, color="#262626", ls="--", lw=1.2)
    ax.set_ylim(top=1.42 * max(float(np.max(v)) for v in series.values()))
    ax.set_xlabel("time in the flight (ms)")
    ax.set_ylabel("residual / calibrated tolerance")
    ax.set_title(f"the flag trips at 1.0 (tolerance {tolerance:.3f} m/s^2, calibrated)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    widths, positions = 0.35, np.arange(len(latency))
    for offset, variant, colour in ((-widths / 2, "fp32", "#0081be"), (widths / 2, "int8", "#a23a00")):
        values = [latency[label][variant]["median_ms"] for label in latency]
        spread = [latency[label][variant]["spread_ms"] for label in latency]
        ax.bar(positions + offset, values, width=widths, yerr=spread, capsize=3, label=variant, color=colour,
               alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels([label.split()[0] for label in latency])
    ax.set_xlabel("intra-op threads")
    ax.set_ylim(0.0, max(0.4, 1.3 * max(v["median_ms"] for label in latency for v in latency[label].values()
                                         if isinstance(v, dict))))
    ax.set_ylabel("median latency per step (ms)")
    ax.set_title("one step: dynamics, residual, sensor check and the decision")
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_out("embedded_flight.json"))
    args = ap.parse_args(argv)

    from pinneapple_physics.embedded_flight import (FlightRuntime, FlightSpec, export_flight, export_probe,
                                                    graph_summary)
    from pinneapple_physics.multirotor import Fault

    spec = FlightSpec(dt=DT)
    path = export_flight(spec, _out("flight.onnx"))
    summary = graph_summary(path)
    print(f"graph: {summary.bytes} bytes, {len(summary.op_types)} op types, straight line = {summary.is_straight_line}")
    print(f"  in  {list(summary.inputs)}")
    print(f"  out {list(summary.outputs)}")

    runtime = FlightRuntime(path, threads=1)
    table = spec.table()                                    # one BEMT tabulation, shared by every tape
    healthy = tape(spec, table, seed=0)
    outs = replay(runtime, healthy)
    agreement = parity(healthy, outs)
    print(f"parity over {len(healthy)} steps: next state to {agreement['next_state_relative']:.2%} relative, "
          f"force channel to {agreement['force_channel_abs']:.4f} m/s^2, torque to "
          f"{agreement['torque_channel_abs']:.4f} rad/s^2, power to {agreement['power_channel_abs']:.4f} W")

    limits = calibrate(spec, outs)
    print(f"calibrated on this flight (99.5th percentile): residual factor {limits['factor']:.3f} -> "
          f"force {limits['force_tolerance']:.3f} m/s^2, torque {limits['torque_tolerance']:.2f} rad/s^2, "
          f"vibration {limits['vibration_tolerance']:.3f} m/s^2")
    calibrated = FlightSpec(dt=DT, **{k: v for k, v in limits.items() if k != "factor"})
    calibrated_path = export_flight(calibrated, _out("flight_calibrated.onnx"))
    runtime = FlightRuntime(calibrated_path, threads=1)

    validation = {"healthy (other seed)": replay(runtime, tape(spec, table, seed=1))}
    faults = {"payload +0.6 kg": (Fault("payload", 0.0, magnitude=0.6),),
              "loose IMU": (Fault("vibration", 0.0, magnitude=2.0),)}
    for name, injected in faults.items():
        validation[name] = replay(runtime, tape(spec, table, faults=injected, seed=0))
    series = {name: np.array([o.residual for o in outs_]) for name, outs_ in validation.items()}
    for name, values in series.items():
        flagged = float(np.mean([o.off_physics for o in validation[name]]))
        suspect = float(np.mean([o.suspect for o in validation[name]]))
        print(f"  {name:22s}: off_physics {flagged:5.0%} of the flight, suspect {suspect:5.0%}")

    feed = FlightRuntime.feed(*[healthy[0][k] for k in ("state", "command", "accel", "gyro", "gyro_previous",
                                                        "power", "history")])
    quant = quantisation(calibrated_path, spec, healthy)
    int8_path = quant["flight"].pop("path")
    latency = {}
    for threads in (1, 2, 4):
        entry = {"fp32": [], "int8": []}
        for variant, graph in (("fp32", calibrated_path), ("int8", int8_path)):
            # three *sessions*, because one batch of 300 calls cannot tell a difference from the spread between
            # runs, and a session is what a flight computer builds once
            entry[variant] = [FlightRuntime(graph, threads=threads).benchmark(feed, n=300)["median_ms"]
                              for _ in range(3)]
        latency[str(threads)] = {v: {"median_ms": float(np.median(times)), "spread_ms": float(np.ptp(times)),
                                     "hz": 1000.0 / float(np.median(times))} for v, times in entry.items()}
        a, b = latency[str(threads)]["fp32"], latency[str(threads)]["int8"]
        latency[str(threads)]["difference_inside_spread"] = bool(
            abs(a["median_ms"] - b["median_ms"]) <= max(a["spread_ms"], b["spread_ms"]))
        fp32_, int8_ = latency[str(threads)]["fp32"], latency[str(threads)]["int8"]
        same = "inside the run-to-run spread" if latency[str(threads)]["difference_inside_spread"] else "outside it"
        print(f"  {threads} thread(s): fp32 {fp32_['median_ms']:.3f} ms ({fp32_['hz']:.0f} Hz) "
              f"+-{fp32_['spread_ms']:.3f}, int8 {int8_['median_ms']:.3f} ms ({int8_['hz']:.0f} Hz) "
              f"+-{int8_['spread_ms']:.3f} -- the difference is {same}")
    print(f"int8 on the flight graph: {quant['flight']['fp32_bytes']} -> {quant['flight']['int8_bytes']} bytes, "
          f"new op types {quant['flight']['new_ops'] or 'none'}, next state moved by "
          f"{quant['flight']['next_state_max_abs']:.3g}, residual {quant['flight']['residual_fp32']:.6f} -> "
          f"{quant['flight']['residual_int8']:.6f}")
    print(f"int8 on a convolutional graph (what quantisation is for): {quant['convolutional']['op_types']}, "
          f"{quant['convolutional']['fp32_bytes']} -> {quant['convolutional']['int8_bytes']} bytes, prediction moved by "
          f"{quant['convolutional']['prediction_max_abs']:.4f}, residual "
          f"{quant['convolutional']['residual_fp32']:.4f} -> {quant['convolutional']['residual_int8']:.4f}")

    breaks = export_probe(_out("probe"))
    for entry in breaks:
        print(f"  does not export -- {entry['piece']}: {entry['result']}")
        print(f"      {entry['detail']}")

    plot = figure(series, limits["force_tolerance"], {f"{k} thread(s)": v for k, v in latency.items()},
                  _out("figures/residual_and_latency.png"))
    print(f"wrote {plot}")

    result = {
        "issue": 350, "graph": summary.to_dict(), "steps": len(healthy), "dt": DT,
        "parity": agreement, "calibration": limits,
        "flags": {name: {"off_physics": float(np.mean([o.off_physics for o in outs_])),
                         "suspect": float(np.mean([o.suspect for o in outs_])),
                         "median_residual": float(np.median([o.residual for o in outs_]))}
                  for name, outs_ in validation.items()},
        "latency": latency, "quantisation": quant, "unexportable": breaks,
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor() or platform.machine(), "python": platform.python_version(),
                    "cpus": os.cpu_count(),
                    "arm_board": None,
                    "arm_board_note": "no Raspberry Pi, Jetson or PX4 on this machine: the latency is the Mac's ARM "
                                      "core, measured here, and is not a board measurement"},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
