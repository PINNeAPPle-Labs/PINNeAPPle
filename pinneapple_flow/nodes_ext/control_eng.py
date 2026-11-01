"""Control-engineering and naval nodes on :mod:`pinneapple_systems.control` and :mod:`pinneapple_simulation.naval`:

* linear analysis of a state-space model and of an open loop (eigenvalues, margins);
* PID tuning rules;
* submarine depth-keeping (instability, command architecture, tuning);
* ship initial stability (GM, free surface, righting arm, roll period).

Simulation only: no node has a path to real equipment.
"""
from __future__ import annotations

import math

import numpy as np

from ..registry import Param, Port, node


def _num(v):
    return float(v) if v is not None and np.isfinite(v) else None


@node("control.linear_analysis", "Linear analysis: stability, controllability, loop margins", category="Systems",
      outputs=[Port("metrics", "metrics")],
      params=[Param("A", "list", [[0.0, 1.0], [-4.0, -0.8]], help="state matrix (1/s), list of rows"),
              Param("B", "list", [[0.0], [1.0]], help="input matrix, list of rows"),
              Param("C", "list", [[1.0, 0.0]], help="output matrix, list of rows"),
              Param("loop_num", "list", [1.0], help="open-loop numerator in s, highest power first"),
              Param("loop_den", "list", [1.0, 3.0, 2.0, 0.0], help="open-loop denominator in s"),
              Param("delay", "float", 0.0, minimum=0.0, help="loop transport delay (s)")],
      calls="pinneapple_systems.control.stability / controllability / observability / loop_margins")
def linear_analysis(ctx, inputs, params):
    """Eigenvalues with growth rates and doubling times, controllability and observability of (A, B, C); gain,
    phase and delay margins and the peak sensitivity of the open loop num/den (with a delay)."""
    from pinneapple_systems.control import controllability, loop_margins, observability, stability
    A = np.asarray(params["A"], float)
    st = stability(A)
    ctx.metric("classification", st["classification"])
    ctx.metric("max_real_1_per_s", st["max_real"], "largest real part of the eigenvalues (1/s)")
    rows = []
    for m in st["modes"]:
        rows.append([f"{m['s'].real:.6g}{m['s'].imag:+.6g}j", m["damping_ratio"], m.get("doubling_time"),
                     m.get("half_life")])
    ctx.table("modes", ["eigenvalue (1/s)", "damping ratio", "doubling time (s)", "half-life (s)"], rows)
    if params.get("B"):
        c = controllability(A, params["B"])
        ctx.metric("controllable", c["controllable"])
        ctx.metric("controllability_condition", c["condition_number"])
    if params.get("C"):
        o = observability(A, params["C"])
        ctx.metric("observable", o["observable"])
    if params.get("loop_num") and params.get("loop_den"):
        mg = loop_margins((params["loop_num"], params["loop_den"]), delay=float(params["delay"]))
        ctx.metric("gain_margin_db", _num(mg["gain_margin_db"]), "gain margin (dB)")
        ctx.metric("phase_margin_deg", _num(mg["phase_margin"]), "phase margin (deg)")
        ctx.metric("delay_margin_s", _num(mg["delay_margin"]), "delay margin (s)")
        ctx.metric("ms", _num(mg["ms"]), "peak sensitivity (-)")
    ctx.summary(f"{st['classification']}; largest real part {st['max_real']:.4g} 1/s")
    return {"metrics": ctx.report["metrics"]}


@node("control.pid_tuning", "PID tuning rules (SIMC, IMC, Ziegler-Nichols, pole placement)", category="Systems",
      outputs=[Port("metrics", "metrics")],
      params=[Param("method", "enum", "simc", choices=["simc", "imc_pi", "imc_pid", "zn_reaction",
                                                       "pole_placement"]),
              Param("K", "float", 2.0, help="FOPDT gain (output per input)"),
              Param("tau", "float", 10.0, minimum=0.0, help="FOPDT time constant (s)"),
              Param("theta", "float", 1.0, minimum=0.0, help="FOPDT dead time (s)"),
              Param("m", "float", 1.0, help="pole placement: inertia of m y'' + c y' + k y = b u"),
              Param("c", "float", 0.0, help="pole placement: damping"),
              Param("k", "float", -1.0, help="pole placement: stiffness (negative: unstable plant)"),
              Param("b", "float", 1.0, help="pole placement: input gain"),
              Param("wn", "float", 2.0, minimum=0.0, help="pole placement: natural frequency (rad/s)"),
              Param("zeta", "float", 0.7, minimum=0.0, help="pole placement: damping ratio")],
      calls="pinneapple_systems.control.tuning")
def pid_tuning(ctx, inputs, params):
    """Parallel-form PID gains from a FOPDT model (SIMC, IMC, Ziegler-Nichols reaction curve), or by placing the
    closed-loop poles of a second-order plant (stable or unstable)."""
    from pinneapple_systems.control import (
        imc_pid,
        place_pid_second_order,
        simc_pid,
        ziegler_nichols,
    )
    meth = params["method"]
    K, tau, theta = float(params["K"]), float(params["tau"]), float(params["theta"])
    if meth == "simc":
        g = simc_pid(K, tau, theta)
    elif meth in ("imc_pi", "imc_pid"):
        g = imc_pid(K, tau, theta, kind=meth.split("_")[1])
    elif meth == "zn_reaction":
        g = ziegler_nichols("pid", K=K, tau=tau, theta=theta)
    else:
        g = place_pid_second_order(float(params["m"]), float(params["c"]), float(params["k"]), float(params["b"]),
                                   float(params["wn"]), float(params["zeta"]))
        ctx.table("closed-loop poles", ["pole (1/s)"], [[f"{p.real:.6g}{p.imag:+.6g}j"] for p in g["poles"]])
    for key in ("kp", "ki", "kd"):
        ctx.metric(key, float(g[key]))
    ctx.summary(f"{meth}: kp {g['kp']:.4g}, ki {g['ki']:.4g}, kd {g['kd']:.4g}")
    return {"metrics": ctx.report["metrics"]}


@node("naval.submarine_depth", "Submarine depth-keeping: instability, what the controller commands, tuning",
      category="Systems", outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("disturbance_kg", "float", 1500.0, help="extra mass that appears (kg)"),
              Param("t_disturbance", "float", 60.0, minimum=0.0, help="time of the disturbance (s)"),
              Param("t_end", "float", 1800.0, minimum=1.0, help="duration (s)"),
              Param("kp", "float", 500.0, help="PID gain, kg of target tank mass per m"),
              Param("ki", "float", 5.0, help="PID integral gain, kg per (m s)"),
              Param("kd", "float", 27500.0, help="PID derivative gain, kg per (m/s)"),
              Param("kh", "float", 8e-6, help="hull compressibility (1/m)"),
              Param("kw", "float", 4.6e-6, help="water compressibility (1/m)"),
              Param("pump", "float", 50.0, minimum=0.0, help="pump flow limit (kg/s)")],
      calls="pinneapple_simulation.naval.submarine.depth_study")
def submarine_depth(ctx, inputs, params):
    """Open loop, PID commanding the target tank mass (pump as a rate-limited servo) and the same PID commanding the
    pump flow, with the eigenvalues of each closed loop; plus gains placed on the linear model. Defaults are the
    illustrative parameters of :data:`pinneapple_simulation.naval.ILLUSTRATIVE`."""
    from dataclasses import replace

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_simulation.naval import ILLUSTRATIVE, depth_study
    prm = replace(ILLUSTRATIVE, kh=float(params["kh"]), kw=float(params["kw"]), pump=float(params["pump"]))
    r = depth_study(prm, gains={k: float(params[k]) for k in ("kp", "ki", "kd")},
                    disturbance_kg=float(params["disturbance_kg"]), t_disturbance=float(params["t_disturbance"]),
                    t_end=float(params["t_end"]))
    lin = r["linear"]
    ctx.metric("open_loop_doubling_time_s", _num(lin["doubling_time"]), "doubling time of the depth instability (s)")
    ctx.metric("open_loop_stopped", r["open_loop"]["stopped"] or "no")
    ctx.metric("open_loop_t_stop_s", r["open_loop"]["t_stop"])
    rows = r["architectures"] + r["tuned"]["rows"]
    ctx.table("architectures", ["case", "linear loop stable", "max real (1/s)", "z min (m)", "z max (m)",
                                "final tank (kg)", "stopped"],
              [[x["name"], x["stable_linear"], _num(x["max_real"]), x["y_min"], x["y_max"], x["final_trim"],
                x["stopped"] or ""] for x in rows])
    for x in rows:
        ctx.metric(f"{x['name']}.z_max_m", x["y_max"])
        ctx.metric(f"{x['name']}.stable_linear", x["stable_linear"])
    for k, v in r["tuned"]["gains"].items():
        ctx.metric(f"tuned.{k}", v)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, run in r["runs"].items():
        ax[0].plot(run["t"], run["y"], label=name)
        ax[1].plot(run["t"], run["applied"], label=name)
    ax[0].set_ylabel("depth (m)")
    ax[0].invert_yaxis()
    ax[0].set_ylim(min(130.0, prm.z_limit), 0.0)
    ax[1].set_ylabel("trim tank (kg)")
    ax[1].set_xlabel("time (s)")
    ax[0].legend(fontsize=7)
    f = ctx.figure(fig, "submarine_depth", "depth and trim-tank mass after the disturbance")
    best = r["tuned"]["rows"][0]
    ctx.summary(f"open loop: {r['open_loop']['stopped'] or 'holds'}; target-mass PID z {rows[0]['y_min']:.2f}-"
                f"{rows[0]['y_max']:.2f} m; pump-flow PID: {rows[1]['stopped'] or 'holds'}; placed gains z max "
                f"{best['y_max']:.2f} m")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("naval.ship_stability", "Ship initial stability: GM, free surface, righting arm, roll period",
      category="Systems", outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("L", "float", 60.0, minimum=0.0, help="barge length (m)"),
              Param("B", "float", 12.0, minimum=0.0, help="beam (m)"),
              Param("T", "float", 3.0, minimum=0.0, help="draught (m)"),
              Param("KG", "float", 4.0, help="centre of gravity above the keel (m)"),
              Param("tanks", "list", [{"l": 10.0, "b": 12.0, "rho": 1000.0}],
                    help="slack tanks: [{l, b, rho}] (m, m, kg/m^3)")],
      calls="pinneapple_simulation.naval.hydrostatics")
def ship_stability(ctx, inputs, params):
    """Box-barge hydrostatics, the free-surface correction of slack tanks, the corrected GM against the 0.15 m
    criterion, the wall-sided righting-arm curve and the natural roll period."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_simulation.naval import box_barge, initial_stability, righting_arm
    h = box_barge(float(params["L"]), float(params["B"]), float(params["T"]))
    st = initial_stability(h["KB"], h["BM"], float(params["KG"]), h["volume"], B=float(params["B"]),
                           tanks=params.get("tanks") or ())
    for k in ("GM_solid", "fsc", "GM", "roll_period"):
        ctx.metric(f"{k}", _num(st[k]))
    ctx.metric("stable", st["stable"])
    ctx.metric("meets_min_gm", st["meets_min_gm"])
    phi = np.linspace(0, 25, 51)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, righting_arm(st["GM"], phi, BM=h["BM"]), label="wall-sided")
    ax.plot(phi, righting_arm(st["GM"], phi), "--", label="GM sin(phi)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="heel (deg)", ylabel="GZ (m)")
    ax.legend()
    f = ctx.figure(fig, "righting_arm", "righting arm with the free-surface correction")
    tp = st["roll_period"]
    ctx.summary(f"GM {st['GM']:.3f} m ({'stable' if st['stable'] else 'unstable'}); roll period "
                f"{tp:.1f} s" if math.isfinite(tp) else f"GM {st['GM']:.3f} m: unstable upright")
    return {"metrics": ctx.report["metrics"], "figure": f}
