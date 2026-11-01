"""Submarine depth-keeping: why the depth equilibrium is unstable, and why what the controller commands matters as
much as its gains.

    python examples/control/submarine_depth.py            # about half a minute
    python examples/control/submarine_depth.py --quick    # 300 s of simulated time, no robustness samples

The parameters are the illustrative ones of ``pinneapple_simulation.naval.ILLUSTRATIVE``, not a real boat.
The example writes ``results/submarine_depth.json`` and the figures in ``results/``:

1. **Hydrostatics.** The hull is more compressible than the water, so buoyancy falls with depth. The linear model
   has a real unstable pole, and this prints its doubling time.
2. **Open loop.** A 1500 kg extra mass at t = 60 s drives the depth to the 600 m study limit.
3. **PID commanding the target tank mass.** The pump is a rate-limited servo (50 kg/s), and the PID holds depth.
4. **The same PID commanding the pump flow.** The tank mass now integrates the command. The closed-loop
   eigenvalues show the loop is unstable: with no linear damping at rest, the characteristic polynomial has no
   ``s^3`` term, and no PID gains can fix that.
5. **Gains placed on the linear model.** Pole placement for the unstable plant, then simulated.
6. **Tuning by optimisation.** Optimisation under a pump-travel constraint, starting from the example's gains.
7. **Robustness.** Monte Carlo over the hull compressibility (+-30 %) and the drag (/3 to x3).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_simulation.naval import (  # noqa: E402
    ILLUSTRATIVE,
    ILLUSTRATIVE_GAINS,
    architectures,
    depth_study,
)
from pinneapple_simulation.naval.submarine import SubmarineDepth  # noqa: E402
from pinneapple_systems.control import simulate_loop, tune_by_optimization  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items() if k != "runs"}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, complex):
        return [v.real, v.imag]
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_end = 300.0 if a.quick else 1800.0
    r = depth_study(t_end=t_end, robustness_samples=0 if a.quick else 32)
    p = ILLUSTRATIVE
    lin = r["linear"]
    print(f"dB/dz = {lin['k']:.1f} N/m (negative: unstable); growth rate {lin['growth_rate']:.5f} 1/s, "
          f"doubling time {lin['doubling_time']:.1f} s")
    ol = r["open_loop"]
    print(f"open loop: {ol['stopped'] or 'holds'} at t = {ol['t_stop']:.1f} s")
    for row in r["architectures"] + r["tuned"]["rows"]:
        print(f"{row['name']:24s} linear loop {'stable' if row['stable_linear'] else 'UNSTABLE'} "
              f"(max Re {row['max_real']:+.5f} 1/s); z {row['y_min']:.2f}-{row['y_max']:.2f} m; tank "
              f"{row['final_trim']:.0f} kg; {row['stopped'] or 'ran to the end'}")
    print("placed gains:", {k: round(v, 3) for k, v in r["tuned"]["gains"].items()})

    # tuning by optimisation: IAE of depth under a cap on pump travel, from the example's gains
    sub = SubmarineDepth(p)

    def dist(t):
        return 1500.0 if t >= 60 else 0.0

    def run(g):
        c, act = architectures(p, g)["pid_target_mass"]
        s = simulate_loop(sub.plant(), c, act, p.z_ref, t_end, 0.25, disturbance=dist, method="sequential_euler",
                          u0=p.trim0, stop=lambda x, t: "limit" if not (0 < x[1] < p.z_limit) else None)
        if s["stopped"]:
            return {"iae": float("inf"), "pump_travel": float("inf")}
        return {"iae": s["iae"], "pump_travel": float(np.sum(np.abs(np.diff(s["applied"])))),
                "z_max": s["y_max"]}

    base = run(ILLUSTRATIVE_GAINS)
    opt = tune_by_optimization(run, ILLUSTRATIVE_GAINS, {"kp": (10, 1e4), "ki": (0.01, 100), "kd": (1e3, 2e5)},
                               objective="iae", constraints={"pump_travel": 1.2 * base["pump_travel"]},
                               maxiter=12 if a.quick else 40)
    r["optimized"] = {"gains": opt["gains"], "metrics": opt["metrics"], "initial_metrics": opt["initial_metrics"],
                      "improved": opt["improved"], "evaluations": opt["evaluations"]}
    print(f"optimised gains (IAE {opt['initial_metrics']['iae']:.1f} -> {opt['metrics']['iae']:.1f} m s, "
          f"pump travel <= 1.2x): {({k: round(v, 3) for k, v in opt['gains'].items()})}")
    if "robustness" in r:
        rb = r["robustness"]
        print(f"robustness ({rb['n']} samples, kh +-30 %, drag /3..x3): {rb['fraction_stable']:.0%} stable; "
              f"excursion 95 % quantile {rb['quantiles']['excursion_m'][0.95]:.2f} m")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    labels = {"open_loop": "no control", "pid_target_mass": "PID -> target tank mass",
              "pid_pump_flow": "same PID -> pump flow", "pid_target_mass_tuned": "placed gains -> target mass"}
    for name, run_ in r["runs"].items():
        ax[0].plot(run_["t"], run_["y"], label=labels.get(name, name))
        ax[1].plot(run_["t"], run_["applied"] / 1000.0, label=labels.get(name, name))
    ax[0].set_ylim(130, 0)
    ax[0].set_ylabel("depth (m)")
    ax[1].set_ylabel("trim tank (t)")
    ax[1].set_xlabel("time (s)")
    ax[0].axvline(60, color="0.6", lw=0.8)
    ax[0].legend(fontsize=8, loc="center right")
    ax[0].set_title("1500 kg extra mass at t = 60 s (illustrative parameters)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "submarine_depth.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(a.out, "submarine_depth.json"), "w") as f:
        json.dump(_plain(r), f, indent=1)
    print("wrote", os.path.join(a.out, "submarine_depth.json"))


if __name__ == "__main__":
    main()
