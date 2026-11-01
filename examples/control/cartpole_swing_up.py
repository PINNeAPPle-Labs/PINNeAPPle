"""Inverted pendulum on a cart: energy swing-up from hanging, LQR catch at upright, push rejection.

    python examples/control/cartpole_swing_up.py          # a few seconds

Writes ``results/cartpole.json`` and ``results/cartpole.png``. The steps:

1. Linearise at upright: the open loop has a real pole at +sqrt(g/l), which gives its doubling time.
2. Compute the LQR gains from the Riccati equation.
3. Swing up from 3 deg off hanging, with the cart acceleration limited to 3 m/s^2.
4. Catch with the LQR, then reject a push at t = 12 s.
5. Compare with the uncontrolled fall from a 1 deg tilt.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_systems.control.cartpole import cartpole_study  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    r = cartpole_study()
    keys = ("unstable_pole", "doubling_time", "controllable", "lqr_gain", "closed_loop_classification", "catch_time",
            "settled_time", "push_peak_deg", "push_recovery", "max_accel", "cart_travel", "free_fall_time_to_90deg")
    summary = {k: r[k] for k in keys}
    summary["closed_loop_poles"] = [[p.real, p.imag] for p in r["closed_loop_poles"]]
    summary["params"] = r["params"]
    print(f"unstable pole +{r['unstable_pole']:.4f} 1/s, doubling time {r['doubling_time'] * 1000:.0f} ms; "
          f"no control: 1 -> 90 deg in {r['free_fall_time_to_90deg']:.2f} s")
    print(f"swing-up caught at {r['catch_time']:.2f} s, within 2 deg from {r['settled_time']:.2f} s; push peak "
          f"{r['push_peak_deg']:.1f} deg, recovered in {r['push_recovery']:.2f} s; max |a| {r['max_accel']:.1f} m/s^2")
    th = np.array(r["angle_deg"], float)
    th[np.where(np.abs(np.diff(th)) > 180)[0]] = np.nan
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax[0].plot(r["t"], th)
    ax[0].set_ylabel("angle from upright (deg)")
    ax[1].plot(r["t"], r["accel"])
    ax[1].set_ylabel("cart acceleration (m/s^2)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cartpole.png"), dpi=110)
    with open(os.path.join(out, "cartpole.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
