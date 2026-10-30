"""Skew of a surface-PM machine: cogging in 3-D (edge elements) against multi-slice 2-D on the same mesh.

Machine: the default ``SPMMachine`` (24 slots, 4 poles, rotor radius 40 mm, 4 mm magnets, 1 mm gap, stack 50 mm),
linear iron (mu_r = 1000). The 3-D model has 720 angular steps (30 per slot pitch: the polar mesh repeats every
two steps, so a slot pitch of an odd number of steps would not be a symmetry), 8 layers in the stack and 20 mm of
air beyond each end; magnet skew by step (2 or 3 segments) or continuous (one offset per stack layer, 0.5 degree
steps); cogging over a whole period every 1.5 degrees. The multi-slice model is the
mean of straight one-layer models at the same offsets: the difference measures the axial flux between segments
and the end leakage that multi-slice neglects.

    python examples/electrodynamics/20_spm_skew_3d.py            # about 1 h; PARDISO (pypardiso) if installed
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machine import SPMMachine
from pinneapple_simulation.electromagnetics.machines.robustness import step_skew_factor
from pinneapple_simulation.electromagnetics.machines3d import SPM3D
from pinneapple_simulation.electromagnetics.machines3d.skew import continuous_factor, skew_study
from pinneapple_simulation.electromagnetics.materials import LinearMaterial

OUT = Path(__file__).parent / "results"
TOTALS_DEG = (0, 5, 8, 10, 15)
N_THETA = 720


def main() -> dict:
    t0 = time.perf_counter()
    m = SPMMachine(iron=LinearMaterial(mu_r=1000.0), n_theta=N_THETA)
    model = SPM3D(m, n_theta=N_THETA, stack_layers=8, end_length=0.02, end_layers=2)
    print(f"3-D model: {model.mesh.summary()['tets']} tetrahedra", flush=True)
    thetas = np.radians(np.arange(0.0, 15.0, 1.5))
    OUT.mkdir(exist_ok=True)
    res = skew_study(model, np.radians(TOTALS_DEG), thetas=thetas, progress=lambda s: print(s, flush=True),
                     checkpoint=OUT / "20_spm_skew_3d.rows.jsonl")      # resumes an interrupted run
    # 2-D reference of the straight machine with the 2-D code (its own polar mesh with a shaft region)
    th = np.asarray(res["thetas"])
    two_d = [m.torque(m.solve(x)) for x in th]     # the 2-D code's own polar mesh (shaft region, sparse core rings)
    rows = res["rows"]
    st = next(r for r in rows if r["kind"] == "straight")
    lam = res["period"]
    for r in rows:
        r["ratio_3d"] = r["pkpk_3d"] / st["pkpk_3d"]
        r["ratio_ms"] = r["pkpk_ms"] / st["pkpk_ms"]
        s = r["total"]
        if r["kind"].startswith("step"):
            K = int(r["kind"][4:])
            r["fundamental_factor"] = float(abs(step_skew_factor(1, K, s / (K - 1), lam))) if s else 1.0
        else:
            r["fundamental_factor"] = float(abs(continuous_factor(1, s, lam)))
    out = {"machine": {"slots": m.slots, "poles": 2 * m.pole_pairs, "length_m": m.length, "r_rotor_m": m.r_rotor,
                       "mu_r_iron": 1000.0, "cogging_period_deg": math.degrees(lam)},
           "model": {**model.mesh.summary(), "stack_layers": model.stack_layers, "end_length_m": model.end_length,
                     "n_theta": model.n_theta},
           "thetas_deg": np.degrees(th).tolist(), "two_d_code_straight": two_d,
           "two_d_code_pkpk": float(np.max(two_d) - np.min(two_d)), "rows": rows,
           "seconds": time.perf_counter() - t0, "date": time.strftime("%Y-%m-%d")}
    OUT.mkdir(exist_ok=True)
    (OUT / "20_spm_skew_3d.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    colours = {"step2": "#2a78d6", "step3": "#eb6834", "continuous": "#1baf7a"}
    muted = "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#fcfcfb")
    rows = out["rows"]
    st = next(r for r in rows if r["kind"] == "straight")
    for kind, c in colours.items():
        rs = [st] + sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["total"])
        s = [math.degrees(r["total"]) for r in rs]
        ax[0].plot(s, [r["pkpk_3d"] for r in rs], "-o", color=c, lw=2, ms=6, label=f"{kind}, 3-D")
        ax[0].plot(s, [r["pkpk_ms"] for r in rs], "--", color=c, lw=1.5, label=f"{kind}, multi-slice")
    ax[0].set(xlabel="total skew (deg)", ylabel="cogging peak-to-peak (N m)",
              title=f"Cogging against skew (period {out['machine']['cogging_period_deg']:.0f} deg)")
    ax[0].legend(frameon=False, fontsize=8, ncol=2)
    th = out["thetas_deg"]
    ax[1].plot(th, st["cogging_3d"], "-o", color=muted, lw=2, ms=5, label="straight, 3-D")
    ax[1].plot(th, st["cogging_ms"], "--", color=muted, lw=1.5, label="straight, one slice")
    best = {k: min((r for r in rows if r["kind"] == k), key=lambda r: r["pkpk_3d"]) for k in colours}
    for kind, r in best.items():
        ax[1].plot(th, r["cogging_3d"], "-o", color=colours[kind], lw=2, ms=5,
                   label=f"{kind} {math.degrees(r['total']):.0f} deg, 3-D")
        ax[1].plot(th, r["cogging_ms"], "--", color=colours[kind], lw=1.5)
    ax[1].set(xlabel="rotor angle (deg)", ylabel="cogging torque (N m)", title="One cogging period")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "20_spm_skew_3d.png", dpi=150)


if __name__ == "__main__":
    r = main()
    print(f"done in {r['seconds']:.0f} s")
