"""Load in 3-D: the torque of a skewed surface-PM machine against multi-slice 2-D, both models on one mesh.

The skew study (``20_spm_skew_3d.py``) is at no load. Here the machine is driven: the slots carry the axial current
density of the stranded winding, the magnets are skewed, and the torque curve over one slot pitch has a mean and a
ripple. Two approximations are compared at the same operating point:

* the 3-D model -- the field of the full stack, with the axial flux between the skewed segments and 20 mm of air
  beyond each end face;
* multi-slice 2-D on the same mesh -- the mean of straight one-layer solves at the same rotor offsets, which is what
  a 2-D design loop uses.

and the 2-D code itself (``SPMMachine``, its own polar mesh) is the independent reference for the straight machine.

Machine: the default ``SPMMachine`` (24 slots, 4 poles, rotor radius 40 mm, 4 mm magnets, 1 mm gap, stack 50 mm),
linear iron (mu_r = 1000), 20 turns per slot, 200 A peak, current on the q-axis (gamma = 0, the maximum-torque
angle of a surface-PM machine). The 3-D model has 288 angular steps (12 per slot pitch: the polar mesh repeats every
two steps, so a slot pitch of an odd number of steps would not be a symmetry), 2 layers in the stack, 20 mm of air
beyond each end, and the torque curve is sampled at the 12 angles of one whole cogging period. The angular step is
1.25 deg and no skew of the table lands on a half step on any layer -- 15 deg over two layers is 3.75 deg each and
8 deg is 4 deg, which both models round to 3.75 -- so the 3-D model and the multi-slice model hold the same discrete
skew. At 144 steps (2.5 deg) the 15 deg continuous skew falls on exactly a half step, the rounding of the two models
then stops agreeing at every angle, and the study is run on this grid for that reason.

    python examples/electrodynamics/27_spm_load_3d.py            # about 10 min; PARDISO (pypardiso) if installed
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machine import SPMMachine
from pinneapple_simulation.electromagnetics.machines3d import SPM3D
from pinneapple_simulation.electromagnetics.machines3d.load import load_study, ripple
from pinneapple_simulation.electromagnetics.materials import LinearMaterial

OUT = Path(__file__).parent / "results"
TOTALS_DEG = (0, 8, 15)
KINDS = ("step2", "continuous")
N_THETA = 288
I_PEAK = 200.0
GAMMA_DEG = 0.0


def main() -> dict:
    t0 = time.perf_counter()
    m = SPMMachine(iron=LinearMaterial(mu_r=1000.0), n_theta=N_THETA)
    model = SPM3D(m, n_theta=N_THETA, stack_layers=2, end_length=0.02, end_layers=1)
    print(f"3-D model: {model.mesh.summary()['tets']} tetrahedra, {model.space.n} edges", flush=True)
    gamma = math.radians(GAMMA_DEG)
    currents_of = lambda th: m.phase_currents(th, I_PEAK, gamma)        # noqa: E731 - a one-line drive
    period = 2 * math.pi / math.lcm(m.slots, 2 * m.pole_pairs)
    steps = int(round(period / model.step))
    thetas = np.arange(steps) * model.step
    OUT.mkdir(exist_ok=True)
    res = load_study(model, np.radians(TOTALS_DEG), kinds=KINDS, thetas=thetas, currents_of=currents_of,
                     progress=lambda s: print(s, flush=True),
                     checkpoint=OUT / "27_spm_load_3d.rows.jsonl")
    # the 2-D code on its own polar mesh, at the same operating point: the independent reference
    t2 = np.array([m.torque(m.solve(float(x), currents_of(float(x)))) for x in thetas])
    mean2, pk2, rip2 = ripple(t2)
    two_d_cog = np.array([m.torque(m.solve(float(x))) for x in thetas])
    out = {"machine": {"slots": m.slots, "poles": 2 * m.pole_pairs, "length_m": m.length, "r_rotor_m": m.r_rotor,
                       "turns_per_slot": m.turns_per_slot, "mu_r_iron": 1000.0, "I_peak_A": I_PEAK,
                       "gamma_deg": GAMMA_DEG, "cogging_period_deg": math.degrees(period)},
           "model": {**model.mesh.summary(), "edges": model.space.n, "stack_layers": model.stack_layers,
                     "end_length_m": model.end_length, "n_theta": model.n_theta},
           "thetas_deg": np.degrees(thetas).round(6).tolist(),
           "two_d_code": {"torque": t2.tolist(), "mean": mean2, "pkpk": pk2, "ripple": rip2,
                          "cogging_pkpk": float(two_d_cog.max() - two_d_cog.min())},
           "rows": res["rows"], "seconds": time.perf_counter() - t0, "date": time.strftime("%Y-%m-%d")}
    (OUT / "27_spm_load_3d.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.style import use_theme
    use_theme("report")

    colours = {"step2": "#2a78d6", "step3": "#eb6834", "continuous": "#1baf7a"}
    muted = "#52514e"
    th = np.asarray(out["thetas_deg"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#fcfcfb")
    for r in out["rows"]:
        if r["kind"] == "straight":
            continue
        c = colours.get(r["kind"], muted)
        lab = f"{r['kind']} {math.degrees(r['total']):.0f} deg"
        ax[0].plot(th, r["torque_3d"], color=c, lw=2.0, label=lab)
        ax[0].plot(th, r["torque_ms"], color=c, lw=1.0, ls="--")
    ax[0].set_xlabel("rotor angle (deg)")
    ax[0].set_ylabel("torque (N m)")
    ax[0].set_title("solid: 3-D, dashed: multi-slice 2-D", fontsize=10, color=muted)
    ax[0].legend(fontsize=8, frameon=False)

    rows = [r for r in out["rows"]]
    x = np.arange(len(rows))
    w = 0.38
    ax[1].bar(x - w / 2, [100 * r["ripple_3d"] for r in rows], w, color="#1f4e79", label="3-D")
    ax[1].bar(x + w / 2, [100 * r["ripple_ms"] for r in rows], w, color="#9dc3e6", label="multi-slice 2-D")
    ax[1].set_xticks(x, [f"{r['kind']}\n{math.degrees(r['total']):.0f} deg" for r in rows], fontsize=8)
    ax[1].set_ylabel("torque ripple, pk-pk / mean (%)")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "27_spm_load_3d.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
