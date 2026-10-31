"""End effects of a surface-PM machine against L/D: the magnet flux a full-pitch search coil links in 3-D (stack
plus 40 mm of air beyond each end) against the same stack without end air (the 2-D solution on the same mesh).

Machine: the default ``SPMMachine`` (24 slots, 4 poles, bore diameter D = 90 mm), linear iron (mu_r = 1000),
stack lengths L = 0.25 D to 2 D; 240 angular steps, stack layers of at most 15 mm, rotor aligned with the search
coil (the coil side at mid-slot radius, a pole pitch wide). The end effect is written as a ratio
``k = lambda_3D / lambda_2D`` and as the length the 2-D model would have to lose to match, ``dL = L (1 - k)``.

    python examples/electrodynamics/22_spm_end_effects_3d.py      # about 5 min, 7 GB
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pinneapple_simulation.electromagnetics.machine import SPMMachine
from pinneapple_simulation.electromagnetics.machines3d.endeffects import end_effect_study
from pinneapple_simulation.electromagnetics.materials import LinearMaterial

OUT = Path(__file__).parent / "results"
RATIOS = (0.25, 0.5, 1.0, 2.0)
OVERHANGS = (0.0, 0.01)


def main() -> dict:
    t0 = time.perf_counter()
    m = SPMMachine(iron=LinearMaterial(mu_r=1000.0))
    res = end_effect_study(m, RATIOS, end_length=0.04, end_layers=4, layer_dz=0.015, n_theta=240,
                           overhangs=OVERHANGS, progress=lambda s: print(s, flush=True))
    # sensitivity at L/D = 0.5: twice the end air, and twice the angular resolution
    sens = {"end_length_0.08": end_effect_study(m, (0.5,), end_length=0.08, end_layers=8, n_theta=240,
                                                overhangs=OVERHANGS)["rows"][0],
            "n_theta_480": end_effect_study(m, (0.5,), end_length=0.04, end_layers=4, n_theta=480,
                                            overhangs=OVERHANGS)["rows"][0]}
    for r in res["rows"]:
        for o in OVERHANGS:
            r[f"dL_mm_o{o * 1e3:g}mm"] = 1e3 * r["length_m"] * (1 - r[f"k_o{o * 1e3:g}mm"])
    out = {**res, "sensitivity": sens, "seconds": time.perf_counter() - t0, "date": time.strftime("%Y-%m-%d")}
    OUT.mkdir(exist_ok=True)
    (OUT / "22_spm_end_effects_3d.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = {0.0: "#2a78d6", 0.01: "#eb6834"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    ld = [r["L_over_D"] for r in out["rows"]]
    for o, c in colours.items():
        ax[0].plot(ld, [r[f"k_o{o * 1e3:g}mm"] for r in out["rows"]], "-o", color=c, lw=2, ms=6,
                   label=f"search coil, ends {o * 1e3:g} mm beyond the stack")
        ax[1].plot(ld, [r[f"dL_mm_o{o * 1e3:g}mm"] for r in out["rows"]], "-o", color=c, lw=2, ms=6)
    ax[0].plot(ld, [r["k_gap_flux"] for r in out["rows"]], "--", color="#52514e", lw=1.5,
               label="mean gap flux density")
    ax[0].set(xscale="log", xlabel="L / D", ylabel="3-D / 2-D", title="Magnet flux linked, 3-D over 2-D")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(xscale="log", xlabel="L / D", ylabel="dL = L (1 - k)  (mm)", title="Equivalent length lost at the ends")
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "22_spm_end_effects_3d.png", dpi=150)


if __name__ == "__main__":
    r = main()
    print(f"done in {r['seconds']:.0f} s")
