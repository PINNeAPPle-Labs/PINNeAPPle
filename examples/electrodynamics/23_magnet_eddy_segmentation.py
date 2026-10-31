"""Eddy-current loss in a segmented surface magnet: 3-D (edge elements) against the 2-D machine model and the
closed form.

The magnet of the default ``SPMMachine`` (52.8 mm wide over the arc, 50 mm long, 4 mm thick, sigma = 6.7e5 S/m)
is flattened into a block and cut into ``n_width x n_length`` insulated pieces (0.2 mm gaps). A uniform field of
50 mT peak, normal to its face, stands for one harmonic of the field ripple it sees (the slot harmonic of this
24-slot machine is at 1.2 kHz at 3000 rpm). For each segmentation and frequency: the 3-D loss (full time-harmonic
eddy currents, so skin effect included), the closed-form resistance-limited loss of the pieces, and the 2-D value
(each piece an infinitely long strip, what the 2-D machine model computes). A mesh-convergence ladder on the
1 x 4 case gives the discretisation error.

    python examples/electrodynamics/23_magnet_eddy_segmentation.py      # about 30 min, 7 GB; resumes
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from pinneapple_simulation.electromagnetics.machine import SPMMachine
from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import segmentation_study

OUT = Path(__file__).parent / "results"
SEGMENTS = ((1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (4, 1), (2, 4))
FREQS = (1e3, 1e4, 5e4)
B_PEAK = 0.05
H = 1.5e-3
H_LADDER = (3e-3, 2e-3, 1.5e-3, 1.2e-3)


def main() -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    m = SPMMachine()
    width = m.magnet_arc * math.pi * (m.r_rotor + m.magnet_thickness / 2) / m.pole_pairs
    geo = (width, m.length, m.magnet_thickness, m.magnet_sigma, B_PEAK)
    ck = OUT / "23_magnet_eddy_segmentation.rows.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    cases = [(f"seg{nw}x{nl}", (nw, nl), FREQS, H) for nw, nl in SEGMENTS]
    cases += [(f"h{h * 1e3:g}mm", (1, 4), (1e3,), h) for h in H_LADDER if h != H]
    rows = {}
    for key, seg, freqs, h in cases:
        if key not in done:
            r = segmentation_study(*geo, segments=[seg], freqs=freqs, h=h, progress=lambda s: print(s, flush=True))
            done[key] = {"key": key, "h": h, "rows": r["rows"]}
            with open(ck, "a") as fh:
                fh.write(json.dumps(done[key]) + "\n")
        rows[key] = done[key]["rows"]
        for x in rows[key]:
            print(f"{key}, {x['f_Hz']:g} Hz: 3-D / closed form {x['ratio_3d_closed_form']:.4f}, 3-D / 2-D "
                  f"{x['ratio_3d_2d']:.4f}", flush=True)
    ladder = [{"h": h, **(rows[f"h{h * 1e3:g}mm"][0] if h != H else
                          next(x for x in rows["seg1x4"] if x["f_Hz"] == 1e3))} for h in H_LADDER]
    out = {"magnet": {"width_m": width, "length_m": m.length, "thickness_m": m.magnet_thickness,
                      "sigma_S_per_m": m.magnet_sigma}, "B_peak_T": B_PEAK, "h_m": H, "freqs_Hz": list(FREQS),
           "rows": [x for nw, nl in SEGMENTS for x in rows[f"seg{nw}x{nl}"]], "ladder": ladder,
           "seconds": time.perf_counter() - t0, "date": time.strftime("%Y-%m-%d")}
    (OUT / "23_magnet_eddy_segmentation.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = {1e3: "#2a78d6", 1e4: "#eb6834", 5e4: "#1baf7a"}
    muted = "#52514e"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#fcfcfb")
    rows = out["rows"]
    axial = [r for r in rows if r["n_width"] == 1]
    n = sorted({r["n_length"] for r in axial})
    for f, c in colours.items():
        rs = sorted((r for r in axial if r["f_Hz"] == f), key=lambda r: r["n_length"])
        ax[0].plot(n, [r["loss_3d_W"] / rs[0]["loss_2d_W"] for r in rs], "-o", color=c, lw=2, ms=6,
                   label=f"3-D, {f / 1e3:g} kHz")
    rs = sorted((r for r in axial if r["f_Hz"] == 1e3), key=lambda r: r["n_length"])
    ax[0].plot(n, [r["loss_closed_form_W"] / rs[0]["loss_2d_W"] for r in rs], "--", color=muted, lw=1.5,
               label="closed form (resistance-limited)")
    ax[0].axhline(1.0, color=muted, lw=1, ls=":")
    ax[0].text(n[-1], 1.03, "2-D (any axial segmentation)", ha="right", va="bottom", color=muted, fontsize=8)
    ax[0].set(xscale="log", yscale="log", xlabel="axial pieces", ylabel="loss / 2-D loss of the whole magnet",
              title="Axial segmentation: 3-D against 2-D")
    ax[0].set_xticks(n, [str(v) for v in n])
    ax[0].minorticks_off()
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    lad = out["ladder"]
    ax[1].plot([x["h"] * 1e3 for x in lad], [100 * (x["ratio_3d_closed_form"] - 1) for x in lad], "-o",
               color="#2a78d6", lw=2, ms=6)
    ax[1].set(xlabel="element size in the magnet plane (mm)", ylabel="3-D - closed form (%)",
              title="Mesh convergence, 1 x 4 pieces, 1 kHz")
    ax[1].axhline(0, color=muted, lw=0.5)
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "23_magnet_eddy_segmentation.png", dpi=150)


if __name__ == "__main__":
    r = main()
    print(f"done in {r['seconds']:.0f} s")
