"""Figures for the fin experiment (matplotlib): the result card, the temperature profile and how h converges.

Colors follow one validated palette: the PINN is series 1 (blue), the thermocouples series 2 (orange), the analytic
solution and the starting guess are neutral ink, and the fin is drawn with a single-hue temperature ramp.
"""
from __future__ import annotations

import os

import numpy as np

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
HEAT = ["#fde9de", "#f9c6ab", "#f39e76", "#eb6834", "#c94d1d", "#9a3510"]       # one hue, light -> dark


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
    })


def _theta_exact(xi, h, s):
    import math
    k, d, l_ = s["k_W_mK"], s["D_m"], s["L_m"]
    area, perim = math.pi * d * d / 4, math.pi * d
    m = math.sqrt(h * perim / (k * area))
    r = h / (m * k)
    return (np.cosh(m * l_ * (1 - xi)) + r * np.sinh(m * l_ * (1 - xi))) / (np.cosh(m * l_) + r * np.sinh(m * l_))


def _profile_axes(ax, summary, run):
    s = summary["setup"]
    tb, ta, lmm = s["T_base_C"], s["T_air_C"], s["L_m"] * 1000
    xi = np.linspace(0, 1, 300)
    guess = ta + (tb - ta) * _theta_exact(xi, s["h_initial_guess"], s)
    exact = ta + (tb - ta) * _theta_exact(xi, s["h_true_W_m2K"], s)
    pinn_xi = np.array(run["profile"]["xi"])
    pinn = ta + (tb - ta) * np.array(run["profile"]["theta"])
    ax.plot(xi * lmm, guess, color=MUTED, lw=2, ls=(0, (2, 3)), zorder=2,
            label=f"what the starting guess h = {s['h_initial_guess']:.0f} predicts")
    ax.plot(pinn_xi * lmm, pinn, color=BLUE, lw=3, zorder=3, solid_capstyle="round",
            label="PINN: temperature field and h learned together")
    ax.plot(xi * lmm, exact, color=INK, lw=1.6, ls=(0, (5, 4)), zorder=4,
            label="analytic solution (only used to check)")
    x_s = np.array(s["sensors_mm"])
    ax.errorbar(x_s, run["readings_C"], yerr=s["noise_C"], fmt="o", ms=9, color=ORANGE, mec=SURFACE, mew=2,
                ecolor=ORANGE, elinewidth=1.6, capsize=4, zorder=5, label="thermocouples (±0.5 °C noise)")
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 1, 2, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", fontsize=12,
              handlelength=2.6, labelcolor=INK)
    ax.set_xlim(0, lmm + 1)
    ax.set_xlabel("distance from the wall (mm)")
    ax.set_ylabel("temperature (°C)")


def make_figures(summary: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    s = summary["setup"]
    runs = summary["runs"]
    hp, q = summary["h_pinn"], summary["q_pinn_W"]
    run = min(runs, key=lambda r: abs(r["h_pinn"] - hp["mean"]))          # the most typical draw
    tb, ta, lmm = s["T_base_C"], s["T_air_C"], s["L_m"] * 1000

    # ---------------------------------------------------------------- 1. result card (4:5, LinkedIn)
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.text(0.07, 0.955, "How strong is the convection on this fin?", fontsize=27, fontweight="bold", color=INK)
    fig.text(0.07, 0.925, "Unknown h, five noisy thermocouples, one physics-informed network.", fontsize=16,
             color=INK2)

    # fin drawing, colored by the PINN temperature
    axf = fig.add_axes([0.07, 0.745, 0.86, 0.15])
    axf.set_axis_off()
    axf.set_xlim(-14, lmm + 6)
    axf.set_ylim(-11, 12)
    axf.add_patch(plt.Rectangle((-14, -11), 12, 23, color="#6b6a65"))
    axf.text(-8, 0, f"wall\n{tb:.0f} °C", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    cmap = LinearSegmentedColormap.from_list("heat", HEAT)
    xi = np.array(run["profile"]["xi"])
    th = np.array(run["profile"]["theta"])
    polys = [[(xi[i] * lmm - 2, -3.2), (xi[i + 1] * lmm - 2, -3.2), (xi[i + 1] * lmm - 2, 3.2), (xi[i] * lmm - 2, 3.2)]
             for i in range(len(xi) - 1)]
    pc = PolyCollection(polys, array=(th[:-1] + th[1:]) / 2, cmap=cmap, clim=(0.25, 1.0), edgecolor="none")
    axf.add_collection(pc)
    for x_mm, t in zip(s["sensors_mm"], run["readings_C"]):
        axf.plot([x_mm - 2, x_mm - 2], [3.2, 7.2], color=INK2, lw=1.4)
        axf.plot(x_mm - 2, 7.2, "o", ms=9, color=ORANGE, mec=SURFACE, mew=2)
        axf.text(x_mm - 2, 9.6, f"{t:.1f} °C", ha="center", fontsize=12, color=INK)
    axf.text(lmm / 2 - 2, -7.6, f"stainless-steel pin fin  ·  Ø{s['D_m'] * 1000:.0f} mm × {lmm:.0f} mm  ·  "
             f"air {ta:.0f} °C", ha="center", fontsize=12.5, color=INK2)
    axf.text(lmm + 1, -1, "air  h = ?", fontsize=13, color=INK, fontweight="bold", va="center")

    ax = fig.add_axes([0.11, 0.335, 0.83, 0.37])
    _profile_axes(ax, summary, run)

    # hero numbers
    tiles = [
        (f"{hp['mean']:.1f} ± {hp['std']:.1f}", "W/m²K", f"h identified  ·  true value {s['h_true_W_m2K']:.0f}"),
        (f"{q['mean']:.3f}", "W", f"heat the fin dissipates  ·  true {summary['q_true_W']:.3f}"),
        (f"{summary['max_field_error_C']:.2f}", "°C", "worst error on the temperature field"),
    ]
    for i, (big, unit, label) in enumerate(tiles):
        y = 0.235 - i * 0.062
        fig.text(0.07, y, big, fontsize=30, fontweight="bold", color=INK)
        fig.text(0.07 + 0.024 * len(big) + 0.015, y + 0.004, unit, fontsize=16, color=INK2)
        fig.text(0.47, y + 0.006, label, fontsize=14.5, color=INK2)
    lsq = summary["h_least_squares"]
    fig.text(0.07, 0.068, f"Same accuracy as fitting the analytic formula ({lsq['mean']:.1f} ± {lsq['std']:.1f}), "
             "but the PINN never needs a formula.", fontsize=13.5, color=INK, style="italic")
    fig.text(0.07, 0.035, f"{s['draws']} independent noise draws  ·  started 4× off (h = {s['h_initial_guess']:.0f})  ·  "
             "examples/use_cases/fin_convection_inverse", fontsize=11.5, color=INK2)
    fig.text(0.07, 0.016, "PINNeAPPle  ·  github.com/PINNeAPPle-Labs/PINNeAPPle", fontsize=11.5, color=INK2,
             fontweight="bold")
    fig.savefig(os.path.join(out, "fin_result_card.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- 2. profile alone (16:9)
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12)
    _profile_axes(ax, summary, run)
    fig.suptitle(f"Temperature along the fin: h = {run['h_pinn']:.1f} W/m²K learned from five readings "
                 f"(true {s['h_true_W_m2K']:.0f})", x=0.08, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, "fin_temperature_profile.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- 3. h during training (16:9)
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12)
    for r in runs:
        st, hv = np.array(r["history"]).T
        ax.plot(st, hv, color=BLUE, lw=1.6, alpha=0.55)
    ax.axhline(s["h_true_W_m2K"], color=INK2, lw=1.6, ls=(0, (6, 4)))
    ax.text(st[-1], s["h_true_W_m2K"] + 2.5, f"true h = {s['h_true_W_m2K']:.0f} W/m²K", ha="right", color=INK2,
            fontsize=12)
    ax.text(60, s["h_initial_guess"] - 2, f"start: h = {s['h_initial_guess']:.0f}", va="top", color=INK2, fontsize=12)
    ax.text(st[-1], hp["mean"] - 4, f"{s['draws']} noise draws: {hp['mean']:.1f} ± {hp['std']:.1f}", ha="right",
            va="top", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("h (W/m²K)")
    ax.set_ylim(0, s["h_initial_guess"] + 5)
    fig.suptitle("The network finds h while it learns the temperature field", x=0.08, ha="left", fontsize=16,
                 fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, "fin_h_convergence.png"))
    plt.close(fig)
    print(f"figures written to {out}")


if __name__ == "__main__":
    import json
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    with open(os.path.join(here, "summary.json")) as fh:
        make_figures(json.load(fh), here)
