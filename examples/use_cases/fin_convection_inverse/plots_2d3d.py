"""Figures for the 2D plate and the 3D block (same palette as plots.py): a result card each, and h during training."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plots import BLUE, GRID, HEAT, INK, INK2, ORANGE, SURFACE, _style  # noqa: E402


def _tiles(fig, tiles, y0, dy=0.062):
    for i, (big, unit, label) in enumerate(tiles):
        y = y0 - i * dy
        fig.text(0.07, y, big, fontsize=30, fontweight="bold", color=INK)
        fig.text(0.07 + 0.024 * len(big) + 0.015, y + 0.004, unit, fontsize=16, color=INK2)
        fig.text(0.47, y + 0.006, label, fontsize=14.5, color=INK2)


def _footer(fig, s, path, extra):
    fig.text(0.07, 0.068, extra, fontsize=13.5, color=INK, style="italic")
    fig.text(0.07, 0.035, f"{s['draws']} independent noise draws  ·  started at h = {s['h_initial_guess']:.0f}  ·  "
             f"{path}", fontsize=11.5, color=INK2)
    fig.text(0.07, 0.016, "PINNeAPPle  ·  github.com/PINNeAPPle-Labs/PINNeAPPle", fontsize=11.5, color=INK2,
             fontweight="bold")


def _h_history(summary, out, name, title, ylim):
    import matplotlib.pyplot as plt
    s, hp = summary["setup"], summary["h_pinn"]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12)
    for r in summary["runs"]:
        st, hv = np.array(r["history"]).T
        ax.plot(st, hv, color=BLUE, lw=1.6, alpha=0.6)
    ax.axhline(s["h_true_W_m2K"], color=INK2, lw=1.6, ls=(0, (6, 4)))
    ax.text(st[-1], s["h_true_W_m2K"] * 1.04, f"true h = {s['h_true_W_m2K']:.0f} W/m²K", ha="right", va="bottom",
            color=INK2, fontsize=12)
    ax.text(st[-1], hp["mean"] * 0.94, f"{s['draws']} noise draws: {hp['mean']:.1f} ± {hp['std']:.1f}", ha="right",
            va="top", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("h (W/m²K)")
    ax.set_ylim(*ylim)
    fig.suptitle(title, x=0.08, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


def plate_figures(out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    summary = json.load(open(os.path.join(out, "summary.json")))
    s, ref = summary["setup"], summary["reference"]
    runs, hp, hs = summary["runs"], summary["h_pinn"], summary["hot_spot_pinn_C"]
    k = min(range(len(runs)), key=lambda i: abs(runs[i]["h_pinn"] - hp["mean"]))
    run = runs[k]
    field = np.load(os.path.join(out, f"field_pinn_{k}.npy"))
    ref_f = np.load(os.path.join(out, "field_reference.npy"))
    lx, ly = s["Lx_m"] * 1000, s["Ly_m"] * 1000
    cmap = LinearSegmentedColormap.from_list("heat", HEAT)
    vmin, vmax = float(np.floor(ref_f.min())), float(np.ceil(ref_f.max()))

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.text(0.07, 0.955, "Where is the hot spot, and how good is the cooling?", fontsize=24, fontweight="bold",
             color=INK)
    fig.text(0.07, 0.925, "8 W device on an aluminium plate, unknown h, eight thermocouples, one PINN.",
             fontsize=16, color=INK2)
    ax = fig.add_axes([0.08, 0.50, 0.80, 0.39])
    im = ax.imshow(field, origin="lower", extent=(0, lx, 0, ly), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.contour(np.linspace(0, lx, field.shape[1]), np.linspace(0, ly, field.shape[0]), field, levels=8,
               colors=SURFACE, linewidths=0.8, alpha=0.7)
    xc, yc, half = (v * 1000 for v in s["device"][:3])
    ax.add_patch(plt.Rectangle((xc - half, yc - half), 2 * half, 2 * half, fill=False, ec=INK, lw=1.6, ls=(0, (4, 3))))
    ax.text(xc, yc - half - 2.5, f"{s['power_W']:.0f} W device", ha="center", va="top", fontsize=12, color=INK)
    j, i = np.unravel_index(np.argmax(field), field.shape)
    hx, hy = (i + 0.5) * lx / field.shape[1], (j + 0.5) * ly / field.shape[0]
    ax.plot(hx, hy, marker="X", ms=13, color=INK, mec=SURFACE, mew=1.5)
    ax.annotate(f"hot spot {field.max():.1f} °C", (hx, hy), xytext=(14, 16), textcoords="offset points",
                fontsize=13, fontweight="bold", color=INK)
    for (x, y), t in zip(s["sensors_mm"], run["readings_C"]):
        ax.plot(x, y, "o", ms=10, color=ORANGE, mec=SURFACE, mew=2)
        ax.text(x, y + 2.6, f"{t:.1f}", ha="center", fontsize=11.5, color=INK)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(False)
    cax = fig.add_axes([0.90, 0.50, 0.022, 0.39])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("temperature (°C), learned by the PINN", color=INK2)
    cb.outline.set_edgecolor(GRID)
    fig.text(0.08, 0.455, "● thermocouple readings (±0.5 °C)   ✕ hottest point of the learned field   "
             "- - device footprint", fontsize=12, color=INK2)

    tiles = [(f"{hp['mean']:.1f} ± {hp['std']:.1f}", "W/m²K", f"h identified  ·  true value {s['h_true_W_m2K']:.0f}"),
             (f"{hs['mean']:.1f}", "°C", f"hot spot  ·  reference simulation {ref['hot_spot_C']:.1f}"),
             (f"{summary['max_field_error_C']:.2f}", "°C", "worst error on the whole map")]
    _tiles(fig, tiles, 0.36)
    fig.text(0.07, 0.17, f"With a guessed h = {s['h_initial_guess']:.0f}, the hot spot would read "
             f"{summary['hot_spot_if_h_guess_C']:.0f} °C: {ref['hot_spot_C'] - summary['hot_spot_if_h_guess_C']:.0f} K "
             "too optimistic.", fontsize=13.5, color=INK)
    hf = summary["h_fv_fit"]
    _footer(fig, s, "examples/use_cases/fin_convection_inverse/plate_2d.py",
            f"No formula exists here. Fitting h by re-running a simulation gives {hf['mean']:.2f} ± {hf['std']:.2f} "
            f"({hf['solves_per_fit']} solves per fit).")
    fig.savefig(os.path.join(out, "plate_result_card.png"))
    plt.close(fig)
    _h_history(summary, out, "plate_h_convergence.png", "2D plate: the network finds h while it learns the map",
               (0, s["h_initial_guess"] * 1.1))
    print(f"2D figures written to {out}")


def block_figures(out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    summary = json.load(open(os.path.join(out, "summary.json")))
    s, ref = summary["setup"], summary["reference"]
    runs, hp, dv = summary["runs"], summary["h_pinn"], summary["device_max_pinn_C"]
    k = min(range(len(runs)), key=lambda i: abs(runs[i]["h_pinn"] - hp["mean"]))
    field = np.load(os.path.join(out, f"field_pinn_{k}.npy"))          # (nz, ny, nx)
    render = os.path.join(out, "block_render.png")
    _render_block(field, s, runs[k]["readings_C"], render)

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.text(0.07, 0.955, "How hot is the chip nobody can reach?", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.07, 0.925, "10 W chip under a steel block, fan of unknown h, nine thermocouples on top, one PINN.",
             fontsize=15, color=INK2)
    ax = fig.add_axes([0.04, 0.44, 0.92, 0.47])
    ax.imshow(plt.imread(render))
    ax.set_axis_off()
    fig.text(0.08, 0.425, "● thermocouples on the top face   ·   cut-away: the learned 3D field   ·   "
             "dark block: the 10 W chip", fontsize=12, color=INK2)
    he = summary["h_energy_balance"]
    tiles = [(f"{dv['mean']:.1f}", "°C", f"chip temperature (not measured)  ·  reference {ref['device_max_C']:.1f}"),
             (f"{he['mean']:.0f}", "W/m²K", f"fan h, energy balance on the learned field  ·  true {s['h_true_W_m2K']:.0f}"),
             (f"{summary['max_field_error_C']:.2f}", "°C", "worst error in the whole 3D field")]
    _tiles(fig, tiles, 0.33)
    fig.text(0.07, 0.165, f"The thermocouples read {ref['top_min_C']:.0f}–{ref['top_max_C']:.0f} °C; the chip under them runs at "
             f"{ref['device_max_C']:.0f} °C.", fontsize=13, color=INK)
    fig.text(0.07, 0.14, f"The network's own h parameter lands at {hp['mean']:.0f} ± {hp['std']:.0f} "
             f"({(hp['mean'] / s['h_true_W_m2K'] - 1) * 100:+.0f} %): the flux it rests on is a ~1 K gradient.",
             fontsize=13, color=INK)
    hf = summary["h_fv_fit"]
    _footer(fig, s, "examples/use_cases/fin_convection_inverse/block_3d.py",
            f"Fitting h by re-running a 3D simulation gives {hf['mean']:.1f} ± {hf['std']:.1f} "
            f"({hf['solves_per_fit']} solves per fit).")
    fig.savefig(os.path.join(out, "block_result_card.png"))
    plt.close(fig)
    _h_history(summary, out, "block_h_convergence.png",
               f"3D block: the h parameter settles {abs(hp['mean'] / s['h_true_W_m2K'] - 1) * 100:.0f} % low; "
               f"the energy balance on the same field gives {summary['h_energy_balance']['mean']:.0f}",
               (0, max(s["h_true_W_m2K"], max(max(v for _, v in r["history"]) for r in runs)) * 1.15))
    print(f"3D figures written to {out}")


def _render_block(field, s, readings, path):
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap
    pv.OFF_SCREEN = True
    nz, ny, nx = field.shape
    lx, ly, lz = (v * 1000 for v in s["L_m"])
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(lx / nx, ly / ny, lz / nz))
    grid.cell_data["T"] = field.ravel(order="C")
    # cut away the front-right quarter (x > half, y < half) to show the inside and the hot zone above the chip
    c = grid.cell_centers().points
    keep = np.where(~((c[:, 0] > lx / 2 + 2) & (c[:, 1] < ly / 2 - 2)))[0]   # centre TC stays on the block
    block = grid.extract_cells(keep)
    cmap = LinearSegmentedColormap.from_list("heat", HEAT)
    p = pv.Plotter(off_screen=True, window_size=(1600, 1150))
    p.set_background(SURFACE)
    p.add_mesh(block, scalars="T", cmap=cmap, show_edges=False, lighting=True, ambient=0.55, diffuse=0.5,
               specular=0.0,
               scalar_bar_args={"title": "temperature (°C)", "color": INK, "vertical": True, "title_font_size": 40,
                                "label_font_size": 36, "position_x": 0.84, "position_y": 0.2, "height": 0.6,
                                "width": 0.05, "fmt": "%.0f", "n_labels": 5})
    xc, yc, half = (v * 1000 for v in s["device"][:3])
    p.add_mesh(pv.Box(bounds=(xc - half, xc + half, yc - half, yc + half, -2.0, 0.0)), color="#3b3a37",
               ambient=0.5)
    for (x, y), _t in zip(s["sensors_mm"], readings):
        p.add_mesh(pv.Sphere(radius=1.1, center=(x, y, lz + 0.7)), color=ORANGE, ambient=0.5)
    p.camera_position = [(80, -55, 48), (lx / 2, ly / 2, lz / 4), (0, 0, 1)]
    p.camera.zoom(1.15)
    p.screenshot(path)
    p.close()


if __name__ == "__main__":
    base = os.path.join(_HERE, "results")
    if os.path.exists(os.path.join(base, "2d", "summary.json")):
        plate_figures(os.path.join(base, "2d"))
    if os.path.exists(os.path.join(base, "3d", "summary.json")):
        block_figures(os.path.join(base, "3d"))
