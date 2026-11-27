"""Figures for a wind POC: power curves, residuals, layout, wakes and the polar.

Each function takes data the other modules already computed and returns a
:class:`matplotlib.figure.Figure`; saving is the caller's business, so a notebook
can show the object and a script can write it to disk.

House style: one colour per model for the whole document (see :data:`COLORS`), the
measured data always in grey, and every axis labelled with units. A figure that
would need a caption to be readable gets its message in the title instead.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

__all__ = [
    "COLORS",
    "plot_power_curves",
    "plot_error_by_speed",
    "plot_wind_rose",
    "plot_layout",
    "plot_wake_polar",
    "plot_farm_deficit_map",
    "plot_learning_curve",
    "plot_efficiency_by_speed",
    "save",
]

COLORS = {
    "measured": "#9e9e9e",
    "reference": "#7a5195",
    "binned": "#ef8636",
    "logistic": "#0f8b8d",
    "physics-informed": "#004c6d",
    "conventional-ml": "#a23a00",
    "operator": "#4c4c4c",
    "model": "#004c6d",
    "observed": "#a23a00",
}

FIG_DPI = 130


def _figure(width: float = 7.0, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=FIG_DPI)
    return fig, ax


def save(fig, path) -> str:
    """Write a figure and close it, returning the path as a string."""
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Turbine level
# ---------------------------------------------------------------------------

def plot_power_curves(
    curves: Mapping[str, object],
    frame,
    *,
    speed_column: str = "u_corrected",
    power_column: str = "power_kw",
    rated_power_kw: float | None = None,
    title: str = "Power curves, fitted on the training window only",
    max_speed: float = 26.0,
    subtitle: str | None = None,
):
    """Measured cloud plus every model's curve, on density-corrected speed."""
    fig, ax = _figure(7.6, 4.8)
    speed = frame[speed_column].to_numpy(dtype=float)
    power = frame[power_column].to_numpy(dtype=float)
    ax.hexbin(speed, power, gridsize=60, cmap="Greys", bins="log", mincnt=1, alpha=0.55,
              linewidths=0)
    grid = np.linspace(0.0, max_speed, 400)
    for name, fn in curves.items():
        values = np.asarray(fn(grid), dtype=float)
        key = next((k for k in COLORS if k in name.lower()), None)
        colour = COLORS.get(key, None)
        ax.plot(grid, values, lw=2.0, label=name, color=colour,
                linestyle="--" if "operator" in name.lower() else "-")
    if rated_power_kw is not None:
        ax.axhline(rated_power_kw, color="#bbbbbb", lw=0.8, ls=":")
        ax.annotate("rated", (max_speed - 0.4, rated_power_kw), ha="right", va="bottom",
                    fontsize=8, color="#666666")
    ax.set_xlabel("density-corrected hub wind speed [m/s]")
    ax.set_ylabel("active power [kW]")
    ax.set_xlim(0, max_speed)
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    return fig


def plot_error_by_speed(rows_by_model: Mapping[str, Sequence[dict]], *, title: str = "Error against wind speed"):
    """Signed bias per speed bin for every model -- where a curve is wrong matters."""
    fig, ax = _figure(7.6, 4.0)
    for name, rows in rows_by_model.items():
        if not rows:
            continue
        centres = [0.5 * (r["speed_lo"] + r["speed_hi"]) for r in rows]
        bias = [r["bias_kw"] for r in rows]
        key = next((k for k in COLORS if k in name.lower()), None)
        ax.plot(centres, bias, marker="o", ms=3, lw=1.6, label=name, color=COLORS.get(key))
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_xlabel("density-corrected wind speed [m/s]")
    ax.set_ylabel("bias, model - measured [kW]")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return fig


def plot_learning_curve(history: Sequence[float], *, title: str = "Physics-informed fit"):
    """Training loss per checkpoint, on a log scale."""
    fig, ax = _figure(4.6, 3.2)
    ax.semilogy(np.arange(len(history)), history, color=COLORS["physics-informed"], lw=1.8)
    ax.set_xlabel("checkpoint (every 25 epochs)")
    ax.set_ylabel("loss (normalised power MSE + penalties)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25, which="both")
    return fig


def plot_wind_rose(frame, *, title: str = "Wind rose (operating samples)"):
    """Direction histogram and the speed distribution beside it."""
    fig = plt.figure(figsize=(8.4, 3.8), dpi=FIG_DPI)
    grid = GridSpec(1, 2, width_ratios=(1, 1.25), wspace=0.3)
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    direction = np.radians(frame["wind_direction_deg"].to_numpy(dtype=float))
    counts, edges = np.histogram(direction, bins=36, range=(0, 2 * np.pi))
    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align="edge", color=COLORS["physics-informed"], alpha=0.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("direction the wind comes from", fontsize=9)
    ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(grid[0, 1])
    speed = frame["wind_speed_ms"].to_numpy(dtype=float)
    ax2.hist(speed, bins=40, range=(0, 26), color=COLORS["measured"], alpha=0.8)
    ax2.set_xlabel("nacelle wind speed [m/s]")
    ax2.set_ylabel("10-min samples")
    ax2.set_title(title, fontsize=10, loc="left")
    ax2.grid(alpha=0.25)
    return fig


# ---------------------------------------------------------------------------
# Farm level
# ---------------------------------------------------------------------------

def plot_layout(layout, *, deficits: Sequence[float] | None = None, title: str | None = None,
                annotate: bool = True):
    """Turbine positions; optionally coloured by a per-turbine deficit.

    One rotor diameter is drawn as a circle around each machine, because spacing
    in diameters -- not in metres -- is what decides whether a wake matters.
    """
    fig, ax = _figure(7.4, 5.2)
    positions = layout.positions
    radii = layout.rotor_diameters / 2.0
    if deficits is None:
        scatter = ax.scatter(positions[:, 0], positions[:, 1], s=42,
                             color=COLORS["physics-informed"], zorder=3)
    else:
        scatter = ax.scatter(positions[:, 0], positions[:, 1], s=90, c=np.asarray(deficits),
                             cmap="inferno_r", vmin=0.0, vmax=max(float(np.max(deficits)), 1e-6),
                             zorder=3, edgecolors="white", linewidths=0.6)
        bar = fig.colorbar(scatter, ax=ax, pad=0.02)
        bar.set_label("predicted velocity deficit [-]")
    for x, y, r, name in zip(positions[:, 0], positions[:, 1], radii, layout.names, strict=True):
        ax.add_patch(plt.Circle((x, y), r, fill=False, lw=0.6, color="#888888", alpha=0.8))
        if annotate:
            ax.annotate(name, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title(title or f"{layout.name}: {len(layout)} turbines, {layout.installed_capacity_mw:.1f} MW",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.2)
    return fig


def plot_wake_polar(observation, comparison: Mapping, *, title: str = "Farm efficiency against wind direction"):
    """Observed vs modelled farm efficiency, sector by sector.

    The model curve is drawn twice: as computed, and rescaled by the single
    calibration factor the comparison fitted. The second one is the only fair
    shape comparison, and the figure says so in the legend.
    """
    fig, ax = _figure(7.6, 4.4)
    sectors = np.asarray(observation.sectors_deg, dtype=float)
    model = np.array(comparison["model_efficiency"], dtype=float)
    observed = np.array(comparison["observed_efficiency"], dtype=float)
    model_sector = np.nanmean(model, axis=1) if model.ndim == 2 else model
    obs_sector = np.nanmean(observed, axis=1) if observed.ndim == 2 else observed
    ax.plot(np.r_[sectors, sectors[0] + 360], np.r_[model_sector, model_sector[0]],
            color=COLORS["model"], lw=2.0, label=f"wake model ({comparison.get('config', '')})")
    if "scale" in comparison:
        scaled = model_sector * comparison["scale"]
        ax.plot(np.r_[sectors, sectors[0] + 360], np.r_[scaled, scaled[0]],
                color=COLORS["model"], lw=1.4, ls="--",
                label=f"model x fitted scale ({comparison['scale']:.2f})")
    ax.plot(np.r_[sectors, sectors[0] + 360], np.r_[obs_sector, obs_sector[0]],
            color=COLORS["observed"], lw=1.6, marker="o", ms=3.5, label="SCADA-observed (reference turbine)")
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("wind direction the wind comes from [deg]")
    ax.set_ylabel("farm efficiency, P / P(no wake)")
    ax.set_ylim(0.4, 1.35)
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=1)
    ax.grid(alpha=0.25)
    return fig


def plot_farm_deficit_map(layout, deficits, *, direction_deg: float, wind_speed: float,
                          title: str | None = None):
    """Layout coloured by the model's predicted deficit at one inflow condition."""
    return plot_layout(
        layout,
        deficits=deficits,
        title=title or f"Predicted wake deficit at {wind_speed:g} m/s from {direction_deg:g} deg",
    )


def plot_efficiency_by_speed(observation, comparison: Mapping, sector_index: int, *,
                             title: str | None = None):
    """Efficiency against wind speed for one sector: model vs observation."""
    fig, ax = _figure(6.4, 3.8)
    speeds = np.asarray(observation.speed_centres, dtype=float)
    model = np.array(comparison["model_efficiency"], dtype=float)[sector_index]
    observed = np.array(comparison["observed_efficiency"], dtype=float)[sector_index]
    counts = observation.counts[sector_index]
    ax.plot(speeds, model, color=COLORS["model"], lw=2.0, label="wake model")
    ax.plot(speeds, observed, color=COLORS["observed"], lw=1.4, marker="o", label="observed")
    for u, o, n in zip(speeds, observed, counts, strict=True):
        if np.isfinite(o):
            ax.annotate(f"{n}", (u, o), xytext=(0, -12), textcoords="offset points",
                        fontsize=6, ha="center", color="#666666")
    ax.set_xlabel("reference wind speed [m/s]")
    ax.set_ylabel("farm efficiency")
    ax.set_title(title or f"Sector {observation.sectors_deg[sector_index]:.0f} deg "
                         f"(labels: samples per bin)", fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return fig
