"""Figures of the ENSO study, in the style of :mod:`pinneapple_systems.time_series.forecast_lab.plots` (same family
colours, grid and spines): forecast plumes, skill against lead, the spring-barrier heatmap, the recharge-oscillator
phase portrait and Hovmoller diagrams."""
from __future__ import annotations

import numpy as np

from ..forecast_lab.plots import _GRID, _INK, _MUTED, FAMILY_COLORS, _plt, _style

__all__ = ["plot_plumes", "plot_skill_vs_lead", "plot_spring_barrier", "plot_phase_portrait", "plot_hovmoller",
           "plot_validation"]

_FAM_ORDER = list(FAMILY_COLORS)


def plot_plumes(panels: list[dict], path: str, *, ncols: int = 3) -> str:
    """Forecast distributions at the evaluated leads from one origin per panel, one family per colour (median
    marker, 50 % thick and 90 % thin interval bars, slightly offset per family), observed Nino 3.4 in black.
    Each panel: ``title``, ``t_obs``/``y_obs`` (months as fractional years), ``origin_t``, and ``fams``: list of
    {family, name, t (lead times), q (5 x n quantiles 5/25/50/75/95 %)}."""
    plt = _plt()
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 2.9 * nrows), squeeze=False)
    for ax, p in zip(axes.ravel(), panels, strict=False):
        _style(ax)
        ax.axhspan(-0.5, 0.5, color=_GRID, alpha=0.5, lw=0)
        ax.plot(p["t_obs"], p["y_obs"], color=_INK, lw=1.2, label="observed")
        ax.axvline(p["origin_t"], color=_MUTED, lw=0.8, ls=":")
        fams = sorted(p["fams"], key=lambda f: _FAM_ORDER.index(f["family"]))
        for k, f in enumerate(fams):
            c = FAMILY_COLORS[f["family"]]
            t = np.asarray(f["t"]) + (k - (len(fams) - 1) / 2) * 0.03
            q = np.asarray(f["q"])
            ax.vlines(t, q[0], q[4], color=c, lw=1.0, alpha=0.6)
            ax.vlines(t, q[1], q[3], color=c, lw=3.0, alpha=0.8)
            ax.scatter(t, q[2], color="white", edgecolor=c, s=16, zorder=4, lw=1.2, label=f"{f['family']}: {f['name']}")
        ax.set_title(p["title"], fontsize=8, color=_INK)
        ax.set_ylabel("Nino 3.4 anomaly [K]", fontsize=7, color=_MUTED)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_skill_vs_lead(curves: dict[str, dict], path: str, *, metrics=("acc", "rmse_skill_vs_best_baseline"),
                       labels: dict[str, str] | None = None) -> str:
    """One panel per metric: each curve (``family``, ``horizons``, values, optional ``significant`` mask) against
    lead; baselines dashed grey."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 3.4), squeeze=False)
    names = {"acc": "anomaly correlation", "rmse_skill_vs_persistence": "RMSE skill vs persistence",
             "rmse_skill_vs_best_baseline": "RMSE skill vs best baseline (val.)",
             "crps_skill_vs_climatology": "CRPS skill vs climatology", **(labels or {})}
    for ax, met in zip(axes[0], metrics, strict=False):
        _style(ax)
        if "skill" in met:
            ax.axhline(0, color=FAMILY_COLORS["baseline"], lw=1.0, ls="--")
        for name, c in curves.items():
            if met not in c:
                continue
            fam = c["family"]
            col = FAMILY_COLORS[fam]
            h = np.asarray(c["horizons"])
            v = np.asarray(c[met], float)
            ls = "--" if fam == "baseline" else "-"
            ax.plot(h, v, color=col, lw=1.8 if fam != "baseline" else 1.1, ls=ls, label=f"{fam}: {name}",
                    alpha=0.95 if fam != "baseline" else 0.8)
            sig = c.get(f"{met}_significant")
            if sig is not None:
                sig = np.asarray(sig, bool)
                ax.scatter(h[sig], v[sig], color=col, s=14, zorder=3)
                ax.scatter(h[~sig], v[~sig], facecolor="white", edgecolor=col, s=14, zorder=3, lw=1.0)
        ax.set_xlabel("lead [months]", fontsize=8, color=_MUTED)
        ax.set_ylabel(names.get(met, met), fontsize=8, color=_MUTED)
        if met == "acc":
            ax.axhline(0.5, color=_MUTED, lw=0.8, ls=":")
        if "skill" in met:
            ax.set_ylim(max(-0.6, ax.get_ylim()[0]), None)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_spring_barrier(maps: dict[str, dict], path: str, *, ncols: int = 3, key: str = "acc") -> str:
    """Heatmaps of ``key`` (ACC) by target month (rows) and lead (columns), one per model; cells with too few
    forecasts are blank, the count range is in each title."""
    plt = _plt()
    names = list(maps)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    im = None
    for ax, name in zip(axes.ravel(), names, strict=False):
        m = maps[name]
        A = np.asarray(m[key], float)
        im = ax.imshow(A, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
        H = m["horizons"]
        ax.set_xticks(range(len(H)))
        ax.set_xticklabels(H, fontsize=6)
        ax.set_yticks(range(12))
        ax.set_yticklabels(m["months"], fontsize=6)
        n = np.asarray(m["n"])
        ax.set_title(f"{name}\n(n = {n.min()}-{n.max()} per cell{', ' + m['period'] if 'period' in m else ''})",
                     fontsize=7, color=_INK)
        ax.set_xlabel("lead [months]", fontsize=7, color=_MUTED)
        ax.set_ylabel("target month", fontsize=7, color=_MUTED)
        if key == "acc":
            cs = ax.contour(A, levels=[0.5], colors=[_INK], linewidths=0.8)
            ax.clabel(cs, fmt="%.1f", fontsize=5)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.6, wspace=0.35)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="anomaly correlation")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_phase_portrait(traj: dict[str, dict], path: str) -> str:
    """(h, T) phase portraits: ``traj[name] = {"h", "T", "family" or "color", "lw", "alpha"}``; arrows show the
    direction of time on the first (observed) curve."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for k, ax in enumerate(axes):
        _style(ax)
        ax.axhline(0, color=_MUTED, lw=0.6)
        ax.axvline(0, color=_MUTED, lw=0.6)
        for i, (name, d) in enumerate(traj.items()):
            if (k == 0) != bool(d.get("left", i == 0 or d.get("family") == "observed")):
                continue
            c = d.get("color") or FAMILY_COLORS.get(d.get("family"), _INK)
            ax.plot(d["h"], d["T"], color=c, lw=d.get("lw", 1.0), alpha=d.get("alpha", 0.9),
                    label=name if d.get("label", True) else None)
            if d.get("left") and d.get("lw", 1.0) > 1.5:                   # mark the start of a model path
                ax.scatter(d["h"][0], d["T"][0], color=c, s=18, zorder=4)
            if d.get("arrows"):
                h, T = np.asarray(d["h"]), np.asarray(d["T"])
                for j in range(0, len(h) - 1, max(1, len(h) // 12)):
                    ax.annotate("", (h[j + 1], T[j + 1]), (h[j], T[j]),
                                arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
        ax.set_xlabel("h: 5S-5N D20 anomaly (WWV proxy) [m]", fontsize=8, color=_MUTED)
        ax.set_ylabel("T: Nino 3.4 anomaly [K]", fontsize=8, color=_MUTED)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[0].set_title("observed, and the fitted oscillators' mean paths from four El Nino onsets (dots)", fontsize=9,
                      color=_INK)
    axes[1].set_title("one stochastic realisation per model (same length as the record)", fontsize=9, color=_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_hovmoller(panels: list[dict], path: str, *, ncols: int = 3) -> str:
    """Hovmoller diagrams (longitude x time): each panel ``title``, ``lon``, ``t`` (fractional years), ``z`` (nt,
    nlon), ``vmax``, ``units``."""
    plt = _plt()
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.0 * nrows), squeeze=False)
    for ax, p in zip(axes.ravel(), panels, strict=False):
        v = p.get("vmax") or float(np.nanpercentile(np.abs(p["z"]), 98))
        im = ax.pcolormesh(p["lon"], p["t"], p["z"], cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
        ax.set_title(p["title"], fontsize=8, color=_INK)
        ax.set_xlabel("longitude [deg E]", fontsize=7, color=_MUTED)
        ax.tick_params(labelsize=7, colors=_MUTED)
        fig.colorbar(im, ax=ax, shrink=0.8, label=p.get("units", ""))
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_validation(kelvin: dict, gill: dict, conv: dict, ro: dict, path: str) -> str:
    """Solver checks: Kelvin-wave speed error under refinement, Gill error against resolution, recharge-oscillator
    period and growth (numerical vs analytic) against the time step."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axes[0]
    _style(ax)
    dx = [r["dlon"] for r in conv["rows"]]
    e = [abs(r["rel_error"]) for r in conv["rows"]]
    ax.loglog(dx, e, "o-", color=FAMILY_COLORS["physical"], label="|c_measured / c - 1|")
    ax.loglog(dx, e[0] * (np.asarray(dx) / dx[0]) ** 2, ls="--", color=_MUTED, label="second order")
    ax.set_xlabel("dlon [deg] (dlat, dt refined with it)", fontsize=8, color=_MUTED)
    ax.set_title(f"Kelvin wave speed (c = {kelvin['speed_theory']} m/s)", fontsize=9, color=_INK)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    _style(ax)
    r = gill["rows"]
    for k, c in (("u", "#2a78d6"), ("v", "#eb6834"), ("p", "#1baf7a")):
        ax.loglog([x["dx_La"] for x in r], [x[f"rel_l2_{k}"] for x in r], "o-", color=c, label=k)
    ax.set_xlabel("dx [L_a]", fontsize=8, color=_MUTED)
    ax.set_title("Gill (1980) response vs closed form", fontsize=9, color=_INK)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    _style(ax)
    sub = [x["sub"] for x in ro["rows"]]
    ax.semilogx(sub, [x["period_rel_error"] for x in ro["rows"]], "o-", color=FAMILY_COLORS["physical"],
                label="period")
    ax.semilogx(sub, [x["growth_rel_error"] for x in ro["rows"]], "s-", color=FAMILY_COLORS["hybrid"],
                label="growth rate")
    ax.set_xlabel("RK4 steps per month", fontsize=8, color=_MUTED)
    ax.set_title("recharge oscillator vs eigenvalues", fontsize=9, color=_INK)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

