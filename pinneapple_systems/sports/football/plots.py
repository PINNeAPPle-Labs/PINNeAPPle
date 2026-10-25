"""Figures of the football studies (matplotlib, static PNG): pitch, pitch-control surface, ball flights in 3-D,
gravity audit, reliability curves, data-efficiency curves, smoothing errors.

Colours: a fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red) assigned per model
name so a model keeps its colour across figures; a blue-grey-orange diverging map for control probability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["COLORS", "color_of", "draw_pitch", "pitch_control_figure", "flights_3d", "gravity_audit_figure",
           "reliability_figure", "efficiency_figure", "smoothing_figure", "holdout_figure"]

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e2dc"
#: one colour per model across every figure: physics in blues/aqua/violet, data-only in orange/yellow/red, hybrids
#: in magenta/green, references in ink grey
MODEL_COLORS = {
    "pc_default": COLORS[6], "pc_reaction": COLORS[0], "pc_accel": COLORS[2], "logistic": COLORS[1], "gbm": COLORS[3],
    "hybrid": COLORS[4], "poly2": COLORS[1], "ml": COLORS[3], "vacuum": COLORS[6], "drag": COLORS[0],
    "drag_free_g": COLORS[2], "constant_velocity": COLORS[6], "physics_limited": COLORS[0],
    "gbm_residual_cv": COLORS[4], "gbm_residual_cv_projected": COLORS[5], "standard_logistic": COLORS[1],
    "standard_gbm": COLORS[3], "physics_logistic": COLORS[0], "standard+physics_logistic": COLORS[4],
    "standard+physics_gbm": COLORS[7], "standard+freeze_geometry_logistic": COLORS[5],
    "statsbomb_xg_reference": "#52514e", "raw_differences": COLORS[7], "savitzky_golay_0.5s": COLORS[3],
    "savitzky_golay_1.0s": COLORS[1], "kalman_rts": COLORS[0],
}


def color_of(name: str) -> str:
    """The fixed colour of a model name (grey for names outside :data:`MODEL_COLORS`)."""
    return MODEL_COLORS.get(name, "#8a8984")


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _fig(w=7.0, h=4.0, ncols=1, nrows=1, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, ax


def _save(fig, path) -> Path:
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def draw_pitch(ax, length: float = 105.0, width: float = 68.0, color: str = "#b9b7b0"):
    """Pitch lines in metres, origin at the centre spot."""
    L, W = length / 2, width / 2
    kw = dict(color=color, lw=1.0, zorder=1)
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], **kw)
    ax.plot([0, 0], [-W, W], **kw)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(9.15 * np.cos(th), 9.15 * np.sin(th), **kw)
    for s in (-1, 1):
        ax.plot([s * L, s * (L - 16.5), s * (L - 16.5), s * L], [-20.16, -20.16, 20.16, 20.16], **kw)
        ax.plot([s * L, s * (L - 5.5), s * (L - 5.5), s * L], [-9.16, -9.16, 9.16, 9.16], **kw)
    ax.set_xlim(-L - 3, L + 3)
    ax.set_ylim(-W - 3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def pitch_control_figure(xs, ys, q, att_pos, att_vel, def_pos, def_vel, ball, target, path, title=""):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pc", [COLORS[1], "#e8e6e1", COLORS[0]])
    fig, ax = _fig(8, 5.4)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(q, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.85,
                   zorder=0)
    draw_pitch(ax)
    for P, V, c, lab in ((att_pos, att_vel, COLORS[0], "attacking team"), (def_pos, def_vel, COLORS[1], "defending team")):
        ok = np.isfinite(P[:, 0])
        ax.scatter(P[ok, 0], P[ok, 1], s=42, color=c, edgecolor=SURFACE, lw=1.5, zorder=3, label=lab)
        ax.quiver(P[ok, 0], P[ok, 1], V[ok, 0], V[ok, 1], color=c, angles="xy", scale_units="xy", scale=1, width=0.003,
                  zorder=3)
    ax.scatter(*ball, s=30, color=INK, zorder=4, label="ball")
    ax.plot([ball[0], target[0]], [ball[1], target[1]], color=INK, lw=1.2, ls="--", zorder=4)
    ax.scatter(*target, s=60, marker="x", color=INK, zorder=4, label="pass target")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("P(attacking team controls the ball)", color=INK2, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=4, bbox_to_anchor=(0, 1.06))
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=18)
    return _save(fig, path)


def flights_3d(flights: list[tuple[np.ndarray, np.ndarray | None, str]], path, title=""):
    """``(observed xyz, fitted xyz or None, label)`` per flight, in a 3-D view over the pitch plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 5), facecolor=SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)
    for i, (obs, fit, lab) in enumerate(flights):
        c = COLORS[i % len(COLORS)]
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=10, color=c, depthshade=False, label=lab)
        if fit is not None:
            fit = fit[fit[:, 2] >= 0]
            ax.plot(fit[:, 0], fit[:, 1], fit[:, 2], color=c, lw=2)
        ax.plot(obs[:, 0], obs[:, 1], 0 * obs[:, 2], color=c, lw=0.8, alpha=0.4)
    ax.set_xlabel("x [m]", fontsize=8, color=INK2)
    ax.set_ylabel("y [m]", fontsize=8, color=INK2)
    ax.set_zlabel("height [m]", fontsize=8, color=INK2)
    ax.set_zlim(bottom=0)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    return _save(fig, path)


def gravity_audit_figure(audit: dict[str, Any], synthetic_g: list[float] | None, path):
    fig, ax = _fig(7, 3.8)
    g = np.asarray(audit["g_drag"])
    bins = np.linspace(0, 16, 41)
    ax.hist(g, bins=bins, color=COLORS[0], alpha=0.85, label=f"SkillCorner flights (n = {len(g)})",
            edgecolor=SURFACE, lw=0.8)
    if synthetic_g is not None:
        ax.hist(synthetic_g, bins=bins, color=COLORS[1], alpha=0.6, label="simulated flights, same fit",
                edgecolor=SURFACE, lw=0.8)
    ax.axvline(9.81, color=INK, lw=1.5, ls="--")
    ax.text(9.95, ax.get_ylim()[1] * 0.92, "g = 9.81", fontsize=8, color=INK)
    med = audit["g_drag_median"]
    ax.axvline(med, color=COLORS[0], lw=1.5)
    ax.text(med - 0.15, ax.get_ylim()[1] * 0.80, f"median {med:.2f}", fontsize=8, color=INK, ha="right")
    _style(ax, "Gravity fitted freely to each aerial ball flight (drag model)", "fitted g [m/s²]", "flights")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def reliability_figure(rel: dict[str, dict], path, title="", models: list[str] | None = None):
    fig, ax = _fig(5, 4.6)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    for m in models or list(rel):
        r = rel[m]
        ax.plot(r["pred"], r["obs"], "-o", color=color_of(m), lw=2, ms=5, label=m, mec=SURFACE, mew=1.5)
    _style(ax, title, "predicted probability", "observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, path)


def efficiency_figure(rows: list[dict], metric: str, path, title="", ylabel=None, models: list[str] | None = None,
                      flat: dict[str, float] | None = None):
    """Mean test ``metric`` against training-set size per model (seeds averaged, min-max band)."""
    fig, ax = _fig(6.5, 4)
    ms = models or sorted({r["model"] for r in rows})
    for m in ms:
        rr = [r for r in rows if r["model"] == m]
        ns = sorted({r["n_train"] for r in rr})
        v = [[r[metric] for r in rr if r["n_train"] == n] for n in ns]
        mean = [np.mean(x) for x in v]
        ax.plot(ns, mean, "-o", color=color_of(m), lw=2, ms=5, label=m, mec=SURFACE, mew=1.5)
        ax.fill_between(ns, [min(x) for x in v], [max(x) for x in v], color=color_of(m), alpha=0.15, lw=0)
    for m, v in (flat or {}).items():
        ax.axhline(v, color=color_of(m), lw=1.5, ls="--", label=f"{m} (no training)")
    ax.set_xscale("log")
    _style(ax, title, "training samples", ylabel or metric)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, path)


def smoothing_figure(rows: list[dict], path, metric: str = "mp_rmse", ylabel="metabolic power RMSE [W/kg]"):
    fig, ax = _fig(7.5, 3.8)
    cases = sorted({(r["fps"], r["noise_m"]) for r in rows}, key=lambda c: (-c[0], c[1]))
    methods = [m for m in dict.fromkeys(r["method"] for r in rows)]
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [next(r[metric] for r in rows if r["method"] == m and (r["fps"], r["noise_m"]) == c) for c in cases]
        x = np.arange(len(cases)) + (i - (len(methods) - 1) / 2) * w
        ax.bar(x, vals, width=w * 0.92, color=color_of(m), label=m)
    ax.set_xticks(np.arange(len(cases)), [f"{int(f)} Hz, noise {s * 100:.0f} cm" for f, s in cases])
    ax.set_yscale("log")
    _style(ax, "Smoothing on simulated runs with known truth", None, ylabel)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return _save(fig, path)


def holdout_figure(res: dict[str, Any], path, models: list[str] | None = None):
    """Distribution of landing errors per model (box plot)."""
    fig, ax = _fig(7, 3.8)
    ms = models or [k for k, v in res.items() if isinstance(v, dict)]
    data = [res[m]["landing_errors"] for m in ms]
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=0.6)
    for patch, m in zip(bp["boxes"], ms, strict=True):
        patch.set_facecolor(color_of(m))
        patch.set_edgecolor(SURFACE)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_xticks(np.arange(1, len(ms) + 1), ms, fontsize=8)
    _style(ax, "Ball landing point predicted from the first half of each flight (held-out matches)", None,
           "horizontal landing error [m]")
    return _save(fig, path)
