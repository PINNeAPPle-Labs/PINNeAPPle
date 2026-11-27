"""Figures for the black-hole weather study: skill against lead time, the resolution study, the model family and the
Earth-2-style animation of one forecast.

    from pinneapple_physics.blackhole.figures import plot_skill, plot_convergence, plot_family, forecast_gif
    plot_skill(skill_report, "docs/assets/blackhole/skill_vs_lead.png")
    plot_convergence(convergence_report, "docs/assets/blackhole/convergence.png")
    plot_family(family_report, "docs/assets/blackhole/family_skill.png")
    forecast_gif("docs/assets/blackhole/family_forecast.gif", ...)

One y-axis per panel, series in a fixed colour order with direct labels and line styles as a second encoding
(two of the colours are below 3:1 contrast on white, so they are never the only cue). The animation follows the
weather module's ``forecast_gif``: the truth and the forecast side by side, and below them the trust bar -- the skill
curve of the forecast, the lead where it stops being useful, and a marker that walks along it with the frame.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_skill", "plot_convergence", "plot_family", "forecast_gif"]

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, in order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GOOD, FADING, POOR, DIM = "#1baf7a", "#eda100", "#d1495b", "#8a8a8a"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _label_ends(ax, x_end, ends, min_gap=0.06):
    """Direct labels at the line ends, nudged apart vertically (``min_gap`` in axis fraction) so none overlap."""
    lo, hi = ax.get_ylim()
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    items = sorted(ends, key=lambda e: e[0])
    frac = [(y - lo) / (hi - lo) for y, _, _ in items]
    pos = list(frac)
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    for (y, text, color), f0, f in zip(items, frac, pos, strict=True):
        ax.plot([x_end], [y], "o", ms=4, color=color)
        ax.annotate(text, (x_end, y), xytext=(8, (f - f0) * height_pt), textcoords="offset points", va="center",
                    fontsize=9, color=INK, annotation_clip=False)


def plot_skill(report: dict, path: str, model: str | None = None, title: str = "") -> str:
    """Left: MAE against lead for the model and the baselines. Right: skill score against persistence with its 95 %
    interval and the skill horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = model or next(iter(report["models"]))
    leads = np.asarray(report["leads_M"])
    mae = report["mean_mae"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ends = []
    for i, (name, ls) in enumerate(((model, "-"), ("persistence", "--"), ("dmd", "-."), ("climatology", ":"))):
        y = np.asarray(mae[name])
        a1.plot(leads, y, ls, lw=2, color=SERIES[i])
        ends.append((y[-1], {"dmd": f"DMD (rank {report['dmd_rank']})"}.get(name, name), SERIES[i]))
    _label_ends(a1, leads[-1], ends)
    a1.set_xlabel("lead time (GM/c³)", color=INK)
    a1.set_ylabel("MAE of normalised log density", color=INK)
    a1.set_xlim(0, leads[-1] * 1.28)
    _style(a1)
    vp = report["models"][model]["vs_persistence"]
    ss = np.asarray(vp["skill_score"])
    a2.fill_between(leads, ss[:, 1], ss[:, 2], color=SERIES[0], alpha=0.18, lw=0)
    a2.plot(leads, ss[:, 0], lw=2, color=SERIES[0])
    a2.axhline(0, color=MUTED, lw=1)
    h = vp["horizon"]
    if h > 0:
        a2.axvline(h, color=INK, lw=1, ls="--")
        a2.annotate(f"skill horizon {h:.0f} M", (h, ss[:, 2].max()), xytext=(4, -4), textcoords="offset points",
                    fontsize=9, color=INK, va="top")
    a2.set_xlabel("lead time (GM/c³)", color=INK)
    a2.set_ylabel("skill vs persistence, 1 − MAE/MAE_pers", color=INK)
    _style(a2)
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(report: dict, path: str) -> str:
    """Time-mean shell density and specific angular momentum on every grid, and the mean accretion rate per grid
    with its 95 % interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [n for n, _ in report["levels"]]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), dpi=150)
    for ax, key, ylab, logy in ((axs[0], "rho_shell", "shell-averaged density", True),
                                (axs[1], "ell_shell", "specific angular momentum l", False)):
        p = report["profiles"][key]
        r = np.asarray(p["r"])
        for i, n in enumerate(levels):
            y = np.asarray(p["by_level"][str(n)] if str(n) in p["by_level"] else p["by_level"][n])
            ax.plot(r, y, lw=1.6, color=SERIES[i], ls=["-", "--", "-.", ":"][i % 4])
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("r (GM/c²)", color=INK)
        ax.set_ylabel(ylab, color=INK)
        _style(ax)
    axs[0].legend([f"{n} × {n // 2}" for n in levels], frameon=False, fontsize=9)
    s = report["scalars"]["mdot_mean"]["by_level"]
    m = [s[str(n)]["mean"] if str(n) in s else s[n]["mean"] for n in levels]
    c = [s[str(n)]["ci95"] if str(n) in s else s[n]["ci95"] for n in levels]
    axs[2].errorbar(levels, m, yerr=c, fmt="o", color=SERIES[0], ms=6, capsize=4, lw=1.5)
    axs[2].set_xscale("log", base=2)
    axs[2].set_xticks(levels, [str(n) for n in levels])
    axs[2].set_xlabel("radial cells", color=INK)
    axs[2].set_ylabel("mean accretion rate (code units), 95 % CI", color=INK)
    _style(axs[2])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


FAMILY_COLORS = {"duarte-unet": SERIES[0], "fourier-operator": SERIES[1], "sphere-backbone": SERIES[2],
                 "ensemble-equal": "#0b0b0b", "ensemble-adaptive": SERIES[4]}


def plot_family(report: dict, path: str, title: str = "") -> str:
    """The trust horizon of every member and of the ensemble, and the ensemble against the best single model.

    Left: MAE against lead for the three members, the two aggregations and the baselines. Middle: skill score against
    persistence with its 95 % interval, one line per member and per aggregation (the comparison the docs report).
    Right: the paired difference between each aggregation and the best single member, per lead, with its bootstrap
    interval -- the panel that answers the question of item 4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [("duarte-unet", "Duarte U-Net (paper)", "-"), ("fourier-operator", "Fourier operator", "-"),
             ("sphere-backbone", "SphereUNet backbone", "-"), ("ensemble-equal", "ensemble (equal)", "--"),
             ("ensemble-adaptive", "ensemble (adaptive)", "-."), ("persistence", "persistence", ":"),
             ("dmd", "DMD", ":")]
    leads = np.asarray(report["leads_M"])
    mae = report["mean_mae"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=150)

    ends = []
    for name, label, ls in lines:
        if name not in mae:
            continue
        y = np.asarray(mae[name])
        a1.plot(leads, y, ls, lw=2 if "ensemble" in name else 1.7, color=FAMILY_COLORS.get(name, MUTED))
        ends.append((y[-1], label, FAMILY_COLORS.get(name, MUTED)))
    _label_ends(a1, leads[-1], ends)
    a1.set_xlabel("lead time (GM/c³)", color=INK)
    a1.set_ylabel("MAE of normalised log density", color=INK)
    a1.set_xlim(0, leads[-1] * 1.55)
    _style(a1)

    skill = report["skill"]["models"]
    ends = []
    for name, label, ls in lines[:5]:
        if name not in skill:
            continue
        ss = np.asarray(skill[name]["vs_persistence"]["skill_score"])
        col = FAMILY_COLORS.get(name, MUTED)
        a2.plot(leads, ss[:, 0], ls, lw=2 if "ensemble" in name else 1.6, color=col)
        ends.append((ss[-1, 0], label, col))
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_xlabel("lead time (GM/c³)", color=INK)
    a2.set_ylabel("skill vs persistence, 1 − MAE/MAE_pers", color=INK)
    a2.set_xlim(0, leads[-1] * 1.06)
    _style(a2)
    if ends:
        _label_ends(a2, leads[-1], ends, min_gap=0.07)

    verdict = report["verdict"]["best_single"]
    for label, ls in (("ensemble-equal", "-"), ("ensemble-adaptive", "--")):
        if label not in report["paired"]:
            continue
        ss = np.asarray(report["paired"][label]["skill_score"])
        col = FAMILY_COLORS[label]
        a3.fill_between(leads, ss[:, 1], ss[:, 2], color=col, alpha=0.16, lw=0)
        a3.plot(leads, ss[:, 0], ls, lw=2, color=col,
                label=f"{label} vs {verdict}")
    a3.axhline(0, color=MUTED, lw=1)
    a3.set_xlabel("lead time (GM/c³)", color=INK)
    a3.set_ylabel(f"skill of the ensemble over {verdict}", color=INK)
    a3.set_xlim(0, leads[-1])
    _style(a3)
    a3.legend(frameon=False, fontsize=8, loc="lower left")
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def forecast_gif(path: str, grid: dict, codec, truth: np.ndarray, forecast: np.ndarray, times: np.ndarray, *,
                 acc_by_lead, horizon_M: float, title: str = "", model_name: str = "ensemble",
                 extra: np.ndarray | None = None, extra_name: str = "", weights: dict | None = None,
                 r_view: float | None = None, fps: float = 3.0, dpi: int = 78, colors: int = 150) -> str:
    """Earth-2-style animation of one forecast: the simulation and the forecast side by side, and the trust bar.

    ``truth``/``forecast``: (n_frames, H, W) normalised log density in the forecast window; ``times``: the lead of each
    frame in GM/c³. ``acc_by_lead``: the anomaly correlation of the forecast by lead (averaged over the start times),
    ``horizon_M`` the lead where it falls below 0.6 -- the bar shades the leads that are still useful, fades them
    after, and its marker walks along with the frame, so the viewer sees *when* the picture stops meaning anything.
    ``extra``: the same curve for another model (drawn faintly, e.g. the best single member). ``weights``: the
    aggregation weights, shown in the caption. Frames are assembled with Pillow and palettised, which keeps the GIF
    small: ``colors`` per frame, ``fps`` frames per second."""
    import matplotlib
    from PIL import Image
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = np.asarray(truth, float)
    forecast = np.asarray(forecast, float)
    lg_t, lg_f = np.log10(np.clip(codec.decode(truth), 1e-12, None)), np.log10(np.clip(codec.decode(forecast), 1e-12, None))
    nr_w, nt_w = lg_t.shape[1:]
    rf = np.asarray(grid["r_faces"])[: nr_w + 1]
    tf = np.asarray(grid["theta_faces"])
    trim = (len(tf) - 1 - nt_w) // 2
    tf = tf[trim:trim + nt_w + 1]
    Rf, Zf = rf[:, None] * np.sin(tf)[None, :], rf[:, None] * np.cos(tf)[None, :]
    view = float(r_view or rf[-1])
    acc = np.asarray(acc_by_lead, float)
    leads = np.asarray(times, float)
    lo, hi = codec.log_min, codec.log_max

    fig = plt.figure(figsize=(8.6, 4.9), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.25], hspace=0.30, wspace=0.06,
                          left=0.04, right=0.96, top=0.90, bottom=0.10)
    ax_t, ax_f, ax_s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    artists = []
    for ax, lab in ((ax_t, "SIMULATION"), (ax_f, f"{model_name.upper()} FORECAST")):
        ax.set_facecolor("black")
        im = ax.pcolormesh(Rf, Zf, lg_t[0], vmin=lo, vmax=hi, cmap="inferno", shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(0, view)
        ax.set_ylim(-view * 0.95, view * 0.95)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, color="#d9d9d9", fontsize=10, pad=3)
        artists.append(im)
    dt = float(np.min(np.diff(leads))) if len(leads) > 1 else float(leads[0])
    bar = dt * (1 + np.arange(len(acc)))                                     # lead of every point of the skill curve
    ax_s.plot(bar, acc, color=INK, lw=2.2)
    if extra is not None:
        ax_s.plot(bar, np.asarray(extra, float), color=MUTED, lw=1.2, ls="--", label=extra_name or "other model")
        ax_s.legend(frameon=False, fontsize=7, loc="upper right")
    ax_s.axhline(0.6, color=POOR, ls=":", lw=1)
    ax_s.axvspan(0, min(horizon_M, bar[-1]), color=GOOD, alpha=0.10)
    ax_s.axvspan(min(horizon_M, bar[-1]), bar[-1], color=POOR, alpha=0.07)
    ax_s.set_xlim(0, bar[-1])
    ax_s.set_ylim(min(0.0, float(np.nanmin(acc)) - 0.05), 1.0)
    ax_s.set_xlabel("lead time (GM/c³)", color=MUTED, fontsize=8.5)
    ax_s.set_ylabel("ACC", color=MUTED, fontsize=8.5)
    ax_s.tick_params(colors=MUTED, labelsize=8)
    for s in ax_s.spines.values():
        s.set_color(GRID)
    ax_s.set_title(f"trust bar: anomaly correlation of the forecast by lead, useful up to {horizon_M:.0f} GM/c³ "
                   f"(ACC 0.6)", color=MUTED, fontsize=8, loc="left")
    fig.suptitle(title, color=INK, fontsize=11, weight="bold", x=0.04, ha="left")
    marker = ax_s.axvline(0, color=INK, lw=1.4)
    cb = fig.colorbar(artists[0], ax=[ax_t, ax_f], orientation="horizontal", fraction=0.045, pad=0.02, aspect=55)
    cb.set_label("log10 density (g/cm³ in code units)", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    caption = fig.text(0.04, 0.93, "", color=INK, fontsize=9)

    frames = []
    for i, lead in enumerate(leads):
        artists[0].set_array(np.clip(lg_t[i].ravel(), lo, hi))
        artists[1].set_array(np.clip(lg_f[i].ravel(), lo, hi))
        j = int(np.argmin(np.abs(bar - lead)))                               # where this lead sits on the skill curve
        marker.set_xdata([bar[j], bar[j]])
        state, colour = _trust_state(float(acc[j]), float(horizon_M), float(lead))
        txt = f"lead +{lead:.0f} GM/c³ · ACC {acc[j]:.2f} · {state}"
        if weights:
            txt += "  ·  weights " + ", ".join(f"{k} {v:.2f}" for k, v in weights.items())
        caption.set_text(txt)
        caption.set_color(colour)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=colors))
    plt.close(fig)
    dur = [max(60, int(1000 / max(fps, 1e-3)))] * len(frames)
    dur[-1] = int(dur[-1] * 3)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    return path


def _trust_state(acc_now: float, horizon_M: float, lead: float) -> tuple[str, str]:
    """The caption of the trust bar at one lead: useful, fading, or no better than the time-mean flow."""
    if lead <= horizon_M:
        return "trusted", GOOD
    if acc_now > 0.0:
        return "fading", FADING
    return "past the horizon", POOR
