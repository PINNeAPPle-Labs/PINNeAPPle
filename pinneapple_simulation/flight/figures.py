"""Static figures of the flight-dynamics results (matplotlib PNG): turbulence spectra against their targets,
take-off trajectories, crosswind tracking, turbulence-encounter load factors and the gust-sensitivity check.

Each aircraft keeps one colour everywhere (``AIRCRAFT_COLORS``, a categorical palette checked for colour-vision
deficiency); identity is never colour alone: lines carry direct labels or a legend.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

AIRCRAFT_COLORS = {"light_ga": "#2a78d6", "regional_turboprop": "#eb6834", "a320": "#1baf7a",
                   "motor_glider": "#eda100"}
LABELS = {"light_ga": "Light GA (C172 class)", "regional_turboprop": "Turboprop (ATR 72 class)",
          "a320": "Airliner (A320 class)", "motor_glider": "Motor glider (AR 28)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FT, KT = 0.3048, 0.514444


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


def _fig(nrows, ncols, w=11.0, h=3.4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, h * nrows), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for a in axs.ravel():
        a.set_facecolor("#fcfcfb")
    return fig, axs


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def psd_figure(curves: Sequence[Mapping], path: str) -> str:
    """One panel per entry of ``curves``: dict(title, W (rad/m or rad/s), measured, target, xlabel, markers=[(W, label)]).
    """
    n = len(curves)
    fig, axs = _fig(1, n, w=3.6 * n, h=3.3)
    for ax, c in zip(axs[0], curves, strict=True):
        ax.loglog(c["W"], c["measured"], color="#2a78d6", lw=1.2, label="generated (Welch)")
        ax.loglog(c["W"], c["target"], color=INK, lw=1.4, ls="--", label="MIL-F-8785C target")
        for k, lab in c.get("markers", []):                     # e.g. the field's resolution cutoff, 2 pi / span
            ax.axvline(k, color=MUTED, lw=0.9, ls=":")
            ax.text(k, ax.get_ylim()[0] * 3, f" {lab}", rotation=90, fontsize=7, color=INK, va="bottom")
        _style(ax, c["title"], c.get("xlabel", "spatial frequency [rad/m]"), "PSD")
    axs[0][0].legend(fontsize=8, frameon=False)
    return _save(fig, path)


def takeoff_figure(runs: Mapping[str, object], metrics: Mapping[str, Mapping], path: str) -> str:
    """Height of the lowest wheel and airspeed against runway distance, pitch attitude against time."""
    fig, axs = _fig(1, 3, w=13.0, h=3.6)
    for name, h in runs.items():
        col = AIRCRAFT_COLORS.get(name, MUTED)
        x = h.x[:, 0] - h.x[0, 0]
        hgt = h.altitude - h.altitude[0]
        axs[0][0].plot(x, hgt / FT, color=col, lw=2)
        axs[0][1].plot(x, h.airspeed / KT, color=col, lw=2)
        axs[0][2].plot(h.t, np.degrees(h.euler[:, 1]), color=col, lw=2, label=LABELS.get(name, name))
        m = metrics.get(name, {})
        if "distance_35ft_m" in m:
            axs[0][0].plot([m["distance_35ft_m"]], [35], "o", color=col, ms=6, mec="#fcfcfb", mew=1.5)
            off = {"motor_glider": (-6, 8), "light_ga": (6, -14)}.get(name, (4, 6))
            axs[0][0].annotate(f"{m['distance_35ft_m']:.0f} m", (m["distance_35ft_m"], 35), textcoords="offset points",
                               xytext=off, fontsize=8, color=INK, ha="right" if off[0] < 0 else "left")
    axs[0][0].axhline(35, color=MUTED, lw=0.8, ls=":")
    _style(axs[0][0], "Height of the CG above its rest height", "distance from brake release [m]", "ft")
    _style(axs[0][1], "Airspeed", "distance from brake release [m]", "kt")
    _style(axs[0][2], "Pitch attitude", "time [s]", "deg")
    axs[0][2].legend(fontsize=8, frameon=False)
    return _save(fig, path)


def lateral_figure(runs: Mapping[str, object], path: str, title: str = "Crosswind take-off") -> str:
    """Lateral deviation from the centreline, bank angle and rudder against time for one or more runs."""
    fig, axs = _fig(1, 3, w=13.0, h=3.4)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for (lab, h), col in zip(runs.items(), palette, strict=False):
        col = AIRCRAFT_COLORS.get(lab, col)
        axs[0][0].plot(h.t, h.x[:, 1], color=col, lw=2, label=LABELS.get(lab, lab))
        axs[0][1].plot(h.t, np.degrees(h.euler[:, 0]), color=col, lw=2)
        axs[0][2].plot(h.t, np.degrees(h.controls["rudder"]), color=col, lw=2)
    _style(axs[0][0], f"{title}: lateral offset", "time [s]", "m (east +)")
    _style(axs[0][1], "Bank angle", "time [s]", "deg")
    _style(axs[0][2], "Rudder", "time [s]", "deg")
    axs[0][0].legend(fontsize=8, frameon=False)
    return _save(fig, path)


def encounter_traces_figure(runs: Mapping[str, object], path: str, window: float = 20.0, t0: float = 10.0,
                            notes: Mapping[str, str] | None = None) -> str:
    """Small multiples: vertical load-factor increment at the CG and roll angle, one row per aircraft; ``notes``
    adds a remark to an aircraft's row title (e.g. the turbulence level shown)."""
    notes = notes or {}
    n = len(runs)
    fig, axs = _fig(n, 2, w=12.0, h=1.9)
    for i, (name, h) in enumerate(runs.items()):
        col = AIRCRAFT_COLORS.get(name, MUTED)
        sel = (h.t >= t0) & (h.t <= t0 + window)
        axs[i][0].plot(h.t[sel], h.nz[sel] - 1, color=col, lw=1.2)
        axs[i][1].plot(h.t[sel], np.degrees(h.euler[sel, 0]), color=col, lw=1.2)
        lab = LABELS.get(name, name) + (f" ({notes[name]})" if name in notes else "")
        _style(axs[i][0], f"{lab}: Δn_z at the CG" if i == 0 else lab,
               "time [s]" if i == n - 1 else None, "g")
        _style(axs[i][1], "Bank angle" if i == 0 else None, "time [s]" if i == n - 1 else None, "deg")
    return _save(fig, path)


def encounter_bars_figure(table: Sequence[Mapping], path: str) -> str:
    """RMS vertical load-factor increment at the CG and at the cabin stations, per aircraft (grouped bars). Rows with
    ``status == "departed"`` (the run left the model's validity) get a note instead of bars."""
    fig, axs = _fig(1, 2, w=12.0, h=3.6)
    names = [r["aircraft"] for r in table]
    xs = np.arange(len(names))
    ax = axs[0][0]
    for i, r in enumerate(table):
        if r.get("status") == "departed":
            for a_ in axs[0]:
                a_.text(i, 0.0, "departed:\noutside the model", ha="center", va="bottom", fontsize=7.5, color=MUTED)
            continue
        st = [("CG", r["dnz_cg_rms"])] + [(k, v["dnz_rms"]) for k, v in r["cabin"].items()]
        w = 0.8 / len(st)
        for j, (lab, v) in enumerate(st):
            ax.bar(i - 0.4 + (j + 0.5) * w, v, w * 0.9, color=AIRCRAFT_COLORS.get(r["aircraft"], MUTED),
                   alpha=1.0 if j == 0 else 0.55)
            ax.text(i - 0.4 + (j + 0.5) * w, v, lab.replace("_", " "), rotation=90, fontsize=6.5, color=INK,
                    ha="center", va="bottom")
    ax.set_xticks(xs, [LABELS.get(n, n) for n in names], fontsize=7.5)
    _style(ax, "RMS Δn_z: CG (solid) and cabin stations (light)", None, "g")
    ax = axs[0][1]
    for i, r in enumerate(table):
        if r.get("status") == "departed":
            continue
        ax.bar(i, r["phi_rms_deg"], 0.55, color=AIRCRAFT_COLORS.get(r["aircraft"], MUTED))
        ax.text(i, r["phi_rms_deg"], f"{r['phi_rms_deg']:.1f}°", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs, [LABELS.get(n, n) for n in names], fontsize=7.5)
    _style(ax, "RMS bank angle", None, "deg")
    return _save(fig, path)


def gust_sensitivity_figure(table: Sequence[Mapping], pratt: Sequence[Mapping], path: str) -> str:
    """Left: measured RMS Δn_z against the quasi-steady prediction A sigma_w (A = rho V CL_a / 2 W/S), y = x for
    reference. Right: discrete 1-cosine gust peak from the simulation against the Pratt formula."""
    fig, axs = _fig(1, 2, w=11.0, h=3.8)
    ax = axs[0][0]
    hi = 0.0
    table = [r for r in table if r.get("status") != "departed"]
    for r in table:
        col = AIRCRAFT_COLORS.get(r["aircraft"], MUTED)
        mk = "o" if r.get("controller") == "stick_fixed" else "s"
        ax.plot(r["quasi_steady_dnz_rms"], r["dnz_cg_rms"], mk, color=col, ms=8, mec="#fcfcfb", mew=1.5)
        hi = max(hi, r["quasi_steady_dnz_rms"], r["dnz_cg_rms"])
    ax.plot([0, hi * 1.1], [0, hi * 1.1], color=MUTED, lw=0.8, ls="--")
    ax.text(hi * 0.75, hi * 0.8, "y = x", fontsize=8, color=MUTED)
    from matplotlib.lines import Line2D
    seen = list(dict.fromkeys(r["aircraft"] for r in table))
    ax.legend([Line2D([], [], marker="o", ls="", color=AIRCRAFT_COLORS.get(n, MUTED), ms=7) for n in seen],
              [LABELS.get(n, n) for n in seen], fontsize=7.5, frameon=False, loc="upper left")
    _style(ax, "RMS Δn_z vs quasi-steady A·σ_w\n○ stick fixed, □ autopilot; departed runs left out",
           "A·σ_w (fixed wing, no alleviation) [g]", "simulated [g]")
    ax = axs[0][1]
    hi = 0.0
    for r in pratt:
        col = AIRCRAFT_COLORS.get(r["aircraft"], MUTED)
        ax.plot(r["dn_pratt"], r["dn_sim"], "o", color=col, ms=8, mec="#fcfcfb", mew=1.5)
        off = {"light_ga": (8, -12), "motor_glider": (-8, 8)}.get(r["aircraft"], (6, -10))
        ax.annotate(LABELS.get(r["aircraft"], r["aircraft"]), (r["dn_pratt"], r["dn_sim"]), textcoords="offset points",
                    xytext=off, fontsize=7.5, color=INK, ha="right" if off[0] < 0 else "left")
        hi = max(hi, r["dn_pratt"], r["dn_sim"])
    ax.plot([0, hi * 1.15], [0, hi * 1.15], color=MUTED, lw=0.8, ls="--")
    _style(ax, "1-cosine gust (H = 12.5 MAC): peak Δn", "Pratt-Walker formula [g]", "simulated, stick fixed [g]")
    return _save(fig, path)


def table_markdown(rows: Sequence[Mapping], cols: Sequence[tuple], digits: dict[str, int] | None = None) -> str:
    """Markdown table from dict rows; ``cols`` = (key, header) pairs; floats rounded per ``digits`` (default 2)."""
    digits = digits or {}
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, float) and math.isfinite(v):
                nd = digits.get(k, 2)
                v = f"{v:.2e}" if 0 < abs(v) < 0.5 * 10 ** -nd else f"{v:.{nd}f}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
