"""Forecast-versus-truth visuals in the style of global AI weather demos: dark background, the ERA5 field and the
forecast on globes side by side, the lead time running, and a skill strip that shows, at each lead, whether the
forecast is still good (ACC above 0.6), fading, or no better than climatology.

No map library is needed: the orthographic projection and the coastlines (Natural Earth 1:110m, public domain,
bundled) are drawn here. The model grid is coarse (5.625 degrees); fields are drawn smoothly interpolated, and the
figures say so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BG, FG, DIM = "#05080d", "#e8eef5", "#7d8a99"
GOOD, FADING, POOR = "#3ddc97", "#f5b942", "#ff5d5d"
CMAPS = {"t2m": "RdYlBu_r", "t850": "RdYlBu_r", "z500": "viridis", "msl": "cividis", "u10": "RdBu_r",
         "tp6": "Blues", "wind10": "magma", "q850": "YlGnBu"}
LABEL = {"t2m": "2 m temperature (°C)", "t850": "temperature at 850 hPa (°C)", "z500": "500 hPa height (dam)",
         "msl": "sea-level pressure (hPa)", "u10": "10 m zonal wind (m/s)", "tp6": "6-hour precipitation (mm)",
         "wind10": "10 m wind speed (m/s)"}


def display_units(var: str, x: np.ndarray) -> np.ndarray:
    return {"t2m": x - 273.15, "t850": x - 273.15, "z500": x / 98.0665, "msl": x / 100.0, "tp6": x * 1000.0
            }.get(var, x)


def coastlines() -> list[np.ndarray]:
    d = np.load(Path(__file__).with_name("coastline_110m.npz"))
    return np.split(d["points"], np.cumsum(d["lengths"])[:-1])


def _sample(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, qlat: np.ndarray, qlon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (lat, lon) field (lat descending, lon from 0 with uniform step), periodic in lon."""
    dlon = lon[1] - lon[0]
    x = (np.mod(qlon - lon[0], 360.0)) / dlon
    y = (lat[0] - qlat) / (lat[0] - lat[1])
    y = np.clip(y, 0, len(lat) - 1.000001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = (x0 + 1) % len(lon), np.minimum(y0 + 1, len(lat) - 1)
    x0 = x0 % len(lon)
    return ((1 - fy) * ((1 - fx) * field[y0, x0] + fx * field[y0, x1]) +
            fy * ((1 - fx) * field[y1, x0] + fx * field[y1, x1]))


def orthographic(field, lat, lon, center=(-15.0, -50.0), size=420):
    """Image (size, size) of the field on the visible hemisphere centred at (lat, lon); NaN outside the disk."""
    la0, lo0 = np.deg2rad(center[0]), np.deg2rad(center[1])
    u = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(u, -u)
    r2 = X ** 2 + Y ** 2
    inside = r2 <= 1
    z = np.sqrt(np.clip(1 - r2, 0, None))
    qlat = np.arcsin(np.clip(Y * np.cos(la0) + z * np.sin(la0), -1, 1))
    qlon = lo0 + np.arctan2(X, z * np.cos(la0) - Y * np.sin(la0))
    img = _sample(field, lat, lon, np.rad2deg(qlat), np.rad2deg(qlon))
    return np.where(inside, img, np.nan)


def project_lines(lines, center, size=420):
    la0, lo0 = np.deg2rad(center[0]), np.deg2rad(center[1])
    out = []
    for ln in lines:
        lo, la = np.deg2rad(ln[:, 0]), np.deg2rad(ln[:, 1])
        cosc = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(lo - lo0)
        x = np.cos(la) * np.sin(lo - lo0)
        y = np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(lo - lo0)
        px, py = (x + 1) / 2 * (size - 1), (1 - y) / 2 * (size - 1)
        vis = cosc > 0
        px, py = np.where(vis, px, np.nan), np.where(vis, py, np.nan)
        out.append((px, py))
    return out


def regional(field, lat, lon, box, size=(360, 260)):
    """Equirectangular image of a region ``box`` = (lat_s, lat_n, lon_w, lon_e), smoothly interpolated."""
    la = np.linspace(box[1], box[0], size[1])
    lo = np.linspace(box[2], box[3], size[0])
    LO, LA = np.meshgrid(lo, la)
    return _sample(field, lat, lon, LA, LO)


def regional_lines(lines, box, size=(360, 260)):
    out = []
    for ln in lines:
        lo = np.where(ln[:, 0] < box[2] - 180, ln[:, 0] + 360, ln[:, 0])
        px = (lo - box[2]) / (box[3] - box[2]) * (size[0] - 1)
        py = (box[1] - ln[:, 1]) / (box[1] - box[0]) * (size[1] - 1)
        ok = (px > -5) & (px < size[0] + 5) & (py > -5) & (py < size[1] + 5)
        out.append((np.where(ok, px, np.nan), np.where(ok, py, np.nan)))
    return out


def _status(lead_h, horizon):
    if lead_h <= horizon["useful_hours"]:
        return "good", GOOD
    if lead_h <= horizon["no_skill_hours"]:
        return "fading", FADING
    return "no better than climatology", POOR


def forecast_gif(path, var, truth, forecast, lat, lon, times, *, skill_leads_h, skill_acc, horizon,
                 center=(-15.0, -50.0), region=None, title="", model_name="PINNeAPPle", vmin=None, vmax=None,
                 fps=4, dpi=90, extra_series=None):
    """Animated GIF: ERA5 (left) and forecast (right) of ``var`` for each lead, and the skill strip below.

    ``truth``/``forecast``: (n_leads, lat, lon) in physical units; ``times``: valid times; ``skill_acc``: average
    ACC of the model by lead (from the evaluation) with ``horizon`` from :func:`evaluate.horizons`;
    ``extra_series``: optional {name: (leads_h, values)} drawn faintly on the strip (other models)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    T = display_units(var, np.asarray(truth))
    F = display_units(var, np.asarray(forecast))
    lo = np.nanpercentile(T, 1) if vmin is None else vmin
    hi = np.nanpercentile(T, 99) if vmax is None else vmax
    cmap = plt.get_cmap(CMAPS.get(var, "viridis")).copy()
    cmap.set_bad(BG)
    lines = coastlines()
    if region is None:
        draw = lambda f: orthographic(f, lat, lon, center)                     # noqa: E731
        shore = project_lines(lines, center)
    else:
        draw = lambda f: regional(f, lat, lon, region)                         # noqa: E731
        shore = regional_lines(lines, region)
    fig = plt.figure(figsize=(9.6, 6.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.3], hspace=0.28, wspace=0.04, left=0.05, right=0.95,
                          top=0.88, bottom=0.1)
    axT, axF, axS = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    ims = []
    for ax, lab in ((axT, "ERA5 (what happened)"), (axF, f"{model_name} forecast")):
        ax.set_facecolor(BG)
        ax.axis("off")
        im = ax.imshow(draw(T[0]), cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
        for px, py in shore:
            ax.plot(px, py, color="#d9e2ec", lw=0.6, alpha=0.75)
        ax.set_title(lab, color=FG, fontsize=11, pad=4)
        ims.append(im)
    cb = fig.colorbar(ims[1], ax=[axT, axF], orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label(LABEL.get(var, var), color=FG, fontsize=9)
    cb.ax.tick_params(colors=DIM, labelsize=8)
    cb.outline.set_edgecolor(DIM)
    ttl = fig.text(0.05, 0.962, title, color=FG, fontsize=12.5, weight="bold")
    sub = fig.text(0.05, 0.925, "", color=DIM, fontsize=9.5)
    badge = fig.text(0.95, 0.925, "", color=GOOD, fontsize=12, weight="bold", ha="right")
    # skill strip
    axS.set_facecolor(BG)
    lh = np.asarray(skill_leads_h, float) / 24
    U, N = horizon["useful_hours"] / 24, horizon["no_skill_hours"] / 24
    xmax = lh.max()
    axS.axvspan(0, min(U, xmax), color=GOOD, alpha=0.12)
    if U < xmax:
        axS.axvspan(U, min(N, xmax), color=FADING, alpha=0.12)
    if N < xmax:
        axS.axvspan(N, xmax, color=POOR, alpha=0.12)
    for j, (name, (xs, ys)) in enumerate((extra_series or {}).items()):
        ys = fill_gaps(xs, ys)
        axS.plot(np.asarray(xs) / 24, ys, color=DIM, lw=1, alpha=0.7)
        axS.text(np.asarray(xs)[-1] / 24 - 0.05, ys[-1] + 0.06 * (j + 1), name, color=DIM, fontsize=7, ha="right")
    axS.plot(lh, skill_acc, color=FG, lw=2)
    axS.axhline(0.6, color=DIM, ls=":", lw=1)
    axS.text(0.02, 0.62, "ACC 0.6: limit of a useful forecast", color=DIM, fontsize=7, transform=axS.get_yaxis_transform())
    marker = axS.axvline(0, color=FG, lw=1.2)
    axS.set_xlim(0, xmax)
    axS.set_ylim(min(0.0, np.nanmin(skill_acc) - 0.05), 1.0)
    axS.set_xlabel("lead time (days)", color=DIM, fontsize=9)
    axS.set_ylabel("ACC", color=DIM, fontsize=9)
    axS.tick_params(colors=DIM, labelsize=8)
    for s in axS.spines.values():
        s.set_color("#253040")
    axS.set_title(f"average skill of {model_name} on 2020 forecasts ({var}): good up to {U:.1f} days, "
                  f"better than climatology up to {N:.1f} days" if np.isfinite(N) else
                  f"average skill of {model_name} on 2020 forecasts ({var}): good up to {U:.1f} days",
                  color=DIM, fontsize=8.5, loc="left")

    def frame(k):
        ims[0].set_data(draw(T[k]))
        ims[1].set_data(draw(F[k]))
        lead = k * (skill_leads_h[1] - skill_leads_h[0]) if len(skill_leads_h) > 1 else 0
        st, col = _status(lead, horizon)
        badge.set_text(f"+{lead / 24:.1f} days · {st}")
        badge.set_color(col)
        sub.set_text(f"valid {np.datetime_as_string(times[k], unit='h').replace('T', ' ')} UTC  ·  "
                     f"5.625° model grid, drawn interpolated")
        marker.set_xdata([lead / 24])
        return ims + [badge, sub, marker]

    anim = FuncAnimation(fig, frame, frames=len(T), blit=False)
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    _ = ttl
    return path


def fill_gaps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs (models published every 12 h on a 6-hour lead axis)."""
    y = np.asarray(y, float).copy()
    ok = np.isfinite(y)
    if ok.sum() >= 2:
        y[~ok] = np.interp(np.asarray(x, float)[~ok], np.asarray(x, float)[ok], y[ok], left=np.nan, right=np.nan)
    return y


def skill_figure(path, leads_h, curves: dict, horizons_by_model: dict, var: str, metric: str = "acc", dpi=110):
    """Skill by lead for every model (dark style), with the useful horizon of each marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    palette = ["#ffffff", "#5fb3ff", "#ff9f43", "#c792ea", "#3ddc97", "#ff5d5d", "#9aa5b1", "#ffd166"]
    for (name, ys), col in zip(curves.items(), palette, strict=False):
        lw = 2.4 if name.startswith("PINNeAPPle") else 1.4
        xs = np.asarray(leads_h[name] if isinstance(leads_h, dict) else leads_h)
        ax.plot(xs / 24, fill_gaps(xs, ys), color=col, lw=lw, label=name)
        h = horizons_by_model.get(name)
        if metric == "acc" and h is not None and np.isfinite(h):
            ax.plot([h / 24], [0.6], "o", color=col, ms=5)
    if metric == "acc":
        ax.axhline(0.6, color=DIM, ls=":", lw=1)
        ax.set_ylim(0, 1.0)
    ax.set_xlabel("lead time (days)", color=DIM)
    ax.set_ylabel({"acc": "anomaly correlation (ACC)", "rmse": "RMSE"}[metric] + f" · {var}", color=DIM)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values():
        s.set_color("#253040")
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=FG)
    _ = leg
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return Path(path)
