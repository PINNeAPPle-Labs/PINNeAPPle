"""3-D renders of a forecasting study (Blender Cycles through :mod:`pinneapple_blender.cycles`).

Each render has a data function (pure NumPy, tested without Blender, returns the numbers that are drawn) and a
scene function that draws them:

* :func:`landscape_tiles` / :func:`render_landscape` -- skill against the best baseline as a surface over horizon
  (Lyapunov times for chaotic series) x model family, one tile per series, with a translucent sea at skill 0.
* :func:`dense_forecast` / :func:`render_cones` -- a direct forecaster refitted at every horizon 1..H, so the
  predictive fan exists at every step; the observed series as a glowing line and the forecast as translucent
  50 % and 90 % bands; optional clip that reveals the fan step by step.
* :func:`embedding_views` / :func:`render_attractors` -- the delay-embedded series next to its IAAFT and shuffled
  surrogates (tube, tube, point cloud).
* :func:`ensemble_fibres` / :func:`render_ensembles` -- members started from one perturbed initial state, drawn as
  fibres over time, with a marker at one Lyapunov time.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["FAMILIES", "landscape_tiles", "render_landscape", "dense_forecast", "render_cones",
           "embedding_views", "render_attractors", "ensemble_fibres", "render_ensembles"]

FAMILIES = ("data", "data+physics", "physical", "hybrid")
_FAM_RGB = {"data": (0.165, 0.471, 0.839), "data+physics": (0.922, 0.408, 0.204),
            "physical": (0.106, 0.686, 0.478), "hybrid": (0.384, 0.314, 0.839)}


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Predictability landscape
# ---------------------------------------------------------------------------

def landscape_tiles(results: Sequence[dict], *, floor: float = -0.5, titles: dict[str, str] | None = None
                    ) -> list[dict[str, Any]]:
    """One tile per case result (from ``pipeline.run_case``): the families present, the horizon axis (log10 of
    Lyapunov times when the series is chaotic, else log10 of samples) and the skill against the best baseline of
    each family's representative at every horizon, clipped below at ``floor``."""
    tiles = []
    for r in results:
        z = r["zoo"]
        H = np.asarray(z["protocol"]["horizons"], float)
        lam = r["truth"].get("lyapunov")
        x = np.log10(H * r["dt"] * lam) if lam else np.log10(H)
        fams = [f for f in FAMILIES if f in z.get("family_representative", {})]
        Z = np.array([np.maximum(np.asarray(z["models"][z["family_representative"][f]]["skill_vs_best_baseline"],
                                            float), floor) for f in fams])
        tiles.append({"case": r["case"], "title": (titles or {}).get(r["case"], r["title"]), "x": x.tolist(), "x_unit": "log10 LT" if lam else
                      "log10 samples", "families": fams,
                      "representatives": [z["family_representative"][f] for f in fams], "skill": Z.tolist(),
                      "max_skill": float(Z.max()), "horizons": H.astype(int).tolist()})
    return tiles


def _height_color(z):
    """Diverging: warm above sea level (skill > 0), blue below, light neutral at 0."""
    z = np.asarray(z, float)
    warm, mid, cool = np.array([0.90, 0.32, 0.06]), np.array([0.86, 0.85, 0.80]), np.array([0.10, 0.32, 0.80])
    t = np.clip(z, -0.5, 1.0)
    pos = (np.clip(t / 0.6, 0, 1) ** 0.6)[..., None]
    neg = (np.clip(-t / 0.3, 0, 1) ** 0.7)[..., None]
    return np.where(t[..., None] >= 0, mid + (warm - mid) * pos, mid + (cool - mid) * neg)


def render_landscape(tiles: list[dict], path: str, *, ncols: int = 4, resolution=(1600, 1000), samples: int = 64,
                     height_scale: float = 1.2) -> str:
    """Draw the tiles as surfaces (x: horizon, y: family, z: skill) on a grid of ``ncols`` columns over a dark
    studio floor, each with a translucent sea at skill 0 bounded by a glowing waterline. The families share
    one labelled axis per row (a tile with a single family is a narrow strip on that family's row); each title
    stands in front of its tile."""
    from pinneapple_blender.cycles import CyclesScene
    sc = CyclesScene(resolution=resolution, samples=samples, background=(0.045, 0.05, 0.065), ambient=0.6)
    W, D, gx, gy = 3.0, 1.6, 1.0, 1.95
    zfloor = -0.5 * height_scale - 0.06
    tilt = (32, 0, 0)                                  # text faces the camera direction (0, -1, 1.6)
    nrows = int(np.ceil(len(tiles) / ncols))
    zmax = 0.0
    fam_y = {f: i / (len(FAMILIES) - 1) * D for i, f in enumerate(FAMILIES)}
    for k, t in enumerate(tiles):
        r, c = divmod(k, ncols)
        ox, oy = c * (W + gx), -r * (D + gy)
        x = np.asarray(t["x"], float)
        Z = np.asarray(t["skill"], float)
        fams = list(t["families"])
        if len(Z) == 1:                                # a strip on that family's row
            y0 = fam_y.get(fams[0], 0.0)
            Z, yrows = np.vstack([Z, Z]), np.array([y0 - 0.17, y0 + 0.17])
        else:
            yrows = np.array([fam_y.get(f, i) for i, f in enumerate(fams)])
        xs = np.linspace(x.min(), x.max(), 48)
        rows = np.array([np.interp(xs, x, row) for row in Z])
        ys = np.linspace(yrows.min(), yrows.max(), 6 * len(rows))
        grid = np.array([np.interp(ys, yrows, rows[:, j]) for j in range(len(xs))]).T
        X, Y = np.meshgrid(ox + (xs - xs.min()) / max(np.ptp(xs), 1e-9) * W, oy + ys)
        sc.add_surface(X, Y, grid * height_scale, _height_color(grid), roughness=0.45)
        zmax = max(zmax, float(grid.max() * height_scale))
        y_lo, y_hi = oy + ys.min(), oy + ys.max()
        sc.add_plane(0.0, (W, y_hi - y_lo), center=(ox + W / 2, (y_lo + y_hi) / 2), color=(0.30, 0.60, 1.0),
                     alpha=0.22, roughness=0.05)
        sc.add_tube([(ox, y_lo, 0), (ox + W, y_lo, 0), (ox + W, y_hi, 0), (ox, y_hi, 0), (ox, y_lo, 0)], 0.012,
                    (0.45, 0.80, 1.0), emission=4.0)
        sc.add_text(t["title"], (ox + W / 2, oy - 0.62, 0.05), size=0.27, rotation=tilt, color=(0.95, 0.95, 0.93),
                    emission=1.6)
        sc.add_text(f"horizon, {t['x_unit']} ->", (ox + W / 2, oy - 0.98, 0.05), size=0.15, rotation=tilt,
                    color=(0.62, 0.68, 0.76), emission=1.2)
    for r in range(nrows):                             # one family axis per row
        oy = -r * (D + gy)
        for f in FAMILIES:
            sc.add_text(f, (-0.55, oy + fam_y[f] - 0.07, 0.05), size=0.24, rotation=tilt, align="RIGHT",
                        color=_FAM_RGB[f], emission=2.0)
    total_w = ncols * (W + gx) - gx
    y_lo, y_hi = -(nrows - 1) * (D + gy) - 1.6, D + 0.2
    sc.add_text("height and colour: RMSE skill of the family's best model vs the best baseline "
                "(warm above 0, blue below, floored at -0.5); translucent plane = skill 0",
                (total_w / 2, y_lo + 0.1, 0.05), size=0.17, rotation=tilt, color=(0.62, 0.68, 0.76), emission=1.2)
    big = (total_w + 30.0, y_hi - y_lo + 30.0)
    sc.add_plane(zfloor, big, center=(total_w / 2, (y_lo + y_hi) / 2), color=(0.022, 0.025, 0.032), alpha=1.0,
                 roughness=0.9)
    sc.fit_camera((-2.5, y_lo, zfloor), (total_w + 0.5, y_hi, zmax), direction=(0.0, -1.0, 1.6), ortho=False,
                  lens=40, margin=1.02)
    sc.light("SUN", rotation=(45, 0, -35), energy=3.2)                 # key, front left
    sc.light("SUN", rotation=(-60, 0, 160), energy=2.2)                # rim, from behind
    return sc.render(path)


# ---------------------------------------------------------------------------
# Forecast cone
# ---------------------------------------------------------------------------

def dense_forecast(data, protocol, make, max_horizon: int, *, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)
                   ) -> dict[str, Any]:
    """Fit ``make()`` on the protocol's final split with horizons 1..``max_horizon`` (every step) and forecast
    every test origin. Returns the quantiles at all test origins and the empirical coverage of the 50 % and 90 %
    bands per horizon (measured on the test period)."""
    from dataclasses import replace

    from .base import targets_at
    H = list(range(1, int(max_horizon) + 1))
    pr = replace(protocol, horizons=H)
    s = pr.final_split(len(data))
    m = make().fit(data, s["train"], s["val"], H)
    f = m.predict(data, s["test"])
    y = targets_at(data, s["test"], H)
    q = f.quantiles(quantiles)
    cov50 = np.mean((y >= q[1]) & (y <= q[3]), axis=0)
    cov90 = np.mean((y >= q[0]) & (y <= q[4]), axis=0)
    return {"model": m.name, "horizons": H, "origins": s["test"].tolist(), "quantiles": q, "truth": y,
            "coverage50": cov50.tolist(), "coverage90": cov90.tolist(), "y": data.y}


def render_cones(lanes: list[dict], path: str, *, resolution=(1280, 720), samples: int = 64,
                 frames_dir: str | None = None, n_frames: int = 0, history: int = 96) -> dict[str, Any]:
    """Draw one lane per entry (``title``, ``dense`` from :func:`dense_forecast`, ``origin_pos`` index into its
    test origins, ``std`` for scaling), stacked vertically: the observed history as an emissive tube, the 90 % and
    50 % bands as translucent ribbons, the median as a thin tube and what happened as a grey line. With
    ``frames_dir`` and ``n_frames`` also renders a clip in which the bands grow step by step."""
    from pinneapple_blender.cycles import CyclesScene
    sc = CyclesScene(resolution=resolution, samples=samples, background=(0.06, 0.07, 0.09), ambient=0.25)
    xscale, lane_gap = 6.0, 3.6
    built = []
    H = len(lanes[0]["dense"]["horizons"])
    x0 = -history / H * xscale
    for k, lane in enumerate(lanes):
        d = lane["dense"]
        zo = -lane_gap * k
        i = int(lane["origin_pos"])
        o = d["origins"][i]
        y, sd = np.asarray(d["y"], float), float(lane["std"])
        mu = float(np.mean(y))
        z = lambda v, mu=mu, sd=sd, zo=zo: zo + (np.asarray(v, float) - mu) / sd * 0.55  # noqa: E731
        th = np.arange(-history, 1) / H * xscale
        sc.add_tube(np.c_[th, np.zeros_like(th), z(y[o - history:o + 1])], 0.025, (1.0, 0.75, 0.3), emission=4.0)
        tf = np.arange(0, H + 1) / H * xscale
        q = np.asarray(d["quantiles"])[:, i, :]
        q = np.concatenate([np.full((q.shape[0], 1), y[o]), q], axis=1)
        sc.add_tube(np.c_[tf, np.full_like(tf, 0.05), z(y[o:o + H + 1])], 0.01, (0.7, 0.7, 0.7), emission=0.5)
        built.append((tf, z(q[0]), z(q[4]), z(q[1]), z(q[3]), z(q[2])))
        sc.add_text(lane["title"], (x0, -0.1, zo + 1.5), size=0.32, color=(0.92, 0.92, 0.92), rotation=(90, 0, 0),
                    align="LEFT")
    sc.add_text("observed", (x0 * 0.5, -0.1, 1.2), size=0.18, color=(1.0, 0.75, 0.3), rotation=(90, 0, 0))
    sc.add_text(f"forecast, every step to h = {H} (50 % and 90 % bands; grey: what happened)",
                (xscale * 0.5, -0.1, 1.2), size=0.18, color=(0.6, 0.8, 1.0), rotation=(90, 0, 0))
    cone_objs = []

    def draw(upto):
        for ob in cone_objs:
            sc.bpy.data.objects.remove(ob, do_unlink=True)
        cone_objs.clear()
        for tf, lo90, hi90, lo50, hi50, med in built:
            n = max(2, upto + 1)
            cone_objs.append(sc.add_ribbon(tf[:n], lo90[:n], hi90[:n], y=0.1, color=(0.35, 0.65, 1.0), alpha=0.22))
            cone_objs.append(sc.add_ribbon(tf[:n], lo50[:n], hi50[:n], y=0.08, color=(0.35, 0.65, 1.0), alpha=0.4))
            cone_objs.append(sc.add_tube(np.c_[tf[:n], np.full(n, 0.06), med[:n]], 0.015, (0.6, 0.85, 1.0),
                                         emission=2.0))

    draw(H)
    lo = (x0 - 0.3, -0.5, -lane_gap * (len(lanes) - 1) - 1.6)
    hi = (xscale + 0.3, 0.5, 1.9)
    sc.fit_camera(lo, hi, direction=(0.0, -1.0, 0.12), ortho=True, margin=1.04)
    sc.light("SUN", rotation=(55, 0, 20), energy=1.5)
    out = {"still": sc.render(path)}
    if frames_dir and n_frames:
        steps = np.linspace(1, H, int(n_frames)).astype(int)
        out["frames"] = sc.render_frames(frames_dir, len(steps), lambda s, j: draw(int(steps[j])))
    return out


# ---------------------------------------------------------------------------
# Attractor vs surrogates
# ---------------------------------------------------------------------------

def embedding_views(x, *, delay: int | None = None, n_points: int = 3000, seed: int = 0) -> dict[str, Any]:
    """Delay embeddings (dimension 3) of ``x``, of an IAAFT surrogate and of a shuffled surrogate, each
    standardised, plus the nonlinear prediction error (h = 1, NN predictor) of the three series."""
    from ..predictability import (
        delay_embedding,
        first_acf_zero,
        iaaft_surrogate,
        nonlinear_prediction_error,
        shuffle_surrogate,
    )
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    d = first_acf_zero(x) if delay is None else int(delay)
    series = {"original": x, "iaaft": iaaft_surrogate(x, rng), "shuffled": shuffle_surrogate(x, rng)}
    out = {"delay": int(d), "embeddings": {}, "npe": {}}
    for k, s in series.items():
        e = delay_embedding((s - s.mean()) / s.std(), 3, d)[:n_points]
        out["embeddings"][k] = e
        out["npe"][k] = nonlinear_prediction_error(s, dim=3, delay=max(1, min(d, 10)))
    return out


def render_attractors(views: dict, path: str, *, resolution=(1280, 480), samples: int = 64,
                      titles=("Lorenz-63 x, delay-embedded", "IAAFT surrogate", "shuffled surrogate")) -> str:
    from pinneapple_blender.cycles import CyclesScene
    sc = CyclesScene(resolution=resolution, samples=samples, background=(0.96, 0.96, 0.95), ambient=0.8)
    cols = {"original": (0.165, 0.471, 0.839), "iaaft": (0.384, 0.314, 0.839), "shuffled": (0.45, 0.45, 0.45)}
    for k, (key, title) in enumerate(zip(("original", "iaaft", "shuffled"), titles, strict=True)):
        e = views["embeddings"][key] + np.array([k * 6.5, 0, 0])
        if key == "shuffled":
            sc.add_points(e[:1500], 0.035, cols[key])
        else:
            sc.add_tube(e, 0.02, cols[key])
        sc.add_text(f"{title}\nNN prediction error / std = {views['npe'][key]:.2f}", (k * 6.5, -3.0, -3.3),
                    size=0.34, rotation=(75, 0, 0))
    sc.fit_camera((-3.3, -3.3, -4.2), (16.3, 3.3, 3.3), direction=(0.1, -1.0, 0.3), ortho=True, margin=1.02)
    sc.light("SUN", rotation=(40, 10, 30), energy=2.5)
    return sc.render(path)


# ---------------------------------------------------------------------------
# Ensemble divergence
# ---------------------------------------------------------------------------

def ensemble_fibres(step, x0: np.ndarray, *, n_members: int = 50, rel_perturbation: float = 0.01,
                    n_steps: int = 300, scale: np.ndarray | None = None, seed: int = 0,
                    components=(0, 1)) -> dict[str, Any]:
    """Integrate ``n_members`` copies of a state, each perturbed by ``rel_perturbation`` x ``scale`` (per
    component, default 1), with ``step(states) -> states`` acting on a batch (m, d). Returns the two chosen
    components over time for every member and the ensemble spread (std across members of the first component,
    divided by ``scale``) at every step."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    sc = np.ones_like(x0) if scale is None else np.asarray(scale, float)
    X = x0[None] + rel_perturbation * sc[None] * rng.standard_normal((n_members, x0.size))
    a, b = components
    traj = np.empty((n_steps + 1, n_members, 2))
    traj[0] = X[:, [a, b]]
    for k in range(1, n_steps + 1):
        X = step(X)
        traj[k] = X[:, [a, b]]
    spread = traj[:, :, 0].std(axis=1) / sc[a]
    return {"trajectories": traj, "spread": spread}


def render_ensembles(panels: list[dict], path: str, *, resolution=(1280, 900), samples: int = 48) -> str:
    """One lane per panel (``title``, ``fibres`` from :func:`ensemble_fibres`, ``times`` (n,), ``lt_time`` or None,
    ``family`` colour key), stacked vertically: members as thin tubes with time along x, the first component as
    height and the second as depth, and a translucent wall at one Lyapunov time."""
    from pinneapple_blender.cycles import CyclesScene
    sc = CyclesScene(resolution=resolution, samples=samples, background=(0.97, 0.97, 0.96), ambient=0.8)
    L, gap = 9.0, 3.4
    for k, p in enumerate(panels):
        tr = p["fibres"]["trajectories"]
        t = np.asarray(p["times"], float)
        ts = (t - t[0]) / (t[-1] - t[0]) * L
        flat = tr.reshape(-1, 2)
        mid = 0.5 * (flat.max(axis=0) + flat.min(axis=0))
        half = np.maximum(0.5 * np.ptp(flat, axis=0), 1e-9)
        zo = -gap * k
        col = _FAM_RGB.get(p.get("family"), (0.3, 0.3, 0.3))
        for m in range(tr.shape[1]):
            sc.add_tube(np.c_[ts, (tr[:, m, 1] - mid[1]) / half[1], zo + (tr[:, m, 0] - mid[0]) / half[0]], 0.007,
                        col, alpha=0.85, resolution=2)
        if p.get("lt_time"):
            xm = (p["lt_time"] - t[0]) / (t[-1] - t[0]) * L
            v = [(xm, -1.2, zo - 1.2), (xm, 1.2, zo - 1.2), (xm, 1.2, zo + 1.2), (xm, -1.2, zo + 1.2)]
            sc.add_mesh(v, [[0, 1, 2, 3]], material=sc.material((0.89, 0.29, 0.28), alpha=0.22), smooth=False)
            sc.add_text("1 Lyapunov time", (xm, -1.3, zo - 1.55), size=0.2, rotation=(90, 0, 0),
                        color=(0.75, 0.2, 0.2))
        sc.add_text(p["title"], (0.0, -1.3, zo + 1.35), size=0.26, rotation=(90, 0, 0), align="LEFT")
    lo = (-0.3, -1.3, -gap * (len(panels) - 1) - 1.8)
    hi = (L + 0.3, 1.3, 1.8)
    sc.fit_camera(lo, hi, direction=(0.12, -1.0, 0.25), ortho=True, margin=1.03)
    sc.light("SUN", rotation=(50, 0, 25), energy=2.5)
    return sc.render(path)
