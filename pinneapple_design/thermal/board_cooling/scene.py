"""3-D scene of the cooled board: enclosure, fans, board temperature map, parts, heat sink and the warm air.

Everything coloured comes from a solve:

* board top: the board-top map of :func:`.simulate.simulate` (``MAP_SHAPE``), on jet;
* parts: their junction temperature;
* heat sink: base from the 3D finite-volume base solve (its top face), fins from the 1D fin equation from the local
  root temperature (``field3d`` of :func:`pinneapple_design.thermal.heatsink.evaluate_design`);
* air paths (forced convection only): straight paths along the flow from the fans, each coloured by the bulk
  temperature of its strip of air from an energy balance (heat from the board-top cells under the strip, and from the
  heat sink for strips through the fins, over ``rho cp V A`` of the strip). Paths do not mix sideways; they show where
  the heat goes, they are not a CFD solution.

:func:`timeline_scene` animates a sequence of states (bare board, still-air heat sink, the design rounds, the verified
design) on one geometry: the heat sink has room for the largest fin count of the sequence and unused fins collapse
to a point, so the vertex count stays fixed across frames.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .mission import CoolingMission
from .simulate import MAP_SHAPE, map_axes

RHO_AIR, CP_AIR = 1.145, 1007.0          # air at 35 °C (Incropera Table A.4)
T_RANGE = (35.0, 120.0)                   # colour range of the board scenes [°C]
ENCLOSURE_MARGIN = (14.0, 18.0, 46.0)     # x, y margins and inner height of the enclosure [mm]


def _box(lo, hi):
    """Closed box, one quad (two triangles) per face with its own vertices (flat shading)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    V = np.concatenate([c[list(q)] for q in quads])
    F = np.concatenate([[[4 * i, 4 * i + 1, 4 * i + 2], [4 * i, 4 * i + 2, 4 * i + 3]] for i in range(6)])
    return V, F


def _grid(xs, ys, z):
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    ny, nx = X.shape
    i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None]
    F = np.concatenate([np.stack([i, i + 1, i + nx + 1], -1).reshape(-1, 3),
                        np.stack([i, i + nx + 1, i + nx], -1).reshape(-1, 3)])
    return V, F


def _fan(cx, y, cz, r, n_blades=7):
    """Frame ring, hub and twisted blades of an axial fan facing +y, centred at (cx, y, cz)."""
    Vs, Fs = [], []
    th = np.linspace(0, 2 * np.pi, 33)
    ring_o, ring_i = 1.08 * r, 1.0 * r
    for k in range(32):                                    # ring as a flat annulus of quads
        a, b = th[k], th[k + 1]
        q = np.array([[cx + ring_i * np.cos(a), y, cz + ring_i * np.sin(a)],
                      [cx + ring_o * np.cos(a), y, cz + ring_o * np.sin(a)],
                      [cx + ring_o * np.cos(b), y, cz + ring_o * np.sin(b)],
                      [cx + ring_i * np.cos(b), y, cz + ring_i * np.sin(b)]])
        o = sum(len(v) for v in Vs)
        Vs.append(q)
        Fs.append(np.array([[o, o + 1, o + 2], [o, o + 2, o + 3]]))
    hv, hf = _box((cx - 0.28 * r, y - 0.3 * r, cz - 0.28 * r), (cx + 0.28 * r, y + 0.1 * r, cz + 0.28 * r))
    o = sum(len(v) for v in Vs)
    Vs.append(hv)
    Fs.append(hf + o)
    for k in range(n_blades):                               # blades: quads from hub to tip, pitched along y
        a = 2 * np.pi * k / n_blades
        w = 0.32
        q = []
        for rr, da in ((0.28 * r, -w / 2), (0.95 * r, -w / 2 - 0.15), (0.95 * r, w / 2 + 0.15), (0.28 * r, w / 2)):
            q.append([cx + rr * np.cos(a + da), y + (0.12 * r if da < 0 else -0.12 * r), cz + rr * np.sin(a + da)])
        o = sum(len(v) for v in Vs)
        Vs.append(np.array(q))
        Fs.append(np.array([[o, o + 1, o + 2], [o, o + 2, o + 3]]))
    return np.concatenate(Vs), np.concatenate(Fs)


def fan_positions(mission: CoolingMission) -> tuple[list[tuple[float, float, float]], float]:
    """Centres (x, y, z) of the three fans on the inlet wall and their radius [mm].

    Returns
    -------
    centres : list[tuple[float, float, float]] [mm]
        Fan centres (x, y, z).
    radius : float [mm]
        Fan radius.
    """
    W = mission.board.width_mm
    mx, my, hz = ENCLOSURE_MARGIN
    r = min((W + 2 * mx) / 6.6, 0.42 * hz)
    return [(W * f, -my + 1.0, 0.5 * hz - 2.0) for f in (1 / 6, 1 / 2, 5 / 6)], r


def heat_sink_mesh(mission: CoolingMission, design: dict[str, Any] | None, n_max: int,
                   field3d: dict[str, Any] | None = None):
    """Vertices, faces and per-vertex temperature of the heat sink with ``n_max`` fin slots (unused fins collapsed
    to a point on the base). ``design`` None: everything collapsed (no heat sink)."""
    src = mission.hot_component()
    z0 = mission.board.thickness_mm + src.h_mm
    cx, cy = src.x_mm, src.y_mm
    if design is None:
        V, F = _box((0, 0, 0), (0, 0, 0))
        Vs = [np.tile([cx, cy, z0], (len(V), 1))]
        Fs = [F]
        for k in range(n_max):
            Vs.append(np.tile([cx, cy, z0], (len(V), 1)))
            Fs.append(F + 24 * (k + 1))
        return np.concatenate(Vs), np.concatenate(Fs), np.full(24 * (n_max + 1), mission.t_ambient_c)
    W, D, tb = design["base_width_mm"], design["base_depth_mm"], design["base_thickness_mm"]
    n, t, H = int(design["n_fins"]), design["fin_thickness_mm"], design["fin_height_mm"]
    gap = (W - n * t) / max(n - 1, 1)
    x0, y0 = cx - W / 2, cy - D / 2
    bv, bf = _box((x0, y0, z0), (x0 + W, y0 + D, z0 + tb))
    Vs, Fs, Ts = [bv], [bf], []
    Ta = mission.t_ambient_c
    if field3d is not None:
        b = field3d["base"]
        top = np.asarray(b["top_c"], float)
        bx, by = np.asarray(b["x"]), np.asarray(b["y"])

        def t_base(px, py):
            ix = np.clip(np.searchsorted(bx, px - x0), 0, len(bx) - 1)
            iy = np.clip(np.searchsorted(by, py - y0), 0, len(by) - 1)
            return top[iy, ix]
        roots = np.asarray(field3d["fins"]["root_c"], float).mean(1)
        prof_z = np.asarray(field3d["fin_profile"]["z"], float)
        prof_r = np.asarray(field3d["fin_profile"]["theta_ratio"], float)
        Ts.append(np.array([t_base(p[0], p[1]) for p in bv]))
    else:
        Ts.append(np.full(len(bv), Ta))
    for k in range(n_max):
        if k < n:
            fx = x0 + k * (t + gap)
            fv, ff = _box((fx, y0, z0 + tb), (fx + t, y0 + D, z0 + tb + H))
            if field3d is not None:
                rt = roots[min(k, len(roots) - 1)]
                ratio = np.interp(fv[:, 2] - z0 - tb, prof_z, prof_r)
                Ts.append(Ta + (rt - Ta) * ratio)
            else:
                Ts.append(np.full(len(fv), Ta))
        else:
            fv, ff = np.tile([cx, cy, z0 + tb], (24, 1)), _box((0, 0, 0), (1, 1, 1))[1]
            Ts.append(np.full(24, Ta))
        Vs.append(fv)
        Fs.append(ff + 24 * (k + 1))
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ts)


def air_paths(mission: CoolingMission, result: dict[str, Any], n_x: int = 18, heights=(4.0, 12.0, 22.0)):
    """Air paths along +y and the bulk temperature of each strip (energy balance), for a forced-air result."""
    d = result.get("design")
    V = 0.0 if d is None else d["air_velocity_m_s"]
    if V <= 0:
        return [], []
    Wb, Db = mission.board.width_mm, mission.board.depth_mm
    mx, my, hz = ENCLOSURE_MARGIN
    yc, xc = map_axes(mission.board)
    tmap = np.asarray(result["map_c"])
    h_top = result["h_board"]["top"]
    dyc = Db / MAP_SHAPE[0] * 1e-3
    src = mission.hot_component()
    sx0, sx1 = src.x_mm - d["base_width_mm"] / 2, src.x_mm + d["base_width_mm"] / 2
    sy0, sy1 = src.y_mm - d["base_depth_mm"] / 2, src.y_mm + d["base_depth_mm"] / 2
    z_sink0 = mission.board.thickness_mm + src.h_mm
    z_sink1 = z_sink0 + d["height_mm"]
    dx = (Wb + 2 * mx) / n_x
    pts, vals = [], []
    ys = np.linspace(-my + 2, Db + my - 2, 60)
    for xi in np.linspace(-mx + dx / 2, Wb + mx - dx / 2, n_x):
        ix = int(np.clip(np.searchsorted(xc, xi), 0, len(xc) - 1))
        q_board = np.where((xi >= 0) & (xi <= Wb), h_top * (tmap[:, ix] - mission.t_ambient_c) * dx * 1e-3 * dyc, 0)
        cum_board = np.interp(ys, yc, np.cumsum(q_board), left=0.0)
        through = sx0 <= xi <= sx1
        for z in heights:
            in_sink = through and z_sink0 <= z <= z_sink1
            if in_sink:                              # ducted through the fins: the sink heat of this strip
                frac = dx / d["base_width_mm"]
                q_sink = result["q_sink_w"] * min(frac, 1.0) * np.clip((ys - sy0) / (sy1 - sy0), 0, 1)
                area = dx * 1e-3 * d["height_mm"] * 1e-3
                rise = q_sink / (RHO_AIR * CP_AIR * V * area)
            else:
                area = dx * 1e-3 * hz * 1e-3
                rise = cum_board / (RHO_AIR * CP_AIR * V * area)
            zz = np.full_like(ys, z)
            pts.append(np.stack([np.full_like(ys, xi), ys, zz], 1))
            vals.append(mission.t_ambient_c + rise)
    return pts, vals


def _board_parts(mission: CoolingMission):
    """Static board surfaces: laminate box and the board-top grid (MAP_SHAPE) for the temperature field."""
    from pinneapple_tools.visualization.studio import Surface
    Wb, Db, tb = mission.board.width_mm, mission.board.depth_mm, mission.board.thickness_mm
    lv, lf = _box((0, 0, 0), (Wb, Db, tb - 0.02))
    yc, xc = map_axes(mission.board)
    xs = np.concatenate([[0.0], xc, [Wb]])
    ys = np.concatenate([[0.0], yc, [Db]])
    gv, gf = _grid(xs, ys, tb)
    return [Surface("laminate", lv, lf, "pcb")], Surface("board_top", gv, gf, "pcb", group="board")


def _pad_map(m: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(m, float), 1, mode="edge").ravel()


def _components(mission: CoolingMission):
    from pinneapple_tools.visualization.studio import Surface
    out = []
    tb = mission.board.thickness_mm
    for c in mission.components:
        r = c.resolved()
        v, f = _box((r.x_mm - r.w_mm / 2, r.y_mm - r.d_mm / 2, tb), (r.x_mm + r.w_mm / 2, r.y_mm + r.d_mm / 2,
                                                                      tb + r.h_mm))
        out.append(Surface(f"part_{c.name}", v, f, "plastic", group="parts"))
    return out


def _enclosure(mission: CoolingMission):
    from pinneapple_tools.visualization.studio import Surface
    Wb, Db = mission.board.width_mm, mission.board.depth_mm
    mx, my, hz = ENCLOSURE_MARGIN
    lo, hi = (-mx, -my, -6.0), (Wb + mx, Db + my, hz)
    v, f = _box(lo, hi)
    base_v, base_f = _box((-mx, -my, -7.0), (Wb + mx, Db + my, -6.0))
    surfs = [Surface("enclosure", v, f, "enclosure_glass", group="enclosure"),
             Surface("enclosure_floor", base_v, base_f, "anodized_black", group="enclosure")]
    e = 0.8                                                  # the 12 edges of the box as thin bars
    xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
    Vs, Fs = [], []
    for a in xs:
        for b in ys:
            Vs.append(_box((a - e, b - e, lo[2]), (a + e, b + e, hi[2])))
    for a in xs:
        for c in zs:
            Vs.append(_box((a - e, lo[1], c - e), (a + e, hi[1], c + e)))
    for b in ys:
        for c in zs:
            Vs.append(_box((lo[0], b - e, c - e), (hi[0], b + e, c + e)))
    for k, (_, bf) in enumerate(Vs):
        Fs.append(bf + 24 * k)
    surfs.append(Surface("enclosure_frame", np.concatenate([bv for bv, _ in Vs]), np.concatenate(Fs), "aluminium",
                         group="enclosure"))
    fans, r = fan_positions(mission)
    for k, (cx, y, cz) in enumerate(fans):
        fv, ff = _fan(cx, y, cz, r)
        surfs.append(Surface(f"fan_{k + 1}", fv, ff, "fan_black", group="fans"))
    return surfs


SCENE_MATERIALS = {
    "enclosure_glass": {"color": [0.85, 0.92, 1.0], "metallic": 0.0, "roughness": 0.35, "alpha": 0.06},
    "fan_black": {"color": [0.08, 0.08, 0.09], "metallic": 0.1, "roughness": 0.5},
}


def board_scene(mission: CoolingMission, result: dict[str, Any], *, title: str = "", paths: bool = True):
    """One state as a static scene: field ``T`` (°C) on the board top, the parts and the heat sink."""
    from pinneapple_tools.visualization.studio import Lines, Scene, Surface
    static, top = _board_parts(mission)
    top.fields = {"T": _pad_map(result["map_c"])}
    parts = _components(mission)
    for s, c in zip(parts, mission.components, strict=True):
        s.fields = {"T": np.full(len(s.vertices), result["tj_c"][c.name])}
    d = result.get("design")
    n = 2 if d is None else int(d["n_fins"])
    hv, hf, ht = heat_sink_mesh(mission, d, n, (result.get("sink") or {}).get("field3d"))
    surfs = static + [top] + parts
    if d is not None:
        surfs.append(Surface("heat_sink", hv, hf, "aluminium", {"T": ht}, group="heat_sink"))
    surfs += _enclosure(mission)
    sc = Scene(surfs, axes="z_up", title=title or "Cooled board", labels={"T": "temperature [°C]"},
               materials=dict(SCENE_MATERIALS))
    if paths:
        P, Vv = air_paths(mission, result)
        if P:
            sc.lines.append(Lines(P, Vv, name="air", label="air temperature [°C], energy balance",
                                  alpha=0.55, radius=0.35, color_range=(mission.t_ambient_c,
                                                                         float(max(v.max() for v in Vv)))))
    return sc


def timeline_scene(mission: CoolingMission, states: Sequence[tuple[str, dict[str, Any]]], *, title: str = "",
                   t_range: tuple[float, float] = T_RANGE):
    """The states (label, result) as frames of one animation: board map, part temperatures and heat sink shape and
    temperature change per frame; the HUD carries the hot part's junction temperature (nominal and pessimistic),
    the heat-sink mass and the fan air power. The air paths of the last state are drawn as static lines."""
    from pinneapple_tools.visualization.studio import Lines, Scene, Surface
    from pinneapple_tools.visualization.studio.animation import Animation
    n_max = max([int(r["design"]["n_fins"]) for _, r in states if r.get("design")] + [2])
    static, top = _board_parts(mission)
    parts = _components(mission)
    hs = [heat_sink_mesh(mission, r.get("design"), n_max, (r.get("sink") or {}).get("field3d")) for _, r in states]
    sink = Surface("heat_sink", hs[-1][0], hs[-1][1], "aluminium", group="heat_sink")
    sc = Scene(static + [top] + parts + [sink] + _enclosure(mission), axes="z_up",
               title=title or "The self-optimizing engineer: board temperature as the design converges",
               labels={"T": "temperature [°C]"}, materials=dict(SCENE_MATERIALS))
    an = Animation(np.arange(len(states), dtype=float), time_label="step", time_unit="", time_decimals=0,
                   interpolate=False)
    lab = f"temperature [°C] ({t_range[0]:.0f}-{t_range[1]:.0f}, clipped)"
    an.add_surface_field("board_top", np.stack([np.clip(_pad_map(r["map_c"]), *t_range) for _, r in states]),
                         label=lab, color_range=t_range)
    for s, c in zip(parts, mission.components, strict=True):
        an.add_surface_field(s.name, np.stack([np.full(len(s.vertices), np.clip(r["tj_c"][c.name], *t_range))
                                               for _, r in states]), label=lab, color_range=t_range)
    an.add_surface_shape("heat_sink", np.stack([h[0] for h in hs]))
    an.add_surface_field("heat_sink", np.stack([np.clip(h[2], *t_range) for h in hs]), label=lab,
                         color_range=t_range)
    hot = mission.hot
    an.add_hud("tj", [r["tj_c"][hot] for _, r in states], f"Tj {hot}", "°C", "{:.1f}")
    an.add_hud("tjp", [(r.get("tj_pessimistic_c") or r["tj_c"])[hot] for _, r in states],
               f"Tj {hot} pessimistic (limit {mission.tj_limit_c:.0f})", "°C", "{:.1f}")
    an.add_hud("mass", [r.get("mass_g", 0.0) for _, r in states], "heat sink", "g", "{:.1f}")
    an.add_hud("air", [(r.get("design") or {}).get("air_velocity_m_s", 0.0) for _, r in states], "fan air speed",
               "m/s", "{:.2f}")
    an.add_hud("fan", [1e3 * r.get("fan_power_w", 0.0) for _, r in states], "fan air power", "mW", "{:.1f}")
    sc.animation = an
    P, Vv = air_paths(mission, states[-1][1])
    if P:
        sc.lines.append(Lines(P, Vv, name="air", label="air temperature [°C], energy balance",
                              alpha=0.5, radius=0.35, color_range=(mission.t_ambient_c,
                                                                     float(max(v.max() for v in Vv)))))
    sc.step_labels = [lab_ for lab_, _ in states]
    return sc
