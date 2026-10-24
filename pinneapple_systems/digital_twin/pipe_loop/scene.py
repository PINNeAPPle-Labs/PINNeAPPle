"""3-D control-room scene of the loop: pipes coloured by pressure, a transparent elbow with its velocity profile,
the T junction, valves, tanks, the pump and the sensors coloured by the twin's verdict, frame by frame.

The layout is schematic (lengths not to scale, labelled in the report); every colour comes from a solve or from the
diagnosis:

* pipe walls: the reference pressure along each pipe, linear between its end pressures (uniform Darcy gradient of a
  constant-diameter pipe), per frame;
* elbow streamlines: the fully developed turbulent profile ``u / U = (1 - r / R)^(1/7)`` (a textbook power law, not a
  CFD solution) scaled by the bulk velocity of pipe A;
* sensor markers: 0 = consistent, 1 = flagged by the diagnosis at that frame (the isolated sensor, the obstructed
  pipe's sensors, or every model-predicted sensor when the twin is outside its domain).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .plant import SENSORS

R_PIPE = 0.12
PATHS = {                     # schematic centrelines (display units)
    "A": [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
    "C": [(4.8, 3.0, 0.0), (4.8, 6.2, 0.0)],
    "B": [(4.8, 3.0, 0.0), (8.4, 3.0, 0.0)],
}
ELBOW = {"centre": (4.0, 0.8, 0.0), "radius": 0.8}                  # pipe A turns from +x to +y
RISER = [(4.8, 0.8, 0.0), (4.8, 3.0, 0.0)]
SENSOR_POS = {"P_pump": (0.6, 0.0, 0.0), "Q_A": (2.0, 0.0, 0.0), "T_A": (3.2, 0.0, 0.0), "P_tee": (4.8, 2.6, 0.0),
              "Q_B": (5.9, 3.0, 0.0), "P_B": (7.3, 3.0, 0.0), "T_B": (7.8, 3.0, 0.0), "Q_C": (4.8, 4.1, 0.0),
              "P_C": (4.8, 5.3, 0.0), "T_C": (4.8, 5.8, 0.0)}


def _elbow_path(n: int = 24) -> np.ndarray:
    c, r = np.array(ELBOW["centre"]), ELBOW["radius"]
    a = np.linspace(-np.pi / 2, 0.0, n)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.zeros(n)], 1)


def tube(path, r: float = R_PIPE, n_theta: int = 20, n_seg: int = 40):
    """Triangulated tube along a polyline (resampled to ``n_seg`` rings); returns vertices, faces and the arc-length
    fraction 0..1 of every vertex."""
    P = np.asarray(path, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ss = np.linspace(0, s[-1], n_seg)
    C = np.stack([np.interp(ss, s, P[:, k]) for k in range(3)], 1)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    V, frac = [], []
    for i, (c, t) in enumerate(zip(C, T, strict=True)):
        up = np.array([1.0, 0.0, 0.0]) if abs(t[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        n1 = np.cross(t, up)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(t, n1)
        for a in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
            V.append(c + r * (np.cos(a) * n1 + np.sin(a) * n2))
            frac.append(ss[i] / ss[-1])
    F = []
    for i in range(n_seg - 1):
        for j in range(n_theta):
            a, b = i * n_theta + j, i * n_theta + (j + 1) % n_theta
            F += [[a, b, b + n_theta], [a, b + n_theta, a + n_theta]]
    return np.array(V), np.array(F), np.array(frac)


def _cyl(c, r, h, n=24):
    V, F, _ = tube([(c[0], c[1], c[2]), (c[0], c[1], c[2] + h)], r, n, 2)
    top = len(V)
    V = np.vstack([V, [c[0], c[1], c[2] + h], [c[0], c[1], c[2]]])
    for j in range(n):
        F = np.vstack([F, [n + j, n + (j + 1) % n, top], [j, top + 1, (j + 1) % n]])
    return V, F


def _cube(c, h):
    from pinneapple_design.thermal.board_cooling.scene import _box
    return _box(np.asarray(c) - h, np.asarray(c) + h)


def elbow_streamlines(v_bulk: float, n_r: int = 3, n_t: int = 8):
    """Lines through the elbow and the straight runs next to it at a few radii, valued by the 1/7 power-law speed."""
    path = np.vstack([np.linspace((2.6, 0, 0), (4.0, 0, 0), 8)[:-1], _elbow_path(), np.linspace(RISER[0], (4.8, 2.2, 0), 8)[1:]])
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    pts, vals = [], []
    for rr in np.linspace(0.15, 0.8, n_r):
        for a in np.linspace(0, 2 * np.pi, n_t, endpoint=False):
            off = []
            for t in T:
                n1 = np.cross(t, [0, 0, 1.0])
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(t, n1)
                off.append(rr * R_PIPE * (np.cos(a) * n1 + np.sin(a) * n2))
            pts.append(path + np.array(off))
            u = v_bulk * (1 - rr) ** (1 / 7) * 60 / 49        # U_max = (60/49) U_bulk for the 1/7 law
            vals.append(np.full(len(path), u))
    return pts, vals


def loop_scene(run: dict[str, Any], diagnosis: dict[str, Any], *, n_frames: int = 24, title: str = ""):
    """Animated scene of one monitored run: pressures on the walls, verdict on the sensors, HUD with the evidence."""
    from pinneapple_tools.visualization.studio import Lines, Scene, Surface
    from pinneapple_tools.visualization.studio.animation import Animation
    t = run["t"]
    idx = np.linspace(0, len(t) - 1, n_frames).round().astype(int)
    Yt = run["y_true"][idx]
    j = {s: SENSORS.index(s) for s in SENSORS}
    surfs, fields = [], {}
    pipeA = np.vstack([np.linspace(*PATHS["A"], 20)[:-1], _elbow_path(), np.linspace(*RISER, 20)[1:]])
    straight = {"A1": np.linspace(*PATHS["A"], 20), "A3": np.linspace(*RISER, 20)}
    s_all = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pipeA, axis=0), axis=1))]
    s_all /= s_all[-1]
    s_e0 = s_all[19]
    s_e1 = s_all[19 + len(_elbow_path()) - 1]
    for name, path, (p0, p1), lo, hi in (
            ("pipe_A_in", straight["A1"], ("P_pump", "P_tee"), 0.0, s_e0),
            ("pipe_A_riser", straight["A3"], ("P_pump", "P_tee"), s_e1, 1.0),
            ("pipe_B", PATHS["B"], ("P_tee", "P_B"), 0.0, 1.0),
            ("pipe_C", PATHS["C"], ("P_tee", "P_C"), 0.0, 1.0)):
        V, F, fr = tube(path)
        surfs.append(Surface(name, V, F, "steel", group="pipes"))
        s = lo + fr * (hi - lo)
        fields[name] = Yt[:, [j[p0]]] + (Yt[:, [j[p1]]] - Yt[:, [j[p0]]]) * s[None, :]
    V, F, _ = tube(_elbow_path(), n_seg=24)
    surfs.append(Surface("elbow_glass", V, F, "pipe_glass", group="pipes"))   # transparent: shows the flow inside
    for nm, (c, r, h, mat) in {"pump": ((0.0, -0.05, -0.5), 0.45, 0.9, "pump_blue"),
                               "tank_B": ((8.9, 3.0, -0.8), 0.6, 1.8, "steel"),
                               "tank_C": ((4.8, 6.8, -0.8), 0.6, 1.4, "steel"),
                               "heater": ((1.3, 0.0, -0.35), 0.25, 0.7, "copper")}.items():
        V, F = _cyl(c, r, h)
        surfs.append(Surface(nm, V, F, mat, group="equipment"))
    for nm, c in {"valve_B": (8.0, 3.0, 0.0), "valve_C": (4.8, 6.0, 0.0)}.items():
        V, F = _cube(c, 0.2)
        surfs.append(Surface(nm, V, F, "valve_red", group="equipment"))
    tv, tf = _cube((4.8, 3.0, 0.0), 0.18)
    surfs.append(Surface("tee", tv, tf, "steel", group="pipes"))
    # sensor status per frame from the diagnosis
    kinds = np.array(diagnosis["kinds"])[idx]
    flag = np.zeros((n_frames, len(SENSORS)))
    for f, k in enumerate(kinds):
        if k == "normal":
            continue
        if diagnosis["kind"] == "sensor" and k in ("shift", "spread"):
            flag[f, j[diagnosis["target"]]] = 1
        elif diagnosis["kind"] == "process" and k == "shift":
            for s_ in {"A": ("P_pump", "P_tee", "Q_A"), "B": ("P_tee", "P_B", "Q_B"),
                       "C": ("P_tee", "P_C", "Q_C")}[diagnosis["target"]]:
                flag[f, j[s_]] = 0.6
        else:
            flag[f] = 0.35
    for s_, c in SENSOR_POS.items():
        V, F = _cube(np.array(c) + np.array([0, 0, R_PIPE + 0.12]), 0.09)
        surfs.append(Surface(f"sensor_{s_}", V, F, "plastic", group="sensors"))
        fields[f"sensor_{s_}"] = np.repeat(flag[:, [j[s_]]], len(V), 1)
    sc = Scene(surfs, axes="z_up", title=title or "Digital twin of a pumped loop", labels={"p": "pressure [bar]"},
               materials={"pipe_glass": {"color": [0.8, 0.9, 1.0], "roughness": 0.2, "alpha": 0.12},
                          "pump_blue": {"color": [0.1, 0.3, 0.75], "metallic": 0.4, "roughness": 0.4},
                          "valve_red": {"color": [0.8, 0.12, 0.1], "metallic": 0.2, "roughness": 0.5}})
    an = Animation(t[idx] / 60.0, time_label="t", time_unit="min", time_decimals=0, interpolate=False)
    pmax = float(max(v.max() for k, v in fields.items() if not k.startswith("sensor")))
    for k, v in fields.items():
        if k.startswith("sensor"):
            an.add_surface_field(k, v, label="sensor flagged (0 ok, 1 suspect)", color_range=(0.0, 1.0))
        else:
            an.add_surface_field(k, v, label="pressure [bar]", color_range=(0.0, pmax))
    st = diagnosis["stats"]
    an.add_hud("shift", st["shift"][idx] / diagnosis["thresholds"]["shift"], "physics residual / threshold", "",
               "{:.2f}")
    an.add_hud("model", st["model"][idx] / diagnosis["thresholds"]["model"], "fast model innovation / threshold", "",
               "{:.2f}")
    an.add_hud("ood", st["ood"][idx], "outside the envelope", "box widths", "{:.2f}")
    ev = diagnosis["evidence"]
    an.add_hud("kB", ev["k_extra"][idx, 1], "Kalman: extra K pipe B", "", "{:.2f}")
    sc.animation = an
    vA = float(run["y_true"][-1, j["Q_A"]] / 3600 / (np.pi * 0.1 ** 2 / 4))
    P, Vv = elbow_streamlines(vA)
    sc.lines.append(Lines(P, Vv, name="elbow_flow", label="velocity, 1/7 power-law profile [m/s]", alpha=0.9,
                          radius=0.012, color_range=(0.0, float(max(v.max() for v in Vv)))))
    return sc
