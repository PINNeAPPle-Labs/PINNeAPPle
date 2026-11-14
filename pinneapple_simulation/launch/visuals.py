"""The flight as a display: a globe with the ascent drawn on it, a chase view of the vehicle, and a mission-control
page whose charts follow the same time cursor.

Two 3-D scenes come out of one recorded ascent, because the two views live at scales 350 000 apart (the Earth's
6 371 km against the vehicle's 18 m) and no single camera holds both without losing the depth buffer:

* ``view="globe"``: the Earth (a sphere with the Natural Earth 1:110m land mask), the vehicle as a marker a few tens
  of kilometres long, and the path flown so far, coloured by speed.
* ``view="vehicle"``: the launcher at its own scale in a body-centred chase camera, with the plume, the trail and the
  head-up numbers.

    from pinneapple_simulation.launch.visuals import flight_scene, mission_control
    scene = flight_scene(hist, view="globe")
    pp.viz.web_viewer(scene, "viewer_globe/", single_file=True)     # opens from file://
    mission_control(hist, "mission_control.html")                    # charts + map + tiles

Scene axes are the inertial frame of the ascent (z along the Earth's spin axis), so the globe does not turn under the
trajectory: the Earth's rotation is already in the recorded positions.
"""
from __future__ import annotations

import math
import os

import numpy as np

from pinneapple_tools.visualization.studio.animation import Animation, quat_from_matrix
from pinneapple_tools.visualization.studio.scene import Scene, Surface

__all__ = ["land_mask", "globe_surfaces", "rocket_surfaces", "marker_surfaces", "flight_scene", "mission_control",
           "flight_figures"]

MATERIALS = {
    "ocean": {"color": [0.045, 0.12, 0.24], "metallic": 0.0, "roughness": 0.85},
    "land": {"color": [0.16, 0.22, 0.16], "metallic": 0.0, "roughness": 0.95},
    "rocket_white": {"color": [0.90, 0.91, 0.90], "metallic": 0.0, "roughness": 0.36},
    "rocket_dark": {"color": [0.10, 0.11, 0.13], "metallic": 0.3, "roughness": 0.5},
    "rocket_nozzle": {"color": [0.30, 0.28, 0.26], "metallic": 0.9, "roughness": 0.45},
    "plume": {"color": [1.0, 0.62, 0.20], "metallic": 0.0, "roughness": 0.4,
              "emissive": [1.0, 0.45, 0.10]},
    "marker": {"color": [1.0, 0.55, 0.15], "metallic": 0.0, "roughness": 0.4, "emissive": [1.0, 0.45, 0.10]},
}

_LAND = {}


def land_mask(cell_deg: float = 1.0):
    """``(lat, lon, mask)`` on a regular grid: whether the centre of each cell is land, from the Natural Earth
    1:110m coastlines that ship with the weather module (public domain). Cached per resolution."""
    if cell_deg in _LAND:
        return _LAND[cell_deg]
    from matplotlib.path import Path

    from pinneapple_physics.weather.viz import coastlines

    nlon = int(round(360.0 / cell_deg))
    nlat = int(round(180.0 / cell_deg))
    lat = -90.0 + (np.arange(nlat) + 0.5) * cell_deg
    lon = -180.0 + (np.arange(nlon) + 0.5) * cell_deg
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.c_[LON.ravel(), LAT.ravel()]
    mask = np.zeros(len(pts), bool)
    for line in coastlines():
        if len(line) >= 3:
            mask |= Path(np.asarray(line, float)).contains_points(pts)
    out = (lat, lon, mask.reshape(nlat, nlon))
    _LAND[cell_deg] = out
    return out


def _unit_sphere(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def globe_surfaces(radius: float = 6371.0, cell_deg: float = 1.0):
    """The Earth as two surfaces — the ocean sphere and the land cells lifted a hair above it — at ``radius``.

    ``radius`` is in whatever unit the scene uses (the globe view of the examples works in kilometres)."""
    lat, lon, mask = land_mask(cell_deg)
    nlat, nlon = mask.shape
    edges_lat = -90.0 + np.arange(nlat + 1) * cell_deg
    edges_lon = -180.0 + np.arange(nlon + 1) * cell_deg
    LA, LO = np.meshgrid(edges_lat, edges_lon, indexing="ij")
    V = _unit_sphere(LA.ravel(), LO.ravel()) * radius
    idx = np.arange((nlat + 1) * (nlon + 1)).reshape(nlat + 1, nlon + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    quads = mask.ravel()
    ocean_faces = np.vstack([np.c_[a, b, c], np.c_[a, c, d]]) if nlat * nlon else np.zeros((0, 3), int)
    land_faces = np.vstack([np.c_[a[quads], b[quads], c[quads]], np.c_[a[quads], c[quads], d[quads]]])
    land_v = V * 1.0008
    return [Surface("earth_ocean", V, ocean_faces, "ocean"),
            Surface("earth_land", land_v, land_faces, "land")]


def _tube_along_x(xs, radii, sides=12, close_start=False, close_end=False):
    """A surface of revolution about the +x axis: one ring of ``sides`` vertices per (x, r) pair."""
    ang = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    rings = [np.c_[np.full(sides, x), r * np.cos(ang), r * np.sin(ang)] for x, r in zip(xs, radii, strict=True)]
    V = np.vstack(rings)
    F = []
    for k in range(len(rings) - 1):
        b0, b1 = k * sides, (k + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            F.append([b0 + j, b0 + j2, b1 + j2])
            F.append([b0 + j, b1 + j2, b1 + j])
    if close_start:
        c = len(V)
        V = np.vstack([V, [[xs[0], 0.0, 0.0]]])
        F += [[c, rings_idx, (rings_idx + 1) % sides] for rings_idx in range(sides)]
    if close_end:
        c = len(V)
        V = np.vstack([V, [[xs[-1], 0.0, 0.0]]])
        base = (len(rings) - 1) * sides
        F += [[c, base + (j + 1) % sides, base + j] for j in range(sides)]
    return V, np.asarray(F, int)


def rocket_surfaces(length: float = 18.0, diameter: float = 1.30, plume: float = 1.0):
    """The launcher at its own scale, body frame x along the thrust axis (nose at +x), origin at the tail.

    Three stacked stages, a nose cone, four fins and (``plume`` > 0) the exhaust: a display model, not the CAD — it
    is the silhouette a chase camera needs, with the real length and diameter of the concept vehicle."""
    r = diameter / 2.0
    out = []
    V1, F1 = _tube_along_x([0.0, 0.35 * length, 0.70 * length], [r, r, r], close_start=True)
    out.append(Surface("booster", V1, F1, "rocket_white"))
    V2, F2 = _tube_along_x([0.70 * length, 0.78 * length, 0.86 * length], [r * 0.92, r * 0.9, r * 0.85])
    out.append(Surface("upper", V2, F2, "rocket_dark"))
    V3, F3 = _tube_along_x([0.86 * length, 0.9 * length, length], [r * 0.85, r * 0.8, 0.01], close_end=True)
    out.append(Surface("fairing", V3, F3, "rocket_white"))
    # four fins at the base
    Vf, Ff = [], []
    for k in range(4):
        a = k * math.pi / 2.0 + math.pi / 4.0
        ca, sa = math.cos(a), math.sin(a)
        quad = np.array([[0.07 * length, 0.0, 0.0],
                         [0.0, 1.75 * r * ca, 1.75 * r * sa],
                         [0.0, 2.10 * r * ca, 2.10 * r * sa],
                         [0.0, 0.55 * r * ca, 0.55 * r * sa]])
        base = sum(len(v) for v in Vf)
        Vf.append(quad)
        Ff.append(np.array([[0, 1, 2], [0, 2, 3]]) + base)
    out.append(Surface("fins", np.vstack(Vf), np.vstack(Ff), "rocket_dark"))
    if plume > 0:
        Vp, Fp = _tube_along_x([0.0, -2.0 * length * plume, -6.0 * length * plume],
                               [r * 0.95, r * 1.3, 0.02], close_end=True)
        out.append(Surface("plume", Vp, Fp, "plume"))
    return out


def marker_surfaces(size: float = 120.0):
    """A chevron a launcher-sized-object cannot be at globe scale: the vehicle as a marker ``size`` long along +x."""
    V, F = _tube_along_x([0.0, 0.55 * size, size], [0.28 * size, 0.28 * size, 0.001], close_start=True)
    return [Surface("vehicle_marker", V, F, "marker")]


def _body_to_inertial(directions, ups):
    """Rotation matrices (T, 3, 3) with column x along ``directions`` and z as close to ``ups`` as possible."""
    X = np.asarray(directions, float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    U = np.asarray(ups, float)
    U = U - np.sum(U * X, axis=1, keepdims=True) * X
    n = np.linalg.norm(U, axis=1, keepdims=True)
    U = np.where(n > 1e-9, U / np.maximum(n, 1e-12), np.array([0.0, 0.0, 1.0]))
    Y = np.cross(U, X)
    R = np.stack([X, Y, U], axis=2)
    return R


def _thrust_directions(hist):
    """The direction of thrust at each recorded time, from the pitch and the plane of the ascent."""
    up = hist.r / np.linalg.norm(hist.r, axis=1, keepdims=True)
    hdir = np.cross(hist.plane_normal, up)
    hdir = hdir / np.maximum(np.linalg.norm(hdir, axis=1, keepdims=True), 1e-12)
    xi = np.radians(hist.pitch_deg)
    d = np.cos(xi)[:, None] * hdir + np.sin(xi)[:, None] * up
    # during the coast the vehicle keeps its last attitude
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def flight_scene(hist, *, view: str = "globe", n_frames: int = 300, earth_radius_km: float = 6371.0):
    """The recorded ascent as a :class:`~pinneapple_tools.visualization.studio.scene.Scene` with an animation.

    ``view="globe"``: kilometres, the Earth and the marker; ``view="vehicle"``: metres, the launcher and its plume in
    a chase camera. Resample the animation with ``n_frames`` to keep the page small.
    """
    k = np.linspace(0, len(hist.t) - 1, min(n_frames, len(hist.t))).astype(int)
    t = hist.t[k]
    dirs = _thrust_directions(hist)
    up = hist.r / np.linalg.norm(hist.r, axis=1, keepdims=True)
    Q = quat_from_matrix(_body_to_inertial(dirs[k], up[k]))

    if view == "globe":
        scale = 1.0 / 1e3                              # kilometres
        pos = hist.r[k] * scale
        sc = Scene(globe_surfaces(earth_radius_km) + marker_surfaces(160.0), axes="z_up",
                   title="Ascent to orbit",
                   materials=MATERIALS)
        group = "vehicle"
        for s in sc.surfaces:
            if s.name == "vehicle_marker":
                s.group = group
        anim = Animation(t, camera="fixed", follow_group=group, time_label="T+", time_unit="s")
        anim.add_track(group, pos, Q)
        anim.add_trail("ground_track", pos, values=hist.speed[k], label="speed [m/s]", trail_seconds=None,
                       fade=False, alpha=1.0, cmap="magma")
    elif view == "vehicle":
        scale = 1.0                                    # metres
        pos = hist.r[k]
        length = hist.vehicle.stages[0].length_m + hist.vehicle.stages[1].length_m + hist.vehicle.stages[2].length_m
        parts = rocket_surfaces(length=length, diameter=hist.vehicle.stages[0].diameter_m)
        for s in parts:
            s.group = "vehicle"
        sc = Scene(parts, axes="z_up", title="Chase view",
                   materials=MATERIALS)
        anim = Animation(t, camera="follow", follow_group="vehicle", follow_offset=[-70.0, 0.0, 22.0],
                         follow_frame="body", time_label="T+", time_unit="s")
        anim.add_track("vehicle", pos, Q)
        anim.add_trail("trail", pos, color="#ff9a3c", trail_seconds=40.0, fade=True, alpha=0.9)
    else:
        raise ValueError("view must be 'globe' or 'vehicle'")

    anim.add_hud("alt", hist.altitude[k] / 1e3, "Altitude", "km", "{:.1f}")
    anim.add_hud("speed", hist.speed[k], "Speed", "m/s", "{:.0f}")
    anim.add_hud("mach", hist.mach[k], "Mach", "", "{:.2f}")
    anim.add_hud("q", hist.q[k] / 1e3, "Dyn. pressure", "kPa", "{:.2f}")
    anim.add_hud("mass", hist.mass[k], "Mass", "kg", "{:.0f}")
    anim.add_hud("thrust", hist.thrust[k] / 1e3, "Thrust", "kN", "{:.0f}")
    sc.animation = anim
    return sc


def flight_figures(hist, out_dir: str):
    """The four plots of the ascent — altitude and speed, Mach and dynamic pressure, mass and thrust, pitch — as
    PNG files in ``out_dir``; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t = hist.t
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 6.4), facecolor="#0b0f14")
    style = dict(color="#8fa3b8")
    for a in ax.ravel():
        a.set_facecolor("#0b0f14")
        a.tick_params(colors=style["color"], labelsize=8)
        for s in a.spines.values():
            s.set_color("#233040")
        a.grid(alpha=0.18, color="#3a4c60")
    ax[0, 0].plot(t, hist.altitude / 1e3, color="#57b6ff", lw=1.6)
    ax[0, 0].set_ylabel("altitude [km]", color=style["color"])
    ax[0, 1].plot(t, hist.speed / 1e3, color="#ffd166", lw=1.6, label="inertial")
    ax[0, 1].plot(t, hist.speed_rel / 1e3, color="#f78c6b", lw=1.2, ls="--", label="relative to the air")
    ax[0, 1].set_ylabel("speed [km/s]", color=style["color"])
    ax[0, 1].legend(frameon=False, labelcolor=style["color"], fontsize=8)
    ax[1, 0].plot(t, hist.mach, color="#a0e7a0", lw=1.4)
    ax[1, 0].set_ylabel("Mach", color=style["color"])
    axq = ax[1, 0].twinx()
    axq.plot(t, hist.q / 1e3, color="#c792ea", lw=1.2, ls=":")
    axq.set_ylabel("q [kPa]", color="#c792ea")
    axq.tick_params(colors="#c792ea", labelsize=8)
    ax[1, 1].plot(t, hist.mass, color="#9ad0c2", lw=1.4)
    ax[1, 1].set_ylabel("mass [kg]", color=style["color"])
    axm = ax[1, 1].twinx()
    axm.plot(t, hist.thrust / 1e3, color="#ff9a3c", lw=1.1, ls="--")
    axm.set_ylabel("thrust [kN]", color="#ff9a3c")
    axm.tick_params(colors="#ff9a3c", labelsize=8)
    for a in ax[1, :]:
        a.set_xlabel("time [s]", color=style["color"])
    fig.suptitle("Ascent: altitude, speed, Mach and dynamic pressure, mass and thrust", color="#e6edf3", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "ascent_profile.png")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig2, a2 = plt.subplots(figsize=(7.6, 4.2), facecolor="#0b0f14")
    a2.set_facecolor("#0b0f14")
    a2.tick_params(colors=style["color"], labelsize=8)
    for s in a2.spines.values():
        s.set_color("#233040")
    a2.grid(alpha=0.18, color="#3a4c60")
    a2.plot(t, hist.pitch_deg, color="#ffd166", lw=1.5, label="thrust above the horizon")
    a2.plot(t, hist.gamma_deg, color="#57b6ff", lw=1.3, ls="--", label="flight-path angle")
    a2.set_xlabel("time [s]", color=style["color"])
    a2.set_ylabel("angle [deg]", color=style["color"])
    a2.legend(frameon=False, labelcolor=style["color"], fontsize=8)
    a2.set_title("Attitude and flight path", color="#e6edf3", fontsize=10)
    p2 = os.path.join(out_dir, "ascent_angles.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    return [path, p2]


def mission_control(hist, out_html: str, *, n_frames: int = 120, cell_deg: float = 2.0, title: str = "Mission control"):
    """Write the mission-control page: a map of the ground track, the telemetry charts and the tiles, all on one
    time cursor (scrubbing a chart seeks the map). Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_physics.weather.viz import coastlines
    from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Series, Tile

    from .ascent import eci_to_geodetic

    k = np.linspace(0, len(hist.t) - 1, min(n_frames, len(hist.t))).astype(int)
    t = hist.t[k]
    lat, lon, _ = eci_to_geodetic(hist.r, hist.t, hist.earth)
    lat, lon = lat[k], lon[k]
    alt = hist.altitude[k] / 1e3

    out_dir = os.path.dirname(os.path.abspath(out_html))
    frames_dir = os.path.join(out_dir, "mission_frames")
    os.makedirs(frames_dir, exist_ok=True)
    shore = [np.asarray(c, float) for c in coastlines()]
    frames = []
    for i in range(len(t)):
        fig, a = plt.subplots(figsize=(7.2, 4.0), facecolor="#070c12")
        a.set_facecolor("#070c12")
        a.set_xlim(-180, 180)
        a.set_ylim(-90, 90)
        a.set_xticks(range(-180, 181, 60))
        a.set_yticks(range(-90, 91, 30))
        a.tick_params(colors="#5b6b7c", labelsize=7)
        for s in a.spines.values():
            s.set_color("#1b2836")
        a.grid(alpha=0.22, color="#22303f")
        for c in shore:
            a.plot(c[:, 0], c[:, 1], color="#2b4a3a", lw=0.55)
        a.plot(np.unwrap(lon[: i + 1]), lat[: i + 1], color="#ffb347", lw=1.6)
        a.plot([lon[i]], [lat[i]], marker="o", ms=5.5, color="#ff5d3b", mec="#ffd9c9", mew=0.8)
        a.set_title(f"T+{t[i]:6.1f} s    altitude {alt[i]:7.1f} km    speed {hist.speed[k][i] / 1e3:5.2f} km/s",
                    color="#d7e2ee", fontsize=9)
        fig.tight_layout(pad=0.6)
        path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        fig.savefig(path, dpi=95, facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(path)

    charts = [
        Chart("Altitude", "altitude [km]", "time [s]",
              [Series("altitude", alt)], slot="left"),
        Chart("Speed", "speed [km/s]", "time [s]",
              [Series("inertial", hist.speed[k] / 1e3), Series("air-relative", hist.speed_rel[k] / 1e3)], slot="left"),
        Chart("Dynamic pressure", "q [kPa]", "time [s]", [Series("q", hist.q[k] / 1e3)], slot="left"),
        Chart("Mach", "Mach", "time [s]", [Series("mach", hist.mach[k])], slot="right"),
        Chart("Mass", "mass [kg]", "time [s]", [Series("mass", hist.mass[k])], slot="right"),
        Chart("Thrust", "thrust [kN]", "time [s]", [Series("thrust", hist.thrust[k] / 1e3)], slot="right"),
    ]
    stage_at = hist.stage[k]
    tiles = [
        Tile("Altitude", float(alt[-1]), "km"),
        Tile("Speed", float(hist.speed[k][-1] / 1e3), "km/s"),
        Tile("Max q", float(np.max(hist.q) / 1e3), "kPa"),
        Tile("Mass", float(hist.mass[k][-1]), "kg"),
        Tile("Stages flown", int(stage_at.max()) + 1, ""),
    ]
    orb = hist.final_orbit()
    db = Dashboard(
        title,
        t,
        frames,
        charts=charts,
        tiles=tiles,
        subtitle=(f"{hist.site.name} -> {hist.target.altitude_km:.0f} km, azimuth {hist.azimuth_deg:.1f} deg, "
                  f"inserted orbit {orb['perigee_km']:.0f} x {orb['apogee_km']:.0f} km at "
                  f"{orb['inclination_deg']:.1f} deg"),
        footer=("PINNeAPPle ascent model: 3-DOF, US Standard Atmosphere 1976, two-body gravity; coastlines from "
                "Natural Earth 1:110m (public domain). A concept study, not a flight demonstration."),
        time_label="T+",
        time_unit="s",
    )
    return db.to_html(out_html)
