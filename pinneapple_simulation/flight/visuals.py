"""3-D scenes of a flight: the aircraft flying its recorded trajectory over a runway, the gust field around it as
arrows, the wing and tail loading (vortex-lattice dCp) and a head-up display, as a
:class:`pinneapple_tools.visualization.studio.Scene` with an :class:`~pinneapple_tools.visualization.studio.animation.
Animation` (browser viewer with a time slider, Blender renders and clips).

Scene axes ``z_up``: x north (along the runway), y west, z up; the aircraft's surfaces are given in its body frame
(x forward, y left, z up) about the CG and moved by the track ``aircraft``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .aero import T_AB
from .aircraft import Aircraft
from .rigid_body import quat_to_dcm
from .simulate import FlightHistory

S_FLIP = np.diag([1.0, -1.0, -1.0])            # NED <-> NWU and body FRD <-> FLU
KT, FT = 0.514444, 0.3048

RUNWAY_MATERIALS = {
    "asphalt": {"color": [0.16, 0.165, 0.17], "metallic": 0.0, "roughness": 0.92},
    "marking": {"color": [0.92, 0.92, 0.90], "metallic": 0.0, "roughness": 0.7},
    "grass": {"color": [0.30, 0.42, 0.20], "metallic": 0.0, "roughness": 0.95},
    "lattice": {"color": [0.95, 0.95, 0.95], "metallic": 0.0, "roughness": 0.6},
}


def _register_materials():
    import pinneapple_design.aero.airliner  # noqa: F401  (adds the airliner materials)
    from pinneapple_design.aero.airframe import MATERIALS as AERO
    from pinneapple_tools.visualization.studio.scene import MATERIALS
    for k, v in {**AERO, **RUNWAY_MATERIALS}.items():
        MATERIALS.setdefault(k, v)


def body_vertices(ac: Aircraft, V_airframe: np.ndarray) -> np.ndarray:
    """Airframe-axes points (x aft, y right, z up) -> body frame x forward, y left, z up about the CG."""
    return (np.asarray(V_airframe, float) - ac.cg) @ (S_FLIP @ T_AB).T


def aircraft_surfaces(ac: Aircraft, group: str = "aircraft", parts=None):
    """The visual geometry of the aircraft (its ``parts_fn``: airframe parts or geometry-library bodies) as studio
    surfaces in the body frame."""
    from pinneapple_tools.visualization.studio.scene import Surface
    _register_materials()
    parts = parts if parts is not None else (ac.parts_fn() if getattr(ac, "parts_fn", None) else [])
    out = []
    for p in parts:
        mats = getattr(p, "materials", None)
        if mats is not None and not isinstance(mats, str):           # geometry-library body, one material per face
            fm = p.face_materials()
            for m in dict.fromkeys(fm):
                sel = fm == m
                out.append(Surface(f"{p.name}__{m}", body_vertices(ac, p.vertices), p.faces[sel], str(m), group=group))
        else:
            out.append(Surface(p.name, body_vertices(ac, p.vertices), p.faces, mats or p.material, group=group))
    return out


def lattice_surface(ac: Aircraft, group: str = "aircraft", lift: float = 0.6):
    """The vortex-lattice panels as a sheet just above the upper skin (offset ``lift`` x local thickness), one quad
    (two triangles, four own vertices) per panel, in the body frame."""
    from pinneapple_design.aero.geometry import properties
    from pinneapple_tools.visualization.studio.scene import Surface
    L = ac.aero.L
    Q = ac.aero.panel_quads().copy()
    tc = properties(ac.af.section)["thickness"]
    chord = L.chord[L.strip]
    Q[:, :, 2] += (lift * tc * chord + 0.01)[:, None]
    V = body_vertices(ac, Q.reshape(-1, 3))
    n = len(Q)
    base = 4 * np.arange(n)[:, None]
    F = np.vstack([base + [0, 1, 2], base + [0, 2, 3]])
    return Surface("vlm_load", V, F, "lattice", group=group)


def runway_surfaces(length: float = 3000.0, width: float = 45.0, x0: float = -300.0, grass: float = 1500.0):
    """Runway along +x (north) from x0, centreline dashes, threshold bars, grass around (scene axes z up)."""
    from pinneapple_tools.visualization.studio.scene import Surface
    _register_materials()

    def rect(xa, xb, ya, yb, z):
        V = np.array([[xa, ya, z], [xb, ya, z], [xb, yb, z], [xa, yb, z]])
        return V, np.array([[0, 1, 2], [0, 2, 3]])

    out = [Surface("grass", *rect(x0 - grass, x0 + length + grass, -grass, grass, -0.05), "grass"),
           Surface("runway", *rect(x0, x0 + length, -width / 2, width / 2, 0.0), "asphalt")]
    Vs, Fs = [], []
    for k, xa in enumerate(np.arange(x0 + 120.0, x0 + length - 120.0, 60.0)):
        V, F = rect(xa, xa + 30.0, -0.45, 0.45, 0.02)
        Vs.append(V)
        Fs.append(F + 4 * k)
    nb = len(Vs)
    for j, y in enumerate(np.linspace(-width / 2 + 3, width / 2 - 3, 12)):
        V, F = rect(x0 + 6.0, x0 + 6.0 + 30.0, y - 0.9, y + 0.9, 0.02)
        Vs.append(V)
        Fs.append(F + 4 * (nb + j))
    out.append(Surface("runway_marking", np.vstack(Vs), np.vstack(Fs), "marking"))
    return out


def frame_indices(hist: FlightHistory, n_frames: int) -> np.ndarray:
    t = np.linspace(hist.t[0], hist.t[-1], n_frames)
    return np.clip(np.searchsorted(hist.t, t), 0, len(hist.t) - 1)


def flight_scene(ac: Aircraft, hist: FlightHistory, *, n_frames: int = 240, wind=None, runway: bool = True,
                 runway_length: float | None = None, gust_grid: int = 7, gust_scale: float | None = None,
                 load: bool = True, title: str | None = None, follow: bool = True,
                 parts=None, arrows: bool = True, wisps: bool = False, trail_nz: bool = True,
                 gust_arrows: str = "vertical") -> Any:
    """Animated scene of ``hist``. ``wind``: the field the aircraft flew through, sampled on a heading-aligned
    horizontal grid around the aircraft (``gust_grid`` x ``gust_grid`` points over 2 spans, at the CG height and one
    semi-span above and below) and drawn as arrows coloured by the vertical gust; ``load``: the lattice dCp per frame
    (needs ``hist.gamma``, i.e. ``record_gamma=True``); ``arrows`` / ``wisps``: the gust as moving arrows and / or
    as static wisps along the flown corridor (:func:`gust_wisps`); ``trail_nz``: the trail coloured by n_z;
    ``gust_arrows``: "vertical" draws the vertical gust w only, as vertical arrows (what the colour bar shows; they are
    never seen end-on from the chase views), "full" the whole wind vector."""
    from pinneapple_tools.visualization.studio.animation import Animation, quat_from_matrix
    from pinneapple_tools.visualization.studio.scene import Scene
    ks = frame_indices(hist, n_frames)
    t = hist.t[ks]
    X = hist.x[ks]
    pos = X[:, 0:3] @ S_FLIP.T
    R = np.array([S_FLIP @ quat_to_dcm(q) @ S_FLIP for q in X[:, 6:10]])
    quats = quat_from_matrix(R)
    surfs = aircraft_surfaces(ac, parts=parts)
    anim = Animation(t)
    anim.add_track("aircraft", pos, quats)
    if trail_nz:
        anim.add_trail("trajectory", pos, values=hist.nz[ks], label="Load factor n_z [g]", cmap="jet")
    else:
        anim.add_trail("trajectory", pos)
    if load and hist.gamma is not None:
        lat = lattice_surface(ac)
        surfs.append(lat)
        L = ac.aero.L
        dx = L.chord[L.strip] / ac.aero.nc
        dcp = 2 * hist.gamma[ks] / (np.maximum(hist.airspeed[ks], 1.0)[:, None] * dx[None])
        lo, hi = np.percentile(dcp, [1, 99])
        anim.add_surface_field("vlm_load", np.repeat(dcp, 4, axis=1), label="Wing load dCp (lattice)",
                               color_range=(float(min(lo, 0.0)), float(hi)))
    lines = []
    if wind is not None and wisps:
        lines.append(gust_wisps(ac, hist, wind))
    if wind is not None and arrows:
        b = ac.af.span
        g = np.linspace(-b, b, gust_grid)
        GX, GY = np.meshgrid(g, g, indexing="ij")
        levels = [-b / 2, 0.0, b / 2]
        Org, W = [], []
        for k, x in zip(ks, X, strict=True):
            psi = math.atan2(*(quat_to_dcm(x[6:10]) @ np.array([1.0, 0, 0]))[1::-1])
            c, s = math.cos(psi), math.sin(psi)
            P = np.array([[x[0] + c * a - s * bb, x[1] + s * a + c * bb, x[2] + dz] for dz in levels
                          for a, bb in zip(GX.ravel(), GY.ravel(), strict=True)])
            Org.append(P @ S_FLIP.T)
            W.append(wind.velocity(hist.t[k], P) @ S_FLIP.T)
        Org, W = np.array(Org), np.array(W)
        Wd = W * np.array([0.0, 0.0, 1.0]) if gust_arrows == "vertical" else W
        sc = gust_scale or 0.5 * b / max(float(np.abs(Wd).max()), 1e-6)
        anim.add_vectors("gust", Org, Wd * sc, values=W[..., 2], label="Vertical gust w [m/s] (up +)", cmap="coolwarm",
                         color_range=(-float(np.percentile(np.abs(W[..., 2]), 98)),
                                      float(np.percentile(np.abs(W[..., 2]), 98))))
    anim.add_hud("V", hist.airspeed[ks] / KT, "Airspeed", "kt", "{:.0f}")
    if runway:                                     # height of the CG above its height at rest on the gear
        anim.add_hud("h", (hist.altitude[ks] - hist.altitude[0]) / FT, "Height", "ft", "{:.0f}")
    else:
        anim.add_hud("h", hist.altitude[ks] / FT, "Altitude", "ft", "{:.0f}")
    anim.add_hud("nz", hist.nz[ks], "n_z", "g", "{:.2f}")
    anim.add_hud("theta", np.degrees(hist.euler[ks, 1]), "Pitch", "deg", "{:.1f}")
    anim.add_hud("phi", np.degrees(hist.euler[ks, 0]), "Bank", "deg", "{:.1f}")
    if follow:
        anim.camera = "follow"
        anim.follow_group = "aircraft"
    if runway:
        L = runway_length or max(1.3 * float(hist.x[:, 0].max()) + 600.0, 1500.0)
        width = 45.0 if ac.af.span > 25 else 25.0
        surfs = runway_surfaces(L, width) + surfs
    sc = Scene(surfs, lines=lines, axes="z_up", title=title or f"{ac.info.get('type', ac.name)}")
    sc.animation = anim
    return sc


def gust_wisps(ac: Aircraft, hist: FlightHistory, wind, n_lat: int = 5, n_vert: int = 3, spacing: float | None = None,
               exaggeration: float = 0.6, ahead: float = 0.15):
    """The frozen gust field as static wisps along the flown track: ``n_lat`` x ``n_vert`` lines offset from the CG
    track by +-1 span on the local horizontal normal and +-0.5 span vertically (extended by ``ahead`` x the flown
    distance along the last heading), sampled every ``spacing`` (default span / 4) of track length, each point lifted
    by ``exaggeration`` x the vertical gust [m per m/s] and coloured by it (up positive) on a diverging scale."""
    from pinneapple_tools.visualization.studio.scene import Lines
    b = ac.af.span
    P = hist.x[:, :3]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_cum = np.r_[0.0, np.cumsum(seg)]
    L = float(s_cum[-1]) or 1.0
    ds = spacing or b / 4
    s = np.arange(0.0, L, ds)
    track = np.stack([np.interp(s, s_cum, P[:, k]) for k in range(3)], 1)
    head = P[-1] - P[max(len(P) - 20, 0)]
    head[2] = 0.0
    head /= np.linalg.norm(head) + 1e-12
    ext = P[-1] + np.outer(np.arange(ds, ahead * L, ds), head)
    track = np.vstack([track, ext])
    tang = np.gradient(track, axis=0)
    tang[:, 2] = 0.0
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    nrm = np.c_[-tang[:, 1], tang[:, 0], np.zeros(len(tang))]
    pts, vals = [], []
    for a in np.linspace(-b, b, n_lat):
        for dz in np.linspace(-b / 2, b / 2, n_vert):
            Q = track + a * nrm + np.array([0.0, 0.0, dz])
            w_up = -wind.velocity(0.0, Q)[:, 2]
            Z = Q @ S_FLIP.T
            Z[:, 2] += exaggeration * w_up
            pts.append(Z)
            vals.append(w_up)
    aw = np.abs(np.concatenate(vals))
    lim = float(np.percentile(aw, 98))
    if lim < 0.25 * aw.max():                    # a discrete gust in calm air: scale on the gust, not the calm
        lim = float(aw.max())
    lim = max(lim, 1e-6)
    lines = Lines(pts, vals, name="gust_wisps", label="Vertical gust w [m/s] (up +), frozen field")
    for k, v in (("cmap", "coolwarm"), ("alpha", 0.45), ("radius", 0.006 * b), ("color_range", (-lim, lim))):
        setattr(lines, k, v)
    return lines


def scene_summary(sc) -> dict[str, Any]:
    return {"surfaces": len(sc.surfaces), "triangles": int(sum(len(s.faces) for s in sc.surfaces)),
            "frames": sc.animation.n_frames if sc.animation is not None else 0}
