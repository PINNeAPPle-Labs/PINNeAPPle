"""Gallery: 3D twins of very different parts built from primitives, to exercise the generator beyond liners.

    from pinneapple_twin3d import gallery
    gallery.export_all("out/gallery")         # one folder per part + index.html
    gallery.get("rocket").export("out/rocket")

Parts: runner channel (blast-furnace trough), mixing tank, oil pipeline, airliner, launch vehicle, quadcopter drone,
lunar rover, wind turbine, satellite.

**Every field here is ILLUSTRATIVE** (simple closed-form patterns in space and time that place the physics where
an engineer expects it: stagnation heating at a nose, erosion at a bend, ...). They are not simulation or measurement
results, and each scene's title and ``source`` say so. Swap in real fields with ``Scene.add_field``.
"""
from __future__ import annotations

import os
from typing import Callable, Dict

import numpy as np

from . import primitives as P
from .scene import Scene

ILLUSTRATIVE = "  [ILLUSTRATIVE FIELDS]"
GALLERY: Dict[str, Callable[[], Scene]] = {}


def _register(fn):
    GALLERY[fn.__name__] = fn
    return fn


def get(key: str) -> Scene:
    if key not in GALLERY:
        raise KeyError(f"unknown part '{key}'; available: {sorted(GALLERY)}")
    return GALLERY[key]()


def _g(x, mu, s):
    return np.exp(-0.5 * ((np.asarray(x, float) - mu) / s) ** 2)


def _smooth(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _scene(key, title, times, unit):
    return Scene(title + ILLUSTRATIVE, length_unit="m", times=np.asarray(times, float), time_unit=unit,
                 source=f"pinneapple_twin3d.gallery.{key}: illustrative fields, not a simulation or measurement result")


def _rounded_path(points, radius, step=1.0, n_arc=14):
    """Polyline with filleted corners, densified to ``step`` m. Returns points and a 0/1 flag for arc samples."""
    pts = np.asarray(points, float)
    out, flag = [pts[0]], [0]
    for i in range(1, len(pts) - 1):
        c, d1, d2 = pts[i], pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        r = min(radius, 0.45 * l1, 0.45 * l2)
        a, b = c - d1 / l1 * r, c + d2 / l2 * r
        for p0, p1 in ((out[-1], a),):  # straight run up to the arc
            n = max(int(np.linalg.norm(p1 - p0) / step), 1)
            for k in range(1, n + 1):
                out.append(p0 + (p1 - p0) * k / n); flag.append(0)
        for t in np.linspace(0, 1, n_arc)[1:]:
            out.append((1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b); flag.append(1)
    p0, p1 = out[-1], pts[-1]
    n = max(int(np.linalg.norm(p1 - p0) / step), 1)
    for k in range(1, n + 1):
        out.append(p0 + (p1 - p0) * k / n); flag.append(0)
    return np.asarray(out), np.asarray(flag)


def _rows(mesh, n_rows):
    v = mesh[0]
    return v.reshape(n_rows, -1, 3)


# ------------------------------------------------------------------------------------------------ runner channel
@_register
def runner_channel() -> Scene:
    """Blast-furnace cast-house runner: refractory-lined U trough from the tap hole to the skimmer."""
    months = np.arange(0, 13)
    sc = _scene("runner_channel", "Blast-furnace runner channel", months, "month")
    s = np.linspace(0, 1, 70)
    path = np.stack([14 * s, -0.5 * s, 1.2 * np.sin(s * 1.6)], 1)
    U = [(0.65, -0.55), (0.06, -0.55), (0.0, -0.42), (0.0, 0.42), (0.06, 0.55), (0.65, 0.55)]  # (height, lateral) of the hot face
    mesh = P.sweep(U, path, closed=False, up=(0, 1, 0))
    sc.add_part("runner", *mesh, group="refractory", color=(0.58, 0.52, 0.47))
    v = mesh[0].reshape(len(s), len(U), 3)
    height = np.array([u for u, _ in U])[None, :].repeat(len(s), 0)
    lateral = np.array([w for _, w in U])[None, :].repeat(len(s), 0)
    sg = s[:, None].repeat(len(U), 1)
    # metal line (low) and slag line (higher) bands, a jet-impact hot spot under the tap hole, wall corner scour
    pat = (0.35 + 0.9 * _g(height, 0.08, 0.1) + 0.7 * _g(height, 0.4, 0.07)) * (0.45 + 0.9 * _g(sg, 0.05, 0.12)
           + 0.35 * _g(sg, 0.62, 0.1)) + 0.5 * _g(sg, 0.04, 0.05) * _g(lateral, 0, 0.25) * _g(height, 0, 0.12)
    wear = np.outer(months, pat.ravel()) * 9.0  # mm, ~100 mm at the worst spot after a year
    sc.add_field("runner", "wear_depth", wear, unit="mm")
    sc.add_field("runner", "remaining_thickness", 300.0 - wear, unit="mm")
    sc.add_sensor("TC-tap", tuple(path[2] + [0, 0.1, 0]), label="Thermocouple at tap end", unit="degC", quantity="temperature",
                  series=1380 + 25 * np.sin(months), envelope=(1300, 1500))
    return sc


# ------------------------------------------------------------------------------------------------ mixing tank
@_register
def mixing_tank() -> Scene:
    """Agitated tank: dished bottom, baffles, two-stage pitched-blade impeller; tracer blending over time."""
    t = np.linspace(0, 120, 25)
    sc = _scene("mixing_tank", "Agitated mixing tank", t, "s")
    R, H = 1.2, 2.4
    tank = P.revolve([(0.0, 0.0), (0.7, 0.02), (1.05, 0.13), (R, 0.32), (R, H)], 56)
    sc.add_part("tank", *tank, group="vessel", color=(0.7, 0.74, 0.78))
    sc.add_part("shaft", *P.cylinder([0, 0.35, 0], [0, H + 0.2, 0], 0.05), group="agitator", color=(0.4, 0.42, 0.45))
    blades = []
    for y in (0.7, 1.55):
        hub = P.cylinder([0, y - 0.05, 0], [0, y + 0.05, 0], 0.12)
        parts = [hub]
        for k in range(4):
            b = P.box([0.36, 0.015, 0.16], [0.30, 0, 0])
            b = P.transform(b, R=P.rotation([1, 0, 0], 35))  # pitch
            parts.append(P.transform(b, R=P.rotation([0, 1, 0], 90 * k), t=[0, y, 0]))
        blades.append(P.merge(parts))
    sc.add_part("impeller_lower", *blades[0], group="agitator", color=(0.3, 0.55, 0.75))
    sc.add_part("impeller_upper", *blades[1], group="agitator", color=(0.3, 0.55, 0.75))
    baf = [P.transform(P.box([0.02, H - 0.35, 0.12], [R - 0.08, (H + 0.35) / 2, 0]), R=P.rotation([0, 1, 0], 90 * k)) for k in range(4)]
    sc.add_part("baffles", *P.merge(baf), group="vessel", color=(0.55, 0.58, 0.62))

    v = tank[0]
    r, y, th = np.hypot(v[:, 0], v[:, 2]), v[:, 1], np.arctan2(-v[:, 2], v[:, 0])
    # tracer fed near the wall at the surface; blending time grows away from the impeller zone
    d2 = (r - 1.0) ** 2 + (y - 2.2) ** 2 + (2 * np.sin((th - 0) / 2)) ** 2 * 0.8
    tau = 10 + 28 * (1 - _g(y, 1.1, 0.7)) + 8 * (r / R)  # blending time constant [s], longest far from the impellers
    conc = 1 + 4.0 * _g(np.sqrt(d2), 0, 0.55)[None, :] * np.exp(-t[:, None] / tau[None, :])
    sc.add_field("tank", "tracer_concentration", conc, unit="-")
    ramp = _smooth(0, 10, t)[:, None]  # motor spin-up
    for name, part_v in (("impeller_lower", blades[0][0]), ("impeller_upper", blades[1][0])):
        rr = np.hypot(part_v[:, 0], part_v[:, 2])
        sc.add_field(name, "shear_rate", ramp * (40 + 260 * (rr / 0.5) ** 2)[None, :], unit="1/s")
    sc.add_sensor("CT-01", (0.2, 0.4, 0.9), label="Conductivity probe (mid-tank)", unit="-", quantity="tracer",
                  series=1 + 0.8 * np.exp(-t / 35), envelope=(0.95, 1.05))
    return sc


# ------------------------------------------------------------------------------------------------ oil pipeline
@_register
def oil_pipeline() -> Scene:
    """Buried-style crude line with two bends and a sag: wall loss from corrosion/erosion over 20 years."""
    years = np.arange(0, 21, 2.0)
    sc = _scene("oil_pipeline", "Crude-oil pipeline segment", years, "year")
    path, bend = _rounded_path([(0, 0, 0), (60, 0, 0), (60, 0, 45), (110, 6, 45), (150, 6, 45)], radius=9, step=1.5)
    n_th = 24
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    rad = 0.3
    section = np.stack([rad * np.cos(th), rad * np.sin(th)], 1)  # u = up, v = lateral
    mesh = P.sweep(section, path, closed=True, up=(0, 1, 0))
    sc.add_part("pipe", *mesh, group="piping", color=(0.5, 0.52, 0.56))
    V = mesh[0].reshape(len(path), n_th, 3)
    cen = V.mean(1)
    d = V - cen[:, None, :]
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    acc = np.gradient(np.gradient(path, axis=0), axis=0)  # points toward the centre of curvature
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    cdir = np.where(nrm > 1e-9, acc / np.where(nrm > 1e-9, nrm, 1), 0)
    extrados = np.clip(-(d * cdir[:, None, :]).sum(2), 0, 1) * bend[:, None]
    bottom = np.clip(-d[:, :, 1], 0, 1) ** 2  # water drop-out along the 6 o'clock line
    s = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
    sag = _g(s, 0.62 * s[-1], 0.04 * s[-1])[:, None]
    rate = 0.08 + 0.55 * extrados + 0.12 * bottom + 0.25 * sag * bottom  # mm / year
    loss = years[:, None] * rate.ravel()[None, :]
    sc.add_field("pipe", "wall_loss", loss, unit="mm")
    sc.add_field("pipe", "remaining_wall", 12.7 - loss, unit="mm")
    sc.add_field("pipe", "pressure", np.repeat((70 - 25 * (s / s[-1]))[:, None].repeat(n_th, 1).ravel()[None, :], len(years), 0), unit="bar")
    for i, name in ((int(np.argmax(bend)), "UT-bend-1"), (len(path) - 25, "UT-straight")):
        sc.add_sensor(name, tuple(cen[i] + [0, 0.32, 0]), label=name + " wall-thickness probe", unit="mm", quantity="wear",
                      series=years * float(rate[i].max()), envelope=(0, 6.0))
    return sc


# ------------------------------------------------------------------------------------------------ airliner
@_register
def airliner() -> Scene:
    """Narrow-body airliner: fuselage, swept wing, tail, two engines; skin temperature and wing-root load over a flight."""
    t = np.arange(0, 181, 10.0)
    sc = _scene("airliner", "Commercial airliner (narrow-body)", t, "min")
    prof = [(0.05, 0), (0.7, 0.8), (1.35, 2.2), (1.95, 4.5), (1.95, 30), (1.5, 34), (0.9, 36.5), (0.25, 37.5)]
    fus = P.transform(P.revolve(prof, 48), R=P.rotation([0, 0, 1], -90))  # y axis -> +x: nose at x=0
    fus = (fus[0] * [1, 1, 1] + [0, 0, 0], fus[1])
    sc.add_part("fuselage", *fus, group="airframe", color=(0.86, 0.88, 0.9))
    w = P.wing(17.0, 6.2, 1.5, sweep_deg=27, dihedral_deg=5, thickness=0.12)
    w = P.transform(w, t=[14.5, -0.9, 1.7])
    sc.add_part("wing_right", *w, group="airframe", color=(0.8, 0.83, 0.86))
    sc.add_part("wing_left", *P.mirror_z(w), group="airframe", color=(0.8, 0.83, 0.86))
    h = P.transform(P.wing(5.5, 3.4, 1.1, sweep_deg=32, dihedral_deg=4, thickness=0.1), t=[32, 0.6, 0.9])
    sc.add_part("tailplane_right", *h, group="airframe", color=(0.8, 0.83, 0.86))
    sc.add_part("tailplane_left", *P.mirror_z(h), group="airframe", color=(0.8, 0.83, 0.86))
    fin = P.transform(P.wing(6.0, 5.0, 1.8, sweep_deg=35, thickness=0.1), R=P.rotation([1, 0, 0], -90), t=[30.5, 1.5, 0])
    sc.add_part("fin", *fin, group="airframe", color=(0.75, 0.2, 0.2))
    for sgn, nm in ((1, "engine_right"), (-1, "engine_left")):
        eng = P.cylinder([12.0, -2.0, sgn * 6.3], [16.3, -2.0, sgn * 6.3], (1.05, 0.9))
        sc.add_part(nm, *eng, group="propulsion", color=(0.45, 0.47, 0.5))
    # flight profile: climb, cruise at M0.78 / 11 km, descent
    alt = np.interp(t, [0, 25, 140, 180], [0, 11000, 11000, 0])
    mach = np.interp(t, [0, 5, 25, 140, 180], [0.05, 0.4, 0.78, 0.78, 0.25])
    Ta = 288.15 - 6.5e-3 * np.minimum(alt, 11000)
    a = np.sqrt(1.4 * 287 * Ta)
    dT = 0.9 * (mach * a) ** 2 / (2 * 1005)  # stagnation temperature rise
    load = np.interp(t, [0, 5, 25, 140, 175, 180], [1.0, 1.4, 1.1, 1.05, 1.3, 1.0])  # wing bending load factor
    for name in ("fuselage", "wing_right", "wing_left", "tailplane_right", "tailplane_left", "fin", "engine_right", "engine_left"):
        part = next(p for p in sc.parts if p.name == name)
        vv = part.vertices.astype(float)
        if name == "fuselage":
            front = np.exp(-(vv[:, 0] - vv[:, 0].min()) / 1.2)
        elif name.startswith(("wing", "tailplane", "fin")):
            K = 2 * (20 if name.startswith("wing") else 20 if name.startswith("tail") else 20) - 2
            rows = vv.reshape(-1, K, 3)
            xmin = rows[:, :, 0].min(1, keepdims=True)
            front = np.exp(-(rows[:, :, 0] - xmin) / 0.35).reshape(-1)
        else:
            front = np.exp(-(vv[:, 0] - vv[:, 0].min()) / 0.4)
        T_skin = (Ta - 273.15)[:, None] + (0.35 + 0.65 * front)[None, :] * dT[:, None] * (1 if name != "fuselage" else 0.9)
        sc.add_field(name, "skin_temperature", T_skin, unit="degC")
        if name.startswith("wing"):
            zroot = 1 - np.clip(np.abs(vv[:, 2]) - 1.7, 0, None) / 17.0
            sc.add_field(name, "wing_bending_stress", (load[:, None] - 0.0) * 120 * (zroot ** 2)[None, :], unit="MPa")
    sc.add_sensor("TAT", (0.5, 1.9, 0.0), label="Total air temperature probe", unit="degC", quantity="temperature",
                  series=(Ta - 273.15) + dT, envelope=(-70, 50))
    return sc


# ------------------------------------------------------------------------------------------------ rocket
@_register
def rocket() -> Scene:
    """Launch vehicle: ogive nose, tank stack, bell nozzle, four fins; aero heating and nozzle ablation during ascent."""
    t = np.arange(0, 161, 8.0)
    sc = _scene("rocket", "Launch vehicle", t, "s")
    sn = np.linspace(0, 1, 13)[1:]  # ogive: radius falls from 1.8 m to a rounded tip over 7 m
    ogive = [(1.8 * np.sqrt(1 - 0.995 * q * q), 36.0 + 7.0 * q) for q in sn]
    body = P.revolve([(1.8, 0.0), (1.8, 6.0), (1.8, 36.0)] + ogive, 48)
    sc.add_part("body", *body, group="airframe", color=(0.9, 0.9, 0.92))
    noz = P.revolve([(0.95, 0.0), (0.75, -0.7), (0.38, -1.5), (0.5, -2.2), (0.95, -3.4), (1.25, -4.4)], 48)
    sc.add_part("nozzle", *noz, group="propulsion", color=(0.35, 0.33, 0.32))
    fins = [P.transform(P.box([0.12, 4.5, 2.2], [1.8 + 1.0, 3.0, 0]), R=P.rotation([0, 1, 0], 90 * k + 45)) for k in range(4)]
    sc.add_part("fins", *P.merge(fins), group="airframe", color=(0.3, 0.3, 0.34))
    # trajectory (illustrative): altitude and speed during the burn
    h = 2.75 * t ** 2  # ~70 km at 160 s
    V = 22.0 * t  # ~3.5 km/s at 160 s
    rho = 1.225 * np.exp(-h / 8400.0)
    qdyn = 0.5 * rho * V ** 2
    qheat = rho ** 0.5 * V ** 3  # Sutton-Graves-like scaling, arbitrary units
    qheat = qheat / qheat.max()
    vb = body[0]
    nose_f = _smooth(34, 41, vb[:, 1]) * _g(np.hypot(vb[:, 0], vb[:, 2]), 0, 0.9) + 0.12 * _smooth(34, 43, vb[:, 1])
    sc.add_field("body", "heat_flux", (qheat[:, None] * (0.05 + 0.95 * nose_f)[None, :]) * 600, unit="kW/m2")
    vn = noz[0]
    thr = _g(vn[:, 1], -1.5, 0.45)
    burn = (t < 150).astype(float)
    flux = burn[:, None] * (1500 + 4500 * thr)[None, :]  # kW/m2, throat is the hottest
    sc.add_field("nozzle", "heat_flux", flux, unit="kW/m2")
    abl = np.cumsum(flux * 8.0, axis=0) / 4.0e5  # mm: integral of flux, arbitrary constant
    sc.add_field("nozzle", "ablation_depth", abl - abl[0], unit="mm")
    vf = fins[0][0]
    sc.add_field("fins", "heat_flux", np.outer(qheat, np.ones(len(P.merge(fins)[0]))) * 120, unit="kW/m2")
    sc.add_sensor("Q-dyn", (0, 30, 1.9), label="Dynamic pressure (derived)", unit="kPa", quantity="pressure",
                  series=qdyn / 1000, envelope=(0, 40))
    return sc


# ------------------------------------------------------------------------------------------------ drone
@_register
def drone() -> Scene:
    """Quadcopter: body, arms, motors, rotor discs, battery; motor and battery heating over a 20-minute flight."""
    t = np.linspace(0, 20, 21)
    sc = _scene("drone", "Quadcopter drone", t, "min")
    sc.add_part("body", *P.box([0.22, 0.07, 0.22], [0, 0.0, 0]), group="airframe", color=(0.2, 0.22, 0.25))
    bat = P.box([0.14, 0.04, 0.09], [0, -0.055, 0])
    sc.add_part("battery", *bat, group="power", color=(0.85, 0.7, 0.15))
    sc.add_part("camera", *P.revolve([(0.0, -0.1), (0.035, -0.095), (0.04, -0.07), (0.0, -0.05)], 24), group="payload", color=(0.1, 0.1, 0.1))
    arms, motors, rotors = [], [], []
    for k in range(4):
        a = np.radians(45 + 90 * k)
        d = np.array([np.cos(a), 0, np.sin(a)])
        arms.append(P.cylinder(d * 0.08, d * 0.30, 0.012, 16))
        c = d * 0.32
        motors.append(P.cylinder(c + [0, -0.015, 0], c + [0, 0.035, 0], 0.028, 24))
        rotors.append(P.cylinder(c + [0, 0.05, 0], c + [0, 0.054, 0], 0.125, 40))
    sc.add_part("arms", *P.merge(arms), group="airframe", color=(0.25, 0.27, 0.3))
    sc.add_part("motors", *P.merge(motors), group="propulsion", color=(0.55, 0.57, 0.6))
    sc.add_part("rotors", *P.merge(rotors), group="propulsion", color=(0.1, 0.1, 0.12))
    load = 0.75 + 0.25 * np.sin(t / 3)  # manoeuvre load
    tau = 6.0
    heat = np.cumsum(load) / np.arange(1, len(t) + 1)
    mt = 28 + 52 * (1 - np.exp(-t / tau)) * heat  # motor temperature
    mv = P.merge(motors)[0]
    sc.add_field("motors", "temperature", np.repeat(mt[:, None], len(mv), 1) + 4 * np.sin(np.arctan2(mv[:, 2], mv[:, 0]) * 1)[None, :] * (t[:, None] / 20), unit="degC")
    av = P.merge(arms)[0]
    dist = np.hypot(av[:, 0], av[:, 2])
    # heat conducts from the motor (at 0.32 m from the centre) towards the body
    sc.add_field("arms", "temperature", 24 + (mt[:, None] - 24) * np.exp(-(0.32 - dist) / 0.12)[None, :] * 0.6, unit="degC")
    bt = 24 + 22 * (1 - np.exp(-t / 9)) * heat
    sc.add_field("battery", "temperature", np.repeat(bt[:, None], len(bat[0]), 1), unit="degC")
    sc.add_sensor("T-motor1", (0.23, 0.0, 0.23), label="Motor 1 winding temperature", unit="degC", quantity="temperature",
                  series=mt, envelope=(0, 85))
    sc.add_sensor("T-batt", (0, -0.075, 0), label="Battery pack temperature", unit="degC", quantity="temperature",
                  series=bt, envelope=(0, 50))
    return sc


# ------------------------------------------------------------------------------------------------ lunar rover
@_register
def lunar_rover() -> Scene:
    """Six-wheel lunar rover over one lunar day-night cycle: chassis temperature and regolith wheel wear."""
    days = np.arange(0, 29, 1.0)
    sc = _scene("lunar_rover", "Lunar rover (six-wheel)", days, "Earth day")
    sc.add_part("chassis", *P.box([1.7, 0.35, 1.0], [0, 0.62, 0]), group="body", color=(0.82, 0.8, 0.72))
    sc.add_part("solar_panel", *P.transform(P.box([1.5, 0.03, 0.9], [0, 0, 0]), R=P.rotation([1, 0, 0], 12), t=[0, 0.86, 0]), group="power", color=(0.1, 0.15, 0.4))
    mast = P.merge([P.cylinder([0.6, 0.8, 0], [0.6, 1.6, 0], 0.025), P.box([0.18, 0.1, 0.22], [0.6, 1.65, 0])])
    sc.add_part("mast_camera", *mast, group="payload", color=(0.3, 0.3, 0.32))
    dish = P.transform(P.revolve([(0.0, 0.0), (0.12, 0.012), (0.24, 0.05), (0.32, 0.11)], 32), R=P.rotation([1, 0, 0], 55), t=[-0.5, 1.0, 0.0])
    sc.add_part("antenna", *dish, group="payload", color=(0.88, 0.88, 0.9))
    wheels = []
    for i, x in enumerate((-0.75, 0.0, 0.75)):
        for sgn in (-1, 1):
            c = np.array([x, 0.3, sgn * 0.7])
            w = P.cylinder(c - [0, 0, 0.13], c + [0, 0, 0.13], 0.3, 40)
            sc.add_part(f"wheel_{'L' if sgn < 0 else 'R'}{i + 1}", *w, group="mobility", color=(0.25, 0.25, 0.27))
            wheels.append((f"wheel_{'L' if sgn < 0 else 'R'}{i + 1}", w, i))
    # thermal: ~14 days of sun then ~14 days of night, first-order lag, partly decoupled by the radiator/heater
    sun = (np.sin(2 * np.pi * days / 29.5) > 0).astype(float)
    T = np.zeros(len(days)); T[0] = 20.0
    for k in range(1, len(days)):
        target = 95 * sun[k] + (-165) * (1 - sun[k])
        T[k] = T[k - 1] + (target - T[k - 1]) * 0.35
    Tc = np.clip(T, -40, 70) * 0.8 + 0.2 * T  # heater / radiator keep the bus nearer to comfortable
    cv = sc.parts[0].vertices
    top = (cv[:, 1] - cv[:, 1].min()) / (np.ptp(cv[:, 1]) + 1e-9)
    sc.add_field("chassis", "temperature", Tc[:, None] + (20 * top * sun[:, None]), unit="degC")
    sc.add_field("solar_panel", "temperature", (T * 0.95)[:, None] + np.zeros((1, len(sc.parts[1].vertices))), unit="degC")
    drive_km = np.cumsum(sun * 0.9)  # driven only in daylight
    for name, w, i in wheels:
        vv = w[0]
        edge = _smooth(0.2, 0.3, np.hypot(vv[:, 1] - 0.3, vv[:, 0] - [-0.75, 0.0, 0.75][i]))  # rim wears
        load = (1.25, 1.0, 0.85)[i]
        sc.add_field(name, "wear_depth", drive_km[:, None] * 0.18 * load * (0.3 + 0.7 * edge)[None, :], unit="mm")
    sc.add_sensor("T-bus", (0, 0.8, 0), label="Bus temperature (derived)", unit="degC", quantity="temperature", series=Tc, envelope=(-40, 60))
    return sc


# ------------------------------------------------------------------------------------------------ wind turbine
@_register
def wind_turbine() -> Scene:
    """Three-blade turbine: tapered tower, nacelle, twisted blades; leading-edge erosion and fatigue over 20 years."""
    years = np.arange(0, 21, 2.0)
    sc = _scene("wind_turbine", "Wind turbine", years, "year")
    HUB = 90.0
    tower = P.revolve([(2.4, 0.0), (2.2, 15), (1.9, 55), (1.6, HUB)], 40)
    sc.add_part("tower", *tower, group="structure", color=(0.88, 0.89, 0.9))
    sc.add_part("nacelle", *P.box([7.0, 3.0, 3.0], [-1.5, HUB + 1.6, 0]), group="drivetrain", color=(0.8, 0.82, 0.85))
    hub = P.transform(P.revolve([(0.0, 0.0), (1.3, 0.5), (1.7, 1.6), (0.9, 3.0), (0.0, 3.4)], 32), R=P.rotation([0, 0, 1], -90), t=[1.5, HUB + 1.6, 0])
    sc.add_part("hub", *hub, group="drivetrain", color=(0.9, 0.9, 0.92))
    blade = P.wing(55.0, 4.2, 0.7, sweep_deg=0, twist_deg=-14, thickness=0.22, n_sections=24, n_pts=22)
    bv0 = blade[0]
    for k in range(3):
        rot = P.rotation([1, 0, 0], -90 + 120 * k)  # blade 1 points up; the others 120 degrees apart around the rotor axis (x)
        b = P.transform(blade, R=rot, t=[2.2, HUB + 1.6, 0])
        name = f"blade_{k + 1}"
        sc.add_part(name, *b, group="rotor", color=(0.95, 0.95, 0.96))
        K = 2 * 22 - 2
        rows = bv0.reshape(-1, K, 3)
        span = rows[:, :, 2] / 55.0
        xle = rows[:, :, 0].min(1, keepdims=True)
        le = np.exp(-(rows[:, :, 0] - xle) / 0.18)  # leading edge
        er = (span ** 3.2) * (0.15 + 0.85 * le)  # tip-speed driven
        fat = _g(span, 0.12, 0.12) * (0.6 + 0.4 * np.abs(rows[:, :, 1]) / 0.3)  # root / max-chord region
        sc.add_field(name, "leading_edge_erosion", np.outer(years, er.ravel()) * 0.35, unit="mm")
        sc.add_field(name, "fatigue_damage", np.outer(years / 20, fat.ravel()) * 0.8, unit="-")
    tv = tower[0]
    sc.add_field("tower", "fatigue_damage", np.outer(years / 20, (_g(tv[:, 1], 0, 6) * 0.55 + _g(tv[:, 1], 55, 4) * 0.25)), unit="-")
    sc.add_sensor("SG-root", (2.4, 2.0, 0), label="Strain gauge, tower base", unit="-", quantity="fatigue",
                  series=years / 20 * 0.55, envelope=(0, 1))
    return sc


# ------------------------------------------------------------------------------------------------ satellite
@_register
def satellite() -> Scene:
    """Small satellite in LEO: bus, two solar wings, dish; thermal cycling through sunlight and eclipse."""
    t = np.linspace(0, 190, 39)  # two orbits of ~95 min
    sc = _scene("satellite", "Earth-observation satellite (LEO)", t, "min")
    bus = P.box([1.8, 1.8, 2.6], [0, 0, 0])
    sc.add_part("bus", *bus, group="bus", color=(0.78, 0.7, 0.35))
    wing = P.box([0.03, 1.6, 6.5], [0, 0, 0])
    sc.add_part("array_north", *P.transform(wing, t=[0, 0, 4.6]), group="power", color=(0.1, 0.16, 0.42))
    sc.add_part("array_south", *P.transform(wing, t=[0, 0, -4.6]), group="power", color=(0.1, 0.16, 0.42))
    sc.add_part("boom", *P.cylinder([0, 0, -1.3], [0, 0, 1.3], 0.05), group="power", color=(0.5, 0.5, 0.52))
    dish = P.transform(P.revolve([(0.0, 0.0), (0.3, 0.03), (0.6, 0.12), (0.9, 0.27)], 36), R=P.rotation([1, 0, 0], 90), t=[0.0, 0.0, 1.3])
    dish = P.transform(dish, R=P.rotation([0, 1, 0], 90), t=[0.9, 0.6, 0.0])
    sc.add_part("antenna", *dish, group="payload", color=(0.9, 0.9, 0.92))
    period, eclipse = 95.0, 35.0
    phase = (t % period)
    sunlit = (phase < period - eclipse).astype(float)
    def lag(target, k):
        out = np.zeros_like(target); out[0] = target[0]
        for i in range(1, len(target)):
            out[i] = out[i - 1] + (target[i] - out[i - 1]) * k
        return out
    arr = lag(np.where(sunlit > 0, 85.0, -110.0), 0.18)
    busT = lag(np.where(sunlit > 0, 28.0, 8.0), 0.07)
    for name in ("array_north", "array_south"):
        pv = next(p for p in sc.parts if p.name == name).vertices
        tipcool = _smooth(1.5, 6.5, np.abs(pv[:, 2]))  # wing tips radiate a little more
        sc.add_field(name, "temperature", arr[:, None] - 6 * tipcool[None, :] * sunlit[:, None], unit="degC")
    bv = bus[0]
    face = 0.5 + 0.5 * bv[:, 0] / 0.9  # sun-facing side (+x) is warmer
    sc.add_field("bus", "temperature", busT[:, None] + 14 * face[None, :] * sunlit[:, None] - 6 * (1 - face)[None, :] * (1 - sunlit[:, None]), unit="degC")
    sc.add_sensor("T-bus", (0.9, 0.0, 0.0), label="Bus thermistor", unit="degC", quantity="temperature", series=busT + 7 * sunlit, envelope=(-10, 45))
    return sc


def export_all(folder: str) -> str:
    """Export every gallery part to ``folder/<key>/`` and write ``folder/index.html`` linking them."""
    os.makedirs(folder, exist_ok=True)
    rows = []
    for key in sorted(GALLERY):
        sc = get(key)
        sc.export(os.path.join(folder, key))
        fields = sorted({n for p in sc.parts for n in p.fields})
        rows.append(f'<li><a href="{key}/index.html?step=last&field={fields[0]}"><b>{sc.title}</b></a> '
                    f'<small>{len(sc.parts)} parts &middot; fields: {", ".join(fields)} &middot; time: {sc.times[0]:g}-{sc.times[-1]:g} {sc.time_unit}</small></li>')
    html = ('<!doctype html><meta charset="utf-8"><title>PINNeAPPle Twin3D gallery</title>'
            '<style>body{font:15px system-ui;max-width:860px;margin:30px auto;padding:0 16px}li{margin:10px 0}small{color:#667;display:block}</style>'
            '<h1>Twin3D gallery</h1><p>Parts built from primitives. <b>All fields are illustrative</b>, not simulation results.</p><ul>'
            + "".join(rows) + "</ul>")
    with open(os.path.join(folder, "index.html"), "w") as f:
        f.write(html)
    return os.path.join(folder, "index.html")
