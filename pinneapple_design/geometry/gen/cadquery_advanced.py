"""Advanced parametric CadQuery geometry for physics-AI design spaces, plus reproductions of the
classic CadQuery documentation examples.

Each builder takes a ``params`` dict (so it plugs into :class:`CadQueryRegistry` and can be driven
by a DoE / optimiser / LLM draft) and returns a ``cq.Workplane``. Every builder exercises CadQuery
features that the two starter templates did not: sweeps along arcs and helices, lofts between
sections, shells, polar/rectangular patterns, fillets, counterbored holes, computed (involute)
profiles. Where a closed-form volume exists, ``tests/test_cadquery_advanced.py`` checks it.

Engineering builders (CadQuery 2.x API; docs: https://cadquery.readthedocs.io):
- ``pipe_bend_flanged``   sweep of an annulus along an arc + bolted flanges (PINNeAPPle-CFD CFD-04)
- ``pipe_tee``            equal tee: unions and cuts of cylinders
- ``concentric_reducer``  loft between circles, hollowed by a second loft
- ``helical_coil``        tube swept along a helix (Frenet frame): heat-exchanger coil / spring
- ``axial_fan``           hub + twisted blades lofted between NACA sections, polar pattern
- ``spur_gear``           involute teeth computed point by point, bore with key slot
- ``pin_fin_heatsink``    base + rectangular array of pin fins

Reproductions of the CadQuery documentation examples (Apache-2.0):
- ``occ_bottle``          "The Classic OCC Bottle"
- ``pillow_block``        "Parametric Bearing Pillow Block"
- ``lego_brick``          "A Parametric Enclosure"/Lego-style brick ("Lego Brick" example)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

Params = Dict[str, Any]


def _cq():
    import cadquery as cq
    return cq


# ── engineering builders ────────────────────────────────────────────────
def pipe_bend_flanged(p: Params):
    """Bend of mean radius R and angle theta, tube OD/wall t, bolted flange at each end.

    The tube is an annulus swept along an arc in the XZ plane; flanges are extruded outward from
    both end faces so the volumes add up exactly:
    V = pi (ro^2 - ri^2) R theta + 2 [pi (rf^2 - ri^2) - n pi rb^2] tf.
    """
    cq = _cq()
    ro, t, R = p["od"] / 2, p["wall"], p["bend_radius"]
    ri = ro - t
    ang = math.radians(p["angle_deg"])
    # path: arc from (0,0,0) turning from +Z towards +X, centre (R,0,0)
    end = (R - R * math.cos(ang), 0.0, R * math.sin(ang))
    mid = (R - R * math.cos(ang / 2), 0.0, R * math.sin(ang / 2))
    path = cq.Workplane("XZ").threePointArc((mid[0], mid[2]), (end[0], end[2]))
    tube = cq.Workplane("XY").circle(ro).circle(ri).sweep(path)
    rf, tf, n, rb, rbc = p["flange_od"] / 2, p["flange_t"], int(p["n_bolts"]), p["bolt_d"] / 2, p["bolt_circle_d"] / 2

    f1 = (cq.Workplane("XY").circle(rf).circle(ri).extrude(-tf)
          .faces("<Z").workplane().polarArray(rbc, 0, 360, n).circle(rb).cutThruAll())
    # second flange on the end face, whose normal is the path tangent at the end
    tangent = cq.Vector(math.sin(ang), 0.0, math.cos(ang))
    plane2 = cq.Plane(origin=end, xDir=cq.Vector(0, 1, 0).cross(tangent), normal=tangent)
    f2 = (cq.Workplane(plane2).circle(rf).circle(ri).extrude(tf)
          .faces(cq.selectors.DirectionMinMaxSelector(tangent, True)).workplane()
          .polarArray(rbc, 0, 360, n).circle(rb).cutThruAll())
    return tube.union(f1).union(f2)


def pipe_tee(p: Params):
    """Equal tee: run along X (length L), branch along +Z (height H), same OD/wall."""
    cq = _cq()
    ro, ri = p["od"] / 2, p["od"] / 2 - p["wall"]
    L, H = p["run_length"], p["branch_height"]
    outer = (cq.Workplane("YZ").circle(ro).extrude(L / 2, both=True)
             .union(cq.Workplane("XY").circle(ro).extrude(H)))
    bore = (cq.Workplane("YZ").circle(ri).extrude(L / 2, both=True)
            .union(cq.Workplane("XY").circle(ri).extrude(H)))
    body = outer.cut(bore)
    r_fillet = p.get("fillet", 0.0)
    if r_fillet:
        # outer intersection curve of the two cylinders: edges not on the end caps, above z = 0
        body = body.edges(cq.selectors.BoxSelector((-ro * 1.2, -ro * 1.2, 0.2 * ro), (ro * 1.2, ro * 1.2, 1.05 * ro))).fillet(r_fillet)
    return body


def concentric_reducer(p: Params):
    """Truncated cone from D1 to D2 over length L, wall t (loft of circles, hollowed by a loft)."""
    cq = _cq()
    r1, r2, L, t = p["d1"] / 2, p["d2"] / 2, p["length"], p["wall"]
    outer = cq.Workplane("XY").circle(r1).workplane(offset=L).circle(r2).loft(combine=True)
    inner = cq.Workplane("XY").circle(r1 - t).workplane(offset=L).circle(r2 - t).loft(combine=True)
    return outer.cut(inner)


def helical_coil(p: Params):
    """Solid tube of radius r swept along a helix (coil radius Rc, pitch, turns)."""
    cq = _cq()
    Rc, pitch, turns, r = p["coil_radius"], p["pitch"], p["turns"], p["tube_d"] / 2
    helix = cq.Wire.makeHelix(pitch=pitch, height=pitch * turns, radius=Rc)
    # profile: circle in the plane normal to the helix tangent at its start (Rc, 0, 0)
    tangent = cq.Vector(0, 2 * math.pi * Rc, pitch).normalized()
    plane = cq.Plane(origin=(Rc, 0, 0), xDir=(1, 0, 0), normal=tangent)
    return cq.Workplane(plane).circle(r).sweep(cq.Workplane().add(helix), isFrenet=True)


def _naca4_points(code: str, chord: float, n: int = 40) -> List[Tuple[float, float]]:
    m, pp, tt = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    pts_u, pts_l = [], []
    for i in range(n + 1):
        x = 0.5 * (1 - math.cos(math.pi * i / n))
        yt = 5 * tt * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if m and pp:
            yc = m / pp ** 2 * (2 * pp * x - x * x) if x < pp else m / (1 - pp) ** 2 * (1 - 2 * pp + 2 * pp * x - x * x)
            dy = 2 * m / pp ** 2 * (pp - x) if x < pp else 2 * m / (1 - pp) ** 2 * (pp - x)
        else:
            yc = dy = 0.0
        th = math.atan(dy)
        pts_u.append(((x - yt * math.sin(th) - 0.25) * chord, (yc + yt * math.cos(th)) * chord))
        pts_l.append(((x + yt * math.sin(th) - 0.25) * chord, (yc - yt * math.cos(th)) * chord))
    pts = pts_u[::-1] + pts_l[1:-1]
    return pts


def axial_fan(p: Params):
    """Hub cylinder + n blades lofted from a root to a tip NACA section with twist, polar pattern."""
    cq = _cq()
    n, r_hub, r_tip, h_hub = int(p["n_blades"]), p["hub_d"] / 2, p["tip_d"] / 2, p["hub_length"]
    c_root, c_tip = p["root_chord"], p["tip_chord"]
    a_root, a_tip = p["root_pitch_deg"], p["tip_pitch_deg"]
    hub = cq.Workplane("XY").circle(r_hub).extrude(h_hub).translate((0, 0, -h_hub / 2))

    # blade: loft from a root to a tip NACA section; section plane normal along the span (+X),
    # local x -> -Y (tangential), local y -> +Z (axial); the pitch angle rotates each section
    blade = (cq.Workplane(cq.Plane(origin=(r_hub * 0.9, 0, 0), xDir=(0, -1, 0), normal=(1, 0, 0)))
             .polyline(_rot(_naca4_points(p["naca"], c_root), a_root)).close()
             .workplane(offset=r_tip - r_hub * 0.9)
             .polyline(_rot(_naca4_points(p["naca"], c_tip), a_tip)).close()
             .loft(ruled=True))  # polylines: a periodic spline through the sharp trailing edge fails in OCC
    rotor = hub
    for k in range(n):
        rotor = rotor.union(blade.rotate((0, 0, 0), (0, 0, 1), 360.0 * k / n))
    return rotor


def _rot(pts, deg):
    a = math.radians(deg)
    return [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)) for x, y in pts]


def involute_tooth_profile(module: float, teeth: int, pressure_deg: float = 20.0, n: int = 12) -> List[Tuple[float, float]]:
    """Closed outline of a spur gear (involute flanks from base to tip circle, root arcs)."""
    alpha = math.radians(pressure_deg)
    rp = module * teeth / 2
    rb = rp * math.cos(alpha)
    ra = rp + module  # addendum
    rf = rp - 1.25 * module  # dedendum
    inv = lambda a: math.tan(a) - a
    half = math.pi / (2 * teeth) + inv(alpha)  # half tooth thickness angle at the base circle
    t_max = math.sqrt((ra / rb) ** 2 - 1)
    flank = []
    for i in range(n + 1):
        t = t_max * i / n
        r = rb * math.sqrt(1 + t * t)
        phi = t - math.atan(t)
        flank.append((r, phi))
    pts = []
    for k in range(teeth):
        c = 2 * math.pi * k / teeth
        start = max(rf, 1e-9)
        pts.append((start * math.cos(c - half), start * math.sin(c - half)))  # root, before the flank
        for r, phi in flank:
            a = c - half + phi
            pts.append((r * math.cos(a), r * math.sin(a)))
        for r, phi in reversed(flank):
            a = c + half - phi
            pts.append((r * math.cos(a), r * math.sin(a)))
        pts.append((start * math.cos(c + half), start * math.sin(c + half)))
    return pts


def spur_gear(p: Params):
    cq = _cq()
    pts = involute_tooth_profile(p["module"], int(p["teeth"]), p.get("pressure_deg", 20.0))
    g = cq.Workplane("XY").polyline(pts).close().extrude(p["width"])
    if p.get("bore_d"):
        g = g.faces(">Z").workplane().hole(p["bore_d"])
        if p.get("key_w"):
            kw, kd = p["key_w"], p.get("key_depth", p["key_w"] / 2)
            key = cq.Workplane("XY").center(p["bore_d"] / 2, 0).rect(2 * kd, kw).extrude(p["width"])
            g = g.cut(key)
    return g


def pin_fin_heatsink(p: Params):
    cq = _cq()
    Lx, Ly, tb = p["base_x"], p["base_y"], p["base_t"]
    nx, ny, pitch, d, h = int(p["nx"]), int(p["ny"]), p["pitch"], p["pin_d"], p["pin_h"]
    base = cq.Workplane("XY").box(Lx, Ly, tb, centered=(True, True, False))
    pins = base.faces(">Z").workplane().rarray(pitch, pitch, nx, ny).circle(d / 2).extrude(h)
    if p.get("fillet"):
        pins = pins.faces(">Z").edges().fillet(p["fillet"])
    return pins


# ── reproductions of the CadQuery documentation examples ───────────────
def occ_bottle(p: Params):
    """CadQuery docs, "The Classic OCC Bottle"."""
    cq = _cq()
    L, w, t = p.get("L", 20.0), p.get("w", 6.0), p.get("t", 3.0)
    s = cq.Workplane("XY")
    body = (s.center(-L / 2.0, 0).vLine(w / 2.0).threePointArc((L / 2.0, w / 2.0 + t), (L, w / 2.0))
            .vLine(-w / 2.0).mirrorX().extrude(p.get("height", 30.0), True))
    body = body.faces(">Z").workplane(centerOption="CenterOfMass").circle(p.get("neck_r", 3.0)).extrude(p.get("neck_h", 2.0), True)
    return body.faces(">Z").shell(p.get("shell", 0.3))


def pillow_block(p: Params):
    """CadQuery docs, "Parametric Bearing Pillow Block"."""
    cq = _cq()
    length, height, bearing_d, thickness, padding = (p.get("length", 30.0), p.get("height", 40.0),
                                                      p.get("bearing_d", 22.0), p.get("thickness", 10.0),
                                                      p.get("padding", 8.0))
    return (cq.Workplane("XY").box(length, height, thickness).faces(">Z").workplane().hole(bearing_d)
            .faces(">Z").workplane().rect(length - padding, height - padding, forConstruction=True)
            .vertices().cboreHole(2.4, 4.4, 2.1))


def lego_brick(p: Params):
    """CadQuery docs example "Lego Brick" (studs on top, tubes underneath)."""
    cq = _cq()
    lbumps, wbumps = int(p.get("lbumps", 6)), int(p.get("wbumps", 2))
    pitch, clearance, bump_d, bump_h = 8.0, 0.1, 4.8, 1.8
    height = 3.2 if p.get("thin", True) else 9.6
    t = (pitch - 2 * clearance - bump_d) / 2.0
    post_d = pitch - t
    s = cq.Workplane("XY").box(lbumps * pitch - 2 * clearance, wbumps * pitch - 2 * clearance, height)
    s = s.faces("<Z").shell(-1.0 * t)
    s = s.faces(">Z").workplane().rarray(pitch, pitch, lbumps, wbumps, True).circle(bump_d / 2.0).extrude(bump_h)
    tmp = s.faces("<Z").workplane(invert=True)
    if lbumps > 1 and wbumps > 1:
        tmp = (tmp.rarray(pitch, pitch, lbumps - 1, wbumps - 1, center=True)
               .circle(post_d / 2.0).circle(bump_d / 2.0).extrude(height - t))
    elif lbumps > 1:
        tmp = tmp.rarray(pitch, pitch, lbumps - 1, 1, center=True).circle(t).extrude(height - t)
    elif wbumps > 1:
        tmp = tmp.rarray(pitch, pitch, 1, wbumps - 1, center=True).circle(t).extrude(height - t)
    return tmp


ADVANCED_TEMPLATES = {
    "pipe_bend_flanged": (pipe_bend_flanged, {
        "od": {"type": "float", "min": 1e-3, "default": 0.1143}, "wall": {"type": "float", "min": 1e-4, "default": 0.006},
        "bend_radius": {"type": "float", "min": 1e-3, "default": 0.1524}, "angle_deg": {"type": "float", "min": 1, "max": 180, "default": 90.0},
        "flange_od": {"type": "float", "default": 0.229}, "flange_t": {"type": "float", "default": 0.024},
        "n_bolts": {"type": "int", "min": 3, "default": 8}, "bolt_d": {"type": "float", "default": 0.019},
        "bolt_circle_d": {"type": "float", "default": 0.1905}}),
    "pipe_tee": (pipe_tee, {
        "od": {"type": "float", "default": 0.1143}, "wall": {"type": "float", "default": 0.006},
        "run_length": {"type": "float", "default": 0.4}, "branch_height": {"type": "float", "default": 0.2},
        "fillet": {"type": "float", "default": 0.0}}),
    "concentric_reducer": (concentric_reducer, {
        "d1": {"type": "float", "default": 0.1143}, "d2": {"type": "float", "default": 0.0603},
        "length": {"type": "float", "default": 0.127}, "wall": {"type": "float", "default": 0.005}}),
    "helical_coil": (helical_coil, {
        "coil_radius": {"type": "float", "default": 0.05}, "pitch": {"type": "float", "default": 0.02},
        "turns": {"type": "float", "min": 0.25, "default": 5.0}, "tube_d": {"type": "float", "default": 0.008}}),
    "axial_fan": (axial_fan, {
        "n_blades": {"type": "int", "min": 2, "max": 20, "default": 5}, "hub_d": {"type": "float", "default": 0.08},
        "tip_d": {"type": "float", "default": 0.3}, "hub_length": {"type": "float", "default": 0.05},
        "root_chord": {"type": "float", "default": 0.06}, "tip_chord": {"type": "float", "default": 0.04},
        "root_pitch_deg": {"type": "float", "default": 45.0}, "tip_pitch_deg": {"type": "float", "default": 20.0},
        "naca": {"type": "str", "default": "4412"}}),
    "spur_gear": (spur_gear, {
        "module": {"type": "float", "default": 2.0}, "teeth": {"type": "int", "min": 8, "default": 24},
        "pressure_deg": {"type": "float", "default": 20.0}, "width": {"type": "float", "default": 10.0},
        "bore_d": {"type": "float", "default": 10.0}, "key_w": {"type": "float", "default": 3.0}}),
    "pin_fin_heatsink": (pin_fin_heatsink, {
        "base_x": {"type": "float", "default": 0.06}, "base_y": {"type": "float", "default": 0.06},
        "base_t": {"type": "float", "default": 0.004}, "nx": {"type": "int", "default": 6}, "ny": {"type": "int", "default": 6},
        "pitch": {"type": "float", "default": 0.009}, "pin_d": {"type": "float", "default": 0.003},
        "pin_h": {"type": "float", "default": 0.02}, "fillet": {"type": "float", "default": 0.0}}),
    "occ_bottle": (occ_bottle, {}),
    "pillow_block": (pillow_block, {}),
    "lego_brick": (lego_brick, {"lbumps": {"type": "int", "min": 1, "default": 6},
                                "wbumps": {"type": "int", "min": 1, "default": 2}, "thin": {"type": "bool", "default": True}}),
}


def register_advanced_templates(reg):
    """Add every builder above to a :class:`CadQueryRegistry` (with its parameter schema)."""
    for name, (fn, schema) in ADVANCED_TEMPLATES.items():
        reg.register(name, fn, schema=schema)
    return reg
