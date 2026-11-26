"""Scale: the metres behind a scene, and the set that shows them.

A render normalises everything it measures on the model's own bounding box, so on its own a picture of a 25 mm pin
and a picture of a 36 m airliner are indistinguishable -- nothing in the frame is measurable. This module supplies
the two things that make a picture measurable: the conversion from the scene's coordinates to metres
(:func:`unit_scale`), and the metric furniture built at that size -- a floor, a cyclorama, a metric grid and props
whose real size is known (:func:`cyclorama`, :func:`grid_lines`, :func:`prop`).

Every builder here returns plain ``(vertices, faces)`` arrays of a closed, triangulated mesh **in scene units**, z up,
so it can be added to a :class:`~.scene.Scene`, exported to glTF and rendered under any background. The sizes are in
metres and are converted by the caller (:func:`~.blender.render`), which is the one place that knows the unit:

    from pinneapple_tools.visualization.studio import scale
    unit = scale.unit_scale("mm")                     # 0.001: one scene unit is a millimetre
    step = scale.grid_step(0.25, squares=10)           # 25 mm squares under a 250 mm part
    print(scale.fmt_length(0.25))                      # "250 mm"
    print(scale.auto_prop(0.25))                       # "coin": the largest prop that does not dwarf the part
"""
from __future__ import annotations

import math
import re

import numpy as np

#: name -> metres per scene unit. A scene's ``unit`` is metres per coordinate unit, so a model drawn in millimetres
#: says ``"mm"``.
UNITS: dict[str, float] = {
    "m": 1.0, "metre": 1.0, "meter": 1.0, "metres": 1.0, "meters": 1.0,
    "cm": 0.01, "centimetre": 0.01, "centimeter": 0.01, "centimetres": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimetre": 0.001, "millimeter": 0.001, "millimetres": 0.001, "millimeters": 0.001,
    "um": 1e-6, "µm": 1e-6, "micron": 1e-6, "microns": 1e-6,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254, '"': 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "'": 0.3048,
}

#: Props whose size a reader already knows, largest first: (name, its real size in metres). The figure is 1.75 m tall,
#: the rule is 300 mm long, A4 is 297 mm on its long side and the coin is 25.4 mm across.
PROPS: tuple[tuple[str, float], ...] = (("human", 1.75), ("ruler", 0.30), ("a4", 0.297), ("coin", 0.0254))

#: The props :func:`prop` can build.
PROP_KINDS = tuple(name for name, _ in PROPS)

#: How much floor each prop takes, as (along its own x, across it), in metres -- what has to fit beside the subject.
PROP_FOOTPRINT: dict[str, tuple[float, float]] = {
    "human": (0.55, 0.35), "ruler": (0.300, 0.030), "a4": (0.297, 0.210), "coin": (0.0254, 0.0254),
}


# ------------------------------------------------------------------ units
def unit_scale(units) -> float:
    """Metres per scene coordinate unit: ``1.0`` (a model already in metres) when ``units`` is None, the number
    itself when it is one (``0.001`` for millimetres), or the :data:`UNITS` entry when it is a name ("mm", "in", ...)."""
    if units is None:
        return 1.0
    if isinstance(units, (int, float)) and not isinstance(units, bool):
        u = float(units)
        if not math.isfinite(u) or u <= 0.0:
            raise ValueError(f"units={units!r}: metres per scene unit must be a positive number")
        return u
    key = str(units).strip().lower()
    if key in UNITS:
        return UNITS[key]
    raise ValueError(f"units={units!r}: a number of metres per scene unit, or one of {', '.join(sorted(UNITS))}")


def parse_length(value) -> float:
    """A length in metres from a number (already metres) or a string with its unit: "0.025", "25mm", "1.5 m",
    "2 ft". Used where one number has to accept "the grid square is 25 mm"."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip().lower().replace(" ", "")
    m = re.fullmatch(r"([-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)(.*)", s)
    if not m:
        raise ValueError(f"length={value!r}: a number of metres or a string like '25 mm'")
    num, suffix = float(m.group(1)), m.group(2)
    if not suffix:
        return num
    if suffix in UNITS:
        return num * UNITS[suffix]
    raise ValueError(f"length={value!r}: {suffix!r} is not a unit ({', '.join(sorted(UNITS))})")


def fmt_length(metres: float) -> str:
    """A length in metres as an engineer writes it, in three significant figures at most: "25.4 mm", "250 mm",
    "1.75 m", "36 m", "100 µm"."""
    x = abs(float(metres))
    if not math.isfinite(x):
        return "?"
    if x < 1e-3:
        unit, v = "µm", x * 1e6
    elif x < 1.0:
        unit, v = "mm", x * 1e3
    else:
        unit, v = "m", x
    if v >= 100.0:
        txt = f"{v:,.0f}"
    elif v >= 10.0:
        txt = f"{v:,.1f}".rstrip("0").rstrip(".")
    else:
        txt = f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{txt} {unit}"


def decade(x: float) -> float:
    """The round engineering value at or just below ``x``: 1, 2 or 5 times a power of ten (the steps a rule is
    graduated in). Used to pick a grid square and a scale-bar length a reader can hold in mind."""
    x = abs(float(x))
    if not math.isfinite(x) or x <= 0.0:
        return 1.0
    e = math.floor(math.log10(x))
    for m in (5.0, 2.0, 1.0):
        v = m * 10.0 ** e
        if v <= x * (1.0 + 1e-9):
            return v
    return 10.0 ** e


def grid_step(extent_m: float, squares: int = 10) -> float:
    """The spacing of a floor grid that puts about ``squares`` squares across ``extent_m``, snapped to a round value
    (:func:`decade`): a 250 mm part gets 25 mm squares, a 36 m airliner 5 m ones."""
    return decade(max(float(extent_m), 1e-9) / max(1, int(squares)))


def auto_prop(extent_m: float, *, fill: float = 0.8) -> str | None:
    """The largest prop of :data:`PROPS` that is at most ``fill`` of ``extent_m`` -- the biggest object of known size
    that does not compete with the subject -- or None when the part is smaller than every prop (a 25 mm pin gets no
    prop: only the grid and the scale bar can measure it)."""
    for name, size in PROPS:
        if size <= fill * float(extent_m):
            return name
    return None


# ------------------------------------------------------------------ mesh helpers
def _perp(n: np.ndarray):
    """Two unit vectors perpendicular to ``n`` and to each other."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u) or 1.0
    return u, np.cross(n, u)


def _circle(nseg: int) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    return np.stack([np.cos(ang), np.sin(ang)], -1)


def merge(*parts):
    """One ``(vertices, faces)`` pair out of several, offsetting the face indices."""
    V, F, off = [], [], 0
    for v, f in parts:
        v, f = np.asarray(v, float).reshape(-1, 3), np.asarray(f, int).reshape(-1, 3)
        if not len(v) or not len(f):
            continue
        V.append(v)
        F.append(f + off)
        off += len(v)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(V), np.vstack(F)


def capsule(a, b, r: float, nseg: int = 14, ncap: int = 3):
    """A capsule from ``a`` to ``b`` of radius ``r`` with hemispherical caps, closed and triangulated: a limb, a strut,
    a stand-in for a person."""
    a, b = np.asarray(a, float).reshape(3), np.asarray(b, float).reshape(3)
    ax = b - a
    L = float(np.linalg.norm(ax))
    if L < 1e-12:
        return sphere(a, r, nseg, 2 * ncap)
    n = ax / L
    u, v = _perp(n)
    circ = _circle(nseg)
    circle = circ[:, :1] * u + circ[:, 1:] * v                       # (nseg, 3), radius 1
    rings = []
    for k in range(1, ncap + 1):                                     # bottom hemisphere: pole -> equator at a
        phi = 0.5 * math.pi * k / ncap
        rings.append(a - n * (r * math.cos(phi)) + circle * (r * math.sin(phi)))
    rings.append(b + circle * r)                                     # equator at b
    for k in range(ncap - 1, 0, -1):                                 # top hemisphere: equator -> pole
        phi = 0.5 * math.pi * k / ncap
        rings.append(b + n * (r * math.cos(phi)) + circle * (r * math.sin(phi)))
    V = np.vstack([a - n * r] + rings + [b + n * r])                 # poles first and last
    F = []
    first = 1
    for j in range(nseg):                                            # bottom fan, pole -> first ring
        F.append([0, first + j, first + (j + 1) % nseg])
    for i in range(len(rings) - 1):                                  # quad bands between the rings
        o0, o1 = first + i * nseg, first + (i + 1) * nseg
        for j in range(nseg):
            jn = (j + 1) % nseg
            F.append([o0 + j, o1 + j, o1 + jn])
            F.append([o0 + j, o1 + jn, o0 + jn])
    top = len(V) - 1
    last = first + (len(rings) - 1) * nseg
    for j in range(nseg):                                            # top fan, last ring -> pole
        F.append([top, last + (j + 1) % nseg, last + j])
    return V, np.asarray(F, int)


def sphere(centre, r: float, nseg: int = 16, nring: int = 8):
    """A UV sphere, rings between two poles."""
    c = np.asarray(centre, float).reshape(3)
    ang = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    out = [c + np.array([0.0, 0.0, -r])]
    for k in range(1, nring):
        th = math.pi * k / nring
        out.append(c + r * np.stack([np.sin(th) * np.cos(ang), np.sin(th) * np.sin(ang),
                                     np.full(nseg, math.cos(th))], -1))
    out.append(c + np.array([0.0, 0.0, r]))
    V = np.vstack(out)
    F = []
    for k in range(1, nring - 1):
        o0, o1 = 1 + (k - 1) * nseg, 1 + k * nseg
        for j in range(nseg):
            jn = (j + 1) % nseg
            F += [[o0 + j, o1 + j, o1 + jn], [o0 + j, o1 + jn, o0 + jn]]
    last = 1 + (nring - 2) * nseg
    top = len(V) - 1
    for j in range(nseg):
        jn = (j + 1) % nseg
        F += [[0, 1 + jn, 1 + j], [top, last + j, last + jn]]
    return V, np.asarray(F, int)


def box(centre, size, rot=None):
    """An axis-aligned box (or one turned by the 3x3 ``rot``) as 12 triangles."""
    c = np.asarray(centre, float).reshape(3)
    s = np.asarray(size, float).reshape(3) / 2.0
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * s
    if rot is not None:
        corners = corners @ np.asarray(rot, float).T
    V = corners + c
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return V, np.asarray(F, int)


# ------------------------------------------------------------------ the set
def cyclorama(centre_xy, floor_z: float, *, radius: float, cove_radius: float, wall_height: float,
              n_theta: int = 48, n_arc: int = 10):
    """A cyclorama: a floor that curves up into a wall all the way round, so the subject stands in a seamless cove
    instead of floating, from whatever side the camera looks. This is the shape a photographic studio actually has --
    an infinity cove on every side -- and being a surface of revolution it needs no knowledge of the camera.

    ``centre_xy`` is the subject's footprint centre, ``floor_z`` the floor, ``radius`` where the flat floor ends and
    the cove begins, ``cove_radius`` the radius of the curve up from there and ``wall_height`` how high the wall rises
    above the floor. All lengths in scene units. A camera placed inside the bowl sees cove and nothing else, which is
    the point: the world image then only lights the set."""
    c = np.asarray(centre_xy, float).reshape(2)
    prof = [(0.0, 0.0), (float(radius), 0.0)]
    for k in range(1, n_arc + 1):                        # the cove: horizontal at `radius`, vertical r higher
        phi = 0.5 * math.pi * k / n_arc
        prof.append((radius + cove_radius * math.sin(phi), cove_radius * (1.0 - math.cos(phi))))
    if wall_height > cove_radius + 1e-12:
        prof.append((radius + cove_radius, float(wall_height)))
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    V = [[float(c[0]), float(c[1]), float(floor_z)]]
    for (r, z) in prof[1:]:
        for t in ang:
            V.append([c[0] + r * math.cos(t), c[1] + r * math.sin(t), floor_z + z])
    F = []
    for j in range(n_theta):                             # the flat floor, fanned from its centre
        F.append([0, 1 + j, 1 + (j + 1) % n_theta])
    for i in range(len(prof) - 2):                       # rings out to the wall
        o0, o1 = 1 + i * n_theta, 1 + (i + 1) * n_theta
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            F += [[o0 + j, o0 + jn, o1 + jn], [o0 + j, o1 + jn, o1 + j]]
    return np.asarray(V, float), np.asarray(F, int)


def floor_plane(centre_xy, floor_z: float, radius: float):
    """A square floor ``2*radius`` across, centred on ``centre_xy``."""
    x, y = float(np.asarray(centre_xy, float).reshape(-1)[0]), float(np.asarray(centre_xy, float).reshape(-1)[1])
    V = np.array([[x - radius, y - radius, floor_z], [x + radius, y - radius, floor_z],
                  [x + radius, y + radius, floor_z], [x - radius, y + radius, floor_z]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]], int)


def grid_lines(centre_xy, floor_z: float, *, step: float, radius: float, width: float,
               heading: float = 0.0, major_every: int = 5, lift: float = 0.0):
    """A metric grid painted on the flat floor: lines every ``step`` along two perpendicular directions, every
    ``major_every``-th twice as wide, each a flat quad ``width`` wide lying ``lift`` above the floor (enough to beat
    z-fighting). ``radius`` clips the grid to the floor's disc, so no line runs up the cove; ``heading`` turns the
    grid about the vertical axis, in radians."""
    c = np.asarray(centre_xy, float).reshape(2)
    ca, sa = math.cos(float(heading)), math.sin(float(heading))
    back = np.array([ca, sa])
    side = np.array([-sa, ca])
    z = floor_z + lift
    R = float(radius)
    n = int(math.floor(R / step))
    if 2 * n + 1 > 400:
        raise ValueError(f"grid: {2 * n + 1} lines per direction is a wall of ink; widen the step or shrink the floor")
    parts = []
    for k in range(-n, n + 1):                           # lines one way, clipped by the disc, both directions
        d = k * step
        half = math.sqrt(max(0.0, R * R - d * d))
        if half <= 0.0:
            continue
        w = width * (2.0 if (major_every and k % major_every == 0) else 1.0)
        parts.append(_strip(c + back * d - side * half, c + back * d + side * half, w, z, back))
        parts.append(_strip(c + side * d - back * half, c + side * d + back * half, w, z, side))
    return merge(*parts)


def _strip(p0, p1, width: float, z: float, across):
    """A flat quad from ``p0`` to ``p1``, ``width`` wide, widened along ``across``."""
    a = np.asarray(p0, float).reshape(2)
    b = np.asarray(p1, float).reshape(2)
    n = np.asarray(across, float).reshape(2)
    n = n / (np.linalg.norm(n) or 1.0) * (width / 2.0)
    V = np.array([[*(a - n), z], [*(b - n), z], [*(b + n), z], [*(a + n), z]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]], int)


# ------------------------------------------------------------------ props of known size
def prop_size(kind: str, height: float = 1.75) -> float:
    """The real size of a prop in metres -- the length that reads on the caption: the figure's height, the rule's
    length, A4's long side, the coin's diameter. How much floor it takes is :func:`footprint`."""
    kind = str(kind).strip().lower()
    if kind == "human":
        return float(height)
    for name, size in PROPS:
        if name == kind:
            return size
    raise ValueError(f"prop kind={kind!r}: one of {', '.join(PROP_KINDS)}")


def footprint(kind: str) -> tuple[float, float]:
    """How much floor a prop takes, as (along its own x, across it), in metres -- what has to fit beside the
    subject."""
    kind = str(kind).strip().lower()
    if kind in PROP_FOOTPRINT:
        return PROP_FOOTPRINT[kind]
    raise ValueError(f"prop kind={kind!r}: one of {', '.join(PROP_KINDS)}")


def prop(kind: str, at_xy, floor_z: float, *, unit: float = 1.0, height: float = 1.75, rotate: float = 0.0):
    """One prop of :data:`PROPS`, standing or lying on the floor at ``at_xy``, built at its real size.

    ``kind`` is "human" (a mannequin ``height`` metres tall, 1.75 m by default), "ruler" (a 300 mm rule with
    millimetre graduations), "a4" (a 210 x 297 mm sheet lying flat) or "coin" (a 25.4 mm disc). ``rotate`` turns it
    about z, in radians. The prop is written in metres and divided by ``unit`` (metres per scene unit), so ``at_xy``
    and ``floor_z`` are in scene units and the result is too."""
    kind = str(kind).strip().lower()
    at = np.asarray(at_xy, float).reshape(-1)
    x, y, z = float(at[0]), float(at[1]), float(floor_z)
    if kind == "human":
        V, F = _mannequin(float(height))
    elif kind == "ruler":
        V, F = _ruler()
    elif kind == "a4":
        V, F = _a4()
    elif kind == "coin":
        V, F = _coin()
    else:
        raise ValueError(f"prop kind={kind!r}: one of {', '.join(PROP_KINDS)}")
    return _place(np.asarray(V, float) / float(unit), x, y, z, rotate), F


def _rot2(rotate: float) -> np.ndarray:
    c, s = math.cos(rotate), math.sin(rotate)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _place(V, x, y, z, rotate):
    """Turn a prop written at the origin about z, then stand it on the floor at (x, y, z)."""
    return np.asarray(V, float).reshape(-1, 3) @ _rot2(rotate).T + np.array([x, y, z])


def _mannequin(height: float):
    """A scale figure -- head, neck, torso, hips, arms in a slight A-pose, legs -- with the proportions of a 1.75 m
    body scaled to ``height``. Enough for the eye to judge the subject by, with no attempt at a person."""
    k = float(height) / 1.75
    parts = [sphere((0.0, 0.0, 1.645 * k), 0.105 * k, 16, 8),                                  # head
             capsule((0.0, 0.0, 1.46 * k), (0.0, 0.0, 1.56 * k), 0.055 * k, 10, 2),            # neck
             capsule((0.0, 0.0, 0.95 * k), (0.0, 0.0, 1.46 * k), 0.150 * k, 16, 3),            # torso
             capsule((-0.085 * k, 0.0, 0.86 * k), (0.085 * k, 0.0, 0.86 * k), 0.115 * k, 12, 2)]   # hips
    for sx in (-1.0, 1.0):
        parts.append(capsule((sx * 0.175 * k, 0.0, 1.40 * k), (sx * 0.255 * k, 0.0, 0.72 * k), 0.052 * k, 10, 2))
        parts.append(capsule((sx * 0.085 * k, 0.0, 0.90 * k), (sx * 0.115 * k, 0.0, 0.08 * k), 0.072 * k, 12, 2))
        parts.append(box((sx * 0.115 * k, -0.045 * k, 0.035 * k), (0.095 * k, 0.23 * k, 0.07 * k)))
    return merge(*parts)


def _ruler():
    """A 300 mm engineer's rule lying on the floor: a plate with a graduation every millimetre, every tenth one longer,
    so the eye can count them."""
    L, W, T = 0.300, 0.030, 0.0018
    parts = [box((0.0, 0.0, T / 2.0), (L, W, T))]
    for i in range(int(round(L * 1000)) + 1):
        long_tick = (i % 10 == 0)
        tick_len = W * (0.75 if long_tick else 0.40)
        d = -L / 2.0 + i * 0.001
        parts.append(box((d, -W / 2.0 + tick_len / 2.0, T + 0.0002), (0.0004, tick_len, 0.0004)))
    return merge(*parts)


def _a4():
    """A sheet of A4 (210 x 297 mm) lying on the floor, 0.8 mm thick."""
    return box((0.0, 0.0, 0.0004), (0.297, 0.210, 0.0008))


def _coin():
    """A 25.4 mm disc, 2 mm thick: the smallest thing anybody has in a pocket to judge a part by."""
    r, t, nseg = 0.0127, 0.002, 32
    circ = _circle(nseg) * r
    ring = np.hstack([circ, np.zeros((nseg, 1))])
    V = np.vstack([ring, ring + [0.0, 0.0, t], [[0.0, 0.0, 0.0], [0.0, 0.0, t]]])
    F = []
    for j in range(nseg):
        jn = (j + 1) % nseg
        F += [[j, jn, nseg + jn], [j, nseg + jn, nseg + j],            # wall
              [2 * nseg, jn, j], [2 * nseg + 1, nseg + j, nseg + jn]]  # two faces, fanned
    return V, np.asarray(F, int)


# ------------------------------------------------------------------ reading the result
def scale_bar_length(target_m: float) -> float:
    """The length of a scale bar, in metres, for a graphic near ``target_m`` -- snapped to a round value so the label
    reads "0 ... 250 mm" instead of "0 ... 237 mm"."""
    return decade(max(float(target_m), 1e-9))


def px_per_metre(size_px, view_angle: float, distance_m: float) -> float:
    """The image scale at the subject: how many pixels one metre spans. ``size_px`` is the rendered (width, height),
    ``view_angle`` the camera's angle across the longer side (radians) and ``distance_m`` the camera-to-subject
    distance in metres. A bar drawn with this number is metrically right *at the subject's plane*, the convention
    every photograph with a scale bar uses."""
    w, h = (float(v) for v in size_px)
    focal_px = 0.5 * max(w, h) / max(1e-9, math.tan(0.5 * float(view_angle)))
    return focal_px / max(1e-9, float(distance_m))
