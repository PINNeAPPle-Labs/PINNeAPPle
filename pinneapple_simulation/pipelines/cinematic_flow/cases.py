"""The cases a film can be made of: the closed surfaces, and how the tunnel is placed around them.

A case is geometry the library already owns — the parametric road car, a launcher, a wing section — plus the
two numbers that decide what the picture can show: how many cells resolve the length of the body, and how
long the wake box is behind it. Everything here is in metres, with the body's own axes: ``x`` downstream,
``z`` up, the ground at ``z = 0`` when the case sits on one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Case", "CASES", "build", "wing_bodies"]

# name -> how it is described in a caption, and the material split the render uses
CASES = {
    "carro": "RoadCar — fastback de rua, rodas abertas",
    "asa": "NACA 2412 a 16°, ponta de asa livre",
    "foguete": "Lançador de duas etapas, quatro aletas em +",
}


@dataclass
class Case:
    """A body (or a set of closed bodies) and the box the flow is solved in around it."""

    name: str
    bodies: dict[str, tuple[np.ndarray, np.ndarray]]   # part -> (vertices [m], faces)
    ground: float | None = None                        # z of the floor; None is free air
    # tunnel proportions, in lengths of the body
    upstream: float = 1.2
    downstream: float = 2.8
    lateral: float = 0.9
    overhead: float = 1.2
    # what the picture needs that a solver run does not care about
    paint: dict[str, str] = field(default_factory=dict)   # part -> "paint" | "tyre" | "glass"
    caption: str = ""
    # the detailed surfaces the render shows (glass, rims, lamps); the solve keeps the closed bodies above
    visual: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    visual_paint: dict[str, str] = field(default_factory=dict)

    @property
    def render_parts(self) -> tuple[dict, dict]:
        return (self.visual or self.bodies), (self.visual_paint or self.paint)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v = np.vstack([np.asarray(p[0], float) for p in self.bodies.values()])
        return v.min(axis=0), v.max(axis=0)

    @property
    def length(self) -> float:
        lo, hi = self.bounds
        return float(hi[0] - lo[0])

    def scaled(self, factor: float) -> Case:
        """The same case in other units — the tunnel only sees ratios, so a 42 m launcher and a 4 m car are the
        same solve once the cells per length are fixed."""
        bodies = {k: (np.asarray(v, float) * factor, f) for k, (v, f) in self.bodies.items()}
        g = None if self.ground is None else self.ground * factor
        return Case(self.name, bodies, g, self.upstream, self.downstream, self.lateral, self.overhead,
                    dict(self.paint), self.caption)


def _loft(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A closed solid through closed profiles of equal length, with a fan at each end."""
    m = len(rings[0])
    V = np.vstack(rings)
    F = []
    for i in range(len(rings) - 1):
        a, b = i * m, (i + 1) * m
        for j in range(m):
            k = (j + 1) % m
            F += [(a + j, a + k, b + k), (a + j, b + k, b + j)]
    c0 = V[:m].mean(axis=0)
    c1 = V[-m:].mean(axis=0)
    V = np.vstack([V, c0, c1])
    i0, i1 = len(V) - 2, len(V) - 1
    for j in range(m):
        k = (j + 1) % m
        F += [(i0, k, j), (i1, (len(rings) - 1) * m + j, (len(rings) - 1) * m + k)]
    return V, np.asarray(F, int)


def wing_bodies(span: float = 3.0, chord: float = 1.0, aoa_deg: float = 16.0, taper: float = 0.55,
                m: float = 0.04, t: float = 0.12, n: int = 72, stations: int = 9) -> dict:
    """A straight tapered wing on a NACA 4-digit section, at incidence ``aoa_deg``, as one closed body.

    The section is the library's own :func:`pinneapple_design.geometry.vehicles3d.naca4`; the twist about the
    quarter chord is what gives the leading-edge vortex the picture is after."""
    from pinneapple_design.geometry.vehicles3d import naca4

    profile = naca4(m, 0.4, t, n)                     # (n, 2), unit chord, closed
    a = math.radians(aoa_deg)
    rings = []
    for s in np.linspace(-0.5, 0.5, stations):
        c = chord * (1.0 - (1.0 - taper) * abs(s) * 2.0)
        xz = profile * c - np.array([0.25 * c, 0.0])   # quarter chord at the origin
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        xz = xz @ R.T                                  # nose up: the flow meets the lower surface
        y = s * span
        rings.append(np.c_[xz[:, 0], np.full(len(xz), y), xz[:, 1]])
    V, F = _loft(rings)
    return {"wing": (V, F)}


def build(name: str) -> Case:
    """The case ``name`` as geometry and tunnel proportions, in metres."""
    if name == "carro":
        from pinneapple_design.geometry.vehicles3d import RoadCar

        car = RoadCar(length=4.62, width=1.87, roof_height=1.38, clearance=0.14, deck_height=1.03,
                      windshield_deg=27.0, slant_deg=21.0, style="fastback", boat_tail=0.07, diffuser_deg=7.0,
                      wheel_radius=0.345, wheel_width=0.245, wheelbase=2.82, front_overhang=0.92,
                      n_x=200, n_ring=112)
        # The render shows the body as it is drawn, not as it is meshed — but the drawn shell is an open surface:
        # its boundary runs along the sills and the arches, and at film scale those free edges read as a sawtooth
        # silhouette. So the paint is the closed CFD body, and the detail parts sit a tenth of a percent outside
        # it, which is enough to keep them out of a depth fight with the paint they are laid on.
        closed = dict(car.cfd_bodies())
        body_v = np.asarray(closed["body"][0], float)
        centre = 0.5 * (body_v.min(axis=0) + body_v.max(axis=0))
        visual = {"body": closed["body"]}
        kinds = {"body": "paint"}
        for part in car.parts():
            mat = part.material
            if mat in ("car_paint", "car_trim"):
                continue
            if mat in ("car_glass", "headlight", "taillight"):
                v = centre + (np.asarray(part.vertices, float) - centre) * 1.003
                visual[part.name] = (v, np.asarray(part.faces, int))
                kinds[part.name] = {"car_glass": "glass", "headlight": "lamp", "taillight": "tail"}[mat]
            elif mat in ("tyre", "rim"):
                visual[part.name] = (np.asarray(part.vertices, float), np.asarray(part.faces, int))
                kinds[part.name] = mat
        return Case("carro", dict(car.cfd_bodies()), ground=0.0,
                    paint={"body": "paint", "wheels": "tyre"},
                    caption=CASES["carro"], visual=visual, visual_paint=kinds)
    if name == "asa":
        bodies = wing_bodies()
        # the box is measured on the span, which is three chords: the wing is thin, so ``overhead`` and
        # ``lateral`` are read against its own 0.29 m height and would leave the wake no room at all
        return Case("asa", bodies, ground=None, upstream=1.0, downstream=2.0, lateral=0.5, overhead=2.5,
                    paint={"wing": "paint"}, caption=CASES["asa"])
    if name == "foguete":
        from pinneapple_design.geometry.vehicles3d import LaunchVehicle

        r = LaunchVehicle(length=12.0, diameter=1.0)   # a sounding rocket: the fins stay resolvable
        # a launcher is 12 m long and 5.8 m across the fins, so the room behind it is measured in lengths and the
        # sides only need to clear the fin tips
        return Case("foguete", dict(r.cfd_bodies()), ground=None, upstream=0.8, downstream=2.0,
                    lateral=0.25, overhead=0.35, paint={"rocket": "paint", "fins": "paint"},
                    caption=CASES["foguete"])
    raise KeyError(f"caso desconhecido: {name!r} (há {sorted(CASES)})")
