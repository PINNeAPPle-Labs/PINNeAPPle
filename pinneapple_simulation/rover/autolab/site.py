"""A synthetic planetary site whose truth is known: relief, soils, and what an orbiter would have said about them.

The site is a height function (planar tilt + Gaussian mounds) and a map of soil *units*. Each unit has a true soil
(hidden from the rover, used only by the simulator and to score what the rover learns) and an orbital class: the
soil an orbital map would assign to that unit before any wheel touched it. A unit whose orbital class is wrong (soft
sand mapped as compact regolith) is how a first route can fail.

    from pinneapple_simulation.rover.autolab import demo_site
    site = demo_site()
    site.unit_at(x, y)              # unit index per point (truth map; units are what the orbiter also sees)
    site.true_soil(k)               # hidden truth, for the simulator and for scoring
    site.prior_soil(k)              # what the orbital map assumes
    site.deformable_terrain(cell)   # DeformableTerrain for pinneapple_simulation.vehicle.VehicleSimulator

All numbers of :func:`demo_site` (positions, sizes, the soft-sand parameters) are choices made for this synthetic
site, not measurements of a real one; the compact regolith and the firm gravelly ground use library soils with
their recorded provenance (``GRC-1`` and ``JSC-1A`` in :mod:`pinneapple_simulation.terramechanics.soils`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...terramechanics.soils import Soil, get_soil


@dataclass(frozen=True)
class Mound:
    """Gaussian hill ``height * exp(-((x-x0)^2/sx^2 + (y-y0)^2/sy^2) / 2)`` [m]."""
    x: float
    y: float
    sx: float
    sy: float
    height: float

    def __call__(self, x, y):
        return self.height * np.exp(-0.5 * (((x - self.x) / self.sx) ** 2 + ((y - self.y) / self.sy) ** 2))


@dataclass(frozen=True)
class Patch:
    """An ellipse of one soil unit (rotated by ``angle_deg``); later patches cover earlier ones."""
    unit: int
    x: float
    y: float
    a: float
    b: float
    angle_deg: float = 0.0

    def contains(self, x, y):
        c, s = math.cos(math.radians(self.angle_deg)), math.sin(math.radians(self.angle_deg))
        dx, dy = np.asarray(x, float) - self.x, np.asarray(y, float) - self.y
        u, v = c * dx + s * dy, -s * dx + c * dy
        return (u / self.a) ** 2 + (v / self.b) ** 2 <= 1.0


@dataclass(frozen=True)
class SoilUnit:
    """A mapped unit: its name, its true soil (hidden) and the soil the orbital map assigns to it."""
    name: str
    truth: Soil
    orbital: Soil
    orbital_note: str = ""
    color: tuple = (0.6, 0.6, 0.6)


@dataclass
class SyntheticSite:
    x_range: tuple
    y_range: tuple
    units: list[SoilUnit]
    patches: list[Patch] = field(default_factory=list)
    mounds: list[Mound] = field(default_factory=list)
    tilt_deg: float = 0.0
    tilt_dir_deg: float = 0.0
    start: tuple = (0.0, 0.0)
    goal: tuple = (1.0, 0.0)
    description: str = ""

    # ------------------------------------------------------------------ relief
    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        a = math.radians(self.tilt_dir_deg)
        h = math.tan(math.radians(self.tilt_deg)) * (x * math.cos(a) + y * math.sin(a))
        for m in self.mounds:
            h = h + m(x, y)
        return h

    def gradient(self, x, y, eps: float = 1e-3):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return ((self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps),
                (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps))

    def grade_deg(self, x, y, heading_rad):
        """Grade along a heading [deg], positive uphill."""
        hx, hy = self.gradient(x, y)
        return np.degrees(np.arctan(hx * np.cos(heading_rad) + hy * np.sin(heading_rad)))

    def slope_deg(self, x, y):
        hx, hy = self.gradient(x, y)
        return np.degrees(np.arctan(np.hypot(hx, hy)))

    # ------------------------------------------------------------------ soils
    def unit_at(self, x, y) -> np.ndarray:
        """Unit index per point (0 = background unit)."""
        idx = np.zeros(np.shape(np.asarray(x, float)), int)
        for p in self.patches:
            idx = np.where(p.contains(x, y), p.unit, idx)
        return idx

    def true_soil(self, k: int) -> Soil:
        return self.units[k].truth

    def prior_soil(self, k: int) -> Soil:
        return self.units[k].orbital

    def true_soils(self) -> list[Soil]:
        return [u.truth for u in self.units]

    def mislabelled_units(self) -> list[int]:
        """Units whose orbital soil differs from the truth (what the rover has to find out)."""
        keys = ("c", "phi_deg", "K", "k_c", "k_phi", "n0", "n1")
        return [k for k, u in enumerate(self.units)
                if any(abs(getattr(u.truth, q) - getattr(u.orbital, q)) > 1e-9 * max(1.0, abs(getattr(u.truth, q)))
                       for q in keys)]

    # ------------------------------------------------------------------ sampling
    def grid(self, cell: float = 0.25):
        xs = np.arange(self.x_range[0], self.x_range[1] + 1e-9, cell)
        ys = np.arange(self.y_range[0], self.y_range[1] + 1e-9, cell)
        X, Y = np.meshgrid(xs, ys)
        return X, Y, self.height(X, Y)

    def deformable_terrain(self, cell: float = 0.02, margin: float = 1.0, relief: float = 0.0, seed: int = 0,
                           x_range=None, y_range=None):
        """A :class:`~pinneapple_simulation.terramechanics.terrain.DeformableTerrain` over the site (ruts recorded on
        a ``cell`` grid) for the multibody simulator."""
        from ...terramechanics.terrain import DeformableTerrain
        xr = x_range or (self.x_range[0] - margin, self.x_range[1] + margin)
        yr = y_range or (self.y_range[0] - margin, self.y_range[1] + margin)
        return DeformableTerrain(self, xr, yr, cell=cell, relief=relief, relief_seed=seed)


# ---------------------------------------------------------------------------------------------- the demo site
SOFT_SAND_NOTE = ("synthetic truth chosen for this demo (loose, weakly cohesive sand: low Bekker k_phi, sinkage "
                  "exponent above 1, strong slip-sinkage, lower friction angle than GRC-1); not a measured soil")


def soft_sand() -> Soil:
    """The hidden soft sand of the demo site: GRC-1 with a looser state (see :data:`SOFT_SAND_NOTE`)."""
    base = get_soil("GRC-1")
    s = base.with_(phi_deg=24.0, k_phi=60_000.0, n0=1.2, n1=1.4)
    src = dict(s.sources)
    for k in ("phi_deg", "k_phi", "n0", "n1"):
        src[k] = SOFT_SAND_NOTE
    from dataclasses import replace
    return replace(s, name="soft sand (synthetic)", description="hidden truth of the demo site", sources=src)


def demo_site() -> SyntheticSite:
    """A 40 m x 24 m site of compact regolith (GRC-1) cut by a north-south ridge (3.6 m high, flanks up to about
    20 deg). The ridge's pass (about 14 deg) is filled with soft sand that the orbital map labels as compact regolith;
    its northern part is firm gravelly ground (JSC-1A). A second soft-sand patch lies on level ground."""
    grc, jsc, sand = get_soil("GRC-1"), get_soil("JSC-1A"), soft_sand()
    units = [
        SoilUnit("compact regolith", grc, grc, "orbital class: smooth bright regolith -> GRC-1", (0.62, 0.60, 0.57)),
        SoilUnit("firm gravelly ground", jsc, jsc, "orbital class: rough, blocky -> JSC-1A (firm)", (0.45, 0.43, 0.42)),
        SoilUnit("smooth mottled unit", sand, grc,
                 "orbital class: smooth, slightly mottled -> assumed compact regolith (GRC-1); truth: soft sand",
                 (0.80, 0.68, 0.48)),
    ]
    patches = [
        Patch(1, 20.0, 20.5, 9.0, 3.5, 0.0),           # firm gravelly ground over the north part of the ridge
        Patch(1, 31.0, 4.0, 6.0, 2.5, -20.0),
        Patch(2, 18.5, 10.0, 7.5, 3.6, 0.0),           # soft sand filling the pass through the ridge
        Patch(2, 9.0, 17.0, 3.0, 2.0, -10.0),          # second soft patch, level ground to the north-west
    ]
    mounds = [Mound(20.0, 12.0, 6.0, 60.0, 3.6),        # ridge across the site (north-south)
              Mound(20.0, 10.0, 6.0, 3.0, -0.9),        # its pass
              Mound(8.0, 3.0, 4.0, 3.0, 0.5), Mound(33.0, 20.0, 4.0, 3.5, 0.6)]
    return SyntheticSite((0.0, 40.0), (0.0, 24.0), units, patches, mounds, tilt_deg=1.0, tilt_dir_deg=90.0,
                         start=(3.0, 6.0), goal=(37.0, 15.0),
                         description="synthetic site: a ridge whose pass is soft sand mapped from orbit as "
                                     "compact regolith, firm gravelly ground on its northern part")


__all__ = ["Mound", "Patch", "SoilUnit", "SyntheticSite", "demo_site", "soft_sand", "SOFT_SAND_NOTE"]
