"""Suspension of a car as a design problem: the double-wishbone hardpoints, the wheels-in-bump loop that moves them, the
arm loads, the structural sizing and the elastokinematic compliance — all from one description.

A suspension corner is where the three models of the vehicle meet, and they cannot be designed one after the other:

* the **CAD** places the hardpoints (six per corner: four inner pivots, two ball joints, the steering-arm point) and
  turns a tube section into an arm whose mass is the objective;
* the **multibody** moves the linkage through wheel travel and reads the wheel plane: camber, toe, caster, scrub,
  half-track, motion ratio and the roll centre — the *kinematic* objectives;
* the **structural** model takes the wheel loads the linkage delivers and sizes the arm for stress and buckling — and
  the section it needs changes the mass, which is the objective the other two constrain.

And the loop does not close there: the bushings that hold the arms are compliant, so a lateral force steers the wheel
(*elastokinematics*). A geometry that is perfect on paper can have 0.3 deg of toe compliance; a stiffer bushing fixes
it and costs ride comfort. :func:`pinneapple_design.suspension.design.suspension_design` runs that loop and returns
the lightest arm whose kinematics, structure and compliance all pass, and
:func:`pinneapple_design.suspension.study.suspension_study` runs it as one Studio tool: the 3-D field of the arm,
the compliance curves, the checks and the numbers the deck shows.

Conventions: SI in the library (metres, newtons, pascals, radians in the solvers), degrees and millimetres in what is
reported. Frame: ``x`` forward, ``y`` left, ``z`` up, origin on the ground under the centre of the modelled corner's
tyre contact patch at the reference pose. One corner is modelled — the **left** one — and the right one is its mirror,
which is what a symmetric axle allows.

The vehicle in :func:`front_corner` is a **design study**, a double-wishbone front corner with the proportions of a
compact rear-drive coupe (tyre 235/40 R18, track 1.58 m). The numbers are chosen to be plausible and are stated, not
measured from a particular car; every one of them is a parameter of :class:`SuspensionSpec`.

    from pinneapple_design.suspension import front_corner, kinematic_sweep
    spec = front_corner()
    sweep = kinematic_sweep(spec)                     # wheels in bump and rebound
    sweep.summary()["camber_gain_deg_per_50mm"]
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["Bushing", "SuspensionSpec", "front_corner", "GRAVITY"]

GRAVITY = 9.80665


@dataclass(frozen=True)
class Bushing:
    """A cylindrical bushing at an inner pivot: radial stiffness in the plane normal to the pivot axis and axial
    stiffness along it [N/m]."""
    radial_n_per_m: float
    axial_n_per_m: float
    name: str = ""

    @property
    def radial_n_per_mm(self) -> float:
        return self.radial_n_per_m / 1e3

    @property
    def axial_n_per_mm(self) -> float:
        return self.axial_n_per_m / 1e3


def _v(x, y, z) -> np.ndarray:
    return np.array([x, y, z], float)


@dataclass(frozen=True)
class SuspensionSpec:
    """One double-wishbone corner: hardpoints [m], bushings, spring, tyre and the elastic constants of the arm.

    ``lca_front``/``lca_rear``/``uca_front``/``uca_rear`` are the inner pivots on the chassis (each bushed),
    ``lca_outer``/``uca_outer`` the ball joints, ``tierod_inner`` the rack end and ``tierod_outer`` the point on the
    upright's steering arm. ``wheel_centre`` and ``wheel_axis`` are the wheel's at the reference pose;
    ``spring_lower``/``spring_upper`` the damper's attachments (on the lower arm and on the chassis).
    """
    name: str
    # hardpoints [m]
    lca_front: np.ndarray
    lca_rear: np.ndarray
    lca_outer: np.ndarray
    uca_front: np.ndarray
    uca_rear: np.ndarray
    uca_outer: np.ndarray
    tierod_inner: np.ndarray
    tierod_outer: np.ndarray
    wheel_centre: np.ndarray
    wheel_axis: np.ndarray
    spring_lower: np.ndarray
    spring_upper: np.ndarray
    # compliance and rates
    bushing_lca: Bushing
    bushing_uca: Bushing
    bushing_tierod: Bushing
    spring_rate_n_per_m: float
    tyre_rate_n_per_m: float
    tyre_radius_m: float
    # structure
    tube_od_m: float
    tube_wall_m: float
    apex_offset_m: float                 # ball-joint normal offset from the tube axis at the apex (bending arm)
    density_kg_m3: float
    yield_pa: float
    elastic_modulus_pa: float
    safety_factor: float
    # vehicle
    corner_mass_kg: float                # sprung mass carried by the corner
    half_track_m: float
    arm_length_scale: float = 1.0        # >1 lengthens both legs (a wider track), <1 shortens them
    sources: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in ("lca_front", "lca_rear", "lca_outer", "uca_front", "uca_rear", "uca_outer", "tierod_inner",
                  "tierod_outer", "wheel_centre", "wheel_axis", "spring_lower", "spring_upper"):
            object.__setattr__(self, k, np.asarray(getattr(self, k), float))
        a = self.wheel_axis / np.linalg.norm(self.wheel_axis)
        object.__setattr__(self, "wheel_axis", a)

    # -- derived lengths ---------------------------------------------------------------------------------------
    @property
    def lca_axis(self) -> np.ndarray:
        return _unit(self.lca_rear - self.lca_front)

    @property
    def uca_axis(self) -> np.ndarray:
        return _unit(self.uca_rear - self.uca_front)

    @property
    def lca_inner_mid(self) -> np.ndarray:
        return 0.5 * (self.lca_front + self.lca_rear)

    @property
    def uca_inner_mid(self) -> np.ndarray:
        return 0.5 * (self.uca_front + self.uca_rear)

    @property
    def lca_length_m(self) -> float:
        return float(np.linalg.norm(self.lca_outer - self.lca_inner_mid))

    @property
    def uca_length_m(self) -> float:
        return float(np.linalg.norm(self.uca_outer - self.uca_inner_mid))

    @property
    def kingpin_length_m(self) -> float:
        return float(np.linalg.norm(self.uca_outer - self.lca_outer))

    @property
    def tierod_length_m(self) -> float:
        return float(np.linalg.norm(self.tierod_outer - self.tierod_inner))

    @property
    def spring_length_m(self) -> float:
        return float(np.linalg.norm(self.spring_upper - self.spring_lower))

    @property
    def motion_ratio_reference(self) -> float:
        """``|d(spring length) / d(wheel travel)|`` at the reference pose (the damper's lever ratio)."""
        from .kinematics import pose

        h = 1e-5
        up = pose(self, h)
        down = pose(self, -h)
        return float(abs(up.spring_length_m - down.spring_length_m) / (2 * h))

    def mirrored(self) -> SuspensionSpec:
        """The same corner on the other side of the car (``y -> -y``): for whole-axle checks."""
        flip = np.array([1.0, -1.0, 1.0])
        pts = {k: getattr(self, k) * flip for k in
               ("lca_front", "lca_rear", "lca_outer", "uca_front", "uca_rear", "uca_outer", "tierod_inner",
                "tierod_outer", "wheel_centre", "wheel_axis", "spring_lower", "spring_upper")}
        return replace(self, name=self.name + " (mirrored)", **pts)

    def scaled_arms(self, scale: float) -> SuspensionSpec:
        """Both legs of both A-arms lengthened by ``scale`` about the inner pivots (a wider track), wheel and
        steering arm following the upright: the CAD half of the design loop."""
        def grow(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
            return inner + (outer - inner) * scale

        l_mid, u_mid = self.lca_inner_mid, self.uca_inner_mid
        l_o, u_o = grow(l_mid, self.lca_outer), grow(u_mid, self.uca_outer)
        shift = 0.5 * ((l_o - self.lca_outer) + (u_o - self.uca_outer))       # the upright moves with the arms
        return replace(self, lca_outer=l_o, uca_outer=u_o, tierod_outer=self.tierod_outer + shift,
                       wheel_centre=self.wheel_centre + shift, spring_lower=self.spring_lower + 0.5 * (l_o - self.lca_outer),
                       half_track_m=self.half_track_m + shift[1], arm_length_scale=scale)

    def with_tube(self, od_m: float, wall_m: float) -> SuspensionSpec:
        return replace(self, tube_od_m=od_m, tube_wall_m=wall_m)

    def with_bushings(self, radial_n_per_m: float) -> SuspensionSpec:
        """All three bushings scaled from the lower arm's radial rate, keeping the reference ratios."""
        r = self.bushing_lca.radial_n_per_m
        f = radial_n_per_m / r
        return replace(self, bushing_lca=replace(self.bushing_lca, radial_n_per_m=self.bushing_lca.radial_n_per_m * f,
                                                 axial_n_per_m=self.bushing_lca.axial_n_per_m * f),
                       bushing_uca=replace(self.bushing_uca, radial_n_per_m=self.bushing_uca.radial_n_per_m * f,
                                           axial_n_per_m=self.bushing_uca.axial_n_per_m * f),
                       bushing_tierod=replace(self.bushing_tierod,
                                              radial_n_per_m=self.bushing_tierod.radial_n_per_m * f,
                                              axial_n_per_m=self.bushing_tierod.axial_n_per_m * f))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def front_corner() -> SuspensionSpec:
    """A double-wishbone front corner with the proportions of a compact rear-drive coupe: tyre 235/40 R18,
    track 1.58 m, corner sprung mass 350 kg. A design study, not a particular car — the limits of the model are
    listed in :mod:`pinneapple_design.suspension.study`."""
    return SuspensionSpec(
        name="front corner, double wishbone (design study)",
        lca_front=_v(0.145, 0.440, 0.145),
        lca_rear=_v(-0.155, 0.440, 0.150),
        lca_outer=_v(0.005, 0.745, 0.155),
        uca_front=_v(0.115, 0.470, 0.455),
        uca_rear=_v(-0.115, 0.470, 0.465),
        uca_outer=_v(-0.010, 0.705, 0.475),
        tierod_inner=_v(0.130, 0.420, 0.300),
        tierod_outer=_v(0.075, 0.700, 0.310),
        wheel_centre=_v(0.0, 0.790, 0.3226),
        wheel_axis=_v(0.0, np.cos(np.radians(1.0)), np.sin(np.radians(1.0))),   # 1 deg of static camber
        spring_lower=_v(0.000, 0.620, 0.155),
        spring_upper=_v(-0.010, 0.520, 0.680),
        bushing_lca=Bushing(900e3, 1400e3, "lower arm"),
        bushing_uca=Bushing(700e3, 1000e3, "upper arm"),
        bushing_tierod=Bushing(500e3, 500e3, "tie rod"),
        spring_rate_n_per_m=65e3,
        tyre_rate_n_per_m=250e3,
        tyre_radius_m=0.3226,
        tube_od_m=0.032,
        tube_wall_m=0.0035,
        apex_offset_m=0.015,
        density_kg_m3=7850.0,
        yield_pa=420e6,
        elastic_modulus_pa=210e9,
        safety_factor=1.5,
        corner_mass_kg=350.0,
        half_track_m=0.790,
        sources={"tyre": "235/40 R18: radius 0.40 x 235 mm + 228.6 mm (nominal section)",
                 "geometry": "design study: the proportions of a compact rear-drive coupe, not a measured car",
                 "bushings": "polyurethane/steel-sleeved rates of the order used on road cars (0.5-1.4 kN/mm)",
                 "camber": "1 deg of static negative camber at the design pose",
                 "tube": "cold-drawn seamless steel tube, 32 x 3.5 mm, yield 420 MPa"},
    )
