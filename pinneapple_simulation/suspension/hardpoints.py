"""The right-front corner of a double wishbone suspension as hardpoints, and the design vector that moves them.

Vehicle axes, ISO 8855: ``x`` forward, ``y`` to the right (outboard on this corner), ``z`` up, the origin on the
ground plane under the wheel centre's nominal fore-aft position. Everything is in SI: metres and radians inside, mm
and degrees only in the reported metrics.

A corner is eleven points. The lower and upper wishbones each swing about the axis through their two chassis pivots;
the ball joints, the outer tie-rod end and the wheel centre are points of one rigid upright. That is the whole
mechanism: given a wheel travel, the pose of the upright is the solution of five scalar constraints
(:mod:`pinneapple_simulation.suspension.kinematics`), and every alignment number -- camber, toe, roll centre,
scrub radius, motion ratio -- is a function of that pose.

The design vector is eight numbers (:data:`PARAMETERS`): four move hardpoints (which changes the elastokinematics),
two are structural sizes of the lower wishbone and one of the upper (which change mass, stress and compliance). The
values marked *baseline* are a small front-wheel-drive hatchback's, not a race car's.

References for the geometry conventions and the metric definitions: Gillespie, *Fundamentals of Vehicle Dynamics*
(SAE, 1992), ch. 7; Milliken & Milliken, *Race Car Vehicle Dynamics* (SAE, 1995), ch. 16-17; Reimpell & Stoll,
*The Automotive Chassis* (Butterworth-Heinemann, 1996), ch. 2.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Parameter", "PARAMETERS", "BOUNDS", "BASELINE", "Corner", "corner", "invalid"]


@dataclass(frozen=True)
class Parameter:
    """One entry of the design vector: what it moves, in the units the engineer uses."""
    name: str
    low: float
    high: float
    unit: str
    description: str

    @property
    def mid(self) -> float:
        return 0.5 * (self.low + self.high)


#: The eight design variables. Bounds are the ranges a designer would actually consider for this corner: wide enough
#: that the surrogate sees a real family of cars, narrow enough that every point is a buildable suspension.
PARAMETERS: tuple[Parameter, ...] = (
    Parameter("lbj_y", 0.680, 0.740, "m",
              "lateral position of the lower ball joint: the lower wishbone's length (wheel centre is at y = 0.730)"),
    Parameter("ubj_y", 0.630, 0.700, "m",
              "lateral position of the upper ball joint: with lbj_y it sets the kingpin inclination and the camber gain"),
    Parameter("ubj_z", 0.420, 0.480, "m",
              "height of the upper ball joint: the upper wishbone's inclination, the other half of the camber gain"),
    Parameter("lca_z", 0.100, 0.160, "m",
              "height of the lower wishbone's inner pivots: the arm's inclination, which sets the roll centre"),
    Parameter("tie_inner_z", 0.150, 0.230, "m",
              "height of the inner tie-rod end on the rack: bump steer is the mismatch between this arc and the arms'"),
    Parameter("lc_od", 0.020, 0.034, "m", "outer diameter of the lower wishbone's tubular legs"),
    Parameter("lc_wall", 0.0025, 0.0050, "m", "wall thickness of the lower wishbone's tubular legs"),
    Parameter("uc_od", 0.016, 0.028, "m", "outer diameter of the upper wishbone's tubular legs (wall fixed at 3 mm)"),
)

BOUNDS: dict[str, tuple[float, float]] = {p.name: (p.low, p.high) for p in PARAMETERS}

#: A representative design: every value at the middle of its range.
BASELINE: dict[str, float] = {p.name: p.mid for p in PARAMETERS}

WHEEL_RADIUS = 0.310            # m, 205/55 R16
WHEEL_CENTRE_Y = 0.730          # m, half track 1.460 m
SPRING_RATE = 100.0             # N/mm, at the damper (the wheel rate follows from the motion ratio)
YIELD_STRENGTH = 620e6          # Pa, quenched-and-tempered forged steel (34CrMo4, Re ~ 620 MPa)
DENSITY = 7850.0                # kg/m^3
UPPER_WALL = 0.003              # m, the upper arm's wall is not a design variable


@dataclass(frozen=True)
class Corner:
    """The eleven hardpoints of the corner plus what the structural model needs."""
    lca_front: np.ndarray
    lca_rear: np.ndarray
    lbj: np.ndarray
    uca_front: np.ndarray
    uca_rear: np.ndarray
    ubj: np.ndarray
    tie_inner: np.ndarray
    tie_outer: np.ndarray
    wheel_centre: np.ndarray
    damper_lower: np.ndarray
    damper_upper: np.ndarray
    lc_od: float
    lc_wall: float
    uc_od: float
    wheel_radius: float = WHEEL_RADIUS
    spring_rate: float = SPRING_RATE
    design: Mapping[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ axes
    @property
    def lca_axis(self) -> tuple[np.ndarray, np.ndarray]:
        """(point, unit direction) of the lower wishbone's chassis axis."""
        d = self.lca_rear - self.lca_front
        return self.lca_front, d / np.linalg.norm(d)

    @property
    def uca_axis(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.uca_rear - self.uca_front
        return self.uca_front, d / np.linalg.norm(d)

    def points(self) -> dict[str, np.ndarray]:
        """The five points of the rigid upright (they move together), for the kinematics."""
        return {"lbj": self.lbj, "ubj": self.ubj, "tie_outer": self.tie_outer, "wheel_centre": self.wheel_centre,
                "damper_lower": self.damper_lower}


def corner(design: Mapping[str, float]) -> Corner:
    """The hardpoints of one design. Values outside :data:`BOUNDS` are not clipped -- :func:`invalid` says so."""
    d = {p.name: float(design[p.name]) for p in PARAMETERS}
    z = d["lca_z"]
    return Corner(
        lca_front=np.array([0.130, 0.300, z]),
        lca_rear=np.array([-0.170, 0.315, z + 0.005]),
        lbj=np.array([0.010, d["lbj_y"], 0.130]),
        uca_front=np.array([0.100, 0.350, 0.400]),
        uca_rear=np.array([-0.100, 0.360, 0.400]),
        ubj=np.array([-0.020, d["ubj_y"], d["ubj_z"]]),
        tie_inner=np.array([-0.155, 0.320, d["tie_inner_z"]]),
        tie_outer=np.array([-0.150, 0.665, 0.200]),
        wheel_centre=np.array([0.000, WHEEL_CENTRE_Y, WHEEL_RADIUS]),
        damper_lower=np.array([0.020, 0.480, z + 0.035]),
        damper_upper=np.array([0.020, 0.330, 0.620]),
        lc_od=d["lc_od"], lc_wall=d["lc_wall"], uc_od=d["uc_od"],
        design=d,
    )


def _perp_distance(p: np.ndarray, q: np.ndarray, a: np.ndarray) -> float:
    v = p - q
    return float(np.linalg.norm(v - (v @ a) * a))


def invalid(design: Mapping[str, float]) -> list[str]:
    """Why this design cannot be built or simulated; empty when it can.

    These are the checks a designer applies on a sketch before spending a solver on it: a wishbone has to be a real
    triangle about its pivot axis (a ball joint on the axis cannot swing), the upright has to be a triangle too, the
    tie rod has to reach, and the tube has to have a wall thin enough to mesh and thick enough to be a tube.
    """
    out: list[str] = []
    for p in PARAMETERS:
        v = float(design.get(p.name, float("nan")))
        if not math.isfinite(v) or not (p.low - 1e-12 <= v <= p.high + 1e-12):
            out.append(f"{p.name} = {v:.4g} {p.unit} is outside its range [{p.low:.4g}, {p.high:.4g}]")
    if out:
        return out
    c = corner(design)
    for name, ball, axis in (("lower", c.lbj, c.lca_axis), ("upper", c.ubj, c.uca_axis)):
        rho = _perp_distance(ball, *axis)
        if rho < 0.10:
            out.append(f"{name} wishbone is {rho * 1e3:.0f} mm from its pivot axis: too short to swing")
    lt = float(np.linalg.norm(c.tie_outer - c.tie_inner))
    if not 0.15 <= lt <= 0.55:
        out.append(f"tie rod length {lt * 1e3:.0f} mm is outside a rack's reach [150, 550] mm")
    if np.linalg.norm(c.ubj - c.lbj) < 0.20:
        out.append("kingpin length below 200 mm: the upright cannot be a triangle")
    if c.uc_od < 2.2 * UPPER_WALL:
        out.append(f"upper leg diameter {c.uc_od * 1e3:.0f} mm is smaller than twice its 3 mm wall")
    if c.lc_wall > 0.25 * c.lc_od:
        out.append(f"lower leg wall {c.lc_wall * 1e3:.1f} mm is more than a quarter of the {c.lc_od * 1e3:.0f} mm "
                   f"diameter: not a tube")
    if c.lc_wall < 0.002:
        out.append(f"lower leg wall {c.lc_wall * 1e3:.1f} mm is below the 2 mm a wishbone is forged to")
    return out


def with_values(design: Mapping[str, float], **changes: float) -> dict[str, float]:
    """A copy of ``design`` with some values replaced (a design vector is a plain dict, on purpose: it is a CSV row,
    a JSON object and a column of the dataset without conversion)."""
    return {**{p.name: float(design[p.name]) for p in PARAMETERS}, **{k: float(v) for k, v in changes.items()}}
