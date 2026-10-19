"""The aircraft as a 6-DOF rigid body: mass properties, vortex-lattice aerodynamics, engines and landing gear.

:class:`Aircraft` evaluates the total force and moment about the CG for a state (13-vector of
:mod:`.rigid_body`), the controls and a wind field (:mod:`.turbulence`), and the state derivative. Atmosphere: ISA
(:func:`pinneapple_design.aero.airliner3d.isa`) at the geometric altitude above the runway plus ``field_elevation``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pinneapple_design.aero.airliner3d import isa

from .aero import T_AB, AeroModel
from .gear import GearLeg, gear_forces
from .propulsion import thrust_direction_body
from .rigid_body import G0, IDX_OMEGA, IDX_VEL, quat_to_dcm, rigid_body_derivative

CONTROL_LIMITS = {"elevator": math.radians(25), "aileron": math.radians(20), "rudder": math.radians(25),
                  "flap": math.radians(40), "steer": math.radians(60), "throttle": 1.0, "brake": 1.0}


@dataclass
class Controls:
    """Positive elevator / rudder / aileron / flap = trailing edge down on the right side (elevator down = nose
    down, rudder: trailing edge left = nose left, aileron: right aileron down = roll left), radians; throttle and
    brake 0..1; steer = nose-wheel angle [rad], positive turns right."""
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    flap: float = 0.0
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    gear_down: bool = True

    def clipped(self) -> Controls:
        d = {k: float(np.clip(getattr(self, k), -CONTROL_LIMITS[k] if k not in ("throttle", "brake") else 0.0,
                              CONTROL_LIMITS[k])) for k in CONTROL_LIMITS}
        return replace(self, **d)

    def deflections(self) -> dict[str, float]:
        return {"elevator": self.elevator, "aileron": self.aileron, "rudder": self.rudder, "flap": self.flap}

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in ("elevator", "aileron", "rudder", "flap", "throttle", "steer",
                                                      "brake")}


@dataclass
class Aircraft:
    name: str
    af: Any                                     # planform / geometry (pinneapple_design.aero.Airframe or Airliner)
    aero: AeroModel
    mass: float
    cg: np.ndarray                              # airframe axes
    inertia: np.ndarray                         # body axes about the CG
    engines: Sequence[Any] = ()
    gear: Sequence[GearLeg] = ()
    cabin: dict[str, np.ndarray] = field(default_factory=dict)     # passenger stations (airframe axes)
    tail_point: np.ndarray | None = None     # lowest aft-fuselage point (tail-strike clearance)
    wingtip: np.ndarray | None = None        # right wing tip (left by symmetry), wing-strike clearance
    clmax_takeoff: float = float("nan")
    clmax_clean: float = float("nan")
    flap_takeoff: float = 0.0                   # rad
    field_elevation: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)              # sources, assumptions, component masses
    parts: list[Any] | None = None           # visual geometry (airframe axes)

    def __post_init__(self):
        self.cg = np.asarray(self.cg, float)
        self.inertia = np.asarray(self.inertia, float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.strip_b = self.to_body(self.aero.strip_points)
        self.fin_b = self.to_body(self.aero.fin.ac) if self.aero.fin is not None else np.zeros(3)
        self.sample_b = np.vstack([self.strip_b, self.fin_b, np.zeros(3)])
        self.cabin_b = {k: self.to_body(v) for k, v in self.cabin.items()}
        self.engine_r = [self.to_body(e.position) for e in self.engines]
        self.engine_d = [thrust_direction_body(e.cant_deg) for e in self.engines]
        self.wing_dz = float(self.af.wing_z_root() - self.cg[2])         # wing root above the CG (ground effect)

    @property
    def weight(self) -> float:
        return self.mass * G0

    @property
    def wing_loading(self) -> float:
        return self.weight / self.af.wing_area

    def to_body(self, pA) -> np.ndarray:
        return (np.asarray(pA, float) - self.cg) @ T_AB

    def atmosphere(self, x: np.ndarray):
        return isa(max(self.field_elevation - float(x[2]), 0.0))

    def stall_speed(self, rho: float = 1.225, clmax: float | None = None) -> float:
        c = self.clmax_takeoff if clmax is None else clmax
        return math.sqrt(2 * self.weight / (rho * self.af.wing_area * c))

    # ------------------------------------------------------------------ forces
    def evaluate(self, t: float, x: np.ndarray, u: Controls, wind=None, ground_z: float = 0.0) -> dict[str, Any]:
        u = u.clipped()
        C = quat_to_dcm(x[6:10])
        v, w = x[IDX_VEL], x[IDX_OMEGA]
        rho, a, _ = self.atmosphere(x)
        if wind is not None:
            P = self.sample_points(x, C)
            Wn = wind.velocity(t, P)
            Wb = Wn @ C                                         # rows: NED -> body (C^T w)
            Ws, Wf, Wc = Wb[:-2], Wb[-2], Wb[-1]
        else:
            Wn = Ws = Wf = Wc = None
        h_agl = ground_z - float(x[2])
        ae = self.aero.loads(v, w, rho, Ws, Wf, Wc, u.deflections(), height_agl=h_agl + self.wing_dz, gear_down=u.gear_down)
        F = ae["F"].copy()
        M = ae["M"].copy()
        # engines
        thrust = 0.0
        for e, r, d in zip(self.engines, self.engine_r, self.engine_d, strict=True):
            T = e.thrust(ae["V"], rho, a, u.throttle)
            F += T * d
            M += np.cross(r, T * d)
            thrust += T
        # gear
        Fg = Mg = None
        loads = [0.0] * len(self.gear)
        if self.gear and u.gear_down:
            Fg, Mg, loads = gear_forces(self.gear, x, C, self.cg, u.steer, u.brake, ground_z)
            F += Fg
            M += Mg
        f_spec = F / self.mass                                   # specific force (no gravity), body axes
        Fg_b = C.T @ np.array([0.0, 0.0, self.mass * G0])
        dx = rigid_body_derivative(x, F + Fg_b, M, self.mass, self.inertia, self.inertia_inv)
        return {"dx": dx, "F": F, "M": M, "aero": ae, "thrust": thrust, "gear_loads": loads, "f_spec": f_spec,
                "rho": rho, "a": a, "C": C, "controls": u, "wind_cg_b": Wc, "h_agl": h_agl,
                "wind_samples": Wn}

    def sample_points(self, x: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
        """NED positions of the wind sample points (lattice strips, fin, CG) for state x."""
        C = quat_to_dcm(x[6:10]) if C is None else C
        return x[0:3] + self.sample_b @ C.T

    def specific_force_at(self, ev: dict[str, Any], x: np.ndarray, r_b: np.ndarray) -> np.ndarray:
        """Specific force [m/s^2] in body axes at body point r_b: f_cg + omega_dot x r + omega x (omega x r)."""
        w = x[IDX_OMEGA]
        wd = ev["dx"][IDX_OMEGA]
        return ev["f_spec"] + np.cross(wd, r_b) + np.cross(w, np.cross(w, r_b))

    def derivative(self, t, x, u: Controls, wind=None) -> np.ndarray:
        return self.evaluate(t, x, u, wind)["dx"]
