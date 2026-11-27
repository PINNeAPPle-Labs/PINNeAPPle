"""A launch vehicle as the ascent model needs it: stages with their masses, their vacuum and sea-level thrust and
specific impulse, their drag area and a Mach-dependent drag coefficient.

A stage is jettisoned when its propellant is spent; what remains is the mass of the stages above it, the payload and
the fairing (dropped at ``fairing_jettison_altitude_km``). Thrust and mass flow follow the ambient pressure:

    T(p) = T_vac - (T_vac - T_sl) p / p_sl          m_dot = T / (Isp g0)
    Isp(p) = T(p) / (m_dot g0)

so a stage that leaves the ground with a sea-level specific impulse finishes its burn with the vacuum one, and the
ideal velocity of the stage is the Tsiolkovsky one at the vacuum impulse for the whole burn (it is an upper bound;
the ascent integrates the real thing).

    from pinneapple_simulation.launch import vehicle
    veh = vehicle.small_launcher()          # the concept vehicle of ``examples/rocket_launch``
    veh.liftoff_mass_kg, veh.delta_v_vac_m_s, veh.stage_delta_v_m_s

Nothing here is a published vehicle: every number is an assumption of the concept study, listed in
``examples/rocket_launch/README.md`` with what it stands on. The published cross-checks in
:mod:`.verification` use closed forms, not a real launcher's flight data.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["G0", "CD_MACH", "Stage", "LaunchVehicle", "cd_of_mach", "small_launcher"]

G0 = 9.80665
P_SL = 101325.0

#: Drag coefficient against Mach number of a slender launcher (assumption: fineness ratio ~ 15, no boat-tail);
#: the transonic rise and the supersonic decay are the shape every launcher shows (e.g. Akin 2005, "Flight
#: vehicle drag", a standard textbook curve; these are the concept's own numbers, not a measured table).
CD_MACH = (
    (0.0, 0.20), (0.3, 0.20), (0.6, 0.22), (0.8, 0.26), (0.9, 0.32), (1.0, 0.44), (1.1, 0.48), (1.2, 0.47),
    (1.5, 0.42), (2.0, 0.36), (3.0, 0.30), (4.0, 0.27), (6.0, 0.23), (8.0, 0.21), (12.0, 0.20),
)


_CD_X = [t[0] for t in CD_MACH]
_CD_Y = [t[1] for t in CD_MACH]


def cd_of_mach(mach, table=CD_MACH):
    """Drag coefficient at ``mach`` (linear interpolation in the table, constant beyond its ends).

    A scalar Mach number takes a plain-Python path: the ascent evaluates it at every integrator substep."""
    if np.ndim(mach) == 0:
        m = abs(float(mach))
        xs, ys = (_CD_X, _CD_Y) if table is CD_MACH else ([t[0] for t in table], [t[1] for t in table])
        if m <= xs[0]:
            return ys[0]
        if m >= xs[-1]:
            return ys[-1]
        k = bisect.bisect_right(xs, m) - 1
        return ys[k] + (ys[k + 1] - ys[k]) * (m - xs[k]) / (xs[k + 1] - xs[k])
    m = np.abs(np.asarray(mach, float))
    xs = np.array([t[0] for t in table])
    ys = np.array([t[1] for t in table])
    return np.interp(m, xs, ys)


@dataclass(frozen=True)
class Stage:
    """One stage: ``dry`` structure jettisoned at separation, ``propellant`` burned at ``isp_vac``, thrust at vacuum
    and at sea level, and the reference area its diameter gives for drag."""
    name: str
    dry_mass_kg: float
    propellant_kg: float
    isp_vac_s: float
    thrust_vac_n: float
    diameter_m: float
    isp_sl_s: float | None = None
    thrust_sl_n: float | None = None
    length_m: float = 0.0

    def __post_init__(self):
        if self.isp_sl_s is None:
            object.__setattr__(self, "isp_sl_s", self.isp_vac_s)
        if self.thrust_sl_n is None:
            # a nozzle of fixed exit area: the thrust lost on the ground is the pressure difference times that area
            object.__setattr__(self, "thrust_sl_n", self.thrust_vac_n * self.isp_sl_s / self.isp_vac_s)

    @property
    def area_m2(self) -> float:
        """Reference area for drag [m^2]."""
        return math.pi * (self.diameter_m / 2.0) ** 2

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_kg

    def thrust_n(self, p_amb_pa: float) -> float:
        """Thrust [N] at ambient pressure ``p_amb_pa`` (linear between sea level and vacuum)."""
        f = min(max(p_amb_pa / P_SL, 0.0), 1.0)
        return self.thrust_vac_n - (self.thrust_vac_n - self.thrust_sl_n) * f

    def mass_flow_kg_s(self, p_amb_pa: float = 0.0) -> float:
        """Propellant mass flow [kg/s]: the engine runs at a fixed flow, so it is the vacuum one at any altitude."""
        return self.mass_flow_vac_kg_s

    @property
    def mass_flow_vac_kg_s(self) -> float:
        return self.thrust_vac_n / (self.isp_vac_s * G0)

    @property
    def burn_time_vac_s(self) -> float:
        """Burn time [s] in vacuum, where the mass flow is the vacuum one."""
        return self.propellant_kg / self.mass_flow_vac_kg_s

    def delta_v_m_s(self, payload_mass_kg: float) -> float:
        """Tsiolkovsky velocity of this stage carrying ``payload_mass_kg`` above it, at the vacuum impulse."""
        m0 = self.wet_mass_kg + payload_mass_kg
        m1 = self.dry_mass_kg + payload_mass_kg
        return self.isp_vac_s * G0 * math.log(m0 / m1)

    def thrust_to_weight(self, mass_above_kg: float) -> float:
        """Lift-off thrust-to-weight with ``mass_above_kg`` (the stages, payload and fairing above this one)."""
        return self.thrust_sl_n / ((self.wet_mass_kg + mass_above_kg) * G0)


@dataclass
class LaunchVehicle:
    """Stages (bottom first), payload and fairing, and what they add up to."""
    stages: tuple
    payload_kg: float
    fairing_kg: float = 0.0
    fairing_jettison_altitude_km: float = 110.0
    name: str = "concept launcher"
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.stages = tuple(self.stages)

    # ------------------------------------------------------------------ masses
    @property
    def stage_masses_above(self):
        """Mass carried above each stage at lift-off [kg]: later stages, payload and fairing."""
        out = []
        above = self.payload_kg + self.fairing_kg
        for st in reversed(self.stages):
            out.append(above)
            above += st.wet_mass_kg
        return list(reversed(out))

    @property
    def liftoff_mass_kg(self) -> float:
        return sum(s.wet_mass_kg for s in self.stages) + self.payload_kg + self.fairing_kg

    @property
    def propellant_kg(self) -> float:
        return sum(s.propellant_kg for s in self.stages)

    @property
    def payload_fraction(self) -> float:
        return self.payload_kg / self.liftoff_mass_kg

    @property
    def stage_delta_v_m_s(self) -> list:
        return [st.delta_v_m_s(above) for st, above in zip(self.stages, self.stage_masses_above, strict=True)]

    @property
    def delta_v_vac_m_s(self) -> float:
        """Sum of the stage Tsiolkovsky velocities (an upper bound on the inertial velocity the vehicle can reach)."""
        return float(sum(self.stage_delta_v_m_s))

    @property
    def thrust_to_weight_liftoff(self) -> float:
        return self.stages[0].thrust_sl_n / (self.liftoff_mass_kg * G0)

    def drag_area_m2(self, stage_index: int) -> float:
        return self.stages[stage_index].area_m2

    def cd(self, mach) -> float:
        """Drag coefficient of the vehicle at ``mach`` (the slender-launcher curve of :data:`CD_MACH`)."""
        return cd_of_mach(mach)

    def with_propellant_scale(self, factor: float) -> LaunchVehicle:
        """A copy with every propellant tank scaled by ``factor`` (structure unchanged): the design-closure handle."""
        return replace(self, stages=tuple(replace(s, propellant_kg=s.propellant_kg * factor) for s in self.stages))

    def summary(self) -> dict:
        """The numbers a data sheet or a report line needs."""
        return {
            "name": self.name,
            "liftoff_mass_kg": self.liftoff_mass_kg,
            "propellant_kg": self.propellant_kg,
            "payload_kg": self.payload_kg,
            "payload_fraction": self.payload_fraction,
            "delta_v_vac_m_s": self.delta_v_vac_m_s,
            "stage_delta_v_m_s": self.stage_delta_v_m_s,
            "thrust_to_weight_liftoff": self.thrust_to_weight_liftoff,
            "stages": [{"name": s.name, "dry_mass_kg": s.dry_mass_kg, "propellant_kg": s.propellant_kg,
                        "isp_vac_s": s.isp_vac_s, "isp_sl_s": s.isp_sl_s, "thrust_vac_n": s.thrust_vac_n,
                        "thrust_sl_n": s.thrust_sl_n, "diameter_m": s.diameter_m, "length_m": s.length_m,
                        "area_m2": s.area_m2} for s in self.stages],
        }


#: the velocity split the design closure converges to (``pinneapple_simulation.launch.sizing.close``): the first stage
#: leaves the dense air, the second builds the transfer to a 500 km apogee, the kick stage circularises there
DELTA_V_SPLIT = (3513.0, 5270.0, 950.0)


def small_launcher(payload_kg: float = 200.0) -> LaunchVehicle:
    """The concept vehicle of the study: three stages, 200 kg to a 500 km near-circular orbit from Tanegashima.

    The stage masses come from :func:`.sizing.size_vehicle` with the velocity split this ascent model measures
    (``DELTA_V_SPLIT``, the converged result of ``examples/rocket_launch/02_mission.py``); the structural coefficients,
    impulses and thrust-to-weight are assumptions of the concept and are listed in the example's README.
    """
    from .sizing import size_vehicle

    return size_vehicle(payload_kg, delta_v=DELTA_V_SPLIT, tw_liftoff=1.45,
                        name="Concept-200 (Tanegashima -> 500 km)")
