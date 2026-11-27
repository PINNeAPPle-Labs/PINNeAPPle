"""The rotor of a multirotor, from blade element momentum theory: thrust, torque and the power they cost.

A drone's dynamics change in flight because its rotors do -- a chipped blade, a degraded motor, a battery that sags.
Fitting a thrust curve hides that; solving the blades shows it, because every one of those faults walks into the same
three numbers a flight controller cannot measure directly.

The model is steady BEMT in axial flow, per annulus, with Prandtl tip loss and the inflow solved by bisection:

    phi = atan((Vc + v_i) / (Omega r))        alpha = theta(r) - phi
    dT/dr = 1/2 rho W^2 B c (Cl cos phi - Cd sin phi)      blade element
    dT/dr = 4 pi rho r F v_i (Vc + v_i)                    momentum
    dQ/dr = 1/2 rho W^2 B c r (Cl sin phi + Cd cos phi)

with ``Cl = cl_alpha alpha`` (clamped at ``cl_max``) and ``Cd = cd0 + k Cl^2``, and the geometric pitch giving
``theta(r) = atan(pitch / 2 pi r)``.

What it is checked against, in :mod:`tests.test_multirotor`:

* the figure of merit ``P_ideal / P`` -- the ideal being momentum theory's ``T^{3/2} / sqrt(2 rho A)`` -- lands in the
  published ``0.5-0.8`` band for a propeller of this size, which is what says the blade element and the momentum side
  agree instead of one of them being decoration;
* ``T ~ Omega^2`` to within 2 % over 2 000-8 000 rpm, the propeller law every bench measurement obeys;
* a chipped tip makes less thrust **at better efficiency** -- the tip is the part Prandtl's loss penalises most, so as
  the span goes the figure of merit rises, 0.650 to 0.787 between an intact blade and one missing 40 % of it. That is
  the *opposite* of what "a damaged prop works harder" suggests, which is why the ratio is measured here and not
  assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Rotor", "RotorState", "PROPS", "figure_of_merit"]

#: Published propellers, with the geometry a bench measurement would give: radius and pitch in metres, chord as the
#: mean of the blade. ``cl_alpha``, ``cd0`` and ``k`` are declared airfoil polars, not measured ones.
PROPS = {
    # APC 10x4.7: 5 in radius, 4.7 in pitch, thin Eppler section at Re ~ 1e5
    "apc-10x4.7": dict(radius_m=0.127, pitch_m=0.1194, chord_m=0.0220, blades=2, cd0=0.022, k=0.018),
    # APC 9x4.5, the prop of a 250-class quad
    "apc-9x4.5": dict(radius_m=0.1143, pitch_m=0.1143, chord_m=0.0195, blades=2, cd0=0.024, k=0.018),
    # the small, high-pitch prop of a racing quad
    "apc-5x4.6": dict(radius_m=0.0635, pitch_m=0.1168, chord_m=0.0125, blades=3, cd0=0.028, k=0.020),
}


@dataclass(frozen=True)
class RotorState:
    """What a rotor is producing at one instant."""

    omega_rad_s: float
    thrust_n: float
    torque_nm: float
    power_w: float
    figure_of_merit: float
    inflow_m_s: float

    def to_dict(self) -> dict[str, float]:
        return {"omega_rad_s": self.omega_rad_s, "thrust_n": self.thrust_n, "torque_nm": self.torque_nm,
                "power_w": self.power_w, "figure_of_merit": self.figure_of_merit, "inflow_m_s": self.inflow_m_s}


def figure_of_merit(thrust_n: float, power_w: float, radius_m: float, rho: float = 1.225) -> float:
    """``P_ideal / P`` for the thrust produced: 1.0 is an actuator disk with no losses, a real propeller is 0.5-0.8."""
    if thrust_n <= 0.0 or power_w <= 0.0:
        return 0.0
    area = math.pi * radius_m ** 2
    return float(thrust_n ** 1.5 / math.sqrt(2.0 * rho * area) / power_w)


@dataclass(frozen=True)
class Rotor:
    """One propeller: its geometry, its airfoil, and the BEMT that turns a rotation rate into thrust and torque.

    ``chord_m`` is the blade chord at 75 % radius and ``taper`` the ratio of the tip chord to it, so a linear taper is
    the blade: what a chipped tip removes is exactly the outer stations, which is the point.
    """

    radius_m: float = 0.127
    pitch_m: float = 0.1194
    chord_m: float = 0.0175
    blades: int = 2
    cl_alpha: float = 6.283
    cd0: float = 0.022
    k: float = 0.018
    cl_max: float = 1.4
    taper: float = 0.55
    hub_fraction: float = 0.15
    stations: int = 60
    rho: float = 1.225
    #: the fraction of the rotor a fault has removed, from a chip to a blade lost
    tip_loss: float = 0.0

    #: the efficiency a motor + ESC + battery chain adds on top of the aerodynamic power (declared, not measured)
    drivetrain_efficiency: float = 0.75

    @classmethod
    def from_prop(cls, name: str, **overrides) -> Rotor:
        if name not in PROPS:
            raise ValueError(f"unknown prop {name!r}; this library knows {sorted(PROPS)}")
        return cls(**{**PROPS[name], **overrides})

    @property
    def area_m2(self) -> float:
        return math.pi * (self.radius_m * (1.0 - self.tip_loss)) ** 2

    def chord_at(self, r: float) -> float:
        """The chord at radius ``r``: linear from the hub to the tip, ``taper`` times ``chord_m`` at the tip."""
        s = (r / self.radius_m - self.hub_fraction) / (1.0 - self.hub_fraction)
        return float(self.chord_m * (1.0 + (self.taper - 1.0) * min(max(s, 0.0), 1.0)))

    def _blade_element(self, omega: float, r: float, v_i: float, climb: float):
        """``(dT/dr, dQ/dr)`` of the annulus at ``r``, from the blade element alone."""
        ut = omega * r
        up = climb + v_i
        w2 = ut * ut + up * up
        phi = math.atan2(up, ut)
        alpha = math.atan2(self.pitch_m, 2.0 * math.pi * r) - phi
        cl = max(min(self.cl_alpha * alpha, self.cl_max), -self.cl_max)
        cd = self.cd0 + self.k * cl * cl
        c = self.chord_at(r)
        common = 0.5 * self.rho * w2 * self.blades * c
        dt = common * (cl * math.cos(phi) - cd * math.sin(phi))
        dq = common * r * (cl * math.sin(phi) + cd * math.cos(phi))
        return dt, dq, phi

    def _momentum(self, r: float, v_i: float, climb: float, phi: float) -> float:
        """``dT/dr`` the momentum theory allows for that inflow, with Prandtl's tip loss."""
        if v_i <= 0.0:
            return 0.0
        sin_phi = math.sin(phi) if phi > 1e-6 else 1e-6
        f = 0.5 * self.blades * (self.radius_m - r) / (r * sin_phi)
        loss = 1.0 if f > 40.0 else (2.0 / math.pi) * math.acos(min(math.exp(-f), 1.0))
        return 4.0 * math.pi * self.rho * r * loss * v_i * (climb + v_i)

    def inflow(self, omega: float, r: float, climb: float = 0.0) -> float:
        """The induced velocity at ``r``: where the blade element and the momentum annulus agree, by bisection."""
        if omega <= 0.0:
            return 0.0
        lo, hi = 1e-6, 60.0
        f_lo = self._blade_element(omega, r, lo, climb)[0] - self._momentum(r, lo, climb, 1e-3)
        if f_lo <= 0.0:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            phi = math.atan2(climb + mid, omega * r)
            f_mid = self._blade_element(omega, r, mid, climb)[0] - self._momentum(r, mid, climb, phi)
            if f_mid > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        return 0.5 * (lo + hi)

    def state(self, omega: float, climb: float = 0.0) -> RotorState:
        """Thrust, torque, power and the figure of merit at ``omega``. Negative speed is refused, not mirrored."""
        if omega < 0.0:
            raise ValueError("a rotor does not turn backwards: pass a non-negative angular velocity")
        tip = self.radius_m * (1.0 - self.tip_loss)
        if omega == 0.0 or tip <= 0.0:
            return RotorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rs = np.linspace(self.hub_fraction * tip, 0.995 * tip, self.stations)
        dr = rs[1] - rs[0]
        thrust = torque = inflow = 0.0
        for r in rs:
            v_i = self.inflow(omega, float(r), climb)
            dt, dq, _ = self._blade_element(omega, float(r), v_i, climb)
            thrust += dt * dr
            torque += dq * dr
            inflow += v_i * dr
        torque = max(torque, 0.0)
        power = torque * omega
        return RotorState(omega, max(thrust, 0.0), torque, power,
                          figure_of_merit(max(thrust, 0.0), power, tip, self.rho), inflow / (tip - self.hub_fraction * tip))

    def electrical_power(self, omega: float, climb: float = 0.0) -> float:
        """What the battery pays for that rotation, after the drivetrain's declared efficiency."""
        return self.state(omega, climb).power_w / self.drivetrain_efficiency

    def omega_for_thrust(self, thrust_n: float, climb: float = 0.0, *, brackets=(1.0, 2000.0)) -> float:
        """The rotation rate that produces ``thrust_n`` -- what a flight controller's mixer solves for, slowly."""
        if thrust_n <= 0.0:
            return 0.0
        lo, hi = brackets
        if self.state(hi, climb).thrust_n < thrust_n:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.state(mid, climb).thrust_n < thrust_n:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-2:
                break
        return 0.5 * (lo + hi)
