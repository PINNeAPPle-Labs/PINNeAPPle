"""The faults a multirotor actually suffers, injected where they physically act, and the checks each one moves.

A payload is mass; a degraded motor is less thrust for the same command; a chipped blade is less thrust at *better*
efficiency, because the tip it lost was the part the induced losses punished most; a gust is wind; a GPS glitch is a
lie in the position fix; a
magnetometer interference is a heading that is wrong while everything else is right; vibration is noise the
accelerometer should not have.

Keeping them there -- and not as a label on a dataset -- is what makes the residual of :mod:`.residuals` able to tell
them apart: each one moves a different subset of the physics, so the signature is the diagnosis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rotor import Rotor

__all__ = ["Fault", "FAULTS", "chip_factors", "faulty_airframe"]

#: Every fault this issue handles, and what it is allowed to touch.
FAULTS = ("payload", "motor", "blade", "gust", "gps", "magnetometer", "vibration", "battery")


def chip_factors(rotor: Rotor, omega: float, tip_loss: float = 0.10) -> tuple[float, float]:
    """``(thrust, torque)`` of a blade with its tip gone, over the intact one, both measured from BEMT at ``omega``.

    A chip removes the outer stations, which are the ones moving fastest: thrust falls and the *torque per newton*
    rises, because what is left of the blade has to work at a less favourable inflow. The numbers are measured here
    rather than assumed -- the ratio depends on where the chip is and how fast the rotor turns.
    """
    intact = rotor.state(omega)
    chipped = Rotor(**{**rotor.__dict__, "tip_loss": tip_loss}).state(omega)
    if intact.thrust_n <= 0.0 or intact.torque_nm <= 0.0:
        return 1.0, 1.0
    return chipped.thrust_n / intact.thrust_n, chipped.torque_nm / intact.torque_nm


@dataclass(frozen=True)
class Fault:
    """One injected fault: what kind, when it starts (and stops, if it does), how big, and on which motor."""

    kind: str
    start_s: float
    magnitude: float = 1.0
    motor: int = 0
    end_s: float | None = None
    #: for a gust, the direction in the world frame; for a payload, unused
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self):
        if self.kind not in FAULTS:
            raise ValueError(f"unknown fault {self.kind!r}; this library knows {FAULTS}")
        if self.start_s < 0.0:
            raise ValueError("a fault cannot start before the flight")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("a fault that ends before it starts is not a fault")
        if self.kind == "payload" and self.magnitude <= 0.0:
            raise ValueError("a payload adds mass: magnitude is kilograms and has to be positive")
        if self.kind == "motor" and not 0.0 < self.magnitude <= 1.0:
            raise ValueError("a motor's efficiency is a fraction in (0, 1]")
        if self.kind == "blade" and not 0.0 < self.magnitude < 1.0:
            raise ValueError("a chipped blade loses a fraction of its span: magnitude in (0, 1)")
        if self.motor not in range(4):
            raise ValueError(f"motor {self.motor} does not exist on a quad")

    def active(self, t: float) -> bool:
        return t >= self.start_s and (self.end_s is None or t < self.end_s)

    def label(self) -> str:
        if self.kind == "payload":
            return f"payload +{self.magnitude:.2f} kg"
        if self.kind == "motor":
            return f"motor {self.motor} at {100 * self.magnitude:.0f} % efficiency"
        if self.kind == "blade":
            return f"motor {self.motor} blade chipped ({100 * (1 - self.magnitude):.0f} % of span lost)"
        if self.kind == "gust":
            return f"gust {self.magnitude:.1f} m/s"
        if self.kind == "gps":
            return f"GPS glitch {self.magnitude:.1f} m"
        if self.kind == "magnetometer":
            return f"magnetometer interference {math.degrees(self.magnitude):.0f} deg"
        if self.kind == "vibration":
            return f"IMU vibration {self.magnitude:.1f} m/s^2"
        return f"battery sag ({100 * self.magnitude:.0f} % of nominal resistance added)"


@dataclass
class FaultState:
    """What the faults have done to the vehicle at one instant."""

    mass_kg: float
    thrust_gain: np.ndarray
    torque_gain: np.ndarray
    wind: np.ndarray
    battery_resistance_ohm: float
    gps_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading_error_rad: float = 0.0
    vibration_m_s2: float = 0.0
    active: tuple[str, ...] = ()


def faulty_airframe(base, *, mass_kg: float, battery_resistance_ohm: float):
    """The airframe with a fault's mass and battery change applied; the rotor table is shared, not rebuilt."""
    from dataclasses import replace

    battery = replace(base.battery, resistance_ohm=battery_resistance_ohm)
    return replace(base, mass_kg=mass_kg, battery=battery)


def apply_faults(faults, t: float, *, base_airframe, base_resistance_ohm: float, rotor, omega_hover: float,
                 frame: str = "world") -> FaultState:
    """The vehicle's state under ``faults`` at time ``t``: mass, motor gains, wind, battery, and what the sensors lie."""
    thrust_gain = np.ones(4)
    torque_gain = np.ones(4)
    mass = base_airframe.mass_kg
    wind = np.zeros(3)
    resistance = base_resistance_ohm
    gps_offset = np.zeros(3)
    heading_error = 0.0
    vibration = 0.0
    active = []
    for f in faults:
        if not f.active(t):
            continue
        active.append(f.label())
        if f.kind == "payload":
            mass += f.magnitude
        elif f.kind == "motor":
            thrust_gain[f.motor] *= f.magnitude
            torque_gain[f.motor] *= f.magnitude
        elif f.kind == "blade":
            gain_t, gain_q = chip_factors(rotor, omega_hover, tip_loss=f.magnitude)
            thrust_gain[f.motor] *= gain_t
            torque_gain[f.motor] *= gain_q
        elif f.kind == "gust":
            wind = wind + np.asarray(f.direction, float) / max(float(np.linalg.norm(f.direction)), 1e-9) * f.magnitude
        elif f.kind == "battery":
            resistance = base_resistance_ohm * (1.0 + f.magnitude)
        elif f.kind == "gps":
            gps_offset = gps_offset + np.asarray(f.direction, float) * f.magnitude
        elif f.kind == "magnetometer":
            heading_error += f.magnitude
        elif f.kind == "vibration":
            vibration = max(vibration, f.magnitude)
    return FaultState(mass, thrust_gain, torque_gain, wind, resistance, gps_offset, heading_error, vibration,
                      tuple(active))
