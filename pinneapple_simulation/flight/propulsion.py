"""Engines for flight simulation: thrust against speed and altitude, and where it acts.

* :class:`Turbofan`: installed thrust lapse of a high-bypass turbofan, Mattingly (Aircraft Engine Design, 2nd ed.,
  2002, eq. 2.45a): T / T_SL = delta_0 (1 - 0.49 sqrt(M)), delta_0 = p_t / p_SL (total pressure ratio).
* :class:`Propeller`: shaft power P(h) (piston: Gagg-Ferrar P/P_SL = 1.132 sigma - 0.132; turboprop: sigma^0.75, an
  assumption for a flat-rated engine below its flat-rating altitude is not modelled) driving a propeller with
  efficiency eta: T = eta P / sqrt(V^2 + V0^2), V0 = eta P / T_static, so T -> T_static at rest and eta P / V in
  flight; static thrust from momentum theory with a figure of merit: T_static = (FM P)^(2/3) (2 rho A)^(1/3).

Positions in airframe axes (x aft, y right, z up); the thrust acts along the body x axis tilted by ``cant_deg``
(nose-up positive). Propeller torque, P-factor and slipstream are not modelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

P_SL, RHO_SL, R_AIR = 101325.0, 1.225, 287.05


def _static_pressure(rho: float, a: float) -> float:
    T = a * a / (1.4 * R_AIR)
    return rho * R_AIR * T


@dataclass
class Turbofan:
    static_thrust: float                 # N at sea level, take-off rating
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cant_deg: float = 0.0
    name: str = "turbofan"

    def thrust(self, V: float, rho: float, a: float, throttle: float = 1.0) -> float:
        M = max(V, 0.0) / a
        delta0 = _static_pressure(rho, a) / P_SL * (1 + 0.2 * M * M) ** 3.5
        return float(np.clip(throttle, 0, 1)) * self.static_thrust * delta0 * (1 - 0.49 * math.sqrt(M))


@dataclass
class Propeller:
    power: float                         # W, sea-level shaft power at full throttle
    diameter: float                      # m
    efficiency: float = 0.80             # in flight (assumption unless given)
    figure_of_merit: float = 0.70        # static (assumption)
    kind: str = "piston"                 # "piston" | "turboprop"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cant_deg: float = 0.0
    name: str = "propeller"

    def shaft_power(self, rho: float) -> float:
        s = rho / RHO_SL
        lapse = 1.132 * s - 0.132 if self.kind == "piston" else s ** 0.75
        return self.power * max(lapse, 0.0)

    def static_thrust(self, rho: float, power: float | None = None) -> float:
        P = self.shaft_power(rho) if power is None else power
        A = math.pi * self.diameter ** 2 / 4
        return (self.figure_of_merit * P) ** (2 / 3) * (2 * rho * A) ** (1 / 3)

    def thrust(self, V: float, rho: float, a: float, throttle: float = 1.0) -> float:
        P = float(np.clip(throttle, 0, 1)) * self.shaft_power(rho)
        if P <= 0:
            return 0.0
        Ts = self.static_thrust(rho, P)
        V0 = self.efficiency * P / Ts
        return self.efficiency * P / math.sqrt(V * V + V0 * V0)


def thrust_direction_body(cant_deg: float) -> np.ndarray:
    c = math.radians(cant_deg)
    return np.array([math.cos(c), 0.0, -math.sin(c)])
