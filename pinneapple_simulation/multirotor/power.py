"""Electric drive of a multirotor: a brushless DC motor (Kv / Kt with copper and no-load losses) and a lithium-polymer
pack (Peukert-corrected capacity, open-circuit voltage against state of charge, internal-resistance sag).

Motor (the standard first-order DC model, e.g. Drela, *First-Order DC Electric Motor Model*, MIT 2007):
``K_t = 1 / K_v`` in SI units (``K_v`` [rad/s per V]; ``K_t = 60 / (2 pi Kv_rpm)`` [N m/A]),
``I = Q / K_t + I_0``, ``V = I R_m + omega / K_v``, ``P_elec = V I = Q omega + I^2 R_m + I_0 omega / K_v``
(shaft + copper + no-load loss: :meth:`Motor.operating_point` returns all three and their sum closes ``V I``).

Battery: ``n_series`` cells, open-circuit voltage ``V_oc(SOC)`` interpolated from ``ocv_curve`` (a stated typical
LiPo curve, 3.3 V empty to 4.2 V full; replace it with the cell's datasheet curve), terminal voltage
``V = n (V_oc(SOC) - I R_cell / n_parallel)``. Peukert: at a constant current ``I`` the delivered capacity is
``C (C / (I H))^{k-1}`` for a rating at ``H`` hours (Peukert 1897; ``k = 1`` switches it off; lithium cells are close
to 1, the default 1.05 is a stated assumption). The discharge is integrated in time with the charge drawn scaled
by ``(I H / C)^{k-1}``, which gives that capacity exactly at constant current.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Motor", "Battery", "LIPO_OCV"]

#: (SOC, cell open-circuit voltage [V]): a typical LiPo curve (stated assumption, not a datasheet)
LIPO_OCV: tuple[tuple[float, float], ...] = ((0.0, 3.30), (0.05, 3.50), (0.10, 3.60), (0.20, 3.70), (0.40, 3.78),
                                             (0.60, 3.87), (0.80, 4.00), (0.90, 4.08), (1.00, 4.20))


@dataclass
class Motor:
    """``kv_rpm`` [rpm/V], winding resistance ``r_ohm`` [ohm], no-load current ``i0`` [A], ``mass`` [kg]."""
    kv_rpm: float
    r_ohm: float = 0.1
    i0: float = 0.5
    mass: float = 0.05
    max_current: float = 30.0

    @property
    def kt(self) -> float:
        """Torque constant [N m / A] = ``60 / (2 pi Kv_rpm)``."""
        return 60.0 / (2.0 * math.pi * self.kv_rpm)

    def operating_point(self, torque: float, omega: float) -> dict:
        """Current [A], voltage [V], electrical power [W] and its split for a shaft ``torque`` [N m] at ``omega``."""
        i = torque / self.kt + self.i0
        emf = omega * self.kt
        v = i * self.r_ohm + emf
        p = v * i
        shaft, cu, nl = torque * omega, i * i * self.r_ohm, self.i0 * emf
        return {"current_A": i, "voltage_V": v, "power_W": p, "shaft_W": shaft, "copper_W": cu, "no_load_W": nl,
                "efficiency": shaft / p if p > 0 else float("nan"), "closure": abs(shaft + cu + nl - p) / max(p, 1e-12)}

    def max_omega(self, voltage: float, torque_of_omega) -> float:
        """Highest speed [rad/s] the motor reaches at ``voltage`` against a load ``torque_of_omega(omega)``."""
        lo, hi = 0.0, voltage / self.kt
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            need = self.operating_point(torque_of_omega(mid), mid)["voltage_V"]
            lo, hi = (mid, hi) if need < voltage else (lo, mid)
        return lo


@dataclass
class Battery:
    """``n_series`` x ``n_parallel`` cells of ``capacity_ah`` [A h] each, cell resistance ``r_cell`` [ohm], Peukert
    exponent ``peukert`` at a rating of ``rating_hours`` [h], ``specific_energy`` [W h/kg] for the pack mass."""
    n_series: int
    capacity_ah: float
    n_parallel: int = 1
    r_cell: float = 0.01
    peukert: float = 1.05
    rating_hours: float = 1.0
    specific_energy: float = 150.0
    ocv_curve: Sequence[tuple[float, float]] = field(default=LIPO_OCV)
    cutoff_cell_v: float = 3.3
    reserve_soc: float = 0.0

    @property
    def capacity(self) -> float:
        """Pack capacity [A h]."""
        return self.capacity_ah * self.n_parallel

    @property
    def nominal_voltage(self) -> float:
        return 3.7 * self.n_series

    @property
    def energy_wh(self) -> float:
        return self.capacity * self.nominal_voltage

    @property
    def mass(self) -> float:
        return self.energy_wh / self.specific_energy

    def ocv(self, soc: float) -> float:
        s, v = zip(*self.ocv_curve, strict=True)
        return float(self.n_series * np.interp(np.clip(soc, 0.0, 1.0), s, v))

    def terminal_voltage(self, soc: float, current: float) -> float:
        return self.ocv(soc) - current * self.r_cell * self.n_series / self.n_parallel

    def peukert_capacity(self, current: float) -> float:
        """Capacity [A h] delivered at a constant ``current`` [A] (Peukert)."""
        c = self.capacity
        return c * (c / max(current * self.rating_hours, 1e-12)) ** (self.peukert - 1.0)

    def draw(self, soc: float, current: float, dt: float) -> float:
        """State of charge after drawing ``current`` [A] for ``dt`` [s]."""
        c = self.capacity
        eff = (max(current, 0.0) * self.rating_hours / c) ** (self.peukert - 1.0) if current > 0 else 1.0
        return soc - current * eff * dt / 3600.0 / c

    def current_for_power(self, soc: float, power: float) -> float:
        """Current [A] that delivers ``power`` [W] at the terminals: ``(V_oc - I R) I = P`` (the smaller root);
        nan when the pack cannot deliver it."""
        v, r = self.ocv(soc), self.r_cell * self.n_series / self.n_parallel
        disc = v * v - 4.0 * r * power
        return (v - math.sqrt(disc)) / (2.0 * r) if disc >= 0 else float("nan")

    def empty(self, soc: float, current: float) -> bool:
        return soc <= self.reserve_soc or self.terminal_voltage(soc, current) <= self.cutoff_cell_v * self.n_series
