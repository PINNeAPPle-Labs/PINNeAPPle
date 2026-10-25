"""Three-element Windkessel (RCR) outlets: the lumped model of the arterial tree downstream of a 3-D vessel.

``P = Rp Q + Pc``, ``C dPc/dt = Q - (Pc - P_venous) / Rd`` (Westerhof, Lankhaar and Westerhof, Med Biol Eng Comput
47:131-141, 2009): proximal (characteristic) resistance ``Rp``, compliance ``C``, distal resistance ``Rd``. SI units:
Q in m^3/s, P in Pa, R in Pa s / m^3, C in m^3 / Pa.

* :class:`Windkessel` integrates the model for any flow history (exact exponential update for piecewise-constant
  Q, so a coupling step of any length is stable) and gives its input impedance
  ``Z(w) = Rp + Rd / (1 + i w Rd C)``, the analytic check of the integrator;
* :func:`split_rcr` sizes one RCR per outlet from a target mean pressure, mean flow, flow fractions, a proximal
  share of the resistance and a total compliance (the usual way RCR outlets are tuned when no patient data are
  available: Q_i fraction by Murray's law ``d^3``, ``R_T,i = P_mean / Q_i``, ``C_i`` proportional to ``Q_i``);
* :func:`half_sine_inflow` is a simple cardiac inflow: a half-sine ejection of ``stroke_volume`` during
  ``systole`` of each period and no flow in diastole.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Windkessel", "split_rcr", "half_sine_inflow", "MMHG"]

MMHG = 133.322387415          # Pa


@dataclass
class Windkessel:
    Rp: float
    C: float
    Rd: float
    P_venous: float = 0.0
    Pc: float = 0.0           # state: pressure across the compliance

    @property
    def tau(self) -> float:
        return self.Rd * self.C

    def step(self, Q: float, dt: float) -> float:
        """Advance ``dt`` with the flow ``Q`` held constant; returns the outlet pressure at the end."""
        target = self.P_venous + self.Rd * Q
        self.Pc = target + (self.Pc - target) * math.exp(-dt / self.tau)
        return self.pressure(Q)

    def pressure(self, Q: float) -> float:
        return self.Rp * Q + self.Pc

    def simulate(self, t: np.ndarray, Q: np.ndarray, Pc0: float | None = None) -> np.ndarray:
        """Outlet pressure for a sampled flow (piecewise-constant between samples)."""
        if Pc0 is not None:
            self.Pc = Pc0
        P = np.empty(len(t))
        P[0] = self.pressure(float(Q[0]))
        for i in range(1, len(t)):
            P[i] = self.step(float(Q[i - 1]), float(t[i] - t[i - 1])) + self.Rp * (float(Q[i]) - float(Q[i - 1]))
        return P

    def impedance(self, omega) -> np.ndarray:
        omega = np.asarray(omega, float)
        return self.Rp + self.Rd / (1 + 1j * omega * self.Rd * self.C)

    def periodic_state(self, t: np.ndarray, Q: np.ndarray, cycles: int = 20) -> float:
        """``Pc`` at the start of a cycle once the response is periodic (repeats the cycle ``cycles`` times)."""
        for _ in range(cycles):
            self.simulate(t, Q)
        return self.Pc


def split_rcr(diameters: Mapping[str, float], *, P_mean: float = 93 * MMHG, Q_mean: float = 70e-6,
              proximal_fraction: float = 0.09, C_total: float = 1.0e-8,
              fractions: Mapping[str, float] | None = None) -> dict[str, Windkessel]:
    """One :class:`Windkessel` per outlet. Flow fractions default to Murray's law (``d^3``); every outlet sees the
    same mean pressure ``P_mean`` at its mean flow, ``Rp = proximal_fraction R_T``, and the compliance is shared in
    proportion to the flow. The defaults (93 mmHg, 70 mL/s, 9 %, 1e-8 m^3/Pa ~ 1.3 mL/mmHg) are round adult values
    chosen for a demonstration, not patient data."""
    if fractions is None:
        w = {k: d ** 3 for k, d in diameters.items()}
        s = sum(w.values())
        fractions = {k: v / s for k, v in w.items()}
    out = {}
    for k, f in fractions.items():
        RT = P_mean / (f * Q_mean)
        out[k] = Windkessel(Rp=proximal_fraction * RT, C=C_total * f, Rd=(1 - proximal_fraction) * RT,
                            Pc=(1 - proximal_fraction) * P_mean)
    return out


def half_sine_inflow(t, *, stroke_volume: float = 70e-6, period: float = 1.0, systole: float = 0.3) -> np.ndarray:
    """Flow (m^3/s): ``Qmax sin(pi tau / systole)`` for ``tau = t mod period < systole``, else 0, with
    ``Qmax = pi stroke_volume / (2 systole)`` so each beat ejects ``stroke_volume``."""
    tau = np.mod(np.asarray(t, float), period)
    qmax = math.pi * stroke_volume / (2 * systole)
    return np.where(tau < systole, qmax * np.sin(np.pi * tau / systole), 0.0)


def check_against_impedance(wk: Windkessel, omega: float, amplitude: float, n_per_period: int = 2000,
                            periods: int = 40) -> dict[str, float]:
    """Drive the integrator with ``Q = amplitude sin(w t)`` and compare the periodic pressure amplitude and phase
    with ``|Z(w)|`` and ``arg Z(w)``. Returns the relative amplitude error and the phase error (rad)."""
    T = 2 * math.pi / omega
    t = np.linspace(0, periods * T, periods * n_per_period + 1)
    Q = amplitude * np.sin(omega * t)
    P = wk.simulate(t, Q, Pc0=0.0)
    last = t >= (periods - 1) * T
    tt, PP = t[last], P[last] - P[last].mean()
    a = 2 * np.mean(PP * np.sin(omega * tt))
    b = 2 * np.mean(PP * np.cos(omega * tt))
    Z = wk.impedance(omega)
    return {"amplitude_rel_error": abs(math.hypot(a, b) / (amplitude * abs(Z)) - 1),
            "phase_error_rad": abs(math.atan2(b, a) - float(np.angle(Z)))}


__all__ += ["check_against_impedance"]


def outlet_flows(Q_in: float, resistances: Sequence[float]) -> np.ndarray:
    """Steady flow split of parallel resistances (the 0-D limit used to check a 3-D coupled run)."""
    g = 1 / np.asarray(resistances, float)
    return Q_in * g / g.sum()
