"""Catalogue electrical steel and magnet data for machine models, with their sources.

``M400_50A``: non-oriented electrical steel grade M400-50A (EN 10106; 0.50 mm, at most 4.00 W/kg at 1.5 T,
50 Hz): the B-H curve (44 points, 0 to 2.3 T) and the specific-loss table (W/kg at 50, 100, 200, 400, 1000 and
2500 Hz, 0.1 to 1.8 T) of pyleecan's material library (``pyleecan/Data/Material/M400-50A.json``, pyleecan 1.4.2,
Apache-2.0), resistivity 4.6e-7 Ohm m, density 7650 kg/m^3. These are the data of a supplier datasheet for the
grade, not of one coil: a design should use its own supplier's curve.

``MAGNET_PRIUS``: sintered NdFeB of pyleecan's ``MagnetPrius.json``: Br = 1.24 T at 20 degC, mu_r = 1.05,
reversible Br temperature coefficient -0.1 %/K (pyleecan ``alpha_Br`` = -0.001), resistivity 1.6e-6 Ohm m.

:func:`stacked` turns a solid-steel curve into the curve of a lamination stack with stacking factor ``kf``
(``B = kf B_steel(H) + (1 - kf) mu0 H``; the flux runs in the plane of the sheets). :func:`fit_iron_loss` fits
:class:`~..postprocess.IronLossModel` (Steinmetz hysteresis + Bertotti classical and excess eddy terms) to a loss
table and returns the fit's errors; :class:`TableIronLoss` interpolates the table itself (log-log), so it
reproduces the catalogue exactly at its points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import MU0
from ..materials import BHCurve
from ..postprocess import IronLossModel, classical_eddy_coefficient

__all__ = ["SteelGrade", "M400_50A", "MagnetGrade", "MAGNET_PRIUS", "stacked", "fit_iron_loss", "LossFit",
           "TableIronLoss"]

_M400_50A_HB = [
    (100.0, 0.5), (150.0, 0.7), (180.0, 0.8), (200.0, 0.9), (250.0, 1.0), (300.0, 1.05), (350.0, 1.1),
    (450.0, 1.15), (550.0, 1.2), (650.0, 1.225), (750.0, 1.25), (850.0, 1.275), (950.0, 1.3), (1100.0, 1.325),
    (1250.0, 1.35), (1400.0, 1.375), (1550.0, 1.4), (1700.0, 1.425), (1900.0, 1.45), (2150.0, 1.475),
    (2450.0, 1.5), (2750.0, 1.525), (3150.0, 1.55), (3600.0, 1.575), (4100.0, 1.6), (4700.0, 1.625),
    (5250.0, 1.65), (6000.0, 1.675), (6700.0, 1.7), (7500.0, 1.725), (8650.0, 1.75), (9500.0, 1.775),
    (10750.0, 1.8), (14500.0, 1.85), (19500.0, 1.9), (25000.0, 1.95), (33000.0, 2.0), (44000.0, 2.05),
    (57000.0, 2.1), (74000.0, 2.15), (96000.0, 2.2), (130000.0, 2.25), (170000.0, 2.3)]

_M400_50A_LOSS = {  # f (Hz): W/kg at B = 0.1, 0.2, ... T
    50.0: [0.02, 0.09, 0.19, 0.31, 0.46, 0.62, 0.81, 1.01, 1.24, 1.49, 1.76, 2.09, 2.46, 2.96, 3.57, 4.38, 5.02, 5.47],
    100.0: [0.07, 0.26, 0.54, 0.88, 1.27, 1.73, 2.24, 2.8, 3.44, 4.15, 4.95, 5.85, 6.88, 8.18, 9.82],
    200.0: [0.16, 0.64, 1.35, 2.25, 3.33, 4.58, 6.03, 7.68, 9.58, 11.7, 14.2, 17.0, 20.2, 23.8, 28.3],
    400.0: [0.48, 1.8, 3.77, 6.29, 9.37, 13.1, 17.5, 22.7, 28.8, 35.9, 44.2, 53.8, 64.9, 77.4, 91.7],
    1000.0: [2.12, 7.49, 15.3, 25.7, 39.0, 56.1, 77.1, 103.1, 135.0, 173.3, 218.8, 272.4, 334.6, 405.6, 488.4],
    2500.0: [8.34, 30.1, 62.7, 109.0, 172.0, 256.0, 367.0, 509.0, 685.0, 899.0, 1155.0, 1453.0, 1793.0, 2130.0],
}


@dataclass(frozen=True)
class SteelGrade:
    name: str
    H: np.ndarray
    B: np.ndarray
    loss_f: np.ndarray       # Hz
    loss_B: np.ndarray       # T (peak)
    loss_W_per_kg: np.ndarray
    thickness: float         # m
    resistivity: float       # Ohm m
    density: float           # kg/m^3
    source: str

    def curve(self, stacking_factor: float = 1.0) -> BHCurve:
        c = BHCurve(self.B, self.H)
        return c if stacking_factor >= 1.0 else stacked(c, stacking_factor)


def _loss_arrays(table):
    f, B, p = [], [], []
    for fr, vals in table.items():
        for k, v in enumerate(vals):
            f.append(fr)
            B.append(0.1 * (k + 1))
            p.append(v)
    return np.array(f), np.round(np.array(B), 10), np.array(p)


M400_50A = SteelGrade(
    "M400-50A", np.array([h for h, _ in _M400_50A_HB]), np.array([b for _, b in _M400_50A_HB]),
    *_loss_arrays(_M400_50A_LOSS), thickness=0.5e-3, resistivity=4.6e-7, density=7650.0,
    source="pyleecan 1.4.2 material library, M400-50A.json (Apache-2.0); grade per EN 10106")


@dataclass(frozen=True)
class MagnetGrade:
    name: str
    Br20: float          # T at 20 degC
    mu_r: float
    alpha_Br: float      # 1/K, reversible
    resistivity: float   # Ohm m
    density: float
    source: str

    def Br(self, temperature: float = 20.0) -> float:
        return self.Br20 * (1.0 + self.alpha_Br * (temperature - 20.0))

    @property
    def Hc(self) -> float:
        """Coercivity of the linear recoil line at 20 degC (A/m)."""
        return self.Br20 / (MU0 * self.mu_r)


MAGNET_PRIUS = MagnetGrade("NdFeB (Prius model)", 1.24, 1.05, -0.001, 1.6e-6, 7500.0,
                           source="pyleecan 1.4.2 material library, MagnetPrius.json (Apache-2.0)")


def stacked(curve: BHCurve, kf: float, n: int = 200) -> BHCurve:
    """B-H curve of a lamination stack: ``B = kf B_steel(H) + (1 - kf) mu0 H``."""
    H = np.concatenate([np.logspace(-1, math.log10(curve.H[-1]), n)])
    Bs = np.interp(H, curve.H, curve.B)
    # the steel curve is defined by H(B); invert it with the monotone interpolant on a dense B grid
    b = np.linspace(0.0, curve.B[-1], 4 * n)
    h = curve.h(b)
    Bs = np.interp(H, h, b)
    return BHCurve(kf * Bs + (1.0 - kf) * MU0 * H, H)


@dataclass
class LossFit:
    model: IronLossModel
    rms_rel_error: float
    max_rel_error: float
    fitted: tuple


def fit_iron_loss(grade: SteelGrade, classical_from_resistivity: bool = True, B_max: float = 1.8,
                  f_max: float = 2500.0) -> LossFit:
    """Least-squares fit (relative errors) of ``k_h f^alpha B^beta + k_c (f B)^2 + k_e (f B)^1.5`` to the
    grade's loss table. With ``classical_from_resistivity`` the classical eddy coefficient is the
    thin-sheet value ``pi^2 d^2 / (6 rho_e density)`` and only ``k_h, alpha, beta, k_e`` are fitted."""
    from scipy.optimize import least_squares

    m = (grade.loss_B <= B_max + 1e-9) & (grade.loss_f <= f_max)
    f, B, p = grade.loss_f[m], grade.loss_B[m], grade.loss_W_per_kg[m]
    kc0 = classical_eddy_coefficient(1.0 / grade.resistivity, grade.thickness, grade.density)

    def unpack(x):
        kh, a, b, ke = math.exp(x[0]), x[1], x[2], math.exp(x[3])
        kc = kc0 if classical_from_resistivity else math.exp(x[4])
        return kh, a, b, kc, ke

    def resid(x):
        kh, a, b, kc, ke = unpack(x)
        return (kh * f ** a * B ** b + kc * (f * B) ** 2 + ke * (f * B) ** 1.5) / p - 1.0

    x0 = [math.log(0.02), 1.0, 2.0, math.log(1e-4)] + ([] if classical_from_resistivity else [math.log(kc0)])
    sol = least_squares(resid, x0, method="lm")
    kh, a, b, kc, ke = unpack(sol.x)
    r = resid(sol.x)
    model = IronLossModel(k_h=kh, alpha=a, beta=b, k_c=kc, k_e=ke, density=grade.density)
    return LossFit(model, float(np.sqrt(np.mean(r ** 2))), float(np.max(np.abs(r))), (kh, a, b, kc, ke))


class TableIronLoss:
    """Specific iron loss interpolated from a catalogue table (same interface as ``IronLossModel``).

    For each tabulated frequency, ``log p`` is interpolated linearly in ``log B`` (beyond the last point:
    the power law of the last two points); between frequencies, linearly in ``log f`` (beyond the table:
    the power law of the two nearest frequencies)."""

    def __init__(self, grade: SteelGrade):
        self.density = grade.density
        self.freqs = np.unique(grade.loss_f)
        self.rows = [(np.log(grade.loss_B[grade.loss_f == f]), np.log(grade.loss_W_per_kg[grade.loss_f == f]))
                     for f in self.freqs]

    @staticmethod
    def _interp(x, xp, yp):
        y = np.interp(x, xp, yp)
        lo, hi = x < xp[0], x > xp[-1]
        y = np.where(lo, yp[0] + (x - xp[0]) * (yp[1] - yp[0]) / (xp[1] - xp[0]), y)
        return np.where(hi, yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y)

    def specific_loss(self, f, B):
        f = np.asarray(f, float)
        B = np.asarray(B, float)
        f, B = np.broadcast_arrays(f, B)
        out = np.zeros(B.shape)
        ok = (B > 0) & (f > 0)
        if not ok.any():
            return out
        lB, lf = np.log(B[ok]), np.log(f[ok])
        per_row = np.array([self._interp(lB, xb, yp) for xb, yp in self.rows])  # (n_f, n)
        lF = np.log(self.freqs)
        k = np.clip(np.searchsorted(lF, lf) - 1, 0, len(lF) - 2)
        t = (lf - lF[k]) / (lF[k + 1] - lF[k])
        idx = np.arange(len(lf))
        out[ok] = np.exp(per_row[k, idx] * (1 - t) + per_row[k + 1, idx] * t)
        return out
