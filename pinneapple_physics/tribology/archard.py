"""Sliding wear of a bar end pressed against a rigid, flat counterface.

Archard's law, local form: the wear depth grows as dh/ds = K p, with K = k / H the specific wear rate
(mm³ / (N·m)), p the contact pressure (MPa = N/mm²) and s the sliding distance (m). The bar end is crowned (a
cylinder of radius R across the sliding direction), so at first only its middle touches; the pressure comes from an
elastic (Winkler) layer, p = E' / t · max(0, δ - g(x) - h(x)), with the approach δ set by the load at every step. Wear
removes the crown, the contact spreads and the pressure tends to the flat-punch value F / A: running-in, then
steady wear at dh/ds = K F / A.

Exact facts the solver must keep (checked in the lab experiment ``bar_wear``):

* worn volume = K F s at every moment (global Archard, any pressure distribution);
* the initial contact is the closed-form Winkler solution for a parabolic gap (``winkler_parabolic_contact``);
* after running-in the wear rate is K F / A.

The materials are the classic pin-on-ring data of Archard and Hirst (1956), as tabulated in Hutchings and Shipway,
*Tribology* (2nd ed., 2017), table 5.2 (k dimensionless, H Vickers hardness): order-of-magnitude values for dry
sliding at low load, not design data for a specific pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WearMaterial:
    name: str
    k: float              # Archard wear coefficient (dimensionless)
    hardness_MPa: float   # Vickers hardness
    E_GPa: float          # Young's modulus (for the elastic layer)

    @property
    def K(self) -> float:
        """Specific wear rate, mm³ / (N·m)."""
        return self.k / self.hardness_MPa * 1e3     # k / H [1/MPa = mm²/N] -> mm³/(N·mm) * 1e3 mm/m


WEAR_MATERIALS = {m.name: m for m in (
    WearMaterial("mild steel", 7e-3, 1860, 210),
    WearMaterial("60/40 brass", 6e-4, 950, 100),
    WearMaterial("hardened tool steel", 1.3e-4, 8500, 210),
    WearMaterial("stellite", 5.5e-5, 6900, 230),
    WearMaterial("PTFE", 2.5e-5, 50, 0.5),
    WearMaterial("ferritic stainless steel", 1.7e-5, 2500, 200),
    WearMaterial("polyethylene", 1.3e-7, 170, 1.0),
    WearMaterial("tungsten carbide", 1e-6, 13000, 600),
)}


def winkler_parabolic_contact(F_N: float, width_mm: float, R_mm: float, k_w: float) -> dict[str, float]:
    """Closed form for a parabolic gap g = x² / (2R) on a Winkler layer of modulus k_w (MPa/mm), load F over a
    width b: p = k_w (δ - x²/2R) on |x| < a, a = sqrt(2 R δ), F = (4/3) b k_w a δ."""
    delta = (3 * F_N / (4 * width_mm * k_w * math.sqrt(2 * R_mm))) ** (2 / 3)
    a = math.sqrt(2 * R_mm * delta)
    return {"delta_mm": delta, "half_width_mm": a, "p_max_MPa": k_w * delta}


@dataclass
class BarWear:
    """A bar of section length (along sliding, crowned) x width, end radius R, pressed with load F."""
    material: WearMaterial
    load_N: float = 50.0
    length_mm: float = 20.0
    width_mm: float = 10.0
    crown_radius_mm: float = 400.0
    layer_mm: float = 5.0          # thickness of the elastic layer (Winkler modulus E' / t)
    nx: int = 401

    def __post_init__(self):
        self.x = np.linspace(-self.length_mm / 2, self.length_mm / 2, self.nx)
        self.dx = self.x[1] - self.x[0]
        self.g = self.x ** 2 / (2 * self.crown_radius_mm)
        nu = 0.3
        self.k_w = self.material.E_GPa * 1e3 / (1 - nu ** 2) / self.layer_mm      # MPa / mm
        self.h = np.zeros_like(self.x)

    @property
    def area_mm2(self) -> float:
        return self.length_mm * self.width_mm

    def pressure(self, ds_m: float = 0.0) -> tuple[np.ndarray, float]:
        """Contact pressure (MPa) and approach δ (mm) that carry the load on the current profile. With ``ds_m`` the
        pressure is the one at the end of a wear step of that length (implicit: p = k (δ - gap - K ds p), i.e. an
        effective stiffness k / (1 + k K ds)), which keeps the wear update stable for any step."""
        k = self.k_w / (1.0 + self.k_w * self.material.K * ds_m)
        gap = self.g + self.h - (self.g + self.h).min()
        lo, hi = 0.0, gap.max() + self.load_N / (k * self.width_mm * self.dx) + 1e-12
        for _ in range(100):
            d = 0.5 * (lo + hi)
            F = self.width_mm * k * np.clip(d - gap, 0, None).sum() * self.dx
            lo, hi = (d, hi) if F < self.load_N else (lo, d)
        d = 0.5 * (lo + hi)
        p = k * np.clip(d - gap, 0, None)
        return p * (self.load_N / (self.width_mm * p.sum() * self.dx)), d     # exact load balance

    def run(self, distance_m: float, n_save: int = 60, max_step_um: float = 0.2) -> dict[str, np.ndarray]:
        """Slide ``distance_m``; returns the history (distance, worn volume, peak and mean pressure, contact
        fraction) and saved profiles / pressures."""
        K = self.material.K
        s, out = 0.0, {k: [] for k in ("s", "volume", "p_max", "contact", "rate")}
        prof, pres, s_saved = [], [], []
        save_at = np.linspace(0, distance_m, n_save)
        si = 0
        while s < distance_m - 1e-12:
            p0, _ = self.pressure()
            ds = min(distance_m - s, max_step_um * 1e-3 / max(K * p0.max(), 1e-30))
            p, _ = self.pressure(ds)
            if si < n_save and s >= save_at[si] - 1e-12:
                prof.append(self.h.copy())
                pres.append(p.copy())
                s_saved.append(s)
                si += 1
            self.h += K * p * ds
            s += ds
            out["s"].append(s)
            out["volume"].append(self.h.sum() * self.dx * self.width_mm)
            out["p_max"].append(p.max())
            out["contact"].append(float((p > 1e-6 * p.max()).mean()))
            out["rate"].append(K * self.load_N / self.width_mm / (self.dx * (p > 1e-6 * p.max()).sum()))
        p, _ = self.pressure()
        prof.append(self.h.copy())
        pres.append(p.copy())
        s_saved.append(s)
        res = {k: np.asarray(v) for k, v in out.items()}
        res.update(profiles=np.asarray(prof), pressures=np.asarray(pres), s_saved=np.asarray(s_saved), x=self.x,
                   crown=self.g)
        return res

    def running_in_distance_m(self) -> float:
        """Distance to wear the whole crown away at the flat-punch rate (an estimate of the running-in)."""
        crown_volume = (self.g.max() - self.g).sum() * self.dx * self.width_mm      # material above the flat end
        return crown_volume / (self.material.K * self.load_N) if self.material.K > 0 else math.inf
