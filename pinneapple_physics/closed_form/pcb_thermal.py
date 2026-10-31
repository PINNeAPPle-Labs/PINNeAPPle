"""PCB thermal building blocks (closed form, citable).

* Laminate properties: copper/FR-4 layers with copper coverage; per-layer
  in-plane and through-plane conductivity from the parallel (area-fraction)
  rule, board-level effective values from the parallel (in-plane) and
  series (through-plane) rules -- the standard compact treatment used by
  JEDEC JESD51-7 style board models.
* Thermal vias: plated barrels add copper area in parallel through the
  dielectric under a footprint.
* Surface heat transfer from a flat board, both faces:
    - natural convection, vertical plate: Churchill & Chu (1975),
      L = board height;
    - natural convection, horizontal plate: hot face up Nu = 0.54 Ra^1/4 /
      0.15 Ra^1/3, hot face down Nu = 0.27 Ra^1/4 (Incropera & DeWitt),
      L = A / P;
    - forced parallel flow: flat plate, laminar 0.664 Re^1/2 Pr^1/3,
      mixed (0.037 Re^0.8 - 871) Pr^1/3 above Re = 5e5;
    - radiation to surroundings at ambient, linearised:
      h_rad = eps sigma (Ts^2 + Ta^2)(Ts + Ta).
* JEDEC JESD15-3 two-resistor component model (theta_JC to the case top,
  theta_JB to the board).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .plate_fin_heatsink import G, air_properties

SIGMA = 5.670374419e-8
K_CU = 390.0
K_FR4_XY, K_FR4_Z = 0.8, 0.3          # woven glass/epoxy: in-plane > through-plane
OZ_UM = 34.8                          # 1 oz/ft2 copper thickness


@dataclass
class Layer:
    name: str
    kind: str                 # "copper" | "dielectric"
    thickness_m: float
    coverage: float = 1.0     # copper area fraction (copper layers only)

    def k_xy(self) -> float:
        if self.kind == "copper":
            return self.coverage * K_CU + (1 - self.coverage) * K_FR4_XY
        return K_FR4_XY

    def k_z(self) -> float:
        if self.kind == "copper":
            return self.coverage * K_CU + (1 - self.coverage) * K_FR4_Z
        return K_FR4_Z


def stackup(n_copper: int = 4, thickness_mm: float = 1.6, outer_oz: float = 1.0,
            inner_oz: float = 1.0, outer_coverage: float = 0.3, inner_coverage: float = 0.9,
            ) -> List[Layer]:
    """Symmetric stack: signal outer layers (partial copper), plane inner layers,
    dielectric filling the rest of the board thickness equally."""
    if n_copper < 1:
        raise ValueError("need at least one copper layer")
    cu = []
    for i in range(n_copper):
        outer = i in (0, n_copper - 1)
        oz = outer_oz if outer else inner_oz
        cu.append(Layer(f"L{i + 1}", "copper", oz * OZ_UM * 1e-6,
                        outer_coverage if outer else inner_coverage))
    t_cu = sum(layer.thickness_m for layer in cu)
    n_diel = max(n_copper - 1, 1)
    t_d = (thickness_mm * 1e-3 - t_cu) / n_diel
    if t_d <= 0:
        raise ValueError("copper thicker than the board")
    layers: List[Layer] = []
    for i, c in enumerate(cu):
        layers.append(c)
        if i < n_copper - 1 or n_copper == 1:
            layers.append(Layer(f"D{i + 1}", "dielectric", t_d))
    return layers


def effective_conductivity(layers: List[Layer]) -> Dict[str, float]:
    t = sum(layer.thickness_m for layer in layers)
    k_xy = sum(layer.k_xy() * layer.thickness_m for layer in layers) / t
    k_z = t / sum(layer.thickness_m / layer.k_z() for layer in layers)
    return {"k_xy": k_xy, "k_z": k_z, "thickness_m": t,
            "copper_thickness_m": sum(la.thickness_m for la in layers if la.kind == "copper")}


def via_copper_fraction(n_vias: int, drill_mm: float, plating_um: float, area_m2: float) -> float:
    """Copper area fraction of plated via barrels over a footprint (unfilled vias).

    Returns
    -------
    phi : float [-]
        Copper area fraction.
    """
    r_o = drill_mm * 1e-3 / 2
    r_i = max(r_o - plating_um * 1e-6, 0.0)
    return min(n_vias * math.pi * (r_o ** 2 - r_i ** 2) / area_m2, 1.0)


# ── surface heat transfer ────────────────────────────────────────────────────

@dataclass
class Environment:
    t_ambient_c: float = 25.0
    air_velocity_m_s: float = 0.0          # 0 = natural convection
    orientation: str = "vertical"          # "vertical" | "horizontal" (natural only)
    emissivity: float = 0.9                # solder mask ~0.9
    extra: Dict[str, float] = field(default_factory=dict)


def convection_coefficients(width_m: float, depth_m: float, t_surface_c: Tuple[float, float],
                            env: Environment) -> Dict[str, float]:
    """h for the (top, bottom) faces at the given mean surface temperatures."""
    Ta = env.t_ambient_c
    out: Dict[str, float] = {}
    warnings: List[str] = []
    for face, ts in zip(("top", "bottom"), t_surface_c, strict=True):
        dt = max(ts - Ta, 1e-3)
        air = air_properties(0.5 * (ts + Ta))
        if env.air_velocity_m_s > 0:
            L = depth_m                                   # flow along the depth
            re = env.air_velocity_m_s * L / air["nu"]
            nu = (0.664 * math.sqrt(re) * air["pr"] ** (1 / 3) if re < 5e5
                  else (0.037 * re ** 0.8 - 871) * air["pr"] ** (1 / 3))
            regime = f"forced, Re={re:.3g}"
        elif env.orientation == "vertical":
            L = depth_m                                   # board height
            ra = G * air["beta"] * dt * L ** 3 / (air["nu"] * air["alpha"])
            nu = (0.825 + 0.387 * ra ** (1 / 6) / (1 + (0.492 / air["pr"]) ** (9 / 16)) ** (8 / 27)) ** 2
            regime = f"natural vertical, Ra={ra:.3g}"
            if ra > 1e9:
                warnings.append(f"{face}: Ra={ra:.2g} > 1e9, turbulent natural convection")
        else:
            L = width_m * depth_m / (2 * (width_m + depth_m))
            ra = G * air["beta"] * dt * L ** 3 / (air["nu"] * air["alpha"])
            if face == "top":
                nu = 0.54 * ra ** 0.25 if ra < 1e7 else 0.15 * ra ** (1 / 3)
            else:
                nu = 0.27 * ra ** 0.25
            regime = f"natural horizontal ({'hot face up' if face == 'top' else 'hot face down'}), Ra={ra:.3g}"
            if ra < 1e4 and face == "top":
                warnings.append(f"{face}: Ra={ra:.2g} below the 1e4 validity limit of the upper-face correlation")
        h_conv = nu * air["k"] / L
        tk, tak = ts + 273.15, Ta + 273.15
        h_rad = env.emissivity * SIGMA * (tk ** 2 + tak ** 2) * (tk + tak)
        out[f"h_conv_{face}"] = h_conv
        out[f"h_rad_{face}"] = h_rad
        out[f"h_{face}"] = h_conv + h_rad
        out[f"regime_{face}"] = regime
    out["warnings"] = warnings
    return out


# ── component packages (JEDEC two-resistor model) ───────────────────────────
# Typical values. theta_JC(top) is the median of published datasheets;
# theta_JB is the *model* value that makes this board model reproduce the
# median published JEDEC JESD51-7 (2s2p, still air) theta_JA of the package
# (target in ``JEDEC_2S2P_THETA_JA``, checked in tests/test_pcb_hotspot.py).
# It lands inside the datasheet theta_JB range but a little lower, because
# here the board node is directly under the part, not at its edge
# (JESD51-8). Users should enter their own datasheet numbers for sign-off.
PACKAGES: Dict[str, Dict[str, float]] = {
    "QFN-32 5x5":   {"w_mm": 5.0, "d_mm": 5.0, "h_mm": 0.9, "theta_jb": 11.0, "theta_jc": 25.0},
    "QFN-64 9x9":   {"w_mm": 9.0, "d_mm": 9.0, "h_mm": 0.9, "theta_jb": 7.3, "theta_jc": 15.0},
    "BGA 17x17":    {"w_mm": 17.0, "d_mm": 17.0, "h_mm": 1.5, "theta_jb": 14.0, "theta_jc": 6.0},
    "BGA 35x35":    {"w_mm": 35.0, "d_mm": 35.0, "h_mm": 2.5, "theta_jb": 1.4, "theta_jc": 0.5},
    "QFP-100 14x14": {"w_mm": 14.0, "d_mm": 14.0, "h_mm": 1.4, "theta_jb": 31.0, "theta_jc": 12.0},
    "SOIC-8":       {"w_mm": 4.9, "d_mm": 6.0, "h_mm": 1.5, "theta_jb": 87.0, "theta_jc": 60.0},
    "TO-252 DPAK":  {"w_mm": 6.6, "d_mm": 10.0, "h_mm": 2.3, "theta_jb": 20.0, "theta_jc": 52.0},
    "TO-263 D2PAK": {"w_mm": 10.2, "d_mm": 15.0, "h_mm": 4.5, "theta_jb": 12.0, "theta_jc": 40.0},
    "LED 3535":     {"w_mm": 3.5, "d_mm": 3.5, "h_mm": 2.0, "theta_jb": 8.0, "theta_jc": 150.0},
    "Inductor 10x10": {"w_mm": 10.0, "d_mm": 10.0, "h_mm": 4.0, "theta_jb": 15.0, "theta_jc": 30.0},
}

# Median published theta_JA [K/W] on the JEDEC 2s2p board in still air, e.g.
# SOIC-8: TI ISO6521 104.6, SN75157 116.7, OPA892 124.5, TL082 147.8;
# LQFP-100: ST 51, JCET 37.2; TO-252: TI LM1117 NDP 45.1; FCBGA: Microchip
# PolarFire 8.2. QFN with exposed pad soldered and via'd to the planes.
JEDEC_2S2P_THETA_JA: Dict[str, float] = {
    "QFN-32 5x5": 33.0, "QFN-64 9x9": 24.0, "BGA 17x17": 28.0, "BGA 35x35": 9.5,
    "QFP-100 14x14": 45.0, "SOIC-8": 120.0, "TO-252 DPAK": 45.0, "TO-263 D2PAK": 30.0,
}
