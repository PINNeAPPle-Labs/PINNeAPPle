"""Plate-fin heat sink thermal/hydraulic model (closed form, citable).

Straight rectangular fins on a flat base, cooled either by ducted forced
air or by natural convection with vertical fins. Every relation below is a
published engineering correlation, not a fitted/invented formula:

* Air properties at 1 atm: tabulated (Incropera & DeWitt, Table A.4),
  linearly interpolated, ideal-gas density.
* Forced convection, parallel-plate channels (developing + fully developed
  composite): Teertstra, Yovanovich & Culham, "Analytical forced
  convection modeling of plate fin heat sinks", J. Electronics
  Manufacturing 10(4), 2000::

      Re_b* = Re_b * b / L,   Re_b = V_ch * b / nu
      Nu_b  = [ (Re_b* Pr / 2)^-3
               + (0.664 sqrt(Re_b*) Pr^(1/3) sqrt(1 + 3.65 / sqrt(Re_b*)))^-3 ]^(-1/3)

  ``h`` is referenced to the INLET air temperature: the ``Re* Pr / 2``
  limit is exactly the energy balance of air leaving at the wall
  temperature, so air heating is already accounted for.
* Pressure drop: Muzychka & Yovanovich apparent-friction model with the
  Shah & London fully developed ``f Re`` for rectangular ducts, plus
  Kays & London contraction/expansion losses.
* Natural convection, vertical isothermal parallel plates: Bar-Cohen &
  Rohsenow, J. Heat Transfer 106, 1984::

      El   = g beta dT b^4 / (nu alpha L)
      Nu_b = [ 576 / El^2 + 2.873 / sqrt(El) ]^(-1/2)

* Fin efficiency: :mod:`.fin_array_conduction` (Incropera, adiabatic-tip
  corrected length).
* Base spreading resistance (source smaller than the base, including 1D
  conduction through the base): Lee, Song, Au & Moran, "Constriction/
  spreading resistance model for electronics packaging", ASME/JSME
  Thermal Engineering Conf., 1995.

Limitations are reported, not hidden: every result carries ``warnings``
when a correlation is used outside its stated range or an assumption
(ducted flow, laminar channels, no radiation) may not hold.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List

from .fin_array_conduction import fin_array_profile_params

G = 9.80665

# Incropera & DeWitt Table A.4 (air, 1 atm): T [C], k [W/mK], nu [m2/s], Pr
_AIR_T = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0)
_AIR_K = (0.0243, 0.0257, 0.0271, 0.0285, 0.0299, 0.0314, 0.0349)
_AIR_NU = (13.3e-6, 15.1e-6, 16.97e-6, 18.9e-6, 20.9e-6, 23.0e-6, 28.5e-6)
_AIR_PR = (0.711, 0.709, 0.705, 0.702, 0.700, 0.697, 0.690)
AIR_CP = 1007.0

# Thermal conductivity [W/mK], density [kg/m3]
MATERIALS: Dict[str, Dict[str, float]] = {
    "al6063": {"k": 209.0, "rho": 2700.0, "label": "Aluminium 6063-T5 (extruded)"},
    "al6061": {"k": 167.0, "rho": 2700.0, "label": "Aluminium 6061-T6 (machined)"},
    "al1050": {"k": 229.0, "rho": 2705.0, "label": "Aluminium 1050 (bonded/skived fins)"},
    "cu110": {"k": 388.0, "rho": 8940.0, "label": "Copper C110"},
}


def _interp(x: float, xs, ys) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def air_properties(t_c: float, p_pa: float = 101325.0) -> Dict[str, float]:
    t_k = t_c + 273.15
    nu = _interp(t_c, _AIR_T, _AIR_NU)
    pr = _interp(t_c, _AIR_T, _AIR_PR)
    return {
        "k": _interp(t_c, _AIR_T, _AIR_K),
        "nu": nu,
        "pr": pr,
        "rho": p_pa / (287.05 * t_k),
        "cp": AIR_CP,
        "alpha": nu / pr,
        "beta": 1.0 / t_k,
    }


@dataclass
class HeatSinkGeometry:
    """All lengths in metres. Fins run along ``base_depth`` (the flow /
    vertical direction) and are distributed across ``base_width``, with a
    fin at each edge."""
    base_width: float
    base_depth: float
    base_thickness: float
    n_fins: int
    fin_thickness: float
    fin_height: float
    material: str = "al6063"

    @property
    def fin_gap(self) -> float:
        return (self.base_width - self.n_fins * self.fin_thickness) / max(self.n_fins - 1, 1)

    def mass_kg(self) -> float:
        rho = MATERIALS[self.material]["rho"]
        vol = (self.base_width * self.base_depth * self.base_thickness
               + self.n_fins * self.fin_thickness * self.fin_height * self.base_depth)
        return rho * vol


@dataclass
class OperatingPoint:
    power_w: float
    t_ambient_c: float
    air_velocity_m_s: float          # approach velocity; 0 = natural convection
    source_width: float              # heat-source footprint on the base [m]
    source_depth: float
    tim_resistance_k_m2_w: float = 1.0e-5   # interface material R'' (thin grease ~ 1e-5)


def geometry_errors(g: HeatSinkGeometry, op: OperatingPoint) -> List[str]:
    errs = []
    if g.material not in MATERIALS:
        errs.append(f"unknown material '{g.material}' (use one of {sorted(MATERIALS)})")
    if g.n_fins < 2:
        errs.append("need at least 2 fins")
    if min(g.base_width, g.base_depth, g.base_thickness, g.fin_thickness, g.fin_height) <= 0:
        errs.append("all dimensions must be positive")
    elif g.fin_gap <= 0:
        errs.append("fins do not fit: n_fins x fin_thickness exceeds the base width")
    if op.power_w <= 0:
        errs.append("power must be positive")
    if op.source_width > g.base_width + 1e-12 or op.source_depth > g.base_depth + 1e-12:
        errs.append("heat source is larger than the base")
    if op.air_velocity_m_s < 0:
        errs.append("air velocity cannot be negative")
    return errs


# ── convection ───────────────────────────────────────────────────────────────

def forced_convection(g: HeatSinkGeometry, v_approach: float, air: Dict[str, float]) -> Dict[str, float]:
    b, H, L, N = g.fin_gap, g.fin_height, g.base_depth, g.n_fins
    open_area = (N - 1) * b * H
    v_ch = v_approach * g.base_width * H / open_area          # ducted: all flow in channels
    re_b = v_ch * b / air["nu"]
    re_star = re_b * b / L
    pr = air["pr"]
    nu_fd = re_star * pr / 2.0
    nu_dev = 0.664 * math.sqrt(re_star) * pr ** (1 / 3) * math.sqrt(1 + 3.65 / math.sqrt(re_star))
    nu_b = (nu_fd ** -3 + nu_dev ** -3) ** (-1 / 3)
    h = nu_b * air["k"] / b

    # pressure drop
    dh = 2 * b * H / (b + H)
    re_dh = v_ch * dh / air["nu"]
    asp = min(b, H) / max(b, H)
    fre_fd = 24 * (1 - 1.3553 * asp + 1.9467 * asp ** 2 - 1.7012 * asp ** 3
                   + 0.9564 * asp ** 4 - 0.2537 * asp ** 5)
    l_plus = L / (dh * re_dh)
    f_app = math.sqrt((3.44 / math.sqrt(l_plus)) ** 2 + fre_fd ** 2) / re_dh
    sigma = open_area / (g.base_width * H)
    kc, ke = 0.42 * (1 - sigma ** 2), (1 - sigma ** 2) ** 2
    dp = (4 * f_app * L / dh + kc + ke) * air["rho"] * v_ch ** 2 / 2
    flow_m3_s = v_approach * g.base_width * H
    return {"h": h, "nu_b": nu_b, "re_star": re_star, "re_dh": re_dh, "v_channel": v_ch,
            "pressure_drop_pa": dp, "flow_m3_s": flow_m3_s, "fan_power_w": dp * flow_m3_s}


def natural_convection(g: HeatSinkGeometry, dt_k: float, air: Dict[str, float]) -> Dict[str, float]:
    b, L = g.fin_gap, g.base_depth
    el = G * air["beta"] * max(dt_k, 1e-6) * b ** 4 / (air["nu"] * air["alpha"] * L)
    nu_b = (576 / el ** 2 + 2.873 / math.sqrt(el)) ** -0.5
    return {"h": nu_b * air["k"] / b, "nu_b": nu_b, "elenbaas": el,
            "pressure_drop_pa": 0.0, "flow_m3_s": 0.0, "fan_power_w": 0.0}


# ── spreading ────────────────────────────────────────────────────────────────

def spreading_resistance(k: float, tb: float, source_area: float, plate_area: float,
                         r_conv: float) -> Dict[str, float]:
    """Lee et al. (1995). Includes 1D conduction through the base."""
    rs, rp = math.sqrt(source_area / math.pi), math.sqrt(plate_area / math.pi)
    eps, tau = rs / rp, tb / rp
    h_eff = 1.0 / (r_conv * plate_area)
    bi = h_eff * rp / k
    lam = math.pi + 1.0 / (math.sqrt(math.pi) * eps)
    th = math.tanh(lam * tau)
    phi = (th + lam / bi) / (1 + (lam / bi) * th)
    psi_max = eps * tau / math.sqrt(math.pi) + (1 - eps) * phi / math.sqrt(math.pi)
    psi_avg = eps * tau / math.sqrt(math.pi) + 0.5 * (1 - eps) ** 1.5 * phi
    denom = math.sqrt(math.pi) * k * rs
    return {"r_max": psi_max / denom, "r_avg": psi_avg / denom,
            "r_1d": tb / (k * plate_area), "epsilon": eps}


# ── full evaluation ──────────────────────────────────────────────────────────

def evaluate(g: HeatSinkGeometry, op: OperatingPoint, max_iter: int = 60,
             h_scale: float = 1.0, t_root_guess_c: float | None = None) -> Dict[str, object]:
    """Thermal resistance network: T_source = T_amb + Q (R_tim + R_spread + R_fins).

    ``h_scale`` multiplies the correlation's heat-transfer coefficient; use
    it to propagate correlation uncertainty (e.g. 0.85 / 1.15).
    ``t_root_guess_c`` is the fin-root temperature the natural-convection
    iteration starts from (default ambient + 20 K); with ``max_iter=1`` the
    coefficient is evaluated once at that temperature (a hand calculation)."""
    errs = geometry_errors(g, op)
    if errs:
        raise ValueError("; ".join(errs))
    mat = MATERIALS[g.material]
    k = mat["k"]
    Q, Ta = op.power_w, op.t_ambient_c
    natural = op.air_velocity_m_s <= 0
    a_src = op.source_width * op.source_depth
    a_base = g.base_width * g.base_depth

    t_root = Ta + 20.0 if t_root_guess_c is None else float(t_root_guess_c)
    conv: Dict[str, float] = {}
    fin: Dict[str, float] = {}
    converged = False
    for _ in range(max_iter):
        air = air_properties(0.5 * (t_root + Ta))
        conv = (natural_convection(g, t_root - Ta, air) if natural
                else forced_convection(g, op.air_velocity_m_s, air))
        conv["h"] *= h_scale
        fin = fin_array_profile_params(
            k_material_w_mk=k, h_conv_w_m2k=conv["h"], n_fins=g.n_fins,
            fin_thickness_m=g.fin_thickness, fin_length_m=g.fin_height,
            transverse_extent_m=g.base_depth, spacing_extent_m=g.base_width,
            base_thickness_m=g.base_thickness, power_w=Q,
        )
        new_root = Ta + Q * fin["r_conv"]
        if abs(new_root - t_root) < 1e-6:
            converged = True
            t_root = new_root
            break
        t_root = 0.5 * (t_root + new_root) if natural else new_root

    r_conv = fin["r_conv"]
    sp = spreading_resistance(k, g.base_thickness, a_src, a_base, r_conv)
    r_tim = op.tim_resistance_k_m2_w / a_src
    r_total = r_tim + sp["r_max"] + r_conv
    m = fin["m"]
    ml = m * fin["lc_m"]
    eta = math.tanh(ml) / ml if ml > 1e-9 else 1.0

    warnings: List[str] = []
    if not converged:
        warnings.append("Natural-convection iteration did not fully converge.")
    if natural:
        warnings.append("Radiation is neglected (conservative: real parts run ~10-25% cooler "
                        "in still air, more if anodised).")
        warnings.append("Assumes vertical fins with air rising along the base depth.")
        if not 1e-1 <= conv["elenbaas"] <= 1e4:
            warnings.append(f"Elenbaas number {conv['elenbaas']:.2g} outside 0.1-1e4.")
    else:
        warnings.append("Assumes ducted flow (shroud): all air passes between the fins. Without "
                        "a shroud, bypass can cut performance by 20-50%.")
        if not 0.26 <= conv["re_star"] <= 175:
            warnings.append(f"Channel Re* = {conv['re_star']:.3g} outside the Teertstra model's "
                            "validated range 0.26-175.")
        if conv["re_dh"] > 2300:
            warnings.append(f"Channel Reynolds {conv['re_dh']:.0f} > 2300: flow may be turbulent; "
                            "the laminar model is then conservative.")
    if eta < 0.5:
        warnings.append(f"Fin efficiency is low ({eta:.0%}): fins are too tall/thin for this "
                        "material -- extra height adds little.")
    if g.fin_gap < 1.5e-3:
        warnings.append(f"Fin gap {g.fin_gap * 1e3:.2f} mm is below ~1.5 mm: hard to extrude and "
                        "prone to dust clogging.")
    if g.fin_height / g.fin_gap > 20:
        warnings.append(f"Fin height/gap = {g.fin_height / g.fin_gap:.0f} > 20: beyond typical "
                        "extrusion limits (consider bonded or skived fins).")

    return {
        "mode": "natural" if natural else "forced",
        "t_source_max_c": Ta + Q * r_total,
        "t_base_max_c": Ta + Q * (sp["r_max"] + r_conv),
        "t_fin_root_c": Ta + Q * r_conv,
        "r_total_k_w": r_total,
        "resistances_k_w": {"interface_tim": r_tim, "spreading_and_base": sp["r_max"],
                            "fins_convection": r_conv},
        "h_w_m2k": conv["h"],
        "fin_efficiency": eta,
        "fin_m_per_m": m,                  # fin parameter sqrt(2h/(k t))
        "fin_lc_m": fin["lc_m"],           # adiabatic-tip corrected fin length
        "fin_gap_mm": g.fin_gap * 1e3,
        "mass_kg": g.mass_kg(),
        "pressure_drop_pa": conv["pressure_drop_pa"],
        "airflow_m3_h": conv["flow_m3_s"] * 3600,
        "fan_power_w": conv["fan_power_w"],
        "dimensionless": {k_: v for k_, v in conv.items()
                          if k_ in ("nu_b", "re_star", "re_dh", "elenbaas", "v_channel")},
        "spreading": sp,
        "material": mat["label"],
        "k_material": k,
        "warnings": warnings,
        "geometry": asdict(g),
        "operating": asdict(op),
    }
