"""Physics engine of the HeatSink Sizer: evaluates one design, fully verified.

Every number returned here comes from physics, not from the neural
surrogate: the closed-form network in
``pinneapple_physics.closed_form.plate_fin_heatsink`` (convection
correlations, fin efficiency, pressure drop) plus a 3D finite-volume solve
of the base (hotspot + temperature map). The two methods cross-check each
other, and each check is reported with a pass/warn/fail status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from pinneapple_physics.closed_form.plate_fin_heatsink import (
    MATERIALS, HeatSinkGeometry, OperatingPoint, evaluate, geometry_errors, spreading_resistance,
)

from .field import solve_base_field

# Typical accuracy of laminar plate-fin convection correlations against
# measurements; propagated as a +/- band on h. An engineering assumption,
# stated in the report -- not a measured uncertainty of this design.
H_BAND = 0.15
MM = 1e-3


@dataclass
class DesignInput:
    base_width_mm: float
    base_depth_mm: float
    base_thickness_mm: float
    n_fins: int
    fin_thickness_mm: float
    fin_height_mm: float
    material: str = "al6063"

    def geometry(self) -> HeatSinkGeometry:
        return HeatSinkGeometry(
            base_width=self.base_width_mm * MM, base_depth=self.base_depth_mm * MM,
            base_thickness=self.base_thickness_mm * MM, n_fins=int(self.n_fins),
            fin_thickness=self.fin_thickness_mm * MM, fin_height=self.fin_height_mm * MM,
            material=self.material,
        )


@dataclass
class OperatingInput:
    power_w: float
    t_ambient_c: float = 25.0
    air_velocity_m_s: float = 2.0          # 0 = natural convection
    source_width_mm: float = 30.0
    source_depth_mm: float = 30.0
    source_x_mm: Optional[float] = None    # centre of the source; default = centred
    source_y_mm: Optional[float] = None
    tim_k_mm2_w: float = 10.0              # interface resistance R'' in K*mm^2/W (grease ~5-20)
    t_limit_c: float = 85.0                # max allowed source/case temperature

    def operating(self) -> OperatingPoint:
        return OperatingPoint(
            power_w=self.power_w, t_ambient_c=self.t_ambient_c,
            air_velocity_m_s=self.air_velocity_m_s,
            source_width=self.source_width_mm * MM, source_depth=self.source_depth_mm * MM,
            tim_resistance_k_m2_w=self.tim_k_mm2_w * 1e-6,
        )


def _check(name: str, status: str, value: Any, detail: str, law: str = "") -> Dict[str, Any]:
    return {"name": name, "status": status, "value": value, "detail": detail, "law": law}


def base_theta(g: HeatSinkGeometry, op: OperatingInput, r_conv: float, n: int = 40,
               ) -> Dict[str, Any]:
    return solve_base_field(
        base_width=g.base_width, base_depth=g.base_depth, base_thickness=g.base_thickness,
        k=MATERIALS[g.material]["k"], power_w=op.power_w, r_fins_k_w=r_conv,
        source_width=op.source_width_mm * MM, source_depth=op.source_depth_mm * MM,
        source_x=None if op.source_x_mm is None else op.source_x_mm * MM,
        source_y=None if op.source_y_mm is None else op.source_y_mm * MM,
        nx=n, ny=n,
    )


def physics_resistance(design: DesignInput, op: OperatingInput, n: int = 36) -> Dict[str, float]:
    """Base-to-ambient resistance (no TIM) from the FVM-resolved hotspot, plus
    pressure drop -- the quantity the surrogate is trained to reproduce."""
    g, o = design.geometry(), op.operating()
    net = evaluate(g, o)
    f = base_theta(g, op, net["resistances_k_w"]["fins_convection"], n=n)
    return {"r_base_k_w": f["theta_max"] / op.power_w, "pressure_drop_pa": net["pressure_drop_pa"],
            "mass_kg": net["mass_kg"]}


def field_3d(g: HeatSinkGeometry, op: OperatingInput, fvm: Dict[str, Any],
             nominal: Dict[str, Any], t_amb: float, max_cells: int = 64, n_z: int = 14,
             ) -> Dict[str, Any]:
    """Surface temperatures of the whole heat sink for the 3D viewer.

    * Base: bottom and top faces straight from the 3D finite-volume solve.
    * Fins: each fin root takes the local top-face temperature under it
      (along the flow direction), and the temperature up the fin follows
      the 1D fin equation used by the resistance network,
      theta(z)/theta_root = cosh(m (Lc - z)) / cosh(m Lc)
      (Incropera, adiabatic tip via the corrected length Lc).
    All lengths in mm, temperatures in °C.
    """
    def pick(n):
        return np.linspace(0, n - 1, min(max_cells, n)).round().astype(int)

    iy, ix = pick(fvm["theta_bottom"].shape[0]), pick(fvm["theta_bottom"].shape[1])
    xs, ys = fvm["x_centers"][ix] / MM, fvm["y_centers"][iy] / MM
    top = fvm["theta_top"][np.ix_(iy, ix)]
    bottom = fvm["theta_bottom"][np.ix_(iy, ix)]

    t, gap = g.fin_thickness / MM, g.fin_gap / MM
    x_fins = t / 2 + np.arange(g.n_fins) * (t + gap)
    root = np.stack([[np.interp(xf, xs, row) for row in top] for xf in x_fins])  # (n_fins, ny)

    m, lc = nominal["fin_m_per_m"], nominal["fin_lc_m"]
    z = np.linspace(0.0, g.fin_height, n_z)
    ratio = np.cosh(m * (lc - z)) / np.cosh(m * lc) if m * lc > 1e-9 else np.ones_like(z)
    r2 = lambda a: np.round(a, 3).tolist()
    return {
        "t_ambient_c": t_amb,
        "base": {"width": g.base_width / MM, "depth": g.base_depth / MM,
                 "thickness": g.base_thickness / MM, "x": r2(xs), "y": r2(ys),
                 "bottom_c": r2(bottom + t_amb), "top_c": r2(top + t_amb)},
        "fins": {"x_centers": r2(x_fins), "thickness": t, "height": g.fin_height / MM,
                 "root_c": r2(root + t_amb)},
        "fin_profile": {"z": r2(z / MM), "theta_ratio": [float(v) for v in ratio]},
        "hotspot": {"x": fvm["hotspot_xy_m"][0] / MM, "y": fvm["hotspot_xy_m"][1] / MM,
                    "t_c": float(fvm["theta_max"] + t_amb)},
        "source": {"w": op.source_width_mm, "d": op.source_depth_mm,
                   "x": op.source_x_mm if op.source_x_mm is not None else g.base_width / MM / 2,
                   "y": op.source_y_mm if op.source_y_mm is not None else g.base_depth / MM / 2},
        "flow": {"mode": nominal["mode"], "velocity_m_s": op.air_velocity_m_s,
                 "direction": "+depth" if nominal["mode"] == "forced" else "up (vertical fins)"},
        "method": "Base: 3D finite volumes. Fins: 1D fin equation from the local root "
                  "temperature (Incropera). Surface temperatures; air not shown.",
    }


# Model assumptions are reported once, in the structured "scope" block, not as warnings.
_SCOPE_WARNINGS = ("Radiation is neglected", "Assumes vertical fins", "Assumes ducted flow")

VALIDATION = [
    "Base conduction: 3D finite volumes vs. Lee et al. (1995) spreading theory, 1-5 %",
    "Forced convection: vs. an independent method (Stephan + ε-NTU), 0-5 % at 0.5-6 m/s",
    "Natural convection: vs. Elenbaas (1942), 2 %; optimum fin gap vs. Bar-Cohen, within 6 %",
    "Pressure drop: vs. developing-flow friction with entry/exit losses, 2-4 %",
    "Energy balance of every solve: better than 1e-12",
]


def model_scope(mode: str) -> Dict[str, Any]:
    """What the model covers, which way each simplification errs, what to do
    about it today and what is planned -- shown in every report."""
    items = []
    if mode == "natural":
        items.append({
            "topic": "Thermal radiation", "effect": "conservative",
            "detail": "Only convection is counted. Radiation removes an extra 10-25 % of the heat "
                      "in still air (more with a black-anodised finish).",
            "today": "Treat the result as an upper bound on temperature.",
            "planned": "Radiation from the fin envelope with finish-dependent emissivity."})
        items.append({
            "topic": "Mounting orientation", "effect": "check",
            "detail": "Fins vertical, air rising along the base depth -- the orientation natural-"
                      "convection sinks are designed for.",
            "today": "Horizontal or fins-across-gravity mounting runs hotter; keep extra margin.",
            "planned": "Horizontal and inclined orientations."})
    else:
        items.append({
            "topic": "Air bypass", "effect": "optimistic",
            "detail": "All air passes between the fins, as with a shroud or a duct. In open flow "
                      "part of the air goes around the sink.",
            "today": "Use a shroud, or enter the air speed expected between the fins rather than "
                     "the fan's free-stream speed.",
            "planned": "Bypass model for unducted sinks (flow split by fin density)."})
    items.append({
        "topic": "Heat source", "effect": "check",
        "detail": "Uniform heat flux over the footprint you enter, with the interface material "
                  "resistance you enter.",
        "today": "For a die smaller than its package lid, enter the die size as the footprint.",
        "planned": "Multiple sources and non-uniform power maps."})
    items.append({
        "topic": "Steady state", "effect": "check",
        "detail": "Continuous power. Short bursts run cooler than shown.",
        "today": "Use the average power for duty-cycled loads, the peak for sign-off.",
        "planned": "Transient response to power profiles."})
    return {"validated": VALIDATION, "items": items,
            "band": f"±{int(H_BAND * 100)} % on the convection coefficient (typical correlation "
                    "accuracy) is already included in the verdict."}


def evaluate_design(design: DesignInput, op: OperatingInput, *, grid: int = 48,
                    map_size: int = 40) -> Dict[str, Any]:
    g, o = design.geometry(), op.operating()
    errs = geometry_errors(g, o)
    if errs:
        raise ValueError("; ".join(errs))

    nominal = evaluate(g, o)
    low_h = evaluate(g, o, h_scale=1 - H_BAND)    # pessimistic
    high_h = evaluate(g, o, h_scale=1 + H_BAND)   # optimistic
    Q, Ta = op.power_w, op.t_ambient_c
    r_conv = nominal["resistances_k_w"]["fins_convection"]
    r_tim = nominal["resistances_k_w"]["interface_tim"]

    fvm = base_theta(g, op, r_conv, n=grid)
    fvm_coarse = base_theta(g, op, r_conv, n=max(16, int(grid * 0.6)))
    theta = fvm["theta_max"]
    t_max = Ta + theta + Q * r_tim

    def shifted(res):  # same FVM spreading, band on the convection/spreading terms
        k = MATERIALS[g.material]["k"]
        a_src, a_base = o.source_width * o.source_depth, g.base_width * g.base_depth
        rc = res["resistances_k_w"]["fins_convection"]
        dsp = (spreading_resistance(k, g.base_thickness, a_src, a_base, rc)["r_max"]
               - nominal["spreading"]["r_max"])
        return t_max + Q * (rc - r_conv) + Q * dsp

    t_hi, t_lo = shifted(low_h), shifted(high_h)
    r_sp_fvm = theta / Q - r_conv

    # ── checks ──────────────────────────────────────────────────────────────
    checks: List[Dict[str, Any]] = []
    eb = fvm["energy_balance_rel_error"]
    checks.append(_check("energy_balance", "pass" if eb < 1e-6 else "fail", eb,
                         "Heat leaving through the fins / heat injected by the source - 1 (FVM).",
                         "energy conservation"))
    lee = nominal["spreading"]["r_max"]
    agree = abs(r_sp_fvm - lee) / max(lee, 1e-12)
    aspect = max(g.base_width / g.base_depth, g.base_depth / g.base_width,
                 op.source_width_mm / op.source_depth_mm, op.source_depth_mm / op.source_width_mm)
    if aspect > 2:
        checks.append(_check(
            "spreading_methods_agree", "n/a", agree,
            f"Aspect ratio {aspect:.1f} > 2: the circular-equivalent closed form (Lee et al.) is "
            f"not applicable ({lee:.4f} K/W); the 3D finite-volume solve of the real rectangle "
            f"({r_sp_fvm:.4f} K/W) is used.", "conduction in the base"))
    else:
        checks.append(_check(
            "spreading_methods_agree", "pass" if agree < 0.10 else "warn" if agree < 0.25 else "fail",
            agree, f"3D finite-volume base solve vs Lee et al. closed form: {r_sp_fvm:.4f} vs "
                   f"{lee:.4f} K/W (independent methods).", "conduction in the base"))
    grid_err = abs(fvm_coarse["theta_max"] - theta) / theta
    checks.append(_check("grid_convergence", "pass" if grid_err < 0.02 else "warn", grid_err,
                         "Hotspot change between the coarse and fine finite-volume grids.",
                         "numerical accuracy"))
    val_warn = [w for w in nominal["warnings"] if "outside" in w or "turbulent" in w]
    checks.append(_check("correlation_in_range", "warn" if val_warn else "pass", None,
                         val_warn[0] if val_warn else "Convection correlation used inside its "
                         "validated range.", "convection model validity"))
    checks.append(_check("temperature_above_ambient", "pass" if theta > 0 else "fail", theta,
                         "Hotspot rise above ambient must be positive (2nd law).", "2nd law"))

    margin = op.t_limit_c - t_hi
    if t_hi <= op.t_limit_c:
        verdict = {"status": "meets", "text": f"Meets the {op.t_limit_c:.0f} °C limit even with "
                   f"pessimistic convection ({t_hi:.1f} °C)."}
    elif t_max <= op.t_limit_c:
        verdict = {"status": "marginal", "text": f"Nominal {t_max:.1f} °C meets the limit, but the "
                   f"pessimistic case ({t_hi:.1f} °C) does not -- add margin."}
    else:
        verdict = {"status": "fails", "text": f"Exceeds the {op.t_limit_c:.0f} °C limit "
                   f"({t_max:.1f} °C nominal)."}

    # temperature map (downsampled for the UI)
    tb = fvm["theta_bottom"] + Ta
    iy = np.linspace(0, tb.shape[0] - 1, min(map_size, tb.shape[0])).round().astype(int)
    ix = np.linspace(0, tb.shape[1] - 1, min(map_size, tb.shape[1])).round().astype(int)
    tmap = tb[np.ix_(iy, ix)]

    return {
        "verdict": verdict,
        "kpis": {
            "t_source_max_c": t_max,
            "t_source_band_c": [t_lo, t_hi],
            "margin_to_limit_c": margin,
            "t_limit_c": op.t_limit_c,
            "r_total_k_w": (t_max - Ta) / Q,
            "pressure_drop_pa": nominal["pressure_drop_pa"],
            "airflow_m3_h": nominal["airflow_m3_h"],
            "fan_power_w": nominal["fan_power_w"],
            "mass_g": nominal["mass_kg"] * 1e3,
            "fin_efficiency": nominal["fin_efficiency"],
            "h_w_m2k": nominal["h_w_m2k"],
            "fin_gap_mm": nominal["fin_gap_mm"],
            "mode": nominal["mode"],
        },
        "resistances_k_w": {"interface_tim": r_tim, "spreading_and_base": r_sp_fvm,
                            "fins_convection": r_conv},
        "temperature_map": {
            "values_c": tmap.round(2).tolist(),
            "width_mm": design.base_width_mm, "depth_mm": design.base_depth_mm,
            "hotspot_mm": [v / MM for v in fvm["hotspot_xy_m"]],
            "source": {"w": op.source_width_mm, "d": op.source_depth_mm,
                       "x": op.source_x_mm if op.source_x_mm is not None else design.base_width_mm / 2,
                       "y": op.source_y_mm if op.source_y_mm is not None else design.base_depth_mm / 2},
            "note": "Bottom face of the base (heat-source side), from the 3D finite-volume solve.",
        },
        "field3d": field_3d(g, op, fvm, nominal, Ta),
        "checks": checks,
        "warnings": [w for w in nominal["warnings"] if not w.startswith(_SCOPE_WARNINGS)],
        "scope": model_scope(nominal["mode"]),
        "details": {"dimensionless": nominal["dimensionless"], "material": nominal["material"],
                    "k_material_w_mk": nominal["k_material"],
                    "fvm_grid": fvm["grid"], "h_uncertainty_band": H_BAND},
        "method": ("Resistance network with published correlations (Teertstra 2000 forced / "
                   "Bar-Cohen & Rohsenow 1984 natural), fin efficiency (Incropera), pressure drop "
                   "(Muzychka & Yovanovich), base conduction by 3D finite volumes cross-checked "
                   "against Lee et al. 1995."),
        "inputs": {"design": design.__dict__, "operating": op.__dict__},
    }
