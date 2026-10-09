"""Sizing: requirements in, verified heat-sink designs out.

1. Sample tens of thousands of manufacturable candidates inside the
   customer's envelope and process rules.
2. Screen them: with the PINNeAPPle surrogate (FVM-accurate, microseconds
   per design) when the request is inside its trained design space, else
   with the closed-form network (vectorised loop, still fast). The screen
   uses the surrogate's conformal 95% upper bound, so it errs on the safe
   side.
3. Round the best candidates to manufacturing precision and re-verify each
   one with the full physics engine (FVM base + pessimistic convection
   band). Only verified designs are recommended.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from pinneapple_physics.closed_form.plate_fin_heatsink import (
    MATERIALS, HeatSinkGeometry, OperatingPoint, evaluate,
)

from .engine import H_BAND, DesignInput, OperatingInput, evaluate_design
from .surrogate import COMMON_RANGES, MODE_RANGES, features, n_fins_for

PROCESSES = {
    "extruded": {"label": "Extruded aluminium", "t_min": 1.0, "gap_min": 1.5, "aspect_max": 20.0,
                 "tb_min": 3.0},
    "bonded": {"label": "Bonded / skived fins", "t_min": 0.6, "gap_min": 1.0, "aspect_max": 40.0,
               "tb_min": 3.0},
}


@dataclass
class SizingRequest:
    power_w: float
    t_limit_c: float
    t_ambient_c: float = 25.0
    air_velocity_m_s: float = 2.0
    source_width_mm: float = 30.0
    source_depth_mm: float = 30.0
    tim_k_mm2_w: float = 10.0
    max_width_mm: float = 100.0
    max_depth_mm: float = 100.0
    max_height_mm: float = 50.0            # base + fins
    materials: List[str] = field(default_factory=lambda: ["al6063"])
    process: str = "extruded"
    max_pressure_drop_pa: Optional[float] = None
    objective: str = "mass"                 # mass | volume | temperature
    n_candidates: int = 40000
    n_verify: int = 24
    n_results: int = 5
    seed: int = 0


def _round_design(W, D, tb, N, t, H, mat) -> DesignInput:
    """Manufacturing precision: 1 mm envelope, 0.5 mm base, 0.1 mm fins."""
    return DesignInput(float(round(float(W))), float(round(float(D))), round(float(tb) * 2) / 2,
                       int(N), round(float(t), 1), float(round(float(H))), str(mat))


def _surrogate_domain_issues(req: SizingRequest, mode: str) -> List[str]:
    ranges = {**COMMON_RANGES, **MODE_RANGES[mode]}
    issues = []
    lo, hi = ranges["power_w"]
    if not lo <= req.power_w <= hi:
        issues.append(f"power {req.power_w} W outside {lo}-{hi} W")
    lo, hi = ranges["t_ambient_c"]
    if not lo <= req.t_ambient_c <= hi:
        issues.append(f"ambient {req.t_ambient_c} °C outside {lo}-{hi} °C")
    if mode == "forced":
        lo, hi = ranges["air_velocity_m_s"]
        if not lo <= req.air_velocity_m_s <= hi:
            issues.append(f"air velocity {req.air_velocity_m_s} m/s outside {lo}-{hi} m/s")
    for m in req.materials:
        k = MATERIALS[m]["k"]
        if not ranges["k_w_mk"][0] <= k <= ranges["k_w_mk"][1]:
            issues.append(f"material {m} outside the trained conductivity range")
    return issues


def _sample(req: SizingRequest, mode: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    proc = PROCESSES[req.process]
    n = req.n_candidates
    wlo = max(req.source_width_mm, COMMON_RANGES["base_width_mm"][0])
    dlo = max(req.source_depth_mm, COMMON_RANGES["base_depth_mm"][0])
    # the surrogate was trained with source >= 15% of the base
    whi = min(req.max_width_mm, COMMON_RANGES["base_width_mm"][1], req.source_width_mm / 0.15)
    dhi = min(req.max_depth_mm, COMMON_RANGES["base_depth_mm"][1], req.source_depth_mm / 0.15)
    if whi < wlo or dhi < dlo:
        raise ValueError("The envelope cannot contain the heat source.")
    logu = lambda a, b, size: np.exp(rng.uniform(math.log(a), math.log(b), size))
    W, D = logu(wlo, whi, n), logu(dlo, dhi, n)
    tb = rng.uniform(proc["tb_min"], min(12.0, req.max_height_mm - 8.0), n)
    t = rng.uniform(max(proc["t_min"], COMMON_RANGES["fin_thickness_mm"][0]), 3.0, n)
    glo, ghi = MODE_RANGES[mode]["fin_gap_mm"]
    gap = logu(max(glo, proc["gap_min"]), ghi, n)
    h_hi = np.minimum(80.0, req.max_height_mm - tb)            # fins + base <= max height
    H = rng.uniform(8.0, np.maximum(h_hi, 8.0))
    N = n_fins_for(W, gap, t)
    real_gap = (W - N * t) / np.maximum(N - 1, 1)
    mats = np.array(req.materials)[rng.integers(0, len(req.materials), n)]
    ok = (h_hi >= 8.0) & (real_gap >= proc["gap_min"]) & (H / np.maximum(real_gap, 1e-9) <= proc["aspect_max"])
    return {"W": W[ok], "D": D[ok], "tb": tb[ok], "t": t[ok], "H": H[ok], "N": N[ok], "mat": mats[ok]}


def _mass_g(c) -> np.ndarray:
    rho = np.array([MATERIALS[m]["rho"] for m in c["mat"]])
    vol_mm3 = c["W"] * c["D"] * c["tb"] + c["N"] * c["t"] * c["H"] * c["D"]
    return rho * vol_mm3 * 1e-9 * 1e3


def _screen_surrogate(req, mode, c, sur) -> Dict[str, np.ndarray]:
    k = np.array([MATERIALS[m]["k"] for m in c["mat"]])
    X = features(mode, c["W"], c["D"], c["tb"], c["N"], c["t"], c["H"], k,
                 np.full_like(c["W"], req.source_width_mm), np.full_like(c["W"], req.source_depth_mm),
                 np.full_like(c["W"], req.t_ambient_c), np.full_like(c["W"], req.power_w),
                 np.full_like(c["W"], max(req.air_velocity_m_s, 1e-6)))
    p = sur.predict(X)
    r95 = p["r_base_k_w"] * math.exp(sur.conformal_log_q["95"])
    out = {"r_upper": r95, "r_pred": p["r_base_k_w"]}
    if "pressure_drop_pa" in p:
        out["dp"] = p["pressure_drop_pa"]
    return out


def _screen_closed_form(req, c, limit: int = 6000) -> Dict[str, np.ndarray]:
    n = min(len(c["W"]), limit)
    for key in c:
        c[key] = c[key][:n]
    r, dp = np.empty(n), np.zeros(n)
    op = OperatingPoint(req.power_w, req.t_ambient_c, req.air_velocity_m_s,
                        req.source_width_mm * 1e-3, req.source_depth_mm * 1e-3, req.tim_k_mm2_w * 1e-6)
    for i in range(n):
        g = HeatSinkGeometry(c["W"][i] * 1e-3, c["D"][i] * 1e-3, c["tb"][i] * 1e-3, int(c["N"][i]),
                             c["t"][i] * 1e-3, c["H"][i] * 1e-3, str(c["mat"][i]))
        res = evaluate(g, op)
        r[i] = res["r_total_k_w"] - res["resistances_k_w"]["interface_tim"]
        dp[i] = res["pressure_drop_pa"]
    # the circular-equivalent spreading term can under-predict elongated bases
    return {"r_upper": r * 1.15, "r_pred": r, "dp": dp}


def size(req: SizingRequest, surrogates: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    if req.process not in PROCESSES:
        raise ValueError(f"unknown process '{req.process}'")
    bad = [m for m in req.materials if m not in MATERIALS]
    if bad or not req.materials:
        raise ValueError(f"unknown material(s) {bad}")
    if req.t_limit_c <= req.t_ambient_c:
        raise ValueError("the temperature limit must be above ambient")
    mode = "natural" if req.air_velocity_m_s <= 0 else "forced"
    rng = np.random.default_rng(req.seed)
    c = _sample(req, mode, rng)
    if len(c["W"]) == 0:
        raise ValueError("No manufacturable design fits this envelope and process.")

    issues = _surrogate_domain_issues(req, mode)
    sur = surrogates.get(mode)
    if sur is not None and not issues:
        s = _screen_surrogate(req, mode, c, sur)
        screen = {"method": f"PINNeAPPle neural surrogate ({mode}), conformal 95% upper bound",
                  "trust": sur.report["verdict"]["status"]}
    else:
        s = _screen_closed_form(req, c)
        why = issues or ["no trained surrogate available"]
        screen = {"method": "closed-form network (surrogate not used: " + "; ".join(why) + ")",
                  "trust": "physics"}
    n_screened = len(s["r_pred"])
    for key in c:
        c[key] = c[key][:n_screened]

    r_tim = req.tim_k_mm2_w / (req.source_width_mm * req.source_depth_mm)
    # Verification requires the pessimistic convection case (h - H_BAND) to meet
    # the limit, so screen against it too (applied to the whole resistance:
    # slightly conservative, since the spreading part barely depends on h).
    t_upper = req.t_ambient_c + req.power_w * (r_tim + s["r_upper"] / (1 - H_BAND))
    mass = _mass_g(c)
    volume = c["W"] * c["D"] * (c["tb"] + c["H"]) * 1e-3   # cm3
    feasible = t_upper <= req.t_limit_c
    if req.max_pressure_drop_pa is not None and "dp" in s:
        feasible &= s["dp"] <= req.max_pressure_drop_pa
    obj = {"mass": mass, "volume": volume, "temperature": t_upper}[req.objective]

    order = np.argsort(np.where(feasible, obj, np.inf))
    if not feasible.any():
        order = np.argsort(t_upper)           # nothing passes: show the coolest designs
    picked, seen = [], []
    for i in order:
        key = np.array([c["W"][i], c["D"][i], c["H"][i], c["N"][i]])
        if any(np.all(np.abs(key - k) <= 0.06 * np.abs(k) + 1) for k in seen):
            continue
        seen.append(key)
        picked.append(i)
        if len(picked) >= req.n_verify:
            break

    op = OperatingInput(power_w=req.power_w, t_ambient_c=req.t_ambient_c,
                        air_velocity_m_s=req.air_velocity_m_s, source_width_mm=req.source_width_mm,
                        source_depth_mm=req.source_depth_mm, tim_k_mm2_w=req.tim_k_mm2_w,
                        t_limit_c=req.t_limit_c)
    verified = []
    for i in picked:
        d = _round_design(c["W"][i], c["D"][i], c["tb"][i], c["N"][i], c["t"][i], c["H"][i], c["mat"][i])
        try:
            res = evaluate_design(d, op, grid=40, map_size=32)
        except ValueError:
            continue
        k = res["kpis"]
        dp_ok = req.max_pressure_drop_pa is None or k["pressure_drop_pa"] <= req.max_pressure_drop_pa
        predicted_t = req.t_ambient_c + req.power_w * (r_tim + s["r_pred"][i])
        verified.append({
            "design": asdict(d),
            "meets_requirements": res["verdict"]["status"] == "meets" and dp_ok,
            "verdict": res["verdict"],
            "kpis": k,
            "volume_cm3": d.base_width_mm * d.base_depth_mm * (d.base_thickness_mm + d.fin_height_mm) / 1e3,
            "screen_vs_physics_c": float(predicted_t - k["t_source_max_c"]),
            "checks_ok": all(ch["status"] in ("pass", "n/a") for ch in res["checks"]),
            "warnings": res["warnings"],
        })

    key = {"mass": lambda v: v["kpis"]["mass_g"], "volume": lambda v: v["volume_cm3"],
           "temperature": lambda v: v["kpis"]["t_source_band_c"][1]}[req.objective]
    good = sorted([v for v in verified if v["meets_requirements"]], key=key)
    diffs = [abs(v["screen_vs_physics_c"]) for v in verified]
    result = {
        "status": "ok" if good else "no_feasible_design",
        "message": (f"{len(good)} verified design(s) meet all requirements." if good else
                    "No design in this envelope meets the limit. Closest options are shown: relax "
                    "the envelope, raise the airflow or use a more conductive material."),
        "recommendations": (good or sorted(verified, key=lambda v: v["kpis"]["t_source_band_c"][1]))
        [: req.n_results],
        "search": {
            "mode": mode, "candidates_sampled": int(req.n_candidates),
            "candidates_screened": int(n_screened), "passed_screen": int(feasible.sum()),
            "verified_with_physics": len(verified), "screening": screen,
            "screen_error_on_verified_c": {"mean_abs": float(np.mean(diffs)) if diffs else None,
                                           "max_abs": float(np.max(diffs)) if diffs else None},
            "process": PROCESSES[req.process]["label"],
            "seconds": time.time() - t0,
        },
        "request": asdict(req),
    }
    return result
