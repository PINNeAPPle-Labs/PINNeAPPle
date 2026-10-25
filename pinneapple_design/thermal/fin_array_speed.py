"""How much faster is a trained surrogate than a numerical solve, and how far apart are they? A measured sweep over
seven plate-fin heat sinks.

The heat sink, its closed form and its surrogate are those of
:mod:`pinneapple_design.surrogate_inverse.heatsink_fins` (80 x 80 mm aluminium base, 3 mm thick, straight fins,
uniform film coefficient; MLP on the thermal resistance, trained on the fin-array formula of
:mod:`pinneapple_physics.closed_form.fin_array_conduction`). The numerical reference is the finite-volume fin pitch
of :func:`pinneapple_design.thermal.fin_cell.solve_fin_cell` (2-D cross-section, which is the same problem as the
3-D periodic cell because nothing depends on the fin depth), with its grid-halving check.

For each design the surrogate forward pass is timed (mean of ``timing_reps`` calls after one warm-up call) and the
reference solve is timed (wall clock, including its grid check), and the agreement is the difference of the
temperature rise above ambient, ``|dT_surrogate - dT_ref| / dT_ref``. The speed ratio depends on the machine and on
the reference solver: this reference is a sub-second 2-D solve, so the ratios are much smaller than against a meshed
3-D FEM run.

The seven designs stay at 55 W or more and inside the manufacturable box of ``heatsink_fins``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from pinneapple_design.surrogate_inverse import heatsink_fins as hf

__all__ = ["CURATED_DESIGNS", "AGREEMENT_THRESHOLD_PCT", "design_by_name", "design_of", "train_surrogate",
           "time_surrogate", "reference_solve", "measure_design", "summarize", "run_sweep", "thermal_field"]

AGREEMENT_THRESHOLD_PCT = 10.0       # decision bound on the surrogate vs reference difference (temperature rise)

CURATED_DESIGNS: list[dict[str, Any]] = [
    {"name": "baseline", "label": "Baseline", "short": "Baseline", "fin_count": 20, "fin_height_m": 0.025,
     "fin_thickness_m": 0.0010, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "dense_thin_fins", "label": "Dense, thin fins", "short": "Dense fins", "fin_count": 44,
     "fin_height_m": 0.020, "fin_thickness_m": 0.0006, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "sparse_thick_fins", "label": "Sparse, thick fins", "short": "Sparse fins", "fin_count": 8,
     "fin_height_m": 0.030, "fin_thickness_m": 0.0025, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "tall_fins", "label": "Tall fins", "short": "Tall fins", "fin_count": 15, "fin_height_m": 0.035,
     "fin_thickness_m": 0.0010, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "short_fins", "label": "Short, dense fins", "short": "Short fins", "fin_count": 25,
     "fin_height_m": 0.008, "fin_thickness_m": 0.0012, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "high_power_hs_fan", "label": "High power + strong fan", "short": "High power", "fin_count": 20,
     "fin_height_m": 0.025, "fin_thickness_m": 0.0010, "power_w": 130.0, "cooling_profile": "high_static_pressure_fan"},
    {"name": "lower_power_quiet", "label": "Lower power + quiet fan", "short": "Low power", "fin_count": 20,
     "fin_height_m": 0.025, "fin_thickness_m": 0.0010, "power_w": 55.0, "cooling_profile": "quiet_fan"},
]


def design_by_name(name: str) -> dict[str, Any]:
    """One of ``CURATED_DESIGNS`` (KeyError otherwise)."""
    for spec in CURATED_DESIGNS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def design_of(spec: dict[str, Any]) -> hf.HeatsinkDesign:
    return hf.HeatsinkDesign(float(spec["fin_count"]), spec["fin_height_m"], spec["fin_thickness_m"])


def train_surrogate(n_samples: int = 6000, epochs: int = 1500, seed: int = 0):
    """``heatsink_fins.train_surrogate``; returns ``(model, val_mean_rel_err)`` (held-out error on the thermal
    resistance, i.e. on the temperature rise)."""
    fit = hf.train_surrogate(n_samples=n_samples, epochs=epochs, seed=seed)
    return fit.model, fit.val_mean_rel_err


def time_surrogate(model, d: hf.HeatsinkDesign, power_w: float, h_conv: float, reps: int = 200) -> dict[str, float]:
    """Mean wall time of one forward pass (``reps`` calls after a warm-up call) and the predicted temperature."""
    x = torch.tensor([[d.fin_count, d.fin_height_m, d.fin_thickness_m, h_conv]], dtype=torch.float32)
    with torch.no_grad():
        model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            r = model(x)
        dt = time.perf_counter() - t0
    return {"predicted_c": hf.T_AMBIENT_C + power_w * float(r[0]), "time_s": dt / reps}


def reference_solve(d: hf.HeatsinkDesign, power_w: float, h_conv: float, refine: bool = True) -> dict[str, Any]:
    """Finite-volume fin pitch (``pinneapple_design.thermal.fin_cell``), timed end to end."""
    from pinneapple_design.thermal.fin_cell import solve_fin_cell

    t0 = time.perf_counter()
    r = solve_fin_cell(n_fins=max(round(d.fin_count), 1), fin_thickness_m=d.fin_thickness_m,
                       fin_height_m=d.fin_height_m, fin_spacing_m=d.fin_spacing_m(), depth_m=hf.BASE_D_M,
                       base_thickness_m=hf.BASE_THICKNESS_M, k_w_mk=hf.K_ALUMINUM, h_w_m2k=h_conv, power_w=power_w,
                       t_ambient_c=hf.T_AMBIENT_C, refine=refine)
    return {"junction_c": r.t_junction_c, "wall_time_s": time.perf_counter() - t0, "n_cells": r.n_cells,
            "energy_balance_rel_error": r.energy_balance_rel_error, "grid_change_rel": r.grid_change_rel}


def _rise_diff_pct(a_c: float, ref_c: float) -> float:
    t = hf.T_AMBIENT_C
    return abs((a_c - t) - (ref_c - t)) / max(abs(ref_c - t), 1e-6) * 100.0


def measure_design(model, spec: dict[str, Any], timing_reps: int = 200, refine: bool = True) -> dict[str, Any]:
    """Closed form, timed surrogate and timed reference for one design."""
    d = design_of(spec)
    power, h = spec["power_w"], hf.COOLING_PROFILES[spec["cooling_profile"]]
    t0 = time.perf_counter()
    analytical = hf.junction_temperature_c(d, power, h)
    analytical_s = time.perf_counter() - t0
    s = time_surrogate(model, d, power, h, timing_reps)
    ref = reference_solve(d, power, h, refine)
    return {"name": spec["name"], "label": spec["label"], "short": spec["short"],
            "design": {"fin_count": d.fin_count, "fin_height_mm": d.fin_height_m * 1e3,
                       "fin_thickness_mm": d.fin_thickness_m * 1e3, "fin_spacing_mm": d.fin_spacing_m() * 1e3},
            "power_w": power, "cooling_profile": spec["cooling_profile"], "h_conv": h,
            "constraints_ok": all(v <= 0 for v in hf.constraint_violations(d).values()),
            "analytical_c": analytical, "analytical_time_s": analytical_s,
            "surrogate_c": s["predicted_c"], "surrogate_time_s": s["time_s"],
            "fea_c": ref["junction_c"], "fea_solve_time_s": ref["wall_time_s"], "fea_wall_time_s": ref["wall_time_s"],
            "fea_num_cells": ref["n_cells"], "fea_heat_balance_error": ref["energy_balance_rel_error"],
            "fea_grid_change": ref["grid_change_rel"],
            "surrogate_vs_fea_diff_pct": _rise_diff_pct(s["predicted_c"], ref["junction_c"]),
            "analytical_vs_fea_diff_pct": _rise_diff_pct(analytical, ref["junction_c"]),
            "surrogate_vs_fea_abs_error_c": abs(s["predicted_c"] - ref["junction_c"]),
            "speedup_factor": ref["wall_time_s"] / s["time_s"] if s["time_s"] > 0 else float("inf")}


def summarize(designs: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = [d["surrogate_vs_fea_diff_pct"] for d in designs]
    sp = [d["speedup_factor"] for d in designs]
    ref_t = sum(d["fea_wall_time_s"] for d in designs)
    sur_t = sum(d["surrogate_time_s"] for d in designs)
    ae = [d["surrogate_vs_fea_abs_error_c"] for d in designs]
    return {"n_designs": len(designs), "mean_speedup_factor": float(np.mean(sp)), "min_speedup_factor": min(sp),
            "max_speedup_factor": max(sp), "aggregate_speedup_factor": ref_t / sur_t if sur_t > 0 else float("inf"),
            "mean_agreement_diff_pct": float(np.mean(diffs)), "max_agreement_diff_pct": max(diffs),
            "agreement_threshold_pct": AGREEMENT_THRESHOLD_PCT,
            "all_within_threshold": max(diffs) <= AGREEMENT_THRESHOLD_PCT,
            "total_fea_wall_time_s": ref_t, "total_surrogate_time_s": sur_t, "mean_abs_error_c": float(np.mean(ae)),
            "max_abs_error_c": max(ae),
            "max_analytical_vs_fea_diff_pct": max(d["analytical_vs_fea_diff_pct"] for d in designs)}


def run_sweep(model, progress_cb: Callable[[int, int], None] | None = None, timing_reps: int = 200,
              refine: bool = True) -> dict[str, Any]:
    """Measure every design of ``CURATED_DESIGNS``; ``progress_cb(i, n)`` after each."""
    out = []
    for i, spec in enumerate(CURATED_DESIGNS):
        out.append(measure_design(model, spec, timing_reps, refine))
        if progress_cb is not None:
            progress_cb(i + 1, len(CURATED_DESIGNS))
    return {"designs": out, "summary": summarize(out)}


def thermal_field(name: str = "baseline") -> dict[str, Any]:
    """``heatsink_fins.thermal_field`` of one design, plus its base temperature."""
    spec = design_by_name(name)
    d, h = design_of(spec), hf.COOLING_PROFILES[spec["cooling_profile"]]
    return {**hf.thermal_field(d, spec["power_w"], h),
            "base_temperature_c": hf.junction_temperature_c(d, spec["power_w"], h)}
