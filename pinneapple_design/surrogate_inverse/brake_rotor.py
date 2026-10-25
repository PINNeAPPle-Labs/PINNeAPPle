"""Vented brake rotor: the radial vanes between the two friction rings, as a fin array, under a sustained load.

Ground truth: the same 1-D fin-array formula as :mod:`.heatsink_fins` (:mod:`pinneapple_physics.closed_form.
fin_array_conduction`) with cast iron (``k = 50 W/m K``) and rotor-scale geometry: the vanes are fins of length =
vent gap spread along the mean circumference, the friction ring is the base (extended-surface analysis of vented
rotor cooling: e.g. Limpert, *Brake Design and Safety*). Lumped and steady: a uniform convection coefficient per
driving profile and a sustained heat input (a long descent), not a single hard stop (that is a transient problem; see
the ``car_brake_thermal`` PDE preset). The temperature rise is linear in power, so the surrogate learns the thermal
resistance.

The analysis of its own: :func:`monte_carlo_uncertainty` propagates machining tolerances on vane thickness and gap
through the formula (:func:`pinneapple_analysis.uncertainty.tolerance.propagate`, uniform within the band) and grades
the 95th-percentile temperature against the safe limit.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from pinneapple_analysis.uncertainty.tolerance import propagate
from pinneapple_physics.closed_form import fin_array_conduction as fac

from . import meshes
from .core import (
    FAIL,
    PASS,
    WARNING,
    CheckResult,
    SurrogateFit,
    VerificationCard,
    banded,
    constraint_checks,
    descend,
    fit_surrogate,
    freeze,
    relative_error,
)

K_CAST_IRON = 50.0             # W/(m K), grey cast iron of brake discs
T_AMBIENT_C = 25.0
MEAN_CIRCUMFERENCE_M = 0.754   # 2 pi R_mean of a ~300 mm rotor (R_mean = 0.12 m)
RADIAL_FLOW_LENGTH_M = 0.06    # R_outer - R_inner, the vane's radial extent
FRICTION_RING_THICKNESS_M = 0.006

VANE_GAP_BOUNDS_M = (0.003, 0.016)
VANE_THICKNESS_BOUNDS_M = (0.0015, 0.006)
VANE_COUNT_BOUNDS = (16, 80)
MIN_VANE_SPACING_M = 0.006     # design rule of this case: cast channels narrower than this are not made
POWER_BOUNDS_W = (150.0, 1600.0)

COOLING_PROFILES: dict[str, float] = {      # assumed lumped convection coefficients, W/(m^2 K)
    "city": 40.0,
    "highway": 120.0,
    "mountain_descent": 200.0,
}
H_SPREAD = (0.85, 1.15)
REL_ERROR_PASS = 0.05          # surrogate vs formula, on the temperature rise
REL_ERROR_WARN = 0.15
MAX_SAFE_ROTOR_TEMP_C = 500.0  # cast-iron rotors: fade / distortion risk beyond roughly 500-600 C (assumed limit)

# Machining tolerances: assumed order-of-magnitude bands for cast and lightly machined vanes, not supplier data.
VANE_THICKNESS_TOLERANCE_M = 0.05e-3
VANE_GAP_TOLERANCE_M = 0.1e-3
_FLOAT_TOL = 1e-6


@dataclass
class RotorDesign:
    vane_count: float
    vane_gap_m: float
    vane_thickness_m: float

    def vane_spacing_m(self) -> float:
        n = max(self.vane_count, 1.0)
        if n <= 1:
            return MEAN_CIRCUMFERENCE_M
        return max((MEAN_CIRCUMFERENCE_M - n * self.vane_thickness_m) / (n - 1), 1e-6)


BASELINE_DESIGN = RotorDesign(vane_count=36.0, vane_gap_m=0.008, vane_thickness_m=0.004)


def profile_params(design: RotorDesign, power_w: float, h_conv: float) -> dict[str, float]:
    """Formula quantities, plus ``theta_vane_base_c`` (alias of the fin-root rise) and ``lc_radial_m`` (corrected
    length along the vane's radial extent, for the radial colour profile)."""
    t = max(design.vane_thickness_m, 1e-6)
    p = fac.fin_array_profile_params(
        k_material_w_mk=K_CAST_IRON, h_conv_w_m2k=h_conv, n_fins=max(design.vane_count, 1e-6), fin_thickness_m=t,
        fin_length_m=max(design.vane_gap_m, 1e-6), transverse_extent_m=RADIAL_FLOW_LENGTH_M,
        spacing_extent_m=MEAN_CIRCUMFERENCE_M, base_thickness_m=FRICTION_RING_THICKNESS_M, power_w=power_w)
    p["theta_vane_base_c"] = p["theta_fin_base_c"]
    p["lc_radial_m"] = RADIAL_FLOW_LENGTH_M + t / 2.0
    return p


def rotor_temperature_c(design: RotorDesign, power_w: float, h_conv: float) -> float:
    """Ground truth: friction-ring temperature under a sustained heat input."""
    return fac.fin_array_base_temperature_c(profile_params(design, power_w, h_conv), T_AMBIENT_C, power_w)


def thermal_resistance_k_w(design: RotorDesign, h_conv: float) -> float:
    return rotor_temperature_c(design, 1.0, h_conv) - T_AMBIENT_C


def vane_temperature_at_radial_position_c(design: RotorDesign, power_w: float, h_conv: float,
                                          r_fraction: float) -> float:
    """The fin profile (same ``m`` and root rise) along the vane's radial extent: hottest at the inner edge
    (``r_fraction = 0``), coolest at the outer edge. Used to colour the vanes; the scalar model integrates over the
    gap direction, this profile is a display of the same fin parameters along the radius."""
    p = profile_params(design, power_w, h_conv)
    x = min(max(r_fraction, 0.0), 1.0) * RADIAL_FLOW_LENGTH_M
    return fac.fin_array_temperature_at_position_c(p, T_AMBIENT_C, x, corrected_length_m=p["lc_radial_m"])


def thermal_field(design: RotorDesign, power_w: float, h_conv: float) -> dict[str, float]:
    p = profile_params(design, power_w, h_conv)
    return {"t_ambient_c": T_AMBIENT_C, "theta_vane_base_c": p["theta_vane_base_c"], "m": p["m"],
            "lc_radial_m": p["lc_radial_m"]}


def constraint_violations(design: RotorDesign) -> dict[str, float]:
    raw = {
        "min_spacing": MIN_VANE_SPACING_M - design.vane_spacing_m(),
        "min_gap": VANE_GAP_BOUNDS_M[0] - design.vane_gap_m,
        "max_gap": design.vane_gap_m - VANE_GAP_BOUNDS_M[1],
        "min_thickness": VANE_THICKNESS_BOUNDS_M[0] - design.vane_thickness_m,
        "max_thickness": design.vane_thickness_m - VANE_THICKNESS_BOUNDS_M[1],
        "min_count": VANE_COUNT_BOUNDS[0] - design.vane_count,
        "max_count": design.vane_count - VANE_COUNT_BOUNDS[1],
    }
    return {k: (v if v > _FLOAT_TOL else min(v, 0.0)) for k, v in raw.items()}


def design_to_dict(design: RotorDesign) -> dict:
    d = asdict(design)
    d["vane_count_rounded"] = round(design.vane_count)
    d["vane_spacing_m"] = design.vane_spacing_m()
    return d


# -- surrogate ---------------------------------------------------------------------------------------------------

SURROGATE_INPUTS = ("vane_count", "vane_gap_m", "vane_thickness_m", "h_conv")
SURROGATE_BOUNDS = (VANE_COUNT_BOUNDS, VANE_GAP_BOUNDS_M, VANE_THICKNESS_BOUNDS_M,
                    (min(COOLING_PROFILES.values()) * H_SPREAD[0], max(COOLING_PROFILES.values()) * H_SPREAD[1]))


def generate_dataset(n_samples: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    hs = list(COOLING_PROFILES.values())
    rows, labels = [], []
    for _ in range(n_samples):
        d = RotorDesign(rng.uniform(*VANE_COUNT_BOUNDS), rng.uniform(*VANE_GAP_BOUNDS_M),
                        rng.uniform(*VANE_THICKNESS_BOUNDS_M))
        h = rng.choice(hs) * rng.uniform(*H_SPREAD)
        rows.append([d.vane_count, d.vane_gap_m, d.vane_thickness_m, h])
        labels.append(thermal_resistance_k_w(d, h))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train_surrogate(n_samples: int = 6000, epochs: int = 1500, seed: int = 0) -> SurrogateFit:
    X, y = generate_dataset(n_samples, seed)
    return fit_surrogate(X, y, SURROGATE_BOUNDS, epochs=epochs, seed=seed, log_output=True)


def predict_temperature_c(surrogate, design: RotorDesign, power_w: float, h_conv: float) -> float:
    with torch.no_grad():
        x = torch.tensor([[design.vane_count, design.vane_gap_m, design.vane_thickness_m, h_conv]])
        return T_AMBIENT_C + power_w * float(surrogate(x)[0])


@dataclass
class OptimizationResult:
    design: RotorDesign
    surrogate_predicted_temp_c: float


def optimize_design(surrogate, power_w: float, h_conv: float, steps: int = 400, lr: float = 0.05,
                    penalty_weight: float = 100.0) -> OptimizationResult:
    """Minimise the surrogate's rotor temperature over (N, gap, t): same loss form (``log R`` plus the relative
    spacing shortfall), sigmoid-bounded descent and integer rounding as :func:`.heatsink_fins.optimize_design`."""
    freeze(surrogate)
    h_t = torch.tensor(float(h_conv))

    def loss(v):
        r = surrogate(torch.stack([v["n"], v["gap"], v["t"], h_t]).unsqueeze(0))[0]
        spacing = (MEAN_CIRCUMFERENCE_M - v["n"] * v["t"]) / (v["n"] - 1).clamp_min(1.0)
        return torch.log(r) + penalty_weight * (torch.relu(MIN_VANE_SPACING_M - spacing) / MIN_VANE_SPACING_M) ** 2

    b = BASELINE_DESIGN
    x = descend(loss, {"n": b.vane_count, "gap": b.vane_gap_m, "t": b.vane_thickness_m}, steps=steps, lr=lr,
                bounds={"n": VANE_COUNT_BOUNDS, "gap": VANE_GAP_BOUNDS_M, "t": VANE_THICKNESS_BOUNDS_M},
                reparam="sigmoid")
    n = round(x["n"])
    while n > VANE_COUNT_BOUNDS[0] and RotorDesign(n, x["gap"], x["t"]).vane_spacing_m() < MIN_VANE_SPACING_M - 1e-9:
        n -= 1
    design = RotorDesign(float(n), x["gap"], x["t"])
    return OptimizationResult(design, predict_temperature_c(surrogate, design, power_w, h_conv))


def verify_design(design: RotorDesign, power_w: float, h_conv: float,
                  surrogate_predicted_temp_c: float) -> VerificationCard:
    real = rotor_temperature_c(design, power_w, h_conv)
    s = surrogate_predicted_temp_c
    err = relative_error(s - T_AMBIENT_C, real - T_AMBIENT_C, 1e-9)
    checks = [banded("surrogate_vs_real_model", err, REL_ERROR_PASS, REL_ERROR_WARN,
                     pass_text=f"Surrogate predicted {s:.0f} C; the formula gives {real:.0f} C (error {err * 100:.1f}% "
                               "of the rise).",
                     warn_text=f"{err * 100:.1f}% error on the temperature rise between surrogate and formula.",
                     fail_text=f"{err * 100:.1f}% error on the temperature rise: do not use the surrogate's number.")]
    checks += constraint_checks(constraint_violations(design))
    if real < T_AMBIENT_C:
        checks.append(CheckResult("energy_sanity", FAIL, "Rotor below ambient while dissipating heat.", real))
    elif real > MAX_SAFE_ROTOR_TEMP_C:
        checks.append(CheckResult("energy_sanity", WARNING, f"{real:.0f} C is above the ~{MAX_SAFE_ROTOR_TEMP_C:.0f} C "
                                  "limit assumed for cast iron (fade / distortion risk).", real, MAX_SAFE_ROTOR_TEMP_C))
    else:
        checks.append(CheckResult("energy_sanity", PASS, f"{real:.0f} C is within the assumed safe range.", real,
                                  MAX_SAFE_ROTOR_TEMP_C))
    return VerificationCard(
        checks=checks,
        values={"real_temperature_c": real, "surrogate_temperature_c": s, "relative_error": err},
        covered=["surrogate-vs-formula agreement", "geometric and manufacturing limits", "thermal safety limit"],
        not_covered=["rotating airflow (CFD)", "single hard-stop transients", "dynamometer measurement"])


# -- manufacturing tolerance -------------------------------------------------------------------------------------

def monte_carlo_uncertainty(design: RotorDesign, power_w: float, h_conv: float, safety_threshold_c: float,
                            n_samples: int = 200, seed: int | None = None,
                            thickness_tolerance_m: float = VANE_THICKNESS_TOLERANCE_M,
                            gap_tolerance_m: float = VANE_GAP_TOLERANCE_M) -> dict:
    """Vane thickness and gap drawn uniformly within their tolerance bands (vane count is a part count and stays
    fixed), each sample through the formula (never the surrogate). Verdict on the 95th percentile ``p95``:
    FRAGILE if above the threshold, MARGINAL within 5 % below it, ROBUST otherwise. ``safe_fraction`` is the share of
    samples at or under the threshold, with its 95 % Wilson interval."""
    def fn(X: np.ndarray) -> np.ndarray:
        return np.array([rotor_temperature_c(RotorDesign(design.vane_count, max(g, 1e-6), max(t, 1e-6)), power_w,
                                             h_conv) for t, g in X])

    res = propagate(fn, [design.vane_thickness_m, design.vane_gap_m], [thickness_tolerance_m, gap_tolerance_m],
                    n=n_samples, dist="uniform", limits=(None, safety_threshold_c),
                    seed=0 if seed is None else seed, percentiles=(95,))
    temps = res.outputs[:, 0]
    p95 = float(res.percentiles[95.0][0])
    margin = (safety_threshold_c - p95) / safety_threshold_c
    verdict = "FRAGILE" if p95 > safety_threshold_c else "MARGINAL" if margin < 0.05 else "ROBUST"
    return {"n_samples": int(res.n), "mean_c": float(res.mean[0]), "std_c": float(res.std[0]),
            "min_c": float(temps.min()), "max_c": float(temps.max()), "p95_c": p95,
            "safe_fraction": float(res.p_within), "safe_fraction_ci95": list(res.p_within_ci),
            "safety_threshold_c": safety_threshold_c, "verdict": verdict,
            "tolerances_m": {"vane_thickness": thickness_tolerance_m, "vane_gap": gap_tolerance_m},
            "method": "uniform Monte Carlo within the tolerance bands (JCGM 101), closed-form model per sample"}


def mesh(design: RotorDesign) -> meshes.Mesh:
    """Two friction rings and the radial vanes, in millimetres."""
    return meshes.vented_rotor(MEAN_CIRCUMFERENCE_M * 1e3 / (2 * math.pi), RADIAL_FLOW_LENGTH_M * 1e3,
                               FRICTION_RING_THICKNESS_M * 1e3, design.vane_gap_m * 1e3,
                               max(round(design.vane_count), 1), design.vane_thickness_m * 1e3)


def export_stl_bytes(design: RotorDesign) -> bytes:
    return meshes.to_stl(mesh(design), "vented rotor")
