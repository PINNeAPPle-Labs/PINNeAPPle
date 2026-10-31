"""Straight-fin CPU heat sink: fin count, height and thickness on a fixed 80 x 80 mm aluminium base.

Ground truth: the 1-D fin-array formula (:mod:`pinneapple_physics.closed_form.fin_array_conduction`, Incropera &
DeWitt fin efficiency with an adiabatic-tip correction, convective plus 1-D base resistance) with a prescribed
convection coefficient per fan profile. ``k = 205 W/m K`` is the aluminium value of the ``cpu_heatsink_thermal``
preset. The formula's temperature rise is linear in power, ``T_j = T_amb + P * R(N, H, t, h)``, so the surrogate
learns the thermal resistance ``R`` (in log scale) and the power enters exactly.

Independent check of the formula itself: :func:`fin_cell_check` solves 2-D finite-volume conduction in one fin pitch
(:mod:`pinneapple_design.thermal.fin_cell`).

Scope: a fixed convection coefficient (no airflow solve, no fin-to-fin flow interaction), steady state, no
measurement. For plate-fin sizing with published forced / natural convection correlations, pressure drop and a 3-D
base-plate solve use :mod:`pinneapple_design.thermal.heatsink` (the HeatSink Sizer app) instead.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch

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

K_ALUMINUM = 205.0             # W/(m K), as the cpu_heatsink_thermal preset
T_AMBIENT_C = 25.0
BASE_W_M = 0.08
BASE_D_M = 0.08
BASE_THICKNESS_M = 0.003

FIN_HEIGHT_BOUNDS_M = (0.005, 0.045)
FIN_THICKNESS_BOUNDS_M = (0.0004, 0.003)
FIN_COUNT_BOUNDS = (3, 60)
MIN_FIN_SPACING_M = 0.0012     # below this, air does not flow between the fins (design rule of this case)
POWER_BOUNDS_W = (15.0, 150.0)  # range the surrogate is trained on

COOLING_PROFILES: dict[str, float] = {      # convection coefficient, W/(m^2 K) (assumed values per fan class)
    "quiet_fan": 18.0,
    "standard_fan": 35.0,
    "high_static_pressure_fan": 65.0,
}
H_SPREAD = (0.85, 1.15)        # training samples spread h around each profile
REL_ERROR_PASS = 0.05          # surrogate vs ground truth, on the temperature rise above ambient
REL_ERROR_WARN = 0.15
MAX_PLAUSIBLE_T_C = 250.0
_FLOAT_TOL = 1e-6              # float32 round trip of a value clamped to a bound


@dataclass
class HeatsinkDesign:
    """A fin array on the fixed base.

    Attributes
    ----------
    fin_count : float [-]
        Number of fins.
    fin_height_m : float [m]
        Fin height.
    fin_thickness_m : float [m]
        Fin thickness.
    """
    fin_count: float
    fin_height_m: float
    fin_thickness_m: float

    def fin_spacing_m(self) -> float:
        n = max(self.fin_count, 1.0)
        if n <= 1:
            return BASE_W_M
        return max((BASE_W_M - n * self.fin_thickness_m) / (n - 1), 1e-6)


BASELINE_DESIGN = HeatsinkDesign(fin_count=20.0, fin_height_m=0.025, fin_thickness_m=0.001)


def profile_params(design: HeatsinkDesign, power_w: float, h_conv: float) -> dict[str, float]:
    """The fin-array formula's intermediate quantities (``m``, ``lc_m``, resistances, fin-root rise)."""
    return fac.fin_array_profile_params(
        k_material_w_mk=K_ALUMINUM, h_conv_w_m2k=h_conv, n_fins=design.fin_count,
        fin_thickness_m=design.fin_thickness_m, fin_length_m=design.fin_height_m, transverse_extent_m=BASE_D_M,
        spacing_extent_m=BASE_W_M, base_thickness_m=BASE_THICKNESS_M, power_w=power_w)


def junction_temperature_c(design: HeatsinkDesign, power_w: float, h_conv: float) -> float:
    """Ground truth: base (junction) temperature. Used for the training labels and for the verification.

    Returns
    -------
    T_j : float [degC]
        Junction temperature.
    """
    return fac.fin_array_base_temperature_c(profile_params(design, power_w, h_conv), T_AMBIENT_C, power_w)


def thermal_resistance_k_w(design: HeatsinkDesign, h_conv: float) -> float:
    """``(T_j - T_amb) / P``: independent of the power in this model.

    Returns
    -------
    R : float [K/W]
        Thermal resistance.
    """
    return (junction_temperature_c(design, 1.0, h_conv) - T_AMBIENT_C) / 1.0


def fin_temperature_at_height_c(design: HeatsinkDesign, power_w: float, h_conv: float, z_above_root_m: float) -> float:
    """The 1-D fin profile ``theta(z) = theta_root cosh(m (Lc - z)) / cosh(m Lc)`` of the same formula.

    Returns
    -------
    T : float [degC]
        Fin temperature at ``z_above_root_m``.
    """
    return fac.fin_array_temperature_at_position_c(profile_params(design, power_w, h_conv), T_AMBIENT_C, z_above_root_m)


def thermal_field(design: HeatsinkDesign, power_w: float, h_conv: float) -> dict[str, float]:
    """What a viewer needs to colour the fins with the fin profile (evaluated per vertex)."""
    p = profile_params(design, power_w, h_conv)
    return {"t_ambient_c": T_AMBIENT_C, "theta_fin_base_c": p["theta_fin_base_c"], "m": p["m"], "lc_m": p["lc_m"],
            "base_thickness_m": BASE_THICKNESS_M}


def constraint_violations(design: HeatsinkDesign) -> dict[str, float]:
    """Signed violations (``<= 0`` satisfied)."""
    raw = {
        "min_spacing": MIN_FIN_SPACING_M - design.fin_spacing_m(),
        "min_height": FIN_HEIGHT_BOUNDS_M[0] - design.fin_height_m,
        "max_height": design.fin_height_m - FIN_HEIGHT_BOUNDS_M[1],
        "min_thickness": FIN_THICKNESS_BOUNDS_M[0] - design.fin_thickness_m,
        "max_thickness": design.fin_thickness_m - FIN_THICKNESS_BOUNDS_M[1],
        "min_count": FIN_COUNT_BOUNDS[0] - design.fin_count,
        "max_count": design.fin_count - FIN_COUNT_BOUNDS[1],
    }
    return {k: (v if v > _FLOAT_TOL else min(v, 0.0)) for k, v in raw.items()}


def design_to_dict(design: HeatsinkDesign) -> dict:
    d = asdict(design)
    d["fin_count_rounded"] = round(design.fin_count)
    d["fin_spacing_m"] = design.fin_spacing_m()
    return d


# -- surrogate ---------------------------------------------------------------------------------------------------

SURROGATE_INPUTS = ("fin_count", "fin_height_m", "fin_thickness_m", "h_conv")
SURROGATE_BOUNDS = (FIN_COUNT_BOUNDS, FIN_HEIGHT_BOUNDS_M, FIN_THICKNESS_BOUNDS_M,
                    (min(COOLING_PROFILES.values()) * H_SPREAD[0], max(COOLING_PROFILES.values()) * H_SPREAD[1]))


def generate_dataset(n_samples: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform designs, ``h`` spread around the fan profiles; label = thermal resistance from the formula."""
    rng = random.Random(seed)
    hs = list(COOLING_PROFILES.values())
    rows, labels = [], []
    for _ in range(n_samples):
        d = HeatsinkDesign(rng.uniform(*FIN_COUNT_BOUNDS), rng.uniform(*FIN_HEIGHT_BOUNDS_M),
                           rng.uniform(*FIN_THICKNESS_BOUNDS_M))
        h = rng.choice(hs) * rng.uniform(*H_SPREAD)
        rows.append([d.fin_count, d.fin_height_m, d.fin_thickness_m, h])
        labels.append(thermal_resistance_k_w(d, h))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train_surrogate(n_samples: int = 6000, epochs: int = 1500, seed: int = 0) -> SurrogateFit:
    """MLP on ``log R``; ``fit.val_mean_rel_err`` is the held-out error on ``R``, i.e. on the temperature rise."""
    X, y = generate_dataset(n_samples, seed)
    return fit_surrogate(X, y, SURROGATE_BOUNDS, epochs=epochs, seed=seed, log_output=True)


def predict_temperature_c(surrogate, design: HeatsinkDesign, power_w: float, h_conv: float) -> float:
    with torch.no_grad():
        x = torch.tensor([[design.fin_count, design.fin_height_m, design.fin_thickness_m, h_conv]])
        return T_AMBIENT_C + power_w * float(surrogate(x)[0])


@dataclass
class OptimizationResult:
    design: HeatsinkDesign
    surrogate_predicted_temp_c: float
    steps_run: int


def optimize_design(surrogate, power_w: float, h_conv: float, steps: int = 400, lr: float = 0.05,
                    penalty_weight: float = 100.0) -> OptimizationResult:
    """Minimise the surrogate's junction temperature over (N, H, t) from the baseline.

    The loss is ``log R + penalty_weight * (relative spacing shortfall)^2`` (``T_j - T_amb = P R``, so the power
    only scales the objective), descended in the sigmoid-bounded space of :func:`.core.descend`: N (tens), H
    (centimetres) and t (tenths of a millimetre) move at comparable rates and stay inside their bounds. The final N
    is rounded and stepped down until the integer design keeps the spacing rule (the coupled rule
    ``N <= (W + s_min) / (t + s_min)`` is not kept by bounding N and t separately).
    """
    freeze(surrogate)
    h_t = torch.tensor(float(h_conv))

    def loss(v):
        r = surrogate(torch.stack([v["n"], v["h"], v["t"], h_t]).unsqueeze(0))[0]
        spacing = (BASE_W_M - v["n"] * v["t"]) / (v["n"] - 1).clamp_min(1.0)
        return torch.log(r) + penalty_weight * (torch.relu(MIN_FIN_SPACING_M - spacing) / MIN_FIN_SPACING_M) ** 2

    b = BASELINE_DESIGN
    x = descend(loss, {"n": b.fin_count, "h": b.fin_height_m, "t": b.fin_thickness_m}, steps=steps, lr=lr,
                bounds={"n": FIN_COUNT_BOUNDS, "h": FIN_HEIGHT_BOUNDS_M, "t": FIN_THICKNESS_BOUNDS_M},
                reparam="sigmoid")
    n = round(x["n"])
    while n > FIN_COUNT_BOUNDS[0] and HeatsinkDesign(n, x["h"], x["t"]).fin_spacing_m() < MIN_FIN_SPACING_M - 1e-9:
        n -= 1
    design = HeatsinkDesign(float(n), x["h"], x["t"])
    return OptimizationResult(design, predict_temperature_c(surrogate, design, power_w, h_conv), steps)


# -- verification ------------------------------------------------------------------------------------------------

def verify_design(design: HeatsinkDesign, power_w: float, h_conv: float,
                  surrogate_predicted_temp_c: float) -> VerificationCard:
    """Re-run the formula on the final design: surrogate error on the temperature rise (PASS <= 5 %, WARNING <= 15 %),
    every design rule, and a plausibility bound on the temperature."""
    real = junction_temperature_c(design, power_w, h_conv)
    rise_err = relative_error(surrogate_predicted_temp_c - T_AMBIENT_C, real - T_AMBIENT_C, 1e-9)
    s, r = surrogate_predicted_temp_c, real
    checks = [banded("surrogate_vs_real_solver", rise_err, REL_ERROR_PASS, REL_ERROR_WARN,
                     pass_text=f"Surrogate predicted {s:.1f} C; the formula gives {r:.1f} C "
                               f"(error {rise_err * 100:.1f}% of the rise above ambient).",
                     warn_text=f"{rise_err * 100:.1f}% error on the temperature rise between the surrogate ({s:.1f} C) "
                               f"and the formula ({r:.1f} C).",
                     fail_text=f"{rise_err * 100:.1f}% error on the temperature rise between the surrogate ({s:.1f} C) "
                               f"and the formula ({r:.1f} C): do not use the surrogate's number.")]
    checks += constraint_checks(constraint_violations(design))
    if real < T_AMBIENT_C:
        checks.append(CheckResult("energy_sanity", FAIL, "Junction below ambient while dissipating power.", real))
    elif real > MAX_PLAUSIBLE_T_C:
        checks.append(CheckResult("energy_sanity", WARNING, f"Junction at {real:.0f} C: no CPU survives this; the "
                                  "geometry does not suit this power and fan.", real, MAX_PLAUSIBLE_T_C))
    else:
        checks.append(CheckResult("energy_sanity", PASS, f"Junction at {real:.1f} C is plausible.", real,
                                  MAX_PLAUSIBLE_T_C))
    return VerificationCard(
        checks=checks,
        values={"real_temperature_c": real, "surrogate_temperature_c": s, "relative_error": rise_err,
                "relative_error_absolute_c": relative_error(s, real, 1.0)},
        covered=["surrogate-vs-formula agreement", "geometric and manufacturing limits", "temperature plausibility"],
        not_covered=["airflow (CFD) or measured h", "measurement on a built heat sink"])


def fin_cell_check(design: HeatsinkDesign, power_w: float, h_conv: float) -> dict:
    """The formula against 2-D finite-volume conduction in one fin pitch (same h, same power per fin)."""
    from pinneapple_design.thermal.fin_cell import solve_fin_cell

    n = max(round(design.fin_count), 1)
    fv = solve_fin_cell(n_fins=n, fin_thickness_m=design.fin_thickness_m, fin_height_m=design.fin_height_m,
                        fin_spacing_m=design.fin_spacing_m(), depth_m=BASE_D_M, base_thickness_m=BASE_THICKNESS_M,
                        k_w_mk=K_ALUMINUM, h_w_m2k=h_conv, power_w=power_w, t_ambient_c=T_AMBIENT_C)
    analytic = {"t_junction_c": junction_temperature_c(design, power_w, h_conv),
                "t_fin_root_c": fin_temperature_at_height_c(design, power_w, h_conv, 0.0),
                "t_fin_tip_c": fin_temperature_at_height_c(design, power_w, h_conv, design.fin_height_m)}
    numeric = {"t_junction_c": fv.t_junction_c, "t_fin_root_c": fv.t_fin_root_c, "t_fin_tip_c": fv.t_fin_tip_c}

    def rel_rise(a, b):
        return abs((a - T_AMBIENT_C) - (b - T_AMBIENT_C)) / max(abs(a - T_AMBIENT_C), 1e-9)

    return {"analytical": analytic, "finite_volume": {**numeric, "n_cells": fv.n_cells,
                                                      "energy_balance_rel_error": fv.energy_balance_rel_error,
                                                      "grid_change_rel": fv.grid_change_rel},
            "agreement": {f"{k[:-2]}_relative_diff": rel_rise(analytic[k], numeric[k]) for k in analytic}}


def mesh(design: HeatsinkDesign) -> meshes.Mesh:
    """Base plate plus fins, in millimetres."""
    return meshes.fin_array(BASE_W_M * 1e3, BASE_D_M * 1e3, BASE_THICKNESS_M * 1e3, max(round(design.fin_count), 1),
                            design.fin_height_m * 1e3, design.fin_thickness_m * 1e3)


def export_stl_bytes(design: HeatsinkDesign) -> bytes:
    return meshes.to_stl(mesh(design), "heat sink")
