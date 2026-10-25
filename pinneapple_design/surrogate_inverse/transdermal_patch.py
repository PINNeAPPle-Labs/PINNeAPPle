"""Membrane-controlled release patch: area, membrane thickness and reservoir concentration for a target release rate
and wear time, with the smallest area.

Ground truth: Fick's first law through a rate-limiting membrane (:mod:`pinneapple_physics.closed_form.
membrane_diffusion`, the reservoir-type delivery model of Robinson & Lee, *Controlled Drug Delivery*):
``rate = D K C_res A / L``, reservoir mass ``M = C_res A L_res``, depletion time ``t_dep = L_res L / (D K)``. The
depletion time depends on the membrane thickness only (area and concentration cancel), so the wear-time constraint is
applied through the closed form and the surrogate learns the rate.

``D`` and ``K`` are generic order-of-magnitude values for a small solute in a polymer membrane, not the measured
coefficients of any drug, and the model is a steady 1-D membrane with sink conditions at the outer face (no skin
resistance, no saturation plateau). This is a mass-transfer design exercise, not a dosing or medical tool.

The analysis of its own: :func:`simulate_release_timeseries`, the release over the wear time (constant zero-order
phase, then a first-order tail once the reservoir is depleted).
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import torch

from pinneapple_physics.closed_form import membrane_diffusion as md

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

DIFFUSION_COEFF_M2_S = 5e-13   # D, generic solute-in-polymer value (published range roughly 1e-13 to 1e-10 m^2/s)
PARTITION_COEFF = 1.0          # K, membrane/reservoir partitioning (dimensionless)
RESERVOIR_THICKNESS_M = 0.001  # L_res
SECONDS_PER_DAY = 86400.0

AREA_BOUNDS_M2 = (5e-4, 40e-4)                 # 5-40 cm^2
MEMBRANE_THICKNESS_BOUNDS_M = (10e-6, 200e-6)  # 10-200 um
RESERVOIR_CONC_BOUNDS_KG_M3 = (5.0, 150.0)     # 5-150 mg/mL

TARGET_RATE_BOUNDS_MG_DAY = (20.0, 250.0)
# t_dep = L_res L / (D K) reaches 4.63 days at the 200 um membrane bound, so a wear target above ~4.6 days cannot be
# met inside this design space; the request bound stops at 4 days.
TARGET_WEAR_BOUNDS_DAYS = (1.0, 4.0)
WEAR_MARGIN = 1.03             # the optimizer aims 3 % above the requested wear time

SURROGATE_ERROR_PASS = 0.08
SURROGATE_ERROR_WARN = 0.20
TARGET_RATE_ERROR_PASS = 0.08
TARGET_RATE_ERROR_WARN = 0.20
_FLOAT_TOL = 1e-9


@dataclass
class PatchDesign:
    area_m2: float
    membrane_thickness_m: float
    reservoir_conc_kg_m3: float


BASELINE_DESIGN = PatchDesign(area_m2=20e-4, membrane_thickness_m=80e-6, reservoir_conc_kg_m3=60.0)


def release_rate_mg_day(design: PatchDesign) -> float:
    """Ground truth: steady release rate (training labels and verification)."""
    return md.steady_state_release_rate_mg_day(DIFFUSION_COEFF_M2_S, PARTITION_COEFF, design.reservoir_conc_kg_m3,
                                               design.area_m2, design.membrane_thickness_m)


def depletion_time_days(design: PatchDesign) -> float:
    return md.depletion_time_days(DIFFUSION_COEFF_M2_S, PARTITION_COEFF, design.membrane_thickness_m,
                                  RESERVOIR_THICKNESS_M)


def reservoir_mass_mg(design: PatchDesign) -> float:
    return design.reservoir_conc_kg_m3 * design.area_m2 * RESERVOIR_THICKNESS_M * md.KG_TO_MG


def concentration_at_membrane_depth(design: PatchDesign, z_from_reservoir_m: float) -> float:
    """Linear steady profile ``C(z) = K C_res (1 - z / L)`` (sink at the outer face)."""
    return md.concentration_at_membrane_depth(design.reservoir_conc_kg_m3, PARTITION_COEFF,
                                              design.membrane_thickness_m, z_from_reservoir_m)


def decay_rate_constant_per_day(design: PatchDesign) -> float:
    """First-order constant of the tail after depletion, ``k = D K / (L L_res) = 1 / t_dep`` (area cancels)."""
    return md.post_depletion_decay_rate_constant_per_day(DIFFUSION_COEFF_M2_S, PARTITION_COEFF,
                                                         design.membrane_thickness_m, RESERVOIR_THICKNESS_M)


def simulate_release_timeseries(design: PatchDesign, duration_days: float | None = None,
                                n_points: int = 100) -> list[dict[str, float]]:
    """Rate, cumulative release and nominal reservoir left over time (default: 1.5 x the depletion time).

    The tail treats the drug still dissolved at depletion as decaying with the steady transport constant; it does not
    track a separate residual mass, so the cumulative release can exceed the nominal reservoir by a bounded amount
    (about 39 % over a tail of half the depletion time)."""
    return md.simulate_release_timeseries(DIFFUSION_COEFF_M2_S, PARTITION_COEFF, design.reservoir_conc_kg_m3,
                                          design.area_m2, design.membrane_thickness_m, RESERVOIR_THICKNESS_M,
                                          duration_days=duration_days, n_points=n_points)


def concentration_field(design: PatchDesign) -> dict[str, float]:
    return {"reservoir_conc_kg_m3": design.reservoir_conc_kg_m3, "membrane_thickness_m": design.membrane_thickness_m,
            "partition_coeff": PARTITION_COEFF, "reservoir_thickness_m": RESERVOIR_THICKNESS_M}


def constraint_violations(design: PatchDesign, target_wear_days: float) -> dict[str, float]:
    raw = {
        "min_area": AREA_BOUNDS_M2[0] - design.area_m2,
        "max_area": design.area_m2 - AREA_BOUNDS_M2[1],
        "min_membrane": MEMBRANE_THICKNESS_BOUNDS_M[0] - design.membrane_thickness_m,
        "max_membrane": design.membrane_thickness_m - MEMBRANE_THICKNESS_BOUNDS_M[1],
        "min_conc": RESERVOIR_CONC_BOUNDS_KG_M3[0] - design.reservoir_conc_kg_m3,
        "max_conc": design.reservoir_conc_kg_m3 - RESERVOIR_CONC_BOUNDS_KG_M3[1],
        "insufficient_wear_duration": target_wear_days - depletion_time_days(design),
    }
    return {k: (v if v > _FLOAT_TOL else min(v, 0.0)) for k, v in raw.items()}


def design_to_dict(design: PatchDesign) -> dict:
    d = asdict(design)
    d["release_rate_mg_day"] = release_rate_mg_day(design)
    d["depletion_time_days"] = depletion_time_days(design)
    d["reservoir_mass_mg"] = reservoir_mass_mg(design)
    d["area_cm2"] = design.area_m2 * 1e4
    return d


# -- surrogate ---------------------------------------------------------------------------------------------------

SURROGATE_INPUTS = ("area_m2", "membrane_thickness_m", "reservoir_conc_kg_m3")
SURROGATE_BOUNDS = (AREA_BOUNDS_M2, MEMBRANE_THICKNESS_BOUNDS_M, RESERVOIR_CONC_BOUNDS_KG_M3)


def generate_dataset(n_samples: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n_samples):
        d = PatchDesign(rng.uniform(*AREA_BOUNDS_M2), rng.uniform(*MEMBRANE_THICKNESS_BOUNDS_M),
                        rng.uniform(*RESERVOIR_CONC_BOUNDS_KG_M3))
        rows.append([d.area_m2, d.membrane_thickness_m, d.reservoir_conc_kg_m3])
        labels.append(release_rate_mg_day(d))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train_surrogate(n_samples: int = 6000, epochs: int = 1500, seed: int = 0) -> SurrogateFit:
    """MLP on ``log rate`` (the rate spans about 0.5 to 2600 mg/day over the design space)."""
    X, y = generate_dataset(n_samples, seed)
    return fit_surrogate(X, y, SURROGATE_BOUNDS, epochs=epochs, seed=seed, log_output=True)


def predict_rate_mg_day(surrogate, design: PatchDesign) -> float:
    with torch.no_grad():
        return float(surrogate(torch.tensor([[design.area_m2, design.membrane_thickness_m,
                                              design.reservoir_conc_kg_m3]]))[0])


@dataclass
class OptimizationResult:
    design: PatchDesign
    surrogate_predicted_rate_mg_day: float


def optimize_design(surrogate, target_rate_mg_day: float, target_wear_days: float, steps: int = 1200,
                    lr: float = 0.15, weight_rate: float = 80.0, weight_wear: float = 300.0) -> OptimizationResult:
    """Minimise ``A / A_max + w_r (rel. rate error)^2 + w_w (rel. wear shortfall)^2`` in a sigmoid-bounded space
    (area, membrane and concentration differ by six orders of magnitude; in that space one learning rate moves all
    three). The wear term uses the closed-form depletion time and aims 3 % above the request."""
    freeze(surrogate)
    d_res = RESERVOIR_THICKNESS_M / (DIFFUSION_COEFF_M2_S * PARTITION_COEFF)

    def loss(v):
        rate = surrogate(torch.stack([v["a"], v["l"], v["c"]]).unsqueeze(0))[0]
        wear = d_res * v["l"] / SECONDS_PER_DAY
        rate_err = (rate - target_rate_mg_day) / max(target_rate_mg_day, 1e-6)
        shortfall = torch.relu(target_wear_days * WEAR_MARGIN - wear) / max(target_wear_days, 1e-6)
        return v["a"] / AREA_BOUNDS_M2[1] + weight_rate * rate_err ** 2 + weight_wear * shortfall ** 2

    b = BASELINE_DESIGN
    x = descend(loss, {"a": b.area_m2, "l": b.membrane_thickness_m, "c": b.reservoir_conc_kg_m3}, steps=steps, lr=lr,
                bounds={"a": AREA_BOUNDS_M2, "l": MEMBRANE_THICKNESS_BOUNDS_M, "c": RESERVOIR_CONC_BOUNDS_KG_M3},
                reparam="sigmoid")
    design = PatchDesign(x["a"], x["l"], x["c"])
    return OptimizationResult(design, predict_rate_mg_day(surrogate, design))


def verify_design(design: PatchDesign, target_rate_mg_day: float, target_wear_days: float,
                  surrogate_predicted_rate_mg_day: float) -> VerificationCard:
    real_rate = release_rate_mg_day(design)
    real_dep = depletion_time_days(design)
    s = surrogate_predicted_rate_mg_day
    surr_err = relative_error(s, real_rate, 1e-9)
    tgt_err = relative_error(real_rate, target_rate_mg_day, 1e-9)
    checks = [
        banded("surrogate_vs_real_formula", surr_err, SURROGATE_ERROR_PASS, SURROGATE_ERROR_WARN,
               pass_text=f"Surrogate predicted {s:.1f} mg/day; the formula gives {real_rate:.1f} mg/day "
                         f"(error {surr_err * 100:.1f}%).",
               warn_text=f"{surr_err * 100:.1f}% discrepancy between surrogate and formula.",
               fail_text=f"{surr_err * 100:.1f}% discrepancy: do not use the surrogate's number."),
        banded("target_rate_hit", tgt_err, TARGET_RATE_ERROR_PASS, TARGET_RATE_ERROR_WARN,
               pass_text=f"Rate {real_rate:.1f} mg/day for a target of {target_rate_mg_day:.1f} "
                         f"(deviation {tgt_err * 100:.1f}%).",
               warn_text=f"{tgt_err * 100:.1f}% deviation from the target rate.",
               fail_text=f"{tgt_err * 100:.1f}% deviation from the target rate: the design misses it."),
    ]
    if real_dep >= target_wear_days:
        checks.append(CheckResult("reservoir_lasts_wear_duration", PASS,
                                  f"Reservoir lasts {real_dep:.2f} days for a {target_wear_days:.1f}-day target.",
                                  real_dep, target_wear_days))
    elif real_dep >= 0.8 * target_wear_days:
        checks.append(CheckResult("reservoir_lasts_wear_duration", WARNING,
                                  f"Reservoir lasts {real_dep:.2f} days, slightly short of {target_wear_days:.1f}.",
                                  real_dep, target_wear_days))
    else:
        checks.append(CheckResult("reservoir_lasts_wear_duration", FAIL,
                                  f"Reservoir depletes after {real_dep:.2f} days, well before {target_wear_days:.1f}.",
                                  real_dep, target_wear_days))
    checks += constraint_checks(constraint_violations(design, target_wear_days), "Within physical/practical limit.",
                                skip=("insufficient_wear_duration",))
    return VerificationCard(
        checks=checks,
        values={"real_rate_mg_day": real_rate, "surrogate_rate_mg_day": s, "target_rate_mg_day": target_rate_mg_day,
                "real_depletion_days": real_dep, "target_wear_days": target_wear_days,
                "surrogate_relative_error": surr_err, "target_relative_error": tgt_err},
        covered=["surrogate-vs-formula agreement", "target-rate accuracy", "reservoir duration"],
        not_covered=["in-vitro release testing (Franz cell)", "skin permeation or absorption"])


def mesh(design: PatchDesign) -> meshes.Mesh:
    """Reservoir disc with the membrane on top, in millimetres (radius from the area)."""
    r_mm = math.sqrt(design.area_m2 / math.pi) * 1e3
    return meshes.layered_disc(r_mm, (RESERVOIR_THICKNESS_M * 1e3, design.membrane_thickness_m * 1e3))


def export_stl_bytes(design: PatchDesign) -> bytes:
    return meshes.to_stl(mesh(design), "patch")
