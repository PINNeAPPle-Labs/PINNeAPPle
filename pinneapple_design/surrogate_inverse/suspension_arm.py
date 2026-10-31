"""Suspension arm as a rectangular cantilever: the lightest width x thickness that still reaches a fatigue life.

Ground truth (:mod:`pinneapple_physics.closed_form.cantilever_fatigue`, e.g. Shigley, *Mechanical Engineering
Design*): root bending stress ``sigma = 6 F L / (w t^2)`` and Basquin's stress-life law ``sigma_a = sigma_f' (2N)^b``.
The life target is turned into an allowable stress (``sigma_allow = sigma_f' (2 N_target)^b``), which is how the
problem is posed in practice and is far better conditioned than cycles, which span orders of magnitude. The material
constants are typical table values for a spring steel (``sigma_f' = 1200 MPa``, ``b = -0.09``), not a measured
material. No endurance limit (the power law keeps extrapolating at low stress), no stress concentrations (holes,
fillets), a constant section, one load case.

The analysis of its own: :func:`compute_sensitivity`, local derivatives of life and mass by central differences on
the closed form (:func:`pinneapple_analysis.sensitivity.local_derivatives`) and their elasticities
``(x / f) df/dx``. For this model the life is a pure power law, ``N ~ w^(-1/b) t^(-2/b)``, so the elasticities are
exactly ``-1/b`` and ``-2/b``: thickness always matters twice as much as width.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from pinneapple_analysis.sensitivity import local_derivatives
from pinneapple_physics.closed_form import cantilever_fatigue as cf

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

ARM_LENGTH_M = 0.35                     # pivot to load point, fixed by the vehicle
FATIGUE_STRENGTH_COEFF_PA = 1200e6      # sigma_f', typical spring steel (table value)
FATIGUE_STRENGTH_EXPONENT = -0.09       # b, typical for steels (-0.05 to -0.12)
STEEL_DENSITY_KG_M3 = 7850.0

WIDTH_BOUNDS_M = (0.02, 0.07)
THICKNESS_BOUNDS_M = (0.006, 0.03)
LOAD_BOUNDS_N = (400.0, 2500.0)
TARGET_LIFE_BOUNDS_CYCLES = (1.0e5, 1.0e7)

SURROGATE_ERROR_PASS = 0.05
SURROGATE_ERROR_WARN = 0.15
H_M = 1e-6                               # finite-difference step of the sensitivity (1 um)
_FLOAT_TOL = 1e-9


@dataclass
class ArmDesign:
    width_m: float
    thickness_m: float

    def mass_kg(self) -> float:
        return STEEL_DENSITY_KG_M3 * self.width_m * self.thickness_m * ARM_LENGTH_M


BASELINE_DESIGN = ArmDesign(width_m=0.05, thickness_m=0.022)


def bending_stress_pa(design: ArmDesign, load_n: float) -> float:
    """Ground truth: bending stress at the clamped root, where the moment is largest.

    Returns
    -------
    sigma : float [Pa]
        Root bending stress.
    """
    return cf.cantilever_root_bending_stress_pa(design.width_m, design.thickness_m, load_n, ARM_LENGTH_M)


def bending_stress_at_position_pa(design: ArmDesign, load_n: float, x_from_root_m: float) -> float:
    """``sigma(x) = 6 F (L - x) / (w t^2)``: largest at the root, zero at the loaded tip.

    Returns
    -------
    sigma : float [Pa]
        Bending stress at ``x_from_root_m``.
    """
    return cf.cantilever_bending_stress_at_position_pa(design.width_m, design.thickness_m, load_n, ARM_LENGTH_M,
                                                       x_from_root_m)


def fatigue_life_cycles(design: ArmDesign, load_n: float) -> float:
    """Cycles to failure at the root stress (Basquin).

    Returns
    -------
    N : float [-]
        Cycles to failure.
    """
    return cf.basquin_fatigue_life_cycles(bending_stress_pa(design, load_n), FATIGUE_STRENGTH_COEFF_PA,
                                          FATIGUE_STRENGTH_EXPONENT)


def allowable_stress_for_life_pa(target_life_cycles: float) -> float:
    """Largest stress amplitude that still reaches ``target_life_cycles`` (Basquin).

    Returns
    -------
    sigma_a : float [Pa]
        Allowable stress amplitude.
    """
    return cf.basquin_allowable_stress_pa(target_life_cycles, FATIGUE_STRENGTH_COEFF_PA, FATIGUE_STRENGTH_EXPONENT)


def stress_field(design: ArmDesign, load_n: float) -> dict[str, float]:
    return {"load_n": load_n, "arm_length_m": ARM_LENGTH_M, "width_m": design.width_m,
            "thickness_m": design.thickness_m}


def constraint_violations(design: ArmDesign, load_n: float, target_life_cycles: float) -> dict[str, float]:
    raw = {
        "min_width": WIDTH_BOUNDS_M[0] - design.width_m,
        "max_width": design.width_m - WIDTH_BOUNDS_M[1],
        "min_thickness": THICKNESS_BOUNDS_M[0] - design.thickness_m,
        "max_thickness": design.thickness_m - THICKNESS_BOUNDS_M[1],
        "fatigue_stress_exceeded": bending_stress_pa(design, load_n) - allowable_stress_for_life_pa(target_life_cycles),
    }
    return {k: (v if v > _FLOAT_TOL else min(v, 0.0)) for k, v in raw.items()}


def design_to_dict(design: ArmDesign) -> dict:
    d = asdict(design)
    d["mass_kg"] = design.mass_kg()
    return d


# -- surrogate ---------------------------------------------------------------------------------------------------

SURROGATE_INPUTS = ("width_m", "thickness_m", "load_n")
SURROGATE_BOUNDS = (WIDTH_BOUNDS_M, THICKNESS_BOUNDS_M, LOAD_BOUNDS_N)


def generate_dataset(n_samples: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n_samples):
        d = ArmDesign(rng.uniform(*WIDTH_BOUNDS_M), rng.uniform(*THICKNESS_BOUNDS_M))
        load = rng.uniform(*LOAD_BOUNDS_N)
        rows.append([d.width_m, d.thickness_m, load])
        labels.append(bending_stress_pa(d, load))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train_surrogate(n_samples: int = 6000, epochs: int = 1500, seed: int = 0) -> SurrogateFit:
    """MLP on ``log sigma`` (the stress spans about 30 MPa to 6 GPa over the space)."""
    X, y = generate_dataset(n_samples, seed)
    return fit_surrogate(X, y, SURROGATE_BOUNDS, epochs=epochs, seed=seed, log_output=True)


def predict_stress_pa(surrogate, design: ArmDesign, load_n: float) -> float:
    with torch.no_grad():
        return float(surrogate(torch.tensor([[design.width_m, design.thickness_m, load_n]]))[0])


@dataclass
class OptimizationResult:
    design: ArmDesign
    surrogate_predicted_stress_pa: float


def optimize_design(surrogate, load_n: float, target_life_cycles: float, steps: int = 500, lr: float = 0.0006,
                    penalty_weight: float = 5e-11) -> OptimizationResult:
    """Minimise ``w t`` (mass at fixed length and density) plus ``penalty_weight * max(0, sigma - sigma_allow)^2``
    through the surrogate's stress, from the baseline, clamped to the bounds."""
    freeze(surrogate)
    sigma_allow = allowable_stress_for_life_pa(target_life_cycles)
    load_t = torch.tensor(float(load_n))

    def loss(v):
        stress = surrogate(torch.stack([v["w"], v["t"], load_t]).unsqueeze(0))[0]
        return v["w"] * v["t"] + penalty_weight * torch.relu(stress - sigma_allow) ** 2

    b = BASELINE_DESIGN
    x = descend(loss, {"w": b.width_m, "t": b.thickness_m}, steps=steps, lr=lr,
                bounds={"w": WIDTH_BOUNDS_M, "t": THICKNESS_BOUNDS_M})
    design = ArmDesign(x["w"], x["t"])
    return OptimizationResult(design, predict_stress_pa(surrogate, design, load_n))


def verify_design(design: ArmDesign, load_n: float, target_life_cycles: float,
                  surrogate_predicted_stress_pa: float) -> VerificationCard:
    real_stress = bending_stress_pa(design, load_n)
    real_life = fatigue_life_cycles(design, load_n)
    s = surrogate_predicted_stress_pa
    err = relative_error(s, real_stress, 1.0)
    checks = [banded("surrogate_vs_formula_real", err, SURROGATE_ERROR_PASS, SURROGATE_ERROR_WARN,
                     pass_text=f"Surrogate predicted {s / 1e6:.1f} MPa; the formula gives {real_stress / 1e6:.1f} MPa "
                               f"(error {err * 100:.1f}%).",
                     warn_text=f"{err * 100:.1f}% discrepancy between surrogate and formula.",
                     fail_text=f"{err * 100:.1f}% discrepancy: do not use the surrogate's number.")]
    if real_life >= target_life_cycles:
        checks.append(CheckResult("fatigue_life_target", PASS, f"Life {real_life:.2e} cycles >= target "
                                  f"{target_life_cycles:.2e}.", real_life, target_life_cycles))
    elif real_life >= 0.7 * target_life_cycles:
        checks.append(CheckResult("fatigue_life_target", WARNING, f"Life {real_life:.2e} cycles is "
                                  f"{(1 - real_life / target_life_cycles) * 100:.0f}% under the target "
                                  f"{target_life_cycles:.2e}.", real_life, target_life_cycles))
    else:
        checks.append(CheckResult("fatigue_life_target", FAIL, f"Life {real_life:.2e} cycles does not reach the target "
                                  f"{target_life_cycles:.2e}: unsafe design.", real_life, target_life_cycles))
    checks += constraint_checks(constraint_violations(design, load_n, target_life_cycles), "Within physical/design limit.")
    return VerificationCard(
        checks=checks,
        values={"real_stress_pa": real_stress, "surrogate_stress_pa": s, "real_life_cycles": real_life,
                "target_life_cycles": target_life_cycles, "relative_error": err},
        covered=["surrogate-vs-formula agreement", "fatigue-life target", "geometric bounds"],
        not_covered=["road-load or rig fatigue testing", "3-D FEA with stress concentrations"])


# -- sensitivity -------------------------------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Derivatives of life and mass at a design point (``compute_sensitivity``).

    Attributes
    ----------
    width_m : float [m]
    thickness_m : float [m]
    load_n : float [N]
    life_cycles : float [-]
        Cycles to failure.
    mass_kg : float [kg]
    d_life_d_width : float [1/m]
        Cycles per metre of width.
    d_life_d_thickness : float [1/m]
        Cycles per metre of thickness.
    d_mass_d_width : float [kg/m]
    d_mass_d_thickness : float [kg/m]
    elasticity_width : float [-]
        Relative change of life over relative change of width.
    elasticity_thickness : float [-]
    elasticity_ratio : float [-]
        Larger over smaller elasticity.
    """
    width_m: float
    thickness_m: float
    load_n: float
    life_cycles: float
    mass_kg: float
    d_life_d_width: float
    d_life_d_thickness: float
    d_mass_d_width: float
    d_mass_d_thickness: float
    elasticity_width: float
    elasticity_thickness: float
    elasticity_ratio: float          # larger |elasticity| / smaller, >= 1
    dominant_variable: str           # "width_m" or "thickness_m"


def compute_sensitivity(width_m: float, thickness_m: float, load_n: float, step_m: float = H_M) -> SensitivityResult:
    """Derivatives of life and mass at the design point by central differences on the closed form (never the
    surrogate), with a step of ``step_m`` metres."""
    def fn(X: np.ndarray) -> np.ndarray:
        return np.array([[fatigue_life_cycles(ArmDesign(w, t), load_n), ArmDesign(w, t).mass_kg()] for w, t in X])

    res = local_derivatives(fn, [width_m, thickness_m], rel_step=step_m, outputs=["life_cycles", "mass_kg"])
    J, life0, mass0 = res["jacobian"], float(res["y0"][0]), float(res["y0"][1])
    e_w, e_t = J[0, 0] * width_m / life0, J[0, 1] * thickness_m / life0
    big, small = max(abs(e_w), abs(e_t)), min(abs(e_w), abs(e_t))
    return SensitivityResult(
        width_m=width_m, thickness_m=thickness_m, load_n=load_n, life_cycles=life0, mass_kg=mass0,
        d_life_d_width=float(J[0, 0]), d_life_d_thickness=float(J[0, 1]), d_mass_d_width=float(J[1, 0]),
        d_mass_d_thickness=float(J[1, 1]), elasticity_width=float(e_w), elasticity_thickness=float(e_t),
        elasticity_ratio=float(big / max(small, 1e-12)),
        dominant_variable="thickness_m" if abs(e_t) >= abs(e_w) else "width_m")


def sensitivity_to_dict(r: SensitivityResult) -> dict:
    dom, other = ("thickness", "width") if r.dominant_variable == "thickness_m" else ("width", "thickness")
    e_dom = r.elasticity_thickness if dom == "thickness" else r.elasticity_width
    e_other = r.elasticity_width if dom == "thickness" else r.elasticity_thickness
    return {
        **{k: v for k, v in asdict(r).items() if k not in ("width_m", "thickness_m")},
        "design": {"width_m": r.width_m, "thickness_m": r.thickness_m},
        "summary": (f"Fatigue life is {r.elasticity_ratio:.2f}x more sensitive to {dom} than {other} at this design "
                    f"point (elasticity {dom}={e_dom:.2f} vs. {other}={e_other:.2f})."),
        "method_note": (f"Central finite differences (h={H_M:g} m) on the closed form (bending stress -> Basquin life), "
                        "not the surrogate's gradients. Elasticity = (df/dx)(x/f), dimensionless, so width and "
                        "thickness compare directly."),
    }


def mesh(design: ArmDesign) -> meshes.Mesh:
    """The beam as a plain box in millimetres (x width, y thickness, z length): the model has no fillets or holes."""
    return meshes.rectangular_beam(design.width_m * 1e3, design.thickness_m * 1e3, ARM_LENGTH_M * 1e3)


def export_stl_bytes(design: ArmDesign) -> bytes:
    return meshes.to_stl(mesh(design), "suspension arm")
