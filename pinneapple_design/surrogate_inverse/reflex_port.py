"""Bass-reflex port: radius and length that tune a cabinet to a target frequency, and the trade-off between a small
port and an accurate tuning.

Ground truth: the Helmholtz resonator (:mod:`pinneapple_physics.closed_form.helmholtz_resonator`),
``f = c / (2 pi) sqrt(A / (V L_eff))`` with ``A = pi r^2`` and ``L_eff = L + 1.7 r`` (end correction of a port
flanged at both ends). Lumped: the air in the port moves as one mass, which holds while the port is short against the
wavelength; no cabinet modes, no port compression or air-speed limits, no driver model.

The analysis of its own: :func:`sweep_pareto_frontier` minimises
``lambda * volume_fraction + (1 - lambda) * rel_freq_error^2`` for a sweep of ``lambda``, where ``volume_fraction``
is the port volume over the largest port this case allows (6 % of the cabinet). Each weight starts gradient descent
from several points and keeps the candidate with the lowest loss evaluated with the closed form, not the surrogate.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch

from pinneapple_physics.closed_form import helmholtz_resonator as hr

from . import meshes
from .core import (
    SurrogateFit,
    VerificationCard,
    banded,
    constraint_checks,
    descend,
    fit_surrogate,
    freeze,
    relative_error,
)

END_CORRECTION_FACTOR = 1.7
PORT_RADIUS_BOUNDS_M = (0.008, 0.06)
PORT_LENGTH_BOUNDS_M = (0.02, 0.35)
CABINET_VOLUME_BOUNDS_L = (5.0, 100.0)
TARGET_FREQ_BOUNDS_HZ = (22.0, 90.0)
MAX_PORT_VOLUME_FRACTION = 0.06        # design rule: the port displaces at most 6 % of the cabinet air
FLANGE_THICKNESS_M = 0.004
FLANGE_RADIUS_MARGIN_M = 0.006

SURROGATE_ERROR_PASS = 0.05
SURROGATE_ERROR_WARN = 0.15
TARGET_ERROR_PASS = 0.05
TARGET_ERROR_WARN = 0.15
_FLOAT_TOL = 1e-6


@dataclass
class PortDesign:
    radius_m: float
    length_m: float

    def port_volume_m3(self) -> float:
        return hr.port_volume_m3(self.radius_m, self.length_m)


BASELINE_DESIGN = PortDesign(radius_m=0.025, length_m=0.10)


def resonant_frequency_hz(design: PortDesign, cabinet_volume_l: float) -> float:
    """Ground truth (training labels and verification).

    Returns
    -------
    f : float [Hz]
        Port resonance.
    """
    return hr.helmholtz_resonant_frequency_hz(design.radius_m, design.length_m, cabinet_volume_l,
                                              end_correction_factor=END_CORRECTION_FACTOR)


def acoustic_field(design: PortDesign, cabinet_volume_l: float) -> dict[str, float]:
    """One value per design (the lumped model has no field along the port): the tuning within the target range."""
    return {"resonant_frequency_hz": resonant_frequency_hz(design, cabinet_volume_l),
            "freq_min_hz": TARGET_FREQ_BOUNDS_HZ[0], "freq_max_hz": TARGET_FREQ_BOUNDS_HZ[1]}


def constraint_violations(design: PortDesign, cabinet_volume_l: float) -> dict[str, float]:
    v_m3 = max(cabinet_volume_l, 1e-6) / 1000.0
    raw = {
        "min_radius": PORT_RADIUS_BOUNDS_M[0] - design.radius_m,
        "max_radius": design.radius_m - PORT_RADIUS_BOUNDS_M[1],
        "min_length": PORT_LENGTH_BOUNDS_M[0] - design.length_m,
        "max_length": design.length_m - PORT_LENGTH_BOUNDS_M[1],
        "port_volume_fraction": design.port_volume_m3() - MAX_PORT_VOLUME_FRACTION * v_m3,
    }
    return {k: (v if v > _FLOAT_TOL else min(v, 0.0)) for k, v in raw.items()}


def design_to_dict(design: PortDesign) -> dict:
    d = asdict(design)
    d["port_volume_m3"] = design.port_volume_m3()
    return d


# -- surrogate ---------------------------------------------------------------------------------------------------

SURROGATE_INPUTS = ("radius_m", "length_m", "cabinet_volume_l")      # the target frequency is never an input
SURROGATE_BOUNDS = (PORT_RADIUS_BOUNDS_M, PORT_LENGTH_BOUNDS_M, CABINET_VOLUME_BOUNDS_L)


def generate_dataset(n_samples: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n_samples):
        d = PortDesign(rng.uniform(*PORT_RADIUS_BOUNDS_M), rng.uniform(*PORT_LENGTH_BOUNDS_M))
        vol = rng.uniform(*CABINET_VOLUME_BOUNDS_L)
        rows.append([d.radius_m, d.length_m, vol])
        labels.append(resonant_frequency_hz(d, vol))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train_surrogate(n_samples: int = 6000, epochs: int = 1500, seed: int = 0) -> SurrogateFit:
    X, y = generate_dataset(n_samples, seed)
    return fit_surrogate(X, y, SURROGATE_BOUNDS, epochs=epochs, seed=seed, log_output=True)


def predict_frequency_hz(surrogate, design: PortDesign, cabinet_volume_l: float) -> float:
    with torch.no_grad():
        return float(surrogate(torch.tensor([[design.radius_m, design.length_m, cabinet_volume_l]]))[0])


def _descend(surrogate, cabinet_volume_l, target_freq_hz, volume_weight, steps, lr, penalty_weight, start):
    vol_t = torch.tensor(float(cabinet_volume_l))
    max_port = MAX_PORT_VOLUME_FRACTION * cabinet_volume_l / 1000.0

    def loss(v):
        f = surrogate(torch.stack([v["r"], v["l"], vol_t]).unsqueeze(0))[0]
        port = hr.port_volume_m3(v["r"], v["l"])           # plain arithmetic: differentiable on tensors
        penalty = (torch.relu(port - max_port) / max_port) ** 2
        rel = (f - target_freq_hz) / target_freq_hz
        return volume_weight * port / max_port + (1.0 - volume_weight) * rel ** 2 + penalty_weight * penalty

    x = descend(loss, {"r": start.radius_m, "l": start.length_m}, steps=steps, lr=lr,
                bounds={"r": PORT_RADIUS_BOUNDS_M, "l": PORT_LENGTH_BOUNDS_M}, reparam="sigmoid")
    design = PortDesign(x["r"], x["l"])
    return design, predict_frequency_hz(surrogate, design, cabinet_volume_l)


@dataclass
class OptimizationResult:
    design: PortDesign
    surrogate_predicted_freq_hz: float


def optimize_design(surrogate, cabinet_volume_l: float, target_freq_hz: float, steps: int = 400, lr: float = 0.05,
                    penalty_weight: float = 100.0) -> OptimizationResult:
    """Minimise ``((f_surrogate - f_target) / f_target)^2`` plus a penalty on the port-volume rule, from the baseline,
    in the sigmoid-bounded space of :func:`.core.descend` (radius and length bounded, comparable step sizes)."""
    freeze(surrogate)
    return OptimizationResult(*_descend(surrogate, cabinet_volume_l, target_freq_hz, 0.0, steps, lr, penalty_weight,
                                        BASELINE_DESIGN))


@dataclass
class FrontierPoint:
    volume_weight: float
    design: PortDesign
    surrogate_predicted_freq_hz: float
    real_achieved_freq_hz: float       # closed form on this design
    real_freq_error_pct: float
    port_volume_m3: float


def sweep_pareto_frontier(surrogate, cabinet_volume_l: float, target_freq_hz: float, n_points: int = 8,
                          steps: int = 400, lr: float = 0.05, penalty_weight: float = 100.0) -> list[FrontierPoint]:
    """Volume-vs-accuracy trade-off for ``lambda`` in ``linspace(0.05, 0.95, n_points)``.

    Per weight, descent starts from the previous weight's best design, the baseline and both corners of the bounds;
    then every design found (plus the target-only optimum) competes for every weight and the lowest closed-form loss
    wins. Selection by the closed form matters: descent towards small ports ends where the surrogate is less accurate
    than its average and reports designs as on target that the formula puts several per cent off. The frontier can
    still jump between adjacent weights (non-convex loss); every reported frequency and volume is from the formula."""
    freeze(surrogate)
    n_points = max(2, n_points)
    weights = [0.05 + 0.9 * i / (n_points - 1) for i in range(n_points)]
    max_port = MAX_PORT_VOLUME_FRACTION * cabinet_volume_l / 1000.0

    def real_loss(d: PortDesign, w: float) -> float:
        rel = (resonant_frequency_hz(d, cabinet_volume_l) - target_freq_hz) / target_freq_hz
        return w * d.port_volume_m3() / max_port + (1.0 - w) * rel ** 2

    accurate = _descend(surrogate, cabinet_volume_l, target_freq_hz, 0.0, steps, lr, penalty_weight, BASELINE_DESIGN)
    corners = [BASELINE_DESIGN, PortDesign(PORT_RADIUS_BOUNDS_M[0], PORT_LENGTH_BOUNDS_M[0]),
               PortDesign(PORT_RADIUS_BOUNDS_M[1], PORT_LENGTH_BOUNDS_M[1])]
    pool, warm = [accurate], accurate[0]
    for w in weights:
        runs = [_descend(surrogate, cabinet_volume_l, target_freq_hz, w, steps, lr, penalty_weight, s)
                for s in [warm] + corners]
        pool += runs
        warm = min(runs + [accurate], key=lambda c: real_loss(c[0], w))[0]
    points = []
    for w in weights:                    # every design found for any weight competes for every weight
        design, pred = min(pool, key=lambda c: real_loss(c[0], w))
        f = resonant_frequency_hz(design, cabinet_volume_l)
        points.append(FrontierPoint(w, design, pred, f, abs(f - target_freq_hz) / max(target_freq_hz, 1.0) * 100.0,
                                    design.port_volume_m3()))
    return points


def verify_design(design: PortDesign, cabinet_volume_l: float, target_freq_hz: float,
                  surrogate_predicted_freq_hz: float) -> VerificationCard:
    real = resonant_frequency_hz(design, cabinet_volume_l)
    s = surrogate_predicted_freq_hz
    surr_err = relative_error(s, real, 1.0)
    tgt_err = relative_error(real, target_freq_hz, 1.0)
    checks = [
        banded("surrogate_vs_real_formula", surr_err, SURROGATE_ERROR_PASS, SURROGATE_ERROR_WARN,
               pass_text=f"Surrogate predicted {s:.1f} Hz; the formula gives {real:.1f} Hz (error {surr_err * 100:.1f}%).",
               warn_text=f"{surr_err * 100:.1f}% discrepancy between surrogate and formula.",
               fail_text=f"{surr_err * 100:.1f}% discrepancy: do not use the surrogate's number."),
        banded("target_frequency_hit", tgt_err, TARGET_ERROR_PASS, TARGET_ERROR_WARN,
               pass_text=f"Port tuned to {real:.1f} Hz for a {target_freq_hz:.1f} Hz target "
                         f"(deviation {tgt_err * 100:.1f}%).",
               warn_text=f"{tgt_err * 100:.1f}% deviation from the {target_freq_hz:.1f} Hz target.",
               fail_text=f"{tgt_err * 100:.1f}% deviation: the design does not reach the requested tuning."),
    ]
    checks += constraint_checks(constraint_violations(design, cabinet_volume_l), "Within physical/practical limit.")
    return VerificationCard(
        checks=checks,
        values={"real_frequency_hz": real, "surrogate_frequency_hz": s, "target_frequency_hz": target_freq_hz,
                "surrogate_relative_error": surr_err, "target_relative_error": tgt_err},
        covered=["surrogate-vs-formula agreement", "target-frequency accuracy", "geometric and practical limits"],
        not_covered=["anechoic measurement", "driver excursion and port air-speed limits"])


def mesh(design: PortDesign) -> meshes.Mesh:
    """Port tube with its flange, in millimetres."""
    return meshes.flanged_tube(design.radius_m * 1e3, design.length_m * 1e3,
                               (design.radius_m + FLANGE_RADIUS_MARGIN_M) * 1e3, FLANGE_THICKNESS_M * 1e3)


def export_stl_bytes(design: PortDesign) -> bytes:
    return meshes.to_stl(mesh(design), "reflex port")
