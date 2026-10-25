"""Worked design study: cut the profile drag of a NACA 4-digit section while holding its lift, with the
surrogate-driven gradient loop of this package.

Ground truth: :mod:`pinneapple_physics.closed_form.thin_airfoil` (thin-airfoil lift of the camber line, Hoerner
profile drag of the thickness). A :class:`~.surrogate.PhysicsSurrogate` MLP learns (camber m, thickness t/c) ->
(Cl, Cd) from samples of that ground truth; :class:`~.pipeline.DesignOptLoop` runs Adam steps on the design through
the surrogate with a :class:`~.objective.CompositeObjective` (drag + weight * (Cl - Cl_baseline)^2) and a
:class:`~.constraints.BoxConstraint` (0-6 % camber, 8-18 % thickness). Every iterate is re-evaluated with the
closed-form physics, so the reported drag and lift are the physics' numbers, not the surrogate's belief.

Training detail that matters: Cl ~ O(1) and Cd ~ O(1e-2). A plain MSE on both is dominated by Cl, and the surrogate
can learn the wrong sign of dCd/d(t/c) at the baseline, which stalls the optimiser. The targets are therefore
normalised per channel (training-set mean and standard deviation) and the trained network is composed with a fixed
affine un-normalisation layer, so the optimiser sees physical units and exact gradients through it.

Scope: the ground truth has no thickness effect on lift and no camber effect on drag (see the physics module), so the
optimum found is "thinnest section allowed, camber nudged to keep Cl"; a panel method or RANS would trade these off.

``run_design_optimization(seed=...)`` is fully reproducible for a given seed; ``seed=None`` draws a fresh one.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from pinneapple_physics.closed_form.thin_airfoil import hoerner_profile_cd, thin_airfoil_cl

from .constraints import BoxConstraint, ConstraintSet
from .objective import CompositeObjective, ObjectiveBase
from .optimizer import DesignOptimizerConfig, ParamSpace
from .pipeline import DesignOptConfig, DesignOptLoop
from .surrogate import PhysicsSurrogate

__all__ = ["CAMBER_POSITION", "ANGLE_OF_ATTACK_DEG", "REYNOLDS", "M_BOUNDS", "T_BOUNDS", "BASELINE_M",
           "BASELINE_T", "section_aero", "airfoil_outline", "train_surrogate", "run_design_optimization"]

CAMBER_POSITION = 0.4          # p of a NACA 24xx / 44xx section
ANGLE_OF_ATTACK_DEG = 4.0
REYNOLDS = 3.0e6               # chord Reynolds number of a light aircraft / UAV wing
M_BOUNDS = (0.0, 0.06)         # NACA 4-digit camber range
T_BOUNDS = (0.08, 0.18)        # thickness range kept for structure
BASELINE_M, BASELINE_T = 0.02, 0.18   # NACA 2418: thick, not drag-optimised

LIFT_PENALTY_WEIGHT = 0.3
BOX_CONSTRAINT_WEIGHT = 50.0
N_TRAIN_SAMPLES, N_VAL_SAMPLES = 2000, 400
SURROGATE_EPOCHS, SURROGATE_LR = 300, 1e-3


def section_aero(m: float, t_c: float, p: float = CAMBER_POSITION, alpha_deg: float = ANGLE_OF_ATTACK_DEG,
                 reynolds: float = REYNOLDS) -> dict[str, float]:
    """(Cl, Cd) of one design from the closed-form ground truth."""
    return {"cl": thin_airfoil_cl(m, p, alpha_deg), "cd": hoerner_profile_cd(t_c, reynolds)}


def airfoil_outline(m: float, t_c: float, p: float = CAMBER_POSITION, n_pts: int = 160) -> dict[str, list[float]]:
    """Surface points of the section (``pinneapple_design.geometry.gen.airfoil.naca_parametric``), true scale."""
    from pinneapple_design.geometry.gen.airfoil import naca_parametric

    arr = naca_parametric(m=m, p=p, t_c=t_c, n_pts=n_pts).detach().cpu().numpy()
    return {"x": arr[:, 0].tolist(), "y": arr[:, 1].tolist()}


class _DragChannel(ObjectiveBase):
    def __call__(self, theta: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u[..., 1]


class _LiftTarget(ObjectiveBase):
    def __init__(self, cl_target: float) -> None:
        self.cl_target = cl_target

    def __call__(self, theta: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (u[..., 0] - self.cl_target) ** 2


class _Unnormalize(torch.nn.Module):
    """Fixed affine layer mapping normalised outputs back to physical (Cl, Cd), differentiable."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _dataset(n: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    m_s = rng.uniform(M_BOUNDS[0] - 0.01, M_BOUNDS[1] + 0.01, n)
    t_s = rng.uniform(T_BOUNDS[0] - 0.02, T_BOUNDS[1] + 0.02, n)
    cl = np.array([thin_airfoil_cl(m, CAMBER_POSITION, ANGLE_OF_ATTACK_DEG) for m in m_s])
    cd = np.array([hoerner_profile_cd(t, REYNOLDS) for t in t_s])
    theta = np.stack([m_s, t_s], 1).astype(np.float32)
    u = np.stack([cl, cd], 1).astype(np.float32)
    return {"theta": torch.from_numpy(theta), "u": torch.from_numpy(u)}


def train_surrogate(seed: int, n_train: int = N_TRAIN_SAMPLES, n_val: int = N_VAL_SAMPLES,
                    epochs: int = SURROGATE_EPOCHS) -> dict[str, Any]:
    """Train the (m, t/c) -> (Cl, Cd) MLP on normalised targets; validate on an independently seeded batch.

    Returns ``surrogate`` (outputs in physical units), ``train_mse``, ``val_mse``, ``val_rel_error`` (relative L2 per
    channel on the validation batch) and the sizes used."""
    torch.manual_seed(seed)
    core = PhysicsSurrogate.build_mlp(in_dim=2, out_dim=2, hidden=64, layers=3)
    train = _dataset(n_train, np.random.default_rng(seed))
    val = _dataset(n_val, np.random.default_rng(seed + 1_000_003))
    mean, std = train["u"].mean(0), train["u"].std(0)
    core.train_on_dataset({"theta": train["theta"], "u": (train["u"] - mean) / std}, epochs=epochs,
                          lr=SURROGATE_LR, verbose=False)
    surrogate = PhysicsSurrogate(torch.nn.Sequential(core.model, _Unnormalize(mean, std)), core.cfg)
    with torch.no_grad():
        tr = surrogate.model(train["theta"])
        va = surrogate.model(val["theta"])
    rel = torch.linalg.norm(va - val["u"], dim=0) / torch.linalg.norm(val["u"], dim=0)
    return {"surrogate": surrogate, "train_mse": float(torch.mean((tr - train["u"]) ** 2)),
            "val_mse": float(torch.mean((va - val["u"]) ** 2)),
            "val_rel_error": {"cl": float(rel[0]), "cd": float(rel[1])},
            "n_train": n_train, "n_val": n_val, "epochs": epochs}


def run_design_optimization(n_iterations: int = 150, seed: int | None = None,
                            surrogate_epochs: int = SURROGATE_EPOCHS) -> dict[str, Any]:
    """Train the surrogate, run the closed loop and return a JSON-able result: baseline and final designs with their
    closed-form (Cl, Cd), the trajectory (design and closed-form Cl/Cd at every iterate), the surrogate's train and
    validation errors, convergence and the outlines of both sections."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    t0 = time.perf_counter()
    baseline = section_aero(BASELINE_M, BASELINE_T)
    info = train_surrogate(seed, epochs=surrogate_epochs)
    objective = CompositeObjective(terms=[(1.0, _DragChannel()),
                                          (LIFT_PENALTY_WEIGHT, _LiftTarget(baseline["cl"]))])
    constraints = ConstraintSet([BoxConstraint(theta_min=torch.tensor([M_BOUNDS[0], T_BOUNDS[0]]),
                                               theta_max=torch.tensor([M_BOUNDS[1], T_BOUNDS[1]]),
                                               weight=BOX_CONSTRAINT_WEIGHT)])
    space = ParamSpace(bounds={"m": M_BOUNDS, "t_c": T_BOUNDS}, x0={"m": BASELINE_M, "t_c": BASELINE_T})
    cfg = DesignOptConfig(n_iterations=n_iterations, refine_top_k=0, verbose=False, seed=seed,
                          optimizer_cfg=DesignOptimizerConfig(method="gradient", grad_optimizer="adam", lr=1e-3,
                                                              grad_clip=1.0))
    result = DesignOptLoop(space, info["surrogate"], objective, constraints=constraints, cfg=cfg).run()

    trajectory = []
    for i, theta in enumerate(result.history_theta):
        m_i, t_i = float(theta[0]), float(theta[1])
        trajectory.append({"iteration": i, "m": m_i, "t_c": t_i, **section_aero(m_i, t_i),
                           "objective": float(result.history_objectives[i])})
    fm, ft = float(result.best_theta[0]), float(result.best_theta[1])
    final = section_aero(fm, ft)
    return {
        "seed": seed,
        "problem": {"camber_position": CAMBER_POSITION, "angle_of_attack_deg": ANGLE_OF_ATTACK_DEG,
                    "reynolds": REYNOLDS, "m_bounds": list(M_BOUNDS), "t_bounds": list(T_BOUNDS),
                    "cl_target": baseline["cl"], "lift_penalty_weight": LIFT_PENALTY_WEIGHT},
        "baseline": {"m": BASELINE_M, "t_c": BASELINE_T, **baseline},
        "final": {"m": fm, "t_c": ft, **final},
        "improvement": {"drag_reduction_pct": (baseline["cd"] - final["cd"]) / baseline["cd"] * 100.0,
                        "lift_delta_pct": (final["cl"] - baseline["cl"]) / baseline["cl"] * 100.0},
        "surrogate": {k: info[k] for k in ("train_mse", "val_mse", "val_rel_error", "n_train", "n_val", "epochs")},
        "trajectory": trajectory,
        "n_iterations_run": len(trajectory),
        "converged_early": result.convergence_iter is not None,
        "convergence_iter": result.convergence_iter,
        "within_bounds": bool(M_BOUNDS[0] - 1e-6 <= fm <= M_BOUNDS[1] + 1e-6
                              and T_BOUNDS[0] - 1e-6 <= ft <= T_BOUNDS[1] + 1e-6),
        "elapsed_s": time.perf_counter() - t0,
        "airfoil_baseline": airfoil_outline(BASELINE_M, BASELINE_T),
        "airfoil_final": airfoil_outline(fm, ft),
    }
