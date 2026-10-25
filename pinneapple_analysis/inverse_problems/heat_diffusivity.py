"""Recover the diffusivity alpha of the 1-D heat equation from noisy readings at one instant, with a PINN that learns
the field and alpha together, and say how well the readings can determine alpha at all.

Problem (dimensionless, the textbook test case)::

    u_t = alpha u_xx on 0 < x < 1,   u(0, x) = sin(pi x),   u(t, 0) = u(t, 1) = 0
    exact solution u(t, x) = exp(-pi^2 alpha t) sin(pi x)

Synthetic readings: ``n_observations`` points at ``t = OBSERVATION_TIME`` drawn from the exact solution with
``alpha_true``, plus Gaussian noise ``noise_std``. The PINN (``pinneapple_physics.pinn_solver.factory``:
``PINNProblemSpec`` with ``inverse_params=["alpha"]``, ``PINN(..., inverse_params_names=["alpha"])``) stores alpha as
an ``nn.Parameter`` updated by the same Adam step and the same loss (PDE residual + initial/boundary conditions +
data misfit) as the network weights; the call sequence is the one of
``examples/pinn_solver/04_factory_inverse_parameter_heat_1d.py``.

Identifiability: the Fisher information of the readings for alpha, linearised at the recovered value
(``LocalSensitivity`` + ``IdentifiabilityAnalyzer``), gives the Cramer-Rao lower bound on the standard error of any
unbiased estimator from these sensor positions and this noise. It is a bound for the exact model, not an error bar
of the PINN estimate (which also carries optimisation and network error).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from pinneapple_physics.pinn_solver.factory.pinn_factory import (
    PINN,
    NeuralNetwork,
    PINNFactory,
    PINNProblemSpec,
)

from .sensitivity import IdentifiabilityAnalyzer, LocalSensitivity

__all__ = ["TRUE_ALPHA", "INITIAL_GUESS", "NOISE_STD", "N_OBSERVATIONS", "N_EPOCHS", "OBSERVATION_TIME",
           "u_exact", "identifiability_report", "run_inverse_discovery"]

TRUE_ALPHA = 0.15
INITIAL_GUESS = 0.5
NOISE_STD = 0.02
N_OBSERVATIONS = 256
N_EPOCHS = 2500
OBSERVATION_TIME = 0.5
LEARNING_RATE = 2e-3


def u_exact(t: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """exp(-pi^2 alpha t) sin(pi x)."""
    return (np.exp(-(math.pi**2) * alpha * np.asarray(t)) * np.sin(math.pi * np.asarray(x))).astype(np.float32)


def identifiability_report(t_obs: np.ndarray, x_obs: np.ndarray, alpha: float, noise_std: float) -> dict[str, Any]:
    """Fisher information and Cramer-Rao bound for alpha from readings at (t_obs, x_obs) with Gaussian noise."""
    t_t = torch.as_tensor(np.asarray(t_obs).reshape(-1), dtype=torch.float64)
    x_t = torch.as_tensor(np.asarray(x_obs).reshape(-1), dtype=torch.float64)

    def forward(theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(math.pi**2) * theta[0] * t_t) * torch.sin(math.pi * x_t)

    res = LocalSensitivity(forward, noise_std=noise_std, param_names=["alpha"]).compute(
        torch.tensor([alpha], dtype=torch.float64))
    ident = IdentifiabilityAnalyzer().analyze(res.fisher_information, param_names=["alpha"])
    crlb = math.sqrt(ident.a_optimality) if math.isfinite(ident.a_optimality) else float("inf")
    return {"fisher_information": float(res.fisher_information[0, 0]), "cramer_rao_standard_error": crlb,
            "d_optimality": float(ident.d_optimality), "e_optimality": float(ident.e_optimality),
            "is_identifiable": len(ident.identifiable_indices) == 1}


def _uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)


def run_inverse_discovery(true_alpha: float = TRUE_ALPHA, initial_guess: float = INITIAL_GUESS,
                          noise_std: float = NOISE_STD, n_observations: int = N_OBSERVATIONS,
                          n_epochs: int = N_EPOCHS, seed: int | None = None, log_every: int = 1,
                          n_collocation: int = 6000, n_condition: int = 1500,
                          progress_cb: Callable[[int, int, float, float], None] | None = None) -> dict[str, Any]:
    """Generate the noisy readings, train network and alpha jointly, and return a JSON-able result: the training
    history (loss and alpha per logged step), alpha recovered vs true, the field fit at the observation time against
    the exact solution, the readings, the identifiability report and the wall time.

    ``true_alpha`` is used only to make the readings and to score the result. ``seed`` fixes the network
    initialisation, the collocation points and the noise (``None``: fresh). ``progress_cb(step, n_epochs, loss,
    alpha)`` is called at every logged step."""
    t0 = time.time()
    dtype = torch.float32
    if seed is not None:
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    spec = PINNProblemSpec(
        pde_residuals=["Derivative(u(t,x), t) - alpha*Derivative(u(t,x), (x,2))"],
        conditions=[{"name": "ic", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0},
                    {"name": "bc_left", "equation": "u(t,x)", "weight": 1.0},
                    {"name": "bc_right", "equation": "u(t,x)", "weight": 1.0}],
        independent_vars=["t", "x"], dependent_vars=["u"], inverse_params=["alpha"],
        loss_weights={"pde": 1.0, "conditions": 50.0, "data": 5.0}, verbose=False)
    loss_fn = PINNFactory(spec).generate_loss_function()
    net = NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=4, num_neurons=64, activation=torch.nn.Tanh())
    model = PINN(net, inverse_params_names=["alpha"], initial_guesses={"alpha": initial_guess}, dtype=dtype).to(
        dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def leaf(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a).to(dtype).requires_grad_(True)

    nc, nb = int(n_collocation), int(n_condition)
    zeros, ones = np.zeros((nb, 1), np.float32), np.ones((nb, 1), np.float32)
    batch = {"collocation": (leaf(_uniform(nc, rng)), leaf(_uniform(nc, rng))),
             "conditions": [(leaf(zeros), leaf(_uniform(nb, rng))),
                            (leaf(_uniform(nb, rng)), leaf(zeros.copy())),
                            (leaf(_uniform(nb, rng)), leaf(ones))]}
    t_d = np.full((n_observations, 1), OBSERVATION_TIME, dtype=np.float32)
    x_d = _uniform(n_observations, rng)          # drawn after the training points, as in the source project
    y_noisy = (u_exact(t_d, x_d, true_alpha) + rng.normal(0.0, noise_std, size=t_d.shape)).astype(np.float32)
    batch["data"] = ((leaf(t_d), leaf(x_d)), torch.from_numpy(y_noisy).to(dtype))

    hist_epoch: list[int] = []
    hist_loss: list[float] = []
    hist_alpha: list[float] = []
    for step in range(1, n_epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model, batch)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step in (1, n_epochs):
            lv, av = float(loss.detach()), float(model.inverse_params["alpha"].detach())
            hist_epoch.append(step)
            hist_loss.append(lv)
            hist_alpha.append(av)
            if progress_cb is not None:
                progress_cb(step, n_epochs, lv, av)

    alpha_hat = float(model.inverse_params["alpha"].detach())
    x_grid = np.linspace(0.0, 1.0, 200, dtype=np.float32)[:, None]
    t_grid = np.full_like(x_grid, OBSERVATION_TIME)
    with torch.no_grad():
        u_pred = model(torch.from_numpy(t_grid), torch.from_numpy(x_grid)).numpy()
    u_true = u_exact(t_grid, x_grid, true_alpha)
    from pinneapple_physics.metrics import relative_l2

    abs_err = abs(alpha_hat - true_alpha)
    return {
        "scenario": {"true_alpha": true_alpha, "initial_guess": initial_guess, "noise_std": noise_std,
                     "n_observations": n_observations, "n_epochs": n_epochs, "observation_time": OBSERVATION_TIME,
                     "seed": seed},
        "history": {"epoch": hist_epoch, "loss": hist_loss, "alpha_estimate": hist_alpha},
        "final": {"alpha_recovered": alpha_hat, "alpha_true": true_alpha, "abs_error": abs_err,
                  "pct_error": 100.0 * abs_err / true_alpha,
                  "rel_l2_field_error": float(relative_l2(u_pred.reshape(-1), u_true.reshape(-1)))},
        "field_fit": {"x_grid": x_grid.reshape(-1).tolist(), "u_predicted": u_pred.reshape(-1).tolist(),
                      "u_true_noiseless": u_true.reshape(-1).tolist(), "x_observed": x_d.reshape(-1).tolist(),
                      "u_observed_noisy": y_noisy.reshape(-1).tolist()},
        "identifiability": identifiability_report(t_d, x_d, alpha_hat, noise_std),
        "wall_time_s": time.time() - t0,
    }
