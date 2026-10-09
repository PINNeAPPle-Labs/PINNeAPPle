"""21_active_learning.py — Residual-based active learning for PINNs.

Demonstrates:
- ResidualBasedAL: adds collocation points where the PDE residual is largest (RAD, sampling ∝ |residual|)
- VarianceBasedAL-style scores: MC-Dropout spread of the prediction as the uncertainty signal
- CombinedAL: residual + variance + diversity score (all from pinneapple_data.active_learning)

At this small budget (8 rounds, 1280 points, about 70 s on CPU) the three strategies end close together; one run
gave L2 = 0.34 (uniform), 0.35 (residual) and 0.31 (combined). Active learning pays off on solutions with sharp
local features and longer training, so treat this as a how-to, not a benchmark.
- Comparison of active vs. uniform sampling convergence
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_data.active_learning import (
    ActiveLearningConfig,
    CombinedAL,
    ResidualBasedAL,
)


# ---------------------------------------------------------------------------
# Problem: 2D Poisson  Δu = f  on [0,1]²  (same as template 01)
# Exact u(x,y) = sin(πx)sin(πy)
# ---------------------------------------------------------------------------

import math

def f_source(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0:1], xy[:, 1:2]
    return -2.0 * math.pi ** 2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def poisson_residual(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    u  = model(xy)
    if hasattr(u, "y"):
        u = u.y
    g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy - f_source(xy)


def residual_numpy(model: nn.Module, device):
    """(N, 2) numpy points -> |PDE residual| (N,): the callable the active-learning selectors expect."""
    def fn(xy: np.ndarray) -> np.ndarray:
        model.eval()                                   # deterministic residual (dropout off)
        out = []
        for i in range(0, len(xy), 4096):
            t = torch.tensor(xy[i:i + 4096], dtype=torch.float32, device=device)
            out.append(poisson_residual(model, t).abs().detach().cpu().numpy().ravel())
        model.train()
        return np.concatenate(out)
    return fn


def variance_numpy(model: nn.Module, device, n_mc: int = 30):
    """(N, 2) numpy points -> MC-Dropout variance of the prediction (N,)."""
    def fn(xy: np.ndarray) -> np.ndarray:
        model.train()                                  # dropout on
        t = torch.tensor(xy, dtype=torch.float32, device=device)
        with torch.no_grad():
            s = torch.stack([model(t) for _ in range(n_mc)])
        return s.var(0).cpu().numpy().ravel()
    return fn


def build_pinn(dropout_p: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(64, 1),
    )


def train_step(model, optimizer, xy_col, xy_bc, u_bc, n_steps: int = 200):
    for _ in range(n_steps):
        optimizer.zero_grad()
        res  = poisson_residual(model, xy_col.clone())
        l_pd = res.pow(2).mean()
        out  = model(xy_bc)
        if hasattr(out, "y"):
            out = out.y
        l_bc = (out - u_bc).pow(2).mean()
        loss = l_pd + 10 * l_bc
        loss.backward()
        optimizer.step()
    return float(loss.item())


def l2_error(model, device) -> float:
    n = 50
    x_ = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x_, x_)
    xy_t = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        u_pred = model(xy_t).cpu().numpy().ravel()
    u_ex = (np.sin(math.pi * xx) * np.sin(math.pi * yy)).ravel()
    return float(np.sqrt(((u_pred - u_ex)**2).mean()) / np.sqrt((u_ex**2).mean()))


def main():
    torch.manual_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    N_INIT   = 256      # initial collocation points
    N_ADD    = 128      # points added per active iteration
    N_ROUNDS = 8        # active learning rounds
    STEPS    = 400      # training steps per round
    CAND     = 4096     # candidate pool size

    # Boundary conditions (Dirichlet u=0)
    from pinneapple_design.geometry.csg import CSGRectangle
    rect = CSGRectangle(x_min=0, y_min=0, x_max=1, y_max=1)
    xy_bc_np = rect.sample_boundary(n=256, seed=0)
    xy_bc = torch.tensor(xy_bc_np, dtype=torch.float32, device=device)
    u_bc  = torch.zeros(len(xy_bc), 1, device=device)

    # The selectors draw their own candidate pool of CAND points inside `bounds`.

    results = {}

    # =========================================================================
    # Strategy A: uniform (no active learning — baseline)
    # =========================================================================
    print("\n[A] Uniform sampling baseline ...")
    xy_col_u = torch.rand(N_INIT, 2, device=device)
    model_u  = build_pinn().to(device)
    opt_u    = torch.optim.Adam(model_u.parameters(), lr=1e-3)
    err_u    = []
    for r in range(N_ROUNDS):
        train_step(model_u, opt_u, xy_col_u, xy_bc, u_bc, STEPS)
        # Add uniformly sampled points
        new_pts = torch.rand(N_ADD, 2, device=device)
        xy_col_u = torch.cat([xy_col_u, new_pts], dim=0)
        err_u.append(l2_error(model_u, device))
        print(f"  round {r+1}: n_col={xy_col_u.shape[0]}  L2={err_u[-1]:.4e}")
    results["uniform"] = err_u

    # =========================================================================
    # Strategy B: Residual active sampling
    # =========================================================================
    print("\n[B] Residual active sampling ...")
    xy_col_r = torch.rand(N_INIT, 2, device=device)
    model_r  = build_pinn().to(device)
    opt_r    = torch.optim.Adam(model_r.parameters(), lr=1e-3)

    al_cfg = ActiveLearningConfig(n_candidates=CAND, n_select=N_ADD, seed=0)
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    res_sampler = ResidualBasedAL(al_cfg, bounds)
    err_r = []
    for r in range(N_ROUNDS):
        train_step(model_r, opt_r, xy_col_r, xy_bc, u_bc, STEPS)
        new_np = res_sampler.select(residual_numpy(model_r, device), mode="weighted")
        new_pts = torch.tensor(new_np, device=device)
        xy_col_r = torch.cat([xy_col_r, new_pts], dim=0)
        err_r.append(l2_error(model_r, device))
        print(f"  round {r+1}: n_col={xy_col_r.shape[0]}  L2={err_r[-1]:.4e}")
    results["residual"] = err_r

    # =========================================================================
    # Strategy C: Combined residual + variance
    # =========================================================================
    print("\n[C] Combined (residual + variance) active sampling ...")
    xy_col_c = torch.rand(N_INIT, 2, device=device)
    model_c  = build_pinn(dropout_p=0.1).to(device)
    opt_c    = torch.optim.Adam(model_c.parameters(), lr=1e-3)

    comb_sampler = CombinedAL(al_cfg, bounds, residual_weight=0.6, variance_weight=0.3, diversity_weight=0.1)
    err_c = []
    for r in range(N_ROUNDS):
        train_step(model_c, opt_c, xy_col_c, xy_bc, u_bc, STEPS)
        new_np = comb_sampler.select(residual_numpy(model_c, device), variance_numpy(model_c, device, n_mc=30))
        new_pts = torch.tensor(new_np, device=device)
        xy_col_c = torch.cat([xy_col_c, new_pts], dim=0)
        err_c.append(l2_error(model_c, device))
        print(f"  round {r+1}: n_col={xy_col_c.shape[0]}  L2={err_c[-1]:.4e}")
    results["combined"] = err_c

    # =========================================================================
    # Visualisation
    # =========================================================================
    rounds = list(range(1, N_ROUNDS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, err in results.items():
        axes[0].semilogy(rounds, err, marker="o", label=label)
    axes[0].set_xlabel("Active learning round")
    axes[0].set_ylabel("Relative L2 error")
    axes[0].set_title("Active learning convergence (2D Poisson)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # Visualise where residual sampler placed points in the last round
    xy_vis = xy_col_r.detach().cpu().numpy()
    axes[1].scatter(xy_vis[:N_INIT, 0], xy_vis[:N_INIT, 1],
                    s=3, c="blue", alpha=0.4, label="Initial uniform")
    axes[1].scatter(xy_vis[N_INIT:, 0], xy_vis[N_INIT:, 1],
                    s=3, c="red", alpha=0.6, label="Residual-added")
    axes[1].set_title("Collocation distribution (Residual strategy)")
    axes[1].legend(fontsize=8)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("21_active_learning_result.png", dpi=120)
    print("Saved 21_active_learning_result.png")


if __name__ == "__main__":
    main()
