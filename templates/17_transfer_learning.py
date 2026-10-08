"""17_transfer_learning.py — Transfer learning for parametric PDEs.

Demonstrates:
- TransferTrainer: fine-tune a pre-trained PINN on a new physical regime
- partial_freeze strategy: freeze backbone layers, train only the head
- layer_lr_scale: different learning rates per layer group
- Comparison of fine-tuned vs. from-scratch training speed
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_adaptation.transfer_learning import (
    TransferConfig,
    TransferTrainer,
    count_trainable,
)


# ---------------------------------------------------------------------------
# Task: 1D heat equation  u_t = α u_xx  on [0,1] × [0,T]
# Steady-state surrogate: α_source = 0.1 → α_target = 0.5
# Exact steady state: u(x) = sin(πx) (same BCs, different transient speed)
# We train a PINN on α_source and fine-tune to α_target.
# ---------------------------------------------------------------------------

import math

ALPHA_SOURCE = 0.1
ALPHA_TARGET = 0.5


def heat_residual(model: nn.Module, xy: torch.Tensor, alpha: float) -> torch.Tensor:
    """Steady-state heat:  α u_xx = 0  →  residual = u_xx."""
    xy.requires_grad_(True)
    u = model(xy)
    if hasattr(u, "y"):
        u = u.y
    u_x = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    return alpha * u_xx      # should be 0 at steady state


def bc_loss(model: nn.Module, device) -> torch.Tensor:
    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    u_bc = torch.zeros(2, 1, device=device)
    out = model(x_bc)
    if hasattr(out, "y"):
        out = out.y
    return (out - u_bc).pow(2).mean()


def build_pinn() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def train_from_scratch(model: nn.Module, alpha: float, device,
                       n_epochs: int = 3000) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    x_col = torch.linspace(0, 1, 300, device=device).unsqueeze(1)
    for _ in range(n_epochs):
        opt.zero_grad()
        res = heat_residual(model, x_col.clone(), alpha)
        loss = res.pow(2).mean() + 10 * bc_loss(model, device)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: pre-train on source domain α=0.1 ----------------------------
    print(f"Pre-training on α_source={ALPHA_SOURCE} ...")
    source_model = build_pinn().to(device)
    _ = train_from_scratch(source_model, ALPHA_SOURCE, device, n_epochs=4000)
    print("Pre-training complete.")

    # --- Step 2: Baseline — train from scratch on α_target -------------------
    print(f"\nTraining from scratch on α_target={ALPHA_TARGET} ...")
    scratch_model = build_pinn().to(device)
    scratch_hist = train_from_scratch(scratch_model, ALPHA_TARGET, device, n_epochs=2000)

    # --- Step 3: TransferTrainer — freeze the input layer --------------------
    # The partial_freeze strategy freezes the parameters whose names start
    # with one of the freeze_prefix entries. In this nn.Sequential those
    # names are "0.weight"/"0.bias" for the input layer (index 1 is its
    # Tanh, which has no parameters). The trainer deep-copies the source
    # model, so source_model itself is never modified.
    #
    # layer_lr_scale replaces the old per-layer LR scheduler: each entry
    # scales finetune_lr for the parameters under that prefix, reproducing
    # the old absolute rates — layer 0: 1e-5, layer 2: 1e-4, layers 4 and
    # 6: 5e-4. (Layer 0 is frozen, so its group is empty; the scale entry
    # documents the intent and applies if the freeze is lifted.)
    config = TransferConfig(
        strategy="partial_freeze",
        freeze_prefix=["0"],
        epochs=2000,
        finetune_lr=5e-4,
        layer_lr_scale={"0": 0.02, "2": 0.2, "4": 1.0, "6": 1.0},
        device=str(device),
        seed=7,
    )
    trainer = TransferTrainer(source_model, config)
    frozen_model = trainer.prepare()
    n_frozen = count_trainable(frozen_model)["frozen"]
    print(f"\nFreezer: {n_frozen} parameters frozen.")

    # --- Step 4: fine-tune on α_target ---------------------------------------
    x_col = torch.linspace(0, 1, 300, device=device).unsqueeze(1)

    def ft_physics_fn(model, batch):
        # The trainer calls this with a batch dict; this template builds
        # its collocation points itself, so the batch goes unused.
        res = heat_residual(model, x_col.clone(), ALPHA_TARGET)
        return res.pow(2).mean() + 10 * bc_loss(model, device)

    print(f"Fine-tuning to α_target={ALPHA_TARGET} ...")
    result = trainer.finetune(target_physics_fn=ft_physics_fn)
    frozen_model = result["model"]
    ft_hist = [entry["loss_total"] for entry in result["history"]]
    print("Fine-tuning complete.")

    # --- Evaluation -----------------------------------------------------------
    x_vis = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    u_exact = torch.sin(math.pi * x_vis).cpu().numpy().ravel()

    with torch.no_grad():
        u_ft    = frozen_model(x_vis).cpu().numpy().ravel()
        u_scr   = scratch_model(x_vis).cpu().numpy().ravel()

    x_plot = x_vis.cpu().numpy().ravel()

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(scratch_hist, label=f"From scratch (α={ALPHA_TARGET})", alpha=0.8)
    axes[0].semilogy(ft_hist, label="Fine-tuned (frozen backbone)", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Transfer vs. from-scratch convergence")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(x_plot, u_exact, "k--", label="Exact")
    axes[1].plot(x_plot, u_ft,    "b-",  label="Fine-tuned")
    axes[1].plot(x_plot, u_scr,   "r-",  label="From scratch")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].set_title(f"Solution at α_target={ALPHA_TARGET}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("17_transfer_learning_result.png", dpi=120)
    print("Saved 17_transfer_learning_result.png")


if __name__ == "__main__":
    main()
