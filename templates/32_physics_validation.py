"""32_physics_validation.py — Physics consistency validation.

Demonstrates:
- PhysicsValidator with custom checks: mean PDE residual and the Dirichlet energy against its analytic value
- ConservationCheck.check_integral_quantity: a domain integral (∫∫ u² = 1/4) against its exact value
- BoundaryCheck.check_dirichlet: the boundary condition at sampled boundary points
- ValidationReport: per-check pass/fail with thresholds, as a table

All from ``pinneapple_analysis.validation``.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_analysis.validation import BoundaryCheck, ConservationCheck, PhysicsValidator


# ---------------------------------------------------------------------------
# Test problem: 2D Poisson  Δu = f  on [0,1]²
# Exact solution: u = sin(πx)sin(πy)
# We train a PINN and then run the full validation suite on it.
# ---------------------------------------------------------------------------

def f_source(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0:1], xy[:, 1:2]
    return -2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def build_and_train(device, n_epochs: int = 4000) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)
    phi = lambda xy: xy[:, 0:1] * (1 - xy[:, 0:1]) * xy[:, 1:2] * (1 - xy[:, 1:2])
    model = lambda xy: phi(xy) * net(xy)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(2048, 2, device=device, requires_grad=True)
        u  = model(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        (u_xx + u_yy - f_source(xy)).pow(2).mean().backward()
        opt.step()

    return model


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Train ---------------------------------------------------------------
    print("Training Poisson PINN for validation ...")
    model = build_and_train(device, n_epochs=4000)
    print("Training complete.\n")

    # --- PDE residual (also used for the residual map below) ---------------
    def pde_residual_fn(m, xy):
        xy = xy.detach().requires_grad_(True)
        u  = m(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        return u_xx + u_yy - f_source(xy)

    coords = ["x", "y"]
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

    def mean_residual(m, coord_names, domain_bounds, n=4096):
        """Custom check: mean |Δu - f| over random interior points (pass below the threshold)."""
        xy = torch.rand(n, 2, device=device)
        return float(pde_residual_fn(m, xy).abs().mean())

    expected_energy = math.pi ** 2 / 2       # ∫∫ |∇u|² for u = sin(πx) sin(πy)

    def energy_rel_error(m, coord_names, domain_bounds, n=50):
        """Custom check: relative error of the Dirichlet energy ∫∫ |∇u|² dx dy."""
        x = np.linspace(0, 1, n, dtype=np.float32)
        xx, yy = np.meshgrid(x, x)
        xy = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), device=device, requires_grad=True)
        g = torch.autograd.grad(m(xy).sum(), xy)[0]
        return abs(float((g ** 2).sum(dim=1).mean()) - expected_energy) / expected_energy

    validator = PhysicsValidator(model, coords, bounds, device=str(device))
    validator.add_custom_check(mean_residual, name="pde_residual_mean", threshold=5e-2)
    validator.add_custom_check(energy_rel_error, name="dirichlet_energy_rel_error", threshold=0.05)
    report = validator.validate(model_name="poisson_pinn")

    # --- Integral of u² against its exact value 1/4 ---------------------------
    cons = ConservationCheck(device=str(device)).check_integral_quantity(
        model, coords, bounds, integrand_fn=lambda u: u.pow(2).ravel(), expected_value=0.25,
        tolerance=0.0125, name="integral_u_squared", n_points=20_000)

    # --- Boundary condition u = 0 ---------------------------------------------
    from pinneapple_design.geometry.csg import CSGRectangle
    xy_bnd = CSGRectangle(0, 0, 1, 1).sample_boundary(n=512, seed=7)
    bc = BoundaryCheck(device=str(device)).check_dirichlet(
        model, xy_bnd, np.zeros(len(xy_bnd)), tolerance=1e-2, name="dirichlet_u_zero")

    report.checks += [cons, bc]
    print(report.summary())
    passed_all = all(c.passed for c in report.checks)
    print(f"\nOverall: {'ALL CHECKS PASSED' if passed_all else 'SOME CHECKS FAILED'}")

    # --- Visualisation -------------------------------------------------------
    n_vis = 60
    x_ = np.linspace(0, 1, n_vis, dtype=np.float32)
    xx, yy = np.meshgrid(x_, x_)
    xy_vis = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), device=device
    )
    with torch.no_grad():
        u_pred = model(xy_vis).cpu().numpy().reshape(n_vis, n_vis)
    xy_pde = torch.rand(1024, 2, device=device)
    res    = pde_residual_fn(model, xy_pde).abs().detach().cpu().numpy().ravel()   # needs autograd

    xy_pde_np = xy_pde.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im1 = axes[0].contourf(xx, yy, u_pred, levels=30, cmap="viridis")
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_title("Predicted u(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    sc = axes[1].scatter(xy_pde_np[:, 0], xy_pde_np[:, 1],
                         c=res, s=4, cmap="Reds", vmin=0)
    plt.colorbar(sc, ax=axes[1])
    axes[1].set_title("PDE residual |Δu - f| (spatial map)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("32_physics_validation_result.png", dpi=120)
    print("\nSaved 32_physics_validation_result.png")


if __name__ == "__main__":
    main()
