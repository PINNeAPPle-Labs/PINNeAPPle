"""PINN on the meshless node set: the interior nodes are the collocation points, the boundary nodes carry the Dirichlet
condition, and training stops at a wall-clock budget (``pinneapple_analysis.cost.train_within_budget``), so the PINN
is compared with the other methods at a stated cost. ``pp.solve(..., "pinn")`` samples a box, which is why the
training step is set up here on the domain's own nodes; the network comes from the model registry and the
derivatives from the PDE compiler's autograd operators."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def train_pinn(problem, disc, *, budget_s: float = 5.0, width: int = 32, depth: int = 3, lr: float = 2e-3,
               bc_weight: float = 10.0, n_times: int = 8, seed: int = 0, **_) -> dict[str, Any]:
    import torch

    import pinneapple_neural.architectures  # noqa: F401
    from pinneapple_analysis.cost import ComputeBudget, cosine_lr, set_lr, train_within_budget
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_physics.pinn_solver.compiler.autograd_ops import grad, laplacian

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nodes = disc.nodes
    X = nodes.X
    td = problem.time_dependent
    lo, hi = X.min(0), X.max(0)
    centre, half = torch.tensor((lo + hi) / 2), torch.tensor(np.maximum((hi - lo) / 2, 1e-12))
    T = float(problem.t_end) if td else 1.0
    net = ModelRegistry.build("bench_mlp", in_dim=3 if td else 2, out_dim=1, width=width, depth=depth).double()

    def u_of(P):                                  # P: (n, 2|3) raw coordinates (and t)
        xs = (P[:, :2] - centre) / half
        z = torch.cat([xs, P[:, 2:3] / T * 2 - 1], 1) if td else xs
        return net(z).y

    Xi, Xb = X[nodes.interior], X[nodes.boundary]
    f_scale = max(float(np.abs(problem.f(Xi)).max()), 1e-12)
    g_scale = max(float(np.abs(problem.g(Xb)).max()), 1e-12)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tt = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731

    if not td:
        Pi = tt(Xi).requires_grad_(True)
        fi, Pb, gb = tt(problem.f(Xi)), tt(Xb), tt(problem.g(Xb))
    else:
        u0 = tt(problem.u0(X))

    def step(progress):
        set_lr(opt, cosine_lr(progress, lr))
        opt.zero_grad()
        if not td:
            u = u_of(Pi)
            res = -laplacian(u, Pi) - problem.c * u
            loss = torch.mean(((res[:, 0] - fi) / f_scale) ** 2) + bc_weight * torch.mean(((u_of(Pb)[:, 0] - gb) / g_scale) ** 2)
        else:
            ts = rng.uniform(0, T, (len(Xi), 1))
            Pi_t = tt(np.c_[Xi, ts]).requires_grad_(True)
            u = u_of(Pi_t)
            ut = grad(u, Pi_t)[:, 2:3]
            lap = laplacian(u, Pi_t, coord_indices=[0, 1])
            fi_t = tt(problem.f(Xi, ts[:, 0]))
            res = ut - problem.alpha * lap
            tb = rng.uniform(0, T, len(Xb))
            gb_t = tt(problem.g(Xb, tb))
            ub = u_of(tt(np.c_[Xb, tb]))[:, 0]
            ui = u_of(tt(np.c_[X, np.zeros(len(X))]))[:, 0]
            loss = torch.mean(((res[:, 0] - fi_t) / f_scale) ** 2) \
                + bc_weight * (torch.mean(((ub - gb_t) / g_scale) ** 2) + torch.mean(((ui - u0) / g_scale) ** 2))
        loss.backward()
        opt.step()
        return float(loss.detach())

    t0 = time.perf_counter()
    run = train_within_budget(step, ComputeBudget(wall_s=float(budget_s)), count_first_step=False, log_every_s=max(1.0, budget_s / 10))
    train_s = time.perf_counter() - t0
    net.eval()

    def evaluate(P):
        with torch.no_grad():
            Q = tt(np.c_[P, np.full(len(P), T)]) if td else tt(P)
            return u_of(Q)[:, 0].numpy()
    u = evaluate(X)
    return {"u": u, "evaluate": evaluate, "solve_s": train_s, "cond": None, "noise_amplification": None,
            "train": {"budget_s": float(budget_s), "steps": run.steps, "used_s": run.wall_s,
                      "stopped_by": run.stopped_by, "final_loss": run.history[-1].get("loss") if run.history else None,
                      "width": width, "depth": depth, "n_parameters": int(sum(p.numel() for p in net.parameters()))}}
