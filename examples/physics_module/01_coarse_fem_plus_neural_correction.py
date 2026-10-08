"""Hybrid model: a coarse finite-element solve plus a neural correction, trained end to end.

-(k(x) u')' = 1 on (0, 1), u(0) = u(1) = 0, with k(x) = 1 + 0.8 sin(2 pi f x + phase). The input is (f, phase). A
6-cell FEM mesh cannot resolve k; a small dilated conv net sees the coarse solution, k and x on a fine grid and
learns the error. Held-out relative L2 error: coarse FEM 3.1e-1, hybrid 1.0e-1 (about a minute on a CPU).

    PYTHONPATH=. python examples/physics_module/01_coarse_fem_plus_neural_correction.py
"""
import math

import torch
import torch.nn as nn

from pinneapple_core import Mesh
from pinneapple_core.fem import solve_poisson
from pinneapple_core.module import Hybrid, Sequential, SolverModule

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
NF = 129                                                     # fine grid points
xf = torch.linspace(0, 1, NF)
k_fun = lambda th, x: 1 + 0.8 * torch.sin(2 * math.pi * th[:, :1] * x + th[:, 1:])          # (B, len(x))


def fem(n):                                                  # (B, 2) parameters -> (B, NF) solution on the fine grid
    m, pts = Mesh.interval(0, 1, n), torch.tensor(Mesh.interval(0, 1, n).points)
    xc = (pts[1:] + pts[:-1]).T / 2
    def solve(th):
        u = torch.stack([solve_poisson(pts, m.cells, 1.0, m.boundary_nodes(), kappa=k) for k in k_fun(th, xc)])
        return torch.nn.functional.interpolate(u[:, None], size=NF, mode="linear", align_corners=True)[:, 0]
    return solve


theta = torch.rand(160, 2) * torch.tensor([3.0, 6.28]) + torch.tensor([2.0, 0.0])      # frequency in [2, 5]
truth = fem(128)(theta)                                                                # fine reference
corrector = Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.Tanh(), nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.Tanh(), nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.Tanh(), nn.Conv1d(32, 1, 5, padding=2), lambda c: c[:, 0])
model = Hybrid(SolverModule(fem(6)), corrector, corrector_input=lambda a, u0: torch.stack([u0, k_fun(a[0], xf), 0 * u0 + xf], 1))
hist = model.fit(theta[:128], truth[:128], epochs=600, lr=1e-2, batch_size=32)

rel = lambda p, t: ((p - t).norm(dim=1) / t.norm(dim=1)).mean().item()
with torch.no_grad():
    print(model.describe())
    print(f"held-out relative error  coarse FEM alone {rel(model.solver(theta[128:]), truth[128:]):.3e}   "
          f"hybrid {rel(model(theta[128:]), truth[128:]):.3e}")
