"""Poisson PINN on the core primitives (Domain, Geometry, FunctionField, Mesh, Field).

Solves  -laplace(u) = 2 pi^2 sin(pi x) sin(pi y)  on the unit square with u = 0 on the
boundary (exact solution sin(pi x) sin(pi y)).

    python examples/core_primitives/01_poisson_pinn.py
"""
import numpy as np
import torch

from pinneapple_core import Field, FunctionField, Geometry

torch.manual_seed(0)
geom = Geometry.box([0, 0], [1, 1])
net = torch.nn.Sequential(
    torch.nn.Linear(2, 48), torch.nn.Tanh(),
    torch.nn.Linear(48, 48), torch.nn.Tanh(),
    torch.nn.Linear(48, 1),
)
u = FunctionField(net, geom.domain)

source = lambda x: 2 * np.pi**2 * torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
x_int = torch.tensor(geom.sample_interior(2000, seed=0), dtype=torch.float32)
x_bnd = torch.tensor(geom.sample_boundary(400, seed=1), dtype=torch.float32)

opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for step in range(3001):
    opt.zero_grad()
    residual = -u.laplacian(x_int) - source(x_int)
    loss = (residual**2).mean() + 100.0 * (u(x_bnd) ** 2).mean()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f"step {step:5d}  loss {loss.item():.3e}")

# Evaluate on a mesh: the same Field API now runs on numpy data.
mesh = geom.mesh(h=1 / 40)
pred = u.to_field(mesh, name="u")
exact = Field.on_mesh(mesh, lambda p: np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
err = np.linalg.norm(pred.values - exact.values) / np.linalg.norm(exact.values)
print(f"relative L2 error      {err:.3e}")
print(f"integral of u          {pred.integrate():.4f}  (exact {4 / np.pi**2:.4f})")
flux_norm = np.linalg.norm(pred.gradient().values - exact.gradient().values) / np.linalg.norm(exact.gradient().values)
print(f"relative gradient error {flux_norm:.3e}")
