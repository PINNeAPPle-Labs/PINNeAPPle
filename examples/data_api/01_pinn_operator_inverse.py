"""A PINN, a neural operator and an inverse problem trained from the same loader API.

    PYTHONPATH=. python examples/data_api/01_pinn_operator_inverse.py
"""
import numpy as np
import torch
import torch.nn as nn

from pinneapple_core import Geometry, loss as L
from pinneapple_core.data import BoundarySampler, CollocationSampler, DataLoader, PhysicsDataset

torch.manual_seed(0)
PI = np.pi
mlp = lambda i, o, w=32: nn.Sequential(nn.Linear(i, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(), nn.Linear(w, o))
second = lambda u, x: torch.autograd.grad(torch.autograd.grad(u.sum(), x, create_graph=True)[0].sum(), x, create_graph=True)[0]

# 1. PINN: -u'' = pi^2 sin(pi x) on (0, 1), u = 0 at both ends (exact u = sin(pi x))
geom = Geometry.box([0.0], [1.0])
net = mlp(1, 1)
loader = DataLoader({"col": CollocationSampler(geom, "lhs"), "bc": BoundarySampler(geom)}, batch_size={"col": 256, "bc": 2}, steps=1500)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for batch in loader:
    x = batch["col"]["x"]                                              # already requires_grad
    r = -second(net(x), x) - PI**2 * torch.sin(PI * x)
    loss = L.combine({"pde": L.pde(r), "bc": L.boundary(net(batch["bc"]["x"]))}, {"bc": 10.0})
    opt.zero_grad(); loss.backward(); opt.step()
xt = torch.linspace(0, 1, 101)[:, None]
print(f"PINN      relative L2 error {((net(xt) - torch.sin(PI * xt)).norm() / torch.sin(PI * xt).norm()).item():.2e}")

# 2. Operator: learn f -> u for -u'' = f on 64 grid points (random sine series)
n, K = 64, np.arange(1, 6)
rng = np.random.default_rng(0)
xs = np.linspace(0, 1, n)
a = rng.normal(size=(600, K.size)) / K**1.5
modes = np.sin(PI * np.outer(K, xs))
data = PhysicsDataset((a * (PI * K) ** 2) @ modes, a @ modes, coords=["x"], fields=["f", "u"]).split([0.8, 0.2])
train, test = data
op = mlp(n, n, 128)
scale = train.inputs.std()
loader = DataLoader(train, batch_size=64)                              # a dataset: epochs, shuffled
opt = torch.optim.Adam(op.parameters(), lr=2e-3)
for epoch in range(150):
    for batch in loader:
        loss = L.supervised(op(batch.x.float() / scale), batch.y.float())
        opt.zero_grad(); loss.backward(); opt.step()
tb = next(iter(DataLoader(test, batch_size=len(test), shuffle=False)))
print(f"operator  relative L2 error {L.supervised(op(tb.x.float() / scale), tb.y.float(), 'relative_l2').item():.2e}")

# 3. Inverse: find kappa in -kappa u'' = 2 pi^2 sin(pi x) from 12 noisy observations (true kappa = 2, u = sin(pi x))
xo = torch.rand(12, 1)
obs = PhysicsDataset(xo, torch.sin(PI * xo) + 0.01 * torch.randn(12, 1), coords=["x"])
net, log_kappa = mlp(1, 1), nn.Parameter(torch.zeros(()))
loader = DataLoader({"col": CollocationSampler(geom), "obs": obs}, batch_size={"col": 256, "obs": 12}, steps=2500)
opt = torch.optim.Adam([*net.parameters(), log_kappa], lr=3e-3)
for batch in loader:
    x = batch["col"]["x"]
    r = -log_kappa.exp() * second(net(x), x) - 2 * PI**2 * torch.sin(PI * x)
    loss = L.combine({"pde": L.pde(r), "data": L.supervised(net(batch["obs"]["x"]), batch["obs"]["y"]), "bc": L.boundary(net(torch.tensor([[0.0], [1.0]])))},
                     {"data": 10.0, "bc": 10.0})
    opt.zero_grad(); loss.backward(); opt.step()
print(f"inverse   kappa = {log_kappa.exp().item():.3f}  (true 2)")
