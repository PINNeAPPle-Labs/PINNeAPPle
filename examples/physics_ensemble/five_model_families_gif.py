"""Adaptive ensemble of five families of physics models (FNO, CNN, DeepONet, MeshGraphNet and a PINN) with an
animated GIF of prediction against the exact field and the model chosen at each case.

Periodic advection-diffusion u_t + c u_x = nu u_xx with exact solutions (``pinneapple_physics.advection_diffusion_1d``).
Each model is trained on its own regime of (c, nu), as happens when models come from different projects:

=========  ===============  ==================  ===========================================================
model      speed c          diffusivity nu      training
=========  ===============  ==================  ===========================================================
FNO        0.8 .. 1.2       0.005 .. 0.02       supervised (exact fields)
GNN        0.3 .. 0.7       0.1 .. 0.3          supervised; MeshGraphNet on a multiscale ring graph
DeepONet   -0.2 .. 0.2      0.02 .. 0.06        supervised; branch = initial condition + (nu, c), trunk = (x, t)
PINN       -0.7 .. -0.3     0.1 .. 0.3          no data: PDE residual only, initial condition exact by construction
CNN        -1.2 .. -0.8     0.005 .. 0.02       supervised; dilated 1-D CNN with circular padding
=========  ===============  ==================  ===========================================================

The stream drifts through the five regimes in that order (c goes from 1 to -1). The ensemble predicts each case with
the weights learned from earlier cases only, then learns from its exact solution, and must find the right model
each time the regime changes. Writes ``physics_ensemble.gif`` and prints the errors. About 4 minutes on a laptop CPU
(``scale`` < 1 trains for fewer steps).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from pinneapple_neural.architectures.convolutions.conv1d import Conv1DModel
from pinneapple_neural.architectures.graphnn.base import GraphBatch
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator
from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.ensemble_viz import animate_ensemble

AD = AdvectionDiffusion1D()
N_MODES = 4
REGIMES = {"FNO": ((0.8, 1.2), (0.005, 0.02)), "GNN": ((0.3, 0.7), (0.1, 0.3)),
           "DeepONet": ((-0.2, 0.2), (0.02, 0.06)), "PINN": ((-0.7, -0.3), (0.1, 0.3)),
           "CNN": ((-1.2, -0.8), (0.005, 0.02))}


def random_case(rng, regime):
    (c0, c1), (n0, n1) = REGIMES[regime]
    return AD.random_case(rng, rng.uniform(n0, n1), c=rng.uniform(c0, c1))


def grid_input(cases):
    """(B, 3, nx): initial condition, scaled nu and c on every grid point."""
    return torch.tensor(np.stack([np.stack([c["u0"], np.full(AD.nx, c["nu"] * 50), np.full(AD.nx, c["c"])])
                                  for c in cases]), dtype=torch.float32)


def train_supervised(forward, params, regime, steps, seed, lr=3e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cases = [random_case(rng, regime) for _ in range(300)]
    X, Y = grid_input(cases), torch.tensor(np.stack([AD.exact(c) for c in cases]), dtype=torch.float32)
    opt = torch.optim.Adam(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(cases), (32,))
        loss = ((forward(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


class CircularCNN(torch.nn.Module):
    """Dilated 1-D CNN; the input is wrapped around so the convolutions see a periodic domain."""

    def __init__(self, pad=32):
        super().__init__()
        self.pad = pad
        self.net = Conv1DModel(3, AD.nt, mode="pde_surrogate", hidden_channels=64, num_blocks=5, kernel_size=5,
                               dilation_schedule="exponential", norm="none")

    def forward(self, X):
        p = self.pad
        return self.net(torch.cat([X[..., -p:], X, X[..., :p]], dim=-1)).y[..., p:-p]


# multiscale ring graph (as in multiscale MeshGraphNets): edges to the neighbours 1, 2, 4 and 8 cells away on both
# sides, with the signed offset as edge attribute, so 4 message-passing steps reach far enough and advection has a
# direction
OFFSETS = [1, -1, 2, -2, 4, -4, 8, -8]
EDGES = torch.tensor([[i for o in OFFSETS for i in range(AD.nx)], [(i + o) % AD.nx for o in OFFSETS for i in range(AD.nx)]])
EDGE_ATTR = torch.tensor([o / 8 for o in OFFSETS for _ in range(AD.nx)], dtype=torch.float32).view(1, -1, 1)


def gnn_forward(gnn, X):
    g = GraphBatch(x=X.transpose(1, 2), edge_index=EDGES, edge_attr=EDGE_ATTR.expand(X.shape[0], -1, -1))
    return gnn(g).y.transpose(1, 2)


_TX = np.stack(np.meshgrid(AD.t, AD.x, indexing="ij"), -1).reshape(-1, 2)
TRUNK = torch.tensor(np.stack([np.sin(_TX[:, 1]), np.cos(_TX[:, 1]), _TX[:, 0]], -1), dtype=torch.float32)


def deeponet_forward(don, X):
    branch = torch.cat([X[:, 0, :], X[:, 1, :1], X[:, 2, :1]], dim=-1)
    return don(branch, TRUNK).y[..., 0].view(-1, AD.nt, AD.nx)


class ParametricPINN(torch.nn.Module):
    """u(x, t; initial condition, nu, c) with periodic features; exact at t = 0 by construction."""

    def __init__(self, width=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4 + 2 * N_MODES + 3, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))

    def forward(self, x, t, coef, nu, c):
        feats = torch.cat([torch.sin(x), torch.cos(x), torch.sin(2 * x), torch.cos(2 * x), coef, nu * 5, c, t], -1)
        u0 = 1.0 + sum(coef[:, 2 * m - 2: 2 * m - 1] * torch.sin(m * x) + coef[:, 2 * m - 1: 2 * m] * torch.cos(m * x)
                       for m in range(1, N_MODES + 1))
        return u0 + t * self.net(feats)


def ic_coefficients(u0):
    U = np.fft.rfft(u0 - 1.0) / (AD.nx / 2)
    return np.concatenate([[-U[m].imag, U[m].real] for m in range(1, N_MODES + 1)])


def train_pinn(steps, seed=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pinn = ParametricPINN()
    opt = torch.optim.Adam(pinn.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        cases = [random_case(rng, "PINN") for _ in range(16)]
        rep = lambda v: torch.tensor(v, dtype=torch.float32).repeat_interleave(64, 0)  # noqa: E731
        coef = rep(np.stack([ic_coefficients(c["u0"]) for c in cases]))
        nu, c = rep([[q["nu"]] for q in cases]), rep([[q["c"]] for q in cases])
        x = (torch.rand(len(nu), 1) * 2 * np.pi).requires_grad_(True)
        t = (torch.rand(len(nu), 1) * AD.t_end).requires_grad_(True)
        u = pinn(x, t, coef, nu, c)
        ux, ut = torch.autograd.grad(u.sum(), (x, t), create_graph=True)
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
        loss = ((ut + c * ux - nu * uxx) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return pinn


def pinn_predict(pinn, case):
    X, T = np.meshgrid(AD.x, AD.t)
    n = X.size
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        u = pinn(f(X.reshape(-1, 1)), f(T.reshape(-1, 1)), f(ic_coefficients(case["u0"])).repeat(n, 1),
                 torch.full((n, 1), case["nu"]), torch.full((n, 1), case["c"]))
    return u.numpy().reshape(AD.nt, AD.nx)


def train_experts(scale=1.0):
    """The five models, each trained on its own regime. ``scale`` multiplies the training steps."""
    steps = lambda n: max(20, int(n * scale))  # noqa: E731
    fno = FourierNeuralOperator(3, AD.nt, width=32, modes=12, layers=4, use_grid=True)
    train_supervised(lambda X: fno(X).y, fno.parameters(), "FNO", steps(800), 0)
    gnn = MeshGraphNet(node_in_dim=3, out_dim=AD.nt, edge_in_dim=1, hidden_dim=64, n_message_passing=4)
    train_supervised(lambda X: gnn_forward(gnn, X), gnn.parameters(), "GNN", steps(800), 1)
    don = DeepONet(AD.nx + 2, 3, 1, hidden=128, modes=64)
    train_supervised(lambda X: deeponet_forward(don, X), don.parameters(), "DeepONet", steps(4000), 2, lr=1e-3)
    pinn = train_pinn(steps(3000))
    cnn = CircularCNN()
    train_supervised(cnn, cnn.parameters(), "CNN", steps(800), 3)
    run_torch = lambda fn: (lambda case: fn(grid_input([case]))[0].detach().numpy())  # noqa: E731
    return [from_callable("FNO", run_torch(lambda X: fno(X).y), kind="neural_operator"),
            from_callable("GNN", run_torch(lambda X: gnn_forward(gnn, X)), kind="graph"),
            from_callable("DeepONet", run_torch(lambda X: deeponet_forward(don, X)), kind="neural_operator"),
            from_callable("PINN", lambda case: pinn_predict(pinn, case), kind="pinn"),
            from_callable("CNN", run_torch(cnn), kind="cnn")]


def main(out="physics_ensemble.gif", scale=1.0, cases_per_regime=16, frames_per_case=3, mode="select", seed=7,
         dpi=64, lookahead=0.0):
    """``lookahead`` > 0 also scores every model on the current case by its PDE residual, against its own recent
    residuals, before choosing (``PhysicsEnsemble(residual_lookahead=...)``); 0 reproduces the GIF in the docs,
    which reacts only to the errors of earlier cases."""
    t0 = time.time()
    experts = train_experts(scale)
    print(f"trained FNO, GNN, DeepONet, PINN and CNN in {time.time() - t0:.0f} s")
    rng = np.random.default_rng(seed)
    cases = [random_case(rng, r) for r in REGIMES for _ in range(cases_per_regime)]
    refs = [AD.exact(c) for c in cases]
    kw = dict(residual_fn=AD.residual, residual_lookahead=lookahead) if lookahead > 0 else {}
    run = PhysicsEnsemble(experts, mode=mode, **kw).run(cases, refs, keep_predictions=True)
    s = run.summary()
    print(f"{mode}: ensemble {s['ensemble']:.4f} | best single in hindsight {s['best_single_in_hindsight']['name']} "
          f"{s['best_single_in_hindsight']['error']:.4f} | coverage {s['coverage']:.2f}")
    for k, r in enumerate(REGIMES):
        sl = slice(k * cases_per_regime, (k + 1) * cases_per_regime)
        print(f"  regime {r:9s} ensemble {np.mean(run.ensemble_errors[sl]):.4f} | "
              + " ".join(f"{n} {np.mean(run.errors[sl, i]):.4f}" for i, n in enumerate(run.names)))
    print("  switches:", s["switches"])
    regimes = [(k * cases_per_regime, f"{r} regime") for k, r in enumerate(REGIMES)]
    path = animate_ensemble(run, refs, out, x=AD.x, times=AD.t, frames_per_case=frames_per_case, regimes=regimes,
                            title="Adaptive ensemble: FNO, GNN, DeepONet, PINN, CNN", dpi=dpi)
    print(f"wrote {path} in {time.time() - t0:.0f} s total")
    return run, s


if __name__ == "__main__":
    main()
