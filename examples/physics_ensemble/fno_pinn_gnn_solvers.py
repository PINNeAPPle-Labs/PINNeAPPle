"""Adaptive ensemble of a neural operator (FNO), a graph network (MeshGraphNet), a parametric PINN and two
finite-difference schemes on periodic advection-diffusion, where the diffusivity moves through three regimes.

Each learned model is trained on one regime only, as happens in practice:
* FNO on low diffusion (nu in [0.005, 0.02]), supervised;
* MeshGraphNet on a ring graph, intermediate diffusion (nu in [0.03, 0.08]), supervised;
* PINN on high diffusion (nu in [0.1, 0.3]) with no data at all: PDE residual + initial condition, conditioned on the
  initial condition's Fourier coefficients and nu, periodic by construction (sin/cos features);
* upwind and coarse Lax-Wendroff schemes work everywhere, with regime-dependent errors.

The ensemble sees 150 cases (50 per regime) one at a time, predicts each with the weights learned so far, then learns
from its exact solution. Prints the errors and the switches, and plots weights over the cases
(physics_ensemble.png). Takes 3-5 minutes on a laptop CPU (training the three networks). With this CPU budget the
MeshGraphNet stays the weakest expert (about 18 % error in its own regime); the ensemble learns to ignore it.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinneapple_neural.architectures.graphnn.base import GraphBatch
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator
from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable, from_torch

AD = AdvectionDiffusion1D()
N_MODES = 4


def operator_input(case):
    return torch.tensor(np.stack([case["u0"], np.full(AD.nx, case["nu"] * 50)])[None], dtype=torch.float32)


def supervised(model, forward, nu_range, steps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cases = [AD.random_case(rng, rng.uniform(*nu_range)) for _ in range(300)]
    X = torch.cat([operator_input(c) for c in cases])
    Y = torch.tensor(np.stack([AD.exact(c) for c in cases]), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(cases), (32,))
        loss = ((forward(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


RING = torch.tensor([list(range(AD.nx)) * 2,
                     [(i + 1) % AD.nx for i in range(AD.nx)] + [(i - 1) % AD.nx for i in range(AD.nx)]])


# direction of each edge (+1 to the right neighbour, -1 to the left): without it message passing is isotropic and
# cannot represent advection, which has a direction
EDGE_DIR = torch.tensor([1.0] * AD.nx + [-1.0] * AD.nx).view(1, -1, 1)


def gnn_forward(gnn, X):
    g = GraphBatch(x=X.transpose(1, 2), edge_index=RING, edge_attr=EDGE_DIR.expand(X.shape[0], -1, -1))
    return gnn(g).y.transpose(1, 2)


class ParametricPINN(torch.nn.Module):
    """u(x, t; c_ic, nu) with periodic features; the initial condition enters through its Fourier coefficients."""

    def __init__(self, width=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4 + 2 * N_MODES + 2, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))

    def forward(self, x, t, coef, nu):
        feats = torch.cat([torch.sin(x), torch.cos(x), torch.sin(2 * x), torch.cos(2 * x), coef, nu * 5, t], dim=-1)
        u0 = 1.0 + sum(coef[:, 2 * (m - 1): 2 * (m - 1) + 1] * torch.sin(m * x) + coef[:, 2 * m - 1: 2 * m] * torch.cos(m * x)
                       for m in range(1, N_MODES + 1))
        return u0 + t * self.net(feats)                               # initial condition exact by construction


def ic_coefficients(u0):
    U = np.fft.rfft(u0 - 1.0) / (AD.nx / 2)
    return np.concatenate([[-U[m].imag, U[m].real] for m in range(1, N_MODES + 1)])


def train_pinn(nu_range=(0.1, 0.3), steps=3000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pinn = ParametricPINN()
    opt = torch.optim.Adam(pinn.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        cases = [AD.random_case(rng, rng.uniform(*nu_range)) for _ in range(16)]
        coef = torch.tensor(np.stack([ic_coefficients(c["u0"]) for c in cases]), dtype=torch.float32).repeat_interleave(64, 0)
        nu = torch.tensor([c["nu"] for c in cases], dtype=torch.float32).repeat_interleave(64)[:, None]
        x = (torch.rand(len(nu), 1) * 2 * np.pi).requires_grad_(True)
        t = (torch.rand(len(nu), 1) * AD.t_end).requires_grad_(True)
        u = pinn(x, t, coef, nu)
        ux, ut = torch.autograd.grad(u.sum(), (x, t), create_graph=True)
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
        loss = ((ut + ux - nu * uxx) ** 2).mean()                    # c = 1
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return pinn


def pinn_predict(pinn, case):
    X, Tm = np.meshgrid(AD.x, AD.t)
    n = X.size
    coef = torch.tensor(ic_coefficients(case["u0"]), dtype=torch.float32).repeat(n, 1)
    with torch.no_grad():
        u = pinn(torch.tensor(X.reshape(-1, 1), dtype=torch.float32), torch.tensor(Tm.reshape(-1, 1), dtype=torch.float32),
                 coef, torch.full((n, 1), case["nu"]))
    return u.numpy().reshape(AD.nt, AD.nx)


def main(out="physics_ensemble.png"):
    t0 = time.time()
    fno_model = FourierNeuralOperator(2, AD.nt, width=32, modes=12, layers=4, use_grid=True)
    supervised(fno_model, lambda X: fno_model(X).y, (0.005, 0.02), 800, 0)
    gnn = MeshGraphNet(node_in_dim=2, out_dim=AD.nt, edge_in_dim=1, hidden_dim=64, n_message_passing=10)
    supervised(gnn, lambda X: gnn_forward(gnn, X), (0.03, 0.08), 800, 1)
    pinn = train_pinn()
    print(f"trained FNO, MeshGraphNet and PINN in {time.time() - t0:.0f} s")

    experts = [from_torch("FNO (low nu)", fno_model, operator_input, lambda y, c: y[0]),
               from_callable("MeshGraphNet (mid nu)", lambda c: gnn_forward(gnn, operator_input(c))[0].detach().numpy()),
               from_callable("PINN (high nu, no data)", lambda c: pinn_predict(pinn, c), kind="neural"),
               from_callable("upwind FD", lambda c: AD.finite_difference(c, "upwind")),
               from_callable("Lax-Wendroff FD (coarse)", lambda c: AD.finite_difference(c, "lax_wendroff", coarsen=2))]
    rng = np.random.default_rng(7)
    cases = ([AD.random_case(rng, rng.uniform(0.005, 0.02)) for _ in range(50)]
             + [AD.random_case(rng, rng.uniform(0.03, 0.08)) for _ in range(50)]
             + [AD.random_case(rng, rng.uniform(0.1, 0.3)) for _ in range(50)])
    refs = [AD.exact(c) for c in cases]
    results = {}
    for mode in ("select", "combine"):
        run = PhysicsEnsemble(experts, mode=mode).run(cases, refs)
        s = run.summary()
        results[mode] = (run, s)
        print(f"\n{mode}: ensemble relative L2 {s['ensemble']:.4f} | best single in hindsight "
              f"{s['best_single_in_hindsight']['name']} {s['best_single_in_hindsight']['error']:.4f} | coverage {s['coverage']:.2f}")
        for k in range(3):
            sl = slice(50 * k, 50 * (k + 1))
            per = {n: float(np.nanmean(run.errors[sl, i])) for i, n in enumerate(run.names)}
            print(f"  regime {k + 1}: ensemble {np.nanmean(run.ensemble_errors[sl]):.4f} | "
                  + " | ".join(f"{n} {v:.4f}" for n, v in per.items()))
        print("  switches:", s["switches"])

    run = results["select"][0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].stackplot(np.arange(len(cases)), run.weights.T, labels=run.names)
    ax[0].set(ylabel="weight", title="Adaptive ensemble weights (select mode)")
    ax[0].legend(loc="upper left", fontsize=8)
    for i, n in enumerate(run.names):
        ax[1].semilogy(run.errors[:, i], lw=0.8, label=n)
    ax[1].semilogy(run.ensemble_errors, "k", lw=2, label="ensemble")
    ax[1].set(xlabel="case", ylabel="relative L2 error")
    ax[1].legend(fontsize=7, ncol=2)
    for x in (50, 100):
        for a in ax:
            a.axvline(x, color="grey", ls=":")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return {m: results[m][1] for m in results}


if __name__ == "__main__":
    main()
