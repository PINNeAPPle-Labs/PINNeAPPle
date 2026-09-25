"""Paper benchmark: 2D Helmholtz with BFGS vs SSBFGS vs SSBroyden vs Adam.

Problem (Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 43(5) 2021, arXiv:2001.04536; the
setup used by Urban et al., JCP 2025, arXiv:2405.04230, and by CrunchOptimizer/PINNs
``Helmholtz_a1_a4_k1_PyTorch.ipynb``)::

    Laplacian(u) + k^2 u = q  on [-1, 1]^2,  u = 0 on the boundary,
    u = sin(a1 pi x) sin(a2 pi y),  a1 = 1, a2 = 4, k = 1.

Network: 4 hidden layers x 30 tanh (as in the notebook), float64. Boundary condition imposed
exactly, u = (1 - x^2)(1 - y^2) N(x, y), following the hard-constraint form of the paper's
eq. (3). Adam warm-up, then quasi-Newton on the same collocation set.

The published runs use 1000 Adam epochs + up to 100k quasi-Newton iterations with RAD
resampling and a 20000 s budget; the defaults below are a reduced budget that fits a laptop.
Results go to ``benchmarks/_out/ssqn_helmholtz.json`` and must be read as "reduced budget",
not as a reproduction of the paper's final numbers.

    python -m benchmarks.ssqn_helmholtz --qn-iters 3000 --n-int 2000   (from the repo root)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from pinneapple_neural.trainer.self_scaled_qn import SelfScaledQuasiNewton

A1, A2, K = 1.0, 4.0, 1.0


def exact(xy):
    return torch.sin(A1 * math.pi * xy[:, :1]) * torch.sin(A2 * math.pi * xy[:, 1:2])


def source(xy):
    return (K ** 2 - (A1 * math.pi) ** 2 - (A2 * math.pi) ** 2) * exact(xy)


def make_model(seed):
    torch.manual_seed(seed)
    layers, width = 4, 30
    mods, d = [], 2
    for _ in range(layers):
        mods += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    mods.append(torch.nn.Linear(d, 1))
    net = torch.nn.Sequential(*mods).double()

    def u(xy):
        return (1 - xy[:, :1] ** 2) * (1 - xy[:, 1:2] ** 2) * net(xy)

    return net, u


def residual_loss(u, xy, q):
    xy = xy.requires_grad_(True)
    uu = u(xy)
    g = torch.autograd.grad(uu.sum(), xy, create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True)[0][:, :1]
    uyy = torch.autograd.grad(g[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return ((uxx + uyy + K ** 2 * uu - q) ** 2).mean()


def rel_l2(u, n=200):
    s = torch.linspace(-1, 1, n, dtype=torch.float64)
    X, Y = torch.meshgrid(s, s, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    with torch.no_grad():
        ref = exact(xy)
        return float(torch.linalg.norm(u(xy) - ref) / torch.linalg.norm(ref))


def run(variant, args):
    torch.manual_seed(args.seed)
    xy = (2 * torch.rand(args.n_int, 2, dtype=torch.float64) - 1)
    q = source(xy).detach()
    net, u = make_model(args.seed)
    t0 = time.time()
    adam = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.99, 0.999))
    for _ in range(args.adam_iters):
        adam.zero_grad()
        loss = residual_loss(u, xy.clone(), q)
        loss.backward()
        adam.step()
    after_adam = rel_l2(u)
    record = {"variant": variant, "adam_warmup_iters": args.adam_iters, "rel_l2_after_adam": after_adam}
    if variant == "adam":
        for _ in range(args.qn_iters):
            adam.zero_grad()
            loss = residual_loss(u, xy.clone(), q)
            loss.backward()
            adam.step()
        record.update(iters=args.qn_iters, loss=float(loss.detach()))
    else:
        opt = SelfScaledQuasiNewton(net.parameters(), variant=variant, max_iter=args.qn_iters,
                                    tolerance_change=0.0, tolerance_grad=0.0)

        def closure():
            opt.zero_grad()
            loss = residual_loss(u, xy.clone(), q)
            loss.backward()
            return loss

        opt.step(closure)
        taus = opt.state["taus"]
        record.update(iters=opt.state["n_iter"], loss=float(closure().detach()),
                      fraction_tau_below_1=sum(t < 1 for t in taus) / max(len(taus), 1))
    record.update(rel_l2=rel_l2(u), seconds=round(time.time() - t0, 1))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adam-iters", type=int, default=1000)
    p.add_argument("--qn-iters", type=int, default=3000)
    p.add_argument("--n-int", type=int, default=2000)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--variants", default="adam,bfgs,ssbfgs,ssbroyden")
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    results = []
    for v in args.variants.split(","):
        r = run(v, args)
        print(json.dumps(r))
        results.append(r)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "ssqn_helmholtz.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"problem": "Helmholtz 2D a1=1 a2=4 k=1", "budget": "reduced", "args": vars(args),
                   "results": results}, f, indent=1)
    print(out)


if __name__ == "__main__":
    main()
