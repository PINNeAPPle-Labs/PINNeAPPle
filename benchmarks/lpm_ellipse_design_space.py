"""Large Physics Model on an exact design space: potential-flow Cp on ellipses.

Design space: ellipses x = a cos(t), y = b sin(t) with a, b in [0.6, 1.4], uniform flow along x.
Exact surface speed (Milne-Thomson, Theoretical Hydrodynamics, §9.61):
    q / U = (a + b) |sin t| / sqrt(a^2 sin^2 t + b^2 cos^2 t),   Cp = 1 - (q/U)^2.
The LPM is trained on 30 designs and evaluated on 10 unseen ones; out-of-distribution flags are
checked on a square (same size) and a circle of radius 3 (outside the range of the family).

    python -m benchmarks.lpm_ellipse_design_space
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

from pinneapple_neural.lpm import Design, LargePhysicsModel, fit


def ellipse(a, b, n=128):
    th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    pts = torch.stack([a * torch.cos(th), b * torch.sin(th)], 1)
    nrm = torch.stack([b * torch.cos(th), a * torch.sin(th)], 1)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    q = (a + b) * torch.sin(th).abs() / torch.sqrt(a ** 2 * torch.sin(th) ** 2 + b ** 2 * torch.cos(th) ** 2)
    return Design(pts, nrm, pts, torch.zeros(0), (1 - q ** 2)[:, None])


def square(s=1.0, n=128):
    pts, nr = [], []
    for u in torch.linspace(0, 4, n + 1)[:-1].tolist():
        k, f = int(u), u - int(u)
        pts.append([(-s + 2 * s * f, -s), (s, -s + 2 * s * f), (s - 2 * s * f, s), (-s, s - 2 * s * f)][k])
        nr.append([(0, -1), (1, 0), (0, 1), (-1, 0)][k])
    P = torch.tensor(pts)
    return Design(P, torch.tensor(nr, dtype=torch.float32), P, torch.zeros(0), None)


def run(n_train=30, n_test=10, n_points=128, epochs=1500, heads=4, width=96, seed=1):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(seed)
    ab = 0.6 + 0.8 * torch.rand(n_train + n_test, 2, generator=g)
    train = [ellipse(float(a), float(b), n_points) for a, b in ab[:n_train]]
    test = [ellipse(float(a), float(b), n_points) for a, b in ab[n_train:]]
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.15, 0.5, 1.5), width=width, n_heads=heads)
    t0 = time.time()
    fit(m, train, epochs=epochs, lr=2e-3)
    rel = lambda d: float((m.predict(d)["mean"] - d.target).norm() / d.target.norm())
    held = [rel(d) for d in test]
    return {
        "train_rel_l2_mean": sum(rel(d) for d in train) / len(train),
        "heldout_rel_l2": held, "heldout_rel_l2_mean": sum(held) / len(held),
        "heldout_ood_scores": [m.ood(d)["ood_score"] for d in test],
        "square": m.ood(square(n=n_points)), "circle_r3": m.ood(ellipse(3.0, 3.0, n_points)),
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "lpm_ellipse_design_space.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)
