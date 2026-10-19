"""Half MBB beam and cantilever: SIMP compliance minimisation with pinneapple_design.topology.

1. MBB beam 60 x 20, volume 0.5, density filter r_min = 2.4, OC and MMA (same problem as the
   classic educational codes of Sigmund 2001 / Andreassen et al. 2011; compare the layout qualitatively).
2. Same beam with Heaviside projection and beta continuation (crisp 0/1 design).
3. Cantilever 80 x 40, volume 0.4, MMA with projection.
Also prints a finite-difference check of the sensitivities. Runs in well under a minute.

    python examples/topology_optimization/01_mbb_beam.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_design.topology import (  # noqa: E402
    Compliance2D,
    DensityFilter,
    DesignChain,
    finite_difference_check,
    optimize,
)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
summary = {}

mbb = Compliance2D.mbb(60, 20)
filt = DensityFilter.grid(60, 20, 2.4)

chain = DesignChain(mbb.n, filt, beta=4.0)


def fun(z):
    f, df = mbb.evaluate(chain.forward(z))
    return f, chain.backward(df)


rng = np.random.default_rng(0)
x = rng.uniform(0.2, 0.9, mbb.n)
_, g = fun(x)
summary["gradient_check_max_rel_err"] = float(finite_difference_check(fun, x, np.argsort(-np.abs(g))[:8], 1e-5).max())

runs = {
    "MBB, OC, filter": optimize(mbb, 0.5, filt, "oc", max_iter=300),
    "MBB, MMA, filter": optimize(mbb, 0.5, filt, "mma", max_iter=300),
    "MBB, MMA, filter + projection": optimize(mbb, 0.5, filt, "mma", max_iter=300, beta0=1.0, beta_max=16.0,
                                              beta_every=40),
}
cant = Compliance2D.cantilever(80, 40)
runs["Cantilever, MMA, projection"] = optimize(cant, 0.4, DensityFilter.grid(80, 40, 2.0), "mma", max_iter=300,
                                               beta0=1.0, beta_max=16.0, beta_every=40)

fig, axes = plt.subplots(len(runs), 1, figsize=(8, 2.6 * len(runs)))
for ax, (name, r) in zip(axes, runs.items(), strict=True):
    shape = (60, 20) if name.startswith("MBB") else (80, 40)
    ax.imshow(r.x_phys.reshape(shape).T, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"{name}: compliance {r.objective:.2f}, {len(r.history['objective'])} it.")
    ax.axis("off")
    gray = float(((r.x_phys > 0.05) & (r.x_phys < 0.95)).mean())
    summary[name] = {"compliance": r.objective, "iterations": len(r.history["objective"]),
                     "volume": r.history["volume"][-1], "gray_fraction": gray}
fig.tight_layout()
fig.savefig(OUT / "mbb_and_cantilever.png", dpi=120)
summary["runtime_s"] = time.time() - t0
(OUT / "mbb_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
