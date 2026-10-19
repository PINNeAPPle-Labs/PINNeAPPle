"""Thermal compliance (heat sink at the middle of the left edge) and the Stokes "double pipe"
(Borrvall & Petersson 2003) with pinneapple_design.topology. Runs in about half a minute.

    python examples/topology_optimization/02_heat_and_stokes_flow.py
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
    DensityFilter,
    StokesDissipation2D,
    ThermalCompliance2D,
    optimize,
)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
summary = {}

heat = ThermalCompliance2D(60, 60, sink_fraction=0.1)
c_uniform = heat.evaluate(np.full(heat.n, 0.3))[0]
rh = optimize(heat, 0.3, DensityFilter.grid(60, 60, 1.5), "mma", max_iter=150, beta0=1.0, beta_max=8.0,
              beta_every=40)
summary["thermal"] = {"uniform": c_uniform, "optimized": rh.objective, "iterations": len(rh.history["objective"])}

pipes = {}
for length in (1.0, 1.5):
    x = None
    for q in (0.01, 0.1):  # continuation on the Brinkman interpolation parameter
        p = StokesDissipation2D.double_pipe(nely=40, length=length, q=q)
        r = optimize(p, 1.0 / 3.0, DensityFilter.grid(p.grid.nelx, p.grid.nely, 1.5), "mma", max_iter=300,
                     move=0.1, tol=1e-3, x0=x)
        x = r.x
    pipes[length] = (p, r)
    summary[f"double_pipe_L{length}"] = {"uniform": p.evaluate(np.full(p.n, 1 / 3))[0], "optimized": r.objective,
                                         "fluid_fraction": float(np.mean(r.x_phys))}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].imshow(rh.x_phys.reshape(60, 60).T, origin="lower", cmap="gray_r")
ax[0].set_title(f"Heat conduction, sink on the left edge\nthermal compliance {rh.objective:.3g} "
                f"(uniform {c_uniform:.3g})")
for k, length in enumerate((1.0, 1.5)):
    p, r = pipes[length]
    ax[k + 1].imshow(r.x_phys.reshape(p.grid.nelx, p.grid.nely).T, origin="lower", cmap="Blues",
                     extent=(0, length, 0, 1))
    ax[k + 1].set_title(f"Double pipe {length} x 1 (fluid dark)\ndissipated power {r.objective:.2f}")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "heat_and_stokes.png", dpi=120)
summary["runtime_s"] = time.time() - t0
(OUT / "heat_stokes_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
