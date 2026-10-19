"""Topology optimization of the iron of a 4-pole reluctance rotor for average torque.

Design: rotor core between a 10 mm shaft and the 40 mm rotor surface, 4-fold rotational symmetry,
density filter, Heaviside projection, MMA, volume fraction 0.5. Torque: Maxwell stress (Arkkio) at three
rotor positions with synchronous currents (60 A peak, current angle 60 electrical degrees from the
rotor's reference axis); sensitivities by the adjoint method (one extra linear solve per position),
checked against finite differences before the run. Iron is linear (mu_r = 1000) during the
optimization; the final design is then re-evaluated with nonlinear iron and with the co-energy
(virtual-work) torque. Runs in about two minutes.

    python examples/topology_optimization/03_reluctance_rotor.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from pinneapple_design.topology import (  # noqa: E402
    DensityFilter,
    DesignChain,
    SymmetryMap,
    finite_difference_check,
    optimize,
)
from pinneapple_design.topology.magnetic import RotorTorqueProblem  # noqa: E402
from pinneapple_simulation.electromagnetics import LinearMaterial, illustrative_steel  # noqa: E402
from pinneapple_simulation.electromagnetics.machine import SPMMachine  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()

geom = {"n_theta": 240, "slots": 24, "pole_pairs": 2, "magnet_thickness": 0.0, "rotor_core_inner": 0.01,
        "rotor_core_layers": 12, "airgap": 0.0006}
m = SPMMachine(iron=LinearMaterial(1000.0), **geom)
thetas = np.radians([0.0, 5.0, 10.0])
gamma = math.radians(60.0)
prob = RotorTorqueProblem(m, thetas, I_peak=60.0, gamma=gamma, objective="mean")
sym = SymmetryMap.rotational(prob.centroids, 4, tol=1e-9)
filt = DensityFilter(prob.centroids, 0.0018, prob.volumes)

chain = DesignChain(prob.n, filt, sym, beta=2.0)


def fun(z):
    f, df = prob.evaluate(chain.forward(z))
    return f, chain.backward(df)


x_test = np.random.default_rng(0).uniform(0.3, 0.8, sym.n_var)
_, g = fun(x_test)
grad_err = float(finite_difference_check(fun, x_test, np.argsort(-np.abs(g))[:5], 1e-5).max())

T_solid = -prob.evaluate(np.ones(prob.n))[0]
res = optimize(prob, 0.5, filt, "mma", max_iter=120, symmetry=sym, beta0=1.0, beta_max=8.0, beta_every=25,
               move=0.2, objective_scale=1.0)
x = res.x_phys

# re-evaluation of the final design: per-position torques, co-energy check, nonlinear iron
T_lin = np.array([m.torque(m.solve(t, prob.currents(t), design=x)) for t in thetas])
T_coe = m.torque_coenergy(thetas[1], prob.currents(thetas[1]), design=x)
m_nl = SPMMachine(iron=illustrative_steel(), **geom)
T_nl = np.array([m_nl.torque(m_nl.solve(t, prob.currents(t), design=x)) for t in thetas])
summary = {
    "design_variables": sym.n_var, "design_elements": prob.n, "gradient_check_max_rel_err": grad_err,
    "solid_round_rotor_torque_Nm": T_solid, "optimized_mean_torque_linear_Nm": float(T_lin.mean()),
    "torque_per_position_linear_Nm": T_lin.tolist(), "coenergy_torque_at_5deg_Nm": T_coe,
    "optimized_mean_torque_nonlinear_Nm": float(T_nl.mean()), "iterations": len(res.history["objective"]),
    "volume_fraction": float(np.sum(prob.volumes * x) / np.sum(prob.volumes)), "runtime_s": time.time() - t0,
}

mesh = m.mesh(0.0)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].add_collection(PolyCollection(mesh.points[mesh.tris[prob.elements]], array=x, cmap="gray_r",
                                    edgecolors="none", clim=(0, 1)))
ax[0].set_aspect("equal")
ax[0].autoscale()
ax[0].set_title("Optimized rotor iron (black)")
ax[1].plot(-np.array(res.history["objective"]))
ax[1].set(xlabel="iteration", ylabel="mean torque (N m), linear iron", title="MMA history")
fig.tight_layout()
fig.savefig(OUT / "reluctance_rotor.png", dpi=120)
(OUT / "reluctance_rotor_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
