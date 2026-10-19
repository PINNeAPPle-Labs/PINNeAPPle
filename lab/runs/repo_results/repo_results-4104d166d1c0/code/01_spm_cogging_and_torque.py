"""Surface-PM machine (4 poles, 24 slots, nonlinear iron): cogging torque, torque vs current angle,
back-EMF and losses with the 2D A_z FEM of pinneapple_simulation.electromagnetics.

Torque is computed twice -- Maxwell stress (Arkkio) and virtual work (co-energy differences) -- and the
two are printed side by side. Dimensions, materials and loss coefficients are illustrative.
Runs in about two minutes on one CPU core; writes PNGs and a JSON summary to ./results/.

    python examples/motor_magnetics/01_spm_cogging_and_torque.py
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

from pinneapple_simulation.electromagnetics.machine import SPMMachine  # noqa: E402
from pinneapple_simulation.electromagnetics.postprocess import IronLossModel, back_emf  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

t0 = time.time()
m = SPMMachine(n_theta=360, slot_opening=2 / 15, slot_width=0.4)  # edges on the 1-degree grid
I_RATED = 30.0  # A peak, 20 turns per slot
summary = {"machine": {"pole_pairs": m.pole_pairs, "slots": m.slots, "length_m": m.length,
                       "airgap_m": m.airgap, "Br_T": m.Br, "n_theta": m.n_theta}}

# --- cogging torque over one period -------------------------------------------------------------
period = m.cogging_period()
th_cog = np.linspace(0.0, period, 16, endpoint=False)
T_cog = m.cogging(th_cog)
check = [math.radians(d) for d in (1.3, 4.1, 9.7)]
cog_pairs = [(math.degrees(t), m.torque(m.solve(t)), m.torque_coenergy(t, (0.0, 0.0, 0.0))) for t in check]
summary["cogging"] = {"period_deg": math.degrees(period), "peak_to_peak_Nm": float(np.ptp(T_cog)),
                      "maxwell_vs_coenergy_Nm": cog_pairs}

# --- torque vs current angle --------------------------------------------------------------------
theta0 = math.radians(1.3)
gammas = np.radians(np.arange(-90, 91, 15))
T_gam = m.torque_vs_current_angle(I_RATED, gammas, theta=theta0)
load_pairs = []
for g in (-30.0, 0.0, 30.0):
    cur = m.phase_currents(theta0, I_RATED, math.radians(g))
    load_pairs.append((g, m.torque(m.solve(theta0, cur)), m.torque_coenergy(theta0, cur)))
summary["torque_vs_gamma"] = {"gamma_deg": np.degrees(gammas).tolist(), "torque_Nm": T_gam.tolist(),
                              "maxwell_vs_coenergy_Nm": load_pairs}

# --- flux linkage and back-EMF at 1500 rpm ------------------------------------------------------
rpm = 1500.0
steps = 72
th_e = (2 * math.pi / m.pole_pairs) * np.arange(steps) / steps
psi = m.no_load_flux(th_e)
emf = back_emf(psi, th_e, rpm * 2 * math.pi / 60)
summary["back_emf"] = {"rpm": rpm, "psi_peak_Wb": float(np.abs(psi).max()), "emf_peak_V": float(np.abs(emf).max()),
                       "emf_rms_V": float(np.sqrt(np.mean(emf[:, 0] ** 2)))}

# --- losses at rated current, 1500 rpm ----------------------------------------------------------
model = IronLossModel(k_h=0.02, alpha=1.0, beta=2.0, k_c=5e-5, k_e=1e-4, density=7650.0)  # illustrative
loss = m.losses(rpm, I_RATED, 0.0, model, steps=24)
summary["losses_1500rpm_rated"] = {k: v for k, v in loss.items() if k not in ("torque", "thetas")}
summary["runtime_s"] = time.time() - t0

# --- plots --------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(np.degrees(th_cog), T_cog, "o-", label="Maxwell stress")
ax[0].plot([p[0] for p in cog_pairs], [p[2] for p in cog_pairs], "s", label="co-energy")
ax[0].set(xlabel="rotor angle (deg)", ylabel="cogging torque (N m)", title="Cogging")
ax[0].legend()
ax[1].plot(np.degrees(gammas), T_gam, "o-", label="Maxwell stress")
ax[1].plot([p[0] for p in load_pairs], [p[2] for p in load_pairs], "s", label="co-energy")
ax[1].set(xlabel="current angle from q-axis (deg)", ylabel="torque (N m)", title=f"I = {I_RATED:g} A peak")
ax[1].legend()
for k, name in enumerate("ABC"):
    ax[2].plot(np.degrees(m.pole_pairs * th_e), emf[:, k], label=name)
ax[2].set(xlabel="electrical angle (deg)", ylabel="back-EMF (V)", title=f"{rpm:g} rpm")
ax[2].legend()
fig.tight_layout()
fig.savefig(OUT / "spm_torque_cogging_emf.png", dpi=120)

ms = m.solve(theta0, m.phase_currents(theta0, I_RATED, 0.0))
fig, ax = plt.subplots(figsize=(6, 6))
pc = PolyCollection(ms.mesh.points[ms.mesh.tris], array=np.hypot(*ms.B.T), cmap="viridis", edgecolors="none")
ax.add_collection(pc)
ax.tricontour(ms.mesh.points[:, 0], ms.mesh.points[:, 1], ms.mesh.tris, ms.A, 16, colors="w", linewidths=0.4)
fig.colorbar(pc, ax=ax, label="|B| (T)")
ax.set_aspect("equal")
ax.autoscale()
ax.set_title("Rated load, flux lines and |B|")
fig.savefig(OUT / "spm_flux_density.png", dpi=120)

(OUT / "spm_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
