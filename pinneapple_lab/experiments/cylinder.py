"""Lattice-Boltzmann flow past a cylinder: regime, Strouhal number and labelled vorticity images."""
from __future__ import annotations

import numpy as np

from ..spec import Experiment, register


@register
class CylinderLBM(Experiment):
    name = "cylinder_lbm"
    version = "2"                    # 2: Strouhal from the windowed, peak-refined spectrum (was the raw FFT bin)
    description = ("2D channel flow past a cylinder with the D2Q9 lattice-Boltzmann solver "
                   "(pinneapple_simulation.numerical_solvers.cylinder_wake). Measures the wake regime (steady or vortex "
                   "shedding) and the Strouhal number; saves vorticity snapshots labelled by regime and Re, a "
                   "dataset for vision (classification / clustering of flow regimes, #415).")
    tags = ["cfd", "lbm", "dataset", "vision"]
    limitations = ["without the Smagorinsky model the solver is unstable from Re 140 (failed runs kept)"]
    params = {"Re": 100.0, "D": 20, "height": 5, "length": 12, "u_in": 0.1, "Cs": 0.0, "steps": 16000,
              "save_every": 100, "seed": 0}
    space = {"Re": ("log", 10.0, 200.0)}

    def run(self, ctx):
        from pinneapple_simulation.numerical_solvers.cylinder_wake import simulate
        p = ctx.params
        D = int(p["D"])
        r = simulate(p["Re"], D=D, height=int(p["height"]), length=int(p["length"]), u_in=p["u_in"],
                     conv_time=p["steps"] * p["u_in"] / D, keep_time=None, save_dt=p["save_every"] * p["u_in"] / D,
                     Cs=p["Cs"])
        ctx.input("setup", {"nx": r["nx"], "ny": r["ny"], "cylinder_center": r["cylinder_center"], "diameter": D,
                            "Re_diameter": p["Re"], "Re_channel_solver": p["Re"] * (r["ny"] - 2) / D,
                            "u_in": p["u_in"], "smagorinsky_Cs": p["Cs"], "tau": r["tau"]})
        w, regime, st = r["vorticity"], r["regime"], r["strouhal"]
        ctx.metric("wake_amplitude", r["wake_amplitude"])
        ctx.metric("strouhal", st)
        ctx.metric("regime_shedding", int(regime == "shedding"))
        ctx.check("finite_fields", r["finite"])
        if regime == "shedding":
            # confined cylinder (blockage 1/5): St ~ 0.15 - 0.35 over this Re range (confinement raises it)
            ctx.check("strouhal_in_physical_range", value=st, min=0.12, max=0.35)
        if p["Re"] < 40:
            ctx.check("steady_below_onset", regime == "steady", detail="no shedding expected below Re ~ 47")
        ds = ctx.dataset("vorticity", description="Vorticity snapshots (normalised by U/D) labelled by regime and Re",
                         units={"vorticity": "U/D"}, shard_size=64)
        for i, ti in enumerate(r["t"]):
            ds.add(vorticity=w[i].astype(np.float16), Re=float(p["Re"]), regime=regime,
                   step=int(round(ti * D / p["u_in"])), strouhal=st)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.imshow(w[-1].T, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"Re = {p['Re']:.0f}: {regime}" + (f", St = {st:.3f}" if regime == "shedding" else ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ctx.figure("vorticity", fig)
        cmap = plt.get_cmap("RdBu_r")
        frames = [(cmap(np.clip((np.flipud(x.T) + 3) / 6, 0, 1))[..., :3] * 255).astype(np.uint8)
                  for x in w[::2]]                         # w is already the second half of the run
        ctx.gif("vorticity", frames, duration_ms=80)
