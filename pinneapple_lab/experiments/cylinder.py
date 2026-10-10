"""Lattice-Boltzmann flow past a cylinder: regime, Strouhal number and labelled vorticity images."""
from __future__ import annotations

import numpy as np

from ..spec import Experiment, register


@register
class CylinderLBM(Experiment):
    name = "cylinder_lbm"
    version = "1"
    description = ("2D channel flow past a cylinder with the D2Q9 lattice-Boltzmann solver "
                   "(pinneapple_simulation.numerical_solvers.lbm). Measures the wake regime (steady or vortex "
                   "shedding) and the Strouhal number; saves vorticity snapshots labelled by regime and Re, a "
                   "dataset for vision (classification / clustering of flow regimes, #415).")
    tags = ["cfd", "lbm", "dataset", "vision"]
    params = {"Re": 100.0, "nx": 240, "ny": 80, "u_in": 0.06, "steps": 12000, "save_every": 100, "seed": 0}
    space = {"Re": ("log", 10.0, 300.0)}

    def run(self, ctx):
        import torch

        from pinneapple_simulation.numerical_solvers.lbm import LBMSolver, cylinder_mask
        p = ctx.params
        nx, ny = p["nx"], p["ny"]
        r = ny / 10.0
        cx, cy = nx / 5.0, ny / 2.0 + 1.0                       # a small offset triggers shedding
        mask = cylinder_mask(nx, ny, cx, cy, r)
        # Re is defined on the cylinder diameter: rescale to the solver's channel-height definition
        re_channel = p["Re"] * (ny - 2) / (2 * r)
        solver = LBMSolver(nx=nx, ny=ny, Re=re_channel, u_in=p["u_in"], obstacle_mask=mask)
        ctx.input("setup", {"nx": nx, "ny": ny, "cylinder_center": [cx, cy], "cylinder_radius": r,
                            "Re_diameter": p["Re"], "Re_channel_solver": re_channel, "u_in": p["u_in"]})
        with ctx.stage("simulate"), torch.no_grad():
            out = solver(steps=p["steps"], save_every=p["save_every"])
        ux = out.extras["trajectory_ux"]
        uy = out.extras["trajectory_uy"]
        ux = np.asarray(torch.stack(ux) if isinstance(ux, list) else ux, dtype=np.float32)
        uy = np.asarray(torch.stack(uy) if isinstance(uy, list) else uy, dtype=np.float32)
        # vorticity
        w = (np.gradient(uy, axis=1) - np.gradient(ux, axis=2)) / p["u_in"] * (2 * r)
        solid = mask.numpy().astype(bool)
        w[:, solid] = 0.0
        # wake probe: transverse velocity downstream, second half of the run
        probe = uy[:, int(cx + 4 * r), int(cy)]
        half = probe[len(probe) // 2:]
        amp = float(np.std(half) / p["u_in"])
        dt_save = p["save_every"]
        spec = np.abs(np.fft.rfft(half - half.mean()))
        freqs = np.fft.rfftfreq(len(half), d=dt_save)
        k = int(np.argmax(spec[1:]) + 1) if len(spec) > 2 else 0
        f = float(freqs[k]) if k else 0.0
        st = f * 2 * r / p["u_in"]
        regime = "shedding" if amp > 0.02 else "steady"
        ctx.metric("wake_amplitude", amp)
        ctx.metric("strouhal", st if regime == "shedding" else 0.0)
        ctx.metric("regime_shedding", int(regime == "shedding"))
        ctx.check("finite_fields", bool(np.isfinite(ux).all() and np.isfinite(uy).all()))
        if regime == "shedding":
            # confined cylinder (blockage 0.2): St ~ 0.15 - 0.30 over this Re range
            ctx.check("strouhal_in_physical_range", value=st, min=0.12, max=0.35)
        if p["Re"] < 40:
            ctx.check("steady_below_onset", regime == "steady", detail="no shedding expected below Re ~ 47")
        ds = ctx.dataset("vorticity", description="Vorticity snapshots (normalised by U/D) labelled by regime and Re",
                         units={"vorticity": "U/D"}, shard_size=64)
        for i in range(len(w) // 2, len(w)):
            ds.add(vorticity=w[i].astype(np.float16), Re=float(p["Re"]), regime=regime, step=int(i * dt_save),
                   strouhal=st if regime == "shedding" else 0.0)
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
                  for x in w[len(w) // 2::2]]
        ctx.gif("vorticity", frames, duration_ms=80)
