"""Solids suspension in a stirred tank: Lagrangian particles (soft-sphere DEM, drag, buoyancy, turbulent dispersion)
in an analytic stirred-tank flow (``pinneapple_simulation.numerical_solvers.particles``), rendered as a process
video: thousands of spheres coloured by speed in a glass tank with a turning pitched-blade impeller, next to charts
that draw themselves (particles in the top third, stirrer speed) (``pinneapple_tools.visualization.studio.particles``).

    python -m pinneapple_lab run particle_suspension
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np

from ..spec import Experiment, register


@register
class ParticleSuspension(Experiment):
    name = "particle_suspension"
    version = "1"
    description = ("Solids suspension in a stirred tank as the impeller speeds up: soft-sphere DEM particles with "
                   "drag, buoyancy and turbulent dispersion in an analytic, divergence-free stirred-tank flow; process "
                   "video (particles coloured by speed, turning impeller, live charts), suspension curves and the "
                   "just-suspended speed against Zwietering's correlation.")
    tags = ["particles", "dem", "mixing", "stirred-tank", "process", "3d", "video", "dataset"]
    references = ["T. N. Zwietering, Chem. Eng. Sci. 8 (1958) 244 (just-suspended speed)",
                  "L. Schiller and A. Naumann, Z. Ver. Dtsch. Ing. 77 (1933) 318 (drag)",
                  "P. A. Cundall and O. D. L. Strack, Geotechnique 29 (1979) 47 (DEM)"]
    limitations = ["one-way coupling: the particles do not change the liquid flow (no hindered settling, no "
                   "damping of the loop by the solids)", "analytic stirred-tank flow (swirl + one circulation loop "
                   "scaled by the tip speed), not a CFD solution of this tank; its constants are not calibrated",
                   "frictionless soft spheres, no particle-impeller collisions, no added mass or lift",
                   "contact stiffness lowered for the time step (overlaps up to about 15 % of the diameter)"]
    params = {"n": 12000, "d_mm": 3.0, "rho_p": 1200.0, "rpm_max": 400.0, "ramp_s": 25.0, "t_end": 30.0,
              "dt": 2.5e-4, "frame_every": 0.25, "render": True, "samples": 16, "size": [800, 640], "seed": 0}

    def run(self, ctx):
        from pinneapple_simulation.numerical_solvers.particles import (
            DEM,
            StirredTank,
            suspend,
            terminal_velocity,
            zwietering_njs,
        )
        p = ctx.params
        d = p["d_mm"] * 1e-3
        tank = StirredTank()
        ctx.input("tank", {"R": tank.R, "H": tank.H, "D_imp": tank.D_imp, "C": tank.C, "blades": tank.blades,
                           "pitch_deg": tank.pitch_deg, "swirl": tank.swirl, "loop": tank.loop,
                           "turbulence": tank.turbulence})
        # 1. the pieces against exact answers
        x = np.random.default_rng(0).uniform([-tank.R, -tank.R, 0.01 * tank.H], [tank.R, tank.R, 0.99 * tank.H], (4000, 3))
        x = x[np.hypot(x[:, 0], x[:, 1]) < 0.98 * tank.R]
        h = 1e-6 * tank.R
        div = sum((tank.velocity(x + h * np.eye(3)[k], 300)[:, k] - tank.velocity(x - h * np.eye(3)[k], 300)[:, k])
                  / (2 * h) for k in range(3))
        V = tank.tip_speed(300)
        ctx.check("flow_is_divergence_free", value=float(np.abs(div).max() * tank.R / V), max=1e-4,
                  detail="max |div u| R / V_tip by central differences (stream-function loop)", kind="physics")
        vt = terminal_velocity(d, p["rho_p"])
        one = DEM(np.array([[0.0, 0.0, 0.8 * tank.H]]), d, p["rho_p"], dt=p["dt"])
        for _ in range(int(2.0 / p["dt"])):
            one.step(np.zeros((1, 3)), 0.0, 1.0)
        ctx.metric("terminal_velocity_m_s", vt)
        ctx.check("settling_velocity_vs_schiller_naumann", value=float(-one.v[0, 2]), reference=vt, rtol=1e-3,
                  detail="an isolated sphere in still liquid", kind="reference")
        # 2. the suspension run
        ramp = p["rpm_max"] / p["ramp_s"]
        rpm = lambda t: min(p["rpm_max"], ramp * t)                       # noqa: E731
        with ctx.stage("simulate"):
            res = suspend(tank, n=int(p["n"]), d=d, rho_p=p["rho_p"], rpm=rpm, t_end=p["t_end"], dt=p["dt"],
                          frame_every=p["frame_every"], seed=int(p["seed"]), log=ctx.log)
        T, N, top, sus = res["times"], res["rpm"], res["top_fraction"], res["suspended_fraction"]
        xs = np.stack([f["x"] for f in res["frames"]])
        r_ = np.hypot(xs[..., 0], xs[..., 1])
        inside = float(np.mean((r_ <= tank.R + 0.5 * d) & (xs[..., 2] >= -0.5 * d) & (xs[..., 2] <= tank.H + 0.5 * d)))
        ctx.check("particles_stay_in_the_tank", value=inside, min=1.0, detail="fraction of particle positions inside "
                  "the tank over all frames", kind="physics")
        ctx.check("contact_overlap_bounded", value=float(res["max_overlap"].max()), max=0.25,
                  detail="largest contact overlap / diameter (soft spheres)", kind="sanity")
        ms = int(p["n"]) * p["rho_p"] * math.pi * d ** 3 / 6
        ml = 998.0 * math.pi * tank.R ** 2 * tank.H
        njs = {S: zwietering_njs(tank, d, p["rho_p"], ms / ml, S=S) for S in (4.0, 6.0, 8.0)}
        k90 = np.nonzero(sus >= 0.9)[0]
        n90 = float(N[k90[0]]) if len(k90) else float("nan")
        ctx.metric("rpm_90pct_suspended", n90)
        ctx.metric("zwietering_njs_S4_rpm", njs[4.0])
        ctx.metric("zwietering_njs_S8_rpm", njs[8.0])
        ctx.metric("top_third_fraction_end", float(top[-1]))
        ctx.metric("suspended_fraction_end", float(sus[-1]))
        ctx.metric("solids_mass_ratio", ms / ml)
        ctx.check("suspension_speed_vs_zwietering", value=n90, min=0.7 * njs[4.0], max=1.3 * njs[8.0],
                  detail="speed with 90 % of the solids off the bottom, against Zwietering's N_js for S = 4 to 8 "
                         "(pitched-blade turbines); an order-of-magnitude consistency, the flow model is not "
                         "calibrated", kind="reference")
        ramp_part = T <= p["ramp_s"]
        ctx.check("suspension_grows_with_speed", value=float(np.corrcoef(N[ramp_part], sus[ramp_part])[0, 1]), min=0.8,
                  detail="correlation of the suspended fraction with the impeller speed during the ramp",
                  kind="physics")
        ds = ctx.dataset("suspension", description="Particle positions and speeds every 4th frame, with time, "
                         "impeller speed and suspension measures", units={"x": "m", "speed": "m/s", "t": "s"})
        for k in range(0, len(T), 4):
            ds.add(t=float(T[k]), rpm=float(N[k]), x=res["frames"][k]["x"], speed=res["frames"][k]["speed"],
                   top_fraction=float(top[k]), suspended_fraction=float(sus[k]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(T, 100 * top, color="#e07b39", label="particles in the top third")
        ax[0].plot(T, 100 * sus, color="#7570b3", label="particles off the bottom bed")
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("%")
        ax[0].legend(frameon=False)
        ax[1].plot(N[ramp_part], 100 * sus[ramp_part], color="#7570b3")
        ax[1].axvspan(njs[4.0], njs[8.0], color="0.9", label="Zwietering N_js, S = 4 to 8")
        ax[1].axhline(90, color="k", lw=0.6, ls="--")
        ax[1].set_xlabel("impeller speed (rpm)")
        ax[1].set_ylabel("off the bottom (%)")
        ax[1].legend(frameon=False)
        fig.tight_layout()
        ctx.figure("suspension_curves", fig)
        if not p["render"]:
            return
        with ctx.stage("render"):
            ang = np.r_[0.0, np.cumsum(np.diff(T) * 0.5 * (N[1:] + N[:-1]) / 60 * 2 * np.pi)]
            from pinneapple_tools.visualization.studio.particles import (
                compose_video,
                render_particle_frames,
            )
            lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
            work = os.path.join(lab_root, "_cases", os.path.basename(ctx.dir))
            vmax = 0.4
            pngs = render_particle_frames(res["frames"], d, os.path.join(work, "frames"),
                                          equipment=lambda k: tank.surfaces(float(ang[k])), field_range=(0, vmax),
                                          samples=int(p["samples"]), size=tuple(p["size"]), log=ctx.log)
            out = ctx.path("figures", "stirred_tank.gif")
            compose_video(pngs, T, out, charts=[("Particles Top [%]", T, 100 * top), ("Stirrer Speed [RPM]", T, N)],
                          colorbar=("Velocity Magnitude (m/s)", 0, vmax), duration_ms=int(1000 * p["frame_every"] / 2),
                          frame_dir=os.path.join(work, "composed"), mp4=True)
            ctx.files["figures"].append(os.path.relpath(out, ctx.dir))
            mp4 = os.path.splitext(out)[0] + ".mp4"
            if os.path.exists(mp4):
                ctx.files.setdefault("outputs", []).append(os.path.relpath(mp4, ctx.dir))
            for k, nm in ((len(pngs) // 3, "early"), (len(pngs) - 1, "end")):
                dst = ctx.path("figures", f"stirred_tank_{nm}.png")
                shutil.copy(os.path.join(work, "composed", f"composed_{k:04d}.png"), dst)
                ctx.files["figures"].append(os.path.relpath(dst, ctx.dir))
            shutil.rmtree(work, ignore_errors=True)
