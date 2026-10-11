"""Black-hole hydrodynamics experiments (``pinneapple_physics.blackhole``)."""
from __future__ import annotations

import numpy as np

from ..spec import Experiment, register


@register
class BondiAccretion(Experiment):
    name = "bondi_accretion"
    version = "1"
    description = ("Spherical accretion onto a Schwarzschild black hole (Paczynski-Wiita potential): the hydro "
                   "solver started from the exact transonic Bondi solution must keep it steady.")
    tags = ["astrophysics", "hydro", "verification"]
    params = {"cs_inf": 0.1, "gamma": 1.4, "nr": 64, "t_end": 300.0, "backend": "numba"}
    space = {"cs_inf": (0.07, 0.2), "gamma": [1.2, 1.3, 1.4], "nr": [48, 64, 96]}

    def run(self, ctx):
        from pinneapple_physics.blackhole import AccretionFlow, RIAFConfig, bondi_pw
        p = ctx.params
        cfg = RIAFConfig(nr=p["nr"], ntheta=4, gamma=p["gamma"], viscosity="none", perturbation=0.0,
                         outer_bc="fixed", t_cap=None, backend=p["backend"])
        flow = AccretionFlow(cfg)
        r = flow.grid()["r"]
        rho, v, pr, mdot, rc = bondi_pw(r, cs_inf=p["cs_inf"], gamma=p["gamma"])
        W = np.zeros((5, len(r), 4))
        W[0], W[1], W[4] = rho[:, None], v[:, None], pr[:, None]
        flow.set_primitives(W)
        ctx.input("initial_profile", {"r": r, "rho": rho, "v": v, "p": pr, "mdot": mdot, "sonic_radius": rc})
        with ctx.stage("evolve"):
            out = flow.run(t_end=p["t_end"], every=p["t_end"])
        Wf = flow.primitives()
        rel = np.abs(Wf[0, :, 1] / rho - 1)
        ctx.metric("mdot_rel_error", abs(out["mdot"][-1] / mdot - 1))
        ctx.metric("density_rel_error_median", float(np.median(rel)))
        ctx.metric("density_rel_error_max", float(rel.max()))
        ctx.metric("sonic_radius", rc)
        ctx.check("steady_accretion_rate", value=abs(out["mdot"][-1] / mdot - 1), max=0.02,
                  detail="vs the exact transonic Bondi accretion rate", kind="reference")
        ctx.check("steady_density", value=float(np.median(rel)), max=0.02, detail="vs the exact Bondi profile",
                  kind="reference")
        ds = ctx.dataset("profiles", description="Radial Bondi profiles: exact and simulated")
        ds.add(r=r, rho_exact=rho, rho_sim=Wf[0, :, 1], v_exact=v, v_sim=Wf[1, :, 1], cs_inf=p["cs_inf"],
               gamma=p["gamma"], mdot=mdot)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.loglog(r, rho, "k-", label="exact")
        ax.loglog(r, Wf[0, :, 1], "--", color="#d95f02", label=f"solver, t={p['t_end']:g}")
        ax.axvline(rc, color="#999", lw=0.8)
        ax.set_xlabel("r [GM/c^2]")
        ax.set_ylabel("density")
        ax.legend(frameon=False)
        ctx.figure("bondi_profile", fig)


@register
class AccretionFlowRun(Experiment):
    name = "accretion_flow"
    version = "1"
    description = ("A hot torus accreting onto a Schwarzschild black hole (viscous 2.5-D hydro): density movie, "
                   "accretion rate and mass / angular-momentum budgets. Frames form a dataset for forecasting.")
    tags = ["astrophysics", "hydro", "dataset"]
    limitations = ["Paczynski-Wiita pseudo-Newtonian gravity, 2.5-D, no magnetic fields", "coarse grid"]
    params = {"alpha": 0.1, "viscosity": "SS", "torus_a": 0.0, "nr": 64, "ntheta": 32, "t_end": 600.0,
              "every": 20.0, "backend": "numba", "seed": 0}
    space = {"alpha": (0.02, 0.3), "viscosity": ["SS", "ST"], "torus_a": (0.0, 0.25)}

    def run(self, ctx):
        from pinneapple_physics.blackhole import AccretionFlow, RIAFConfig, torus_state
        p = ctx.params
        alpha = p["alpha"] if p["viscosity"] == "SS" else p["alpha"] * 0.1      # ST nu = alpha sqrt(r) is stronger
        cfg = RIAFConfig(nr=p["nr"], ntheta=p["ntheta"], alpha=alpha, viscosity=p["viscosity"], torus_a=p["torus_a"],
                         seed=p["seed"], backend=p["backend"])
        flow = AccretionFlow(cfg)
        flow.set_primitives(torus_state(flow))
        b0 = flow.budget()
        with ctx.stage("evolve"):
            out = flow.run(t_end=p["t_end"], every=p["every"])
        b1 = flow.budget()
        g = flow.grid()
        ctx.output("grid", {k: v for k, v in g.items()})
        ctx.output("mdot", out["mdot"])
        ctx.metric("mdot_final", float(out["mdot"][-1]))
        ctx.metric("mdot_mean", float(np.mean(out["mdot"][1:])))
        ctx.metric("capped_cells_last_step", int(flow.capped))
        dm = abs(b1["mass"] - b0["mass"]) / b0["mass"]
        dl = abs(b1["angmom"] - b0["angmom"]) / b0["angmom"]
        ctx.metric("mass_budget_error", dm)
        ctx.metric("angmom_budget_error", dl)
        # the caps on near-empty cells may act; the budgets must still close to well below a percent
        ctx.check("mass_budget", value=dm, max=1e-3)
        ctx.check("angular_momentum_budget", value=dl, max=1e-3)
        ctx.check("accretes", value=float(np.mean(out["mdot"][1:])), min=0.0)
        ds = ctx.dataset("frames", description="Primitive fields (rho, v_r, v_theta, v_phi, p) on the (r, theta) grid",
                         units={"frame": "G=M=c=1", "t": "GM/c^3"}, shard_size=64)
        for t, fr, md in zip(out["t"], out["frames"], out["mdot"], strict=True):
            ds.add(frame=fr.astype(np.float32), t=float(t), mdot=float(md), alpha=float(p["alpha"]),
                   viscosity=p["viscosity"], torus_a=float(p["torus_a"]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rf, tf = g["r_faces"], g["theta_faces"]
        R, Z = rf[:, None] * np.sin(tf)[None], rf[:, None] * np.cos(tf)[None]
        idx = np.linspace(0, len(out["t"]) - 1, 4).astype(int)
        fig, ax = plt.subplots(1, 4, figsize=(10, 3.2))
        for a_, i in zip(ax, idx, strict=True):
            a_.pcolormesh(R, Z, np.log10(out["frames"][i, 0]), vmin=-5, vmax=0, cmap="inferno")
            a_.set_xlim(0, 70)
            a_.set_ylim(-50, 50)
            a_.set_aspect("equal")
            a_.set_title(f"t = {out['t'][i]:.0f}", fontsize=9)
            a_.set_xticks([])
            a_.set_yticks([])
        ctx.figure("density", fig)
        frames = []
        cmap = plt.get_cmap("inferno")
        for fr in out["frames"][:: max(1, len(out["frames"]) // 30)]:
            z = np.clip((np.log10(fr[0]) + 5) / 5, 0, 1)
            frames.append((cmap(np.flipud(z.T))[..., :3] * 255).astype(np.uint8))
        ctx.gif("density_rtheta", frames, duration_ms=120)
        ctx.metric("t_end", float(out["t"][-1]))
        ctx.metric("frames", int(len(out["t"])))
