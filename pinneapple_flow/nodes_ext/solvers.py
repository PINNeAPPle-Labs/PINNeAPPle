"""Numerical-solver nodes from ``examples/numerical_solvers``: the NVE Lennard-Jones gas
(:class:`pinneapple_simulation.particle_dynamics.molecular_dynamics.MolecularDynamicsSimulator`, velocity Verlet,
periodic box) and the shallow-water dam break with a gate and an obstacle
(:class:`pinneapple_simulation.numerical_solvers.shallow_water_fv.ShallowWater2D`, MUSCL-HLL)."""
from __future__ import annotations

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node


@node("md.lj_gas", "Lennard-Jones gas (NVE molecular dynamics)", category="Particles",
      outputs=[Port("metrics", "metrics"), Port("series", "series")],
      params=[Param("dim", "int", 2, minimum=2, maximum=3), Param("n_side", "int", 6, minimum=2, maximum=30),
              Param("epsilon", "float", 1.0), Param("sigma", "float", 1.0), Param("box_size", "float", 15.0),
              Param("cutoff", "float", 2.5, help="in units of sigma"), Param("steps", "int", 500, minimum=1),
              Param("dt", "float", 1e-3, minimum=1e-6), Param("kick", "float", 0.05, help="velocity noise"),
              Param("seed", "int", 0)],
      calls="pinneapple_simulation.particle_dynamics.molecular_dynamics.MolecularDynamicsSimulator + MDState")
def lj_gas(ctx, inputs, params):
    """Particles on a lattice at the LJ well separation with a small random kick; velocity-Verlet steps; total
    energy against its initial value (examples/numerical_solvers/12_molecular_dynamics_lj_gas.py)."""
    import torch

    from pinneapple_simulation.particle_dynamics.molecular_dynamics import (
        MDState,
        MolecularDynamicsSimulator,
    )
    torch.manual_seed(int(params["seed"]))
    dim, n, s = int(params["dim"]), int(params["n_side"]), float(params["sigma"])
    sim = MolecularDynamicsSimulator(epsilon=float(params["epsilon"]), sigma=s, mass=1.0,
                                     cutoff=float(params["cutoff"]) * s, box_size=float(params["box_size"]), dim=dim)
    torch.manual_seed(int(params["seed"]))
    c1 = torch.arange(n, dtype=torch.float32) * (2.0 ** (1.0 / 6.0) * s) + 2.0
    pos = torch.stack([g.reshape(-1) for g in torch.meshgrid(*([c1] * dim), indexing="ij")], dim=-1)
    state = MDState(pos=pos, vel=float(params["kick"]) * torch.randn_like(pos))
    e0 = sim.total_energy(state).item()
    t, E = [0.0], [e0]
    dt = float(params["dt"])
    for k in range(1, int(params["steps"]) + 1):
        state = sim.step(state, dt)
        if k % max(1, int(params["steps"]) // 50) == 0:
            t.append(k * dt)
            E.append(sim.total_energy(state).item())
    ef = sim.total_energy(state).item()
    ctx.metric("n_particles", int(state.n_particles))
    ctx.metric("E_initial", e0)
    ctx.metric("E_final", ef)
    ctx.metric("max_abs_dE", float(np.max(np.abs(np.asarray(E) - e0))))
    ctx.metric("relative_energy_drift", abs(ef - e0) / max(abs(e0), 1.0), "|E_final - E0| / max(|E0|, 1)")
    inside = bool(torch.all(state.pos >= 0.0) and torch.all(state.pos < float(params["box_size"])))
    ctx.metric("inside_box", int(inside))
    a = ctx.save_npz("energy", t=np.asarray(t), values=np.asarray(E)[:, None])
    a["channels"] = ["E_total"]
    ctx.summary(f"{state.n_particles} particles, energy drift {abs(ef - e0) / max(abs(e0), 1.0):.2e}")
    return {"metrics": ctx.report["metrics"], "series": a}


@node("swe.dam_break", "Shallow-water dam break with a gate and an obstacle", category="Numerical solvers",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("nx", "int", 120, minimum=8, maximum=2000), Param("ny", "int", 30, minimum=4, maximum=2000),
              Param("length", "float", 3.0), Param("reservoir_depth", "float", 0.5),
              Param("gate_x", "float", 1.0), Param("release_time", "float", 0.1),
              Param("obstacle", "list", [2.2, 2.4, 0.3, 0.6], help="x0, x1, y0, y1 of the submerged pillar"),
              Param("t_end", "float", 1.2, minimum=0.01), Param("snapshots", "int", 30, minimum=2, maximum=500)],
      calls="pinneapple_simulation.numerical_solvers.shallow_water_fv.ShallowWater2D (add_gate, add_wall, run, "
            "health_report)")
def dam_break(ctx, inputs, params):
    """examples/numerical_solvers/13_shallow_water_twin3d_browser_demo.py without the Twin3D export and its blocking
    server: the run and its health report (volume conservation, positivity, finiteness)."""
    from pinneapple_simulation.numerical_solvers.shallow_water_fv import ShallowWater2D
    nx, ny, L = int(params["nx"]), int(params["ny"]), float(params["length"])
    sw = ShallowWater2D(nx, ny, L / nx)
    X, Y = sw.centers()
    gx = float(params["gate_x"])
    sw.h[X < gx] = float(params["reservoir_depth"])
    sw.add_gate("gate", (X > gx) & (X < gx + 2 * sw.dx), release_time=float(params["release_time"]))
    x0, x1, y0, y1 = map(float, params["obstacle"])
    obstacle = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    sw.add_wall(obstacle)
    times = np.linspace(0.0, float(params["t_end"]), int(params["snapshots"]))
    H = []
    for t in times:
        if t > 0:
            sw.run(t)
        H.append(sw.h.copy())
    rep = sw.health_report()
    for k, v in rep.items():
        if isinstance(v, (bool, np.bool_)):
            ctx.metric(f"health.{k}", int(bool(v)))
        elif isinstance(v, (int, float)):
            ctx.metric(f"health.{k}", float(v))
    if not rep.get("ok", False):
        raise StepFailed(f"solver health check failed: {rep}")
    ctx.metric("max_depth_final", float(np.max(sw.h)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(8, 6.5))
    for ax, k in zip(axs, (0, len(times) // 2, len(times) - 1), strict=True):
        im = ax.pcolormesh(X[:, 0], Y[0], np.ma.masked_where(sw.solid, H[k]).T, shading="auto", cmap="Blues",
                           vmin=0, vmax=float(params["reservoir_depth"]))
        ax.set_title(f"depth at t = {times[k]:.2f} s", fontsize=9)
        ax.set_aspect("equal")
    fig.colorbar(im, ax=list(axs), label="h [m]")
    f = ctx.figure(fig, "dam_break", "depth snapshots")
    ctx.summary(f"health ok; {len(times)} snapshots to {times[-1]:g} s")
    return {"metrics": ctx.report["metrics"], "figure": f}


def _newton_available():
    import importlib.util
    return (True, "") if importlib.util.find_spec("newton") else (False, "needs NVIDIA Newton (pip install newton)")


@node("newton.crate_and_pendulum", "Rigid bodies: a sliding crate and a pendulum (NVIDIA Newton)",
      category="Particles", outputs=[Port("metrics", "metrics")],
      params=[Param("mu", "float", 0.4, minimum=0.0, maximum=2.0, help="friction of the crate and the ground"),
              Param("v0", "float", 2.0, help="initial crate speed [m/s]"),
              Param("pendulum_deg", "float", 20.0), Param("duration", "float", 2.0, minimum=0.01),
              Param("dt", "float", 1e-3, minimum=1e-5)],
      calls="pinneapple_simulation.external_solvers.newton_physics.RigidScene (ground, box, sphere, revolute) + "
            "simulate", available=_newton_available)
def crate_and_pendulum(ctx, inputs, params):
    """examples/rigid_body/newton_quickstart.py: where the crate stops against Coulomb friction, v0^2 / (2 mu g)."""
    from pinneapple_simulation.external_solvers.newton_physics import RigidScene, simulate
    mu, v0 = float(params["mu"]), float(params["v0"])
    scene = RigidScene().ground(mu=mu)
    scene.box("crate", (0.1, 0.1, 0.1), pos=(0, 0, 0.1), vel=(v0, 0, 0), mu=mu)
    a = np.radians(float(params["pendulum_deg"]))
    bob = scene.sphere("bob", 0.05, pos=(1.0 + np.sin(a), 0, 2.0 - np.cos(a)))
    scene.revolute(-1, bob, anchor=(1.0, 0, 2.0), axis=(0, 1, 0))
    traj = simulate(scene, duration=float(params["duration"]), dt=float(params["dt"]))
    x = float(traj.pos[-1, traj.index("crate"), 0])
    coulomb = v0 ** 2 / (2 * mu * 9.81)
    ctx.metric("crate_x_final_m", x, "where the crate came to rest")
    ctx.metric("coulomb_stop_m", coulomb, "v0^2 / (2 mu g)")
    ctx.metric("crate_over_coulomb", x / coulomb)
    ctx.metric("wall_time_s", float(traj.wall_time_s))
    ctx.summary(f"crate stopped at {x:.3f} m (Coulomb {coulomb:.3f} m)")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="md.lj_gas", title="Lennard-Jones gas: NVE molecular dynamics with energy conservation",
        domain="numerical solvers", example="numerical_solvers",
        keywords={"lennard-jones": 3, "lennard jones": 3, "molecular dynamics": 3, "nve": 2, "velocity verlet": 2,
                  "lj gas": 3}, min_score=3, nodes=("md.lj_gas",), references=("hairer1996odes2",),
        template="numerical_solvers/12_molecular_dynamics_lj_gas",
        description="examples/numerical_solvers/12_molecular_dynamics_lj_gas.py"))
    register_capability(Capability(
        id="swe.dam_break", title="Shallow-water dam break (finite volumes)", domain="numerical solvers",
        example="numerical_solvers",
        keywords={"dam break": 3, "dam-break": 3, "shallow water": 3, "shallow-water": 3, "saint-venant": 2.5,
                  "flood": 1.5, "gate release": 2}, min_score=3, nodes=("swe.dam_break",),
        references=("leveque2007fd",), template="numerical_solvers/13_shallow_water_dam_break",
        description="examples/numerical_solvers/13_shallow_water_twin3d_browser_demo.py (without the server)"))
    register_capability(Capability(
        id="newton.rigid_body", title="Rigid bodies with NVIDIA Newton (validation and a small scene)",
        domain="multibody", example="rigid_body",
        keywords={"rigid body": 3, "rigid bodies": 3, "rigid-body": 3, "newton": 1.5, "friction": 1.5, "pendulum": 1.5, "crate": 2,
                  "sliding": 1.5, "contact": 1}, min_score=3, nodes=("solver.newton", "newton.crate_and_pendulum"),
        uncited="closed forms (Coulomb friction, pendulum period) are the checks", template="rigid_body/newton_quickstart",
        description="examples/rigid_body/newton_quickstart.py"))


_register_capabilities()
