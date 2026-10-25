"""Black-hole accretion nodes (``examples/black_hole_weather``): one RIAF torus simulation around a Schwarzschild black
hole (:func:`pinneapple_physics.blackhole.runs.simulate_run`) and the time-window statistics the resolution study
uses (:func:`pinneapple_physics.blackhole.convergence.run_statistics`). The forecaster training and skill scripts of
the example need the Hub dataset and are not nodes here."""
from __future__ import annotations

import importlib.util
import time

import numpy as np

from ..registry import Param, Port, node
from ..runtime import artifact
from ..types import register_port_type

register_port_type("bh_run", "a black-hole accretion simulation run folder (frames, times, mdot, grid; "
                             "pinneapple_physics.blackhole.runs)")


def _backend(name: str) -> str:
    return "numba" if name == "auto" and importlib.util.find_spec("numba") else ("torch" if name == "auto" else name)


@node("blackhole.simulate", "Black-hole accretion flow (RIAF torus)", category="Astrophysics",
      outputs=[Port("run", "bh_run"), Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("nr", "int", 128, minimum=16, maximum=1024), Param("ntheta", "int", 64, minimum=8, maximum=512),
              Param("alpha", "float", 0.1, minimum=0.0, maximum=1.0, help="viscosity parameter"),
              Param("viscosity", "enum", "SS", choices=["SS", "ST", "none"]),
              Param("torus_a", "float", 0.0, help="angular-momentum power law of the torus"),
              Param("t_end", "float", 10000.0, minimum=10.0, maximum=1e6, help="end time [M]"),
              Param("every", "float", 10.0, minimum=0.1, help="frame interval [M]"),
              Param("seed", "int", 0, minimum=0),
              Param("perturbation_kind", "enum", "modes", choices=["modes", "cell"],
                    help="modes: the same seeded perturbation on every grid (resolution study)"),
              Param("backend", "enum", "auto", choices=["auto", "numba", "torch"])],
      calls="pinneapple_physics.blackhole.RIAFConfig + runs.simulate_run / load_run")
def simulate(ctx, inputs, params):
    """Evolve the viscous, radiatively inefficient flow that forms from an equilibrium torus (Paczynski-Wiita
    potential), saving the primitives every ``every`` M (examples/black_hole_weather/simulate.py)."""
    import torch

    from pinneapple_physics.blackhole import RIAFConfig
    from pinneapple_physics.blackhole.runs import load_run, simulate_run
    torch.set_num_threads(1)
    cfg = RIAFConfig(nr=int(params["nr"]), ntheta=int(params["ntheta"]), alpha=float(params["alpha"]),
                     viscosity=params["viscosity"], torus_a=float(params["torus_a"]), seed=int(params["seed"]),
                     perturbation_kind=params["perturbation_kind"], backend=_backend(params["backend"]))
    folder = ctx.path("run")
    t0 = time.perf_counter()
    simulate_run(cfg, str(folder), t_end=float(params["t_end"]), every=float(params["every"]), log=ctx.log)
    ctx.metric("seconds", round(time.perf_counter() - t0, 1), f"{cfg.backend} backend, one core")
    run = load_run(str(folder))
    t, md = np.asarray(run["t"]), np.asarray(run["mdot"])
    ctx.metric("frames", int(len(t)))
    ctx.metric("t_end_M", float(t[-1]))
    half = t >= t[-1] / 2
    ctx.metric("mdot_mean_second_half", float(md[half].mean()), "mass accretion rate through r_in, mean over the "
                                                                 "second half of the run")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = run["grid"]
    rho = np.asarray(run["frames"][-1, 0], float)
    r, th = np.asarray(g["r"]), np.asarray(g["theta"])
    R, TH = np.meshgrid(r, th, indexing="ij")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    im = axs[0].pcolormesh(R * np.sin(TH), R * np.cos(TH), np.log10(np.clip(rho, 1e-12, None)), shading="gouraud",
                           cmap="inferno")
    axs[0].set_aspect("equal")
    axs[0].set_xlim(0, min(100, r[-1]))
    axs[0].set_ylim(-50, 50)
    axs[0].set_title(f"log10 density at t = {t[-1]:.0f} M")
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(t, md)
    axs[1].set(xlabel="t [M]", ylabel="mdot", title="accretion rate")
    f = ctx.figure(fig, "bh_density", "density and accretion rate")
    ctx.summary(f"{len(t)} frames to t = {t[-1]:.0f} M on {cfg.nr} x {cfg.ntheta}")
    a = artifact(folder, "dir")
    return {"run": a, "metrics": ctx.report["metrics"], "figure": f}


@node("blackhole.statistics", "Accretion-run statistics (time window)", category="Astrophysics",
      inputs=[Port("run", "bh_run")], outputs=[Port("metrics", "metrics")],
      params=[Param("t0", "float", 2000.0, minimum=0.0), Param("t1", "float", 10000.0, minimum=1.0)],
      calls="pinneapple_physics.blackhole.convergence.run_statistics")
def statistics(ctx, inputs, params):
    """Mean accretion rate, density and specific angular momentum profiles and the persistence error at leads of
    20, 100 and 1000 M over the window (the quantities of the resolution study, results/convergence.json)."""
    from pinneapple_physics.blackhole.convergence import run_statistics
    st = run_statistics(inputs["run"]["$artifact"], float(params["t0"]), float(params["t1"]))
    ctx.metric("mdot_mean", float(np.mean(st["mdot"])), f"t in [{params['t0']:g}, {params['t1']:g}] M")
    ctx.metric("mass_window_mean", float(np.mean(st["mass_window"])))
    for lead in (20, 100, 1000):
        v = st.get(f"persistence_{lead}")
        if v is not None and len(v):
            ctx.metric(f"persistence_mae_{lead}M", float(np.mean(v)))
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="blackhole.accretion", title="Black-hole accretion flow (RIAF torus) and its statistics",
        domain="astrophysics", example="black_hole_weather",
        keywords={"black hole": 3, "black-hole": 3, "buraco negro": 3, "accretion": 2.5, "riaf": 3, "torus": 1.5,
                  "schwarzschild": 2, "event horizon": 1.5, "black hole weather": 2},
        min_score=3, nodes=("blackhole.simulate", "blackhole.statistics"),
        inputs=("the viscosity (alpha, SS / ST), the grid and the end time",),
        uncited="Duarte, Nemmen & Navarro (2022, MNRAS 512, 5848) is the study the library reproduces; not in the "
                "flow reference registry",
        template="black_hole_weather/accretion_run", quick=True,
        description="a RIAF torus simulation and the window statistics of the resolution study "
                    "(examples/black_hole_weather/simulate.py, convergence.py)"))


_register_capabilities()
