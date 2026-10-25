"""Lattice Boltzmann nodes (``examples/lbm/01_sphere_wind_tunnel.py``): drag on a sphere in the 3-D wind tunnel of
:class:`pinneapple_simulation.numerical_solvers.lbm_tunnel.WindTunnelLBM`, against Stokes' law
(:func:`~pinneapple_simulation.numerical_solvers.lbm_tunnel.stokes_sphere_field` far field on the open sides) and
against the Schiller-Naumann correlation
(:func:`~pinneapple_simulation.numerical_solvers.lbm_tunnel.drag_sphere_schiller_naumann`), with FP32 and FP16
storage of the shifted distributions."""
from __future__ import annotations

import math
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node


@node("lbm.sphere_stokes", "Stokes drag on a sphere (lattice Boltzmann)", category="Lattice Boltzmann",
      outputs=[Port("metrics", "metrics")],
      params=[Param("radius", "float", 5.0, minimum=1.0, help="sphere radius [cells]"),
              Param("nu", "float", 1 / 6, minimum=1e-6, help="viscosity [lattice units]"),
              Param("u", "float", 0.002, minimum=1e-6, help="free-stream velocity [lattice units]"),
              Param("box_radii", "float", 8.0, minimum=2.0, help="cube side in sphere radii"),
              Param("steps", "int", 1200, minimum=1), Param("threads", "int", 4, minimum=1, maximum=256)],
      calls="pinneapple_simulation.numerical_solvers.lbm_tunnel.WindTunnelLBM (open faces, add_solid, initialize, "
            "run, force_on_solid) + sphere + stokes_sphere_field")
def sphere_stokes(ctx, inputs, params):
    """A sphere in a cube whose six sides hold the analytic Stokes far field, started from that field; the
    momentum-exchange drag against F = 6 pi rho nu R U."""
    import torch

    from pinneapple_simulation.numerical_solvers.lbm_tunnel import (
        WindTunnelLBM,
        sphere,
        stokes_sphere_field,
    )
    torch.set_num_threads(int(params["threads"]))
    R, nu, u = float(params["radius"]), float(params["nu"]), float(params["u"])
    L = int(params["box_radii"] * R)
    c = ((L - 1) / 2,) * 3
    t0 = time.perf_counter()
    lbm = WindTunnelLBM((L, L, L), nu=nu, u_in=(u, 0, 0), faces={k: "open" for k in WindTunnelLBM.FACES})
    lbm.add_solid(sphere(lbm.shape, c, R))
    lbm.initialize(stokes_sphere_field(lbm.shape, c, R, (u, 0, 0)))
    lbm.run(int(params["steps"]))
    F = float(lbm.force_on_solid()[0])
    Fs = 6 * np.pi * nu * R * u
    ctx.metric("Re", 2 * R * u / nu)
    ctx.metric("F_lbm", F, "momentum exchange on the sphere, x component [lattice units]")
    ctx.metric("F_stokes", Fs, "6 pi rho nu R U")
    ctx.metric("error", F / Fs - 1, "F_lbm / F_stokes - 1")
    ctx.metric("wall_s", round(time.perf_counter() - t0, 1))
    ctx.summary(f"Stokes drag {100 * (F / Fs - 1):+.1f} % from 6 pi rho nu R U (R = {R:g} cells, box {L}^3)")
    return {"metrics": ctx.report["metrics"]}


@node("lbm.sphere_drag", "Sphere drag coefficient in a wind tunnel, FP32 and FP16 (lattice Boltzmann)",
      category="Lattice Boltzmann", outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("D", "float", 8.0, minimum=2.0, help="sphere diameter [cells]"),
              Param("U", "float", 0.05, minimum=1e-4, help="inlet velocity [lattice units]"),
              Param("Re", "float", 20.0, minimum=0.01),
              Param("shape", "list", [96, 48, 48], help="tunnel cells (x, y, z)"),
              Param("center", "list", [24.0, 23.5, 23.5], help="sphere centre [cells]"),
              Param("storages", "list", ["fp32", "fp16s"], help="fp32 and / or fp16s (16-bit shifted DDFs)"),
              Param("steps", "int", 3600, minimum=1), Param("every", "int", 200, minimum=1,
                                                             help="force sampling interval [steps]"),
              Param("average_last", "int", 4, minimum=1, help="force samples averaged for Cd"),
              Param("threads", "int", 4, minimum=1, maximum=256)],
      calls="pinneapple_simulation.numerical_solvers.lbm_tunnel.WindTunnelLBM (storage, add_solid, run, "
            "force_on_solid, fields, bytes_per_cell) + sphere + drag_sphere_schiller_naumann")
def sphere_drag(ctx, inputs, params):
    """Inlet at x-, outlet at x+, open sides; Cd from the mean of the last force samples, against Schiller &
    Naumann (1933); with both storages, the relative Cd difference and the largest velocity difference."""
    import torch

    from pinneapple_simulation.numerical_solvers.lbm_tunnel import (
        WindTunnelLBM,
        drag_sphere_schiller_naumann,
        sphere,
    )
    torch.set_num_threads(int(params["threads"]))
    D, U, Re = float(params["D"]), float(params["U"]), float(params["Re"])
    shape = tuple(int(s) for s in params["shape"])
    center = tuple(float(c) for c in params["center"])
    storages = list(params["storages"])
    if not storages or any(s not in ("fp32", "fp16s") for s in storages):
        raise StepFailed("storages: a list of 'fp32' and / or 'fp16s'")
    nu = U * D / Re
    q = 0.5 * U ** 2 * math.pi * D ** 2 / 4
    cd_sn = drag_sphere_schiller_naumann(Re)
    ctx.metric("Cd_schiller_naumann", cd_sn, "24/Re (1 + 0.15 Re^0.687)")
    hist, fields = {}, {}
    for st in storages:
        ctx.check_stop()
        t0 = time.perf_counter()
        lbm = WindTunnelLBM(shape, nu=nu, u_in=(U, 0, 0), storage=st)
        lbm.add_solid(sphere(shape, center, D / 2))
        h = []
        lbm.run(int(params["steps"]), every=int(params["every"]),
                callback=lambda m, h=h: h.append((m.t, float(m.force_on_solid()[0]))))
        cd = float(np.mean([f for _, f in h[-int(params["average_last"]):]])) / q
        hist[st], fields[st] = np.array(h), lbm.fields()["u"]
        ctx.metric(f"{st}.Cd", cd, f"mean x force of the last {params['average_last']} samples / (rho U^2 A / 2)")
        ctx.metric(f"{st}.error", cd / cd_sn - 1, "Cd / Cd_schiller_naumann - 1")
        ctx.metric(f"{st}.bytes_per_cell", int(lbm.bytes_per_cell()))
        ctx.metric(f"{st}.wall_s", round(time.perf_counter() - t0, 1))
        ctx.log(f"{st}: Cd {cd:.4f} ({100 * (cd / cd_sn - 1):+.1f} %), {time.perf_counter() - t0:.0f} s")
    if "fp32" in fields and "fp16s" in fields:
        m = ctx.report["metrics"]
        ctx.metric("fp16_vs_fp32.Cd_rel_diff", m["fp16s.Cd"]["value"] / m["fp32.Cd"]["value"] - 1)
        ctx.metric("fp16_vs_fp32.u_max_abs_diff", float(np.abs(fields["fp16s"] - fields["fp32"]).max()),
                   "largest |u_fp16 - u_fp32| [lattice units]")
        ctx.metric("fp16_vs_fp32.u_max_abs_diff_over_U",
                   float(np.abs(fields["fp16s"] - fields["fp32"]).max()) / U)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(storages) + 1, 1, figsize=(8, 3.2 * (len(storages) + 1)))
    jy = int(round(center[1]))
    for ax, st in zip(axs, storages, strict=False):
        im = ax.imshow(fields[st][0, :, jy, :].T / U, origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=1.3)
        ax.set_title(f"u_x / U at mid-plane, {st.upper()}: Cd = {ctx.report['metrics'][f'{st}.Cd']['value']:.2f}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025)
    for st in storages:
        axs[-1].plot(hist[st][:, 0], hist[st][:, 1] / q, label=st.upper())
    axs[-1].axhline(cd_sn, color="k", ls="--", label="Schiller & Naumann (1933)")
    axs[-1].set(xlabel="time step", ylabel="Cd", ylim=(0, 6))
    axs[-1].legend(fontsize=8)
    f = ctx.figure(fig, "sphere_tunnel", f"sphere at Re = {Re:g}, D = {D:g} cells")
    ctx.summary(", ".join(f"{st} Cd {ctx.report['metrics'][f'{st}.Cd']['value']:.3f}" for st in storages)
                + f" (Schiller-Naumann {cd_sn:.3f})")
    return {"metrics": ctx.report["metrics"], "figure": f}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="lbm.sphere_drag", title="Lattice Boltzmann wind tunnel: drag on a sphere", domain="fluid dynamics",
        example="lbm",
        keywords={"lattice boltzmann": 3, "lbm": 3, "d3q19": 3, "fluidx3d": 2.5, "wind tunnel": 1.5,
                  "sphere drag": 2.5, "drag on a sphere": 2.5, "stokes drag": 2, "schiller-naumann": 2.5,
                  "fp16": 1},
        min_score=3, nodes=("lbm.sphere_stokes", "lbm.sphere_drag", "examples.compare"),
        uncited="checked against Stokes' law and the Schiller & Naumann (1933) correlation, neither in "
                "references.yaml (docs/core_concepts/lbm_wind_tunnel.md)",
        template="lbm/01_sphere_wind_tunnel", quick=True,
        description="examples/lbm/01_sphere_wind_tunnel.py"))


_register_capabilities()
