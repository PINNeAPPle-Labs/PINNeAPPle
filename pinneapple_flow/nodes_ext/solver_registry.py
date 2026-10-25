"""Nodes over the solver registry (``examples/solvers``): every solver built by name through
:class:`pinneapple_simulation.numerical_solvers.SolverRegistry` after
:func:`~pinneapple_simulation.numerical_solvers.register_all` -- the time-series decompositions (EEMD, VMD, wavelet,
STL, SSA, synchrosqueezing), the weakly compressible SPH box with particles from
:func:`pinneapple_design.geometry.sample.sample_box_particles`, and the 2-D Poisson equation by SOR finite differences
and by the periodic FFT solver."""
from __future__ import annotations

import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node

_DECOMP = {"eemd": {"n_ensembles": 30, "noise_std": 0.15, "max_imfs": 6}, "vmd": {"K": 4},
           "wavelet": {"wavelet": "db4", "level": 4}, "stl": {"period": 200}, "ssa": {"L": 200, "r": 5},
           "sst": {"fs": 1.0, "nperseg": 128}}
_ADDITIVE = ("eemd", "ceemdan", "vmd", "ssa", "stl")      # components that add up to (an approximation of) the signal


def _registry():
    from pinneapple_simulation.numerical_solvers import SolverRegistry, register_all
    return SolverRegistry, register_all()


@node("solvers.decompose", "Time-series decompositions from the solver registry", category="Time series",
      outputs=[Port("metrics", "metrics")],
      params=[Param("t_end", "float", 10.0, minimum=1e-6), Param("n", "int", 2000, minimum=16, maximum=1_000_000),
              Param("tones", "list", [[1.2, 1.0], [6.0, 0.5]], help="[[frequency, amplitude], ...] of the sines"),
              Param("noise_std", "float", 0.1, minimum=0.0), Param("seed", "int", 0),
              Param("solvers", "dict", _DECOMP, help="{registry name: constructor arguments}")],
      calls="pinneapple_simulation.numerical_solvers.SolverRegistry.build (eemd, vmd, wavelet, stl, ssa, sst, ...) "
            "+ register_all")
def decompose(ctx, inputs, params):
    """A sum of sines plus Gaussian noise decomposed by each registered solver (examples/solvers/
    01_time_series_decompositions.py, with a seeded noise draw). Per solver: the output shape and, for the additive
    decompositions (EEMD, CEEMDAN, VMD, SSA, STL), how far the sum of the components is from the signal. A solver
    whose optional dependency is missing is skipped with a warning."""
    import torch
    reg, failed = _registry()
    t = np.linspace(0.0, float(params["t_end"]), int(params["n"]))
    x = sum(float(a) * np.sin(2 * np.pi * float(f) * t) for f, a in params["tones"])
    x = x + float(params["noise_std"]) * np.random.default_rng(int(params["seed"])).standard_normal(len(t))
    xt = torch.tensor(x, dtype=torch.float32)
    rows, ran = [], 0
    for name, kw in params["solvers"].items():
        ctx.check_stop()
        if name not in reg.list():
            why = failed.get(name, "not registered")
            ctx.warn(f"{name}: skipped ({why})")
            ctx.metric(f"{name}.available", 0)
            rows.append([name, "-", "-", "-", f"skipped: {why}"])
            continue
        t0 = time.perf_counter()
        out = reg.build(name, **(kw or {}))(xt)
        dt = time.perf_counter() - t0
        shape = list(out.result.shape)
        ctx.metric(f"{name}.available", 1)
        ctx.metric(f"{name}.components", int(shape[-2]) if len(shape) >= 2 else 1, f"output shape {shape}")
        ctx.metric(f"{name}.seconds", round(dt, 3))
        rec = "-"
        if name in _ADDITIVE:
            parts = out.result.detach().cpu().numpy().reshape(-1, shape[-1])
            rec = float(np.linalg.norm(parts.sum(0) - x) / np.linalg.norm(x))
            ctx.metric(f"{name}.reconstruction_rel_error", rec, "||sum of components - signal|| / ||signal||")
        rows.append([name, str(shape), round(dt, 2), rec if rec == "-" else f"{rec:.2e}", "ok"])
        ran += 1
    ctx.metric("solvers_run", ran)
    ctx.table("decompositions", ["solver", "output shape", "seconds", "reconstruction error", "status"], rows)
    ctx.summary(f"{ran} of {len(params['solvers'])} decompositions ran")
    return {"metrics": ctx.report["metrics"]}


@node("solvers.sph_box", "Weakly compressible SPH: a fluid block in a box", category="Particles",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("block_lo", "list", [0.1, 0.1]), Param("block_hi", "list", [0.4, 0.6]),
              Param("spacing", "float", 0.03, minimum=1e-4), Param("jitter", "float", 0.2, minimum=0.0),
              Param("seed", "int", 0), Param("h", "float", 0.04, minimum=1e-5), Param("rho0", "float", 1000.0),
              Param("c0", "float", 25.0), Param("mu", "float", 0.02), Param("gravity", "list", [0.0, -9.81]),
              Param("box_lo", "list", [0.0, 0.0]), Param("box_hi", "list", [1.0, 1.0]),
              Param("dt", "float", 2e-3, minimum=1e-8), Param("steps", "int", 50, minimum=1, maximum=100_000)],
      calls="pinneapple_simulation.numerical_solvers.SolverRegistry.build('sph') + "
            "pinneapple_design.geometry.sample.sample_box_particles")
def sph_box(ctx, inputs, params):
    """examples/solvers/02_sph_box.py: the trajectory, the final bounding box and the fraction of particles still
    inside the box (a warning when particles leave it: the run is then not a valid fluid simulation)."""
    import torch

    from pinneapple_design.geometry.sample import sample_box_particles
    reg, _ = _registry()
    torch.manual_seed(int(params["seed"]))
    pos = sample_box_particles(lo=tuple(map(float, params["block_lo"])), hi=tuple(map(float, params["block_hi"])),
                               spacing=float(params["spacing"]), jitter=float(params["jitter"]))
    sph = reg.build("sph", h=float(params["h"]), rho0=float(params["rho0"]), c0=float(params["c0"]),
                    mu=float(params["mu"]), gravity=[float(g) for g in params["gravity"]],
                    boundary_lo=[float(v) for v in params["box_lo"]], boundary_hi=[float(v) for v in params["box_hi"]])
    t0 = time.perf_counter()
    traj = sph(pos, torch.zeros_like(pos), dt=float(params["dt"]), steps=int(params["steps"])).result
    traj = traj.detach().cpu().numpy()
    lo, hi = np.asarray(params["box_lo"], float), np.asarray(params["box_hi"], float)
    final = traj[-1]
    inside = np.all((final >= lo) & (final <= hi), axis=1)
    ctx.metric("particles", int(final.shape[0]))
    ctx.metric("frames", int(traj.shape[0]))
    for k, ax in enumerate("xy"[:final.shape[1]]):
        ctx.metric(f"final_min_{ax}", float(final[:, k].min()))
        ctx.metric(f"final_max_{ax}", float(final[:, k].max()))
    ctx.metric("fraction_inside_box", float(inside.mean()), "particles inside the box at the last step")
    ctx.metric("finite", int(np.isfinite(traj).all()))
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    if not inside.all():
        ctx.warn(f"{int((~inside).sum())} of {len(inside)} particles left the box: the run blew up "
                 f"(final extent {final.min(0).round(2).tolist()} to {final.max(0).round(2).tolist()})")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, k, title in ((ax[0], 0, "initial"), (ax[1], -1, f"step {traj.shape[0] - 1}")):
        a.plot(traj[k][:, 0], traj[k][:, 1], ".", ms=3)
        a.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, color="k"))
        a.set_title(title, fontsize=9)
        a.set_aspect("equal", adjustable="datalim")
    f = ctx.figure(fig, "sph_box", "particle positions")
    ctx.summary(f"{final.shape[0]} particles, {100 * inside.mean():.0f} % inside the box after "
                f"{traj.shape[0] - 1} steps")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("solvers.poisson_fdm_spectral", "2-D Poisson: SOR finite differences against the periodic FFT solver",
      category="Numerical solvers", outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("n", "int", 64, minimum=8, maximum=2048, help="grid points per side on [0, 1]"),
              Param("k", "int", 2, minimum=1, maximum=64, help="source f = sin(k pi x) sin(k pi y)"),
              Param("iters", "int", 2000, minimum=1), Param("omega", "float", 0.9, minimum=0.01, maximum=1.99)],
      calls="pinneapple_simulation.numerical_solvers.SolverRegistry.build('fdm' (SOR), 'spectral' (FFT))")
def poisson_fdm_spectral(ctx, inputs, params):
    """examples/solvers/03_poisson_solvers.py: the source f with u = 0 on the boundary (SOR) and with periodic
    boundaries (FFT), their mean squared difference as the script prints it, and each against the exact solution of
    its own equation: the registry's FDM solver solves lap(u) = -f (u = f / (2 (k pi)^2)), the spectral one
    lap(u) = f (u = -f / (2 (k pi)^2))."""
    import torch
    reg, _ = _registry()
    n, k = int(params["n"]), int(params["k"])
    xs = torch.linspace(0, 1, n)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    f = torch.sin(k * torch.pi * X) * torch.sin(k * torch.pi * Y)
    t0 = time.perf_counter()
    u_fdm = reg.build("fdm", iters=int(params["iters"]), omega=float(params["omega"]))(
        f, torch.zeros_like(f), dx=1 / (n - 1), dy=1 / (n - 1)).result
    t_fdm = time.perf_counter() - t0
    u_spec = reg.build("spectral")(f).result
    exact = f / (2 * (k * np.pi) ** 2)              # solves lap(u) = -f; lap(u) = f has -exact
    ctx.metric("mse_fdm_vs_spectral", float(torch.mean((u_fdm - u_spec) ** 2)), "the script's number")
    ctx.metric("mse_fdm_vs_minus_spectral", float(torch.mean((u_fdm + u_spec) ** 2)),
               "the FDM solver solves lap(u) = -f, the spectral one lap(u) = f")
    for name, u, sgn in (("fdm", u_fdm, 1.0), ("spectral", u_spec, -1.0)):
        ctx.metric(f"{name}.rel_l2_vs_exact", float(torch.linalg.norm(u - sgn * exact) / torch.linalg.norm(exact)),
                   "against u = -+ f / (2 (k pi)^2) in the solver's own sign convention")
    ctx.warn("SolverRegistry 'fdm' solves lap(u) = -f and 'spectral' solves lap(u) = f: their difference "
             "(mse_fdm_vs_spectral, the number the script prints) is mostly the sign")
    ctx.metric("fdm.seconds", round(t_fdm, 2))
    if not (torch.isfinite(u_fdm).all() and torch.isfinite(u_spec).all()):
        raise StepFailed("a solver returned non-finite values")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, u, title in zip(ax, (u_fdm, u_spec, u_fdm - u_spec), ("FDM (SOR, Dirichlet)", "spectral (FFT, periodic)",
                                                                  "difference"), strict=True):
        im = a.imshow(u.numpy(), origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r")
        a.set_title(title, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig_a = ctx.figure(fig, "poisson", "u by finite differences and by FFT")
    m = ctx.report["metrics"]
    ctx.summary(f"relative L2 error against the exact solution: FDM {m['fdm.rel_l2_vs_exact']['value']:.2e}, "
                f"spectral {m['spectral.rel_l2_vs_exact']['value']:.2e}")
    return {"metrics": m, "figure": fig_a}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="solvers.decompose", title="Time-series decompositions (EEMD, VMD, SSA, STL, wavelets)",
        domain="time series", example="solvers",
        keywords={"eemd": 3, "emd": 2, "vmd": 3, "variational mode decomposition": 3, "empirical mode": 3,
                  "ssa": 2, "singular spectrum": 3, "stl": 2, "synchrosqueez": 3, "decomposition": 1.5,
                  "imf": 2},
        min_score=3, nodes=("solvers.decompose",),
        uncited="the registry's own decompositions; no decomposition reference in references.yaml",
        template="solvers/01_time_series_decompositions",
        description="examples/solvers/01_time_series_decompositions.py"))
    register_capability(Capability(
        id="solvers.sph_box", title="SPH fluid block in a box (solver registry)", domain="particles",
        example="solvers",
        keywords={"sph": 3, "smoothed particle": 3, "particle hydrodynamics": 3, "wcsph": 3},
        min_score=3, nodes=("solvers.sph_box",),
        uncited="no SPH reference in references.yaml", template="solvers/02_sph_box",
        description="examples/solvers/02_sph_box.py"))
    register_capability(Capability(
        id="solvers.poisson_fdm_spectral", title="2-D Poisson by finite differences and by FFT",
        domain="numerical solvers", example="solvers",
        keywords={"poisson": 2, "sor": 2, "successive over-relaxation": 3, "fft": 1.5, "spectral": 1.5,
                  "finite difference": 1},
        requires_all=(("poisson",),), min_score=3, nodes=("solvers.poisson_fdm_spectral",),
        references=("leveque2007fd", "trefethen2000spectral"), template="solvers/03_poisson_solvers",
        description="examples/solvers/03_poisson_solvers.py"))


_register_capabilities()
