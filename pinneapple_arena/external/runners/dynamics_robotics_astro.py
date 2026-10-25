"""
Nonlinear dynamics / system identification, robotics-adjacent differentiable
engines, and astro (astrophysical fluids / orbital mechanics / ocean) models
added from the second catalog-expansion research pass.

Same contract as every other runner module: try the real import, build the
smallest real instance of the thing, run one real call, return via
ok()/failed()/not_installed()/manual_install()/skipped_heavy().
"""

from __future__ import annotations

import numpy as np

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.result import (
    ModelResult,
    Timer,
    failed,
    manual_install,
    not_installed,
    ok,
    skipped_heavy,
    summarize_array,
)


def infer_pysindy(ds: dict) -> ModelResult:
    meta = dict(name="PySINDy", slug="pysindy", category="domain", domains=["dynamics"],
                repo="https://github.com/dynamicslab/pysindy")
    if not is_installed("pysindy"):
        return not_installed(**meta, pip_hint="pip install pysindy", dataset=ds)
    with Timer() as t:
        try:
            import pysindy as ps
            model = ps.SINDy()
            model.fit(ds["xyz"], t=ds["dt"])
            score = model.score(ds["xyz"], t=ds["dt"])
            equations = model.equations(precision=3)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["xyz"].shape),
              output_shape=f"{len(equations)} discovered ODEs", output_summary={"r2_score": float(score)},
              note="Fit a SINDy model to the Lorenz-63 trajectory and discovered: " + " | ".join(equations))


def infer_diffrax(ds: dict) -> ModelResult:
    meta = dict(name="Diffrax", slug="diffrax", category="diffsim", domains=["dynamics", "general"],
                repo="https://github.com/patrick-kidger/diffrax")
    if not is_installed("diffrax"):
        return not_installed(**meta, pip_hint="pip install diffrax", dataset=ds)
    with Timer() as t:
        try:
            import diffrax
            import jax.numpy as jnp

            sigma, rho, beta = ds["sigma"], ds["rho"], ds["beta"]

            def vector_field(t, y, args):
                x, y_, z = y
                return jnp.array([sigma * (y_ - x), x * (rho - z) - y_, x * y_ - beta * z])

            term = diffrax.ODETerm(vector_field)
            solver = diffrax.Dopri5()
            save_times = jnp.linspace(0, 2.0, 200)  # fixed, evenly-spaced — avoids SaveAt(steps=True)'s
                                                      # inf-padded buffer for unused adaptive-step slots
            sol = diffrax.diffeqsolve(term, solver, t0=0, t1=2.0, dt0=0.01,
                                       y0=jnp.array([1.0, 1.0, 1.0]),
                                       saveat=diffrax.SaveAt(ts=save_times), max_steps=4096)
            out = np.asarray(sol.ys)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape=str(out.shape),
              output_summary=summarize_array(out),
              note="Integrated the Lorenz-63 ODE forward from t=0 to t=2 with Diffrax's Dopri5 solver.")


def infer_genesis(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="Genesis", slug="genesis", category="diffsim",
                domains=["rigidbody", "particulate", "fluids"],
                repo="https://github.com/Genesis-Embodied-AI/genesis-world")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install genesis-world", dataset=ds)
    if not is_installed("genesis"):
        return not_installed(**meta, pip_hint="pip install genesis-world", dataset=ds)
    with Timer() as t:
        try:
            import genesis as gs
            version = getattr(gs, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"genesis {version} imported; a real scene needs gs.init() + a Scene/entity build, which "
                   "requires a GPU-backed session — beyond this CPU demo harness.")


def infer_nimble_physics(ds: dict) -> ModelResult:
    meta = dict(name="Nimble Physics", slug="nimble-physics", category="rigidbody",
                domains=["rigidbody", "biomechanics"], repo="https://github.com/keenon/nimblephysics")
    if not is_installed("nimblephysics"):
        return not_installed(**meta, pip_hint="pip install nimblephysics", dataset=ds)
    with Timer() as t:
        try:
            import nimblephysics as nimble
            world = nimble.simulation.World()
            world.setGravity([0, -9.81, 0])
            world.setTimeStep(0.001)
            state = world.getState()
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape=str(np.asarray(state).shape),
              output_summary=summarize_array(np.asarray(state)),
              note="Built an empty Nimble World with gravity + a 1ms timestep and read its initial state vector.")


def infer_dojo_jl(ds: dict) -> ModelResult:
    meta = dict(name="Dojo.jl", slug="dojo-jl", category="rigidbody", domains=["rigidbody"],
                repo="https://github.com/dojo-sim/Dojo.jl")
    return manual_install(**meta, instructions="Julia package — `julia -e ']add Dojo'`. A Python interface "
                                                "exists via CondaPkg/PythonCall per the repo README, but the "
                                                "reference implementation is not a pip package. Archived — no "
                                                "longer actively developed.", dataset=ds)


def infer_astronomix(ds: dict) -> ModelResult:
    meta = dict(name="astronomix (jf1uids)", slug="astronomix", category="diffsim",
                domains=["astrofluids"], repo="https://github.com/leo1200/astronomix")
    if not is_installed("astronomix"):
        return not_installed(**meta, pip_hint="pip install astronomix", dataset=ds)
    with Timer() as t:
        try:
            import astronomix
            version = getattr(astronomix, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"astronomix {version} imported; a real MHD run needs a configured grid + initial "
                   "conditions per the repo's example notebooks.")


def infer_dsgp4(ds: dict) -> ModelResult:
    meta = dict(name="dSGP4", slug="dsgp4", category="diffsim", domains=["astrodynamics"],
                repo="https://github.com/esa/dSGP4")
    if not (is_installed("dsgp4") and is_installed("torch")):
        return not_installed(**meta, pip_hint="pip install dsgp4", dataset=ds)
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")
            import dsgp4
            tle = dsgp4.tle.TLE([ds["line1"], ds["line2"]])
            dsgp4.initialize_tle(tle)
            tsince = torch.tensor([0.0, 10.0, 60.0, 120.0])
            state = dsgp4.propagate(tle, tsince)
            out = np.asarray(state.detach() if hasattr(state, "detach") else state)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape=str(out.shape),
              output_summary=summarize_array(out),
              note="Propagated the ISS TLE forward to 0/10/60/120 minutes with dSGP4.")


def infer_veros(ds: dict) -> ModelResult:
    meta = dict(name="Veros", slug="veros", category="domain", domains=["ocean"],
                repo="https://github.com/team-ocean/veros")
    if not is_installed("veros"):
        return not_installed(**meta, pip_hint="pip install veros", dataset=ds)
    with Timer() as t:
        try:
            import veros
            version = getattr(veros, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"veros {version} imported; a real run needs a full VerosSetup subclass (grid, forcing, "
                   "initial state) per the repo's setup gallery.")


def infer_diffpd(ds: dict) -> ModelResult:
    meta = dict(name="DiffPD", slug="diffpd", category="diffsim", domains=["softbody"],
                repo="https://github.com/mit-gfx/diff_pd_public")
    return manual_install(**meta, instructions="Research code — `git clone` and build the C++ core + Python "
                                                "bindings per the repo README, no pip package.", dataset=ds)


def infer_difftaichi(ds: dict) -> ModelResult:
    meta = dict(name="DiffTaichi", slug="difftaichi", category="diffsim",
                domains=["general", "structural", "fluids"],
                repo="https://github.com/taichi-dev/difftaichi")
    return manual_install(**meta, instructions="Examples-only repo, not pip-packaged — `git clone` for the "
                                                "10 reference simulators. Its differentiable-programming "
                                                "functionality now ships inside the `taichi` package (see "
                                                "the 'Taichi' entry in this catalog).", dataset=ds)


def infer_diffsph(ds: dict) -> ModelResult:
    meta = dict(name="diffSPH", slug="diffsph", category="diffsim", domains=["fluids", "particulate"],
                repo="https://github.com/tum-pbs/diffSPH")
    if not (is_installed("diffSPH") and is_installed("torch")):
        return not_installed(**meta, pip_hint="pip install diffSPH", dataset=ds)
    with Timer() as t:
        try:
            import diffSPH
            version = getattr(diffSPH, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"diffSPH {version} imported; a real SPH step needs a configured particle set via its "
                   "companion torchCompactRadius neighbor-search package.")
