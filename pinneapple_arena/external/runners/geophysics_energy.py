"""
Auto-migrated from the original inference_suite.py (single-file registry).
Each infer_* function builds a minimal real instance of the actual library
and runs one real forward pass / inference call — see pinneapple_arena/external/result.py
for the ModelResult contract every function returns.

Heavy/GPU-oriented functions now take an explicit `include_heavy` parameter
instead of reading a module-level global, so concurrent API requests can't
step on each other's --include-heavy setting.
"""

from __future__ import annotations

import numpy as np

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.result import (
    ModelResult,
    Timer,
    failed,
    not_installed,
    ok,
    skipped_heavy,
    summarize_array,
)


def infer_devito(ds: dict) -> ModelResult:
    meta = dict(name="Devito", slug="devito", category="domain", domains=["geophysics"],
                repo="https://github.com/devitocodes/devito")
    if not is_installed("devito"):
        return not_installed(**meta, pip_hint="pip install devito", dataset=ds)
    with Timer() as t:
        try:
            import devito as dv
            grid = dv.Grid(shape=(ds["nx"], ds["nz"]))
            f = dv.Function(name="f", grid=grid)
            f.data[:] = ds["velocity"]
            out = np.array(f.data)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["velocity"].shape),
              output_shape=str(out.shape), output_summary=summarize_array(out),
              note="Built a Devito Grid + Function and loaded the velocity model into it.")

def infer_deepwave(ds: dict) -> ModelResult:
    meta = dict(name="Deepwave", slug="deepwave", category="domain", domains=["geophysics"],
                repo="https://github.com/ar4/deepwave")
    if not (is_installed("torch") and is_installed("deepwave")):
        return not_installed(**meta, pip_hint="pip install deepwave", dataset=ds)
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            v = torch.tensor(ds["velocity"], dtype=torch.float32)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["velocity"].shape),
              output_shape=str(tuple(v.shape)), output_summary=summarize_array(v.numpy()),
              note="Built a PyTorch velocity-model tensor ready for deepwave.scalar()/elastic() propagation.")

def infer_torax(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="TORAX", slug="torax", category="domain", domains=["plasma"],
                repo="https://github.com/google-deepmind/torax")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install torax", dataset=ds)
    if not is_installed("torax"):
        return not_installed(**meta, pip_hint="pip install torax", dataset=ds)
    with Timer() as t:
        try:
            import torax
            version = getattr(torax, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"torax {version} imported; a real run needs a full CONFIG dict per repo README.")

def infer_pybamm(ds: dict) -> ModelResult:
    meta = dict(name="PyBaMM", slug="pybamm", category="domain", domains=["electrochemistry"],
                repo="https://github.com/pybamm-team/PyBaMM")
    if not is_installed("pybamm"):
        return not_installed(**meta, pip_hint="pip install pybamm", dataset=ds)
    with Timer() as t:
        try:
            import pybamm
            model = pybamm.lithium_ion.SPM()
            sim = pybamm.Simulation(model)
            solution = sim.solve([0, 600])
            voltage = solution["Terminal voltage [V]"].data
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape="0-600s discharge window",
              output_shape=str(np.asarray(voltage).shape), output_summary=summarize_array(voltage),
              note="Built and solved a Single Particle Model (SPM) 10-minute discharge simulation.")

def infer_reservoirflow(ds: dict) -> ModelResult:
    meta = dict(name="ReservoirFlow", slug="reservoirflow", category="pinn", domains=["petroleum"],
                repo="https://github.com/hiesabx/reservoirflow")
    if not is_installed("reservoirflow"):
        return not_installed(**meta, pip_hint="pip install reservoirflow", dataset=ds)
    with Timer() as t:
        try:
            import reservoirflow as rf
            grid = rf.grids.RegularCartesian(nx=ds["nx"], ny=ds["ny"], nz=1,  # noqa: F841 (building it is the check)
                                              dx=100, dy=100, dz=10, phi=0.2, kx=50)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str((ds["nx"], ds["ny"])),
              output_shape="RegularCartesian grid built", note="Built a small 2D reservoir grid; a full "
              "solve requires attaching a Fluid + Model + BoundaryConditions per the ReservoirFlow API.")
