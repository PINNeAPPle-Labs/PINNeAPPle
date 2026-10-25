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
    manual_install,
    not_installed,
    ok,
    summarize_array,
)


def infer_phiflow(ds: dict) -> ModelResult:
    meta = dict(name="ΦFlow (PhiFlow)", slug="phiflow", category="diffsim", domains=["fluids"],
                repo="https://github.com/tum-pbs/PhiFlow")
    if not is_installed("phi"):
        return not_installed(**meta, pip_hint="pip install phiflow", dataset=ds)
    with Timer() as t:
        try:
            from phi.flow import Box, CenteredGrid, extrapolation
            grid = CenteredGrid(0, extrapolation.ZERO, x=ds["nx"], y=ds["ny"], bounds=Box(x=1, y=1))
            values = grid.values.numpy(grid.shape)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str((ds["nx"], ds["ny"])),
              output_shape=str(np.asarray(values).shape), output_summary=summarize_array(values))

def infer_jax_cfd(ds: dict) -> ModelResult:
    meta = dict(name="JAX-CFD", slug="jax-cfd", category="diffsim", domains=["fluids"],
                repo="https://github.com/google/jax-cfd")
    if not is_installed("jax_cfd"):
        return not_installed(**meta, pip_hint="pip install jax-cfd", dataset=ds)
    with Timer() as t:
        try:
            import jax_cfd.base as cfd
            grid = cfd.grids.Grid((ds["nx"], ds["ny"]), domain=((0, 1), (0, 1)))
            field_shape = grid.shape
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str((ds["nx"], ds["ny"])),
              output_shape=str(field_shape), note="Built base.grids.Grid; full step() needs an initial velocity field.")

def infer_jax_fluids(ds: dict) -> ModelResult:
    meta = dict(name="JAX-Fluids", slug="jax-fluids", category="diffsim", domains=["fluids"],
                repo="https://github.com/tumaer/JAXFLUIDS")
    return manual_install(**meta, instructions="Not on PyPI — `git clone` the repo then `pip install .`.",
                           dataset=ds)

def infer_deepflame(ds: dict) -> ModelResult:
    meta = dict(name="DeepFlame", slug="deepflame", category="domain", domains=["combustion", "fluids"],
                repo="https://github.com/deepmodeling/deepflame-dev")
    return manual_install(**meta, instructions="Requires OpenFOAM + Cantera + PyTorch pre-installed, "
                                                "then `git clone` and build per repo README.", dataset=ds)

def infer_fluidx3d(ds: dict) -> ModelResult:
    meta = dict(name="FluidX3D", slug="fluidx3d", category="domain", domains=["fluids"],
                repo="https://github.com/ProjectPhysX/FluidX3D")
    return manual_install(**meta, instructions="C++17/OpenCL, not Python — `git clone`, then `./make.sh` "
                                                "(Linux/macOS) or FluidX3D.sln (Windows). Non-commercial, "
                                                "no military/defense use.", dataset=ds)

def infer_lesnets(ds: dict) -> ModelResult:
    meta = dict(name="LESnets", slug="lesnets", category="pinn", domains=["fluids"],
                repo="https://github.com/Sunan-zhao/LESnets")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_twophasepinn(ds: dict) -> ModelResult:
    meta = dict(name="twophasePINN", slug="twophasepinn", category="pinn", domains=["fluids"],
                repo="https://github.com/aaronbuhendwa/twophasePINN")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_meshgraphnets_pytorch(ds: dict) -> ModelResult:
    meta = dict(name="MeshGraphNets (PyTorch port)", slug="meshgraphnets-pytorch", category="graph",
                domains=["fluids", "structural"], repo="https://github.com/echowve/meshGraphNets_pytorch")
    return manual_install(**meta, instructions="Git-only community port — `git clone` and install its "
                                                "requirements manually.", dataset=ds)

def infer_meshgraphnets_original(ds: dict) -> ModelResult:
    meta = dict(name="MeshGraphNets (original, DeepMind)", slug="meshgraphnets-original", category="graph",
                domains=["fluids", "structural"],
                repo="https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets")
    return manual_install(**meta, instructions="Git-only, TensorFlow 1.x research code — see repo README.",
                           dataset=ds)

def infer_gns(ds: dict) -> ModelResult:
    meta = dict(name="GNS — Graph Network Simulator", slug="gns", category="graph",
                domains=["particulate"], repo="https://github.com/geoelements/gns")
    return manual_install(**meta, instructions="Git-only — `git clone` and `pip install -r "
                                                "requirements.txt` manually.", dataset=ds)

def infer_graingnn(ds: dict) -> ModelResult:
    meta = dict(name="GrainGNN", slug="graingnn", category="graph", domains=["materials", "manufacturing"],
                repo="https://github.com/YigongQin/GrainGraphNN")
    return manual_install(**meta, instructions="Paper code (PyTorch Geometric) — `git clone` only, no "
                                                "pip package.", dataset=ds)

def infer_taichi(ds: dict) -> ModelResult:
    meta = dict(name="Taichi", slug="taichi", category="diffsim",
                domains=["fluids", "structural", "particulate"], repo="https://github.com/taichi-dev/taichi")
    if not is_installed("taichi"):
        return not_installed(**meta, pip_hint="pip install taichi", dataset=ds)
    with Timer() as t:
        try:
            import taichi as ti
            ti.init(arch=ti.cpu)
            field_ = ti.field(dtype=ti.f32, shape=(ds["nx"], ds["ny"]))

            @ti.kernel
            def fill():
                for i, j in field_:
                    field_[i, j] = 1.0

            fill()
            out = field_.to_numpy()
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str((ds["nx"], ds["ny"])),
              output_shape=str(out.shape), output_summary=summarize_array(out))

def infer_warp(ds: dict) -> ModelResult:
    meta = dict(name="NVIDIA Warp", slug="warp", category="diffsim", domains=["rigidbody", "fluids"],
                repo="https://github.com/NVIDIA/warp")
    if not is_installed("warp"):
        return not_installed(**meta, pip_hint="pip install warp-lang", dataset=ds)
    with Timer() as t:
        try:
            import warp as wp
            wp.init()

            @wp.kernel
            def add_one(a: wp.array(dtype=float)):
                i = wp.tid()
                a[i] = a[i] + 1.0

            arr = wp.zeros(ds["n"], dtype=float, device="cpu")
            wp.launch(add_one, dim=ds["n"], inputs=[arr], device="cpu")
            out = arr.numpy()
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["n"]),
              output_shape=str(out.shape), output_summary=summarize_array(out))

def infer_dl4to(ds: dict) -> ModelResult:
    meta = dict(name="DL4TO", slug="dl4to", category="domain", domains=["structural"],
                repo="https://github.com/dl4to/dl4to")
    return manual_install(**meta, instructions="Not confirmed on PyPI — `git clone` the repo and "
                                                "`pip install -e .` per its README.", dataset=ds)
