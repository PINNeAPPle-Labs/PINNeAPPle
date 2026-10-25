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
    skipped_heavy,
    summarize_array,
)


def infer_deepxde(ds: dict) -> ModelResult:
    meta = dict(name="DeepXDE", slug="deepxde", category="pinn", domains=["general"],
                repo="https://github.com/lululxvi/deepxde")
    if not is_installed("deepxde"):
        return not_installed(**meta, pip_hint="pip install deepxde", dataset=ds)
    with Timer() as t:
        try:
            import deepxde as dde
            geom = dde.geometry.Interval(-1, 1)

            def pde(x, y):
                dy_xx = dde.grad.hessian(y, x)
                return dy_xx - 2.0

            bc = dde.icbc.DirichletBC(geom, lambda x: x[:, 0] ** 2, lambda x, on: on)
            data = dde.data.PDE(geom, pde, bc, num_domain=ds["n"], num_boundary=2)
            net = dde.nn.FNN([1, 8, 8, 1], "tanh", "Glorot uniform")
            model = dde.Model(data, net)
            model.compile("adam", lr=1e-3)
            y_pred = model.predict(ds["x"])
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["x"].shape),
              output_shape=str(np.asarray(y_pred).shape), output_summary=summarize_array(y_pred))

def infer_pina(ds: dict) -> ModelResult:
    meta = dict(name="PINA", slug="pina", category="pinn", domains=["general"],
                repo="https://github.com/mathLab/PINA")
    if not is_installed("pina"):
        return not_installed(**meta, pip_hint="pip install pina-mathlab", dataset=ds)
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from pina.model import FeedForward
            net = FeedForward(input_dimensions=1, output_dimensions=1, layers=[8, 8])
            x = torch.tensor(ds["x"], dtype=torch.float32)
            y_pred = net(x)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(tuple(x.shape)),
              output_shape=str(tuple(y_pred.shape)), output_summary=summarize_array(y_pred.detach().numpy()))

def infer_physicsnemo(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="NVIDIA PhysicsNeMo", slug="physicsnemo", category="pinn",
                domains=["general", "industrial"], repo="https://github.com/NVIDIA/physicsnemo")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install nvidia-physicsnemo", dataset=ds)
    if not is_installed("physicsnemo"):
        return not_installed(**meta, pip_hint="pip install nvidia-physicsnemo", dataset=ds)
    with Timer() as t:
        try:
            import physicsnemo
            version = getattr(physicsnemo, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"physicsnemo {version} imported; full model construction needs a specific "
                   "architecture module (e.g. physicsnemo.models.fno).")

def infer_sciann(ds: dict) -> ModelResult:
    meta = dict(name="SciANN", slug="sciann", category="pinn", domains=["general"],
                repo="https://github.com/sciann/sciann")
    if not is_installed("sciann"):
        return not_installed(**meta, pip_hint="pip install sciann", dataset=ds)
    with Timer() as t:
        try:
            import sciann as sn
            x = sn.Variable("x")
            y = sn.Functional("y", x, 2 * [8], "tanh")
            model = sn.SciModel(x, y)
            y_pred = model.predict([ds["x"].flatten()])
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    out = np.asarray(y_pred[0] if isinstance(y_pred, list) else y_pred)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["x"].shape),
              output_shape=str(out.shape), output_summary=summarize_array(out))

def infer_neuralpde_jl(ds: dict) -> ModelResult:
    meta = dict(name="NeuralPDE.jl", slug="neuralpde-jl", category="pinn", domains=["general"],
                repo="https://github.com/SciML/NeuralPDE.jl")
    return manual_install(**meta, instructions="Julia package — `julia -e ']add NeuralPDE'`. "
                                                "Not invokable from this Python harness.", dataset=ds)

def infer_pinns_torch(ds: dict) -> ModelResult:
    meta = dict(name="PINNs-Torch", slug="pinns-torch", category="pinn", domains=["general"],
                repo="https://github.com/rtqichen/PINNs-Torch")
    return manual_install(**meta, instructions="Git-only research code — `git clone` and install "
                                                "requirements.txt manually, no pip package.", dataset=ds)

def infer_pinns_original(ds: dict) -> ModelResult:
    meta = dict(name="PINNs (original reference code)", slug="pinns-original", category="pinn",
                domains=["general"], repo="https://github.com/maziarraissi/PINNs")
    return manual_install(**meta, instructions="Archived TensorFlow 1.x code, not pip-packaged — "
                                                "`git clone` only; repo itself now points to DeepXDE/PINA.",
                           dataset=ds)

def infer_pinn_tfi_hss(ds: dict) -> ModelResult:
    meta = dict(name="PINN_TFI-HSS", slug="pinn-tfi-hss", category="pinn", domains=["general"],
                repo="https://github.com/liuxu97531/PINN_TFI-HSS")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.",
                           dataset=ds)

def infer_neuraloperator(ds: dict) -> ModelResult:
    meta = dict(name="NeuralOperator (FNO)", slug="neuraloperator", category="operator",
                domains=["operator"], repo="https://github.com/neuraloperator/neuraloperator")
    if not is_installed("neuralop"):
        return not_installed(**meta, pip_hint="pip install neuraloperator", dataset=ds)
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from neuralop.models import FNO
            model = FNO(n_modes=(8,), hidden_channels=8, in_channels=1, out_channels=1, n_layers=1)
            x = torch.tensor(ds["u0"], dtype=torch.float32).reshape(1, 1, -1)
            y_pred = model(x)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(tuple(x.shape)),
              output_shape=str(tuple(y_pred.shape)), output_summary=summarize_array(y_pred.detach().numpy()))

def infer_deeponet_original(ds: dict) -> ModelResult:
    meta = dict(name="DeepONet (original)", slug="deeponet-original", category="operator",
                domains=["operator"], repo="https://github.com/lululxvi/deeponet")
    return manual_install(**meta, instructions="Research code built on DeepXDE — `git clone` the repo; "
                                                "no standalone pip package.", dataset=ds)

def infer_deeponet_fno_benchmark(ds: dict) -> ModelResult:
    meta = dict(name="DeepONet-FNO benchmark", slug="deeponet-fno-benchmark", category="operator",
                domains=["operator"], repo="https://github.com/lu-group/deeponet-fno")
    return manual_install(**meta, instructions="Benchmark/comparison scripts — `git clone` the repo; "
                                                "no standalone pip package.", dataset=ds)

def infer_pino(ds: dict) -> ModelResult:
    meta = dict(name="PINO (Physics-Informed Neural Operator)", slug="pino", category="operator",
                domains=["operator"], repo="https://github.com/neuraloperator/physics_informed")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.",
                           dataset=ds)
