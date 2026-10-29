"""One-step rollout functions of linear reduced-order models fitted on snapshot trajectories: ``step(u) -> u_next``
in the full space, the shape ``pp.validate(step, {"rollout": ...})`` audits (horizon, drift, spectrum, invariants).

``pod_dmd_stepper``: POD basis + dynamic mode decomposition of the POD coefficients (cylinder wake of
``examples/validation/05_pod_dmd_cylinder_wake.py``). ``pod_opinf_stepper``: POD basis cut at an energy fraction +
linear operator inference on the coefficients (``examples/validation/06_opinf_weather_advection.py``)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .dmd import DynamicModeDecomposition
from .opinf import OperatorInference
from .pod import POD

__all__ = ["pod_dmd_stepper", "pod_opinf_stepper"]


def pod_dmd_stepper(snapshots: np.ndarray, r: int = 8) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """POD (rank ``r``, centred) of ``snapshots`` ``(n_t, *shape)`` and exact DMD of its coefficients. Returns
    ``(step, max_abs_eig)``: ``step(u)`` advances one snapshot interval in the full space; ``max_abs_eig`` is the
    largest DMD eigenvalue modulus (above 1: the ROM amplifies)."""
    X = torch.tensor(snapshots.reshape(len(snapshots), -1), dtype=torch.float64)
    pod = POD(r=r, center=True).fit(X)
    dmd = DynamicModeDecomposition(r=r, center=False).fit(pod.encode(X)[None])
    shape = snapshots.shape[1:]

    def step(u):
        a = pod.encode(torch.tensor(u.reshape(1, -1), dtype=torch.float64))
        return pod.decode(dmd.rollout(a, 1)[:, -1]).numpy().reshape(shape)
    return step, float(np.abs(np.asarray(dmd.eig()[0])).max())


def pod_opinf_stepper(train: np.ndarray, energy: float) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    """POD of ``train`` ``(n_traj, n_t, n_x)`` keeping ``energy`` of the snapshot energy, and linear operator
    inference (no quadratic term, no bias) on the coefficient trajectories. Returns ``(step, rank)``."""
    X = torch.tensor(train, dtype=torch.float64)
    pod = POD(r=train.shape[-1], center=True, energy=energy).fit(X)
    a = pod.encode(X).reshape(train.shape[0], train.shape[1], -1)
    op = OperatorInference(a.shape[-1], use_quadratic=False, use_bias=False, center=False).fit(a)

    def step(u):
        a1 = op.rollout(pod.encode(torch.tensor(u[None], dtype=torch.float64)), 1)[:, -1]
        return pod.decode(a1)[0].numpy()
    return step, a.shape[-1]
