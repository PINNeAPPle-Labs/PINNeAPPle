"""End-to-end demo of the analytic (mesh-free) real-geometry fixtures for
``selector_type="tag"`` presets.

Background
----------
``solve_pde()`` raises ``pinneapple_physics.TagConditionsUnresolved`` for any
preset with ``selector_type="tag"`` boundary/initial conditions unless the
caller supplies real geometry -- see that exception's docstring and
``docs/dev/AUDIT_REPORT.md``. 23 of the 65 registered presets have a real
canonical box/cylinder domain fully described by their own parameters (no
mesh file needed): see
``pinneapple_physics/pde_environment/presets/tag_geometry.py`` for the full
per-preset face-mapping justification and the list of presets that do NOT
qualify (real airfoil/car-body/furnace geometry, or an unlocated-face
ambiguity) and therefore correctly keep raising ``TagConditionsUnresolved``.

This script trains through ``solve_pde()`` for real on one box preset
(``laplace_2d``) and one cylinder preset (``pipe_flow_3d``), and prints the
per-tag boundary losses both before and after training via a direct
``compile_problem`` call, so the per-tag masking can be seen to be doing
real, distinct work (not a coincidental pass-through -- the same check
``04_heat3d_stl_box.py`` does for the STL-based builder).
"""
from __future__ import annotations

import numpy as np
import torch

from _utils import ensure_repo_on_path

ensure_repo_on_path()

import pinneapple_physics as pp
from pinneapple_physics import LossWeights, compile_problem
from pinneapple_physics.pde_environment.presets.tag_geometry import (
    build_tag_batch,
    solve_pde_kwargs_from_batch,
)
from pinneapple_neural.architectures.registry import ModelRegistry
import pinneapple_neural.architectures  # noqa: F401  registers the model zoo


def _per_tag_losses(spec, model, batch) -> dict:
    loss_fn = compile_problem(spec, weights=LossWeights(w_pde=1.0, w_bc=10.0))
    tb = {}
    for k, v in batch.items():
        if k == "ctx":
            tb[k] = v
            continue
        tb[k] = torch.as_tensor(np.asarray(v), dtype=torch.bool if k.startswith("mask_") else torch.float32)
    out = loss_fn(model, None, tb)
    return {k: float(v.detach()) for k, v in out.items()}


def run(name: str, *, epochs: int = 200, hidden_dim: int = 32, n_layers: int = 4):
    print(f"\n=== {name} ===")
    torch.manual_seed(0)
    np.random.seed(0)

    spec = pp.get_preset(name)
    model = ModelRegistry.build(
        "modified_mlp", in_dim=len(spec.coords), out_dim=len(spec.fields),
        hidden_dim=hidden_dim, n_layers=n_layers,
    )

    batch = build_tag_batch(name, spec, n_col=4000, n_bc_per_face=500, seed=1)
    print("tag_masks sizes:", {k: int(v.sum()) for k, v in batch["ctx"]["tag_masks"].items()})

    before = _per_tag_losses(spec, model, batch)
    print("per-tag losses BEFORE training:", before)

    kwargs = solve_pde_kwargs_from_batch(batch)
    result = pp.solve_pde(spec, model, epochs=epochs, n_collocation=4000, seed=1, lr=1e-3, **kwargs)
    losses = result["history"]["loss"]
    print(f"aggregate loss: {losses[0]:.4g} -> {losses[-1]:.4g} (ratio {losses[-1] / losses[0]:.4g})")

    after = _per_tag_losses(spec, result["model"], batch)
    print("per-tag losses AFTER training:", after)


def main():
    # Box domain, single "boundary" tag covering all 4 sides.
    run("laplace_2d")
    # Cylinder domain (inlet/outlet disks + lateral wall ring).
    run("pipe_flow_3d")


if __name__ == "__main__":
    main()
