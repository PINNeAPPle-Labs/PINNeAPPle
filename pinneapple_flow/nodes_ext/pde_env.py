"""The PDE environment examples (``examples/pde_environment``): a ``selector_type="tag"`` preset trained through
``solve_pde`` on its analytic real-geometry fixture, with the per-tag loss terms before and after training
(``07_tagged_presets_real_geometry``)."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("pde_env.tagged_preset", "Tagged preset on its real geometry: per-tag losses before and after training",
      category="PINN", outputs=[Port("metrics", "metrics")],
      params=[Param("preset", "str", "laplace_2d", help="a preset with an analytic tag-geometry fixture "
                                                         "(TAG_GEOMETRY_FIXTURES), e.g. laplace_2d, pipe_flow_3d"),
              Param("epochs", "int", 200, minimum=1), Param("hidden_dim", "int", 32, minimum=4),
              Param("n_layers", "int", 4, minimum=1), Param("n_col", "int", 4000, minimum=16),
              Param("n_bc_per_face", "int", 500, minimum=1), Param("lr", "float", 1e-3, minimum=1e-6),
              Param("seed", "int", 1, minimum=0, help="seed of the batch and of solve_pde")],
      calls="pinneapple_physics.get_preset + pde_environment.presets.tag_geometry.build_tag_batch / per_tag_losses / "
            "solve_pde_kwargs_from_batch + pinneapple_physics.solve_pde + ModelRegistry.build('modified_mlp')")
def tagged_preset(ctx, inputs, params):
    """Per-tag loss terms of a tagged preset before and after training on its real geometry. The boundary points
    per tag come from the preset's box / cylinder fixture, a modified MLP is trained on them with ``solve_pde``,
    and every compiled loss term (``pde``, ``bc_<tag>``, ``total``) is reported before and after."""
    import numpy as np
    import torch

    import pinneapple_neural.architectures  # noqa: F401  registers the model zoo
    import pinneapple_physics as pp
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_physics.pde_environment.presets.tag_geometry import (
        build_tag_batch,
        per_tag_losses,
        solve_pde_kwargs_from_batch,
    )
    name = str(params["preset"])
    seed = int(params["seed"])
    torch.manual_seed(0)
    np.random.seed(0)
    spec = pp.get_preset(name)
    model = ModelRegistry.build("modified_mlp", in_dim=len(spec.coords), out_dim=len(spec.fields),
                                hidden_dim=int(params["hidden_dim"]), n_layers=int(params["n_layers"]))
    batch = build_tag_batch(name, spec, n_col=int(params["n_col"]), n_bc_per_face=int(params["n_bc_per_face"]),
                            seed=seed)
    tags = {k: int(v.sum()) for k, v in batch["ctx"]["tag_masks"].items()}
    for k, v in tags.items():
        ctx.metric(f"tag_points.{k}", v)
    before = per_tag_losses(spec, model, batch)
    result = pp.solve_pde(spec, model, epochs=int(params["epochs"]), n_collocation=int(params["n_col"]), seed=seed,
                          lr=float(params["lr"]), **solve_pde_kwargs_from_batch(batch))
    losses = result["history"]["loss"]
    after = per_tag_losses(spec, result["model"], batch)
    ctx.metric("loss_first", float(losses[0]), "aggregate solve_pde loss, first epoch")
    ctx.metric("loss_last", float(losses[-1]), "aggregate solve_pde loss, last epoch")
    ctx.metric("loss_ratio", float(losses[-1] / losses[0]))
    for k in before:
        ctx.metric(f"before.{k}", before[k])
        ctx.metric(f"after.{k}", after.get(k))
    ctx.table(f"{name}: loss terms (w_pde = 1, w_bc = 10)", ["term", "before", "after"],
              [[k, before[k], after.get(k)] for k in before])
    ctx.summary(f"{name}: aggregate loss {losses[0]:.4g} -> {losses[-1]:.4g}")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="pde_env.tagged_preset", title="A tagged-boundary preset trained on its real geometry, per-tag losses",
        domain="PINN", example="pde_environment",
        keywords={"tagged preset": 4, "tag geometry": 4, "per-tag loss": 4, "tag_masks": 4, "selector_type": 3,
                  "pde environment": 3, "tagged boundary": 3},
        min_score=3, nodes=("pde_env.tagged_preset",), references=("raissi2019pinn",),
        template="pde_environment/07_tagged_presets_real_geometry", quick=True,
        description="examples/pde_environment/07_tagged_presets_real_geometry.py"))


_register_capabilities()
