# Adaptive ensembles of physics models

`pinneapple_physics.ensemble.PhysicsEnsemble` (also `pp.ensemble.PhysicsEnsemble`) selects and combines physics
models case by case: neural operators (FNO, DeepONet), graph networks (MeshGraphNet), PINNs, numerical solvers and
closed forms. It uses the same prequential core as the [adaptive forecaster](adaptive_forecasting.md)
(`pinneapple_physics.online_learning`): Fixed-Share weights, AdaHedge over the learning and switching rates, adaptive
conformal intervals. Every case is predicted with weights learned from earlier cases only, so the model choice
cannot overfit.

```python
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable, from_solution, from_torch

experts = [
    from_torch("fno", fno_model, to_input=lambda case: make_fno_input(case), from_output=lambda y, case: y[0]),
    from_torch("gnn", meshgraphnet, to_input=lambda case: make_graph(case), from_output=lambda y, case: y[0]),
    from_solution("pinn", pp.solve(problem, "pinn"), points=lambda case: case["points"]),
    from_callable("solver", lambda case: run_solver(case), cost=50.0),
]
ens = PhysicsEnsemble(experts, mode="select")      # or "combine"
for case in stream:
    out = ens.predict(case)                        # field, interval, per-expert fields, weights, active model
    ens.update(case, reference=high_fidelity(case))  # when a reference exists
```

## What the weights can learn from

| Signal | Argument | When to use |
|---|---|---|
| Reference field (sensor, experiment, high-fidelity run) | `update(case, reference)` | relative L2 error; validation campaigns, digital twins with sensors |
| Physics residual, no reference | `residual_fn`, `residual_weight` | operation without ground truth: models that violate the PDE lose weight |
| Cost | `cost_weight`, `Expert(cost=...)` | adaptive fidelity: the cheap surrogate while it is good enough, the solver when it is not |

With `mode="select"` and `predict(case, select_only_leader=True)` only the leading model runs, except every
`explore_every` cases, when all run so their errors stay current. Experts that raise or return non-finite values are
left out of that case and scored with the worst loss.

Other options: `conserve=(weights, target_fn)` projects the combined field onto a conserved total;
`min_weight` keeps near-zero weights out of the "combine" average; `point_weights` gives quadrature weights for the norms.
`fit_static_weights(predictions, references)` fits fixed convex weights on a validation set and reports their
cross-validated error next to each single model.

## Results (advection-diffusion with exact solutions, `pinneapple_physics.advection_diffusion_1d`)

150 cases whose diffusivity moves through three regimes; every learned model was trained on one regime only
(`examples/physics_ensemble/fno_pinn_gnn_solvers.py`):

| | low nu | mid nu | high nu | all cases |
|---|---|---|---|---|
| FNO (trained on low nu) | 0.57 % | 4.4 % | 40 % | |
| MeshGraphNet (trained on mid nu) | 19 % | 16 % | 38 % | |
| PINN (high nu, physics loss only, no data) | 16 % | 13 % | 6.5 % | |
| upwind FD | 4.1 % | 3.4 % | 2.5 % | |
| Lax-Wendroff FD, coarse grid | 2.5 % | 1.2 % | 0.10 % | 1.25 % (best single model) |
| **Adaptive ensemble, select** | **0.57 %** | 2.0 % | **0.10 %** | **0.88 %** |

Relative L2 error of the space-time field. 90 % intervals covered 93 % of the points. In the tests
(`tests/test_physics_ensemble.py`) the residual-only mode (no reference at all) removes the out-of-distribution FNO
in the high-diffusion regime, and the cost-aware lazy mode cuts the compute by more than 70 % against running every
model while keeping the error low. With CPU-sized training the MeshGraphNet stays the weakest expert; the ensemble
learns to ignore it. In "combine" mode a model that fails badly still costs a little during a regime change, so
"select" is the better default when some experts can be far off.

The PDE residual is a good detector of failure (it flags the out-of-distribution FNO at once), not a ranking of the
most accurate model: a learned field can have a larger residual than a slightly less accurate numerical solution.
