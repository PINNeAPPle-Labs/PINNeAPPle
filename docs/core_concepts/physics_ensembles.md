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
| Physics residual of the **current** case, before predicting | `residual_fn`, `residual_lookahead` | regime changes: the choice reacts on the first new case instead of after past errors pile up |

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

## Five model families and an animated view

![Adaptive ensemble of FNO, GNN, DeepONet, PINN and CNN: reference field (black) against the prediction coloured by the chosen model, weights and timeline](../assets/physics_ensemble/five_model_families.gif)

`examples/physics_ensemble/five_model_families_gif.py` puts an FNO, a MeshGraphNet (multiscale ring graph), a
DeepONet, a PINN (physics loss only) and a dilated 1-D CNN in one ensemble. Each is trained on its own regime of
advection speed and diffusivity; the stream of 80 cases drifts through the five regimes. Relative L2 error per regime
(select mode, 16 cases each):

| regime (training data of) | FNO | GNN | DeepONet | PINN | CNN | **ensemble** |
|---|---|---|---|---|---|---|
| FNO (c = 1, low nu) | **1.2 %** | 240 % | 38 % | 45 % | 72 % | **1.2 %** |
| GNN (c = 0.5, high nu) | 18 % | **5.4 %** | 18 % | 26 % | 74 % | 10 % |
| DeepONet (c = 0, mid nu) | 37 % | 63 % | **11 %** | 18 % | 29 % | 16 % |
| PINN (c = -0.5, high nu) | 39 % | 28 % | 16 % | **4.2 %** | 51 % | 7.0 % |
| CNN (c = -1, low nu) | 128 % | 153 % | 37 % | 27 % | **0.96 %** | 4.7 % |
| all cases | | | | 24 % (best single) | | **7.8 %** |

The ensemble switches to the right family 2 to 6 cases after each regime change (the gap to the diagonal is that
delay); no single model comes close over the whole stream. 90 % intervals covered 90 % of the points.

### Detecting the regime change before the errors pile up

The weights above learn only from cases already seen, so every switch lags. `residual_lookahead` adds a signal that
exists at prediction time: before choosing, each model's PDE residual on the current case is compared with **its own**
median over the last `lookahead_window` cases, and the weights are tilted by `(r / median) ** -residual_lookahead`. A
model whose residual jumps has just left its domain; the learned weights still decide between models whose residuals
did not move. Same five models and stream, three seeds:

| choice of model | mean error | cases with the wrong model (of 80) | cases until the right model, per regime change |
|---|---|---|---|
| oracle (best model of each case, in hindsight) | 0.046 | 0 | 0 |
| past errors only (the GIF) | 0.080 | 13-19 | 5-6, 2-3, 3, 2-3 |
| + residual against each model's own history, `residual_lookahead=3` | **0.051** | 4-6 | 1-2, 0, 0-1, 0 |
| + residual levels compared across models (`lookahead_reference="experts"`) | 0.064 | 12-14 | 0, 3-4, 0, 0 |

Comparing residual levels across models switches instantly too, but it ranks models by how well they satisfy the
PDE, which is not how accurate they are: a numerical scheme that solves its own discrete equation has a tiny residual
and a larger discretisation error than a well-trained neural operator (in the test suite it takes the FNO's own regime
away from it). The self-referenced version only reads jumps, so it has no such bias. The residual still needs to be
informative: a model can be wrong in a way the PDE residual does not see (a wrong boundary condition, a wrong
parameter that is consistent with the equation), and then only the reference errors catch it.

`examples/physics_ensemble/five_model_families_gif.py` takes `main(lookahead=3.0)` to run this version.

`pinneapple_physics.ensemble_viz.animate_ensemble(run, references, "ensemble.gif")` animates any run made with
`run(..., keep_predictions=True)`: the reference field against the ensemble's prediction (coloured by the chosen
model, with its interval and every expert as a faint line), the weights sliding between cases, and a timeline of
weights, chosen model and error revealed case by case. Space-time fields `(n_t, n_x)` are stepped through time so
the waves move; 1-D fields are shown one frame per case.
