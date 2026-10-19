# PhysicsNeMo trains, PINNeAPPle says when to trust

PhysicsNeMo's `FNO` and `MeshGraphNet` are trained on the zoo dataset `darcy_2d_li2021`
(`-div(a grad u) = 1`, `u = 0` on the boundary, `a ∈ {3, 12}`, 1000 train / 200 test pairs on 128 × 128).
Both models get the same wall-clock budget. PINNeAPPle trains nothing. It checks every prediction against the
physics, without the reference solution, and gives a deploy decision per model and per input regime:

| Check | How (`pinneapple_physics.darcy_checks`) |
|---|---|
| PDE residual | finite-volume residual of the reference solver's own discretisation, turned into guaranteed bounds on the energy-norm error (`residual_error_bound`) and the L2 error (`residual_l2_bound`) |
| Boundary condition | RMS of `u` extrapolated to the boundary, over max \|u\| |
| Flux balance | net outward flux through the boundary against the integral of the source |
| Positivity | `u >= 0` (maximum principle, `f >= 0`) |
| Applicability | coefficient values and statistics inside the training envelope (`CoefficientEnvelope`) |
| Reference (evaluation only) | relative L2 against the reference solver |

The figure shows where each prediction errs (error maps of `u` and of the flux, estimated from the residual
alone) and which uses it can serve (`use_verdicts`: interior field, walls and fluxes, inputs). The certified
verdicts are given per use profile (`darcy_checks.PROFILES`: `fields` for maps of `u`, `fluxes` for flow
rates and wall gradients). Each profile's limits are set a priori and come with a ratio *value / limit*. A second
layer compares each model with its own values on held-out in-distribution inputs (`calibrate_tolerances`, no labels).

```bash
python run.py data                                   # shifted test sets (contrast {1, 20}; 3x shorter length scale)
python run.py train --model fno --budget-s 3600
python run.py train --model meshgraphnet --budget-s 3600
python run.py evaluate                               # results.json, trust cards, figure, GIF, check/predictions
python run.py datapipe                               # the same kind of checks on PhysicsNeMo's Darcy2D datapipe
python run.py report                                 # Veriphysics Decision Record + Evidence Report PDF (scope of use)
pinneapple check --config check/pinneapple-check.toml check/predictions/contrast   # exit code 1: out of range
```

Needs `pip install nvidia-physicsnemo torch_geometric torch_scatter` (MeshGraphNet uses PyTorch Geometric).
Results, sizes and timings are in [docs/guides/physicsnemo_trust.md](../../../docs/guides/physicsnemo_trust.md).
`check/` holds two exported predictions per model and regime. It also has their configuration (validated limits,
training envelope of `a`), the per-sample verdicts the figure shows (`sample_verdicts.json`, equal to what
`pinneapple check` returns), and the rules the out-of-range predictions break (`expected.json`). The `darcy-trust`
job of `.github/workflows/physics-check.yml` runs the GitHub Action on them.
