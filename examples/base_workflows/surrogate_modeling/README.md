# Surrogate modeling: recommended family, checked on new solves, exported

A design of experiments over a tool (or a table), the family recommend_architecture picks or none, the library's families compared on one protocol, checks on new tool calls, and an exported model that refuses to extrapolate (pinneapple_analysis.surrogate.workflow).

Flow: `pinneapple_flow/flows/surrogate_modeling.json` (as the example flow `base_workflows/surrogate_modeling`). Run it with `python -m pinneapple_flow example-flows run base_workflows/surrogate_modeling [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries surrogate_modeling`.

Quick configuration (parameter overrides per node): `{}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 15.6 s (commit ca74b8a, 2026-11-03).

| metric | value | how |
|---|---|---|
| `surrogate.cost.solve_seconds` | 0.0177455 | measured: mean over the design of experiments |
| `surrogate.drag_N.rel_rmse` | 0.00108583 | relative RMSE of the output range on solves never trained on |
| `surrogate.drag_N.coverage90` | 0.94 | share of those solves inside the 90 % interval |
| `surrogate.M_root_Nm.rel_rmse` | 0.000109587 | relative RMSE of the output range on solves never trained on |
| `surrogate.M_root_Nm.coverage90` | 0.955 | share of those solves inside the 90 % interval |
| `surrogate.cl_local_max_landing.rel_rmse` | 0.00380997 | relative RMSE of the output range on solves never trained on |
| `surrogate.cl_local_max_landing.coverage90` | 0.9 | share of those solves inside the 90 % interval |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| surrogate.drag_N.held_out_error | pass | gp: relative RMSE 0.11% of the output range on 200 new tool calls (Sobol, seed 1000) (target 5%) |
| surrogate.drag_N.coverage | pass | 90 % interval covers 94.0% of 200 new tool calls (Sobol, seed 1000) (at least 85.8%: the binomial band at that size) |
| surrogate.drag_N.beats_baseline | pass | gp 0.11% vs quadratic response surface 1.48% |
| surrogate.M_root_Nm.held_out_error | pass | gp: relative RMSE 0.01% of the output range on 200 new tool calls (Sobol, seed 1000) (target 5%) |
| surrogate.M_root_Nm.coverage | pass | 90 % interval covers 95.5% of 200 new tool calls (Sobol, seed 1000) (at least 85.8%: the binomial band at that size) |
| surrogate.M_root_Nm.beats_baseline | pass | gp 0.01% vs quadratic response surface 0.22% |
| surrogate.cl_local_max_landing.held_out_error | pass | gp: relative RMSE 0.38% of the output range on 200 new tool calls (Sobol, seed 1000) (target 5%) |
| surrogate.cl_local_max_landing.coverage | pass | 90 % interval covers 90.0% of 200 new tool calls (Sobol, seed 1000) (at least 85.8%: the binomial band at that size) |
| surrogate.cl_local_max_landing.beats_baseline | pass | gp 0.38% vs quadratic response surface 1.77% |
| surrogate.no_extrapolation | pass | the exported predictor refuses inputs outside aspect_ratio in [5, 12], taper in [0.3, 1], twist in [-6, 0] |

## full

done in 10.4 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |
| `surrogate.M_root_Nm.coverage90` | 0.955 | share of those solves inside the 90 % interval |
| `surrogate.M_root_Nm.rel_rmse` | 0.000109587 | relative RMSE of the output range on solves never trained on |
| `surrogate.cl_local_max_landing.coverage90` | 0.9 | share of those solves inside the 90 % interval |
| `surrogate.cl_local_max_landing.rel_rmse` | 0.00380997 | relative RMSE of the output range on solves never trained on |
| `surrogate.cost.solve_seconds` | 0.019853 | measured: mean over the design of experiments |
| `surrogate.drag_N.coverage90` | 0.94 | share of those solves inside the 90 % interval |
| `surrogate.drag_N.rel_rmse` | 0.00108583 | relative RMSE of the output range on solves never trained on |

| check | status | detail |
|---|---|---|
| surrogate.drag_N.held_out_error | pass | gp: relative RMSE 0.11% of the output range on 200 new tool calls (Sobol, seed 1000) (target 5%) |
| surrogate.drag_N.coverage | pass | 90 % interval covers 94.0% of 200 new tool calls (Sobol, seed 1000) (at least 85.8%: the binomial band at that size) |
| surrogate.drag_N.beats_baseline | pass | gp 0.11% vs quadratic response surface 1.48% |
| surrogate.M_root_Nm.held_out_error | pass | gp: relative RMSE 0.01% of the output range on 200 new tool calls (Sobol, seed 1000) (target 5%) |
| surrogate.M_root_Nm.coverage | pass | 90 % interval covers 95.5% of 200 new tool calls (Sobol, seed 1000) (at least 85.8%: the binomial band at that size) |
| surrogate.M_root_Nm.beats_baseline | pass | gp 0.01% vs quadratic response surface 0.22% |
| surrogate.cl_local_max_landing.held_out_error | pass | gp: relative RMSE 0.38% of the output range on 200 new tool calls (Sobol, seed 1000) (target 5%) |
| surrogate.cl_local_max_landing.coverage | pass | 90 % interval covers 90.0% of 200 new tool calls (Sobol, seed 1000) (at least 85.8%: the binomial band at that size) |
| surrogate.cl_local_max_landing.beats_baseline | pass | gp 0.38% vs quadratic response surface 1.77% |
| surrogate.no_extrapolation | pass | the exported predictor refuses inputs outside aspect_ratio in [5, 12], taper in [0.3, 1], twist in [-6, 0] |
