# Use a trained model (zoo or yours) instead of training, gated and re-verified

A zoo model, a surrogate-workflow export or a model file with its card; the applicability gate (validity range, units, reference solves) before use; forward or inverse through the model; the result re-verified by the reference solver (pinneapple_analysis.surrogate.model_source).

Flow: `pinneapple_flow/flows/use_trained_model.json` (as the example flow `base_workflows/use_trained_model`). Run it with `python -m pinneapple_flow example-flows run base_workflows/use_trained_model [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries use_trained_model`.

Quick configuration (parameter overrides per node): `{}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 0.2 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `model.estimate.air_velocity_m_s` | 2.49315 | bounded least squares through the model |
| `model.misfit_rel` | 4.6781e-09 | relative misfit of the model at the estimate |
| `model.reverified.r_base_k_w.rel_diff` | 0.000780443 | reference solver at the estimate vs observed |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| gate.inputs | pass | every input of the model is given |
| gate.validity_range | pass | all 1 query points inside the model's validity range |
| gate.units | pass | the problem's units convert to the model's |
| gate.reference | pass | largest relative error against 1 reference solves 0.82% (allowed 2.1%; reference 0.0358 s per solve) |
| gate.inputs | pass | every input of the model is given |
| gate.validity_range | pass | all 1 query points inside the model's validity range |
| gate.units | pass | the problem's units convert to the model's |
| gate.reference | pass | the card's 8 metrics recomputed on the stored reference: largest change 0.00% |
| inverse.misfit | pass | the model reproduces the observations to 0.00% (allowed 2.1%, twice the model's claimed error) |
| inverse.inside_bounds | pass | the estimate is inside the model's range |
| inverse.reverified | pass | the reference solver at the estimate reproduces the observations to 0.08% (allowed 2.1%) |

## full

done in 0.2 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `model.estimate.air_velocity_m_s` | 2.49315 | bounded least squares through the model |
| `model.misfit_rel` | 4.6781e-09 | relative misfit of the model at the estimate |
| `model.reverified.r_base_k_w.rel_diff` | 0.000780443 | reference solver at the estimate vs observed |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| gate.inputs | pass | every input of the model is given |
| gate.validity_range | pass | all 1 query points inside the model's validity range |
| gate.units | pass | the problem's units convert to the model's |
| gate.reference | pass | largest relative error against 1 reference solves 0.82% (allowed 2.1%; reference 0.0549 s per solve) |
| gate.inputs | pass | every input of the model is given |
| gate.validity_range | pass | all 1 query points inside the model's validity range |
| gate.units | pass | the problem's units convert to the model's |
| gate.reference | pass | the card's 8 metrics recomputed on the stored reference: largest change 0.00% |
| inverse.misfit | pass | the model reproduces the observations to 0.00% (allowed 2.1%, twice the model's claimed error) |
| inverse.inside_bounds | pass | the estimate is inside the model's range |
| inverse.reverified | pass | the reference solver at the estimate reproduces the observations to 0.08% (allowed 2.1%) |
