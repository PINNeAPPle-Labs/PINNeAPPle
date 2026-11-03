# VeriPhysics grade of a model trained elsewhere

The equations, conditions and reference data a model claims to satisfy; its residuals by finite differences of its outputs (any framework), applicability and robustness; APPROVED / REVIEW / REJECT with the evidence (pinneapple_veriphysics.model_grade).

Flow: `pinneapple_flow/flows/veriphysics_grade.json` (as the example flow `base_workflows/veriphysics_grade`). Run it with `python -m pinneapple_flow example-flows run base_workflows/veriphysics_grade [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries veriphysics_grade`.

Quick configuration (parameter overrides per node): `{"grade": {"steps": 400}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 6.1 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `grade.grade.pde_residual.u_t = alpha*lap(u)` | 0.0130786 |  |
| `grade.grade.conditions.initial t_min: u = sin(pi*x)` | 0.00120781 |  |
| `grade.grade.conditions.dirichlet boundary: u = 0` | 0.000706927 |  |
| `grade.grade.reference` | 0.0019364 |  |
| `grade.grade.robustness` | 10.1326 |  |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| grade.pde_residual | pass | u_t = alpha*lap(u): relative RMS residual 1.31e-02 at 2000 fresh points (allowed 0.05) |
| grade.conditions | pass | initial t_min: u = sin(pi*x): 1.21e-03; dirichlet boundary: u = 0: 7.07e-04 (allowed 0.02) |
| grade.reference | pass | relative L2 error 1.94e-03 on 400 reference points (allowed 0.05) |
| grade.applicability | pass | the graded box is inside the model's declared validity |
| grade.robustness | pass | output change for a 0.001-span perturbation is 10.1x the change for a 10x smaller one (about 10 for a smooth model) |

## full

done in 17.6 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `grade.grade.pde_residual.u_t = alpha*lap(u)` | 0.0113425 |  |
| `grade.grade.conditions.initial t_min: u = sin(pi*x)` | 0.00102559 |  |
| `grade.grade.conditions.dirichlet boundary: u = 0` | 0.00107509 |  |
| `grade.grade.reference` | 0.00153376 |  |
| `grade.grade.robustness` | 10.1341 |  |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| grade.pde_residual | pass | u_t = alpha*lap(u): relative RMS residual 1.13e-02 at 2000 fresh points (allowed 0.05) |
| grade.conditions | pass | initial t_min: u = sin(pi*x): 1.03e-03; dirichlet boundary: u = 0: 1.08e-03 (allowed 0.02) |
| grade.reference | pass | relative L2 error 1.53e-03 on 400 reference points (allowed 0.05) |
| grade.applicability | pass | the graded box is inside the model's declared validity |
| grade.robustness | pass | output change for a 0.001-span perturbation is 10.1x the change for a 10x smaller one (about 10 for a smooth model) |
