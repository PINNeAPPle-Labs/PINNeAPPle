# Model benchmarking: equal data, intervals, a winner only when separated

Library families trained on the same rows and existing models on the same test points over several seeds, ranked with paired-bootstrap intervals (pinneapple_analysis.surrogate.benchmark_workflow).

Flow: `pinneapple_flow/flows/model_benchmarking.json` (as the example flow `base_workflows/model_benchmarking`). Run it with `python -m pinneapple_flow example-flows run base_workflows/model_benchmarking [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery base_workflows --entries model_benchmarking`.

Quick configuration (parameter overrides per node): `{"bench": {"n_test": 60}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 7.7 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `bench.gp.rel_rmse` | 0.00200424 | mean over seeds, inside the box |
| `bench.mlp_ensemble.rel_rmse` | 0.00585586 | mean over seeds, inside the box |
| `bench.quadratic.rel_rmse` | 0.0147537 | mean over seeds, inside the box |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

## full

done in 10.6 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `bench.gp.rel_rmse` | 0.00201712 | mean over seeds, inside the box |
| `bench.mlp_ensemble.rel_rmse` | 0.0055055 | mean over seeds, inside the box |
| `bench.quadratic.rel_rmse` | 0.0146705 | mean over seeds, inside the box |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |
