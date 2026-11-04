# Base workflows

The library's method workflows as flows: surrogate modeling, a trained or zoo model used instead of training, VeriPhysics for a model trained elsewhere, inverse modeling, equation discovery, model benchmarking, design optimization, reduced-order modeling and multiphysics coupling, each with its checks and the measured run times and results.

Regenerate with `python -m pinneapple_flow example-flows gallery base_workflows` (runs every entry, quick and full, and rewrites these files from the runs).

| entry | flow | quick | full |
|---|---|---|---|
| [surrogate_modeling](surrogate_modeling/README.md) | `surrogate_modeling` | done, 15.6 s | done, 10.4 s |
| [use_trained_model](use_trained_model/README.md) | `use_trained_model` | done, 0.2 s | done, 0.2 s |
| [veriphysics_grade](veriphysics_grade/README.md) | `veriphysics_grade` | done, 6.1 s | done, 17.6 s |
| [inverse_modeling](inverse_modeling/README.md) | `inverse_modeling` | done, 0.1 s | done, 0.4 s |
| [equation_discovery](equation_discovery/README.md) | `equation_discovery` | done, 3.1 s | done, 3.1 s |
| [model_benchmarking](model_benchmarking/README.md) | `model_benchmarking` | done, 7.7 s | done, 10.6 s |
| [design_optimization](design_optimization/README.md) | `design_optimization` | done, 9.8 s | done, 7.2 s |
| [reduced_order_modeling](reduced_order_modeling/README.md) | `reduced_order_modeling` | done, 1.7 s | done, 3.4 s |
| [multiphysics_coupling](multiphysics_coupling/README.md) | `multiphysics_coupling` | done, 0.1 s | done, 0.1 s |
