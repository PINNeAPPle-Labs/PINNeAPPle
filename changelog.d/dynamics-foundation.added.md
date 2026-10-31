- `pinneapple_analysis.dynamics`: the shared foundation of the dynamical-systems toolkit. No numerical method
  is in it yet; each framework arrives in its own pull request with its benchmarks.
  - A method contract: `MethodCard` (inputs, assumptions, config, outputs, role, failure conditions) and
    `DynamicsResult` (estimate, uncertainty, checked and unchecked assumptions, diagnostics, convergence,
    applicability, reproducibility record, `to_study_result()`).
  - Backend detection that imports nothing (`capabilities`, `require`, `to_numpy`).
  - `Trajectory` / `TrajectoryEnsemble`: irregular sampling, missing values, right-censored windows, and
    interpolation that never extrapolates.
  - Adapters from arrays, tables and CSV files (units read from the headers), `pp.solve` solutions, callables,
    and PDE / FNO / DeepONet snapshots.
  - Guide: `docs/guides/dynamics_toolkit.md`.
