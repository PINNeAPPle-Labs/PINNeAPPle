- `pinneapple_analysis.dynamics.fisher`: Fisher information, Cramér-Rao bounds and experimental design on the
  `pinneapple_analysis.inverse_problems` forward models (including `ODEForwardModel` sensitivities), guide
  `docs/guides/dynamics_fisher.md`.
  - `FisherInformation`: Gaussian (with optional parameter-dependent noise), Poisson, linear Gaussian, and any
    likelihood by score Monte Carlo (finite differences, PyTorch or JAX, detected at call time).
  - `CramerRaoBound` on parameters and on targets `g(theta)` (a prediction, a first-passage time), kept apart from
    the Laplace posterior (with a prior) and from `estimator_performance` (empirical least squares).
  - `ParameterIdentifiability`: local verdict from the information matrix, with the full analysis delegated to
    `analyze_identifiability`.
  - `ExperimentalDesign` / `ObservationScheduler`: D, A, E, c and Bayesian D criteria, greedy selection plus exchange
    passes.
  - `InformationGain`, `SamplingStrategy`, `SensitivityAnalyzer`.
  - Benchmarks: `python -m pinneapple_analysis.dynamics.benchmarks fisher`.
