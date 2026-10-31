# Fisher information and experimental design

`pinneapple_analysis.dynamics.fisher` answers four questions about a model and a set of measurements:

- how much the measurements say about the parameters;
- what the best achievable precision is, on a parameter or on any quantity computed from the parameters;
- which parameters cannot be told apart;
- where and when to measure next.

It is built on the models of `pinneapple_analysis.inverse_problems`: `ForwardModel`, `CallableModel`,
`ExpressionModel`, and the SBML-based `ODEForwardModel`. It uses their derivatives through
`inverse_problems.forward.jacobian`, in this order of preference:

1. the model's own sensitivities;
2. PyTorch autograd through `predict_torch`;
3. central differences.

For the full identifiability analysis (profile likelihood, posterior) it delegates to the existing
`analyze_identifiability`. Results follow the toolkit's [shared contract](dynamics_toolkit.md).

## Three quantities that are not the same

| Quantity | Where | What it is |
|---|---|---|
| Theoretical bound | `CramerRaoBound`, `FIM.crlb()` | `Cov(θ̂) ⪰ I(θ)⁻¹` for **unbiased** estimators, under the regularity conditions below. It depends on the model and the design, not on any estimator. |
| Empirical performance | `estimator_performance(...)` | The spread of least-squares (Gaussian maximum-likelihood) fits of simulated data sets. It can exceed the bound (non-linearity, few data) or fall below it (bias, active parameter bounds). |
| Bayesian uncertainty | `FIM.with_prior(P).crlb()` | The Laplace posterior covariance `(I + P)⁻¹`. A prior makes the estimator biased, so this is **not** a Cramér-Rao bound. The result says so in `kind`. |

**Regularity conditions.** Each result lists them as unchecked assumptions:

- the likelihood is differentiable in θ;
- its support does not depend on θ;
- the estimator is unbiased;
- the model is correct.

## What it does

| Task | Call |
|---|---|
| Information, Gaussian noise | `FisherInformation.gaussian(model, obs, parameters)`: `JᵀΣ⁻¹J`. With `sigma_model=` (noise that depends on θ, e.g. a relative error) it adds `2 ∂σ ∂σᵀ / σ²`. |
| Information, counts | `FisherInformation.poisson(model, obs)`: `Jᵀ diag(1/λ) J` |
| Information, any likelihood | `FisherInformation.from_loglik(loglik, theta, simulate, backend="fd" \| "torch" \| "jax")`: `E[s sᵀ]` by Monte Carlo, with the standard error of each entry in `mc_se` |
| Information, linear regression | `FisherInformation.linear_gaussian(X, sigma)`, exact |
| Bounds | `fim.crlb(target)`: per-parameter std and correlations, plus `target_std` for `g(θ)` from `∇gᵀ I⁻¹ ∇g` |
| Identifiability | `ParameterIdentifiability(model, obs).local()` (rank, null directions, status per parameter) and `.full()` (delegates to `analyze_identifiability`) |
| Design | `ExperimentalDesign(model, candidates, base=..., prior_precision=...).select(n, "D" \| "A" \| "E" \| "c" \| "bayes-D", target=...)` |
| When to measure | `ObservationScheduler(model, (t0, t1), sigma=...).schedule(n, criterion)` and `.next_best(base)` |
| Value of one more measurement | `InformationGain(fim, model).of(new_obs, target)`: Δ log det, the expected information gain in nats, and the variance reduction of a target |
| Comparing schedules | `SamplingStrategy.compare(model, {"uniform": ..., "log": ..., ...})`, with the builders `uniform`, `logspaced`, `random` and `endpoints` |
| Local sensitivities | `SensitivityAnalyzer(model).local(obs)` (raw, relative and noise-scaled) and `.target(g)`. Global Sobol and Morris analysis is in `pinneapple_analysis.sensitivity`. |

### Targets

A target can be any function of the parameters, for example a prediction at a given time, or a first-passage time
computed by [`FirstPassageAnalyzer`](dynamics_first_passage.md) on the model's own predicted trajectory. It is
differentiated by central differences in the physical parameters.

### Backends

PyTorch and JAX are optional. They are imported only when `backend="torch"` or `"jax"` is requested. If the
requested one is missing, `BackendUnavailable` gives the install command.

## Implementation notes

**Physical and scaled parameters.** `jacobian` differentiates with respect to the scaled parameters `z` of
`ParameterSet`: `ln θ` for log parameters, `θ/|θ₀|` for linear ones. The module converts to physical parameters, so
every bound is in the parameter's own unit.

**Design.** The design is a greedy selection from the candidates, followed by Fedorov exchange passes, with
replicates allowed. The criteria are:

- D: maximise `log det`;
- A: minimise `tr I⁻¹`;
- E: maximise `λ_min`;
- c: minimise `gᵀ I⁺ g`;
- Bayesian D: D with the prior precision.

The c criterion accepts a *singular* information matrix when `g` lies in its range. This is the usual case: the best
design for one target often cannot identify every parameter. For example, to estimate the time a decay reaches a
level, measure near that time.

The result is a local optimum of a combinatorial problem, not a certified global one. It is also *locally* optimal,
that is, optimal at the nominal parameters.

**Identifiability.**
- A zero eigenvalue of `I` (relative threshold `rtol`) is a structurally non-identifiable direction at that point.
  The direction is reported as a parameter combination; for `y = a b t` it is `d ln a = −d ln b`.
- A finite but large bound (above `practical_threshold` times the value) is a practical non-identifiability.
- The verdict is numerical and local, not a symbolic proof.

## Examples

**When it works.** Take the decay `y = A e^{−kt}` with σ = 0.05. The D-optimal two-point design is known to be
`{0, 1/k}` (Box & Lucas 1959), and the scheduler finds it:

```python
import math
import numpy as np
from pinneapple_analysis.inverse_problems.forward import CallableModel, Parameter
from pinneapple_analysis.dynamics.fisher import ObservationScheduler

m = CallableModel(lambda th, inp: th["A"] * np.exp(-th["k"] * inp["t"]), inputs=["t"],
                  parameters=[Parameter("A", 2.0, 0.1, 10), Parameter("k", 0.5, 0.01, 5)])
sch = ObservationScheduler(m, (0, 10), n_grid=401, sigma=0.05)
print(sch.schedule(2, "D").estimate["times"])                # [0.0, 2.0]

tau = lambda th: math.log(th["A"] / 0.5) / th["k"]           # time at which y reaches 0.5
r = sch.schedule(3, "c", target=tau)
print(r.estimate["times"], r.uncertainty["target_std"])      # three points near tau = 2.77; std 0.1161
```

The delta-method value for three measurements exactly at τ is `σ/(k c √3) = 0.1155`. The D-optimal design gives the
same target a bound of 0.189.

**When it tells you the data cannot answer.** With `y = A e^{−kt} + B` measured at a single time, the rank is 1, so
two parameter combinations are invisible. Measured at two times, the rank is 2. Only a spread of times identifies all
three parameters. `ParameterIdentifiability(...).local()` reports each case. With `y = a b t` no schedule helps: the
null direction `a/b` is structural.

**When the bound is optimistic.** Take four noisy points (σ = 0.3) of the decay. The least-squares estimate of `k`
has efficiency 0.82 (95 % interval 0.75–0.90), and its bias is 0.023 ± 0.005. That is the empirical estimator
falling short of the theoretical bound, and the benchmark row reports it.

## Cost

| Operation | Cost |
|---|---|
| Gaussian or Poisson information | one Jacobian, so `p` extra model evaluations by finite differences (2`p` for central differences), or one sensitivity solve, plus `O(n p²)` |
| Score Monte Carlo | `n_mc` simulated data sets, each with 2`p` log-likelihood evaluations (or one autodiff pass) |
| Design of `n` points from `m` candidates | `O(n m p³)` greedy, plus `O(passes · n m p³)` exchange. The 401-candidate, 2-parameter designs above take under 0.11 s. |

## Benchmarks

To reproduce the table:

```bash
python -m pinneapple_analysis.dynamics.benchmarks fisher
```

It uses `seed=0`, 1000 replicates and 20 000 Monte Carlo samples. The values were measured on 2026-10-31.

| Case | Quantity | Reference | Computed | z / tolerance | Pass |
|---|---|---|---|---|---|
| linear_fim | max relative error of F | 0 (`XᵀX/σ²`) | 5.0e-13 | tol 1e-6 | yes |
| ols_efficiency_b0 | CRLB var / empirical var | 1 | 1.079 ± 0.048 | z = 1.6 | yes |
| ols_efficiency_b1 | CRLB var / empirical var | 1 | 1.103 ± 0.049 | z = 2.1 | yes |
| decay_k_optimal_time | best single time for k | 1/k = 2 | 2.000 | grid 0.025 | yes |
| decay_d_optimal_second_time | second D-optimal time (first at 0) | 2 | 2.000 | grid 0.025 | yes |
| tau_c_optimal_std | c-optimal bound on τ (n = 3) | σ/(k c √n) = 0.1155 | 0.1161 | 2 % | yes |
| poisson_score_mc | F by score Monte Carlo | Σt/θ = 2.5 | 2.465 ± 0.025 | z = 1.4 | yes |
| heteroscedastic_score_mc | F with relative noise | 8851 | 8675 ± 86 | z = 2.0 | yes |
| conjugate_posterior | max relative error | 0 | 8.6e-13 | tol 1e-6 | yes |
| nonlinear_efficiency_small_n | efficiency for k, 4 points, σ = 0.3 | 1 (bound) | 0.82 (CI 0.75–0.90) | demonstration | — |

**The OLS efficiencies.** The `ols_efficiency_*` rows scatter around 1 across seeds, by about 0.045. With seeds 1–4
the b1 efficiency was 1.016, 0.992, 1.076 and 1.021.

**What the benchmarks do not cover.**
- They cover models with exact information or exact optimal designs, so they show that the computation is right for
  those models. They do not establish how a locally optimal design performs when the nominal parameters are wrong.
- `estimator_performance` measures that effect at the nominal point only.
- JAX is not installed in the environment that produced this table, so the JAX score is tested only where JAX is
  present (the test skips otherwise).
