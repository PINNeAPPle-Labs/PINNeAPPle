# First-passage and hitting times

`pinneapple_analysis.dynamics.first_passage` estimates when a trajectory first reaches a boundary:

`tau = inf{ t >= 0 : X(t) in boundary of Omega }`

It works on a single trajectory, on an ensemble of observed or simulated trajectories, or on a stochastic model that
it simulates itself. Trajectories can come from measurements, solvers, closed forms or a trained network's
predictions (see [the shared foundation](dynamics_toolkit.md) for how to load them).

All results follow the toolkit's contract. Each one carries:

- the estimate and its uncertainty;
- the assumptions, each marked as checked or not;
- numerical diagnostics;
- the conditions under which the method does not apply;
- a reproducibility record (seed, input hash, versions).

## What it does

| Question | Call | Method |
|---|---|---|
| When did this trajectory first cross the boundary? | `FirstPassageAnalyzer(bc).crossing(trajectory)` | Linear interpolation inside the sampling interval that contains the crossing. Works on irregular grids and skips missing samples. |
| What is the distribution of τ over these observed runs? | `.from_trajectories(ensemble)` | One crossing per run. A run that never crosses is right-censored at the end of its window. The survival function is the Kaplan-Meier estimate with Greenwood log-log bands. |
| What does this stochastic model predict? | `.monte_carlo(sde, x0, t_max=..., dt=..., n_paths=...)` | Euler-Maruyama paths, with a Brownian-bridge crossing correction and optional reflecting walls. Paths not hit by `t_max` are censored there. |
| How much does the step size `dt` still matter? | `.dt_convergence(sde, x0, dts=[...])` | Repeats the run at each step size. Reports the successive changes, the observed order and a Richardson estimate. |
| How sensitive is the answer to a parameter, or to the choice of noise model? | `.sensitivity(lambda v: SDE(...), values, ...)` or `.sensitivity({"additive": ..., "multiplicative": ...}, ...)` | Common random numbers (same seed for every model), plus a central-difference derivative for a parameter sweep. |

Each result has a `HittingTimeDistribution` for every boundary (`result.distribution` when there is only one). It
gives:

| Method | Returns |
|---|---|
| `survival(t)` | `S(t) = P(τ > t)` |
| `survival_function()` | `S(t)` with its confidence band |
| `cdf(t)` | `P(τ ≤ t)` |
| `hitting_probability(T)` | `P(τ ≤ T)` and its interval, as a `CrossingProbability`. `.no_crossing` gives the probability of *not* crossing within `T`. |
| `quantile(q)`, `quantile_ci(q)`, `median` | quantiles of τ and their intervals |
| `prediction_interval(level)` | interval for the τ of a new realisation |
| `mean()` | the mean, or the restricted mean when some samples are censored (see below) |
| `density(bins)` | histogram density of the event times |

The probability interval depends on censoring:

- If no sample is censored before `T`, it is a binomial proportion with a Wilson interval.
- Otherwise it is the Kaplan-Meier estimate with its band.

### Boundaries

| Boundary | Call | Hit when |
|---|---|---|
| upper threshold | `BoundaryCondition.threshold(b, "up", component)` | `x ≥ b` |
| lower threshold | `BoundaryCondition.threshold(b, "down", component)` | `x ≤ b` |
| two-sided threshold | `BoundaryCondition.threshold(b, "either")` | `x` reaches `b` from the side it started on |
| interval (exit) | `BoundaryCondition.interval(l, u)` | `x` leaves `(l, u)` |
| level set, any dimension | `BoundaryCondition.level_set(phi)` | `phi(x) ≥ 0`, e.g. `phi = |x| - R` to leave a ball |
| reflecting wall | `BoundaryCondition.threshold(L, "down", behaviour="reflecting")`, passed as `reflecting=[...]` | never: the state is mirrored back into the domain |

A list of boundaries gives one τ per boundary on the same trajectories. Use this for several thresholds, or for the
two sides of an interval separately.

Reflecting walls are supported only in `monte_carlo`, and only as thresholds.

## Foundations

**Deterministic crossing.** Let `g(x) ≥ 0` mean "on or past the boundary", and let a crossing be bracketed by the
samples `i-1` and `i`. The crossing time is

`tau = t[i-1] + (t[i] - t[i-1]) * (-g[i-1]) / (g[i] - g[i-1])`.

For a twice-differentiable trajectory the error is at most `|f''| h² / (8 |f'|)`, so it is second order in the
sampling step `h`. This is measured in the benchmark below.

The bracket `[t_{i-1}, t_i]` is always reported. It is an exact bound on τ *only if* the trajectory does not leave
and come back between two samples, which the samples themselves cannot show. That assumption is therefore always
listed as unchecked.

**Censoring.** A run observed until time `c` without a crossing tells you only that `τ > c`. Treating `c` as an event
time, or dropping the run, biases the distribution. In the benchmark, ignoring censoring is off by 0.15 in `S(t)`.

The Kaplan-Meier estimator uses each run exactly up to its own end:

`S(t) = prod over t_j <= t of (1 - d_j / n_j)`

where `d_j` is the number of events at `t_j` and `n_j` the number of runs still at risk. Its variance is Greenwood's
formula, and the band is built on the log-log scale so it stays inside [0, 1].

Kaplan-Meier requires *independent censoring*: when a run stops being observed must not depend on how close it is to
the boundary.

**Monte Carlo of an SDE.** The model is `dX = a(t, X) dt + b(t, X) dW` with diagonal noise. Each Euler-Maruyama step
is

`X[k+1] = X[k] + a h + b sqrt(h) Z`

which is weak order 1 in `h`.

Checking the boundary only at the steps misses paths that cross and come back within a step. This makes τ late by
`O(√h)`: the effective barrier moves out by about `0.5826 σ √h` (Broadie, Glasserman & Kou 1997). For scalar
thresholds and intervals, the Brownian-bridge correction adds the probability of such an excursion,

`p = exp(-2 d_k d_{k+1} / (b^2 h))`

where `d_k` and `d_{k+1}` are the distances to the barrier at the two ends of the step. The correction:

- reads `b` at the start of the step;
- places a bridge crossing at the middle of the step;
- for an interval, adds the two one-sided probabilities, which is first order.

Level-set boundaries have no bridge correction. They are checked only at the steps, and the result says so in its
diagnostics.

**When the mean is not the mean.** If some samples have not hit the boundary by the horizon `T`, the mean of τ cannot
be estimated from the data. `mean()` then returns the *restricted mean*

`integral from 0 to T of S(t) dt`

labelled as such. It is a lower bound of the true mean. In that case `censored_fraction` is a warning, and upper
quantiles may be `inf`.

## Closed forms

These functions serve as references in the tests and are available to callers. Here `a` is the distance to the
barrier and `ν` the drift towards it.

| Function | Result | Source |
|---|---|---|
| `brownian_first_passage_pdf`, `_cdf`, `_mean` | Inverse-Gaussian law of `x0 + μt + σW` reaching `b`. It is defective when the drift points away: total mass `exp(2νa/σ²)`. | Karatzas & Shreve (1991), sec. 3.5.C. The tests check it against `scipy.stats.invgauss`. |
| `brownian_exit_probability`, `brownian_exit_mean` | Exit side and mean exit time from `(l, u)`, from the scale function | Cox & Miller (1965), sec. 5.7 |
| `brownian_exit_survival` | `P(τ > t)` for an interval, by eigenfunction expansion after removing the drift | The tests check that its integral equals `brownian_exit_mean` to 1e-6 |
| `ou_mean_first_passage` | Ornstein-Uhlenbeck mean first-passage time, `T = √π/(σ√θ) ∫ erfcx(-(y-m)√θ/σ) dy`, by quadrature | Gardiner, *Stochastic Methods*, sec. 6.2. The tests compare it with an independent finite-difference solve of `D T'' + a T' = -1`. |

## Examples

**When it works.** This is Brownian motion with drift towards a barrier, where the exact answer is known:
`E[τ] = a/μ = 2`.

```python
import numpy as np
from pinneapple_analysis import dynamics as dyn

an = dyn.FirstPassageAnalyzer(dyn.BoundaryCondition.threshold(1.0, name="b"))
r = an.monte_carlo(dyn.SDE.brownian(mu=0.5, sigma=1.0), 0.0, t_max=60.0, dt=2e-3, n_paths=20_000, seed=0)
d = r.distribution
print(r.summary())                   # ok; diagnostics: MC relative s.e. 0.010, dt / median tau 0.0019
print(d.mean())                      # sample mean 2.0034, s.e. 0.0204 (truth 2)
print(d.hitting_probability(2.0))    # 0.7174, Wilson 95 % interval (0.7111, 0.7235)
print(d.prediction_interval(0.9))    # (0.211, 7.137): where the tau of a new path falls 90 % of the time
```

**When it does not answer the question asked.** Here the drift points away from the barrier. The barrier is reached
with probability `e^{-1} ≈ 0.368` over infinite time, and the mean of τ is infinite.

```python
r = an.monte_carlo(dyn.SDE.brownian(mu=-0.5, sigma=1.0), 0.0, t_max=20.0, dt=1e-2, n_paths=5000, seed=0)
print(r.estimate["b"])   # p_hit 0.376 by t = 20 (exact 0.3667), median inf, mean 13.2 = restricted mean up to 20
print(r.diagnostic("b: censored_fraction"))   # warn: 62 % of paths not hit; the mean is a lower bound
```

The number labelled "mean" in this second run is not `E[τ]`. The result says so in `mean_kind` and in the
`censored_fraction` warning.

**Observed data.** A cooling test logged to CSV with units in the header:

```python
tr = dyn.from_csv("cooling.csv", time="time")                     # "time [s]", "T [°C]" -> units kept
r = dyn.FirstPassageAnalyzer(dyn.BoundaryCondition.threshold(50.0, "down", component="T")).crossing(tr)
r.estimate, r.uncertainty            # tau, and the sampling interval that brackets it
```

## Diagnostics and failure conditions

| Diagnostic | Meaning |
|---|---|
| `censored_fraction` | Share of samples not hit by the horizon. Above 1 % it is a warning (restricted mean). At 100 % it is a failure, and the result is marked not applicable. |
| `mc_relative_se_mean` | Monte Carlo relative standard error of the mean. A warning above 2 %. |
| `dt_over_median_tau` | The step compared with the hitting time scale. Above 1 %, run `dt_convergence`. |
| `monitoring` | `Brownian bridge`, or `discrete steps only`, which is a warning: biased late by `O(√dt)`. |
| `started_on_boundary` | The trajectory or path starts on or past the boundary, so τ = t0. |
| `crossing_across_gap`, `max_bracket_width` | Missing samples, or coarse sampling, around a crossing. The interpolated time is uncertain by the bracket width. |

The method does not apply, or should not be trusted, when:

- the SDE is not an adequate model of the system (always an unchecked assumption);
- the noise is non-diagonal (not supported);
- `dt` is not small against the hitting time scale;
- most paths are censored;
- in observed data, censoring depends on the state;
- in observed data, the trajectory leaves and re-enters between samples.

## Cost

| Operation | Cost |
|---|---|
| Monte Carlo | `O(n_paths × n_steps × (d + n_boundaries))`. Paths stop once they have hit every boundary. Measured: 20 000 paths with up to 30 000 steps of `dt = 2e-3` took 4.8 s (`bm_drift_mean` row of the table below). The test suite caps 10 000 paths × 10 000 steps at 30 s. |
| Kaplan-Meier | `O(n log n)`. 10⁶ censored samples took 0.47 s on the same machine. |
| Deterministic crossing | `O(n)` per boundary |

## Benchmarks

To reproduce the table:

```bash
python -m pinneapple_analysis.dynamics.benchmarks first_passage
```

It uses `seed=0`, 20 000 paths and the full sample sizes. Each row compares one result with a reference that is
independent of the code under test.

- For a Monte Carlo row, "Pass" means `|error| ≤ 4` Monte Carlo standard errors.
- For a whole distribution, "Pass" means the error is within the 99 % Dvoretzky-Kiefer-Wolfowitz bound.

The values below were measured on 2026-10-30 (Python 3.13, NumPy 2.5, SciPy 1.18, one CPU core).

| Case | Quantity | Reference | Computed | Error | MC s.e. | z | Pass | Cost [s] |
|---|---|---|---|---|---|---|---|---|
| bm_drift_mean | E[tau] | 2 | 2.003 | 0.00337 | 0.02036 | 0.1655 | yes | 4.79 |
| bm_drift_cdf | sup_t \|F_MC - F\| | 0 | 0.004586 | 0.004586 | - | - | yes (DKW 0.0115) | 0.00 |
| bm_drift_mean_coarse_bridge | E[tau], dt = 0.02 | 2 | 2.043 | 0.04286 | 0.02086 | 2.055 | yes | 1.03 |
| bm_drift_mean_coarse_no_bridge | E[tau], dt = 0.02, discrete monitoring | 2.165 | 2.165 | 0.0002474 | 0.02097 | 0.0118 | yes | 1.02 |
| bm_interval_mean | E[exit time] | 0.3457 | 0.3481 | 0.002359 | 0.002098 | 1.125 | yes | 1.83 |
| bm_interval_p_up | P(exit through upper) | 0.542 | 0.5497 | 0.00773 | 0.003518 | 2.197 | yes | 0.00 |
| bm_interval_survival | sup_t \|S_MC - S\| | 0 | 0.00631 | 0.00631 | - | - | yes (DKW 0.0115) | 0.00 |
| ou_mfpt | E[tau] | 12.93 | 12.82 | -0.1107 | 0.1268 | 0.8731 | yes | 5.12 |
| reflect_absorb | E[tau] | 0.91 | 0.907 | -0.003023 | 0.005786 | 0.5224 | yes | 3.60 |
| crossing_order | observed order of the interpolation error | 2 | 2.001 | 0.0005045 | - | - | yes | 0.28 |
| km_censored | sup_t≤2 \|S_KM - S\| | 0 | 0.006988 | 0.006988 | - | - | yes | 0.43 |

What the rows show, and what they do not:

- **Discrete monitoring without the bridge.** At `dt = 0.02` it gives E[τ] = 2.165, which is 0.165 late. This
  matches the continuity-corrected prediction 2.165. The bridge removes most of that bias: 2.043, with z = 2.1 at
  this coarse step. The residual 0.04 sits near the edge of Monte Carlo noise and is consistent with the first-order
  placement of bridge crossings in the middle of the step.
- **Kaplan-Meier.** About 33 % of the samples are censored. The pointwise 95 % band covered the true `S(t)` at 99.5 %
  of the 201 points checked. The estimate that ignores censoring is off by 0.151.
- **Interpolation.** The worst-case error over 16 grid offsets falls from 1.6e-3 to 2.5e-5 as `h` goes from 0.1 to
  0.0125.

These benchmarks cover Brownian motion and Ornstein-Uhlenbeck, the processes with exact answers. Agreement there does
not establish accuracy for strongly non-linear drifts, multiplicative noise near zero, or boundaries crossed on time
scales comparable to `dt`. For those cases, measure with `dt_convergence` and `sensitivity`.
