# Critical slowing down and early-warning indicators

`pinneapple_analysis.dynamics.early_warning` computes, from a time series, the statistical signatures of a system
losing stability. It reports how strongly those signatures trend, and checks the trend against a null model and
against its own sensitivity to the analysis choices.

The series can come from measurements, from a simulation, or from a trained model's prediction (see
[the shared foundation](dynamics_toolkit.md)). Results follow the toolkit's contract.

## An indicator is not a prediction

Near a local bifurcation the dominant eigenvalue λ of the linearised dynamics goes to zero, and perturbations decay
more slowly. For `dx = λx dt + σ dW` with λ < 0:

| Quantity | Formula |
|---|---|
| variance | `σ²/(2|λ|)` |
| lag-Δt autocorrelation | `e^{λΔt}` |

Both rise as λ → 0⁻ (Scheffer et al. 2009; Dakos et al. 2012). A significant rise is **consistent with** this, but it
does not show that a transition will happen, nor when.

Several things produce the same rise without any loss of stability:

- a trend left in the data;
- noise whose amplitude grows;
- a change of sampling;
- slow forcing far from a bifurcation.

The benchmarks below measure the first two. Conversely, many transitions give no warning at all: noise-induced,
rate-induced and global ones. Every result states this in `limits`, and `TransitionDiagnostics` keeps the indicator
and the prediction apart: `is_prediction` is always `False`.

## What it does

| Step | Call | Notes |
|---|---|---|
| Detrend | `detrend(t, x, "loess" \| "gaussian" \| "linear" \| "none", bandwidth)` | `loess` (local linear) is the default: it has no first-order edge bias. The Gaussian kernel of Dakos et al. is biased near the ends by about slope × bandwidth. On the saddle-node benchmark, where the equilibrium moves, that bias swamped the fluctuations: residual variance 9e-5 against a theoretical 2.6e-6. |
| Rolling statistics | `RollingStatistics(window).compute(t, x, ("variance", "std", "cv", "skewness", "kurtosis"))` | `window` is a fraction of the record (or `by="time"`). Missing values are skipped, and a window with fewer than `min_count` values gives NaN. |
| Autocorrelation | `AutocorrelationAnalyzer(window, lag=1)` | On an irregular grid, use `lag_time=` and `tol=`: the estimate is built from pairs of samples whose spacing is within the tolerance of the lag. |
| Recovery rate | `RecoveryRateEstimator.from_ar1(ac, dt)` gives `ln φ / dt`; `.from_pulse(t, x)` fits an exponential relaxation | The AR(1) rate approaches zero from below near a bifurcation. |
| Spectral indicators | indicators `low_frequency_power` and `spectral_exponent` | Welch periodogram per window (spectral reddening). They need a regular grid. |
| All of it, with significance | `EarlyWarningAnalyzer(...).analyze(trajectory)` | See below. |
| Variance, AC1 and recovery rate together | `CriticalSlowingDown(...)` | The critical-slowing-down signature. |
| Robustness | `.sensitivity(windows=..., bandwidths=...)` and `.sampling_sensitivity(factors=...)` | Kendall τ over a grid of choices, and over coarser sampling. |
| Verdict | `TransitionDiagnostics(result, sensitivity).report()` | One of `no signal`, `indicator signal (not a prediction)` or `inconclusive`, with the doubts listed. |

`EarlyWarningAnalyzer.analyze` returns, for each indicator:

- its values per window and their Kendall τ against time;
- a **surrogate p-value**: the share of stationary surrogates whose τ is at least the observed one.

Kendall's own p-value assumes independent points, and overlapping windows violate that, so it is kept only in the
`StabilityIndicator` and is not used for the verdict.

The surrogates keep the autocorrelation of the residuals and remove any trend:

- an AR(1) fitted to the residuals (the default);
- phase-randomised or IAAFT surrogates from `pinneapple_systems.time_series.predictability`.

The analysis also checks the residuals. It reports:

- the missing fraction and the regularity of the grid;
- a Kendall trend left in the residuals, which means the detrending was incomplete;
- a warning when no detrending was done;
- the ADF and KPSS p-values, from `pinneapple_systems.time_series.audit.tests`, when statsmodels is installed.

### Benchmark series

`ramp_series(kind)` simulates stochastic normal forms by Euler–Maruyama. In every one the decay rate `r` of
perturbations goes linearly from 1 to `end` (> 0), so the system approaches the bifurcation but does not cross it:

| `kind` | Equation |
|---|---|
| `saddle_node` | `dx = (μ − x²)dt` |
| `transcritical` | `dx = (−r x − x²)dt` |
| `pitchfork` | `dx = (−r x − x³)dt` |
| `ou_decreasing` | `dx = −r x dt` |

Three negative controls keep the stability constant:

- `stable`;
- `stable_trend`: an accelerating trend in the mean;
- `stable_growing_noise`: the noise amplitude multiplied by 3 over the record.

The saddle-node and transcritical basins shrink with `r`. A path that leaves the basin is reported through
`tipped_at`, and the rest of the series is NaN.

## Examples

**When it works.** The OU process with decreasing stability:

```python
from pinneapple_analysis.dynamics.early_warning import EarlyWarningAnalyzer, TransitionDiagnostics, ramp_series

s = ramp_series("ou_decreasing", seed=1, end=0.1)
an = EarlyWarningAnalyzer(n_surrogates=99)
r = an.analyze((s["t"], s["x"]))
print(r.estimate["variance"]["tau"], r.estimate["ac1"]["tau"])    # 0.89 and 0.88, both p = 0.01
print(TransitionDiagnostics(r, an.sensitivity((s["t"], s["x"]))).report()["verdict"])
# 'indicator signal (not a prediction)'
```

**When it misleads.** In both cases below the stability never changes:

```python
s = ramp_series("stable_trend", seed=2)                          # accelerating trend in the mean only
EarlyWarningAnalyzer(detrend="none").analyze((s["t"], s["x"]))   # AC1 rises "significantly"; flagged no_detrending
EarlyWarningAnalyzer(detrend="loess").analyze((s["t"], s["x"]))  # the rise disappears

s = ramp_series("stable_growing_noise", seed=3)                  # noise amplitude x3
EarlyWarningAnalyzer().analyze((s["t"], s["x"]))                 # variance rises significantly, AC1 does not
```

This is why variance alone is not evidence of slowing down, and why the verdict requires the indicators that a
linearisation predicts to agree.

## Assumptions, failure conditions, limits

**Unchecked assumptions.** The data cannot test these, and each result lists them:

- the transition, if any, is a local bifurcation reached slowly;
- the control parameter changes slowly compared with the recovery time;
- the noise amplitude and the sampling are stationary;
- the detrending removes the slow trajectory and keeps the fluctuations.

**The analysis is not applicable when:**

- there are fewer than `3 × min_count` values;
- the grid is irregular and no `lag_time` is given for the autocorrelation;
- spectral indicators are requested on an irregular grid.

**Known weaknesses, measured in the benchmarks:**

- White measurement noise dilutes the autocorrelation and adds constant variance.
- Coarse sampling leaves too few points per window.
- The p-value has a floor of `1/(n_surrogates + 1)`.

**Cost.**
- On a regular grid without missing values, the rolling statistics are vectorised: a 1000-point series with 99
  surrogates and two indicators took 1.2 s here, LOESS detrending included.
- With missing values or an irregular grid, the statistics loop over windows.
- LOESS detrending is `O(n²)`.

## Benchmarks

To reproduce the table:

```bash
python -m pinneapple_analysis.dynamics.benchmarks early_warning
```

It uses `seed=0`, 40 replicate series per row, 99 AR(1) surrogates and α = 0.05. "Detection" means that variance
and lag-1 autocorrelation **both** rise significantly.

| Case | What is measured | Reference | Measured | Pass |
|---|---|---|---|---|
| ou_ac1 | mean rolling lag-1 autocorrelation, stationary OU (40 series of 2000 points) | e^(−θΔt) = 0.7787 | 0.7739 ± 0.0022 (z = 2.2) | yes |
| ou_variance | mean rolling variance, same series | σ²/(2θ) = 0.01 | 0.009952 ± 0.0001 (z = 0.47) | yes |
| detect_saddle_node | detection rate, ramp of r from 1 to 0.1 | ≥ 0.8 | 0.975 | yes |
| detect_transcritical | detection rate, ramp of r from 1 to 0.1 | ≥ 0.8 | 0.975 | yes |
| detect_pitchfork | detection rate, ramp of r from 1 to 0.1 | ≥ 0.8 | 0.975 | yes |
| detect_ou_decreasing | detection rate, ramp of r from 1 to 0.1 | ≥ 0.8 | 0.975 | yes |
| fp_stable | false-positive rate, constant stability | α = 0.05 | 0 | yes |
| fp_trend_raw | false-positive rate, accelerating trend, no detrending | — | 0.65 | demonstration |
| fp_trend_loess | the same, detrended with LOESS | α = 0.05 | 0 | yes |
| fp_growing_noise_variance | variance alone, noise ×3, constant stability | — | 1 | demonstration |
| fp_growing_noise_ac1 | AC1 alone, the same series | α = 0.05 | 0.025 | yes |
| detect_ou_obs_noise_1 | detection, white measurement noise of 1× the initial std | — | 0.8 | demonstration |
| detect_ou_obs_noise_3 | detection, measurement noise of 3× the initial std | — | 0.2 | demonstration |
| detect_ou_subsampled_4 | detection, every 4th sample kept (250 points) | — | 0.625 | demonstration |
| detect_ou_subsampled_16 | detection, every 16th sample kept (62 points) | — | 0.05 | demonstration |

The values above were measured on 2026-10-31. The `ou_ac1` estimate sits slightly below the theory (z = 2.2). This is consistent with the known small-sample bias of the lag-1 estimator, about (1 + 4φ)/n_window. The false-positive rate of the constant-stability control was 0 of 40. The AR(1)-surrogate test is conservative here: it rejects less often than α.

The rows marked "demonstration" have no pass/fail reference: they measure a known failure mode.

What the benchmarks do not cover:

- These are normal forms with additive noise and a linear ramp, so detection here is easier than in field data.
- The rates depend on the record length (1000 points), the noise, and how close the ramp ends to the bifurcation
  (`r = 0.1`).
- They do not measure **how early** a signal appears before a transition, and the toolkit makes no claim about lead
  time.
