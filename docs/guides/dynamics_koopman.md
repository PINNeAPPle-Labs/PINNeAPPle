# Koopman operator and spectral analysis

`pinneapple_analysis.dynamics.koopman` approximates the Koopman operator of a dynamical system from snapshot data.
It returns the eigenvalues, eigenfunctions and modes of that approximation, and attaches a residual to each eigenpair
that says whether it can be trusted.

The data can be measured trajectories, simulation output, PDE snapshots, or the prediction of a trained network such
as an FNO or DeepONet rollout. They are loaded with the adapters of [the shared foundation](dynamics_toolkit.md)
(`from_array`, `from_snapshots`, `TrajectoryEnsemble`).

## Foundations

The Koopman operator of a map `F` acts on observables `g`:

```text
(K g)(x) = g(F(x))
```

It is linear even when `F` is not. This means a nonlinear system has a spectral decomposition, but in a space of
observables, which is generally infinite-dimensional.

The data-driven methods work on a finite dictionary `ψ = (ψ₁, …, ψ_m)` evaluated on snapshot pairs `(x_n, y_n = F(x_n))`.

### EDMD

EDMD (Williams, Kevrekidis & Rowley 2015) builds the operator from three matrices:

| Matrix | Formula |
|---|---|
| Gram matrix | `G = Ψ_Xᵀ Ψ_X / N` |
| Cross matrix | `A = Ψ_Xᵀ Ψ_Y / N` |
| Operator | `K = G⁺ A` |

The operator satisfies `ψ(F(x)) ≈ ψ(x) K`. An eigenpair `K ξ = λ ξ` gives the eigenfunction `φ(x) = ψ(x) ξ`. That
eigenfunction is exact when the dictionary spans an invariant subspace, and approximate otherwise.

### DMD

DMD is EDMD with the state itself as the dictionary. It is computed through a rank-`r` SVD of the snapshot matrix, so
a 10⁴-point PDE snapshot costs `O(D N r)` rather than `O(D²)`. The modes are the exact DMD modes of Tu et al. (2014).

The rank `r` can be chosen in three ways:

- `"gavish-donoho"` (the default): the optimal hard threshold for an unknown noise level (Gavish & Donoho 2014);
- a fraction of the singular-value energy, for example `0.99`;
- a fixed integer.

`tls=True` applies total-least-squares DMD (Hemati et al. 2017). This removes the bias that noise in the data puts on
the eigenvalues, which otherwise shifts them towards damping.

### Residuals (ResDMD)

For each eigenpair, the residual of Colbrook & Townsend (2024) measures how far `φ` is from a true eigenfunction:

```text
res(λ, ξ)² = ξ* (L − λ A* − λ̄ A + |λ|² G) ξ / ξ* G ξ        with L = Ψ_Yᵀ Ψ_Y / N
```

The residual is zero for an exact eigenfunction. It is large for a *spurious* eigenvalue, one that comes only from
truncating the dictionary. A spectrum should not be read without its residuals.

### Continuous time

The continuous-time eigenvalue is `ω = log λ / Δt`, on the principal branch. Frequencies above `π/Δt` alias, and the
DMD result warns when eigenvalues sit near that limit.

`EDMD.fit_generator(X, dX/dt)` fits the generator directly from derivative data (gEDMD, Klus et al. 2020). It
differentiates the dictionary along the velocity by central differences.

## What it does

| Task | Call |
|---|---|
| Fit linear dynamics or modal content of snapshots | `DMD(rank="gavish-donoho", tls=False).fit(data)` |
| Fit nonlinear dynamics | `EDMD(ObservableDictionary.monomials(d, 3), ridge=0).fit(data)` |
| Choose a dictionary | `ObservableDictionary`: `identity`, `monomials`, `rbf`, `fourier` or `custom`. You can combine dictionaries with `+`. `neural(model, n_out, d=d)` uses a trained PyTorch module as fixed observables, for example the encoder of `KoopmanAutoencoder`. |
| Fit from derivatives | `EDMD(...).fit_generator(X, dXdt)` |
| Inspect the spectrum | `result.details["operator"]`, a `KoopmanOperator` with `.eigenvalues`, `.eigenfunctions`, `.modes(X)`, `.residuals()`, `.stability()`, `.predict(x0, steps)` and `.generator()` |
| Get a mode table | `SpectralAnalyzer.spectrum(result, residual_tol=...)`: frequency, growth rate, amplitude, residual, and a `trusted` flag |
| Measure errors | `KoopmanReconstruction.evaluate(operator, trajectories, horizon)` for one-step and open-loop multi-step errors |
| Check robustness | `SpectralAnalyzer.noise_sensitivity(...)` and `SpectralAnalyzer.select_order(...)` (held-out error by rank or dictionary) |
| Compare with a nonlinear model | `SpectralAnalyzer.compare_with_state_regression(dictionary, train, test)` |

The last row compares Koopman (linear in the lifted space) against a nonlinear state-space regression,
`x_{n+1} = ψ(x_n) W`, which uses the same dictionary and the same data.

### Name

This `KoopmanOperator` is a fitted, finite-dimensional NumPy operator, with no training involved. It is **not**
`pinneapple_neural.architectures.rom.koopman.KoopmanOperator`, the learnable linear layer inside the deep Koopman
autoencoder. That class is unchanged. The two meet through `ObservableDictionary.neural`, which lets a trained
encoder serve as the dictionary here.

### Role

DMD and EDMD are trajectory operators: they take data and return a fitted operator.

Nothing here is an `nn.Module`. The only differentiable piece is a neural dictionary, and it is used frozen.

## Examples

**When it works.** The system `x1' = μ x1`, `x2' = λ (x2 − x1²)` has the exact invariant subspace `{x1, x2, x1²}`:

```python
import numpy as np
from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics.koopman import EDMD, ObservableDictionary

mu, lam = -0.05, -1.0
b = lam / (lam - 2 * mu)
flow = lambda x1, x2, t: (x1 * np.exp(mu * t), (x2 - b * x1**2) * np.exp(lam * t) + b * x1**2 * np.exp(2 * mu * t))
t = np.arange(30) * 0.1
rng = np.random.default_rng(0)
ens = dyn.TrajectoryEnsemble([dyn.from_array(t, np.column_stack(flow(*rng.uniform(-1, 1, 2), t))) for _ in range(10)])

d = ObservableDictionary.monomials(2, 2, constant=False, only=[(1, 0), (0, 1), (2, 0)])
r = EDMD(d).fit(ens)
print(r.details["operator"].eigenvalues.continuous)   # -0.05, -0.1, -1.0 (error 2.5e-14)
print(r.uncertainty["residuals"])                     # 5e-8, 0, 0
```

**When it warns you.** Add the monomials `1`, `x1 x2` and `x2²`. The last two do not close under the dynamics.
EDMD then returns two extra eigenvalues, `−1.08` and `−2.42`, which are not Koopman eigenvalues. Their residuals
(0.028 and 0.034) flag them, while the four true eigenpairs keep residuals below 1e-6.

**When a linear lifting is the wrong tool.** The Van der Pol oscillator has a limit cycle, and no finite polynomial
dictionary is invariant for it. Here is the 100-step forecast error from 15 training and 5 test trajectories:

| Dictionary | Koopman (EDMD) | Nonlinear state regression |
|---|---|---|
| monomials, degree 3 | 0.507 | 0.080 |
| monomials, degree 5 | 0.477 | 0.00078 |

On this system the lifted linear propagation decays towards the fixed point and misses the cycle. The nonlinear
regression on the same dictionary does far better.

## Assumptions, failures, limits

**Unchecked assumptions.** The data cannot test these, and each result lists them:

- the dynamics are autonomous;
- the dictionary (nearly) spans an invariant subspace; the residuals measure this;
- the sampling resolves the fastest relevant frequency and covers the region where the result will be used.

**Known failure modes:**

| Situation | What goes wrong | How it shows |
|---|---|---|
| DMD on nonlinear dynamics | the spectrum mixes harmonics | — |
| Noisy data | eigenvalues are biased towards damping | `tls=True` removes most of the bias (benchmark below) |
| Dictionary larger than the data | the Gram matrix is ill-conditioned | `gram_condition` and `n_pairs` diagnostics |
| Limit cycles and chaos | eigenvalues are spurious | large residuals |
| Prediction outside the sampled region | the error is unbounded | — |

**Irregular sampling.** It is refused, because the operator needs a fixed time step. `Trajectory.resample` marks
interpolated data if you resample first.

**Cost:**
- DMD costs one thin SVD, `O(D N min(D, N))`.
- EDMD costs `O(N m²)` to assemble and `O(m³)` to solve.
- ResDMD residuals cost `O(m²)` per eigenpair.

## Benchmarks

To reproduce the table:

```bash
python -m pinneapple_analysis.dynamics.benchmarks koopman
```

It uses `seed=0`. The values were measured on 2026-10-31.

| Case | Reference | Measured | Pass |
|---|---|---|---|
| linear_map | eig(A) | max error 3.4e-16 | yes |
| oscillator | ±iω (ω = 2, Δt = 0.1) | 7.4e-15 | yes |
| nonlinear_subspace | {μ, λ, 2μ} with dictionary {x1, x2, x1²} | 2.5e-14 (max residual 5.1e-8) | yes |
| nonlinear_residual_true | residual 0 for the 4 true eigenpairs (all monomials, degree ≤ 2) | max 5.1e-8 | yes |
| nonlinear_residual_spurious | — | min residual of the 2 spurious eigenpairs 0.028 (eigenvalues −1.08, −2.42) | demonstration |
| generator_edmd | {μ, λ, 2μ} from derivatives | 9.5e-13 | yes |
| heat_modes | −νk², k = 1..3, from 101-point PDE snapshots | 1.7e-13; Gavish–Donoho rank 3 | yes |
| gavish_donoho_rank | 4 (rank-4 signal plus noise, 100 × 199) | 4 | yes |
| torch_dmd_crosscheck | the existing `rom.DynamicModeDecomposition` | max difference 1.2e-15 | yes |
| noise_bias_dmd | eig(A), 20 % white noise, 50 replicates | mean max error 0.0379 ± 0.0028 | demonstration |
| noise_bias_tls | the same with TLS-DMD | 0.00226 ± 0.0011 | demonstration |
| koopman_vs_state_regression | 0 (invariant subspace, 25 steps) | Koopman 1.1e-14, regression 8.5e-15 | yes |
| van_der_pol_deg3 / _deg5 | — (limit cycle, 100 steps) | Koopman 0.507 / 0.477; regression 0.080 / 0.00078 | demonstration |

The exact cases show that the algebra is correct, and that the residuals separate true eigenpairs from spurious ones.
They do not show that a given dictionary is adequate for a new system. That is what the residuals and the held-out
errors are for, case by case.
