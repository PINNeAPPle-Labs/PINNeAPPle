# Raissi, Perdikaris & Karniadakis, physics-informed neural networks (continuous time): frozen targets

Written and committed **before any run**. Every number is quoted from the PDF named below, with its table or
figure and page.

## Source

- M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics Informed Deep Learning (Part I): Data-driven Solutions of
  Nonlinear Partial Differential Equations", arXiv:1711.10561v1 (28 Nov 2017), 22 pages.
  - Downloaded from `https://arxiv.org/pdf/1711.10561` on 2026-10-30.
  - Its continuous-time examples are the ones that appear in J. Comput. Phys. 378 (2019) 686-707 (§3.1 there).
    We quote the arXiv text because it is openly available.
- **Official code:** `https://github.com/maziarraissi/PINNs` at commit `932f50a2d8ef4e80d1456bbae6887a73ff5166ef`.
  - `appendix/continuous_time_inference (Burgers)/Burgers.py` and `Burgers_systematic.py`.
  - `main/continuous_time_inference (Schrodinger)/Schrodinger.py`.
  - It is read for details the paper leaves out. Its code is not copied or run, and its data are not
    redistributed.

## Method as the paper states it (§2, p. 3-11)

- **Network:** an MLP with tanh activations for u(t, x). For Schrödinger it has 2 outputs, the real and imaginary
  parts of h.
- **Loss:** MSE = MSE_u + MSE_f for Burgers (eq. 4), and MSE_0 + MSE_b + MSE_f for Schrödinger (eq. 6). The terms
  are unweighted means.
- **Optimizer:** "we chose to optimize all loss functions using L-BFGS" (p. 6).
- **Collocation points:** Latin hypercube sampling.
- **Error:** the relative L2 error against the exact or high-resolution solution.

## Targets

### Burgers, continuous time (§2.1, p. 5-9)

- **Problem.** u_t + u u_x − (0.01/π) u_xx = 0 on x ∈ [−1, 1], t ∈ [0, 1], with u(0, x) = −sin(πx) and
  u(t, ±1) = 0. The exact solution is "analytically available".
- **B0 (headline, p. 7 and Fig. 1).**
  - Settings: N_u = 100 random initial and boundary points; 10 000 Latin-hypercube collocation points; a "9-layer"
    network with 20 neurons per hidden layer, tanh, and "all 3021 parameters".
  - Target: **relative L2 error 6.7·10⁻⁴**.
- **Table 1 (p. 9).** Relative L2 error against N_u (rows) and N_f (columns), with a 9-layer, 20-neuron network:

  | N_u \ N_f | 2000 | 4000 | 6000 | 7000 | 8000 | 10000 |
  |---|---|---|---|---|---|---|
  | 20 | 2.9e-01 | 4.4e-01 | 8.9e-01 | 1.2e+00 | 9.9e-02 | 4.2e-02 |
  | 40 | 6.5e-02 | 1.1e-02 | 5.0e-01 | 9.6e-03 | 4.6e-01 | 7.5e-02 |
  | 60 | 3.6e-01 | 1.2e-02 | 1.7e-01 | 5.9e-03 | 1.9e-03 | 8.2e-03 |
  | 80 | 5.5e-03 | 1.0e-03 | 3.2e-03 | 7.8e-03 | 4.9e-02 | 4.5e-03 |
  | 100 | 6.6e-02 | 2.7e-01 | 7.2e-03 | 6.8e-04 | 2.2e-03 | 6.7e-04 |
  | 200 | 1.5e-01 | 2.3e-03 | 8.2e-04 | 8.9e-04 | 6.1e-04 | 4.9e-04 |

- **Table 2 (p. 9).** Relative L2 error against the number of hidden layers (rows) and neurons per layer
  (columns), with N_u = 100 and N_f = 10 000:

  | Layers \ Neurons | 10 | 20 | 40 |
  |---|---|---|---|
  | 2 | 7.4e-02 | 5.3e-02 | 1.0e-01 |
  | 4 | 3.0e-03 | 9.4e-04 | 6.4e-04 |
  | 6 | 9.6e-03 | 1.3e-03 | 6.1e-04 |
  | 8 | 2.5e-03 | 9.6e-04 | 5.6e-04 |

- **Text claims** (p. 7-9). These are reported with each table, not scored:
  - "increased prediction accuracy as the total number of training data N_u is increased, given a sufficient
    number of collocation points N_f";
  - "as the number of layers and neurons is increased ... the predictive accuracy is increased".

### Schrödinger, continuous time (§2.2, p. 9-11)

- **Problem.** i h_t + 0.5 h_xx + |h|² h = 0 on x ∈ [−5, 5], t ∈ [0, π/2], with h(0, x) = 2 sech(x) and periodic
  h and h_x.
- **Reference.** Chebfun, 256 Fourier modes, explicit RK4 with ∆t = π/2·10⁻⁶.
- **S0 (headline, p. 11).**
  - Settings: N_0 = 50 initial points, N_b = 50 boundary times, N_f = 20 000 LHS points; a "5-layer deep neural
    network with 100 neurons per layer", tanh.
  - Target: **relative L2 error 1.97·10⁻³**, of |h| (Fig. 2 shows |h|).

### Not attempted in this phase

- The discrete-time Burgers model: 8.2·10⁻⁴ (Fig. 3), Table 3 (9 cells) and Table 4 (36 cells).
- The discrete-time Allen-Cahn model: 6.99·10⁻³.

They can be added by an amendment to this file, written before they run.

**Scored targets:** B0 (1) + Table 1 (36) + Table 2 (12) + S0 (1) = **50**.

## Ambiguities and our fixed choices

1. **"9-layer" network.** This means 8 hidden layers of 20 plus the output layer: 2·20 + 20 + 7·(20·20 + 20) + 21
   = 3021 parameters, as the text says. The official `Burgers.py` uses `layers = [2] + 8·[20] + [1]`.
2. **Table 1 network and Table 2 data.**
   - The official `Burgers_systematic.py` runs Table 1 with 8 hidden layers of **40** neurons, and Table 2 with
     **N_u = 200**. Both differ from the paper's captions.
   - **Primary: the captions** (8 × 20 for Table 1; N_u = 100 and N_f = 10 000 for Table 2).
   - The code's settings are run as a labelled secondary variant at seed 0, if time allows.
3. **Optimizer.** Both scripts use `scipy.optimize` L-BFGS-B with maxiter 50 000, maxfun 50 000, maxcor 50, maxls
   50 and ftol = machine epsilon. We use the same, on a float32 PyTorch model, with the gradient in float64 for
   scipy.
4. **Schrödinger training.**
   - `Schrodinger.py` runs **50 000 Adam steps** (TensorFlow defaults, lr 1e-3) before L-BFGS-B. The text mentions
     L-BFGS only.
   - We follow the code (Adam, then L-BFGS-B), since it produced the figure. L-BFGS-B alone is run as a labelled
     secondary variant at seed 0.
   - The error scored is that of |h|, `error_h` in the script.
5. **Init and input scaling** (both scripts).
   - Weights are drawn from a normal with std sqrt(2 / (fan_in + fan_out)), truncated at 2 std, and the biases are
     zero.
   - Inputs are mapped to [−1, 1] by the domain bounds.
6. **Data and collocation** (both scripts).
   - **Burgers:**
     - The candidate data points are the 256 grid points at t = 0 and the 100 times at each wall; N_u of them
       are drawn without replacement.
     - The collocation points are N_f LHS points, plus the chosen data points.
     - The test grid is 256 uniform x on [−1, 1] by t = 0, 0.01, ..., 0.99.
   - **Schrödinger:**
     - N_0 points are drawn from the reference's x grid at t = 0 (256 points on [−5, 5), step 10/256, endpoint
       excluded), N_b times from its t grid (201 points on [0, π/2]), and N_f LHS points.
     - The test grid is that 256 × 201 grid.
   - LHS uses `scipy.stats.qmc.LatinHypercube` (random within strata, like pyDOE's default).
7. **References.**
   - **Burgers:** `pinneapple_physics.closed_form.burgers_sine_exact` (Cole-Hopf with Gauss-Hermite, 120 nodes).
     Before any training, it was compared with the official `burgers_shock.mat` on that file's 256 × 100 grid:
     the maximum difference is 4.2e-11.
   - **Schrödinger:** our own Fourier pseudo-spectral solver, with 256 modes and a fine RK4 or exponential
     integrator. Before any training, it is to be compared with the official `NLS.mat` (a check, not a result).
     The comparison is recorded in the README.
8. **Seeds.** The paper reports one run per number, with one seed (1234) set for each whole sweep.
   - B0: **10 seeds**.
   - Every Table 1 and Table 2 cell: **3 seeds**.
   - S0: **3 seeds**.
   - The score is the median over seeds, and min-max is reported.

## Status rule (`pinneapple_analysis.reproduction.status`, PR #891)

- **Reproduced:** the median over seeds is within a factor of 2 of the paper's number.
- **No ordering claim is scored.** The two text claims above are reported with each table: whether the error
  falls with N_u at N_f = 10 000, and with depth and width.
- **Expected spread.** Table 1's own cells move by more than 10× between neighbours (for example N_u = 20,
  N_f = 7000: 1.2; then N_f = 8000: 9.9e-2). A single seed of the paper can fall far from our median, so our
  min-max is reported next to every cell.

## Baseline at equal budget

- The same network trained without the physics term (N_f = 0, the paper's footnote 1 on p. 8).
- Optionally, a classical solution of the same problem with PINNeAPPle's own solvers.
- Both are reported, not scored.

## Amendment 1 (2026-10-31, after B0 and Table 2 ran in float32): scope and post-hoc variants

**Post hoc, not the frozen protocol.** The scored results stay those of the float32 runs above.

1. **float64 diagnostic.** B0 in float32 ran 10 seeds; the median was 6.1e-3 against 6.7e-4.
   - In every run, L-BFGS-B stopped on its relative-reduction test after about 5 000 iterations.
   - B0 therefore also ran in float64 (seeds 0-2), and Table 2 in float64 (seed 0).
   - These runs are reported as labelled variants, with a "(post hoc)" status. They are not counted in the score.
2. **Not attempted, because of compute.** One CPU (4 cores) is shared with another reproduction.
   - **Table 1:** 108 runs at 3 seeds per cell, about 10 min each on one thread, so about 18 h.
   - **S0:** 50 000 Adam steps plus L-BFGS-B, about 14 h per seed on one thread (measured from a 20-step smoke
     run).
   - Both are reported as "not attempted", with these costs.
   - The extras (code-setting variants, L-BFGS-only Schrödinger, no-physics baseline) are likewise not run.
