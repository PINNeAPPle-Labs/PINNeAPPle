# Gated X-TFC (Dwivedi, Schiassi, Sigovan, Sixou): frozen targets

Written and committed **before any run**. Every number is quoted from the PDF below, with its page. Our settings
and the scoring rule are fixed here.

## Source and access

- V. Dwivedi, E. Schiassi, M. Sigovan, B. Sixou, "Gated X-TFC: Soft Domain Decomposition for Forward and Inverse
  Problems in Sharp-Gradient PDEs", arXiv:2510.01039v1 (1 Oct 2025), 23 pages. Downloaded from
  `https://arxiv.org/pdf/2510.01039` on 2026-10-25.
- No code is linked. The text mentions MATLAB R2022b and `fminbnd`.
- **Why this paper.** The owner's list names "De Florio et al. (sharp-gradient BVPs / X-TFC follow-ups)":
  - The De Florio sharp-gradient paper found is "Physics-Informed Neural Networks for 2nd order ODEs with sharp
    gradients", J. Comput. Appl. Math. 436 (2024) 115396 (from api.crossref.org).
  - It is **not on arXiv**, and the publisher and doi.org are refused by this session's network. It is therefore
    "not attempted: paper not accessible".
  - This paper is the Schiassi-coauthored X-TFC follow-up for sharp-gradient BVPs that *is* accessible.
- The text references an "Appendix A" (for ε_phy), but the PDF contains only Appendix B. The values used below
  are the ones the main text states.

## Method as the paper states it (§3.2, p. 3-5; §4.1, p. 7-10)

- **Problem.** Steady convection-diffusion u' − ν u'' = 0 on (0, 1), with u(0) = 0 and u(1) = 1. Exact solution
  u = (e^{x/ν} − 1)/(e^{1/ν} − 1) (eq. 2, p. 3).
- **Trial function.** u = g + Σ c_i ψ_i, with g = x. The features are ψ_i(x) = ϕ(z_i(x)) − (1 − x) ϕ(z_i(0))
  − x ϕ(z_i(1)), where ϕ(z) = exp(−z²) and z_i = m_i x + b_i. The parameters are m_i = 1/(√2 σ_{x,i}) and
  b_i = −m_i α_i.
- **Layout.** A split x_s defines two blocks. Each block gets N_c collocation points and N⋆ centers on an
  "endpoint grid": x_k = x_L + (k − 1)/N_c (x_s − x_L) for k = 1..N_c, and likewise on [x_s, 1].
- **Widths.**
  - The nominal widths are σ_L = k d_L and σ_R = k d_R, with d_L = x_s/N⋆, d_R = (1 − x_s)/N⋆ and k = 1.5.
  - A logistic gate blends them: s_i = 1/(1 + exp(−(α_i − x_s)/ε_b)) and σ_{x,i} = (1 − s_i) σ_L + s_i σ_R.
  - The gate scale is ε_b = max(ε_scale · min(d_L, d_R), 5ν) (p. 10).
- **Inner solve.** Minimise Σ w_k r_k² + λ‖c‖², with W = (1/N_f) I. This is a linear least-squares problem.
- **Outer search.** Nested `fminbnd`:
  - outer over x_s ∈ [0.80, 0.999] with TolX 1e-4;
  - inner over ε_scale ∈ [10, 100] with TolX 1e-1;
  - objective: the residual energy on an independent validation grid of N_v = 800 points.
- **Test.** N_test = 2 × 10^4 points. Fig. 1 (p. 7) shows the validation and test points uniform on [0, 1].
- **Forward configuration:** N_c = N⋆ = 1000 per block, so 2000 points and 2000 centers (p. 7).

## Targets

### F1-F3. Forward, Fig. 3 (p. 9)

The panel titles, quoted:

- ν = 0.01: "x⋆s = 0.837, ε⋆scale = 99.93, max |e| = 4.95e−04"
- ν = 0.001: "x⋆s = 0.933, ε⋆scale = 44.43, max |e| = 1.40e−04"
- ν = 0.0001: "x⋆s = 0.956, ε⋆scale = 52.76, max |e| = 5.86e−04"

**Targets: max abs test error 4.95e-4, 1.40e-4, 5.86e-4.** The selected x⋆_s and ε⋆_scale are reported next to
ours, but they are not scored.

### F4. Table 1 (p. 10), ν = 1e-4

| Method | Training points | Neurons | Maximum error | Training time (s) |
|---|---|---|---|---|
| X-TFC (ungated) | 10,000 | 10,000 | O(10−3) | 153 |
| Gated X-TFC | 2,000 | 2,000 | O(10−4) | 52 |

- **Target (ordering):** at ν = 1e-4, Gated X-TFC with 2000/2000 has a lower max error than ungated X-TFC with
  10000/10000, and the ungated max error is O(1e-3).
- The ungated configuration is not specified. Ours: the same RBF X-TFC with x_s fixed at 1/2 (the paper: "When
  x_s = 1/2 the layout is uniform") and k = 1.5, 5000 points and 5000 centers per block, no gate search.
- Times depend on hardware (theirs is an i7-12700H). We report ours, but they are not scored.

### I1-I4. Inverse, Table 2 (p. 12)

Quoted rows: "νtrue ν̂ % error Time (s)":

| νtrue | ν̂ | % error | Time (s) |
|---|---|---|---|
| 0.1 | 0.0897 | 10.3% | 12.0 |
| 0.01 | 0.00951 | 4.9% | 12.0 |
| 0.005 | 0.00537 | 7.4% | 11.5 |
| 0.001 | 0.00117 | 17.0% | 14.0 |

Fig. 4d (p. 11) prints ν̂ = 0.00118 for the last case, against 0.00117 in the table.

**Setting (p. 12):**
- N_data = 50 points, with noise standard deviation 1e-2, on a "right-clustered grid (exponent p = 3)".
- N_c = 400 collocation points and N⋆ = 300 centers per block, k = 1.5, ε_phy = 5ν.
- β_data = 1/σ² and β_pde = 1e2.
- The coefficient prior precision η is set by type-II maximum likelihood in [1e-12, 1e-2], starting at 1e-7. The
  fixed-point update is γ = Σ κ²/(κ² + η), η ← γ/‖m_N‖² (p. 6).
- Bayesian optimisation ("expected-improvement-plus") with 30 evaluations over log10 ν ∈ [−3, 1],
  x_s ∈ [0.85, 0.995] and ε_scale ∈ [10, 100], maximising the log-evidence of eq. on p. 6.

**Targets: % error in ν̂ of 10.3, 4.9, 7.4 and 17.0.**

### T1. Twin boundary layer, Fig. 8 (p. 15), ν = 1e-4

- Setting: 2(2x − 1) u_x − ν u_xx + 4u = 0 on [0, 1], u(0) = u(1) = 1. Exact u = e^{−2x(1−x)/ν}.
- Run settings: "two split locations", "1200 RBFs per block", Bayesian optimisation in place of fminbnd.
- Quote: "Twin BL: ν = 0.0001, x⋆s1 = 0.045, x⋆s2 = 0.940, ε⋆1 = 98.76, ε⋆2 = 68.97, max |e| = 2.97e−03".
- **Target: max abs error 2.97e-3.**
- The multi-split construction is only sketched ("composing the same partition-and-gate construction"), so this
  target is **partially specified**. Our construction is fixed in ambiguity 6.

The 2D Poisson case (§4.4.2) reports no number (only Figs. 12-13 against an FD solution), so it has no target. The
meta-learning section (§4.3) reports only timings of a search strategy, so it has no accuracy target either.

## Ambiguities and our fixed choices

1. **λ, the forward Tikhonov term**, is not given ("a small Tikhonov term").
   - We use λ = 1e-12 (relative to W = I/N_f).
   - We solve it as the stacked least-squares problem [√W R; √λ I] c = [−√W f; 0] with `numpy.linalg.lstsq`.
     This is the same minimiser as the paper's normal equations, without squaring the condition number.
2. **The validation objective "residual energy"** is taken as mean(r²) over the 800 uniform validation points
   x = (j − 0.5)/800.
3. **Test points**: 2 × 10^4 uniform points on [0, 1], endpoints included.
4. **`fminbnd`** becomes `scipy.optimize.minimize_scalar(method="bounded")`. It uses the same algorithm (Brent's
   golden-section with parabolic interpolation), with xatol equal to the paper's TolX. The structure is nested:
   for each x_s, the best ε_scale.
5. **Inverse problem.**
   - The data sites are x_n = 1 − (1 − t_n)^3, with t_n uniform on [0, 1] including the endpoints. The
     collocation points come from the gated layout.
   - Noise is redrawn per seed: **10 seeds** per ν_true. The score is the median of the % error over the seeds.
   - The paper's Bayesian optimiser (MATLAB `bayesopt`, EI-plus) is not available. Ours is a Gaussian-process
     EI optimiser with the same 30-evaluation budget and the same box:
     - a Matérn-5/2 GP on the box scaled to [0, 1]^3;
     - 5 Sobol initial points;
     - EI maximised over 2000 random candidates;
     - seeded together with the noise seed.
   - The η fixed point starts at η = 1e-7, runs at most 50 iterations, is clipped to [1e-12, 1e-2], and its
     damping is 0.5 (a log-space average).
6. **Twin boundary layer.**
   - Three blocks [0, x_s1), [x_s1, x_s2) and [x_s2, 1], with 1200 centers and 1200 collocation points each.
   - The nominal width of each block is k × (block length)/1200.
   - Each center's width is blended across both splits by its own logistic gate, with
     ε_b,j = max(ε_scale,j · min(adjacent spacings), 5ν).
   - The trial function is u = 1 + Σ c_i ψ_i, the same constrained expression (g = 1).
   - The search uses the GP-EI optimiser of ambiguity 5, with 60 evaluations over:
     - x_s1 ∈ [0.001, 0.2] and x_s2 ∈ [0.8, 0.999];
     - ε_scale,1 and ε_scale,2 ∈ [10, 100];
     - objective: the validation residual energy;
     - 10 optimiser seeds.
   - The 60 evaluations and both boxes are our choice; the paper does not give them.
7. **Seeds.** The forward problem is deterministic (deterministic layouts and a deterministic search), so it runs
   once. The inverse problem and the twin boundary layer use 10 seeds.

## Status rule

`pinneapple_analysis.reproduction.status`, as fixed in PR #891:

- **F1-F3, T1:** the band is a factor of 2 on the max abs error. ordering_holds is None.
- **F4:** "reproduced" when both hold:
  - the gated error < the ungated error;
  - the ungated error is in [1e-3, 1e-2), which is O(1e-3).
  - If only the first holds, the status is "partially reproduced".
- **I1-I4:** the band is a factor of 2 on the % error. ordering_holds is None.

## Baselines at equal budget (after the reproduction is scored)

All methods have 2000 points and 2000 basis functions, at ν ∈ {1e-2, 1e-3, 1e-4}:

- ungated RBF X-TFC (x_s = 1/2);
- `solve_xtfc` (tanh ELM, unif(−1, 1) and unif(−10, 10) weights, uniform points, 10 seeds);
- the same tanh ELM with constraints as rows (PIELM).

All are scored on the same 2 × 10^4 test points.

## Amendment 1 (before any run): the Bayesian optimiser

Ambiguity 5 described a new GP-EI optimiser. We use the library's existing one instead:
`pinneapple_design.design_optimizer.optimizer.BayesianDesignOptimizer`. It does the same job, and the code is
reused rather than duplicated.

- **Surrogate:** scikit-learn 1.9.1 `GaussianProcessRegressor` with a Matérn 5/2 kernel, `normalize_y` and 3
  restarts.
- **Acquisition:** EI with xi = 0.01, maximised over 500 uniform random candidates in the box (the box is scaled to
  [0, 1]^d).
- **Start:** `n_initial_random = 5` uniform random points.
- **Budget and seeds:** 30 evaluations for the inverse problem and 60 for the twin boundary layer. The optimiser is
  seeded with the run seed.

Everything else in ambiguities 5 and 6 is unchanged.
