# De Florio, Schiassi, Calabrò, Furfaro (2024): frozen targets

Written and committed **before any run**. Our settings and the scoring rule are fixed here too.

## Source

- M. De Florio, E. Schiassi, F. Calabrò, R. Furfaro, "Physics-Informed Neural Networks for 2nd order ODEs with
  sharp gradients", *J. Comput. Appl. Math.* 436 (2024) 115396, doi:10.1016/j.cam.2023.115396.
- **Source of the numbers:** PDF text supplied by the owner on 2026-10-25, pasted into this session. This session
  cannot reach the publisher or doi.org.
- The article is copyrighted (Elsevier). Only target numbers and short phrases are quoted here; the text is not in
  the repository.
- Page numbers are the article's printed page numbers.
- Two references the paper defers to were read from arXiv, because the paper does not restate them:
  - **[9]** F. Calabrò, G. Fabiani, C. Siettos, "Extreme learning machine collocation for the numerical solution
    of elliptic PDEs with sharp gradients", arXiv:2012.05871v2 (CMAME 387 (2021) 114188): the weight sampling.
  - **[14]** S. Mishra, R. Molinaro, "Estimates on the generalization error of PINNs for approximating PDEs",
    arXiv:2006.16144v3: the definitions of ε_T and ε_G.

## Problem (§3, p. 3)

- **Equation:** u_xx − Pe u_x = 0 on x ∈ [0, 1], with u(0) = 1, u(1) = 0 and Pe = 1/ν.
- **Exact solution:** u = (1 − e^{Pe(x−1)})/(1 − e^{−Pe}).
- **Values:** ν ∈ {1, 0.1, 0.01, 0.001, 0.0001}.
- **Constrained expression** (eq. 9, p. 4), for X-TFC and Deep-TFC:
  u = g + (x_f − x)/(x_f − x_0) (u(0) − g(0)) + (x − x_0)/(x_f − x_0) (u(1) − g(1)).

## Settings (Table 1, p. 4)

| | PINN | PIELM | Deep-TFC | X-TFC |
|---|---|---|---|---|
| layers | 10 | 1 | 10 | 1 |
| activation | e^{−(•)²} | e^{−(•)²} | e^{−(•)²} | e^{−(•)²} |
| optimiser | ADAM/LBFGS | least squares | ADAM/LBFGS | least squares |
| N_int | 10000 | 10000 | 10000 | 10000 |
| N_BC | 2 | 2 | 0 | 0 |
| epochs | 15000 | 1 | 15000 | 1 |
| mini-batch (ADAM) | 2000 | 10000 | 2000 | 10000 |
| batches (ADAM) | 5 | 1 | 5 | 1 |

- **Neurons per layer** (superscripts of Tables 2-6):
  - PINN and Deep-TFC: 2e1 for ν = 1 and 0.1; 2e2 for 0.01; 1e3 for 0.001 and 1e-4.
  - PIELM and X-TFC: 1e2, 1e2, 1e3, 1e4 and 1e4.
- **Weight sampling** for PIELM and X-TFC (p. 4): "sampled according to the sampling technique proposed in [9]".
- The authors ran MATLAB R2022a on an i7-9700 with 64 GB.

## Targets: the PIELM and X-TFC columns (Tables 2-6, p. 5-6)

These are the primary targets. Each cell gives max|u*−u|, mean|u*−u| and ε_G, in that order.

| ν | Table | PIELM: max / mean / ε_G | X-TFC: max / mean / ε_G |
|---|---|---|---|
| 1 | 2 | 6.11e−15 / 2.10e−15 / 2.99e−15 | 3.89e−16 / 1.17e−16 / 1.57e−16 |
| 0.1 | 3 | 3.18e−13 / 1.64e−13 / 1.90e−13 | 1.49e−13 / 5.93e−14 / 7.31e−14 |
| 0.01 | 4 | 2.18e−08 / 2.18e−08 / 2.16e−08 | 1.73e−12 / 9.45e−14 / 6.70e−13 |
| 0.001 | 5 | 4.89e−05 / 4.89e−05 / 4.88e−05 | 2.83e−12 / 7.09e−13 / 1.18e−12 |
| 0.0001 | 6 | 5.0e−01 / 5.0e−01 / 5.0e−01 | 2.89e−03 / 1.39e−03 / 1.74e−03 |

Also quoted (reported, not scored):

- **ε_T** (PIELM / X-TFC):

  | ν | PIELM | X-TFC |
  |---|---|---|
  | 1 | 5.26e−11 | 2.44e−12 |
  | 0.1 | 3.33e−09 | 1.46e−09 |
  | 0.01 | 1.38e−06 | 1.23e−06 |
  | 0.001 | 1.48e−04 | 6.83284e−06 |
  | 1e-4 | 1e0 | 4.87e2 |

- **Training time** in seconds (PIELM / X-TFC):

  | ν | PIELM | X-TFC |
  |---|---|---|
  | 1 | 0.02 | 0.02 |
  | 0.1 | 0.02 | 0.02 |
  | 0.01 | 2.11 | 2.14 |
  | 0.001 | 156.06 | 153.78 |
  | 1e-4 | 155.60 | 153.81 |

- **Boundary errors** |u*(x_0) − u(x_0)| and |u*(x_f) − u(x_f)|:
  - X-TFC: 0 everywhere.
  - PIELM: 2.22e−15 and 3.29e−16 (ν = 1), then 1.36e−13 and 5.89e−14, then 2.18e−08 at both ends, then 4.89e−05
    at both ends, then 5.0e−01 at both ends.
- **Text claims** (p. 4):
  - At ν = 1 the mean absolute errors are "of the order of e-05 and e-08 for PINN and Deep-TFC, respectively, and
    e-15 and e-16, for PIELM and X-TFC".
  - "The first failure on the part of PINN is seen for the case ν=0.01".
  - At ν = 0.001 "the failure of Deep-TFC appears", while PIELM and X-TFC keep e-08 and e-14.
  - At ν = 0.0001 "all the methods fail, but X-TFC, which learns the solution with a mean absolute error of
    e-03".
  - The text's "e-08" for PIELM at ν = 0.001 disagrees with Table 5 (4.89e−05). The table is the target.

## Secondary targets: the PINN and Deep-TFC columns (max|u*−u|, Tables 2-6)

| ν | PINN_A | PINN_L | Deep-TFC_A | Deep-TFC_L |
|---|---|---|---|---|
| 1 | 3.15e−05 | 9.96934e−08 | 1.04022e−06 | 1.619e−07 |
| 0.1 | 5.7474e−05 | 1.47079e−05 | 1.03e−04 | 1.10e−04 |
| 0.01 | 5.00e−01 | 5.00e−01 | 2.59e−02 | 6.18332e−05 |
| 0.001 | 5.00e−01 | 5.00e−01 | 9.58e−01 | 9.58e−01 |
| 0.0001 | 5.00e−01 | 5.00e−01 | 9.58e−01 | 9.58e−01 |

- **These are reduced-scale runs** (see ambiguity 7). They can only be "reduced-scale". What we compare is the
  qualitative pattern: whether each method learns or fails at each ν.
- **A paper oddity, recorded and not resolved:** the PINN and Deep-TFC columns of Table 6 (ν = 1e-4) are
  identical to Table 5, including the times. A copy error in the paper is likely.

## Ambiguities and our fixed choices

1. **Weight sampling [9]**, read from arXiv:2012.05871v2 §3, eq. (3.3):
   - α ~ U(−((n−10)/10 + 4), (n−10)/10 + 4) for n neurons;
   - centers C ~ U(0, 1);
   - bias β = −α C.
   - [9] states this for sigmoids. De Florio et al. apply "the sampling technique" to e^{−(•)²}, so the neuron
     is exp(−(α x + β)²) = exp(−α² (x − C)²).
   - Implemented in `pinneapple_simulation.numerical_solvers.xtfc_general.sample_calabro`.
2. **ε_G and ε_T [14]**, from arXiv:2006.16144v3 eqs. (3.7)-(3.8):
   - ε_G = (∫_0^1 |u − u*|² dx)^{1/2}, computed on the test grid of item 4 by the trapezoid rule.
   - ε_T² = Σ w_n |R(x_n)|², with quadrature weights w_n = 1/N_int on the training points. PIELM (and PINN) add
     the two squared boundary residuals with weight 1.
3. **Training points.** N_int = 10000 equispaced interior points x_j = j/10001, j = 1..10000, which matches
   [9]'s "M − 2 equidistant collocation points in (0, 1)".
   - PIELM adds the two boundary rows u(0) = 1 and u(1) = 0 (N_BC = 2), each with weight 1.
   - X-TFC uses the constrained expression of eq. (9) with x_0 = 0 and x_f = 1.
4. **Test points:** 20 001 equispaced points on [0, 1], endpoints included. max, mean and std of |u* − u| are
   taken there.
5. **Least squares:**
   - `scipy.linalg.lstsq` with the LAPACK `gelsy` driver (complete orthogonal factorisation, like MATLAB's
     backslash on a rectangular system).
   - The system is unweighted: one row per collocation point, plus the BC rows for PIELM.
   - Times on our CPU are reported, not scored.
6. **Seeds:**
   - 10 for ν ∈ {1, 0.1, 0.01}.
   - 3 for ν ∈ {0.001, 0.0001}, because the 10000 × 10000 solves are expensive. This is declared, and the
     configuration is not reduced.
   - Score: the median over seeds.
7. **PINN and Deep-TFC: reduced scale, declared.**
   - Kept from the paper: 10 hidden layers, activation e^{−(•)²}, N_int = 10000 equispaced points, mini-batches of
     2000 for Adam. Deep-TFC uses the eq. (9) constrained expression; PINN uses 2 BC points.
   - Neurons per layer: 20 for ν ∈ {1, 0.1}, as in the paper, and 50 for ν ≤ 0.01. The paper uses 200 and 1000
     there.
   - Training: 2000 Adam epochs (lr 1e-3; the paper's rate is "adaptive", unspecified) and then 500 L-BFGS steps
     on the full batch. The paper uses 15000 epochs.
   - Weights: PyTorch's default Linear initialisation (the paper says "randomly initialized").
   - The loss is the plain sum of squared residuals over the points, plus the BC terms for PINN (eq. 6).
   - 3 seeds for each ν.
   - Columns "_A" are scored after the Adam phase, and "_L" after Adam + L-BFGS. The paper does not say whether
     L-BFGS started from scratch or after Adam.

## Status rule (`pinneapple_analysis.reproduction.status`, PR #891)

- **Primary** (PIELM and X-TFC max|u*−u| per ν):
  - If the paper's value is < 1e-6, "reproduced" means our median is within one order of magnitude of it.
    Otherwise the band is a factor of 2.
  - `ordering_holds`: X-TFC median ≤ PIELM median at the same ν (the paper's "X-TFC outperforms PIELM").
- The mean and ε_G are reported with the same band, as secondary entries.
- **PINN and Deep-TFC:** always "reduced-scale". For each method and ν we also report learned (max error < 1e-2)
  or failed, and compare it with the paper's pattern.

## Baselines at equal budget (after the reproduction is scored)

- Same points, neurons and solver as the primary runs, scored on the same test grid:
  - our default tanh X-TFC (`solve_xtfc`, unif(−1, 1));
  - the X-TFC of this paper with the [9] weights;
  - the Gated X-TFC of arXiv:2510.01039 (`gated_xtfc`), with the same number of basis functions and its own
    gate search.

## Amendment 1 (before any run): the PINN and Deep-TFC trainer

Ambiguity 7 is implemented with the library's trainer, `pinneapple_analysis.validation.pinn.train_pinn`, instead of
a new training loop. That changes three details of item 7, all within a run that is reduced-scale anyway:

- **Collocation:** 10 000 points drawn uniformly at random in [0, 1], seeded, and fixed during training. They
  replace the equispaced points.
- **Batching:** full-batch Adam (lr 1e-3, 2000 steps), then L-BFGS with `max_iter = 500` and a strong-Wolfe
  line search. Mini-batches of 2000 are not used.
- **Loss and precision:**
  - The loss is the mean squared residual plus 1.0 × the mean squared boundary misfit (PINN).
  - Weight 1 mirrors eq. (6), which weights the two sums equally.
  - Computed in float32, the trainer's default.
- **Network:** `mlp(1, 1, width, 10, activation="gaussian")`. For Deep-TFC it is wrapped in
  `ConstrainedExpression1D` (eq. 9).
- **_A vs _L:** the "_A" column is read at the end of the Adam phase of the same run whose final state gives "_L".
