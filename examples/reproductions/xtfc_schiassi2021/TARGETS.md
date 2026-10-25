# X-TFC (Schiassi et al.): frozen targets

Written and committed **before any run**. Nothing here comes from memory: every number is quoted from the PDF
named below, with its page. Our settings and our scoring rule are fixed here too, so they cannot be tuned after
seeing results.

## Source

- E. Schiassi, C. Leake, M. De Florio, H. Johnston, R. Furfaro, D. Mortari, "Extreme Theory of Functional
  Connections: A Physics-Informed Neural Network Method for Solving Parametric Differential Equations",
  arXiv:2005.10632**v1** (15 May 2020), 28 pages. Downloaded from `https://arxiv.org/pdf/2005.10632` on
  2026-10-25. It is the only arXiv version (`/abs/2005.10632v2` returns 404).
- Published as "Extreme theory of functional connections: A fast physics-informed neural network method for
  solving ordinary and partial differential equations", *Neurocomputing* 457 (2021) 334-356,
  doi:10.1016/j.neucom.2021.06.015 (metadata from api.crossref.org). **The journal version was not read**:
  doi.org and the publisher are refused by this session's network. Its numbers may differ from the preprint's.
  Every target below is the preprint's.
- No official code is linked in the preprint. leakec/tfc (the TFC library by co-author C. Leake) is a different
  code base and was not used as a source of numbers.

Page numbers are the PDF's printed page numbers.

## Method as the paper states it

- Solution = TFC constrained expression with a free function g, and g = an ELM. An ELM is a single hidden layer
  with random, untrained input weights and biases. Only the output weights β are solved for (§2.3, p. 7-8).
- Linear problems: one linear least-squares solve. Non-linear problems: iterative least squares (§2.1, p. 6).
- **ODEs (§3.1, p. 8):** "the activation function used in the ELM was a logistic activation function, and the
  weights and biases of the ELM were randomly sampled from a uniform distribution, (w_j, b_j) ∼ unif(−10, 10)".
  The paper ran "10^3 Monte Carlo simulations" per problem. It solved the ODEs in MATLAB.
- **PDEs (§3.2, p. 14):** "All input weights and biases for all problems were randomly sampled from a uniform
  distribution, (w_j, b_j) ∼ unif(−1, 1)". The activation was tanh, and every solve used NumPy's `lstsq`. The
  PDEs were solved in Python with autograd.

## Targets

Metric: absolute error |f − f_true| against the exact solution the paper gives. "max" is the maximum over the
set, "average" the mean of |error|. The paper never defines its test sets (see Ambiguities).

### T1. ODE Problem 1 (§3.1.1, p. 8-10; Table 1, p. 13; Fig. 3, p. 11)

- Setting: y_tt = (1 / (2 t^2)) (y^3 − 2 y^2) on t ∈ [1, 2], y(1) = 1, y(2) = 4/3, exact y = 2t / (t + 1). The
  paper prints "2 x^2" in the denominator, a typo for t: the exact solution only satisfies it with t.
- Constrained expression (p. 10): y(t) = [σ + (t − 2) σ0 + (1 − t) σf]^T β + (t + 2)/3.
- Run settings: logistic, unif(−10, 10). Fig. 2 caption (p. 10): "n = 50, and L = 50". Fig. 3 caption (p. 11):
  "n = 51, and L = 51".
- Quotes:
  - p. 10: "the average absolute errors for both training and testing are on the order of O(10−16)".
  - p. 11, Fig. 3 caption: "The convergence is achieved in 5 iterations with a tolerance set to
    4.440892098500626 × 10−16. ... the maximum absolute test error is 6.66134 × 10−16".
  - Table 1: X-TFC absolute errors at test points t = 1, 1.1, ..., 2 are all 0 or 2.2×10−16.
- **Target: max abs test error 6.66134e-16 (single run, Fig. 3), n = L = 51.**

### T2. ODE Problem 2 (§3.1.2, p. 10-14; Table 2, p. 14; Table 3, p. 16; Figs. 5-6)

- Setting: y1_t = cos t + y1^2 + y2 − (1 + t^2 + sin^2 t), y2_t = 2t − (1 + t^2) sin t + y1 y2 on t ∈ [0, 3],
  y1(0) = 0, y2(0) = 1. Exact: y1 = sin t, y2 = 1 + t^2.
- Constrained expressions (p. 13): y1 = [σ − σ0]^T β1, y2 = [σ − σ0]^T β2 + 1.
- Run settings: logistic, unif(−10, 10), "n = 100, and L = 100". The tolerance is "10 × 10−6", and convergence
  took "9 iterations" (Figs. 5-6 captions, p. 14).
- Quotes:
  - p. 13: "For y1, the average absolute errors for both training and testing are on the order of O(10−12),
    while for y2, the average absolute errors for both training and testing are on the order of O(10−11)".
  - Fig. 5 (y1), p. 14: "the maximum absolute test error is 5.5434 × 10−10".
  - Fig. 6 (y2), p. 14: "the maximum absolute test error is 1.58122 × 10−9".
  - Table 2 (y1): the largest X-TFC entry is 1.8×10−10, at t = 3.
  - Table 3 (y2, captioned "Problem #4 ... scattered flux y2", a caption error): the largest X-TFC entry is
    5.2×10−10, at t = 3.
- **Targets: max abs test error y1 5.5434e-10, y2 1.58122e-9 (Figs. 5-6); average y1 O(1e-12), y2 O(1e-11).**

### T3. PDE Problem 1 (§3.2.1, p. 15-17; Table 4, p. 17)

- Setting: f_xx + f_yy = e^{−x}(x − 2 + y^3 + 6y) on [0, 1]^2. Dirichlet on all four sides, from the exact
  solution f = e^{−x}(x + y^3).
- Run settings: "an ELM with 170 neurons that used tanh"; "discretized over 30×30 training points"; NumPy
  `lstsq`.
- Quotes (p. 17): "the training set maximum error was 3.808 × 10−13, and the training set average error was
  6.475 × 10−14. The test set maximum error was 5.054 × 10−13, and the test set average error was
  7.640 × 10−14".
- Table 4 (test max): X-TFC 5.1 × 10−13, FEM 1.5 × 10−5, Ref. [24] (Lagaris) 5 × 10−7, Ref. [38] 3.2 × 10−2,
  Ref. [39] 2.4 × 10−4.
- **Target: test max 5.054e-13, test mean 7.640e-14.**

### T4. PDE Problem 2 (§3.2.2, p. 17-18; Table 5, p. 18)

- Setting: f_xx + f_yy = (2 − π^2 y^2) sin(πx) on [0, 1]^2. f(0, y) = f(1, y) = f(x, 0) = 0, and
  f_y(x, 1) = 2 sin(πx). Exact f = y^2 sin(πx).
- Run settings: 170 neurons, tanh, 30×30 training points, `lstsq`.
- Quotes (p. 18): "the training set maximum error was 6.332 × 10−12, and the training set average error was
  1.187 × 10−12. The test set maximum error was 7.581 × 10−12, and the test set average error was
  1.322 × 10−12".
- Table 5 (test max): X-TFC 7.6 × 10−12, FEM 4 × 10−5, Ref. [24] 6 × 10−6, Ref. [38] 3 × 10−3.
- **Target: test max 7.581e-12, test mean 1.322e-12.**

### T5. PDE Problem 3, non-linear (§3.2.3, p. 19-20; Table 6, p. 20)

- Setting: f_xx + f_yy + f f_y = sin(πx)(2 − π^2 y^2 + 2 y^3 sin(πx)) on [0, 1]^2. Same constraints and exact
  solution as T4.
- Run settings: 150 neurons, tanh, 20×20 training points, non-linear least squares that "needed 10
  iterations".
- Quotes (p. 20): "the training set maximum error was 7.634 × 10−11, and the training set average error was
  9.497 × 10−12. The test set maximum error was 8.977 × 10−11, and the test set average error was
  1.068 × 10−11".
- Table 6 (test max): X-TFC 9.0 × 10−11, FEM 4 × 10−5, Ref. [24] 1.5 × 10−5. Table 6 gives the training max
  as 8.8 × 10−11, against 7.634 × 10−11 in the text.
- **Target: test max 8.977e-11, test mean 1.068e-11.**

### T6. PDE Problem 4, 1D heat (§3.2.4, p. 21)

- Setting: f_xx = κ f_t, κ = 1, on (x, t) ∈ [0, 1]^2. f(0, t) = f(1, t) = 0, f(x, 0) = sin(πx). Exact
  f = sin(πx) e^{−π^2 t}.
- Run settings: 196 neurons, tanh, 30×30 training points, `lstsq`.
- Quote (p. 21): "the training set max error was 5.611×10−7, and the training set average error was
  6.551×10−8. The test set maximum error was 5.785×10−7, and the test set average error was 6.928 × 10−8".
- **Target: test max 5.785e-7, test mean 6.928e-8.**

### T7. PDE Problem 5, 2D heat (§3.2.5, p. 21-23)

- Setting: f_xx + f_yy = κ f_t on [0, L] × [0, H] × [0, 1], with L = 2, H = 1, κ = 1. Zero Dirichlet on the four
  sides; f(x, y, 0) = sin(πx/L) sin(πy/H). Exact f = sin(πx/L) sin(πy/H) e^{−(π^2/L^2 + π^2/H^2) t}.
- Run settings: 400 neurons, tanh, 13×13×13 training points, `lstsq`.
- Quote (p. 23): "the training set maximum error was 3.985 × 10−4, and the training set average error was
  1.705 × 10−5. The test set maximum error was 4.025 × 10−4, and the test set average error was 2.090 × 10−5".
- **Target: test max 4.025e-4, test mean 2.090e-5.**

### T8. PDE Problem 6, non-linear 2D + t (§3.2.6, p. 23-24)

- Setting: z_t z_x + z_y = t^2 + (t − 1)x + 2πx cos(2πxy) + (2ty + xy)((t − 1)y + 2πy cos(2πxy)) on [0, 1]^3.
  Constraints: z(0, y, t) = t^2 y, z(x, 0, t) = 0, z(x, y, 1) = y + sin(2πxy). Exact
  z = sin(2πxy) + t^2 y + (t − 1)xy.
- Hand check of the printed right-hand side against the printed exact solution: z_t = 2ty + xy,
  z_x = 2πy cos(2πxy) + (t − 1)y and z_y = 2πx cos(2πxy) + t^2 + (t − 1)x. So
  z_t z_x + z_y = t^2 + (t − 1)x + 2πx cos(2πxy) + (2ty + xy)((t − 1)y + 2πy cos(2πxy)), which matches the
  printed RHS. A test re-checks this numerically.
- Run settings: 255 neurons, tanh, 8×8×8 training points, non-linear least squares that "needed 10
  iterations".
- Quote (p. 24): "the training set maximum error was 1.657 × 10−8, and the training set average error was
  1.006 × 10−9. The test set maximum error was 1.657 × 10−8, and the test set average error was 1.018 × 10−9".
- **Target: test max 1.657e-8, test mean 1.018e-9.**

### T9. PDE Problem 7, non-linear 3D + t (§3.2.7, p. 24-25)

- Setting: f_x f_y f_z + f_tt = (RHS printed on p. 24) on [0, 1]^4. Exact
  f = t^2 sin(2πz) + sin(x^2 y) + x y^{3/2} z + x y t (z − 1)(t − 1).
- The printed RHS is garbled in text extraction. **We build the RHS from the exact solution and the operator
  f_x f_y f_z + f_tt**, and we say so. We do not transcribe the printed RHS.
- Constraints: f(0, y, z, t), f(x, 0, z, t), f(x, y, 1, t), f(x, y, z, 0), f(x, y, z, 1), from the exact
  solution.
- Run settings: 340 neurons, tanh, 5×5×5×5 training points, non-linear least squares that "needed 10
  iterations".
- Quote (p. 25): "the training set maximum error was 2.744 × 10−5, and the training set average error was
  6.641 × 10−7. The test set maximum error was 2.978 × 10−5, and the test set average error was
  8.082 × 10−7".
- **Target: test max 2.978e-5, test mean 8.082e-7.**

Run times (ms for `lstsq`, seconds in total) are reported for context only. Their hardware (MATLAB on an i7-9700
for the ODEs, autograd on a 2016 MacBook Pro for the PDEs) differs from ours, so times are not a target.

## Ambiguities and how we resolve them (fixed before any run)

1. **Training-point distribution.** The paper only says "N×N training points that spanned the domain". Primary
   choice: a uniform grid including the boundary. TFC usually uses Chebyshev-Gauss-Lobatto points, so we also
   run those as a declared secondary variant. Both are reported, and the primary is not swapped afterwards.
2. **Test set.** Not defined for the PDEs. Ours: 10 000 points drawn uniformly at random in the domain, from a
   fixed seed that is independent of the ELM seed. For the ODEs: the paper's Table 1/2/3 test points (t = 1:0.1:2,
   and t = 0.3:0.3:3), plus 1 000 uniform points. The status uses the paper's test points where it lists them.
3. **Seeds.** The PDE results are a single run with unknown seed. We run **10 seeds** per PDE problem and score
   the median. For the ODEs we run 1 000 seeds, as the paper's 10^3 Monte Carlo runs. No best-seed selection.
4. **Weight ranges.** unif(−10, 10) for the ODEs and unif(−1, 1) for the PDEs, as stated. Inputs are used in their
   physical coordinates, since the paper states no input scaling.
5. **Non-linear solver.** Gauss-Newton (iterative linear least squares on the Jacobian), started from β = 0. The
   iteration cap is the paper's iteration count +5: 10 for ODE P1, 14 for ODE P2, 15 for PDE P3/P6/P7. Stop on
   ||Δβ|| ≤ the paper's tolerance (ODE) or on 1e-14 (PDE, not stated).
6. **ODE n and L.** T1 uses n = L = 51 (the Fig. 3 run whose number we target). Fig. 2's n = L = 50 is also
   reported, as a secondary run.

## Status rule (from `pinneapple_analysis.reproduction.status`, PR #891)

- The scored quantity is the **median over seeds of the test max abs error**. The test mean is reported next to
  it.
- A paper number below 1e-6 counts as "reproduced" when our median is within one order of magnitude of it.
  Otherwise the band is a factor of 2.
- `ordering_holds` asks whether our median beats the best non-X-TFC entry the paper lists for that problem:
  - T3: Ref. [24] 5e-7.
  - T4: Ref. [24] 6e-6.
  - T5: Ref. [24] 1.5e-5.
  - T1: LeNN O(1e-3) (average).
  - T2: ANN [24] O(1e-4) and O(1e-5).
  - T6-T9 list no competitor, so "partially reproduced" is not available there.
- Errors at machine precision (T1, ~1e-16) cannot be told apart below double-precision rounding. T1 is
  "reproduced" if our median max error is ≤ 6.66e-15 (one order of magnitude, the rule above).

## Baselines at equal budget (step 4, after the reproduction)

Run only after the reproduction is scored. Each baseline gets the same training points (the paper's grid) and the
same test set. The budgets differ by kind, and that is declared:

- **ELM-only:** the same random features without the constrained expression, with boundary rows added to the
  least-squares system.
- **PINN (pp.solve "pinn"):** same collocation points. Wall-clock budget = 100× the X-TFC solve time, or 5 000
  Adam steps, whichever ends first.
- **FEM (pp.solve "fem")** for the linear 2D problems: P1, P2 and the 1D heat problem.

All methods are scored by `pinneapple_analysis.reproduction.evaluate` on the same points.
