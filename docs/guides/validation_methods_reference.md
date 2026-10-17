# Validation methods reference

This page lists every check that PINNeAPPle's validation layer runs on a result, a model or a dataset. For each one it says:

- what the check detects;
- what it assumes;
- its default threshold and where to change it;
- the reference it follows;
- what it cannot detect.

It covers the existing modules: `pinneapple check`, `pp.trust_card`, `pinneapple_analysis.validation`, `pinneapple_analysis.uncertainty`, `pinneapple_analysis.verification`, `pinneapple_veriphysics`, the physics audit (`pinneapple_physics.physics_audit`), conservation projections, and the zoo query gate.

A check passing is evidence, not proof. Every check here can be passed by a wrong result that happens to satisfy what the check measures. Section 10 explains how to measure the power of a check (mutation testing, metamorphic relations, adversarial search). The limitations listed for each check are the known ones. If you find a new one, add it here.

## 1. Status and decision vocabulary

The modules grew separately and use different words for the same outcomes. Until one schema exists (#696, #706), use this table to read them consistently.

| Unified status | `pinneapple check` | `trust_card` | `TrustReport` | physics audit monitor | zoo query gate | veriphysics map |
|---|---|---|---|---|---|---|
| pass | (no finding) | `PASS` | `supports` | `OK` | `pass` | `PASS` |
| warn | `warning` | (not used) | (not used) | `WARN` | `warn` | `INFO` |
| fail | `error` | `FAIL` | `contradicts` | `ALARM` (critical or persistent) | `fail` | `FAIL` |
| not-applicable / not run | (rule not applicable, PC002 if nothing ran) | `N/A` | `not_run` | (check absent) | (check absent) | `NOT_RUN` |

The deploy decisions `APPROVED` / `REVIEW` / `REJECT` (`TrustReport.decisao`, `FieldTrustCard.decision`, `pp.zoo.validate(...).decisao`) follow fixed rules:

- any failed critical check gives `REJECT`;
- any other failure gives `REVIEW`;
- coverage (the fraction of checks that ran) below `min_coverage`, default 0.6 (0.5 for `trust_card`), gives `REVIEW`;
- a weighted score below `min_score` (default 0.8) gives `REVIEW`;
- otherwise the decision is `APPROVED`.

A check that could not run is never counted as a pass. It lowers coverage.

Stable check IDs: `pinneapple check` rules are `PC001` to `PC901` (`pinneapple_check.rules.RULES`), grouped by hundreds (`--select PC1` picks every conservation rule). The zoo query gate reuses these IDs: PC601, PC602, PC402, PC403, PC302.

## 2. `pinneapple check` (results and models in CI)

Guide: [Physics checks in CI](physics_checks_in_ci.md). Every tolerance is relative and dimensionless. You can override each one per target under `tolerances` (`pinneapple_check.config.DEFAULT_TOLERANCES`).

| Rule | Detects | Default | Assumes / limitations |
|---|---|---|---|
| PC101 mass | net boundary volume flux / total absolute boundary flux | 1e-3 | incompressible or steady flow. On OpenFOAM cases it uses `phi` when written (exact to solver tolerance), otherwise the interpolated face velocity, which carries interpolation error. A leak that is balanced elsewhere on the boundary is invisible. |
| PC102 divergence | RMS(div u) / RMS(sum of abs velocity gradients) | 1e-2 (stored), 5e-2 (model) | the discrete operator (FV, FD or least squares) has its own truncation error. On coarse meshes the value is dominated by it. |
| PC103 energy | net boundary heat flow plus source / total absolute heat flow | 1e-2 | steady conduction with constant k. Convection and radiation terms are not included. |
| PC104 momentum | resultant of boundary tractions plus body force / total traction | 1e-2 | static linear elasticity, plane strain in 2-D. |
| PC105 conserved total | max \|Q(t) - Q(t0)\| / \|Q(t0)\| | 1e-3 | a closed domain, or a declared boundary-flux column. Q(t0) = 0 makes the ratio undefined. |
| PC201/PC202 BC/IC | max \|value - declared\| / field scale | 1e-3 / 1e-2 | Dirichlet values only. Neumann conditions are checked through the balances. |
| PC203 wall velocity | max \|U\| on `wall` patches / velocity scale | 1e-6 | a moving wall must be declared. |
| PC301/PC302 residual | RMS residual / RMS of the sum of absolute terms | 5e-2 | the ratio is 0 when the equation holds, near 1 when the terms do not balance. Discretisation error makes it non-zero on coarse grids, which is why the default is loose. A small residual does not mean a small error: the residual does not see a wrong boundary condition or a wrong parameter that the PDE still satisfies. On point clouds the weighted quadratic fit gives first-order second derivatives (second order on tensor-product grids), so the residual of an exact solution falls more slowly with density there. |
| PC401/PC402/PC403 | values outside declared bounds; physically impossible values (T, rho at or below 0, fractions outside [0, 1], k, epsilon, omega, nut below 0); NaN/Inf | 0 slack (1e-6 on fractions) | PC402 infers the quantity from the field name and unit. |
| PC501–PC504 units | unit mismatch or missing unit; magnitudes implausible for the unit; parameters of the wrong dimension | none (exact) | units must be written in the file or declared. |
| PC601/PC602 OOD | query outside the training box; query inside the box but more than 3 typical spacings from every training sample (standardised) | none | box bounds miss correlated inputs. A query inside the box can still be far from the data manifold, and PC602 only partly covers this. |
| PC701/PC702 regression | relative L2 against a stored reference; QoI relative/absolute tolerance | 1e-2 | the reference must have been produced by a trusted run. |
| PC801–PC804 integrity | SHA-256 manifest, signatures, unsafe pickles, untracked files | none (exact) | |
| PC901 catalog | the method is not `validated` in `pinneapple_catalog` | none | |

## 3. Trust card of a field on a grid (`pp.trust_card`)

`pinneapple_analysis.trust.field_card` runs on a tensor-product grid and uses NumPy only. Tolerances are in `DEFAULT_TOLERANCES` and can be overridden with `tolerances={...}`.

| Check | Measures | Default | Reference / limitations |
|---|---|---|---|
| finite values (critical) | NaN or Inf anywhere | 0 | a non-finite field stops the card: REJECT. |
| PDE residual (critical) | `laplace(u) = f` or `u_t = alpha laplace(u) + f`, three-point finite differences on the interior. RMS residual / RMS of the sum of absolute terms. | 0.05 | Second order on uniform grids. The tests check order 2.00 on uniform and stretched grids against exact Laplace, Poisson and heat solutions. The scale has a floor at the round-off of the stencil, so exactly harmonic linear fields pass (#708). It needs 4 or more points per axis and only checks interior points. |
| boundary conditions (critical) | max \|u - g\| / range of u on each named face | 0.02 | Dirichlet only. |
| conservation | \|net boundary flux of grad u - integral of f\| / total absolute flux (divergence theorem) | 0.05 | Steady problems only. Uses second-order one-sided boundary gradients and trapezoidal quadrature, so it needs 3 or more points per axis. A field with the right total flux but the wrong distribution passes. |
| maximum principle | interior overshoot beyond the boundary range / range of u | 0.01 | Laplace without a source only. |
| reference (critical) | relative L2 against an array or a closed form | 0.05 | A zero reference makes the ratio meaningless. Give an absolute comparison instead. |
| applicability (OOD) | fraction of query points inside `train_bounds` | 1.0 | Box bounds only. |

`pinneapple_analysis.trust.TrustGate` scores one prediction of a trained model:

- **OOD:** each query point gets the chi-squared p-value of its Mahalanobis distance to the training coordinates. The score is the fraction of points with p >= `ood_level` (default 0.01) (#713). `fit_from_bounds` treats a box as a Gaussian with the box variance; in 6 or more dimensions the corners of the training box fall outside the 99 % region.
- **Residual:** `1 / (1 + RMS residual)`. This depends on the units of the residual, so compare it only between models of the same problem.
- **Ensemble:** `1 / (1 + mean relative std)`.

`trust_report.from_trust_score` turns these scores into checks with a design threshold of 0.7.

## 4. Model validation (`pinneapple_analysis.validation`)

`PhysicsValidator` evaluates a model (with autograd) at sampled points. Each check defaults to `tolerance=1e-3`.

- **ConservationCheck** (mass, energy, custom integrals): Monte Carlo integration. The estimate has O(n^-1/2) sampling noise, so `n_points` must be large enough that the noise is below the tolerance.
- **BoundaryCheck** (Dirichlet, Neumann, periodic).
- **SymmetryCheck** (reflection, 2-D rotation): `max |u(Sx) - s u(x)|`. Use it only when the problem, including its boundary conditions and sources, has the symmetry.
- **compare_to_analytical / validate_against_solver**: RMSE, relative L2 and max error against a closed form or solver data.

Limitation: sampled checks can miss narrow features such as boundary layers or shocks. Use the grid checks in sections 2 and 3 on a resolved field when such features matter.

## 5. Uncertainty (`pinneapple_analysis.uncertainty`)

| Method | What it guarantees or measures | Assumptions / limitations |
|---|---|---|
| `ConformalPredictor` (split conformal; Vovk et al. 2005, Angelopoulos and Bates 2022) | marginal coverage of at least 1 - alpha in finite samples. The quantile is the ceil((n+1)(1-alpha))-th order statistic, never interpolated. A Monte Carlo check with n_cal = 20 gives 0.902 for alpha = 0.1, inside [0.90, 0.948]. | Exchangeable calibration and test data. Coverage is marginal, not conditional on x. The score is the L-infinity residual over outputs, so coverage is joint across outputs. The guarantee does not survive distribution shift: check drift. |
| `QuantileHead` / `QuantileLoss` (pinball loss; Koenker and Bassett 1978) | learned conditional quantiles | No coverage guarantee without conformal calibration (CQR; Romano, Patterson and Candès 2019). `QuantileHead` sorts its outputs so they cannot cross (`rearrange=False` turns that off; `crossing_rate` measures it); the head sees only the point prediction, not the input. |
| `CalibrationMetrics` (Kuleshov et al. 2018) | ECE of the PIT of standardised residuals against the uniform distribution; coverage of Gaussian intervals; NLL; sharpness | Gaussian predictive distribution. Multi-output metrics are element-wise averages (#715). ECE is below 0.01 for a calibrated model and about 0.1 when the std is off by a factor of 2. Default review tolerances in veriphysics: ECE 0.10, coverage within 0.05. |
| `assess_drift` (`DriftThresholds`) | ok / review / retrain from nRMSE (0.2 / 0.5), bias z-test (3 sigma), coverage z-test (3 sigma), per-feature KS with Bonferroni (alpha 0.01), out-of-range fraction (0.01 / 0.10), Mann-Kendall trend (alpha 0.01) | Tests assume independent residuals. Autocorrelated time series inflate the false-alarm rate. |
| `propagate` (tolerances) | Monte Carlo propagation of input tolerances, Wilson intervals on pass fractions | The input distribution (uniform or normal) is an assumption you supply. |
| `crps_gaussian` / `crps_ensemble` (Gneiting et al. 2005; Gneiting and Raftery 2007; Ferro 2014) | proper scores of a predictive distribution | `crps_gaussian` matches the closed form exactly. `fair=True` is unbiased for small ensembles: with 5 members, the average over 20 000 draws is 0.4218 against the exact 0.4216, while the plain version gives 0.532. |
| `block_bootstrap` / `group_bootstrap` (Künsch 1989) | percentile intervals of a statistic of dependent data | Percentile intervals undercover in small or strongly dependent samples. In a Monte Carlo test of the mean, iid data with n = 100 gave 0.93 coverage at a nominal 0.95. An AR(1) series with coefficient 0.9, n = 1200 and blocks of 60 gave 0.90. Choose blocks several correlation times long, and read the interval as approximate. |
| ensembles, MC dropout, decomposition, posterior metrics, resampling, reachability | epistemic spread, aleatoric/epistemic split (law of total variance), bootstrap intervals, reachable sets | Ensemble spread is not calibrated by itself: calibrate it against a reference (veriphysics `calibration_against_reference`) or with conformal prediction. |

## 6. Solution and code verification (`pinneapple_analysis.verification`)

- **`richardson_extrapolate` / `mesh_independence_study`** (Roache 1994; Celik et al. 2008). These compute the observed order, the Richardson value, GCI_fine and GCI_coarse with Fs = 1.25, and the asymptotic ratio `GCI_coarse / (r^p GCI_fine)`, which is near 1 in the asymptotic range (tolerance 0.10).
  - Non-constant refinement ratios use Celik's fixed-point iteration (#710). The tests reproduce Celik et al. (2008) Table 1: p = 1.53, phi_ext = 6.1685, GCI_fine = 2.17 %.
  - Ratios inferred from resolution counts assume h ~ 1/N in one dimension. For cell counts in D dimensions, pass `r = (N_fine/N_medium)^(1/D)` explicitly.
  - Oscillatory convergence gives a warning. The GCI is then not a reliable error band.
  - Array-valued quantities use L2 norms of the differences.
- **Manufactured solutions** (Roache 2002; Salari and Knupp 2000). These are in `tests/test_manufactured_solutions.py`. Code verification means the observed order matches the scheme's formal order on a solution chosen in advance.
- **`dimensional_analysis`**: unit consistency and Buckingham Pi groups.
- **`geometry_ood_guardrail`**: z-scored geometry descriptors against a reference set.
- **`physics_confidence_score`**: a bounded combination of calibration, convergence and residual evidence. The weights are design choices, not a standard.
- **`provenance`**: hashes of the inputs of a run.

## 7. Veriphysics applicability and robustness (`pinneapple_veriphysics`)

The applicability map sorts every claim into three tiers:

- `VERIFIED`: a check measured it in this run;
- `INFERRED`: plausible, but not measured;
- `UNSUPPORTED`: no evidence; do not use the surrogate there.

It also answers an eight-item review checklist. The tolerances are reviewer aids, not standards, and can be replaced per request (`Tolerances`):

- training plateau: below 5 % change over the last 10 % of epochs;
- ensemble and parameter-shift spread: 0.10;
- ECE: 0.10;
- coverage: within 0.05.

The robustness studies are:

- ensembles over seeds;
- extrapolation shells at 10, 25 and 50 % beyond the training box;
- loss-weight sensitivity (x0.25 and x4);
- parameter shifts (10 and 30 %);
- calibration against a reference;
- coverage of the boundary and initial conditions.

A small spread under these perturbations shows stability, not correctness: an ensemble can agree on a wrong answer.

## 8. Physics audit monitor and conservation projections (`pinneapple_physics`)

`physics_audit.monitor.PhysicsValidityMonitor` runs residual, conservation, bounds, boundary, constitutive, invariant and envelope checks on every prediction. A failed check gives `WARN`. A failed critical check, a failure that persists for `persistence` cases in a row (default 3), or a residual whose EWMA reaches 3x its baseline gives `ALARM`.

- Conservation and invariant checks are relative to the expected value. When the expected value is zero (net charge, net momentum of a closed system), give `abs_tol` (#711).
- `boundary_check` scales the mismatch by the larger of the RMS of the condition and the RMS of the prediction, so homogeneous conditions can pass (#711).

`pinneapple_physics.conservation.project_integral` enforces a weighted integral exactly, with additive, multiplicative or positive modes. It guarantees the total, not the local balance: a projected field can still violate the PDE locally.

## 9. Zoo query gate (`pinneapple_hub.zoo.checks`)

Each query is checked for:

- the validity range (PC601);
- sparse training regions (PC602);
- non-finite or unphysical outputs (PC403, PC402);
- ensemble agreement against its held-out level;
- the model's physics residual at the query (PC302).

Any failing check gives `fail`, any warning gives `warn`, otherwise the verdict is `pass`. Design constraints are reported but never enter the verdict. The noise floor is 0.2 % of the output range. `pp.zoo.validate` reproduces the card metrics on the model as a whole.

## 10. How good are the checks? Credibility, check power and attacks

### Credibility frameworks

ASME V&V 40-2018 (*Assessing Credibility of Computational Modeling through Verification and Validation: Application to Medical Devices*) and NASA-STD-7009A (*Standard for Models and Simulations*, 2016) do not prescribe thresholds. They ask that the rigour of each credibility factor scales with the model risk, which combines the influence of the model on the decision and the consequence of a wrong decision, within a stated context of use. The factors are code verification, solution verification, validation against references, uncertainty and applicability.

The checks above map to those factors:

| Credibility factor | Checks |
|---|---|
| code verification | manufactured solutions, solver order tests |
| solution verification | GCI, residual checks |
| validation | reference, regression, field metrics |
| uncertainty | section 5 |
| applicability | OOD, envelope, extrapolation shells |

The default thresholds here are engineering defaults for a low-to-medium-risk context. For a high-risk context of use, tighten them and require more factors to be `VERIFIED`. Do not loosen them to get a pass. See also Oberkampf and Roy (2010), *Verification and Validation in Scientific Computing*.

### Mutation testing: measuring check adequacy

A check that never fails is not evidence. Mutation testing injects known faults and counts how many each check catches; the mutation score measures how adequate the checks are. Faults can be injected into the model (weights, scaling, sign flips) or into the data (drift, leakage, units). Relevant work:

- DeepCrime (Humbatova, Jahangirova and Tonella, ISSTA 2021): mutation operators derived from real deep-learning faults.
- MuFF (arXiv:2501.09846): stable post-training mutants.

### Metamorphic relations

When no oracle exists, check relations that must hold between runs: symmetry of the inputs, scaling laws, superposition for linear problems, refinement consistency. Relevant work:

- Chen, Cheung and Yiu (1998), *Metamorphic testing: a new approach for generating next test cases*, HKUST-CS98-01;
- Segura et al. (2016), *A survey on metamorphic testing*, IEEE TSE;
- METTLE (arXiv:1807.10453).

The symmetry check in section 4 is a metamorphic relation.

### Adversarial robustness of neural operators

Small input perturbations can make neural-operator errors grow by orders of magnitude while the outputs stay plausible:

- Adesoji and Chen (2022, arXiv:2204.04259) show this for FNO on Burgers, Darcy and Navier-Stokes;
- a 2026 study (arXiv:2604.13316) reports accuracy losses of several orders of magnitude when fewer than 1 % of the inputs are perturbed.

No check on this page searches for such perturbations. A pass is evidence only for the inputs that were checked.

### Red team / blue team arena

Issue #707 proposes an arena, implemented separately. Generators search for wrong-but-plausible cases that escape the checks and for correct cases that raise false alarms. The checks are then hardened, and their thresholds calibrated with a stated false-alarm rate. When it lands, its detection and false-alarm rates per check are the measured power of the methods on this page.

## References

- Roache, P. J. (1994). Perspective: A method for uniform reporting of grid refinement studies. *ASME J. Fluids Eng.* 116(3), 405–413.
- Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H., Raad, P. E. (2008). Procedure for estimation and reporting of uncertainty due to discretization in CFD applications. *ASME J. Fluids Eng.* 130(7), 078001.
- Roache, P. J. (2002). Code verification by the method of manufactured solutions. *ASME J. Fluids Eng.* 124(1), 4–10.
- Salari, K., Knupp, P. (2000). *Code verification by the method of manufactured solutions*. Sandia report SAND2000-1444.
- Oberkampf, W. L., Roy, C. J. (2010). *Verification and Validation in Scientific Computing*. Cambridge University Press.
- ASME V&V 40-2018. *Assessing Credibility of Computational Modeling through Verification and Validation: Application to Medical Devices*.
- NASA-STD-7009A (2016). *Standard for Models and Simulations*.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., Bates, S. (2022). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
- Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1), 33–50.
- Gneiting, T., Raftery, A. E., Westveld, A. H., Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Mon. Weather Rev.* 133, 1098–1118.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102, 359–378.
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Q. J. R. Meteorol. Soc.* 140, 1917–1923.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17(3), 1217–1241.
- Kuleshov, V., Fenner, N., Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*.
- Humbatova, N., Jahangirova, G., Tonella, P. (2021). DeepCrime: mutation testing of deep learning systems based on real faults. *ISSTA*.
- MuFF: arXiv:2501.09846.
- Chen, T. Y., Cheung, S. C., Yiu, S. M. (1998). Metamorphic testing: a new approach for generating next test cases. Technical report HKUST-CS98-01.
- Segura, S., Fraser, G., Sanchez, A. B., Ruiz-Cortés, A. (2016). A survey on metamorphic testing. *IEEE Trans. Softw. Eng.* 42(9), 805–824.
- METTLE: arXiv:1807.10453.
- Adesoji and Chen (2022): adversarial perturbations of Fourier neural operators. arXiv:2204.04259.
- arXiv:2604.13316 (2026): adversarial perturbations of neural PDE surrogates, with a defence by attack-driven active learning and input denoising.
