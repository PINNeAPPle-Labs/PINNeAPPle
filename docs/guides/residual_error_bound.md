# From residual to error bound: how wrong is this prediction, without the true answer?

At query time the physics residual of a surrogate's prediction can be computed without the true answer, but a
residual of 0.03 means nothing to an engineer. `pinneapple_analysis.uncertainty.residual_bound` turns it into an error
bound in two ways, and reports both.

```python
from pinneapple_analysis.uncertainty.residual_bound import (ResidualErrorCalibrator, darcy_problem,
                                                             energy_error_bound)

# 1. theory: guaranteed, no data needed (linear elliptic problems)
P = darcy_problem(a)                                   # -div(a grad u) = 1, the discretisation of darcy_2d
b = energy_error_bound(P, u_pred)                      # ||u_h - u_pred||_A <= b["bound"]; b["relative"] = / ||u||_A

# 2. calibration: tight, guaranteed for queries like the calibration set
res_cal = [energy_error_bound(darcy_problem(a), p)["relative"] for a, p in zip(a_cal, pred_cal)]
cal = ResidualErrorCalibrator(alpha=0.1).calibrate(res_cal, err_cal)          # err_cal: relative L2 errors
bound = cal.bound(res_new)                             # 90 % bound on the relative L2 error of new queries
print(cal.evaluate(res_shifted, err_shifted, "shifted inputs"))   # coverage measured, "BROKEN" if it fails
```

## Theory

For `A u = f` with `A` symmetric positive definite and `A >= c L` (`L` a reference SPD operator),
`||u_h - u||_A <= ||r||_{L^-1} / sqrt(c)` with `r = f - A u`, because `A^-1 <= L^-1 / c`. For Darcy and steady heat
conduction `-div(k grad u) = f` in cell-centred finite volumes (`darcy_problem`, `heat_conduction_problem`; the
matrix is `operator_benchmarks.darcy_matrix`), `L` is the same operator with `k = 1` and `c = min k`: every face
coefficient is a harmonic mean and never below `min k`. The residual is evaluated with the reference solver's own
discretisation, so the reference solution has a residual of round-off size (a finite-difference residual of the same
field is of order one).

The bound is about the discrete solution on the same grid; the discretisation error of that solution is not
included. It is loose when the coefficient contrast is large (it scales with `1 / sqrt(min k)`).

## Calibration

Split conformal prediction (the finite-sample quantile of `uncertainty.conformal`) on the score
`error / g(residual)`, with `g` the identity (`mode="ratio"`, the default) or a power law fitted on a separate set
(`mode="power"`). The bound of a new query is `q * g(residual)`. Coverage is guaranteed only for queries exchangeable
with the calibration set. `evaluate` measures it and says "BROKEN" when it falls more than two standard deviations
(test and calibration sampling) below the target; `coverage_study` averages it over random calibration / test
splits; `to_check` gives the trust-report check "error bound without ground truth".

## Results

`tests/test_residual_bound.py`; Darcy on 32 x 32 cells (Li et al. 2021 coefficients, 400 training pairs, operators
trained with `pinneapple_neural.train_model`), 1000 held-out queries, calibration on 500:

| Model | Median rel. L2 error | 90 % bound: mean coverage over 200 splits | Theory violated |
|---|---|---|---|
| FNO | 4.1 % | 90.1 % (std 1.9 %) | 0 of 1000 |
| DeepONet | 20.9 % | 89.9 % (std 2.0 %) | 0 of 1000 |
| Heat conduction, FNO (source -> temperature) | 1.0 % | 90.1 % (std 1.8 %) | 0 of 1000 |

Under the two input shifts of the PhysicsNeMo trust experiment (150 queries each):

| Model | Shift | Median error | Coverage of the 90 % bound (ratio map) | Theory violated |
|---|---|---|---|---|
| FNO | contrast a in {1, 20} | 76 % | **2.7 % (BROKEN)** | 0 |
| FNO | length scale tau 9 | 4.8 % | 94.0 % | 0 |
| DeepONet | contrast a in {1, 20} | 51 % | 100 % | 0 |
| DeepONet | length scale tau 9 | 20 % | 92.7 % | 0 |

The power-law map breaks under the contrast shift for both models (0 % coverage): it extrapolates the
residual-to-error relation outside the calibrated range. These numbers come from one training run each; another seed
moves them (the FNO's contrast-shift coverage was 100 % in an earlier run), which is the point: under a shift the
calibrated bound must be measured, and only the theoretical bound is guaranteed.

## Limits

- The theoretical bound needs a linear, coercive problem and its discrete operator; for others only the calibrated
  map applies.
- The calibrated bound is marginal (90 % of queries), not per query, and assumes exchangeability with the
  calibration set.
- Not yet shown in `pp.trust_card`; `pp.validate` (#696) reports it as "error bound without ground truth".
