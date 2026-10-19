# Viscous Burgers fields u(x, t) across viscosity

Run 2026-10-17T11:13:26+00:00 on Intel(R) Xeon(R) Processor @ 2.80GHz (4 logical CPUs, torch 2.14.1+cu130), 248 s in total. Dataset sha256 `eefeef72b0f22b6a`, 96 samples (field), inputs nu, outputs u.

## Protocol

- Metric: rel_l2 — relative L2 error per held-out field, averaged.
- Budget per training run: 10 wall_s; seeds [0, 1, 2]; split seed 0.
- one split shared by every method and seed; the model seed varies, the data does not.
- Stopping: iterative methods: until the budget is spent, best of 10 validation checkpoints (evaluation time not charged); one-shot fits: fitted once, time reported.
- Hyper-parameters: each method's defaults in pinneapple_analysis.surrogate.methods; no per-method search.
- One-shot fits (ridge, gp, gbt) use the time they need, which can be less than the budget; their train_s is that time.

## Regimes

- **interpolation**: random held-out samples inside the training distribution (54 training, 16 evaluated)
- **few data**: 14 of 54 training samples (25%); same held-out samples as interpolation (14 training, 16 evaluated)
- **extrapolation**: nu <= 0.006179 (14 samples): extrapolation in nu (54 training, 14 evaluated)
- **geometry change**: not run — the data has no groups (geometry or case labels)
- **steep fronts**: held-out fields with the steepest gradients (top quartile of max |grad u|) (54 training, 8 evaluated)

## Rankings (mean ± std over seeds; lower is better)

| method | interpolation | few data | extrapolation | steep fronts | mean rank |
|---|---|---|---|---|---|
| MLP | 0.0008739 ± 0.0001191 (#1) | 0.001412 ± 0.0005484 (#1) | 0.00768 ± 0.0006017 (#1) | 0.01022 ± 0.0008768 (#1) | 1 |
| Gaussian process | 0.0051 ± 3.249e-10 (#2) | 0.005302 ± 6.113e-09 (#2) | 0.01713 ± 1.645e-09 (#2) | 0.0192 ± 2.555e-09 (#2) | 2 |
| Polynomial ridge | 0.01763 ± 0 (#3) | 0.0175 ± 0 (#3) | 0.04942 ± 0 (#3) | 0.05249 ± 0 (#3) | 3 |
| FNO | 0.04741 ± 0.003578 (#4) | 0.03831 ± 0.0006361 (#4) | 0.1199 ± 0.006522 (#4) | 0.1244 ± 0.006596 (#4) | 4 |
| DeepONet | 0.04946 ± 0.005351 (#5) | 0.04213 ± 0.00493 (#5) | 0.1261 ± 0.007614 (#5) | 0.1305 ± 0.007584 (#5) | 5 |
| PINN | 0.2012 ± 0.02147 (#6) | 0.1967 ± 0.02606 (#6) | 0.3598 ± 0.02055 (#6) | 0.3669 ± 0.02003 (#6) | 6 |

## Cost and physics

| method | train s | inference s/query | model bytes | parameters | constraint violations | PDE residual / data |
|---|---|---|---|---|---|---|
| Polynomial ridge | 0.006031 | 0.0002951 | 2.892e+04 | 2304 | abs(u) <= 1.005: 46.7 % | 1.061 |
| Gaussian process | 0.4208 | 0.0004246 | 1.202e+05 | 216 | abs(u) <= 1.005: 46.7 % | 1.018 |
| MLP | 10 | 0.0001582 | 1.973e+05 | 4.589e+04 | abs(u) <= 1.005: 17.8 % | 1.003 |
| FNO | 10.09 | 0.007251 | 2.093e+06 | 2.606e+05 | abs(u) <= 1.005: 35.6 % | 1.009 |
| DeepONet | 10 | 0.0008226 | 1.069e+05 | 2.112e+04 | abs(u) <= 1.005: 53.3 % | 1.044 |
| PINN | 10 | 0.0006204 | 7.594e+04 | 1.28e+04 | abs(u) <= 1.005: 0.0 % | 0.3297 |
| baseline: fdm_201 (explicit upwind / FTCS finite differences, 201 points, CFL 0.4) | — | 0.1027 | — | — | abs(u) <= 1.005: 0.0 % | 0.8024 |
|  error of the baseline (interpolation): 0.02087 | | | | | | |

## Not run

- Gradient-boosted trees: fields: one tree ensemble per grid point is not offered (use gp on POD modes or an operator)
- gnn: graph networks (MeshGraphNet) need mesh connectivity; parametric tables and regular grids have none. pp.benchmark runs MeshGraphNet on the airfoil mesh fields.
- r_oams: R-OAMS is not in the PINNeAPPle library.

## Dataset audit: fail

pinneapple_data.audit.audit_dataset (pinneapple check PC451-PC453)

- pass: finite — every value finite
- not-applicable: physics — no PDE stated for this dataset
- pass: constraint: abs(u) <= 1.005 — 0 of 48384 rows violate it
- pass: diversity.duplicates — train: 0 exact, 0 near copies of 54; test: 0 exact, 0 near copies of 30
- fail: diversity.effective — effective sample count 1 of 54 (no generator stated: only a collapse to one pattern is flagged)
- pass: leakage — 0 of 30 test samples are (near) copies of train samples (closest: 0.0349 x the median train spacing)
- not-applicable: units — no unit metadata
