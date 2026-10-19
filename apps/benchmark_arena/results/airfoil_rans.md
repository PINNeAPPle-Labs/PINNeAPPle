# Airfoil Cl and Cd from OpenFOAM RANS (CST shapes, Re 4e6)

Run 2026-10-17T11:09:17+00:00 on Intel(R) Xeon(R) Processor @ 2.80GHz (4 logical CPUs, torch 2.14.1+cu130), 122 s in total. Dataset sha256 `d8653c66678b8ac6`, 374 samples (scalar), inputs T0, T1, T2, M0, M1, M2, alpha_deg, outputs Cl, Cd.

## Protocol

- Metric: nrmse — RMSE / std of the training targets per output, averaged over outputs and held-out samples.
- Budget per training run: 10 wall_s; seeds [0, 1, 2]; split seed 0.
- one split shared by every method and seed; the model seed varies, the data does not.
- Stopping: iterative methods: until the budget is spent, best of 10 validation checkpoints (evaluation time not charged); one-shot fits: fitted once, time reported.
- Hyper-parameters: each method's defaults in pinneapple_analysis.surrogate.methods; no per-method search.
- One-shot fits (ridge, gp, gbt) use the time they need, which can be less than the budget; their train_s is that time.

## Regimes

- **interpolation**: random held-out samples inside the training distribution (206 training, 64 evaluated)
- **few data**: 52 of 206 training samples (25%); same held-out samples as interpolation (52 training, 64 evaluated)
- **extrapolation**: the 56 samples farthest from the centre of the input space (normalised distance > 1.17); training covers the interior (206 training, 56 evaluated)
- **geometry change**: 15 of 100 groups never seen in training (56 samples) (206 training, 56 evaluated)
- **steep fronts**: not run — scalar outputs: no field to have a front

## Rankings (mean ± std over seeds; lower is better)

| method | interpolation | few data | extrapolation | geometry change | mean rank |
|---|---|---|---|---|---|
| Gaussian process | 0.01338 ± 7.041e-08 (#2) | 0.03223 ± 0.005109 (#1) | 0.03961 ± 2.509e-07 (#1) | 0.02209 ± 1.568e-06 (#1) | 1.25 |
| MLP | 0.01231 ± 0.001214 (#1) | 0.04927 ± 0.001937 (#2) | 0.04419 ± 0.003266 (#2) | 0.02308 ± 0.002725 (#2) | 1.75 |
| Polynomial ridge | 0.0653 ± 0 (#3) | 0.1219 ± 0 (#3) | 0.1117 ± 1.7e-17 (#3) | 0.09184 ± 1.7e-17 (#3) | 3 |
| Gradient-boosted trees | 0.07534 ± 0.001666 (#4) | 0.1469 ± 0.002996 (#4) | 0.152 ± 0.004698 (#4) | 0.09421 ± 0.005443 (#4) | 4 |

## Cost and physics

| method | train s | inference s/query | model bytes | parameters | constraint violations | PDE residual / data |
|---|---|---|---|---|---|---|
| Polynomial ridge | 0.08461 | 0.0002835 | 1974 | 74 | Cd > 0: 0.0 % | — |
| Gaussian process | 3.992 | 0.0003236 | 7.218e+05 | 1854 | Cd > 0: 0.0 % | — |
| Gradient-boosted trees | 0.4945 | 0.0007821 | 1.423e+06 | 2.333e+04 | Cd > 0: 0.0 % | — |
| MLP | 10 | 0.0001311 | 4.045e+04 | 8962 | Cd > 0: 0.0 % | — |

## Not run

- FNO: FNO maps parameters to a field; the outputs here are scalars
- DeepONet: DeepONet maps parameters to a field; the outputs here are scalars
- PINN: a PINN needs a field over coordinates (the outputs here are scalars)
- gnn: graph networks (MeshGraphNet) need mesh connectivity; parametric tables and regular grids have none. pp.benchmark runs MeshGraphNet on the airfoil mesh fields.
- r_oams: R-OAMS is not in the PINNeAPPle library.

## Dataset audit: pass

pinneapple_data.audit.audit_dataset (pinneapple check PC451-PC453)

- pass: finite — every value finite
- not-applicable: physics — no PDE stated for this dataset
- pass: constraint: Cd > 0 — 0 of 326 rows violate it
- pass: diversity.duplicates — train: 0 exact, 0 near copies of 206; test: 0 exact, 0 near copies of 120
- pass: diversity.effective — effective sample count 6.37 of 206 (no generator stated: only a collapse to one pattern is flagged)
- pass: leakage — 0 of 120 test samples are (near) copies of train samples (closest: 0.0442 x the median train spacing)
- not-applicable: units — no unit metadata
