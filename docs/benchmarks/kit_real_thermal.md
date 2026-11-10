# KIT thermal measurements benchmark

Status: **protocol only; no real-data training has been executed**.

Dataset: [KIT LLEC experimental thermal campaign, Zenodo v1](https://zenodo.org/records/21157950), DOI `10.5281/zenodo.21157950`, CC BY 4.0.

The public dataset contains minute-level indoor temperatures, environmental conditions, and measured heating variables. B3 includes delivered thermal power. Use the metadata to map *matching* room temperature and delivered thermal power measurements, not electrical input power as if it were thermal output.

## Research objective

Test whether PINNeAPPle physics-informed learning is more accurate on untouched experimental observations than a **calibrated** numerical thermal balance, and whether gains persist under temporal extrapolation, limited training data, sensor noise, and autonomous rollouts.

## Design

1. Acquire Exp1_tot.csv, Exp2_tot.csv, and Metadata.json and record their SHA256, provenance, units, missingness and temporal gaps.
2. Construct one-step thermodynamic baseline from `C dT/dt = UA*(T_out-T)+Q_delivered+Q_unmodelled`, then calibrate unknown coefficients on **training** data only. Compare the uncalibrated conventional engineering estimate where genuine engineering coefficients are available.
3. Train (a) data-only model, (b) physics-only model, (c) combined continuous-time thermal PINN, plus the calibrated numerical solver. Use native `pp.build_model('vanilla_pinn', ...)` or the PINNeAPPle model factory, and `pinneapple_lab.Experiment`/`RunContext` for tracked results.
4. Temporal split and entire-campaign holdout; use the exact same observations for numerical calibration and neural training. Validation data may select models but never influence test predictions through tuning.
5. Evaluate one-step **and** free-running predictions. Record test RMSE, MAE, relative L2, max error, physics residual, energy-balance diagnostics, model mismatch, extrapolation, training/inference time, and hardware.
6. Repeat at seeds 0–4 and training fractions 10%, 25%, 50%, 75%; bootstrap or otherwise quantify uncertainty at the campaign/window level, not independently across autocorrelated minute readings.
7. Preserve failed or inconclusive results, generate figure panels and markdown report, and include a numerical-calibration fairness audit.

## Implementation constraints

- No synthetic substitution if downloads fail. Fail with clear error.
- A discrete regularizer or pure neural one-step forecaster is **not a full PDE PINN**. Label approaches accurately.
- Physical consistency is not synonymous with low prediction error; calculate it separately.
- Do not claim physics-based superiority until test measurements and paired estimates have been checked.

Tracking issue: #1592.
