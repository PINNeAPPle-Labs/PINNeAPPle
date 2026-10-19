# Model Calibration Lab (day 18)

**Fit model parameters to real measurements, with intervals only where the data determine them, checked on
measurements the fit never saw.** Live at <https://calibration.pinneapple.org> (port 8101).

![Cooling curve: three parameters calibrated, the next 30 minutes predicted within the sensor noise](../../docs/assets/apps/d18-calibration-cooling.jpg)

## Pain

Material properties, heat-transfer coefficients and boundary conditions are uncertain, or differ from the ideal
model. Fitting them by hand, or with a spreadsheet solver, gives numbers without error bars. Worse, it gives numbers
for parameters the data cannot determine at all, and the fit is never checked on data it did not see.

## Input

- **Measurements**: a CSV, Excel, Parquet or JSON table, or a sensor export (decimal commas and `;` separators work).
  You map the columns to inputs, values, σ and observable. Rows can also be typed.
- **A model**, in one of three forms:
  - Built in: pin fin (analytic), fin with zone-wise h(x), 2D heat-spreader plate, lumped-capacitance cooling,
    Arrhenius kinetics.
  - Your formula, for example `T_air + (T0 - T_air)*exp(-h*A*t_s/C)`. It is parsed safely, never evaluated as code.
  - **Your simulated results**: a table of solver runs (parameter columns, probe columns, output), interpolated
    between runs.
- **The parameters to adjust**: start values, bounds, units and log scale. They can also be loaded from a
  `ParameterSpace` contract.
- **Validation**: held-out measurements (a separate table), a random fraction of the rows, or the last rows by an input
  such as time (an extrapolation test).

## Output

- **Calibrated parameters** with 95 % intervals from the Jacobian and from a residual bootstrap.
  - A parameter the measurements cannot determine is reported **not estimable**, with the reason and the combination
    the data do fix. The check uses the day-25 identifiability core at the calibrated values.
  - The page links to the Parameter Identifiability Lab, to find which measurement would make it estimable.
- **Before / after**: the model at the start values and at the calibrated values, against every measurement. It comes
  with the curve, the residuals with ±2σ, and measured vs model.
- **Validation**: fit and held-out RMSE, before and after, and warnings:
  - a parameter on its bound;
  - a reduced χ² far from 1;
  - a held-out error much larger than the fit error;
  - no validation at all.
- **The same data solved by other estimators**: Bayesian regularisation (MAP + MCMC), automatic differentiation,
  adjoint gradient, or a physics-informed network. They are ranked by held-out error, with truth recovery on the
  examples.
- **Reconstructed fields** where the model has them: the fin profile and the plate temperature map.
- **Downloads**: the report (JSON), a `StudyResult` (`study_type="calibration"`), a `ModelRef` of the calibrated model
  with its held-out metrics, and CSV tables of parameters and residuals.

## Integrations

- **REST API** at `/docs`. `POST /api/calibrate` queues a job, and `GET /api/jobs/{id}` returns its status, progress
  and result (shared job queue).
- **API keys** (`X-API-Key`), and **usage per caller** at `GET /api/usage`.
- **Tables**: CSV, Excel, Parquet, JSON and sensor exports.
- **Contracts**: `ParameterSpace` in; `StudyResult` and `ModelRef` out (`pinneapple_data.contracts`). The calibrated
  values and held-out errors can go on to uncertainty, sensitivity or optimisation studies.
- **Your own solver**:
  - This server does not run uploaded code or call external endpoints.
  - Upload its runs as a table, or run `pinneapple_analysis.inverse_problems.calibrate` on your machine with
    `CallableModel(your_solver)`. The API tab shows the code.

## Methods

All of them come from `pinneapple_analysis.inverse_problems` (guide:
[docs/guides/identifiability_calibration_inversion.md](../../docs/guides/identifiability_calibration_inversion.md)):

- **Estimation**: weighted least squares in scaled parameters (ln for positive constants), trust-region reflective
  inside the bounds. Robust losses are available (soft L1, Huber, Cauchy).
- **Jacobian intervals**: Student t. With `σ is the real sensor noise` (the default), the intervals are never narrower
  than the stated noise implies, and widened by sqrt(χ²/dof) when the residuals are larger. Untick it when σ is only a
  weight.
- **Bootstrap**: residual bootstrap (refit on ŷ + resampled standardised residuals), percentile interval.
- **Identifiability**: rank of the Fisher information at the calibrated values (day 25).
- **Comparison**: the inverse-studio estimators on the same fitted rows, ranked by held-out RMSE.

## What the examples show

Measured on this server's examples (synthetic measurements from a known truth plus Gaussian noise, fixed seeds;
95 % Jacobian intervals):

| Example | Calibrated (truth) | Held-out RMSE before → after |
|---|---|---|
| Cooling curve, extrapolated 30 min ahead | h 10.96 [9.14, 13.14] W/m²K (11); T0 93.99 (94); T_air 22.2 [12.1, 32.3] °C (23) | 18.8 → 0.33 °C (σ 0.3) |
| Fin, h and k with a heat-flux meter | h 25.16 [23.73, 26.67] (25); k 16.12 [15.15, 17.15] (16) | 5.6 → 0.41 °C (σ 0.5) |
| Fin, h and k from temperatures only | both **not estimable** (only h/k is determined); the fit still predicts | 5.6 → 0.41 °C |
| Plate, 2D finite volumes | h 14.96 [14.82, 15.10] (15) | 33.4 → 0.65 °C (σ 0.5) |
| Plate, table of 8 solver runs | h 14.84 [14.71, 14.97] (15): the interval **misses** the truth; interpolating between runs adds 0.43 °C of error at h = 15 that the interval does not include | 22.1 → 0.62 °C |
| Kinetics at 360 K and 390 K | Ea 74.4 [71.8, 77.0] kJ/mol (75); A 1.6e7 [7.2e6, 3.7e7] 1/s (2e7) | 0.71 → 0.011 |
| Fin, unknown wall temperature | T_base 108.7 [106.1, 111.3] °C (110); h 33.9 [31.6, 36.4] (35) | 33.9 → 0.59 °C (σ 0.3) |

The examples take 0.3 to 10 s. On the 1D fin, a PINN comparison (1500 steps) took 21 s on one core: h 24.88 for a
true 25, with a held-out RMSE of 0.405 °C against 0.410 °C for least squares.

## Limits

- **The model form is assumed right.** Intervals are statistical only. A missing physical effect biases every
  parameter, and only the reduced χ² and the held-out error reveal it, so hold out measurements.
- **A table of solver runs** adds an interpolation error that the intervals do not include (the plate-table example
  shows it). Use more runs near the answer, or a model that runs the solver.
- **Your own code is not run here.** Upload a table of runs, or use the library call on your machine.
- **Noise** is independent and Gaussian with the σ given; there is no model-discrepancy term.
- **The PINN** estimates h only (fin and plate), and a single run gives no interval.
- **Size**: at most 12 free parameters and 5000 measurements per calibration; bootstrap at most 500 refits.

## Monetization

Charged per calibration and per extra run. The metered units, per caller and month, are `calibrations`,
`method_runs` (each extra estimator on the same data) and `pinn_runs` (network trainings, the costly ones). Free
limits are set by `CALIB_FREE_*`. A team plan raises the limits, adds API keys for CI, and covers calibrations against
customer data with a validation report.

## Run it

```bash
pip install -e . -r apps/calibration_lab/requirements.txt
cd apps/calibration_lab && uvicorn calibration_lab.api:app --port 8101
# or: docker build -f apps/calibration_lab/Dockerfile -t calibration-lab . && docker run -p 8101:8101 calibration-lab
```

| Variable | Default | Meaning |
|---|---|---|
| `CALIB_USER` / `CALIB_PASSWORD` | unset | HTTP Basic login on everything but `/health` |
| `CALIB_MAX_JOBS` | 2 | calibrations running at once over all workers; the rest wait in the queue |
| `CALIB_FREE_CALIBRATIONS`, `CALIB_FREE_METHOD_RUNS`, `CALIB_FREE_PINN_RUNS` | 50, 200, 10 | free use per caller and month (`unlimited` lifts it) |
| `CALIB_ADMIN_TOKEN` | unset | `X-Admin-Token` to manage API keys without the login |
| `CALIB_IDENTIFIABILITY_URL` | https://identifiability.pinneapple.org | link to the Parameter Identifiability Lab |
| `CALIB_DATA_DIR` | `/data/calibration` | SQLite with jobs, keys and usage; uploads |

Tests: `pytest tests/test_app_calibration_lab.py`.
