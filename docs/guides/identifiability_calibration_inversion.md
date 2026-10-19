# Identifiability, calibration and inverse problems

One core in `pinneapple_analysis.inverse_problems` answers three questions in order:

1. **Can these parameters be recovered from these measurements?** `analyze_identifiability`. Run it before the
   experiment, when only the sensor positions and the noise are known.
2. **What are the parameters, and how sure are we?** `calibrate`. It fits the measurements, gives confidence
   intervals only where the data determine the parameter, and checks the fit on held-out measurements.
3. **Which inversion method should I trust for this problem?** `compare_methods`. It runs least squares, an adjoint
   gradient, automatic differentiation, Bayesian regularisation and a PINN on the same data, and ranks them by
   held-out error.

The Parameter Identifiability Lab, the Model Calibration Lab and the Inverse Model Studio apps are built on these
calls.

## Models, parameters, measurements

| Object | What it holds |
|---|---|
| `Parameter(name, value, lower, upper, unit, log=None, fixed=False, truth=None)` | One unknown. Positive parameters are analysed in `ln` (a sensitivity is then per 100 % change). `truth` is used only to check synthetic studies. |
| `Observations(inputs, sigma, values=None, observable=None)` | One row per measurement: inputs such as position or time, the noise `sigma`, the measured value (or none yet), and which observable the row measures (`"T"`, `"q_base"`, ...). `Observations.from_frame(df, ...)` reads a table loaded by `pinneapple_data.data_health.load_table` (CSV, Excel, Parquet, JSON, sensor exports, decimal commas). |
| `ForwardModel` | `predict(theta, inputs, observable)`. Ready-made models: `PinFinModel` (analytic fin), `RodModel` (finite-volume fin with h(x) zones: exact sensitivities, one-solve adjoint, torch twin), `PlateModel` (the Inverse Heat Lab's 2D plate), `LumpedCoolingModel`, `ArrheniusModel`. Your own model: `ExpressionModel("T_air + (T0 - T_air)*exp(-k*t)", ...)` (parsed safely, nothing is passed to `eval`), `CallableModel(fn, ...)` (any Python function or solver interface), or `TabulatedModel(table, ...)` (simulated runs interpolated over the parameters). |

`model_from_spec({...})` builds any of these from JSON. `get_preset(name)` returns ten problems with known answers,
for example `fin_h`, `fin_h_k`, `fin_h_k_flux`, `rod_h_field`, `rod_base_bc`, `plate_h`, `cooling_h`, `cooling_h_C`
and `arrhenius_one_T` / `_two_T`. Their measurements are synthetic: the model at the stated truth plus Gaussian
noise from a fixed seed.

## 1. Identifiability before the experiment

```python
import numpy as np
import pinneapple_analysis.inverse_problems as inv

fin = inv.PinFinModel(d_mm=5, length_mm=50)
params = fin.default_parameters()
params[1].fixed = False                                   # k unknown as well as h
sensors = inv.Observations({"x_mm": [10, 20, 30, 40, 50]}, sigma=0.5,
                           observable=np.array(["T"] * 5, dtype=object))
report = inv.analyze_identifiability(fin, sensors, params)
print(report.summary)
```

The output: *0 of 2 parameters identifiable. Moving h up and k up together leaves the predictions unchanged: the
data fix h/k, not h and k separately. Most informative extra measurement: q_base at x_mm=0.* The fin temperatures
depend on h and k only through h/k. The base heat flow scales with sqrt(h k), so a heat-flux meter makes both
identifiable.

The report combines four methods:

- **Fisher information** (rank, condition number, Cramér-Rao bounds, expected correlations). The eigen-directions
  with zero or small eigenvalues are reported as ambiguities, with the combination the data do fix.
- **Local sensitivity** of every measurement to every parameter, and the **leverage** of each measurement (its share
  of the identifiable information).
- **Profile likelihood** (Raue et al., 2009) with 95 % intervals. A profile that rises only because another
  parameter reaches its bound is reported as such, not as a closed interval.
- **Posterior by adaptive Metropolis** (flat priors inside the bounds) for cheap models.

It then gives a **D-optimal design**: from candidate measurements (other positions, times, temperatures or another
observable), the greedy picks that most increase `log det F`, with the rank and the bounds each pick would give.

Limits: the Fisher rank is a local numerical test at one point, not a symbolic proof of global identifiability. The
profile and the posterior are only as good as the bounds you give.

## 2. Calibration

```python
b = inv.get_preset("fin_h_k_flux").build()
cal = inv.calibrate(b["model"], b["observations"], b["parameters"], holdout=b["validation"])
```

With the heat-flux meter, the run above gives h = 25.17 W/m²K (Jacobian 95 % interval 23.74 to 26.68, bootstrap
24.81 to 25.49) and k = 16.13 W/m·K (15.16 to 17.16, bootstrap 15.88 to 16.38). The truth is 25 and 16. The held-out
RMSE is 0.41 °C for a sensor noise of 0.5 °C. Without the meter (`fin_h_k`), both parameters come back
**not estimable**, with no interval and the reason.

- Identifiability is checked at the calibrated values. A non-identifiable parameter gets no interval.
- Jacobian intervals use the stated sigma (`absolute_sigma=True`, a sensor specification). They are widened by
  sqrt(chi2/dof) when the residuals are larger than the stated noise, and never narrowed. Pass `absolute_sigma=False`
  when sigma is only a weight.
- Bootstrap intervals come from residual resampling and refitting.
- Held-out measurements are never fitted. They can be given explicitly, as a random fraction, or as the last rows by
  an input (`holdout_by="t_s"`, an extrapolation test).
- `warnings` lists a parameter on its bound, a reduced chi-square far from 1, a held-out error much larger than the
  fit error, and a missing validation set.

## 3. Comparing inversion methods

```python
out = inv.compare_methods(inv.get_preset("rod_h_field").problem(),
                          ["least_squares", "adjoint", "autodiff", "bayesian"])
```

This preset reconstructs a convection field h(x) in four zones and the temperature field T(x) from twelve
thermocouples. Every method recovers the truth inside its 95 % interval. Held-out RMSE: 0.176 °C for least squares,
adjoint and autodiff, and 0.180 °C for the Bayesian MAP, whose smoothness prior pulls the zones together. Forward
model evaluations: 11 (least squares, exact sensitivities), 39 (adjoint, one extra solve per gradient whatever the
number of zones), 48 (autodiff) and 3012 (Bayesian, of which 3000 are the MCMC chain).

| Method | Needs | Uncertainty |
|---|---|---|
| `least_squares` | any model | Gauss-Newton interval |
| `adjoint` | `adjoint_gradient` (e.g. `RodModel`) | Gauss-Newton interval |
| `autodiff` | `predict_torch` | Gauss-Newton interval |
| `bayesian` | any model (MCMC if cheap) | posterior interval, flagged as set by the prior for non-identifiable parameters |
| `pinn` | `pinn` (the Inverse Heat Lab fin and plate; h only) | none from one run |

`available_methods(problem)` says which methods apply and why the others do not. With a known truth, every solution
reports its parameter and field errors and `truth_recovered`. That flag is true when the truth lies inside the
interval of every identifiable parameter, or within 5 % when the method gives no interval.

Measured on the presets (synthetic data, same seeds): the Inverse Heat Lab plate PINN (`plate_h`, 1500 steps,
56 s on one CPU core) gave h = 14.0 W/m²K for a true 15. Least squares on the finite-volume model gave 14.96.
