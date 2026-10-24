Day 18 of 30: Model Calibration Lab

Heat-transfer coefficients, material properties and boundary conditions are rarely known well enough. Fitting them to measurements by hand, or with a spreadsheet solver, gives numbers without error bars. It also gives numbers for parameters the data cannot determine, and the fit is never checked on data it did not see.

Model Calibration Lab fits model parameters to your measurements: a built-in physical model, your own formula, or a table of your solver runs. It returns 95 % intervals from the Jacobian and from a bootstrap, before/after plots and residuals, and an error on held-out measurements.

In the video, an aluminium part cools in air. From 31 thermocouple readings it calibrates h = 10.96 W/m²K, 95 % [9.14, 13.1] (truth 11), and T0 = 93.99 °C (truth 94). On the next 8 readings, which the fit never saw, the RMSE drops from 18.8 to 0.332 °C, close to the 0.3 °C sensor noise. The reduced χ² is 0.70, and a Bayesian estimator on the same data agrees.

Then a pin fin with h and k fitted from temperatures alone. The app reports both as not estimable and prints no interval, because the data only fix their ratio. The fit still predicts the held-out readings (5.6 → 0.41 °C), and the page points to the identifiability lab to find the measurement that would separate them.

Downloads: report, StudyResult, ModelRef with held-out metrics, CSV tables.

Live: https://calibration.pinneapple.org
GitHub: https://github.com/pinneapple-labs/pinneapple (apps/calibration_lab)

#Engineering #HeatTransfer #ModelCalibration #UncertaintyQuantification #InverseProblems #Simulation #Python
