- `pinneapple_analysis.dynamics.early_warning`: critical slowing down and early-warning indicators, guide
  `docs/guides/dynamics_early_warning.md`.
  - `EarlyWarningAnalyzer` / `CriticalSlowingDown`: detrending (local linear by default, Gaussian kernel or linear),
    rolling variance, CV, skewness, kurtosis, lag autocorrelation (pair-based on irregular grids), AR(1) recovery
    rate, low-frequency power and spectral exponent.
  - Kendall trends with a surrogate p-value (fitted AR(1), phase or IAAFT surrogates from
    `pinneapple_systems.time_series.predictability`), residual-trend and ADF/KPSS checks (statsmodels optional), and
    window / bandwidth / sampling sensitivity.
  - `TransitionDiagnostics` keeps indicator and prediction apart.
  - `ramp_series`: stochastic saddle-node, transcritical, pitchfork and OU ramps, and stable controls.
  - Benchmarks: `python -m pinneapple_analysis.dynamics.benchmarks early_warning`.
