- `pinneapple_analysis.dynamics.first_passage`: first-passage and hitting times, guide
  `docs/guides/dynamics_first_passage.md`.
  - `FirstPassageAnalyzer`: deterministic crossing of a trajectory (irregular grids, gaps, several thresholds,
    level sets), Kaplan-Meier distributions from observed trajectories with their own censoring windows, and
    Monte Carlo of an `SDE` with a Brownian-bridge correction and reflecting walls.
  - `dt_convergence` and `sensitivity`, the latter covering parameters and noise hypotheses with common random
    numbers.
  - `HittingTimeDistribution`: survival function with Greenwood bands, `P(tau <= T)` with Wilson or
    Kaplan-Meier intervals, quantiles, prediction intervals, and a restricted mean labelled as such.
  - Closed forms: inverse Gaussian, interval exit (probability, mean, survival series) and the
    Ornstein-Uhlenbeck mean first-passage time.
  - Benchmarks: `python -m pinneapple_analysis.dynamics.benchmarks first_passage`.
