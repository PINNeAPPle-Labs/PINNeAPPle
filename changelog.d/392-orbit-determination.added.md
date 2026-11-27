- `pinneapple_physics.astro.orbit_determination` is orbit determination and space traffic: a numerical propagator
  (two-body + J2 + drag, with the atmosphere of `astro.decay` and an optional `storm` factor on the density), the
  analytic two-body flow in universal variables it is checked against, the position covariance by lead time (state
  transition matrix and Monte Carlo) with `trust_horizon` — the lead time at which the covariance reaches a stated
  tolerance — conjunctions through the propagated covariance in the encounter plane (`conjunction_probability`), and
  the trust-facing pieces: `regime_changes` flags a manoeuvre (a step in the semi-major axis) and a storm-driven drag
  regime (a step in its rate) that the quiet model does not explain, `tle_check` / `tle_measurement_check` flag a bad
  TLE fit (checksum, element ranges, and whether the propagation reproduces its own element set), and
  `LearnedTleCorrection` learns the systematic part of the SGP4 error from later TLEs, combined with SGP4 and the
  numerical propagator in a `PhysicsEnsemble` whose residual is the two-body equation residual of the predicted
  trajectory. Measured offline in `tests/test_astro_orbit_determination.py`: energy and angular momentum conserved to
  2e-11, the propagator within 1e-7 km of the closed form, the decay rate within 6 % of the orbit-averaged formula and
  a factor-3 density jump reproduced as a factor 3.15 of it, the covariance within 4 % of Monte Carlo and the
  along-track growth within a few percent of `3 dv t`, the collision probability within 1.5 sigma of a Monte Carlo and
  1e-9 of its closed form, and the learned correction taking a 30 % drag bias from 5.42 km to 0.14 km on a day it
  never saw. What needs the network — CelesTrak GP data and its archive, the F10.7 and Ap indices, the SGP4-VER
  suite — is declared rather than faked, together with the February 2022 Starlink case and the inputs it would take to
  reproduce it ([docs](docs/core_concepts/orbit_determination.md)).
