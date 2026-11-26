- Planetary defence, measured against the mission that flew it (#381): `pinneapple_physics.astro.defence` carries the
  chain from the impulse to the guarantee. `delta_v` is the momentum a kinetic impactor delivers (`beta m v / M`), and
  `dart_beta` recovers the enhancement factor from the **published** DART numbers -- 570 kg at 6.1 km/s shortening
  Dimorphos' period around Didymos by 32 minutes gives `beta = 3.65` against the published 3.61, and feeding it back
  reproduces the -32.6 min change, so nothing downstream is a free parameter; `period_change_fraction` is the
  `dP/P = 3 dv/v` of an along-track impulse. On the orbit, `impulse_body` and `deflect` apply the nudge at a date and
  rebuild the osculating elements there (in the frame the bundled `SmallBody` stores, which is what makes a micrometre
  per second stay a micrometre per second), `miss_distance` measures the resulting encounter and `deflection_table`
  is the miss against impactor mass and warning time. On the decision, `yarkovsky_drift` turns JPL's `A2` into the
  semi-major-axis drift (-166 m a year for Apophis against the ~170 m/yr published), `sample_ensemble`,
  `success_probability` and `trust_horizon` turn the two uncertain parameters into the fraction of the ensemble that
  clears the Earth and the warning time beyond which the spread is wider than the planet -- after it, "we will miss"
  is not a guarantee whatever the model says. Measured on Apophis, whose 2029-04-13 encounter the shipped propagator
  reproduces at 38 356 km and 7.41 km/s from the bundled JPL elements: a 100 kg impactor five years out moves that
  miss by 21 km on a 5.4e10 kg body, and the same mass is 400 times more effective on a 50 m body, which is why the
  regime a real mission works in is a small target or a long warning. `examples/planetary_defence` runs the three acts
  and writes the figures and the numbers.
