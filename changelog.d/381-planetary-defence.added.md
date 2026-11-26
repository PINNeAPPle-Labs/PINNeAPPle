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
  reproduces at 38 356 km and 7.41 km/s from the bundled JPL elements: ten tonnes ten years ahead (4.08 mm/s) move
  that miss by 2 409 km, while the same nudge on a 50 m body (1.3e8 kg) on the same orbit moves it by 87 263 km --
  planetary defence is a problem of small targets and long warning times. The ensemble over beta (+-0.35) and A2
  (+-3e-15 au/day^2) spreads 2 km at one year, 18 km at five and 21 km at ten, against the Earth's 6 371 km radius:
  at this scale the deflection parameters are not what limits the guarantee, which is the honest result and the
  reason the guide says the dominant term for a real object is the orbit solution's own covariance, not carried here. `examples/planetary_defence` runs the three acts
  and writes the figures and the numbers.
