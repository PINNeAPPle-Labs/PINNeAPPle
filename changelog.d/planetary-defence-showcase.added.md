- **Planetary defence is now an animation, a surface and a budget, not two static figures** (#381, PR #2546 follow-up).
  `pinneapple_physics.astro.defence` gained the accelerations the answer depends on -- the Yarkovsky `A2` term and
  solar radiation pressure in the propagator (`smallbodies.Trajectory(..., a2_au_d2=, srp=, perturbers=)`), the
  `perturbation_budget` that drops each perturber in turn and reports what the 2029 Apophis flyby does, the
  `impact_probability` and `jpl_miss_sigma_km` that turn a miss distance into a probability against JPL's own
  1-sigma, and `ensemble_arcs`, which flies every draw and keeps the trajectories so that a cloud drawn from it is
  the ensemble and not a drawing of one. `examples/planetary_defence/apophis_deflection.py` writes
  `figures/impact_probability.png` (the table read as the probability that the body hits, against impactor mass and
  warning time) and `figures/planetary_defence.gif` (the ensemble widening year by year, the impactor arriving, the
  encounter plane to scale with the Earth on it), and corrects the JPL comparison in the manifest: the bundled record
  says 37 724 km, not the ~31 000 km it claimed, so the model's 632 km is inside JPL's 697 km 3-sigma.
