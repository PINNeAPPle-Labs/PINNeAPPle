- Interplanetary trajectory design: `pinneapple_physics.astro.trajectory_design` solves Lambert's problem in
  universal variables (`lambert`, `propagate_universal`, one formulation for elliptic, parabolic and hyperbolic arcs,
  with the nearly-full-revolution long-way transfer that a Cassini Venus-to-Venus leg needs), scores a multiple
  gravity assist sequence the way the ESA GTOP toolbox does (`MGAProblem`, `cassini1`: the departure impulse, the
  powered flyby charged as the change of `|v_inf|` at the pericentre that turns it, the arrival as an orbit insertion
  and the pericentre minimum as a penalty — `CASSINI1_BEST_X` reproduces the published 4.9307 km/s as 4.9309 in the
  ACT ephemeris the benchmark is stated in), flies low-thrust legs the way Sims & Flanagan posed them
  (`SimsFlanaganLeg`, `design_leg`: bounded impulses per segment, mass falling with the flow, Gauss-Newton closure and
  an L-BFGS-B impulse cut), searches the design space with the library's own evolutionary machinery (`optimise` over
  NSGA-II / SciPy DE / random, `map_elites` over flight time, impulse and arrival speed), and re-flies the answer in
  the restricted N-body field (`verify_design`, `TrustReport`: the miss at every encounter in km, planet radii and Hill
  radii, the correction impulse to re-target them, and the verdict `reproduces-the-conic` / `needs-correction` /
  `conic-only`). The guide `docs/core_concepts/trajectory_design.md` states the reproduced-versus-declared split with
  the measured tables: the objective reproduces the published optimum, and a cold differential evolution reaches
  5.3033 km/s against it — 7.6 % above — which is reported rather than tuned away (#397).
- The N-body trust check of a patched conic: the departure body is handed over at its sphere of influence instead of
  being pulled from its own centre, where the design puts the spacecraft (the point-mass term was `0/0`, so the whole
  propagation came back `nan` from its first evaluation, and a spacecraft leaving the Earth's centre at 2.755 km/s is
  bound to it and was integrated as an orbit inside the planet). Every other planet still pulls everywhere, the
  acceleration is floored at each body's radius, and the ephemeris interpolation error of the check is measured and
  reported with every verdict (#397).
