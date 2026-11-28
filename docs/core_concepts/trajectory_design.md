# Interplanetary trajectory design: patched conics, low-thrust legs and evolutionary search

`pinneapple_physics.astro.trajectory_design` answers the three questions a preliminary mission design asks, in the
order it asks them:

1. **What does it cost to fly from one planet to another in a given time?** `lambert` solves Lambert's problem in
   universal variables -- one formulation for elliptic, parabolic and hyperbolic arcs -- and `propagate_universal`
   flies the resulting conic anywhere along it.
2. **What does the whole sequence cost?** `MGAProblem` decodes `[t0, T1, ..., Tn]` into planet states and Lambert
   arcs and adds up the departure impulse, the powered flybys and the arrival insertion. `cassini1()` is the ESA GTOP
   benchmark, so a number from this package can be put next to a published one.
3. **Is the answer a mission, or only a solution of the conic problem?** `verify_design` re-flies the design in the
   restricted N-body field and reports the miss at every encounter, in kilometres, planet radii and Hill radii.

```python
import numpy as np
from pinneapple_physics.astro.trajectory_design import (
    CASSINI1_BEST_X, GTOPEphemeris, cassini1, map_elites, optimise, verify_design)

problem = cassini1(ephemeris=GTOPEphemeris())            # the ephemeris the benchmark is stated in
design = problem.decode(np.array(CASSINI1_BEST_X))
print(design.dv_total_km_s)                              # 4.9309 km/s (published: 4.9307)
print(design.dv_components)                              # where the budget goes: departure, flybys, insertion
print(verify_design(design, ephemeris=problem.ephemeris).verdict)     # "conic-only"

search = optimise(cassini1(), method="scipy-de", budget=20_000, seed=0)
print(search.summary())                                  # the measured best and the gap against the published one
print(map_elites(cassini1(), budget=1_200, seed=0).fill)              # share of the behaviour map that filled
```

Units are km, s and km/s throughout; the planets come from the JPL approximate elements (`AnalyticEphemeris`, the one
to design with) or from the ESA ACT table the GTOP numbers are quoted in (`GTOPEphemeris`), and every report quotes
`.source`.

## Reproduced, and only approached

Every number below was measured on this checkout, and `tests/test_trajectory_design.py` is what keeps measuring it.

| Claim | Declared | Measured here | Status |
|---|---|---|---|
| Lambert, 90° on a circular orbit | `\|v\| = sqrt(mu/r)`, tangential | agrees to 1e-9 relative | reproduced |
| Cassini1's Venus-to-Venus leg (0.008° of transfer angle, 449.4 d) | one full revolution, `z = 39.4772` | `z = 39.4772`, transfer angle > 359°, terminal miss 1e-5 of the radius | reproduced |
| Cassini1 optimum, scored in the ACT ephemeris | 4.9307 km/s (GTOP database) | **4.9309 km/s** (2e-4 high) | reproduced |
| The same decision vector in the modern JPL elements | -- | 5.8107 km/s of impulse, 0.856 km/s of pericentre penalty on top | declared: the two ephemerides are not interchangeable |
| A cold differential evolution finding that optimum | -- | 5.3033 km/s, **7.6 % above it** (247 296 sequences) | *not* reproduced: the gap is the result of §The search |
| A search seeded within 1 % of the published design | -- | 4.9306 km/s | reproduced: the landscape holds the optimum |

The 2e-4 km/s on the objective is the ACT table evaluated in double precision here against the single-precision
toolbox the published number comes from: smaller than the 5e-3 the test allows and 1 800 times smaller than the
0.37 km/s the search leaves.

## The objective: what a patched-conic budget actually charges

`CASSINI1_BEST_X` (`[-789.8117, 158.302027105278, 449.385873819743, 54.7489684339665, 1024.36205846918,
4552.30796805542]`, MJD2000 and days) decoded in the ACT ephemeris:

| Term | km/s |
|---|---|
| departure from the Earth (`\|v_sc - v_earth\|`) | 2.755 |
| powered flyby 1, Venus (`v_inf` 4.53 -> 6.77 km/s, turn 77.4°, `rp` = 6 351.8 km = its minimum) | 1.091 |
| powered flyby 2, Venus (`v_inf` 6.77 -> 7.72 km/s, turn 48.7°) | 0.616 |
| powered flyby 3, Earth (`v_inf` 13.73 km/s unchanged, turn 27.5°) | 0.000 |
| powered flyby 4, Jupiter (`v_inf` 6.10 km/s unchanged, turn 106.9°, `rp` 834 075 km) | 0.000 |
| Saturn orbit insertion (108 950 km x 0.98) | 0.470 |
| pericentre penalty | 0.000 |
| **total** | **4.9309** |

The three terms a patched-conic budget is easy to get wrong, and what this module does instead:

* **A gravity assist is not a velocity mismatch.** The planet turns `v_inf` for free; only the *change of its
  magnitude* is bought, at the pericentre of the flyby hyperbola where the spacecraft is fastest,
  `dv = |sqrt(vout**2 + 2 mu/rp) - sqrt(vin**2 + 2 mu/rp)|` with `rp` the pericentre that turns `v_inf` through the
  angle the two Lambert legs ask for (`powered_flyby`). Charging `|v_out - v_in|` instead -- the naive reading -- and
  the Cassini1 optimum comes out a factor of six high.
* **The arrival is an insertion, not a stopping burn.** `capture_dv` takes the incoming hyperbola down to the stated
  ellipse at the same pericentre; charging `|v_inf|` instead charges arriving dead.
* **The flyby pericentre is a constraint.** `MGASequence.flyby_rp_penalty` charges 0.01 km/s per kilometre below the
  minimum for the terrestrial planets and 0.001 for Jupiter, exactly as `mga.cpp` does, so a design that turns too
  hard at too low an altitude is expensive instead of free. The published optimum sits *on* the Venus pericentre
  (6351.8 km), which is what an optimum against a constraint looks like.

The other two readings are kept as `flyby_cost="difference"` and `arrival_cost="v_inf"` because published numbers do
not always say which one they mean; `flyby_cost="powered"` and `arrival_cost="auto"` are the GTOP ones and the
default. `tests/test_trajectory_design.py` asserts the difference between the three on the same design.

## The search: what a real budget reaches, and the gap it leaves

The objective is verified; the search is the hard part. A cold differential evolution on the ACT ephemeris, seed 0,
all measured on this checkout:

| Search | Sequences | Wall time | Best, km/s | Against 4.9307 |
|---|---|---|---|---|
| DE, Latin hypercube, `popsize=12, maxiter=120` | 8 712 | 1.2 s | 11.0319 | +124 % |
| DE, Latin hypercube, `popsize=40, maxiter=1500` | 360 240 | 82.6 s | 12.5402 | +154 % |
| DE, Sobol, `popsize=40, maxiter=1500` | 247 296 | 47.4 s | **5.3033** | **+7.6 %** |
| DE, Halton, `popsize=60, maxiter=1000` | 360 360 | 75.5 s | 5.3033 | +7.6 % |
| DE, Sobol, `popsize=100, maxiter=500` | 513 024 | 88.4 s | 5.3033 | +7.6 % |
| DE, Sobol + L-BFGS-B polish | 247 408 | 37.4 s | 5.3033 | +7.6 % |
| DE seeded with the published design +-1 % of the box | 8 712 | 1.3 s | 4.9306 | +0.0 % |
| DE seeded with the published design +-5 % of the box | 8 712 | 1.1 s | 5.3033 | +7.6 % |

Three things follow, and they are the honest reading of the landscape:

* **The objective is not the problem.** Seeded within 1 % of the box around the published design, an 8 712-sequence DE
  walks back onto 4.9306 km/s -- the published optimum, to 1e-4. The landscape contains it.
* **The basin is narrow.** At 5 % of the box the same run stops one optimum over, at 5.3033 km/s, and every
  space-filling population tried lands on that second optimum whatever the population, the budget or the polish. A
  uniform sample of 20 000 designs contains one point below 10 km/s, and it is inside the published design's
  neighbourhood in the first four coordinates at once.
* **The gap is therefore coverage, not tuning.** The cold search reaches the neighbourhood and stops 7.6 % short;
  closing the last 0.37 km/s is a multi-start / restart question (a DE from a few dozen seeded starts, seconds each),
  not a different objective, and nothing here weakens the objective to pretend otherwise.

`optimise(budget=600)` uses the library's own drivers, whose budget is counted in *designs* -- at 600 designs they are
a baseline, not a search, and the three report 43.5755 km/s (`random`), 31.4777 (`scipy-de`) and 21.9844 (`nsga2`) on
the JPL ephemeris in 0.2 s each. The numbers above are the ones a real budget produces.

The two search tests are the record: `test_cassini1_search_reports_a_measured_gap_within_a_short_budget` pins the
1.2 s run at its measured 11.03 km/s, `test_cassini1_seeded_search_converges_back_to_the_published_optimum` pins the
seeded return to 4.9306, and the 47 s run that reaches 5.3033 is `test_cassini1_search_reaches_the_published_neighbourhood`,
which only runs with `PINNEAPPLE_SLOW=1`:

```bash
PINNEAPPLE_SLOW=1 PYTHONPATH=. python -m pytest tests/test_trajectory_design.py -q    # ~60 s, 23 tests
PYTHONPATH=. python -m pytest tests/test_trajectory_design.py -q                      # ~12 s, one test skipped
```

## MAP-Elites: a catalogue of missions, not one optimum

`map_elites` runs the library's `QualityDiversitySearch` over three *outcomes* of the trajectory -- flight time,
impulse budget, arrival speed -- with the behaviour ranges measured on a probe of the design box, not declared.
Measured (700 sequences, 6 x 6 x 5 grid, JPL ephemeris, 0.24 s): **79 of 180 cells filled (44 %)**, 8 generations,
cheapest cell 22.89 km/s, ranges 1560-9270 d, 0-573 km/s, 0-13 km/s. Every row of `MapElitesResult.cells()` is the
archive's own record -- `cell`, `value`, `design`, `descriptors`, `condition` -- so one elite per cell is a property of
the archive and not of the report.

## The low-thrust leg

`SimsFlanaganLeg` is Sims & Flanagan's model: bounded impulses at segment midpoints, mass falling with the flow of the
engine, and `design_leg` closing the leg by Gauss-Newton and then cutting the impulse with L-BFGS-B under a growing
closure penalty. Measured on an Earth-to-Mars leg:

| Leg | Impulse ideal | Leg's impulse | Closure | Propellant |
|---|---|---|---|---|
| 6 segments, 5 kN chemical, 300 d | 30.5895 km/s | 30.5895 km/s | 3.4e-7 km | 0.0 kg |
| 8 segments, 0.5 N ion / 1000 kg, free departure, 400 d | 23.3063 km/s | 23.3063 km/s | 6.3e-7 km | 0.0 kg |
| 8 segments, same engine, **fixed** departure, 400 d | 23.3063 km/s | 19.8271 km/s | 2.3e+3 km | 493.8 kg -- does not close |

The high-thrust leg is the impulsive arc to the last digit, which is the limit the model must reproduce. With a free
departure the ion leg is also the impulsive arc: the launcher supplies the 23.3 km/s and the engine only corrects, so
the low-thrust penalty does not appear -- and it should not, that is what a free departure means. Fix it and the
0.5 N engine's 17.28 km/s ceiling over 400 days is below the 23.31 km/s the leg needs: 493.8 kg of propellant, 2 260
km of miss, `converged=False`. The model says which of the two it was asked for; a report has to say it too.

## The N-body trust check

`verify_design` re-flies the design with the Sun and the eight planets as point masses, takes their states from the
same ephemeris the design used, and compares the reached position and relative velocity at every node. The verdict is
a statement about the conic approximation, not about the mission:

| Design | Verdict | Largest miss | Correction impulse | Ephemeris error | Cost |
|---|---|---|---|---|---|
| Published Cassini1 (E-V-V-E-J-S, 6 239 d) | `conic-only` | 1.39e9 km (Venus missed by 3.8e6 km) | 167.5 km/s | 20.3 km | 1.3 s |
| One Earth-to-Mars leg, 90 d | `needs-correction` | 4.1e4 km (inside the Hill spheres) | 0.0 km/s -- one leg has no intermediate node to re-target | 1.3 km | 0.1 s |

Two defects were found here and are fixed; both matter for anyone reading a `TrustReport`:

* **The departure node is the planet's own centre**, so the point-mass term is `0/0` at the first evaluation of
  *every* design and the whole propagation came back `nan` from its first call.
* **The departure body is now handed over**: its pull is switched off inside its own Hill sphere, and from the moment
  the spacecraft leaves that sphere it pulls like any other planet. A patched conic *has* no dynamics at the departure
  node -- the state is the planet's own position with only `v_inf` on top of its velocity -- and at the Earth's centre
  a 2.755 km/s excess leaves the spacecraft *bound*: the propagation resolved an orbit inside the planet at 43 895
  steps per 200 days, which is where the twelve minutes of this test file used to go. Every other body pulls
  everywhere, so a flyby the design relies on is integrated as the hyperbola it is; the flyby the Cassini design does
  not fly is judged where the conic defines it, by the miss and by the `v_inf` the path arrives with.

The interpolation error is measured, not assumed: the ephemeris is evaluated on a grid and interpolated cubically, and
`_EphemerisPath.error_km` re-evaluates the exact ephemeris at the midpoints of that grid (20.3 km on the 6 239-day
span, 1.3 km on the 90-day one). It is reported with every verdict, and the trust check is not claimed to be more
accurate than it is.

## Where the numbers live

* `tests/test_trajectory_design.py` -- the Lambert solve, the flyby and capture terms, the published optimum, the
  seeded and cold searches, MAP-Elites, the Sims-Flanagan legs and both trust checks. The default suite is 12 s (it
  was over ten minutes before the trust check stopped resolving an orbit inside the Earth); the one real search that
  needs a real budget -- 47 s, reaching 5.3033 km/s -- is behind `PINNEAPPLE_SLOW=1`.
* `PUBLISHED_BEST` in `gtop.py` carries the published 4.9307 km/s with its source (ESA GTOP database, Cassini1).
* The JPL approximate elements are Standish's Table 1 (https://ssd.jpl.nasa.gov/planets/approx_pos.html); the ESA ACT
  elements are the toolbox's own `Pl_Eph_An.cpp`; the Lambert formulation is Vallado (2013), algorithm 57, and the
  flyby term is the toolbox's `PowSwingByInv`.
