# Planetary defence: what a kinetic impactor buys, how far ahead the miss can be guaranteed, and what each
# acceleration in the field is worth

An asteroid's orbit is an uncertainty cloud that grows with time. A kinetic impactor nudges it, and the question a
decision-maker actually asks is not "what is the delta-v" but **"for how long ahead can we still guarantee a miss, and
what does it cost"**. This module answers with the numbers that can be checked against the missions and objects whose
answer is already known — and where a number is too small to matter, it says so with the number next to it.

## 1. The impulse, held to the mission that flew it

DART hit Dimorphos on 2022-09-26 with 570 kg at 6.1 km/s. The moonlet's orbital period around Didymos dropped from
11.92 h to 11.37 h — 32 minutes — and that is more than the impactor's own momentum can buy: the ejecta carried extra.

```python
from pinneapple_physics.astro.defence import dart_beta, delta_v, period_change_fraction

dart_beta()        # 3.65  (the published value is 3.61)
dv = delta_v(570.0, 6.1, 4.8e9, beta=dart_beta())      # 2.6 mm/s on Dimorphos
period_change_fraction(-dv, 0.174) * 11.92 * 3600 / 60  # -32.6 min (published: -32)
```

That is the calibration of everything below: `beta` is not a free parameter here, it is the one the mission measured.
A body of mass `M` hit by `m` at speed `v` gets `delta_v = beta m v / M`; on a circular orbit the period follows from
`dP/P = 3 dv/v` (the radius does not change at the impulse, so `da/a = 2 dv/v` and Kepler's third law triples it).

## 2. The nudge, on an object whose encounter is known

Apophis passes **38 356 km** from the Earth on 2029-04-13 at 7.41 km/s, from the bundled JPL elements. The same
record carries JPL's own uncertainty for that flyby — 0.000252172816 au with a 3-sigma range of
0.000247515–0.000256837 au, which is **37 724 km ± 232 km (1 sigma)**. The model is 632 km away from it: 2.7 sigma,
inside JPL's 697 km 3-sigma range, which is the repository's own gate (`smallbodies.validation_summary`).

```python
from pinneapple_physics.astro.defence import APOPHIS_2029_JD, deflection_table, jpl_miss_sigma_km, miss_distance
from pinneapple_physics.astro.smallbodies import load_body

jpl_miss_sigma_km()["sigma_km"]      # 232.4 -- 1 sigma of the flyby distance, from the record's 3-sigma range
body = load_body("apophis")
rows = deflection_table(body, APOPHIS_2029_JD - 5 * 365.25, APOPHIS_2029_JD,
                        masses_kg=(100.0, 1000.0, 10000.0), speed_km_s=6.1, target_kg=5.4e10)
```

Each row is `delta_v` in mm/s and the miss distance it produces. Measured, over the undeflected 38 356 km:

| impactor | delta_v | 1 year of warning | 5 years | 10 years |
|---|---|---|---|---|
| 30 kg | 0.012 mm/s | 38 357 km | 38 362 km | 38 363 km |
| 100 kg | 0.041 mm/s | 38 359 km | 38 377 km | 38 380 km |
| 300 kg | 0.122 mm/s | 38 363 km | 38 417 km | 38 428 km |
| 1 000 kg | 0.408 mm/s | 38 380 km | 38 560 km | 38 597 km |
| 3 000 kg | 1.223 mm/s | 38 428 km | 38 969 km | 39 078 km |
| 10 000 kg | 4.078 mm/s | 38 596 km | 40 401 km | 40 765 km |

The physics is unforgiving in a way worth seeing: on a 5.4e10 kg body even ten tonnes ten years ahead buy **2 409 km**
on a 38 356 km miss — a few tenths of the Earth's radius, not a guarantee. The same impulse on a 50 m body (1.3e8 kg)
on the very same orbit buys **87 263 km**: 1 000 kg there is 169 mm/s, and the miss is no longer an encounter. Which is
the honest summary of planetary defence — it is a problem of *small* targets and *long* warning times, and the useful
question about a big one is how early you have to go.

## 3. The table read as the probability that it hits

A miss distance is not a decision; the probability that the body hits is. It is the same table, read with the two
uncertainties that are already measured: the ensemble below gives the spread of the miss at each warning time, and
JPL gives the orbit's own 232 km. Added in quadrature, they are the sigma of a Gaussian whose mean is the table's miss,
and the probability that the true miss falls inside the Earth is `impact_probability`:

```python
from pinneapple_physics.astro.defence import impact_probability

impact_probability(miss_km, sigma_km)        # P(|miss + error| < R_earth), the 1-D reading of the b-plane
```

Two geometries are read off the same rows, and they are two panels of `figures/impact_probability.png`:

* **a body on a collision course** — the undeflected miss is the Earth's limb, so every kilometre the table bought is
  margin, and the surface is the probability that it still hits. This is the only case a deflection changes, and the
  surface falls across it by more than twenty orders of magnitude: from 0.50 with no deflection to
  **1.6e-23** for 10 000 kg with ten years of warning.
* **Apophis 2029** — the undeflected miss is the 38 356 km measured above, and the question is whether any cell walks
  it back. None does: the closest cell is still **156 sigma** from an impact, and the table's best cell buys 4 of them.

That is what the surface shows that the table did not: the table's 38 357 → 40 765 km reads like "nothing changes",
and in the probability it is a change of twenty-three orders of magnitude. It also shows the *floor* of the exercise —
the 232 km of orbit uncertainty is what stands between the impactor and a guarantee, and no mass in the catalogue
clears it.

## 4. What can be guaranteed

`beta`, the Yarkovsky `A2` and the area-to-mass ratio that sets solar radiation pressure are not known exactly, so the
orbit is an ensemble, and the two questions have the same shape:

```python
from pinneapple_physics.astro.defence import sample_ensemble, success_probability, trust_horizon

draws = sample_ensemble(n=6, a2_au_d2=-2.9e-14, a2_sigma=3e-15, beta=3.61, beta_sigma=0.35,
                        area_to_mass_m2_kg=1.54e-6, am_sigma=0.5, seed=0)
success_probability(misses_km)                      # fraction of the ensemble that clears the Earth
trust_horizon(warning_days=days, spread_km=spread)  # the warning time where the spread reaches the Earth's radius
```

Every draw is flown with *its own* parameters — `ensemble_arcs` integrates each member, so the `A2` draw is an
acceleration the body actually feels and not a column in a table.

The **trust horizon** is the point of the whole exercise: beyond it, "we will miss" is not a guarantee whatever the
model says, because the ensemble is wider than the planet. It is the same object as the trust horizon of a forecast,
applied to a deflection.

Measured on the Apophis deflection with `beta` at ±0.35, `A2` at ±3e-15 au/day^2 and the area-to-mass ratio at ±50 %
(6 draws): the spread of the miss is **<!--SPREAD-->**, against the Earth's 6 371 km radius, and the success fraction
is 1.00 at all three. The horizon function returns the last warning time it was given when the ensemble never reaches
the Earth — read it as "at least this far", not "exactly".

That is a result about *what this ensemble does not contain*, and §5 puts a number on it: the deflection parameters
move the miss by tens of kilometres, while the orbit solution JPL publishes for the same flyby is 232 km wide. The
deflection is not what limits the guarantee at this scale — the orbit is.

The Yarkovsky drift is the largest of the small accelerations: for Apophis (`A2 = -2.9e-14 au/day^2`) the module gives
**-166 m a year** of semi-major-axis drift, against the ~170 m/yr published — which is why `A2` is one of the uncertain
parameters and not a detail.

## 5. What each acceleration in the field is worth

`perturbation_budget` takes the field apart: it drops each perturber in turn, switches solar radiation pressure on at
the rock's own area-to-mass ratio and at ten times it, adds the Yarkovsky term, and repeats the run once at a tighter
integrator tolerance — that last row is the numerical floor the small rows have to be read against. One long
integration per row, over the whole arc, because an asteroid's trajectory is not reusable.

| term taken out or added | miss at 2029-04-13 | change | where the body was at the encounter |
|---|---|---|---|
| the field the module ships (Sun, 9 planets, Moon, 1PN) | 38 356 km | — | — |
| without Venus | 2 808 518 km | +2 770 162 km | <!--DR:venus--> |
| without the Earth | 2 193 147 km | +2 154 790 km | <!--DR:earth--> |
| without Jupiter | 1 288 361 km | +1 250 005 km | <!--DR:jupiter--> |
| without Saturn | 181 374 km | +143 017 km | <!--DR:saturn--> |
| without the Moon | 60 722 km | +22 366 km | <!--DR:moon--> |
| without Mars | 30 284 km | −8 072 km | <!--DR:mars--> |
| without Uranus | 31 168 km | −7 188 km | <!--DR:uranus--> |
| without Mercury | 43 800 km | +5 444 km | <!--DR:mercury--> |
| without Neptune | 35 631 km | −2 726 km | <!--DR:neptune--> |
| Yarkovsky, `A2 = -2.9e-14 au/day^2` | 38 029 km | −327 km | <!--DR:a2--> |
| solar radiation pressure at 10x (A/M = 1.54e-5) | 38 593 km | +237 km | <!--DR:srp10--> |
| solar radiation pressure at 1x (A/M = 1.54e-6) | 38 380 km | +24 km | <!--DR:srp1--> |
| the numerical floor (rtol 1e-11 → 1e-12) | 38 357 km | +0.7 km | <!--DR:floor--> |

Two things come out of it, and the first is the honest one about this whole exercise:

* **the 2029 miss is the result of the entire twenty-one-year arc, not of the last encounter.** Every perturber in the
  field is worth at least a few thousand kilometres, so none of them can be dropped, and the Earth itself — which pulls
  on Apophis at its earlier approaches as much as at this one — is worth two million. That is also why the model lands
  632 km from JPL's solution rather than on it: the sixteen perturbing asteroids a real impact solution uses are not in
  this field.
* **solar radiation pressure does not matter in this window, and here is the number.** At the rock's own area-to-mass
  ratio it moves the 2029 miss by **24 km**: 0.4 % of the Earth's radius, a tenth of JPL's own uncertainty, and thirty
  times the floor of the computation — real, measurable, and irrelevant to the decision. Ten times the pressure (a body
  a tenth of the width) is 237 km, which is where it starts to matter.
* The Yarkovsky term is −327 km, more than the uncertainty JPL quotes: the acceleration is worth including, and
  `ensemble_arcs` finds the same thing from the other side, since it is the largest term in the spread.

## 6. The animation

`figures/planetary_defence.gif` is the picture the deliverable asked for — the uncertainty cloud growing over the
years, the impactor arriving, and the approach that is not an impact — and it is made of the ensemble, not drawn:

* every dot is a member of `result["ensemble"]["arcs"]`, flown with its own `beta`, `A2` and area-to-mass ratio, so
  the widening is the members diverging;
* the top panel is the distance from the Earth falling over the ten years, one trajectory per member, with the
  Earth's surface drawn on it;
* the second is the cloud's width — each member's offset from the undeflected body, which grows from the 11 km it
  already has at the impulse (the `A2` draw acts on the whole arc) to <!--CLOUD--> km at the flyby. It is measured
  *along* the orbit, where the body is moving at 30 km/s: the cloud is long, not thick, and the miss distance it
  moves is thousands of times smaller;
* the third is the encounter plane to scale, Earth's disk and two dots — undeflected and deflected — with a zoomed
  inset against JPL's 1 sigma, because at the full scale the deflection is a few pixels.

## 7. What the example produces

`examples/planetary_defence/apophis_deflection.py` runs the four acts and writes `apophis_deflection.json` plus four
figures: `miss_vs_mass.png` (the table), `trust_horizon.png` (the spread against warning time), `impact_probability.png`
(§3) and `planetary_defence.gif` (§6). `--quick` runs the DART anchor and one deflection in under a minute; the full
study takes about 35 minutes, because it propagates a twenty-one-year arc once per table row, once per ensemble member
and once per acceleration in §5.

## 8. What this does not do

- The n-body field is the one the repository already ships (Sun, planets, Moon, the Sun's relativistic term, DE421).
  The 16 perturbing asteroids that a real impact solution uses are not there, which is the 632 km between the 38 356 km
  above and JPL's 37 724 km — inside the 697 km 3-sigma JPL publishes, and the first thing to fix if it has to be
  better.
- The impact probability is the **1-D** reading of the b-plane: the Gaussian of the miss distance, not the integral of
  a 2-D error ellipse over the capture cross-section. It is the right reading while the uncertainty is much smaller
  than the Earth (232 km against 6 371 km) and the wrong one for a cloud wider than the planet, which is exactly the
  regime a last-minute deflection is in — that case needs JPL Sentry's covariance and the full Rice distribution.
- The "body on a collision course" panel is a what-if, and says so: the table's rows applied to an undeflected miss
  of one Earth radius. It is not a published impact solution for any object.
- The impactor is a momentum impulse at one instant: no cratering, no ejecta modelling beyond the single `beta`.
- The uncertainty ensemble is normal and seeded; it is not a covariance-propagated orbit solution. What it carries
  (beta, `A2`, area-to-mass) are the deflection's parameters; the orbit's own covariance enters only through the JPL
  sigma of §3.
