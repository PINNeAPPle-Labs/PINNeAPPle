# Planetary defence: what a kinetic impactor buys, and how far ahead the miss can be guaranteed

An asteroid's orbit is an uncertainty cloud that grows with time. A kinetic impactor nudges it, and the question a
decision-maker actually asks is not "what is the delta-v" but **"for how long ahead can we still guarantee a miss, and
what does it cost"**. This module answers with the numbers that can be checked against the missions and objects whose
answer is already known.

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

Apophis passes **38 356 km** from the Earth on 2029-04-13 at 7.41 km/s — the encounter the propagator in this
repository reproduces from the bundled JPL elements (0.000256 au against the ~0.0002 au of the published solution).

```python
from pinneapple_physics.astro.defence import APOPHIS_2029_JD, deflection_table, miss_distance
from pinneapple_physics.astro.smallbodies import load_body

body = load_body("apophis")
rows = deflection_table(body, APOPHIS_2029_JD - 5 * 365.25, APOPHIS_2029_JD,
                        masses_kg=(100.0, 1000.0, 10000.0), speed_km_s=6.1, target_kg=5.4e10)
```

Each row is `delta_v` in mm/s and the miss distance it produces. The physics is unforgiving in a way worth seeing: a
100 kg impactor five years out changes the miss by **21 km** on a 5.4e10 kg body, and the useful regime needs either a
much smaller target or a much bigger impactor. That is why the same table over a 50 m body (1.3e8 kg) on the same
orbit is in the example: the same 100 kg is 400 times more effective.

## 3. What can be guaranteed

`beta` and the Yarkovsky drift `A2` are not known exactly. The orbit is therefore an ensemble, and the two questions
have the same shape:

```python
from pinneapple_physics.astro.defence import sample_ensemble, success_probability, trust_horizon

draws = sample_ensemble(n=64, a2_au_d2=-2.9e-14, a2_sigma=3e-15, beta=3.61, beta_sigma=0.35, seed=0)
success_probability(misses_km)                    # fraction of the ensemble that clears the Earth
trust_horizon(warning_days=days, spread_km=spread) # the warning time where the spread reaches the Earth's radius
```

The **trust horizon** is the point of the whole exercise: beyond it, "we will miss" is not a guarantee whatever the
model says, because the ensemble is wider than the planet. It is the same object as the trust horizon of a forecast,
applied to a deflection.

The Yarkovsky drift is the largest unmodelled acceleration: for Apophis (`A2 = -2.9e-14 au/day^2`) the module gives
**-166 m a year** of semi-major-axis drift, against the ~170 m/yr published — which is why `A2` is one of the two
uncertain parameters and not a detail.

## 4. What the example produces

`examples/planetary_defence/apophis_deflection.py` runs the three acts and writes `apophis_deflection.json` plus two
figures: the miss distance against impactor mass for three warning times, and the ensemble spread against warning time
with the trust horizon drawn on it. `--quick` runs the DART anchor and one deflection in under a minute; the full study
takes about 25 minutes on four cores, because it propagates a 15-year arc once per row.

## 5. What this does not do

- The n-body field is the one the repository already ships (Sun, planets, Moon, the Sun's relativistic term, DE421);
  the 16 perturbing asteroids that a real impact solution uses are not there, which is the ~20 % between the 38 356 km
  above and the published 31 000 km.
- The impactor is a momentum impulse at one instant: no cratering, no ejecta modelling beyond the single `beta`.
- The uncertainty ensemble is normal and seeded; it is not a covariance-propagated orbit solution from JPL Sentry.
