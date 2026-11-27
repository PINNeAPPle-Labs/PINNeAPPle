# Orbit determination and space traffic (`pinneapple_physics.astro.orbit_determination`)

An orbit is a prediction with an expiry date. This module propagates one with drag, carries its covariance to the lead
time a traffic decision needs, screens a conjunction, and says when the propagation has stopped being usable — because
of a manoeuvre, of a geomagnetic storm, or of an element set that was never a good fit.

```python
import numpy as np
from pinneapple_physics.astro import orbit_determination as od

state = od.DragModel(B_m2_kg=0.005)                     # C_D A / m, the density table of astro.decay
x0 = np.concatenate(od.sgp4_states(*tle, [0.0])[:2])    # the TLE's own osculating state at its epoch
traj = od.propagate(x0, np.arange(0, 86400, 60.0), drag=state)
h = od.trust_horizon(x0, P0, tolerance_km=5.0)           # {"t_hours": 0.41, "criterion": "covariance (3-D 1 sigma)"}
od.tle_check(*tle)                                       # suspect: False, with the checks it passed
od.regime_changes(t_days, a_mean, model_km=quiet_model)  # [{"kind": "drag", "t_days": 1.0, ...}]
```

What already existed and is reused here rather than re-implemented: `astro.satellites` (TLE parsing, CelesTrak fetch
with cache, SGP4 through the `sgp4` package, TEME), `astro.decay` (the density models and the orbit-averaged decay
ODE), `astro.conjunction` (the 2-D probability of collision in the encounter plane) and `astro.orbits` (elements from a
state, the mean motion). New here: the numerical propagator with drag and a storm factor, the analytic two-body
propagator in universal variables, the covariance by lead time, the encounter-plane projection of a *propagated*
covariance, the manoeuvre/storm residual, the TLE check, the learned correction and the `PhysicsEnsemble` that ties the
three propagations together.

## Accuracy: what is measured, and against what

Every number below is produced by `tests/test_astro_orbit_determination.py`, which runs offline in about twenty
seconds. Nothing in this table was downloaded.

| Check | Reference | Measured |
|---|---|---|
| Two-body propagation, one period | energy and angular momentum are exact invariants | `dE/E` 2.5e-11, `dL/L` 1.2e-11, the orbit closes to 8e-8 km |
| Numerical propagator vs the analytic two-body flow | universal-variable Kepler (Stumpff functions), same state | 1.1e-7 km over one period |
| J2 is in the model | the J2 term is not small: it moves the orbit | 68 km in one revolution at 7000 km |
| Semi-major axis decay vs `da/dt = -B rho sqrt(mu a)` | the orbit-averaged formula of `astro.decay` (independent implementation) | 6 % apart at 420 km, F10.7 = 150, B = 0.005 m²/kg |
| Density jump of a factor 3 at day 1 | the same decay rate on both sides of it | ratio 3.15 of the pre/post rates |
| Covariance (STM) vs Monte Carlo, 300 samples | the same two-body orbit, same P0 | along-track within 3.5-4.6 % at 1, 6, 12 and 24 h |
| Along-track growth vs `3 dv t` | closed form of an in-plane velocity error | +6 %, −1 %, +2 % at 6, 12, 24 h; +27 % at 1 h (see below) |
| Collision probability vs the closed form | `F_noncentral_chi2(r²/σ²; 2, m²/σ²)`, the Marcum Q of the isotropic case | 1e-9 relative, three miss distances |
| A propagated conjunction vs a Monte Carlo of the same encounter | 4000 draws from both covariances, propagated to the same TCA | Pc 9.9e-3 against 1.20e-2 ± 0.0014 (1.5σ) |
| Manoeuvre and storm flags | injected events in the propagator, and a clean arc | both flagged at their time; nothing on the clean arc |
| The TLE check | the ISS element set of `astro/data/reference/`, and three corrupted copies | accepts the real one, rejects each corruption for its own reason |
| Learned correction, controlled experiment | a "TLE" whose ballistic coefficient is 30 % off | 5.42 km → 0.14 km (97 %) on a day it never saw |

### The propagator and its decay

Two-body, J2 and drag on a Cartesian state, integrated with DOP853; the density comes from NRLMSISE 2.0 when `pymsis`
is installed and from the Vallado piecewise-exponential table of `astro.decay` otherwise (`density_model_name()` says
which, and it is the exponential table on a stock install). The drag acceleration is
`a = -B rho |v_rel| v_rel / 2` with the velocity relative to the rotating atmosphere.

The decay rate the propagator produces is compared with the orbit-averaged formula the decay module integrates,
`da/dt = -B rho(h) sqrt(mu a)` — two independent codes for the same physics. At 420 km, B = 0.005 m²/kg, they agree
to 6 %; the difference is the density at the *current* altitude against the initial one, which is what the orbit-averaged
formula cannot see. A factor 3 on the density produces a factor 3.15 on the measured rate: this is the mechanism of the
February 2022 Starlink loss, and it is what the injection measures — not the event itself.

### SGP4, and whether it reproduces its own element set

SGP4 is the reference implementation's package (`sgp4` 2.27, Vallado et al. 2006); what is checked here is not SGP4
but the element set: does the propagation reproduce the trajectory it was fitted to? On the ISS element set of
`pinneapple_physics/astro/data/reference/iss_passes_reference.json` (epoch 2018-07-03, already used against skyfield in
`tests/test_astro_accuracy.py`):

| Quantity | TLE | SGP4 propagation | Difference |
|---|---|---|---|
| Mean motion | 15.540056 rev/day | 15.541482 rev/day measured over one day | +9.2e-5 relative |
| Period | 5559.83 s | — | — |
| Semi-major axis | 6783.18 km (from the mean motion) | 6778.72 km (osculating at the epoch) | −4.46 km (−6.6e-4) |
| Eccentricity | 0.0003435 | 0.0002746 | −6.9e-5 |
| Inclination | 51.6414° | 51.6248° | −0.017° |

The osculating–mean differences are the Brouwer short-period terms, not an error, and the tolerance in `tle_check`
(2e-3 in `a`, 0.2° in `i`) is set from this measurement with a factor three of headroom.

The same element set, propagated two ways — SGP4, and this module's numerical propagator seeded from the SGP4 state at
the epoch:

| Ballistic coefficient | 0.5 day | 1 day | 2 days |
|---|---|---|---|
| from the TLE's own B* (4.0e-4 m²/kg) | 0.46 km | 1.18 km | 3.17 km |
| a physical ISS value (5.0e-3 m²/kg) | 1.32 km | 4.89 km | 17.7 km |

That is a divergence between two drag models on one real element set, not an error against truth — and the honest
caveat is that TEME is treated as inertial here, which is worth about 0.2 km over a day (precession-nutation rotates
the frame at ~50″/yr, five orders below the Earth's rotation rate).

Validating SGP4 itself needs the SGP4-VER suite, which is a file to download; it is listed under "what needs the
network" and is not claimed.

### Covariance by lead time, and the trust horizon

The covariance is carried by the state transition matrix (central differences of the analytic propagator, so it costs
six propagations per lead time) and checked against Monte Carlo. Initial σ: 0.1 km in each position axis, 1 mm/s in each
velocity axis; circular orbit at 7000 km; RTN frame of the reference state.

| Lead time | σ radial (km) | σ along-track (km) | σ cross-track (km) | MC / linear (along-track) | `3 σ_v t` |
|---|---|---|---|---|---|
| 1 h | 3.30 | 13.75 | 0.63 | 1.046 | 10.8 km (ratio 1.27) |
| 6 h | 2.54 | 68.79 | 0.89 | 1.037 | 64.8 km (1.06) |
| 12 h | 3.48 | 128.45 | 0.50 | 1.037 | 129.6 km (0.99) |
| 24 h | 1.33 | 264.02 | 0.83 | 1.035 | 259.2 km (1.02) |

The along-track direction is where a LEO covariance lives: it is the linear drift `3 dv t` of an in-plane velocity
error, and it agrees with the closed form to a few percent once the secular term has overtaken the radial-velocity term
`(2/n)(1 − cos nt)` — which is why the 1 h row is 27 % high. The radial direction departs from the linearisation at 24 h
(9.1 km in the Monte Carlo against 1.3 km linear): the along-track error is 264 km there, and the second-order term is
`O(ds²/2a)` ≈ 5 km, so the linear covariance *under*-states the radial uncertainty exactly where it matters.

`trust_horizon` turns that growth into a number: the lead time at which the 3-D 1σ reaches a stated tolerance. With a
1 mm/s velocity uncertainty it reaches

| Tolerance | Trust horizon |
|---|---|
| 1 km | 0.070 h (4.2 min) |
| 5 km | 0.407 h (24 min) |
| 10 km | 0.829 h (50 min) |

linear in the tolerance, as it must be while the velocity error dominates. The demonstration is deliberately a
two-body orbit with a TLE-grade covariance: a real LEO orbit's horizon is set by the *density* uncertainty, which is
why `propagate_covariance` accepts an `along_track_floor_km_day` — a relative error `e` in `B·rho` is `1.5 e a` of
semi-major axis and about `3 e v t` down-track. A 20 km/day floor, for instance, is a 20 % density error at 420 km.

### Conjunctions

`conjunction_probability` finds the TCA by a dense scan of the relative distance and a golden-section refinement,
propagates both covariances to it, sums them, and projects the sum onto the encounter plane perpendicular to the
relative velocity (Foster & Estes 1992; Alfano 2005). Two checks:

* the projection with an isotropic covariance, against the closed form. The 2-D Pc of a Gaussian centred at miss
  distance `m` with isotropic σ is `F_noncentral_chi2(r_c²/σ²; 2, m²/σ²)` (Marcum Q; it reduces to `1 − exp(−r_c²/2σ²)`
  at `m = 0`): 1.98e-2, 1.21e-2 and 1.69e-4 against the quadrature of `astro.conjunction` to 1e-9;
* a synthetic crossing encounter (two planes 0.3 rad apart, 20 m miss, relative speed 2.26 km/s) against 4000 Monte
  Carlo draws of both covariances propagated to the same TCA: Pc 9.85e-3 against 1.20e-2 ± 0.0014, i.e. within 1.5σ.
  Halving the miss moves the Pc to 9.94e-3 — with σ = 0.14 km and a 20 m hard body, the Pc is `r²/2σ²` and hardly
  knows about the miss. That is the regime the method is known to be hard to interpret in, and it is measured rather
  than asserted.

### The residual: manoeuvres and storm-driven drag

`regime_changes` reads the measured semi-major axis (from the propagator's states, or the mean motion of a later TLE),
takes the quiet model's own prediction as the baseline, and at every candidate time fits a jump and a hinge in the same
least-squares model, testing each by its partial F given the other. On the injected events:

| Case | Flagged | Time | Magnitude | `z = sqrt(F)` |
|---|---|---|---|---|
| Clean arc, 2 days, drag only | nothing | — | — | — |
| Density × 3 at day 1 | `drag` | 1.004 d | −0.124 km/day (truth −0.124) | 8900 |
| +5 m/s along-track at day 1 (raises `a` by 8.9 km) | `manoeuvre` | 1.035 d | 8.86 km (truth 8.91) | 305 |

Two things to know: the one-period averaging of the mean element lags the break by about half an hour, and after the
main event is removed the residual still holds the drag-rate change the burn itself caused — real, but below the
materiality a caller states (`min_magnitude`). Without a model baseline the detector removes a quadratic trend instead
and becomes conservative: the curvature of a real decay eats into a rate break, so an event may be missed rather than
invented on a quiet arc.

### A bad TLE fit

`tle_check` is the `measurement_check` of the ensemble: it returns every check with its value and threshold, and
`suspect` when any fails. On the ISS element set: all eight pass. On corrupted copies of it:

| Corruption | Flags |
|---|---|
| last digit retyped (checksum broken) | `checksum` |
| the satellite number of line 2 changed | `checksum`, `satnum` |
| eccentricity 0.9 with the checksum repaired | `perigee` (−5700 km), `reproduces_mean_motion`, `osculating_inclination` |

The third is the interesting one: with the checksum out of the way, what rejects the set is that its own propagation no
longer reproduces it — the perigee is below the ground and the revolutions no longer match the mean motion. A *plausible*
wrong number (a mean motion retyped to another plausible value) is self-consistent by construction and cannot be caught
this way; that is a data-quality problem, not a structural one, and the module does not pretend otherwise.

### The learned correction, and the ensemble

The correction is a ridge regression on `[τ, τ², B*τ, ln(1+τ)]` (τ in days), fitted on the residual of an SGP4
propagation against a *later TLE* — `tle_pair_residuals` builds that training set from a chronological archive. Every
feature vanishes at τ = 0, so the correction leaves the TLE's own state where it is.

Measured offline on a controlled experiment — a "TLE" whose ballistic coefficient is 30 % off, fitted on the first day,
scored on the second:

| | Held-out error at day 2 |
|---|---|
| SGP4 alone (no correction) | 5.42 km along-track |
| SGP4 + the learned correction | 0.14 km (97 % removed) |

The target carries 0.05 km of noise standing for the TLE-as-truth floor. The fit is that good because the basis *is* the
τ² growth of a drag mismatch: on a pure drag mismatch it should recover it, and the real test is a real archive.

`traffic_ensemble` puts the three in a `PhysicsEnsemble`: `sgp4`, `numerical` (the same osculating state with drag) and
`sgp4+correction`, each scored by `orbit_residual` — how badly its own trajectory satisfies the two-body equation, by a
fourth-order finite difference, with no reference needed. On the ISS element set over four hours the three score 1.50e-3,
1.50e-3 and 1.51e-3: the residual is dominated by the J2 term a two-body residual must see, so it does not rank the
propagations — it catches a trajectory that violates the dynamics, and it needs no truth to do so. The spread between
the two drag models over those four hours is 0.12 km. A suspect element set (the `tle_check` above) makes the ensemble
predict with its learned weights and learn nothing.

### Storm warning for a satellite near re-entry

`reentry_warning` integrates the decay ODE of `astro.decay` twice, quiet and stormed, and reports the lifetime
(censored at the horizon), the shortening, the altitude lost, and whether the storm brings the re-entry into the
horizon. At B = 0.005 m²/kg and a factor 2 on the density:

| Initial altitude | Lifetime, quiet | Lifetime, stormed | Warning |
|---|---|---|---|
| 210 km | 5.5 d | 2.7 d (49 % removed) | yes |
| 300 km | > 60 d | 38.6 d | yes — the storm brings the re-entry inside the horizon |
| 550 km | > 60 d | > 60 d (0.9 km lost in 60 d) | no |

The 210 km row is the deployment altitude of the February 2022 Starlink launch, and this is the mechanism: at that
altitude a storm that doubles the density halves the time a satellite has to raise itself.

## What needs the network, and what is therefore declared rather than reproduced

| Input | Source | Used for |
|---|---|---|
| GP element sets, current and archived | CelesTrak (`celestrak.org/NORAD/elements/gp.php`); `astro.satellites.fetch_tles` caches them | any real object; `tle_pair_residuals` needs a *chronological* series, so the GP history of one object over weeks |
| Space-weather indices, F10.7 and Ap | GFZ Potsdam, NOAA SWPC, CelesTrak's space-weather files | the NRLMSISE-00 / JB2008 density that a storm factor stands in for here |
| SGP4-VER verification suite | CelesTrak / Vallado's published distribution | validating SGP4 itself against its published vectors |

**The February 2022 Starlink case.** On 2022-02-03 a Falcon 9 launched 49 Starlink satellites into a low deployment
orbit of about 210 km, from which they raise themselves to about 340 km. A geomagnetic storm began on 2022-02-04; in
its update of 2022-02-08 SpaceX reported that the storm warmed the atmosphere, that density at the deployment altitude
rose by up to 50 %, and that 38 of the 49 satellites had re-entered or would re-enter. `STARLINK_FEB_2022` records the
event and the inputs a reproduction needs:

* the GP history of the 49 objects, 2022-02-03 to 2022-02-15 (at least two element sets per object per day), which
  gives the along-track decay rate before and after the storm object by object;
* 3-hourly Ap and daily F10.7 for 2022-02-01 to 2022-02-15, the drivers of NRLMSISE-00 or JB2008;
* the ballistic coefficient of a Starlink v1.5 in its low-drag attitude — a mass near 260 kg and the deployed
  cross-section, from the operator's data, not assumed here;
* what this module already gives: `propagate` with a `storm` factor on the density, `regime_changes` on the measured
  semi-major axis to time the drag change, `reentry_warning` for the lifetime against it.

**What is reproduced here and what is declared.** Reproduced, offline: the propagator against the analytic two-body
solution and its invariants, the decay against an independent orbit-averaged formula, the response of that decay to a
density jump, the covariance against Monte Carlo and against the `3 dv t` law, the collision probability against its
closed form and against a Monte Carlo of the encounter, the manoeuvre and storm flags on injected events, the TLE check
on a real element set, the learned correction on a controlled drag mismatch, and the storm warning on the decay ODE.
Declared, not reproduced: the loss of the 38 Starlink satellites, any position error against SGP4 *as truth*, and any
validation of SGP4 itself — each needs one of the inputs above, and none of them was downloaded for this page.

**Not in this module.** The globe animation of the debris field with uncertainty ellipsoids needs a real catalogue of
objects — the CelesTrak GP data again — to have anything to draw, and it is a rendering task of the app layer, not a
computation: what this module supplies for it is the covariance that sizes each ellipsoid (`position_covariance`) and
the probability that says which conjunctions deserve one (`conjunction_probability`).
