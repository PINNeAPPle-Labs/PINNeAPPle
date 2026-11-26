# Positional astronomy and astrophysics (`pinneapple_physics.astro`)

An independent NumPy implementation of the apparent sky, validated against USNO NOVAS, JPL Horizons, the USNO almanac,
NASA eclipse predictions and skyfield. It powers the [Observatory app](../guides/observatory.md).

```bash
pip install "pinneapple[sky]"     # sgp4 + skyfield-data (bundles the public-domain JPL DE421 kernel)
```

```python
from pinneapple_physics.astro import Observer, Time, apparent, rise_transit_set, moon_phase

here = Observer(-23.5505, -46.6333, 760)                  # latitude, longitude (east +), metres
t = Time.from_iso("2026-10-13T23:30:00Z")
p = apparent("saturn", t, here)
p.alt_apparent, p.az, p.ra, p.dec, p.distance_km, p.light_time_s, p.range_rate_km_s
rise_transit_set("moon", here, t.utc)                      # UTC Julian dates
moon_phase(t)["illuminated_fraction"]
```

| Module | What |
|---|---|
| `timescales` | UTC, TAI (leap seconds), TT, TDB, UT1 (IERS table when available, else held / 0), ERA, GMST (IAU 2006), GAST |
| `spk` | JPL SPK reader (DAF, segment types 2 and 3), no SPICE dependency |
| `frames` | frame bias, IAU 2006 precession, IAU 1980 nutation (terms > 5 mas; < 0.02″ from IAU 2000A), WGS84, horizon, ecliptic, separations |
| `astrometry` | apparent places: light time, solar deflection, relativistic aberration; refraction (Saemundsson), airmass (Kasten & Young), extinction; catalogue stars with proper motion and parallax |
| `phenomena` | rise / transit / set, twilight, Moon phase and bright limb, magnitudes, oppositions / conjunctions / greatest elongations, phases, seasons, lunar and solar eclipses (with local circumstances), transits |
| `satellites` | TLE parsing and CelesTrak fetch with cache, SGP4 → TEME → ITRS, look angles, Earth shadow, magnitudes, visible passes |
| `stars` | Yale BSC5 (9,096 stars), constellation figures, T_eff from B−V, black-body bolometric correction, luminosity, radius, HR diagram, Planck spectra |
| `orbits` | Kepler's equation, elements ↔ state, planet elements from DE421, light time and relativistic Doppler |
| `nbody` | leapfrog, Yoshida-4, RK4, DOP853 on Sun + planets + Moon with energy / angular-momentum diagnostics |
| `surrogate` | physics-informed orbit surrogate (Kepler + τ² × network, full N-body residual) with conformal bands, physics checks and a verdict |
| `lensing` | the real sky lensed by a Schwarzschild black hole (geodesics from `pinneapple_physics.blackhole.raytrace`), Einstein radius, shadow size vs EHT |
| `validation` | every accuracy check as data; `run_all()` |

## Accuracy

| Check | Reference | Result |
|---|---|---|
| Apparent places, Sun and planets (36 cases) | USNO NOVAS C 3.1 | ≤ 0.41″ (≤ 0.13″ without Pluto) |
| Moon | USNO NOVAS C 3.1 | ≤ 0.022″ |
| Sun, Mars, topocentric Moon | JPL Horizons | ≤ 0.065″ |
| Topocentric alt/az, 216 random cases 1972–2049 | skyfield | ≤ 0.011″ |
| Sunrise/sunset, phases, seasons | USNO tables (minute) | ≤ 25 s |
| Eclipses and transits | NASA | ≤ 9 s |
| ISS passes (69 events) | independent pass finder | ≤ 0.18 s |
| Yoshida-4, 2 years | energy, angular momentum | 1.4e-10, 3e-15 |

The reference values are in `pinneapple_physics/astro/data/reference/` with their sources.

## The orbit surrogate

```python
from pinneapple_physics.astro.surrogate import train_orbit_pinn
rep = train_orbit_pinn("mars", jd_tdb=2461327.0, days=365, iters=800)   # ~30 s on one CPU core
rep["verdict"], rep["summary"]["max_angle_arcsec_test"], rep["checks"]
```

The network only learns the perturbation of the exact two-body solution, `r = r_Kepler + s τ² g_θ(τ)`, so the initial
position and velocity are hard constraints. Its loss is the residual of
`r'' = −μ r/|r|³ + Σ_j μ_j [(r_j − r)/|r_j − r|³ − r_j/|r_j|³]` at collocation points (perturbers from the reference
integration) plus a few anchor states. It is then scored on held-out integrator states: angular error, split-conformal
coverage (`pinneapple_analysis.uncertainty.conformal`), the error in two-body energy and angular momentum relative to
their perturbation-driven variation, the equation-of-motion residual and the gain over Kepler. Mars passes every check
(0.09″ over a year); the Earth fails them, because of the Moon, and the verdict says so.

## PINNs and machine learning in the observatory

Each model is shown with a trust card in the app (verdict, checks against an independent reference, "how this was
computed, how much to trust it") and tested in `tests/test_astro_ai.py`.

| Module | Model | Independent reference | Result |
|---|---|---|---|
| `decay` | PINN for drag decay of a LEO orbit, NRLMSIS 2.0 density (`pymsis`), Monte-Carlo input envelope | RK integration; SGP4's own drag model from a TLE | PINN within metres of the integrator over weeks; decay rate within 6 % of SGP4 for an ISS TLE (B = 0.005 m²/kg, F10.7 = 70) |
| `smallbodies` | test-particle propagator (DE421 Sun + planets + Moon, solar 1PN term) from JPL SBDB elements | JPL Horizons (Ceres), JPL close-approach table (Apophis 2029) | Ceres 3.9 km after 4.2 yr; Apophis 2029-04-13: 38,356 km vs JPL 37,725 ± 697 km (3σ), 11 s |
| `surrogate.train_orbit_pinn(SmallBody)` | per-body PINN: Kepler orbit + τ² network trained on the N-body residual (planets from DE421) | the integrator above, held-out times | Apophis, 1 year: 64 km max (0.11″ from the Sun) vs 16,835 km for Kepler alone, 88 % coverage of the 90 % band; *trustworthy* |
| `neural_propagator` | one network for many NEOs: elements + time → deviation from Kepler (R/T/N frame) | the full integrator on 150 unseen bodies | median max error 53,326 km vs 61,896 km for Kepler: fails its gate (≥ 5× better than Kepler), *not trustworthy*; `trust_report()` says so and the app uses the per-body PINN instead |
| `spaceweather` | Kp quantile network (residual on persistence), conformal intervals; aurora probability from the IGRF dipole and the NOAA oval rule | observed Kp (GFZ) on held-out months; persistence | 24 h: MAE 1.15 vs persistence 1.18, 84 % coverage of the 80 % interval; storm onsets mostly missed (verdict: use with caution) |
| `stars_ml` | Gaussian-mixture clustering of the HR diagram; k-NN luminosity and spectral class | BSC5 MK spectral types, 5-fold CV | luminosity class 70 % (baseline 40 %), spectral class from B−V 79 % (baseline 25 %), cluster purity 71 % |
| `conditions` | Open-Meteo ensemble → P(clear), cloud quantiles, transparency, seeing proxy; reliability calibration | none yet (needs a forecast/observation history) | labelled *not verified* |
| `lensing.trace_thin_disc`, `disc_image`, `accretion_view` | thin disc (ISCO, Novikov-Thorne flux, Keplerian gas, Doppler beaming and gravitational redshift, g⁴ intensity) lensed in front of the real sky; exact null geodesics u'' + u = 3u² | Schwarzschild closed forms: shadow edge at b = 3√3 GM/c², face-on g = √(1 − 3/r) | both to 1 % in the tests |

## Reading a simulation's output: the `yt` reader (`pinneapple_data.astro`)

`pinneapple_physics.astro` looks at the sky; this section is the other end — the output of the codes that compute it.
One reader covers Enzo, FLASH, RAMSES, GADGET, AREPO, SWIFT, Athena/Athena++, Cholla, ART and generic HDF5, because
[yt](https://yt-project.org) (Turk et al., ApJS 2011, BSD-3-Clause) already does, and the result is the same
`PhysicalSample` the CFD readers return, so the trainers and the physics audit take it without a branch.

```bash
pip install "pinneapple[astro]"      # yt; optional on purpose -- nothing imports it until the reader is called
```

```python
from pinneapple_data.astro import yt_reader

yt_reader.available()                                   # (False, 'the yt reader needs yt: pip install yt ...') without it
yt_reader.list_fields("output_0042")                    # [{"field": "gas_density", "code": "gas", "unit": "g/cm**3"}, ...]
s = yt_reader.load("output_0042")                       # a PhysicalSample: grid state, yt's units, the box in provenance
s.state["gas_density"].values                           # xarray, dims (x, y, z) -- one cell per grid cell
s.schema["units"], s.provenance["dataset_type"], s.provenance["time"]

yt_reader.load("output_0042", level=2)                            # the AMR resampled at level 2
yt_reader.load("output_0042", fields=["density", ("gas", "temperature")])
yt_reader.load("output_0042", region=((0, 0, 0), (0.5, 0.5, 0.5)))  # a box, in the dataset's length unit
yt_reader.load("snapshot_000.hdf5", to="points")                   # particles stay particles (domain type "points")
yt_reader.load("output_0042", physics_audit=True).extras["physics_audit"]   # mass and energy sums, with their caveat
```

Decisions worth knowing:

- **A grid sample is resampled, a particle sample is not.** AMR data is sampled onto the uniform grid at `level` (a
  surrogate wants one array), while particles are returned as a table of point arrays: averaging them would invent
  positions no particle has.
- **The axes are (x, y, z), measured not assumed.** yt's grid arrays were checked on yt 4.4.4 with a dataset whose
  density is a function of `x` alone and whose temperature is a function of `z` alone: `grid[field][:, 0, 0]` walks
  `x` and `[0, 0, :]` walks `z`, for both `covering_grid` and `arbitrary_grid`. `tests/test_astro_yt_reader.py` keeps
  it that way, because a silently transposed field is a plot that looks plausible and is wrong.
- **The units travel.** `schema["units"]` carries yt's unit for every field (`g/cm**3`, `K`, …) and the state records
  the unit system, so a number is never bare.
- **`extras["yt_dataset"]` is the live dataset**, for the operations this reader does not wrap (projections, phase
  plots, radial profiles).
- **The audit's totals say what they are.** `physics_audit` returns sums of the sampled values with unit, maximum and
  a note that an integral needs the cell volume; the divergence of B and the virial checks of the issue's plan are the
  next step, and they are missing rather than silently zero.

Not done yet in #515: the two public sample datasets (an Enzo cosmology box and a FLASH Sedov blast) with the Sedov
shock radius against the self-similar solution, and the MHD/virial audit hooks. The reading, the units, the axis
order and the optional dependency are in place and tested; tests that need yt are skipped with the install line when
it is absent, so the suite stays green on a machine without it.
