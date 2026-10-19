# PINNeAPPle Observatory

**The sky and the solar system as they are right now, from where you stand, in 3-D.**

![Dusk over São Paulo: Venus, Mercury and the crescent Moon above the western horizon](../../docs/assets/apps/14-observatory-sky.jpg)

Open it, allow your location (or pick a city), and you see the sky over you: the stars of the Yale Bright Star
Catalogue with their real colours, the constellations, the Milky Way, the Sun, the Moon with its phase and the planets
where they are this minute, twilight colours, refraction and extinction near the horizon, and the satellites going
over. One click zooms out from your sky to the solar system: the Earth turning under its day/night terminator with your
pin on it, the Moon's orbit, the planets on their orbits (linear or log distances). On a phone, *Point* uses the
motion sensors to show the part of the sky the phone is aimed at.

Every number comes from `pinneapple_physics.astro`, an independent implementation checked against independent references
(table below):

```
JPL DE421 (public domain) ──► SPK reader ──► light time, solar deflection, aberration ──► IAU 2006 precession,
                                                                                          nutation, sidereal time
                                     │                                                    │
             Kepler elements, N-body integrators, PINN surrogate        topocentric alt/az, refraction, airmass
                                     │                                                    │
             astrophysics panel: separation, observing window,          rise / transit / set, twilight, phases,
             light time, Doppler, star luminosity / radius, HR,         elongations, oppositions, eclipses,
             black-body spectra, black-hole lensing of your sky         transits, satellite passes (SGP4)
```

The interface is in **English and Portuguese** (EN / PT in the top bar; `#lang=pt` in the URL). All text lives in the
message catalogues `observatory/static/i18n/{en,pt}.json`; constellations use their Portuguese names, stars their IAU
names.

![O céu de São Paulo ao anoitecer, em português](../../docs/assets/apps/14-observatory-sky-pt.jpg)

## Panels

- **Tonight**: sky state (day / twilight / night), Moon phase, twilight times, what is up now, rise · transit · set of
  the Sun, Moon and planets.
- **Object** (click anything): alt/az, RA/Dec of date and J2000, distance, light time, range rate, apparent diameter,
  magnitude, airmass and extinction, elongation and phase; for stars: distance, M_V, T_eff, luminosity, radius and its
  place on the HR diagram with the stars above your horizon highlighted.
- **Satellites**: next passes of the ISS, Tiangong and Hubble from your location (rise/culmination/set, max altitude,
  visibility, magnitude). TLEs: CelesTrak, cached 6 h.
- **Analyses**: angular separation, best observing window tonight (dark sky, airmass, Moon distance), orbital elements
  + light time + Doppler shift, black-body spectrum, a year of events (phases, equinoxes, oppositions, conjunctions,
  greatest elongations, lunar and solar eclipses with local circumstances, transits of Mercury and Venus), and a
  Schwarzschild black hole placed in front of the sky you are looking at (null geodesics from
  `pinneapple_physics.blackhole`), with the predicted shadow sizes of M87* and Sgr A* against the EHT rings.
- **Observe**: an observing session: plan the night for a target, set up a telescope (presets or your own), slew a
  GoTo mount, then track the target in an eyepiece / camera view at 1x-3600x with field rotation, extinction, sky
  brightness, seeing and the cloud forecast ("real sky") or none ("perfect sky"); a session log.
- **Journey**: fly from your place to a planet, moon, the Sun or a star in 3-D (real positions for the date, stars at
  their parallax distance), with scale markers, counters (distance, light time, trip times at real speeds, the
  target's size and magnitude) and, at a star, the Sun behind you at the right magnitude.
- **Learn**: an info card for every object (planets, moons, bright stars, constellations, Messier objects,
  satellites): key numbers with a source each and everyday comparisons, curiosities, history, how to see it tonight,
  a quiz; search in English or Portuguese. Data: `pinneapple_physics/astro/data/facts/`.
- **Engine**: the PINNeAPPle part. A physics-informed neural network learns a planet's orbit for the next year,
  `r(t) = r_Kepler(t) + τ² g_θ(τ)`, from the residual of the full N-body equation of motion plus 8 anchor states, and is
  judged against the classical integrator: error, split-conformal band, energy / angular-momentum checks, residual, a
  verdict. Next to it, leapfrog / Yoshida-4 / RK4 conservation, and the validation table of every method.

## PINNeAPPle models (each with a trust card)

| Where | Model | Checked against | Verdict |
|---|---|---|---|
| Satellites | orbit-decay PINN, NRLMSIS 2.0 drag, Monte-Carlo envelope over solar flux / Ap / drag area | RK integration; SGP4's own drag model from the TLE | trustworthy (metres vs RK; decay rate within 6 % of SGP4 for an ISS TLE) |
| Analyses › Asteroids | small-body propagator (DE421 Sun, planets, Moon + solar relativity) | JPL Horizons (Ceres: 3.9 km after 4 yr), JPL close approaches (Apophis 2029: within JPL's 3σ, 11 s) | trustworthy |
| Analyses › Asteroids | per-body orbit PINN (Kepler + τ² network, N-body residual) | the integrator, held-out times (Apophis: 64 km, 0.11″ over a year) | trustworthy |
| Analyses › Asteroids | learned propagator for many NEOs (one network) | the integrator on 150 unseen bodies | not trustworthy (only 1.2× better than Kepler; gate needs 5×) |
| Analyses › Space weather | Kp forecast (quantile network on persistence, conformal intervals) + aurora probability | observed Kp, held-out months; persistence | 3 h trustworthy (persistence median), 24 h use with caution (misses storm onsets) |
| Analyses › Stellar ML | HR-diagram clustering, luminosity / spectral class | BSC5 MK types, cross-validation | trustworthy (70 % / 79 % vs 40 % / 25 % baselines) |
| Analyses › Conditions | Open-Meteo ensemble: P(clear), cloud quantiles, transparency, seeing proxy | none yet | not verified (uncalibrated ensemble spread) |
| Analyses › Black hole | thin accretion disc on exact null geodesics, lensed in front of your sky; zoo forecaster when it passes its gate | Schwarzschild shadow and redshift closed forms | render checked; forecaster placeholder until the zoo model exists |

## Accuracy

Computed by `pinneapple_physics.astro.validation` (also `tests/test_astro_accuracy.py` and the Engine tab):

| Method | Reference | Result | Limit |
|---|---|---|---|
| Apparent places, Sun and planets | USNO NOVAS C 3.1 (DE405), 4 epochs, 36 cases | ≤ 0.41″ (≤ 0.13″ without Pluto; the rest is DE405 vs DE421) | 1″ |
| Apparent place, Moon | USNO NOVAS C 3.1 | ≤ 0.022″ | 2″ |
| Apparent / astrometric places (Sun, Mars, topocentric Moon) | JPL Horizons | ≤ 0.065″ | 1″ |
| Topocentric alt/az, all bodies, 216 random epochs and sites | skyfield | ≤ 0.011″ | 0.5″ |
| Sunrise / sunset, Moon phases, equinoxes | USNO tables (to the minute) | ≤ 25 s | 60 s |
| Moon illuminated fraction | USNO (2 decimals) | exact | 0.01 |
| Eclipses (2022, 2024) and transits (2012, 2016, 2019), greatest | NASA predictions | ≤ 9 s | 120 s |
| ISS rise / culmination / set, 69 events, 3 cities | independent pass finder, same TLE | ≤ 0.18 s | 2 s |
| Yoshida-4 N-body, 2 years | energy / angular momentum | 1.4e-10 / 3e-15 | 1e-8 / 1e-12 |
| PINN orbit surrogate, Mars, 1 year | the integrator, held-out times | 0.09″ (122 km), verdict *trustworthy* | 1″ |

The Earth surrogate fails its checks (the Moon's 4700 km monthly wobble of the Earth around the Earth–Moon barycentre
is outside the network's Fourier features): the app shows it *not trustworthy* rather than hiding it.

![The solar system and the Engine tab: the Mars surrogate passes every check](../../docs/assets/apps/14-observatory-system.jpg)

## Run locally

```bash
pip install -e . -r apps/observatory/requirements.txt          # from the repository root
cd apps/observatory && uvicorn observatory.api:app --port 8094
# http://localhost:8094        (?lat=..&lon=..&t=2026-10-13T22:00:00Z&az=250&alt=15&fov=60&view=system also work)
```

Docker: `docker build -f apps/observatory/Dockerfile -t observatory . && docker run -p 8094:8094 observatory`.
Deployment with the other apps: `apps/deploy` (service `observatory`, `OBS_DOMAIN`).

| Environment | Meaning |
|---|---|
| `OBS_USER` / `OBS_PASSWORD` | HTTP Basic login (everything except `/health`) |
| `OBS_MAX_HEAVY` | concurrent heavy computations (PINN training, lensing, events), default 2; excess gets 429 |
| `OBS_TLE_GROUPS` | CelesTrak groups, default `stations,visual` |
| `OBS_TLE_FILE` | use a local TLE file instead of CelesTrak (offline demos, tests) |
| `OBS_CACHE_DIR` | TLE cache directory |
| `OBS_SURVEY` | `0` disables the DSS2 survey cut-outs (CDS hips2fits); the eyepiece field is then drawn from the catalogue |
| `OBS_WEATHER_FILE` | a saved Open-Meteo forecast (JSON) used instead of the live one (offline demos) |
| `OBS_PRECOMPUTE` | `0` skips the start-up validation run and the default Mars surrogate |

Geolocation and the phone's motion sensors need HTTPS (or `localhost`).

Browser tests (Playwright, Chromium; the server runs offline with a local TLE file):

```bash
cd apps/observatory && npm ci && npx playwright test
node tools/capture.mjs http://localhost:8797        # regenerate docs/assets/apps/14-observatory-*.jpg (EN and PT)
```

## Data and licences

- Planetary ephemeris: JPL DE421 (public domain), from the `skyfield-data` package (MIT); IERS UT1−UTC from the same.
- Stars: Yale Bright Star Catalogue 5th rev. ed. (Hoffleit & Warren 1991, NASA ADC / CDS V/50); constellation figures,
  names and the Milky Way contours: d3-celestial (BSD-3, O. Frohn). Details: `pinneapple_physics/astro/data/README.md`.
- Satellites: CelesTrak GP element sets; SGP4 by the `sgp4` package (MIT, Vallado et al. 2006).
- Earth textures: NASA Blue Marble and Earth at Night (public domain) via the three-globe package (MIT). The Moon,
  planet and Sun textures are **procedural** (`tools/make_textures.py`), not imagery; NASA/USGS maps with the same
  file names can replace them in `observatory/static/textures/`.
- Deep-sky fields in the Observe tab: DSS2 colour (STScI Digitized Sky Survey) via CDS hips2fits, fetched by the
  server, cached and credited in the view. Cloud forecast: Open-Meteo (CC BY 4.0).
- No code from GPL planetarium software was used; the algorithms follow the IERS Conventions, the Explanatory
  Supplement, Meeus, and the papers cited in the module docstrings.
