# A wind farm from public SCADA: power curves and analytical wakes

This guide runs the Penmanshiel case end to end and states, for every number, what
produced it and what it does not establish. It is the written half of
`examples/wind_farm_penmanshiel/`; the code is `pinneapple_systems/wind`.

**The short version.** A physics-informed power curve trained on 13 months of
SCADA is 54 kW RMSE on a later 4-month window, against 47 kW for the IEC binned
curve and 45 kW for boosted trees. The constrained model gives up 0.4 % of rated
power in accuracy and is the only data-driven model whose curve cannot break the
four physical constraints — the binned curve and both boosted-tree models break
three of them. The analytical wake models over-predict the farm's loss by 11.8
percentage points with their textbook constants; one constant fitted on a
validation window and scored on a later one cuts that bias to 3.0 points and the
RMSE by a third. What they still do not reproduce is the loss's *direction
dependence*.

## The data

| | |
|---|---|
| Source | Plumley & Takeuchi, *Penmanshiel wind farm data*, Zenodo, [doi:10.5281/zenodo.16807304](https://doi.org/10.5281/zenodo.16807304) |
| License | CC-BY-4.0 |
| Descriptor | Plumley (2022), *Scientific Data* 9:331 |
| Farm | 14 × Senvion MM82, 2.05 MW, 82 m rotor, 59 m hub, 28.7 MW, Scottish Borders |
| Records used | 2016-06 → 2017-12 (the 2016 and 2017 archives, 1.12 M turbine-samples) |
| Resolution | 10 minutes, ~300 signals per row |

Three things in this export change how the analysis has to be done, and all three
are used rather than reinvented:

- **Curtailment, downtime and availability are recorded**, in the operator's own
  lost-production columns. The operating mask is therefore evidence-based, not a
  heuristic guess at "is it curtailed?" from a power deficit.
- **`Potential power default PC (kW)`** is the operator's expected power at the
  measured wind speed — an independent reference curve, used as a yardstick and
  never as a model this package trained.
- **`Density adjusted wind speed (m/s)`** lets the package's own IEC 61400-12-1
  density correction be checked against the vendor's. They agree to machine
  precision (correlation 1 − 1e-16) on the 2016 record, which is a real
  verification of `density_corrected_speed` rather than a claim about it.

Operating fractions run 71-82 % per turbine. The rest is downtime, availability
and missing samples, counted per rule in `results.json` under `data.micro_quality`.

There is no WT03: the farm numbers 01-15 skipping 03. Code that assumes 14
turbines named 01..14 crashes on the missing one, which is why the loader reports
the turbine list it found.

## Temporal split

| Window | Period | Rows (all turbines) |
|---|---|---|
| train | 2016-06-02 → 2017-06-30 | 747,632 |
| validation | 2017-07-01 → 2017-08-31 | 124,992 |
| test | 2017-09-01 → 2017-12-31 | 245,952 |

No shuffling, no lagged target, every scaler fitted on train only, and the wake
decay constant fitted on validation and scored on test. The test window is 3-15
months after the last training sample.

## One turbine: five curves and an honest ranking

Focus turbine T01 was chosen because it has the most operating samples — stated,
not assumed. Test window: 15,964 operating samples, Sep-Dec 2017.

| Model | RMSE [kW] | MAE [kW] | energy-weighted RMSE [kW] | R² | energy error |
|---|---|---|---|---|---|
| reference (analytic, Cp = 0.45) | 102.5 | 66.4 | 130.7 | 0.977 | +4.3 % |
| IEC binned power curve | **46.6** | 34.4 | 53.5 | 0.9952 | −0.6 % |
| logistic, 5 parameters | 48.9 | 36.0 | 55.1 | 0.9947 | −0.3 % |
| **physics-informed (monotone hinge)** | 54.3 | 38.0 | 65.6 | 0.9935 | +0.2 % |
| boosted trees (wind speed only) | 46.7 | 34.4 | 53.4 | 0.9952 | −0.6 % |
| boosted trees (met features) | **45.3** | 31.1 | 55.0 | 0.9955 | −0.1 % |
| constant (training mean) | 682.5 | 577.0 | 846.3 | −0.03 | −12.8 % |
| operator potential power *(measured reference)* | 57.1 | 40.9 | 64.0 | 0.9928 | +2.2 % |

Read this table with three facts next to it:

1. **The physics-informed model is 8.9 kW (0.43 % of rated) behind the binned
   curve** and 9.0 kW behind boosted trees. That is the price of the constraints,
   and it is small enough to be worth paying — but it is a price, not a win.
2. **It is the only data-driven model that satisfies all four physical
   constraints**, re-measured on the fitted curve rather than asserted:

   | model | monotone | zero outside cut-in/out | ≤ rated | ≤ Betz |
   |---|---|---|---|---|
   | reference | ✓ | ✓ | ✓ | ✓ |
   | IEC binned | ✗ | ✗ | ✓ | ✗ |
   | logistic | ✓ | ✓ | ✓ | ✓ |
   | physics-informed | ✓ | ✓ | ✓ | ✓ |
   | boosted trees (speed only) | ✗ | ✗ | ✓ | ✗ |
   | boosted trees (met features) | ✗ | ✗ | ✓ | ✗ |

   The binned curve is the industry's *definition* of a turbine's power curve, and
   it is not monotone. That is not a defect of this implementation: bins are
   averages, and averages of noisy bins wobble. It is, however, a real reason not
   to hand a raw binned table to an optimizer, an AEP calculation or an inverter
   controller without shaping it first.
3. **The operator's own potential power is not beaten here** (57.1 kW vs 46.6),
   and it should not be: it is fitted by the operator over years of data with
   information this POC does not use. It is in the table as a yardstick, labelled
   as a measured reference, not as a sixth model.

The physics-informed curve implies **Cp = 0.430 in region 2** against the Betz
limit of 0.593, and the region-2 cubic law is enforced as a penalty, so the check
is re-run and reported rather than assumed.

### Uncertainty

A split-conformal interval built on the validation residuals has half-width
**80 kW at a 90 % target**, and covers **87.7 %** of the test residuals. The
gap is the honest one: a year-later window is not exchangeable with the
calibration window, so the marginal guarantee does not transfer exactly, and the
measured coverage is what gets reported.

The joint training envelope (wind speed 3.0-24.4 m/s, density ratio 0.948-1.030,
ambient −2.9 to 26.4 °C) contains 100 % of the test window — so this particular
holdout is an *interpolation* test, not an extrapolation test. That is a property
of the split, not a strength of the model, and the applicability map says so.

## Wakes: two analytical models against what the farm did

The published layout gives nearest-neighbour spacings of **4.2-5.2 rotor
diameters**, which is tight enough for wake losses to matter.

The comparison is built from SCADA alone. For each direction sector and wind-speed
bin, the *reference* is the group of three most-upwind turbines; their metered
power is inverted through the site's power curves to a reference speed, and the
farm's efficiency is observed power over what every machine would make at that
speed. Three decisions in that sentence are deliberate and each was forced by the
data:

- **Group of three, not one turbine.** A single reference put the observed
  efficiency between 0.64 and 1.50 across sectors. The cause was one machine
  (T07) running 15 % below the farm mean; a three-turbine group dilutes exactly
  that kind of individual offset.
- **Power, not anemometry.** Nacelle anemometers are not calibrated against each
  other, and in region 2 a 5 % speed offset is a 15 % power offset — bigger than
  the wake loss. Metered active power is comparable across machines.
- **The model is evaluated at the reference group's own inflow.** The group is
  itself waked in row-aligned sectors, so a model evaluated at the free stream
  would describe a different quantity than the observation.

Result on the test window (96 sector × speed cells, 12,942 timestamps):

| Wake model | farm efficiency (model) | observed | bias | RMSE | correlation |
|---|---|---|---|---|---|
| Gaussian, k* = 0.0325 (textbook) | 0.818 | 0.936 | −11.8 pp | 0.185 | +0.17 |
| Gaussian, k* = 0.090 (fitted) | 0.906 | 0.936 | −3.0 pp | **0.126** | +0.16 |
| Jensen/PARK, k = 0.075 (textbook) | 0.830 | 0.936 | −10.7 pp | 0.169 | +0.21 |

The fitted decay constant comes from the **validation** window and is then scored
on the test window, where it still improves the RMSE from 0.185 to 0.126 — so it
transfers across periods, which is the only evidence that would justify using it.

Three statements this comparison supports, and three it does not:

- **Supported.** The textbook constants over-predict this farm's wake loss by
  about 11 percentage points; the site's effective decay constant is roughly three
  times the Bastankhah-Porté-Agel default. A farm with a published layout and
  SCADA can and should measure that constant instead of importing it.
- **Supported.** The effect is large enough to matter for AEP: an 11-point error
  in farm efficiency is roughly a 10 % error in annual energy.
- **Supported.** The two model families bracket the observation in the same way,
  so the conclusion is not an artefact of one formulation.
- **Not supported:** that the wake model reproduces the *direction shape*. The
  correlation is +0.16 to +0.21, which is weak. The model's deepest predicted dip
  (135°, 0.59) is not the observed one (0.88). Whatever dominates the
  sector-to-sector variation in this farm — terrain, calibration, sector
  management, real per-turbine performance — is not in the model.
- **Not supported:** that the observed 0.936 is the true wake-free-normalised
  efficiency. It is an upper bound: it absorbs curtailment that survived the
  filter, icing, and genuine per-turbine under-performance.
- **Not supported:** any claim of CFD-grade fidelity. These are engineering
  models: depth-averaged momentum, a self-similar profile, no terrain, no
  stability, no wake meandering, no yaw deflection.

### Surrogate and operational optimisation

- A small MLP fitted to the analytical farm model reproduces its efficiency on a
  finer, phase-shifted grid with **RMSE 0.020 and R² 0.956**. That is a speed
  device — the surrogate cannot be more accurate than its teacher.
- The curtailment study finds **no modelled gain** at this spacing. That result
  is only trustworthy because an earlier version of the model was wrong: it
  scaled the thrust coefficient without scaling the derated machine's own power,
  and reported a 10-16 % gain that came from a turbine producing full power and
  no wake. Fixing `farm_power` to scale both made the gain disappear. The
  remaining assumption — that a C_T multiplier stands in for yaw or derating, with
  the wake not deflected — is stated in every output and is not validated here.

### The 3D twin and the demo film

The same wake model drives an interactive 3D scene
(`pinneapple_systems.wind.twin`), built as a `pinneapple_twin3d.Scene`:

- a ground grid whose per-vertex `velocity_ratio` field is the **plan-view wake
  footprint**, one slice per wind-direction step;
- towers and rotor discs at the published positions;
- one sensor per turbine carrying that machine's modelled power at each
  direction, so the viewer's time slider reads out the array.

`outputs/twin/` is the exported scene plus the bundled web viewer (`serve` opens
it). `outputs/wind_farm_demo.mp4` is the same scene rasterised frame by frame and
encoded to H.264 — 120 direction steps, about 12 seconds.

Two things are worth stating about the field. It is a **plan-view** footprint:
each point is treated as lying at hub height, so it is the deficit a turbine
there would see, not the deficit at ground level. And every wake is seeded with
the free-stream speed, so overlapping wakes **superpose** rather than cascade —
that keeps the picture independent of the order the turbines are drawn in, and
it makes the field slightly optimistic far downwind compared with `farm_power`.
At the turbines themselves the two agree exactly when no two share a wake path;
the example's tests pin both facts down.

The film is rendered, not screen-captured, so it is reproducible from the
committed code and needs no display.

## What VeriPhysics says

`power_applicability_map` assembles the run into an applicability map from the
checks that actually ran: the four conservation checks (all VERIFIED on the fitted
curve), the held-out error against a declared threshold of 5 % of rated (PASS at
2.6 %), the conformal calibration, and the training envelope. Items that were not
run stay `NOT_RUN` rather than becoming a neutral pass. The rendered map is
written to `outputs/applicability_map.txt`.

## Reproducing

```bash
python examples/wind_farm_penmanshiel/run_poc.py \
    --raw ~/pn-wind-data/raw --cache ~/pn-wind-data/cache \
    --out examples/wind_farm_penmanshiel/outputs
```

Three minutes on a laptop once the parse cache is warm (about 15 on a cold cache).
The data download instructions are in the example's README.

Two notebooks walk the same pipeline cell by cell, with the outputs committed:
`examples/wind_farm_penmanshiel/notebooks/01_turbine_power_model.ipynb` (the
power curve, the guardrails, the interval, the applicability map) and
`02_wake_models.ipynb` (the wake shapes, the SCADA comparison, the calibration,
one frame of the twin). They read the same pickle cache and take under a minute
warm.

## Where this goes next

1. **Per-turbine calibration.** The dominant residual in the farm comparison is
   per-machine, not per-direction. Estimating a calibration offset per turbine
   from the sectors where it is free-stream would make the comparison an actual
   measurement of wake loss instead of an upper bound.
2. **A learned wake correction.** With a calibrated baseline and five more years
   of data, a physics-informed correction on top of the analytical wake field is
   the natural surrogate — and the measured efficiency is the target.
3. **Yaw steering with a deflection model.** The current curtailment study cannot
   represent wake deflection, which is the whole point of wake steering. That
   needs a yaw-aware model (the FLORIS family), not a C_T multiplier.
4. **Terrain and stability.** The sector-level mismatch is consistent with flow
   effects these models do not contain; a mesoscale or CFD reference over the
   layout would test that hypothesis.
5. **External data.** The dataset also ships PMU, grid-meter and event tables
   that this POC does not touch, and the 2018-2024 archives that were not
   downloaded.
