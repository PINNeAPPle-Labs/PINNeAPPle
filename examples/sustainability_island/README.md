# Sustainability Island — coupled physics demonstrator

This is a **screening-level, explicitly hypothetical** tropical coastal campus with
60 residents, 40 daily visitors, 250 kWp PV, 1 MWh battery, rainwater capture,
desalination, wastewater reuse, and irrigated agriculture. It is not a detailed
engineering design, an actual surveyed island, a trained PINN, or a claim of
commercially validated performance.

## Run

From the repository root (Python 3.10+; only Python standard library required):

```bash
python examples/sustainability_island/run.py --weather nasa --year 2024
python examples/sustainability_island/run.py --weather synthetic --year 2024
```

Outputs: `outputs/sustainability_island/timeseries.csv` and `report.json`.
Default downloads a complete hourly meteorological year using NASA POWER;
network access is required. Synthetic mode is only a deterministic smoke-test
fixture and MUST NOT be cited as real climate data.

## Data provenance

- NASA POWER hourly API: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- PVWatts V8 reference for a future validated PV integration:
  https://developer.nlr.gov/docs/solar/pvwatts/v8/
- FAO-56 reference evapotranspiration methodology:
  https://www.fao.org/4/X0490E/x0490e05.htm

The current PV performance ratio is a screening assumption, not PVWatts.
The irrigation model is a screening proxy, not FAO-56 Penman-Monteith.
Weather is spatially gridded and not on-site measured. All installed capacities,
loads, occupancies and water-use coefficients are design assumptions.

## Physical coupling and constraints

At every hourly step: PV and load balance; battery charge/discharge efficiency,
power and state-of-charge limits; rainfall capture; potable consumption;
irrigation demand; wastewater treatment and reclaimed irrigation; desalination
capacity, specific energy and water tank limits. Shortfalls and curtailment are
reported rather than silently assumed away.

## Scientific development required before engineering deployment

1. Site selection, topography, coastal setbacks, geotechnics, water quality,
   ecological assessment and legal permits.
2. Replace PV approximation with PVWatts/PySAM or pvlib using site irradiance,
   module geometry, losses, shading and thermal models; validate against meters.
3. Replace irrigation proxy with daily FAO-56 Penman-Monteith, soil-water
   dynamics, crop-stage Kc, salinity, irrigation scheduling and measured yields.
4. Water quality and treatment design: RO recovery, intake/brine discharge,
   storage, pump hydraulics, sanitation and water safety.
5. Wind-turbine siting, measured wind distribution, power curve, turbulence
   and storm shutdown (wind generation is NOT included in this simulation).
6. Battery degradation, electrical protection, dispatch optimization,
   islanding and black-start, critical-load prioritization and resilience.
7. Explicit stochastic scenario ensemble, extreme-event analysis, uncertainty
   propagation, conservation audits, independent reference comparisons.
8. Export datasets to PINNeAPPle training format, train and validate FNO/
   DeepONet/PINN surrogates, and connect to its digital-twin serving stack.

**Scope boundary:** the runnable script is a physically constrained systems
screening example stored inside PINNeAPPle's examples library, not yet a
registered public `pp.*` API module or end-to-end trained digital twin.
