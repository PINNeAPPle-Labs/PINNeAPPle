# BrakeCool AI

Inverse design of the radial vanes of a vented brake rotor (vane count, vent gap, vane thickness) under a sustained
heat load, through a frozen neural surrogate re-checked against the physics, plus a Monte Carlo of machining
tolerances. Imported from PINNeAPPle-apps (`brakecool_ai`); the logic lives in
`pinneapple_design.surrogate_inverse.brake_rotor`, this folder holds routes and the page.

## What it does

- Ground truth: the vanes as a fin array (`pinneapple_physics.closed_form.fin_array_conduction`), grey cast iron
  `k = 50 W/m K`, a ~300 mm rotor (mean circumference 0.754 m, 60 mm radial vanes, 6 mm friction ring), lumped
  convection 40 / 120 / 200 W/m²K for city / highway / mountain descent (assumed values). Extended-surface analysis of
  vented rotors: e.g. Limpert, *Brake Design and Safety*.
- Surrogate on the thermal resistance (log scale), Adam through it in a sigmoid-bounded space with a vane-spacing
  penalty (min 6 mm), integer vane count; verification card against the formula (surrogate error on the rise,
  design rules, an assumed 500 °C cast-iron limit) and its `TrustReport`.
- Uncertainty: vane thickness ±0.05 mm and gap ±0.1 mm (assumed bands, not supplier data), uniform within the band,
  propagated through the formula with `pinneapple_analysis.uncertainty.tolerance.propagate`; verdict on the 95th
  percentile against the 500 °C limit (ROBUST / MARGINAL within 5 % / FRAGILE), safe fraction with its Wilson interval.

## Measured here

| Check | Result |
|---|---|
| Baseline (36 vanes, 8 mm gap, 4 mm, 800 W highway) | 112.4 °C (same as the source) |
| Surrogate held-out error on the rise, 4 seeds | mean 0.5-0.7 %, p95 1.4-2.0 % |
| Error at the optimized design, 4 seeds x 9 conditions | max 3.3 %, median 1.1 %; 36/36 PASS |
| Optimized vs grid search, 800 W | city 133.2 vs 131.8 °C, highway 65.0 vs 64.6 °C, descent 51.5 vs 51.1 °C (baselines 279.5 / 112.4 / 79.0 °C) |
| Monte Carlo, 200 samples | 800 W highway: p95 112.8 °C, ROBUST; 1480 W city: p95 498.0 °C, MARGINAL |

Changed from the source: the surrogate (log resistance, 1500 epochs) and the optimizer (sigmoid-bounded, replacing
Adam at lr 0.4 on raw values) as for HeatSink Design AI; the Monte Carlo now uses the library's tolerance propagation
(numpy sampling, so individual numbers differ from the source's); the CadQuery compound is a triangle mesh (rings and
vanes as separate closed shells); three.js from `/vendor`.

## Limitations

Lumped, steady, uniform `h`: no rotating airflow (CFD), no single hard-stop transient (see the `car_brake_thermal`
PDE preset), no dynamometer data. Tolerance bands and the 500 °C limit are assumptions. The radial colour profile on
the vanes reuses the fin parameters along the radius for display; the scalar model integrates across the gap.

## Run

```bash
cd apps/brakecool_ai && PYTHONPATH=../..:../_shared uvicorn brakecool_ai.api:app --port 8161
```

Environment: `BRAKE_USER` / `BRAKE_PASSWORD` (or `_PASSWORD_HASH`), `BRAKE_EPOCHS` (1500), `BRAKE_THREADS` (1).
Routes: `GET /health`, `/api/status`; `POST /api/evaluate`, `/api/mesh.stl`, `/api/uncertainty` (design + `power_w`,
`cooling_profile`, `n_samples`), `/api/optimize` (`power_w`, `cooling_profile`).
