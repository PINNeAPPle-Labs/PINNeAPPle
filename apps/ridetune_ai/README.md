# RideTune AI

Lighter cross-section (width x thickness) of a suspension arm modelled as a rectangular cantilever, for a fatigue
life target, found through a frozen neural surrogate of the bending stress and re-checked against the closed form,
plus local sensitivities of life and mass. Imported from PINNeAPPle-apps (`ridetune_ai`); the logic lives in
`pinneapple_design.surrogate_inverse.suspension_arm`.

It is about the arm's structure, not ride dynamics: no spring/damper tuning or vehicle model (for vehicles see
`pinneapple_simulation.vehicle` and `pinneapple_simulation.multibody`).

## What it does

- Ground truth: `pinneapple_physics.closed_form.cantilever_fatigue` (Shigley): `sigma = 6 F L / (w t^2)` at the root,
  Basquin `sigma_a = sigma_f' (2N)^b` turned into an allowable stress; `sigma_f' = 1200 MPa`, `b = -0.09` (table
  values for a spring steel), arm length 0.35 m.
- Surrogate on log(stress); Adam on (w, t) minimising `w t` with a penalty above the allowable stress; verification
  card (surrogate error, life vs target, bounds).
- Sensitivity: central differences (1 µm) on the closed form with `pinneapple_analysis.sensitivity.local_derivatives`,
  elasticities `(x/f) df/dx`. Exact here: life ~ `w^(-1/b) t^(-2/b)`, so thickness is always 2x as influential.

## Measured here

| Check | Result |
|---|---|
| Surrogate held-out error, 4 seeds | mean 0.6-0.9 % |
| 4 load/target cases x 4 seeds | surrogate error <= 1.4 % at the design, life >= target in every case, 16/16 PASS; mass 1.54-3.29 kg vs 3.02 kg baseline |
| Elasticities | -1/b = 11.1 and -2/b = 22.2 within 1e-4 at three design points |

**Honest gap:** the optimizer (Adam, lr 6e-4, 500 steps, as in the source) stops well inside the feasible region:
at 600 N / 3e5 cycles the design lives ~6000x the target and weighs 1.56 kg, while the closed-form minimum (width at its
20 mm bound, thickness ~13 mm where the stress meets the allowable 362 MPa) is about 0.73 kg. The verification card checks safety, not
optimality; the mass is lower than the baseline but not minimal.

## Limitations

Constant rectangular section, one load case, no stress concentrations or 3-D FEA, no endurance limit (Basquin
extrapolates at low stress), no rig or road-load testing.

## Run

```bash
cd apps/ridetune_ai && PYTHONPATH=../..:../_shared uvicorn ridetune_ai.api:app --port 8163
```

Environment: `RIDE_USER` / `RIDE_PASSWORD` (or `_PASSWORD_HASH`), `RIDE_EPOCHS` (1500), `RIDE_THREADS` (1).
Routes: `GET /health`, `/api/status`; `POST /api/evaluate`, `/api/mesh.stl`, `/api/sensitivity` (`width_m`,
`thickness_m`, `load_n`), `/api/optimize` (`load_n`, `target_life_cycles`).
