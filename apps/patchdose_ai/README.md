# PatchDose AI

Smallest membrane-controlled release patch (area, membrane thickness, reservoir concentration) for a target release
rate and wear time, found through a frozen neural surrogate and re-checked against Fick's law, plus the release curve
over the wear time. Imported from PINNeAPPle-apps (`patchdose_ai`); the logic lives in
`pinneapple_design.surrogate_inverse.transdermal_patch`.

**A mass-transfer design exercise, not a dosing or medical tool.** The diffusion and partition coefficients are
generic order-of-magnitude values for a small solute in a polymer membrane, not the data of any drug; no patient or
clinical data is involved.

## What it does

- Ground truth: `pinneapple_physics.closed_form.membrane_diffusion` (Robinson & Lee, *Controlled Drug Delivery*):
  `rate = D K C A / L`, depletion time `t = L_res L / (D K)` (area and concentration cancel), linear steady profile
  with sink conditions, a first-order tail after depletion. `D = 5e-13 m²/s`, `K = 1`, 1 mm reservoir.
- Surrogate on log(rate); Adam in a sigmoid-bounded space minimising area with the rate error and the wear shortfall
  (closed form, 3 % margin) as penalties; verification card (surrogate error, target-rate error, wear time, bounds).

## A unit bug found and fixed

The source app and `membrane_diffusion` converted kg to "mg" with a factor of 1000, so every "mg/day" figure was in
grams per day. Fixed in the library (`KG_TO_MG = 1e6`, test checks rate x depletion time = reservoir mass: 64.8
mg/day x 1.852 days = 120 mg for the baseline). The rates of this design space are therefore 0.5 to ~2600 mg/day, and
the request range is now 20-250 mg/day (it was 0.02-0.25). The source README's claim that the model produces rates
"realistic for low-dose transdermal drugs" rested on the wrong unit and is withdrawn.

## Measured here

| Check | Result |
|---|---|
| Surrogate held-out error, 4 seeds | mean 0.9-1.2 % |
| 5 requests (50/1, 100/2, 120/2.5, 200/1.5, 30/4 mg/day / days) x 4 seeds | rate within 0.0-2.5 % of target, every wear time met, 20/20 PASS |
| Release curve | constant phase equals the scalar rate; tail strictly decreasing; cumulative 1.39 x reservoir over a 0.5 t_dep tail |

## Limitations

Steady 1-D membrane with sink conditions: no skin resistance or saturation plateau; the tail is an approximation
(cumulative release can exceed the nominal reservoir by a bounded amount). No in-vitro (Franz cell) or permeation data.
Wear targets above ~4.6 days are unreachable inside the 200 µm membrane bound, so requests stop at 4 days.

## Run

```bash
cd apps/patchdose_ai && PYTHONPATH=../..:../_shared uvicorn patchdose_ai.api:app --port 8162
```

Environment: `PATCH_USER` / `PATCH_PASSWORD` (or `_PASSWORD_HASH`), `PATCH_EPOCHS` (1500), `PATCH_THREADS` (1).
Routes: `GET /health`, `/api/status`; `POST /api/evaluate`, `/api/mesh.stl`, `/api/simulate` (`area_m2`,
`membrane_thickness_m`, `reservoir_conc_kg_m3`), `/api/optimize` (`target_rate_mg_day`, `target_wear_days`).
