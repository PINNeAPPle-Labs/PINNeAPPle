# HeatSink Design AI

Gradient-based inverse design of a straight-fin CPU heat sink (fin count, height, thickness on a fixed 80 x 80 mm
aluminium base) through a frozen neural surrogate, with every answer re-checked against the physics it learned.
Imported from PINNeAPPle-apps (`heatsink_design_ai`); the logic now lives in
`pinneapple_design.surrogate_inverse.heatsink_fins` and this folder holds only routes and the page.

**Not the HeatSink Sizer.** [`apps/heatsink_sizer`](../heatsink_sizer/README.md) sizes plate-fin heat sinks from
published forced / natural-convection correlations, pressure drop and a 3-D base-plate solve. This app uses the 1-D
fin-array formula with a *prescribed* convection coefficient per fan class: it demonstrates the surrogate +
inverse-design + verification loop, it does not size a real heat sink.

## What it does

1. Ground truth: `pinneapple_physics.closed_form.fin_array_conduction` (Incropera & DeWitt fin efficiency with an
   adiabatic-tip correction, convective + 1-D base resistance), `k = 205 W/m K`, `h` = 18 / 35 / 65 W/m²K for the
   three fan profiles (assumed values).
2. Surrogate: an MLP on the thermal resistance `R = (T_j - T_amb) / P` (log scale; the formula is exactly linear in
   power), 6000 samples, 1500 epochs, trained in a background thread at start-up.
3. Inverse design: Adam on (N, H, t) through the frozen surrogate in a sigmoid-bounded space, a penalty on the
   minimum fin spacing, integer N stepped down until the spacing rule holds.
4. Verification card: the formula re-run on the final design; surrogate error graded on the **temperature rise above
   ambient** (PASS <= 5 %, WARNING <= 15 %), every design rule, a plausibility bound; also as a
   `pinneapple_analysis.trust.TrustReport` (airflow and measurement listed as *not run*).
5. Cross-check of the formula: `pinneapple_design.thermal.fin_cell`, 2-D finite volumes in one fin pitch (the 3-D
   unit cell of a periodic array is exactly 2-D when nothing varies along the depth), with energy balance and a
   grid-refinement check.

## Measured here (`tests/pinneapple_design/`, numbers from this repository)

| Check | Result |
|---|---|
| Surrogate held-out error on the rise, 4 seeds | mean 1.2-1.8 %, p95 3.6-5.3 % |
| Surrogate error at the optimized design, 4 seeds x 9 conditions (3 fans x 35/65/120 W) | max 3.9 %, median 1.5 %; 36/36 PASS |
| Optimized vs grid search (seed 0, 9 conditions) | within 0.06-0.52 °C of the grid optimum; 65 W standard fan: 32.83 °C vs 32.77 °C (baseline 48.12 °C) |
| Formula vs 2-D finite volume | 0.07 % (baseline), 0.27 % (50 fins x 0.4 mm), 0.03 % (30 x 1.5 mm); **8.2-8.8 % with 3-5 fins** (see limitations) |

### The TODO item "surrogate errs 9.1 %, threshold 5 %"

Reproduced on the source code as imported: at the UI default (65 W, standard fan, seed 0) the source surrogate
erred 7.7 % (WARNING); over 4 seeds x 9 conditions its error at the optimized design ranged 0.8-29.8 %, held-out mean
5.6-16.0 % (400 full-batch epochs on raw temperature). The exact 9.1 % depends on seed and condition and was not
re-obtained. The threshold is unchanged (5 %) and the metric is *stricter* now (relative to the temperature rise, not
to the absolute °C). The fix is in the surrogate: learn `log R` with inputs scaled to their bounds and a cosine
learning-rate decay for 1500 epochs (results above).

### Changed from the source

- The source optimizer (Adam, lr 0.4 on raw SI values) pinned height and thickness to their bounds at the first step
  and ended at N = 19, t = 3 mm: 39.15 °C at 65 W / standard fan against 32.77 °C for a grid search. Replaced by the
  sigmoid-bounded descent above (32.83 °C).
- The dolfinx/gmsh 3-D FEA cross-check (a subprocess into a conda env at a hard-coded macOS path) is replaced by the
  finite-volume fin cell. The source's reported FEA agreement (48.08 °C vs 48.12 °C) was not reproduced here (no
  FEniCS); the finite-volume cell gives 48.10 °C.
- The CadQuery solid is replaced by a triangle mesh from `pinneapple_design.geometry.bodies` (fins and base are
  separate closed shells, not fused). No CadQuery dependency.
- three.js comes from the repository's vendored copy (`/vendor`), not a CDN; login and security headers via appkit.

## Limitations

- Fixed convection coefficient: no airflow, no fin-to-fin flow interaction, no bypass. Not for sizing hardware.
- The fin-cell check gives each fin `P/N` over one pitch; with few, widely spaced fins this differs from the
  formula's footprint by the array edge effect (8-9 % at 3-5 fins). It checks the fin theory, not the airflow.
- Steady state; no measurement on a built heat sink.

## Run

```bash
cd apps/heatsink_inverse_ai
PYTHONPATH=../..:../_shared uvicorn heatsink_inverse_ai.api:app --port 8160
# or: docker build -f apps/heatsink_inverse_ai/Dockerfile -t heatsink-design-ai .   (from the repository root)
```

Environment: `HSD_USER` / `HSD_PASSWORD` (or `HSD_PASSWORD_HASH`), `HSD_EPOCHS` (1500), `HSD_THREADS` (1).

| Method | Path | Body |
|---|---|---|
| GET | `/health`, `/api/status` | |
| POST | `/api/evaluate`, `/api/mesh.stl`, `/api/cross_check` | `{"fin_count": 20, "fin_height_m": 0.025, "fin_thickness_m": 0.001, "power_w": 65, "cooling_profile": "standard_fan"}` |
| POST | `/api/optimize` | `{"power_w": 65, "cooling_profile": "standard_fan"}` (503 until the surrogate is trained) |
