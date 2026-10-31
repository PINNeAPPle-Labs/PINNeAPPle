# Axial-Flux Motor Designer

**A quick quasi-3D estimate of an axial-flux PM machine, then "Verify in 3-D": the same design solved in full 3-D on
the same grid, with what the estimate left out and a 3-D view of the field.**

Set the poles, slots, radii, gap, magnets, yokes, teeth and topology (single-sided, or double rotor around a yokeless
stator). **Estimate** cuts the active annulus with flux-parallel faces at the inner and outer radius (the usual
quasi-3D model) and gives the cogging over one period, its peak-to-peak, the no-load flux linkage of a coil around a
stator tooth and the back-EMF constant per tooth turn. **Verify in 3-D** solves the same design with air inside,
outside, above and below, on the same edge-element grid, and gives the same numbers with the ratio estimate / 3-D:
the fringing at the inner and outer radius that quasi-3D leaves out. The 3-D view shows `|B|` and the axial field on
the magnets, the iron and the gap layers.

![The default design after "Verify in 3-D": the results table with the ratios, the cogging over one period and the
3-D view of |B|](../../docs/assets/apps/afpm-designer-verify-3d.png)

## What it computes

| Piece | Library |
|---|---|
| Axial-flux machine on a structured `(r, theta, z)` grid closed round the axis; rotor angle as a relabelling; gap Maxwell-stress torque; tooth search coil | `pinneapple_simulation.electromagnetics.machines3d.afpm` (`AFPM3D`, `quasi3d_slices`) |
| Estimate, check in 3-D, the grid of a design, the 3-D view | `pinneapple_simulation.electromagnetics.machines3d.afpm_design` (`estimate`, `verify_3d`, `grid_for`, `field_view_of`) |
| 3-D edge elements (tree gauge, PARDISO or SuperLU) | `pinneapple_simulation.electromagnetics.fem3d` |
| 3-D view (glTF + description) | `pinneapple_tools.visualization.studio.fieldview.field_view`, `/shared/fieldview3d.js` |

Magnet and tooth edges are shares of the grid cells they cut (`AFPM3D(edges="fraction")`), so the magnet arc and the
slot opening are the ones entered, and the rotor turns by any angle: the cogging is sampled at 16 angles per period
(32 give the same peak-to-peak). With whole cells the coarse grid moved the magnet arc in steps of 1/12 of the pole
pitch (the cogging of the default design jumped from 0.52 to 11.67 N m between an arc of 0.79 and 0.80) and turned
the rotor by even steps only, 4 angles per period, which put the peak-to-peak 29 % low (6.13 against 8.60 N m).

Measured on the default design (12 slots, 8 poles, r = 50 to 90 mm, 1 mm gap, 4 mm magnets over 80 % of the pole
pitch, single-sided), 16 angles per period:

| Grid (steps round the axis, radial x axial) | Cogging pk-pk, estimate / 3-D (N m) | Estimate / 3-D | Tooth flux linkage, estimate / 3-D (mWb) | Estimate / 3-D | Time, estimate / 3-D |
|---|---|---|---|---|---|
| 192, 8 x 4 mm (the app) | 8.60 / 7.63 | 1.128 | 1.063 / 1.041 | 1.021 | 21 s / 34 s |
| 384, 8 x 4 mm | 10.62 / 9.61 | 1.106 | 1.070 / 1.048 | 1.021 | 49 s / 81 s |
| 768, 8 x 4 mm | 11.56 / 10.50 | 1.101 | 1.073 / 1.051 | 1.021 | |
| 384, 4 x 2.5 mm | 11.15 / 9.96 | 1.119 | 1.076 / 1.053 | 1.022 | 133 s / 222 s (9.2 GB) |

The ratio estimate / 3-D, the app's result, stays within 1.10 to 1.13 for the cogging and 1.021 to 1.022 for the flux
linkage on every grid. The absolute cogging is not converged on the app's grid: it rises by 2.0 and then 0.9 N m in
quasi-3D as the steps round the axis double, and the app's 3-D value is 27 % below the 768-step one. Treat the app's
absolute cogging as a coarse-grid value. Times on 4 cores with nothing else running (the 16 angles cost 2.5 times
the 4 of whole cells: 8.5 s and 13 s).

Limits: linear iron and magnets, no load (no winding currents), open rectangular-sector teeth, magnets magnetised
axially. A slot-pole combination whose cogging period needs more than 480 steps round the axis is refused (9 slots
and 8 poles: 576).

Screenshots of the default design, taken with Playwright against the running app: the form
(`docs/assets/apps/afpm-designer-form.png`), after **Estimate** (`afpm-designer-estimate.png`), after **Verify in 3-D**
(`afpm-designer-verify-3d.png`) and the 3-D view alone (`afpm-designer-field-view.png`).

## Run it

```bash
# from the repository root
pip install . -r apps/afpm_designer/requirements.txt
cd apps/afpm_designer && PYTHONPATH=../_shared uvicorn afpm_designer.api:app --port 8191
# or
docker build -f apps/afpm_designer/Dockerfile -t afpm-designer . && docker run -p 8191:8191 afpm-designer
```

`POST /api/estimate` and `POST /api/verify` take the design (`GET /api/defaults` for the fields and ranges) and queue a
job; `GET /api/jobs/{id}` gives its progress and result; `GET /api/view/{token}` and `/api/view/{token}.glb` the 3-D
view of a finished run. Login and the other settings are those of every app (`apps/_shared/README.md`, prefix
`AFPM`).

Tests: `tests/test_app_afpm_designer.py` (the page, refused designs, a check in 3-D against the measured numbers, the
view).
