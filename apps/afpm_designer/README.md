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

## What it computes

| Piece | Library |
|---|---|
| Axial-flux machine on a structured `(r, theta, z)` grid closed round the axis; rotor angle as a relabelling; gap Maxwell-stress torque; tooth search coil | `pinneapple_simulation.electromagnetics.machines3d.afpm` (`AFPM3D`, `quasi3d_slices`) |
| Estimate, check in 3-D, the grid of a design, the 3-D view | `pinneapple_simulation.electromagnetics.machines3d.afpm_design` (`estimate`, `verify_3d`, `grid_for`, `field_view_of`) |
| 3-D edge elements (tree gauge, PARDISO or SuperLU) | `pinneapple_simulation.electromagnetics.fem3d` |
| 3-D view (glTF + description) | `pinneapple_tools.visualization.studio.fieldview.field_view`, `/shared/fieldview3d.js` |

Measured on the default design (12 slots, 8 poles, r = 50 to 90 mm, 1 mm gap, 4 mm magnets, single-sided), on the
coarse grid the app uses (192 steps round the axis, 8 mm radially, 4 mm axially):

| | Quasi-3D estimate | 3-D check | Estimate / 3-D |
|---|---|---|---|
| Cogging, peak to peak | 11.67 N m | 10.37 N m | 1.125 |
| Tooth flux linkage | 1.083 mWb | 1.060 mWb | 1.022 |
| Size, time | 109 440 tetrahedra, 10 s | 196 992 tetrahedra, 15 s | |

On a grid twice as fine (`examples/electrodynamics/24_afpm_quasi3d.py`) the estimate overstates the cogging by
11.3 % and the flux linkage by 2.2 %, so the ratio holds within about a point; the cogging itself is 13 % lower
there (9.20 N m), so treat the app's absolute cogging as a coarse-grid value.

Limits: linear iron and magnets, no load (no winding currents), open rectangular-sector teeth, magnets magnetised
axially. A slot-pole combination whose cogging period needs more than 480 steps round the axis is refused (9 slots
and 8 poles: 576).

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
