# Surrogate Speed

**A trained surrogate answers in microseconds; how much faster is that than a numerical solve, and how far apart are
they?**

Seven fixed plate-fin heat sinks (80×80 mm aluminium base, 3 mm thick, 55-130 W, h = 18-65 W/(m² K)). For each: the
closed-form fin-array temperature, a surrogate prediction (timed, mean of 200 calls) and a finite-volume solve of the
fin unit cell (timed, plus two finer grids for the grid convergence index); the agreement is the difference of the
temperature rise above ambient, with a 10 % decision band.

Ported from the `poc_surrogate_speed` case study of PINNeAPPle-apps. The source project timed its surrogate against a
3-D FEniCS (dolfinx + gmsh) solve run as a subprocess in a separate conda environment, and imported the surrogate and
CAD model from its `heatsink_design_ai` product. Neither FEniCS nor CadQuery is available here; this port uses the
library's own pieces instead (below), so **its speed ratios are of a different reference solver and are not
comparable with the source project's**.

## What it computes

| Piece | Library |
|---|---|
| Closed form (1-D fins with adiabatic-tip correction + base conduction, Incropera & DeWitt) | `pinneapple_physics.closed_form.fin_array_conduction` |
| Reference: base slab + one fin, flux from below, film on the fin and exposed base, symmetry at the cell sides; cell-centred finite volumes (the 3-D cell has nothing depending on the fin depth, so its 2-D section is the same problem) | `pinneapple_simulation.numerical_solvers.fin_cell_fv.solve_fin_cell` |
| Surrogate (MLP 3×48 tanh on 6,000 closed-form samples), timings, agreement, grid check, designs | `pinneapple_design.thermal.fin_array_speed` |
| Heat-sink mesh for the 3-D view | `pinneapple_design.thermal.meshes.plate_fin_mesh` |

Measured here (one sweep, heavily shared 4-core container):

- Finite-volume reference vs closed form: within 0.4 % on six designs, 1.26 % on the sparse thick fins; baseline
  48.11 °C (the source project's FEniCS cell gave 48.08 °C); grid convergence index ≤ 1.4·10⁻⁴; heat balance to
  round-off.
- Surrogate: held-out error 5.7 % (mean |error| / T over 600 samples); against the reference **5.2 % mean, 10.6 %
  worst** (dense thin fins), so **6 of 7 designs inside the 10 % band, 1 outside**; mean absolute error 1.4 °C, worst
  2.5 °C. The source project reported 4.46 % mean / 7.35 % worst for its own surrogate; not reproduced here.
- Time: surrogate 33-138 µs per call, finite-volume solve 1.7-8.5 ms, **ratio 29-190× (mean 84×)**. The source
  project's 176,267× / 171,187× / 227,059× mean speed-ups were against a 0.6-5.9 s FEniCS subprocess and are not
  reproduced here. Surrogate training took 174 s at startup on this machine.

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/surrogate_speed
PYTHONPATH=../_shared uvicorn surrogate_speed.api:app --port 8165      # open http://localhost:8165
```

The surrogate trains and a sweep starts at startup (background thread). `GET /api/sweep_status`,
`POST /api/run_sweep`, `GET /api/mesh.json?design=`, `GET /api/thermal_field?design=`, `GET /health`.

Environment: `SSPEED_USER` / `SSPEED_PASSWORD` (or `SSPEED_PASSWORD_HASH`), `SSPEED_AUTOSTART=0`,
`SSPEED_TRAIN_EPOCHS` (default 400), `SSPEED_GRID_CHECK=0` to skip the finer grids.

## Limitations

- Uniform film coefficient, no fan or flow model, no spreading from a small die, no radiation; the closed form and
  the reference share these assumptions, so their agreement says nothing about the convection model.
- The surrogate is trained over 15-150 W; relative errors grow at small temperature rises.
- Timings depend on the machine and its load; the ratio depends on the reference solver chosen.
