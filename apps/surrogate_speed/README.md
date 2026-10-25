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
library's own pieces instead (below; the heat sink and its surrogate are the ones `apps/heatsink_inverse_ai` uses), so **its speed ratios are of a different reference solver and are not
comparable with the source project's**.

## What it computes

| Piece | Library |
|---|---|
| Heat sink, closed form (1-D fins with adiabatic-tip correction + base conduction, Incropera & DeWitt) and surrogate (MLP 3×48 tanh on log thermal resistance, 6,000 closed-form samples, 1,500 epochs), mesh for the 3-D view | `pinneapple_design.surrogate_inverse.heatsink_fins` (shared with `apps/heatsink_inverse_ai`) on `pinneapple_physics.closed_form.fin_array_conduction` |
| Reference: one fin pitch (base slab + fin, flux from below, film on the fin and exposed base, symmetry at the cell sides), cell-centred finite volumes with a grid-halving check (the 3-D periodic cell has nothing depending on the fin depth, so its 2-D section is the same problem) | `pinneapple_design.thermal.fin_cell.solve_fin_cell` |
| Designs, timings, agreement | `pinneapple_design.thermal.fin_array_speed` |

Measured here (one sweep, single-threaded torch on a shared 4-core container):

- Finite-volume reference vs closed form: within 0.4 % on six designs, 1.28 % on the sparse thick fins; baseline
  48.10 °C (the source project's FEniCS cell gave 48.08 °C); grid-halving change ≤ 1.9·10⁻⁴; energy balance to
  round-off.
- Surrogate: held-out error 1.7 % (on the thermal resistance); against the reference **2.0 % mean, 4.0 % worst**,
  all 7 designs inside the 10 % band; mean absolute error 0.7 °C, worst 1.8 °C. The source project reported 4.46 %
  mean / 7.35 % worst for its own surrogate.
- Time: surrogate 59-96 µs per call, finite-volume solve with its grid check 21-89 ms, **ratio 256-926× (mean 618×)**.
  The source project's 176,267× / 171,187× / 227,059× mean speed-ups were against a 0.6-5.9 s FEniCS subprocess and
  are not reproduced here. Surrogate training took 11 s at startup.

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/surrogate_speed
PYTHONPATH=../_shared uvicorn surrogate_speed.api:app --port 8185      # open http://localhost:8185
```

The surrogate trains and a sweep starts at startup (background thread). `GET /api/sweep_status`,
`POST /api/run_sweep`, `GET /api/mesh.json?design=`, `GET /api/thermal_field?design=`, `GET /health`.

Environment: `SSPEED_USER` / `SSPEED_PASSWORD` (or `SSPEED_PASSWORD_HASH`), `SSPEED_AUTOSTART=0`,
`SSPEED_TRAIN_EPOCHS` (default 1500), `SSPEED_GRID_CHECK=0` to skip the grid-halving check.

## Limitations

- Uniform film coefficient, no fan or flow model, no spreading from a small die, no radiation; the closed form and
  the reference share these assumptions, so their agreement says nothing about the convection model.
- The surrogate is trained over 15-150 W; relative errors grow at small temperature rises.
- Timings depend on the machine and its load; the ratio depends on the reference solver chosen.
