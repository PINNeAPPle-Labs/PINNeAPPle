# PICT: a differentiable multi-block PISO solver

`pinneapple_simulation.external_solvers.pict` drives [PICT](https://github.com/tum-pbs/PICT) — Franz, Wei, Guastoni
and Thuerey, *A differentiable, GPU-accelerated multi-block PISO solver for simulation-coupled learning tasks in fluid
dynamics*, Journal of Computational Physics 544 (2026) 114433, Apache-2.0 — from PINNeAPPle. PICT solves the
incompressible Navier–Stokes equations with PISO on structured, deformed multi-block grids, keeps its fields as plain
PyTorch tensors, and has hand-written CUDA backward kernels, so a network can sit inside the CFD solve and be trained
through it.

```bash
git clone https://github.com/tum-pbs/PICT.git && cd PICT
cd extensions && python setup.py install      # Linux, an NVIDIA GPU, CUDA (12.8 in the paper's environment)
```

PICT is not on PyPI and there is no `pinneapple[pict]` extra: it is built from source, on a machine with CUDA.

## What the bridge is, and what it is not

It builds the grid, the `PISOtorch.Domain`, the boundaries and the `PISOtorch_simulation.Simulation` configuration
from a case in SI units; it runs the solver; it records the velocity and pressure it reads back and returns them as a
`PICTResult` and a `PhysicalSample`. It adds no physics of its own, and it does not fake a run: on a machine without
PICT, `run_cavity` raises with the install line, and `validate()` returns `{"available": False, "reason": ...}`.

```python
from pinneapple_simulation.external_solvers.pict import (
    CavityCase, PICTRunConfig, pict_availability, run_cavity, validate,
)

pict_availability()          # which part is missing, with the fix, before anything is imported far enough to fail

case = CavityCase(nx=32, ny=32, Re=1000)          # .check() lists problems; .grid() is (1, 2, ny+1, nx+1)
result = run_cavity(case, PICTRunConfig(iterations=100, record_every=10))
result.velocity.shape                             # (snapshots, 2, ny, nx), PICT's channel-first layout
checks = result.checks()                           # continuity, lid, walls, Ghia u and v
sample = result.to_upd()                           # PhysicalSample: u, v, p over (time, x, y)
```

`PICTRunConfig` mirrors the arguments of PICT's `Simulation` (`substeps="ADAPTIVE"`, `corrector_steps`,
`pressure_tol`, `non_orthogonal`, `differentiable`, ...); `differentiable=True` records the steps on the torch graph,
which is what a training loop through the solver needs.

## The grid is ours, and it is checked

PICT's examples build their meshes with a helper (`lib/data/shapes.py`) that is not part of what the package installs
as an importable module, so the bridge carries its own generator, `wall_refined_grid`, and the tests check it rather
than trusting it: the layout is `(1, 2, ny+1, nx+1)`, the corners are exact, the spacing is monotone, the ratio
between neighbouring cells equals `base`, the refinement is symmetric when both walls are named, and naming one wall
packs only that side.

## The checks a cavity result is held to

`cavity_checks(result)` returns one line per check, each with the number it was decided on:

| Check | What it measures |
|---|---|
| `continuity` | the largest discrete `du/dx + dv/dy`, normalised by `U/L`: central differences inside and one-sided at the border, because the largest divergence sits right below the moving lid |
| `lid` | the mean u velocity on the top row against the lid velocity |
| `walls` | the largest speed on the three closed walls (the bottom row and the side columns below the lid) |
| `ghia_u` / `ghia_v` | the centreline profiles against [Ghia, Ghia and Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4), Tables I and II, at the run's Reynolds number |

The reference data are the tables reproduced by PICT's own validation suite (Re = 100, 1000, 5000, 10000); an unknown
Reynolds number raises naming the ones the table has instead of interpolating between them. The profile comparison
interpolates the computed field onto the 17 published stations and reports rms and max, with the rms tolerance a
parameter: a 32×32 run at Re = 1000 is a screenshot, a 256×256 run is a result.

The lid convention matters: the reference assumes the lid at the **y+** wall moving in **+x** (`lid_side="+y"`,
`lid_velocity=1`). Move the lid elsewhere and the profile is mirrored — which the rms will show.

## What is verified here, and what is not

Verified without the solver (`tests/test_pict_bridge.py`, 20 checks): the grid generator, the case validation, the
reference tables' consistency, the centreline reduction (including the even/odd middle line), the Ghia comparison on
exact, shifted and resampled data, a divergence that is small for a solenoidal field and large for a stretching one,
and the boundary checks applied to fields whose answer is known by construction.

**Not verified**: no run of PICT itself. This machine has no NVIDIA GPU and no compiled PICT extensions, so nothing
in this repository has executed the solver. Report a PICT number only with the grid, the Reynolds number and the
checks next to it.
