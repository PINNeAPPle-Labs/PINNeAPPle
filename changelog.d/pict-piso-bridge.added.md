**PICT bridge** (`pinneapple_simulation.external_solvers.pict`): the tum-pbs differentiable multi-block PISO solver
(Journal of Computational Physics 544 (2026) 114433, Apache-2.0) as a cavity case, a runner and a set of checks.
The grid generator is PINNeAPPle's own (PICT's lives in its example scripts and is not importable from the installed
package) and is checked: layout `(1, 2, ny+1, nx+1)`, exact corners, monotone spacing, geometric ratio, symmetric
when both walls are named. `run_cavity(CavityCase(...), PICTRunConfig(...))` returns a `PICTResult` with the recorded
velocity and pressure, a `PhysicalSample` and `checks()`: discrete continuity (central inside, one-sided at the
border, where the largest divergence sits below the moving lid), the lid and the closed walls, and the centreline
profiles against Ghia, Ghia and Shin (1982) Tables I-II, with the rms tolerance left to the caller. `pict_availability`
names the missing part and `run_cavity` raises instead of returning numbers when PICT is absent: nothing here ran the
solver, which has no CUDA device on this machine, and the docs say so. 20 checks in `tests/test_pict_bridge.py` cover
everything that does not need the solver. Docs: `docs/core_concepts/pict_piso_solver.md`.
