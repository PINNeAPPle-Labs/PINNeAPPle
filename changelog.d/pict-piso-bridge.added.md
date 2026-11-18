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

**Local engine** (`pict.local_solver`, `run_cavity(..., engine="auto"|"pict"|"local")`): PICT needs Linux, an NVIDIA
GPU and its own CUDA extensions, so on a Mac or a CI box the bridge now runs the cavity with this repository's own
vorticity-streamfunction solver (numpy/scipy) and returns the same `PICTResult` — `engine` says which solver produced
the numbers, and PICT is never claimed for a run it did not do. Measured against Ghia et al. (1982): rms 4.7e-03 /
1.4e-03 / 8.1e-03 on 32², 64² (Re=100) and 128² (Re=1000), divergence at machine precision, 0.1 s / 0.7 s / 23 s. The
engine refuses a wall-refined grid, an off-centre lid and a non-square cavity rather than approximating them, and it
reports no pressure field (recovering one from the momentum balance was tried; the residual did not close, so zeros
with an explicit note are shipped instead).
