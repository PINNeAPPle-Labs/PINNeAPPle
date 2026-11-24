# Classical CFD validation cases, measured against their published tables

Every surrogate in PINNeAPPle is scored against a reference. This folder is where the reference is checked: the
published numbers are transcribed as data with the table they came from, an OpenFOAM case is written for each problem,
and the answer is compared with the table station by station. The numbers below are the output of
`run_cases.py` (`results/*.json` next to this file) on OpenFOAM v2606, macOS arm64.

    python examples/cfd/classical_cases/run_cases.py --only reference          # the tables, no solver
    python examples/cfd/classical_cases/run_cases.py --only cavity --re 400
    python examples/cfd/classical_cases/run_cases.py --only cavity --reuse     # read the cases already solved
    python examples/cfd/classical_cases/run_cases.py                           # every case (~40 min of solver time)

Without OpenFOAM the reference part still runs and the solver parts say they were skipped. `--cases` chooses where the
OpenFOAM cases are written (default `/tmp/pinneapple-cfd-cases`); they are never committed.

## What was measured

### 1. Lid-driven cavity — Ghia, Ghia and Shin (1982), `results/cavity.json`

`simpleFoam`, laminar, `linearUpwind` for momentum, `n x n x 1` uniform cells, `Re = U L / nu` with the lid at
`U = 1`, converged to a normalised residual of `1e-6`. Errors are the L2 of the difference against the published
centreline profiles at Ghia's own 17 stations, relative to the RMS of the table, **excluding the two wall stations**
(there the published value *is* the boundary condition, not a solved value).

| Re | cells | iterations | wall time | u error | v error |
|---:|---:|---:|---:|---:|---:|
| 100 | 1 024 | 415 | 3.2 s | 0.38 % | 2.78 % |
| 100 | 4 096 | 1 232 | 6.6 s | 0.45 % | 3.32 % |
| 100 | 16 384 | 3 810 | 46.8 s | 0.56 % | 3.35 % |
| 400 | 1 024 | 444 | 2.9 s | 3.86 % | 6.41 % |
| 400 | 4 096 | 1 165 | 5.2 s | 0.87 % | 0.89 % |
| 400 | 16 384 | 3 285 | 38.3 s | 0.31 % | 0.83 % |
| 1 000 | 1 024 | 693 | 2.1 s | 11.81 % | 13.02 % |
| 1 000 | 4 096 | 1 638 | 5.4 s | 2.74 % | 1.49 % |
| 1 000 | 16 384 | 3 953 | 45.9 s | 0.94 % | 1.74 % |

At Re = 400 and 1000 the error against the table falls cleanly with the mesh — the table is doing its job as a
reference. At Re = 100 it does not: it sits at 0.4–0.6 % on every mesh. That is the **floor of the comparison**, not a
failure of the mesh study, and the two effects are separated by refining the solution against *itself*:
`richardson_extrapolate` over the three meshes gives

| Re | observed order (u) | GCI of the fine mesh | asymptotic |
|---:|---:|---:|---|
| 100 | 1.10 | 1.54 % | yes |
| 400 | 1.34 | 2.06 % | yes |
| 1 000 | 1.43 | 3.38 % | yes |

The observed order is close to **one**, not the two a `linearUpwind` scheme is nominally worth: the upwind bias costs
the order on a mesh this coarse, and the near-wall stations are interpolated from cell centres whose spacing there is
the largest error term. That is what the ladder measures, and it is the number to quote for this case rather than the
nominal order of the scheme.

Re = 10000 is in the transcribed table and **not** run: the steady incompressible solver on these meshes does not have
that flow's steady solution to find, and reporting a number there would report the solver's failure as agreement. The
honest result is that the case stops here, at Re = 1000, which is also where Ghia's own tables are unambiguous.

**One transcribed station is disputed.** At `Re = 400`, `x = 0.9063`, the `v` table says `-0.23827` and the converged
128x128 solution says `-0.3870`; every other station of that column agrees to 0.005 or better, and the neighbours of
both values put the true profile near `-0.38`. One of the two rows is wrong and the paper has to settle which, so the
value is left exactly as transcribed, the station is recorded in
`reference.disputes()`, and it is excluded from the L2 above rather than deleted from the record.

### 2. Backward-facing step — Armaly et al. (1983), `results/step.json`

`simpleFoam`, laminar, expansion ratio 1 : 1.9423 (measured 1.941–2.0 depending on the mesh), parabolic inlet of mean
`U = 1`, `Re = U·2h_i/ν` on the inlet hydraulic diameter, reattachment read off the first cell row as the sign change
of the streamwise velocity. The mesh is refined by two in cells per step height.

| Re | 8 cells/S | 16 cells/S | 32 cells/S |
|---:|---:|---:|---:|
| 100 | 2.818 | 2.825 | 2.849 |
| 200 | 4.751 | 4.831 | 4.870 |
| 400 | 7.423 | 8.018 | 8.137 |

Every run converged (237 to 4 251 iterations, 4.8 s to 352 s). The reattachment length moves by 0.8 % from 8 to 16
cells per step height and by 0.1 % from 16 to 32 at Re = 200, so it is converged on the finest mesh to about 1 %; at
Re = 100 the three meshes agree to 1.1 % already. At Re = 400 the coarse mesh is 9 % short of the fine one — the
recirculation there is long enough (`x1/S ~ 8`) that 8 cells per step height does not resolve it, which is the rung
where a coarse run stops being a stand-in for a fine one.

**Re = 800 does not converge**, and that is reported rather than tabulated: the 8 and 16 cells-per-step runs reached
the 20 000-iteration cap with the residual still moving (the 32-cell run was stopped to keep the campaign bounded).
Armaly's own data go three-dimensional and transitional above `Re ~ 400`, so a 2D steady solver having nothing to
converge to there is consistent with the experiment rather than a defect of the case. **The published curve is not transcribed**: Armaly's reattachment length is published as a
figure and the digitisation that circulates in the literature is a secondary source with no single provenance, so
`reference.ARMALY_REATTACHMENT` is empty and there is no error column here. Filling it in from the paper is a
one-line change to that dictionary.

### 3. NASA TMR 2D zero-pressure-gradient flat plate — `results/tmr_flat_plate.json`

`simpleFoam`, `Re = 5e6` per unit length (the case page of `benchmark_suites.NASA_TMR_CASES`), plate on `x ∈ [0, 1]`,
inflow a third of a chord upstream, far field at `y = 0.5`, three grids refined by two, wall functions with the first
cell at `y+ = 30`. The skin friction is the `wallShearStress` function object's answer at `x = 0.97`, which is the
value the solver itself used.

| model | coarse (720 cells) | medium (2 880) | fine (11 520) | observed order | Richardson-extrapolated | GCI of the fine | 1/7-power law |
|---|---:|---:|---:|---:|---:|---:|---:|
| SpalartAllmaras | 0.002932 | 0.002975 | 0.002985 | 2.10 | **0.002988** | 0.13 % | 0.002994 |
| kOmegaSST | 0.002659 | 0.002678 | 0.002684 | 1.70 | **0.002687** | 0.12 % | 0.002994 |

Both converge, at an observed order of 2.1 and 1.7, and the extrapolated values land within 0.3 % (SA) and 10 % (SST)
of the 1/7-power turbulent flat-plate law (`cf = 0.027 Re_x^-1/7`, Schlichting) — an independent check that needs no
data file. The 10 % is the model, not the mesh: the fine grid's GCI is 0.12 %.

**The first version of this case never converged**, and the reason is worth recording because OpenFOAM keeps quiet
about it: the relaxation entry was written as `kOmegaSST 0.7` — the *model's* name — where OpenFOAM reads a per
**field** entry, so `k` and `omega` were left at a relaxation factor of 1.0 and the residuals oscillated between 0.11
and 0.26 for six thousand iterations. With `k 0.7; omega 0.7` the same case converges in 153 to 665 iterations. The
second lesson is in `skin_friction`: estimating `tau_w` from the first cell as `(nu + nut) du/dy` over-predicts by
5.6x (SA) and 2.0x (SST) at `y+ = 30`, because the eddy viscosity of a log-layer cell is not the wall's.

**Not compared with CFL3D/FUN3D.** The TMR grids are regenerated here rather than converted from the published Plot3D
files, and the reference tables are not in this repository, so there is no error column against the TMR itself —
`tmr.read_reference(path)` reads one when it is supplied.

### 4. Surrogate warm start — `results/warm_start.json` (#494)

A parametric surrogate over Reynolds number (POD of six solved cavities at `Re = 50 … 800` on 64 x 64, mode coefficients
interpolated in `Re`) writes the predicted `U` and `p` into `0/` — the `internalField` only, the boundary conditions
untouched — and the case is solved cold and warm at two Reynolds numbers held out of the fit:

| Re (held out) | cold | warm | iterations | wall time | largest field difference |
|---:|---:|---:|---:|---:|---:|
| 400 | 1 165 it, 7.4 s | 694 it, 4.5 s | **1.68x** | **1.63x** | 2.0e-5 (U), 2.8e-5 (p) |
| 1 000 | 1 638 it, 7.5 s | 1 109 it, 5.5 s | **1.48x** | **1.37x** | 6.3e-4 (U), 1.0e-3 (p) |

Both runs converge to the same answer to the residual tolerances they stop at, which is the condition that makes the
saving a saving: the solver still produces the answer. The speedup is smaller than the 1.9-7.4x reported for
CFDNet (Obiols-Sales et al., ICS 2020) — that work warm-starts RANS cases whose potential-flow start is much further
from the answer than a POD prediction of the same case is.

### 5. The low-fidelity ladder — `results/ladder.json` (#294)

NACA 0012 at `alpha = 0, 4, 8` degrees, `Re = 4e6` (k-omega SST, the app's O-grid), three RANS resolutions plus the
closed-form rung (`CL = 2 pi sin(alpha)`). "Error" is against the finest RANS level, which is the rung's job: being a
cheaper stand-in for the expensive one.

| rung | cells | wall time per design | mean abs. CL error | mean abs. CD error | Spearman CL | Spearman CD |
|---|---:|---:|---:|---:|---:|---:|
| thin-airfoil | 0 | ~0 s | **0.0036** | — (no drag) | 1.00 | — |
| rans-coarse | 2 624 | 20.3 s | 0.0238 | 0.0045 | 1.00 | 1.00 |
| rans-medium | 5 130 | 37.5 s | 0.0070 | 0.0014 | 1.00 | 1.00 |
| rans-fine | 10 368 | 79.6 s | — (reference) | — | 1.00 | 1.00 |

The result that decides how this ladder should be used: the **closed-form rung is the most accurate of the three that
are not the reference** — 0.0036 in `CL` against the coarse mesh's 0.0238 — at a hundredth of a second against twenty.
For a thin symmetric section below stall, thin-airfoil theory beats a coarse RANS run at ranking the designs, and the
coarse run only earns its twenty seconds when the drag, the separation or the stall margin is what the design is
being chosen on.

The column that matters for a design search is the rank correlation, and every rung ranks these three designs exactly
as the fine one does. Calibrating the coarse rung on two designs (`alpha = 0` and `4`) gives an offset of `+0.0071` in
`CL` and takes its mean absolute error from 0.0238 to 0.0215; the ranking does not move, which is what a nearly
constant bias does.

Relative errors are **not** quoted for `CL`: the section's lift crosses zero between two of these designs, so a
relative error there divides by nothing (the printed 34 % and 30 % are that artefact, and the reason the table carries
absolute differences). At `alpha = 4` the absolute differences are 0.014 (coarse) and 0.004 (medium); at `alpha = 8`,
0.057 and 0.017.

The vortex-lattice and LBM rungs are **not** on this family: a lattice gives a finite wing's `CL`, not a section's, and
the LBM solver works in a low-Reynolds regime while this family is at `Re = 4e6`. `ladder.MULTIFIDELITY` exports the
rungs as data levels (name and relative cost) for the multi-fidelity workflow (#35).

## What is not here

* **A PINN for the cavity at Re = 100, 400, 1000** (#272's other half). Not done: the tables, the cases and the mesh
  study are the part that had to exist first, and a PINN trained here would have been scored against the same table
  this file reads. The parametric operator over `Re` **is** done, as the surrogate of section 4: it predicts the field
  at a `Re` it was not fitted on, and its error against the solver is reported there.
* **The CFDBench subset.** Not done: the licence could not be confirmed offline, and the dataset is not in this
  repository.
* **The TMR bump-in-channel, NACA 0012 and backward-facing-step cases of #265**, and the TMR reference tables: the
  flat plate is the case whose verification stands without an external file, and the other three need the published
  grids and tables this machine could not download. `tmr.read_reference(path)` is the reader for when they arrive.
* **The VLM and LBM rungs on the same family as the RANS rungs** (#294), for the reasons in section 5.

## The tables

`pinneapple_simulation.cfd_validation.reference` holds them, as data:

| table | what | citation |
|---|---|---|
| `GHIA_U`, `GHIA_V` | `u` along `x = 0.5` and `v` along `y = 0.5`, Re = 100, 400, 1000, 10000 | Ghia, Ghia and Shin (1982), *J. Comput. Phys.* 48(3), 387–411, Tables I–II, doi:10.1016/0021-9991(82)90058-4 |
| `DEV_VAHL_DAVIS_NU` | mean Nusselt number of the square cavity, Ra = 1e3…1e6 | de Vahl Davis (1983), *Int. J. Numer. Methods Fluids* 3(3), 249–264, Table I, doi:10.1002/fld.1650030305 |
| `ARMALY_REATTACHMENT` | reattachment length against Re | Armaly et al. (1983), *J. Fluid Mech.* 127, 473–496 — **empty**: the published curve is a figure |

The tables are checked on their own, with no solver: the lid moves at `u = 1`, the walls do not move, the coordinates
increase, `u` crosses the centreline exactly once, and the strongest backflow is below the centreline
(`tests/test_cfd_validation.py`). A transcription error that survives those checks would have to be a plausible number
in the wrong row — which is exactly what the Re = 400 station above turned out to be, and why the runs are part of the
check too.

## References

* U. Ghia, K.N. Ghia, C.T. Shin, "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a
  Multigrid Method", *J. Comput. Phys.* 48(3), 387–411, 1982.
* B.F. Armaly, F. Durst, J.C.F. Pereira, B. Schönung, "Experimental and theoretical investigation of backward-facing
  step flow", *J. Fluid Mech.* 127, 473–496, 1983.
* G. de Vahl Davis, "Natural Convection of Air in a Square Cavity: A Bench Mark Numerical Solution", *Int. J. Numer.
  Methods Fluids* 3(3), 249–264, 1983.
* P.J. Roache, "Perspective: A Method for Uniform Reporting of Grid Refinement Studies", *ASME J. Fluids Eng.*
  116(3), 405–413, 1994 (the GCI, in `pinneapple_analysis.verification.convergence`).
