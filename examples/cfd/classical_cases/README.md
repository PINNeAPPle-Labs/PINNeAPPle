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

### 6. The same cavity solved by a network — `results/cavity_pinn.json` (#272)

Everything above measures a *discretisation* against Ghia's table. This section measures a method that has no
discretisation at all: a tanh perceptron (4 hidden layers of 64) trained on the strong form of the steady
incompressible equations at random collocation points, with no-slip on the three fixed walls and the moving lid as a
penalty rather than an ansatz, and the pressure held to zero mean. **No solved field is ever shown to it** — not the
OpenFOAM result, not the table.

    python examples/cfd/classical_cases/run_cases.py --only pinn --pinn-steps 10000

It is scored through the same door as the meshes: `cavity.profile_error` at Ghia's own 17 stations, the two wall
stations excluded, the disputed `Re = 400` station dropped, the RMS of the table in the denominator. The network is
asked for `u` at `x = 0.5` itself where the mesh interpolates from two neighbouring columns; the stations and the norm
— the two things that decide whether two numbers may share a column — are identical.

**What was trained, and at what cost.** 4 hidden layers of 64 (12 867 parameters), float32 on a CPU, 2 000 collocation
points redrawn every 200 steps, 250 boundary points per wall, the boundary conditions and the zero-mean pressure as
penalties, Adam at 2e-3 with a cosine decay, **10 000 steps** — 36 minutes of wall time per model on this machine, and
the budget the machine had at the time rather than one chosen to make the table look good.

| model | Re | u error | v error | training |
|---|---:|---:|---:|---:|
| one network per Re | 100 | 16.31 % | 53.99 % | 2 179 s |
| one network per Re | 400 | 54.60 % | 96.97 % | 2 181 s |
| one network per Re | 1 000 | 71.48 % | 98.72 % | 2 176 s |
| one operator, `Re` as an input | 100 | 30.71 % | 89.38 % | 2 179 s, one model for all three |
| one operator, `Re` as an input | 400 | 47.46 % | 96.08 % | — (the same model) |
| one operator, `Re` as an input | 1 000 | 62.12 % | 98.16 % | — (the same model) |

**The result is negative, and it is the finding the issue asked for.** At this budget neither model matches the table
at any of the three Reynolds numbers: not at 5 % — the level a middling mesh reaches in section 1 — and not at 1 %.
`stops_matching` therefore reports `below the lowest measured Reynolds number` for both, which is what it is for: a
bracket between two *measured* numbers, or the statement that there is none. Two things in the rows are worth more
than the headline error:

* **`u` degrades with `Re` and it is wrong everywhere rather than unresolved somewhere.** 16 % at Re = 100, 55 % at
  400, 71 % at 1 000 — and the four stations nearest the moving lid, where a 64-cell grid would be expected to lose the
  layer, carry only 13 %, 32 % and 35 % of the squared error. This is not the boundary layer: the whole column is off.
* **`v` is not learned at the two higher Reynolds numbers.** The strongest cross-flow the network returns is 0.022 at
  Re = 400 and 0.021 at 1 000 against 0.450 and 0.515 in the table — a factor of twenty. That is the family the
  equations admit for free (`u = u(y)`, `v = 0`, `p = p(x)`), where the side walls' no-slip condition is only weakly
  enforced, and it is what the module's `balance` switch (Wang, Teng and Perdikaris, 2021) exists to break out of: tried
  at a quarter of this budget it came out worse (44 % against 34 % in `u` at Re = 100), so it is off in the run above.

The cost is the other half of the comparison: 2 176–2 181 s of CPU per model against 2.1–46.8 s for `simpleFoam` on the
three meshes of section 1, for a worse answer. What this section contributes is not a faster method — it is not one —
but a measurement of how far the method gets at a budget of minutes, on the same ruler as the solver, which is the
number a reader deciding whether to use a PINN for this problem actually needs.

**The operator, asked where the table is silent.** A network whose input carries `Re` can answer anywhere in its
training range, and the three published columns are a thin test of that: they are the points it was asked to
reproduce. So the same network is asked at `Re = 250` and `Re = 750` — no column of Ghia's table has either — and
scored against the case *solved* there (`simpleFoam`, two meshes, `results/cavity_pinn.json` keeps both), through the
same stations, the same exclusion of the walls and the same norm, with the solved profile where the published one
would be.

| Re (no published column) | solver | u error | v error |
|---:|---|---:|---:|
| 250 | 64² and 128², converged in 1 256 / 3 683 iterations | 41.20 % | 95.46 % |
| 750 | 64² and 128², converged in 1 378 / 3 424 iterations | 57.09 % | 97.83 % |

The reference is not what the error is made of: the two meshes differ by 0.44 % in `u` at Re = 250 and 1.85 % at 750
(0.89 % and 2.55 % in `v`), an order of magnitude below the operator's numbers. What those two rows show is not
accuracy — there is none — but that the operator **is** a function of `Re` and not three profiles wearing one hat: its
error at 250 (41 %) lands between the error at the two columns that bracket it (31 % at 100 and 47 % at 400), and 57 %
at 750 between 47 % and 62 %. A model that had fitted the three columns separately would have no reason to be ordered
that way in between.

**Where it stops.** The answer, for both models and at both tolerances, is "below the lowest measured Reynolds
number": the network does not clear 5 % anywhere it was measured, and the honest row is the failure rather than a
crossing nobody observed. Making this a bracket rather than a floor needs a run that converges at `Re = 100` first,
and the budget for that is the open item — not a different reading of these three columns.

### 7. The CFDBench cavity family, scored with the same ruler — `results/cfdbench.json` (#272)

CFDBench (Luo, Chen and Zhang, 2023) is a machine-learning benchmark rather than a validation case, and the issue asks
for its numbers beside ours. It is published on HuggingFace (`chen-yingfa/CFDBench`, licence `apache-2.0` in the
dataset card), one zip per problem. The archive answers at

    https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cavity.zip      # 786 MB; tube 213 MB, dam 1.4 GB

and the addresses that fail are the *per-case* ones: the repository ships seven zips and no loose `u.npy`, so a path
like `.../resolve/main/cavity/bc/case0000/u.npy` is a 404 even though the dataset is there (checked: the dataset page
and the zip both answer 200, that file path does not).

    python examples/cfd/classical_cases/run_cases.py --only cfdbench --cfdbench ~/cfdbench-data/cavity

The 158 cavity cases are read through the project's own loader (`pinneapple_data.datasets.cfdbench`, which was written
against the project's repository layout and is what this case uses — a second reader would measure the reader), the
last frame of each is sampled at Ghia's stations, and the profile is divided by the case's own lid speed so it is in
the table's normalisation. Then it goes through `cavity.profile_error` exactly as our own solution does.

| Re | cases in the archive | u error | v error | u error, lid stations left out |
|---:|---:|---:|---:|---:|
| 100 | 5 | 13.28 % | 9.85 % | 10.89 % |
| 400 | 2 | 19.93 % | 12.54 % | 4.19 % |
| 1 000 | 6 | 25.47–25.55 % | 15.35–15.39 % | 3.43–3.65 % |

Three things had to be got right to read it at all, and each of them shows up as error if it is not:

* **the Reynolds number includes the density.** `case.json` gives `vel_top`, `width`, `density` and `viscosity`, so
  `Re = rho U L / mu`. The `prop` family varies the density *and* the viscosity: read as `U L / nu`, six different
  flows come out as one Reynolds number, and they span 13 % to 44 % against the same column of the table. With the
  density in, the cases that share a Reynolds number agree with each other to the last digit (which is also why the
  table has 5, 2 and 6 rows of the same number: the archive reaches the same flow through different parameter
  combinations, and that is kept rather than deduplicated).
* **the grid is 64 x 64 and it is where the difference is.** The archive interpolates its fields to 64 x 64, and the
  whole of the large error sits in the four stations nearest the moving lid (`y >= 0.9531`) — the layer a 64-cell grid
  cannot resolve. Leaving those four stations out takes the `u` error to 3.4–4.2 % at Re = 400 and 1 000. Both
  readings are reported: quoting only the first blames the data for the grid, quoting only the second hides it. At
  Re = 100 the difference is not confined to the lid (10.9 % without those stations either), and the case is the one
  where the archive's own field is furthest from the table away from any wall.
* **what is on disk is a transient, not a solution.** The cavity cases keep 4 to 1 000 frames (the ones at the three
  Reynolds numbers compared here keep 4 to 10); `results/cfdbench.json` records the frame index and the largest change
  between the last two frames (`drift`, 1.1e-3 to 4.9e-3 in lid speeds) so a reader can see that "the last frame" is
  an approximation of the steady answer rather than the steady answer.

Beside ours, at the finest mesh of the ladder, the same ruler gives 0.31–0.94 % in `u` and 0.83–3.35 % in `v`: the
benchmark's own fields sit an order of magnitude further from the table than a 128 x 128 `simpleFoam` run, which is
what 64 interpolated cells cost, and the reason a dataset built for training surrogates is not a substitute for a
mesh-converged case when the question is the flow.

**The other families load, and the tube has a table of its own.** `cfdbench.families(root)` opens every unpacked
family of the archive through the same adapter and records its case count, subsets and field shape — the cavity (158
cases) and the tube (175, `64 x 64`, 10 to 107 frames) are what has been unpacked here. Only the cavity has a table of
*answers* to be scored against, so only the cavity has a column beside ours. The tube has something weaker, which the
paper does publish and this repository now transcribes:

    python examples/cfd/classical_cases/run_cases.py --only cfdbench-tube --cfdbench-tube ~/cfdbench-data/tube

CFDBench's Table 3 gives the **operating parameters** of the tube's cases — the baseline tube (`l = 1 m`, `d = 0.1 m`,
`rho = 100`, `mu = 0.1`, `u_in = 1 m/s`, velocity inlet, pressure outlet), the fifty inlet velocities `0.1 … 5 m/s` by
`0.1`, the 10 x 10 grid of density `10 … 1000 kg/m3` by `110` against viscosity `0.01 … 1 Pa s` by `0.11`, and the five
diameters by the ten length-to-diameter ratios that make the paper's 25 geometries. It is not a table of answers and
is not treated as one: the tube is a water-air two-phase flow solved in Fluent, so the paper's tube tables (12 and 13)
are neural operators' NMSE against the archive's own fields, and a physical solution to put beside `simpleFoam` does
not exist. What transcribing the parameters buys is that the archive can be held against the paper that describes it,
case by case, and `cfdbench.tube_against_table` does that (`results/cfdbench_tube.json`):

| parameter | on disk | published | agrees |
|---|---|---|---:|
| `bc` inlet velocity | 50 values, 0.1 … 5.0 | 50 values, 0.1 … 5.0 by 0.1 | yes |
| `prop` viscosity | 10 values, 0.01 … 1.0 | 10 values, 0.01 … 1.0 by 0.11 | yes |
| `geo` diameter (`height`) | 5 values | 5 values, 0.01 … 0.5 | yes |
| `geo` ratio (`width/height`) | all 10 | all 10, 1 … 100 | yes |
| `prop` density | **the index 0 … 9** | **10 … 1000 kg/m3 by 110** | **no** |
| case counts (`bc`/`prop`/`geo`) | 50 / 100 / 25 | 50 / 100 / 25 | yes |

The one disagreement is in the data rather than in the transcription, and it matters to anyone reading that family:
its `case.json` stores the **density as the ten-value index**, not as the value, while storing the viscosity as the
value. So `Re = rho U L / mu` computed from `case.json` comes out a hundredfold short for all hundred `prop` cases,
and `cfdbench.reynolds` — which is right for the cavity, whose `prop` family stores the physical density — is not a
Reynolds number for this one. The archive does not state the mapping, so this repository does not assume it: the
family is reported as labelled-with-an-index and is not scored. Everything else in the tube matches Table 3 value for
value, which is the whole of what a family with no published answer can be asked to prove.

## What is not here

* **The `dam` and `cylinder` families of CFDBench.** Not downloaded: 1.4 GB and 12 GB of zips (three for the
  cylinder) against 25 GB free on this machine, and neither has a reference table here to be scored against. The
  reader is the one all four go through and `cfdbench.PROBLEMS` holds the four archives' addresses.
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
| `CFDBENCH_TUBE` | the tube flow's operating parameters: baseline case, 50 inlet velocities, the 10 x 10 density/viscosity grid, 5 diameters x 10 ratios | Luo, Chen and Zhang (2023), arXiv:2310.05963, Table 3 — a table of **cases**, not of answers |

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
