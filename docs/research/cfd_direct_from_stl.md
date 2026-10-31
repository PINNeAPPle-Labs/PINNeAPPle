# CFD directly on the polygon geometry (STL), without a body-fitted mesh: pre-registration

Pre-registered on 2026-10-31, before any of the runs below, with the same protocol as
[condensation_falsification.md](condensation_falsification.md). This file states the hypotheses, the metrics, the
success criterion and the **kill criterion** of each. It is committed before the measurements. Results and deviations
are appended at the end, under the original text; nothing above the results section changes after the runs, except
obvious typos, which are listed in the change log at the end.

**Question.** Can incompressible CFD run directly on a polygon geometry (an STL surface in 3-D, a closed polyline in
2-D) without generating a mesh that follows the surface, be faster than CFD on such a mesh **in total time**, and give
similar results?

What is already known:
- point clouds (RBF-FD / GFDM) and the FV + point-cloud hybrid were refuted in #900: errors up to 31x the FV error,
  GFDM unstable. They are not re-tested here;
- `pinneapple_simulation.immersed` (#981, #1001, #1002) solves heat and elasticity on SDF / voxel / STL geometry
  with cut-cell quadrature, ghost penalty and Nitsche conditions, at optimal rates. It has no flow solver;
- `pinneapple_simulation.numerical_solvers.lbm_tunnel` (D3Q19/D3Q27, TRT, Smagorinsky, STL voxelisation, force by
  momentum exchange) has never been compared with body-fitted CFD in time or accuracy.

## Machine and conditions (declared up front)

- This container: 4 cores (Intel Xeon @ 2.10 GHz), 15 GB RAM, no swap, no GPU. Every method runs on the CPU with
  4 threads (`torch.set_num_threads(4)`, `OMP_NUM_THREADS=4`, BLAS 4 threads).
- Each timed run is a fresh subprocess. Recorded per run: wall time by stage, peak resident memory
  (`resource.getrusage(RUSAGE_SELF).ru_maxrss` of that subprocess; for the LBM also the tensor bytes), the load
  average before and after (`os.getloadavg()`), and whether other runs shared the machine. Runs whose time enters a
  verdict are not run concurrently with other timed runs; if that cannot be guaranteed, the load is reported with the
  number.
- Every number in the results comes from a run in this container. A hypothesis or case that does not fit in time or
  memory is reported as **not tested**, with the measured reason (the size, the memory or the time at which it was
  stopped).
- All three methods are Python (NumPy / SciPy / PyTorch) on the same machine. The comparison is between these
  implementations; it says nothing direct about OpenFOAM, Fluent or a production LBM on a GPU. This limits the
  generalisation and is stated with every verdict.

## Methods

### Reference R: body-fitted mesh + standard FEM

- **Mesh:** `pinneapple_design.geometry.mesh.generate.generate_mesh` (#1025): gmsh triangles (2-D) / tetrahedra
  (3-D), local refinement near the body, prism / quad layers at the wall where the tool offers them. The mesh is built
  **from the same polygon / STL the candidates read** (no analytic CAD), so the geometric input is identical for all
  methods.
- **Solver:** Taylor-Hood P2/P1 with [scikit-fem](https://github.com/kinnala/scikit-fem) (pip-installable, pure
  Python, open source), steady incompressible Navier-Stokes by Newton's method from the Stokes solution, stopping at
  a relative residual of 1e-8; linear systems by a sparse direct solver (SciPy SuperLU / UMFPACK if installed). In 3-D,
  if the direct solve does not fit in memory, GMRES with a block preconditioner (AMG on the velocity block, pressure
  mass matrix on the Schur complement) is allowed, to the same residual.
- **Why this one:** OpenFOAM is not installed in this container and FEniCS/dolfinx is not pip-installable here;
  scikit-fem installs with pip, Taylor-Hood is the textbook inf-sup-stable pair, and it keeps all three methods in the
  same language, so the time comparison is not dominated by the language.
- **Total time** = geometry read + mesh generation + assembly + solve + post-processing (forces, Δp, reattachment).
- **The reference must match the published values (gate R).** Per case, the tolerance is given in the case table.
  If no resolution of the reference passes gate R within the budget, the case is reported, and the candidates are
  then compared with the published values only (the time criterion uses the finest reference run, marked).

### Candidate A: lattice Boltzmann on the voxelised STL

- `WindTunnelLBM` as it is: D3Q19, TRT (magic 3/16), FP32 storage, staircase bounce-back on the voxelised STL
  (`voxelize_stl`), force by momentum exchange. 2-D cases run quasi-2-D: one cell in z, periodic in z.
- Needed boundary conditions that do not exist yet (a parabolic inflow profile, no-slip channel walls, a pressure
  outlet) are added to the library with tests; they are case set-up, not deviations.
- Steady state: run until the drag (or Δp, or the reattachment point) changes by less than 1e-4 relative over
  1000 steps, or 10 flow-through times, whichever comes first (reported).
- Lattice velocity at most 0.05 at the inflow (Mach ≈ 0.09); τ > 0.5 is enforced by the code.
- **Total time** = STL read + voxelisation + solve to the steady criterion + force / Δp evaluation.
- **Allowed deviation A':** interpolated bounce-back (Bouzidi, Firdaouss & Lallemand 2001) instead of staircase,
  only if A fails a case on accuracy. A' is then reported with the label **deviation (post hoc)**, with its own
  numbers, next to A's verdict, never replacing it.

### Candidate B: immersed (cut-cell) flow solver

- Extension of `pinneapple_simulation.immersed` to flow, in the library, with tests of known answer:
  1. **Stokes** on the cut Cartesian grid: Q2/Q1 Taylor-Hood on the background grid; no-slip on the immersed surface
     by symmetric Nitsche (penalty γ/h); ghost penalty on the velocity gradients and on the pressure across the faces of
     cut cells (Burman-Hansbo, Massing-Larson-Logg-Rognes 2014) so that the conditioning and inf-sup stability do not
     depend on the cut position; grid-aligned box faces with strong Dirichlet inflow / walls and do-nothing outflow.
  2. **Navier-Stokes** steady, by Newton (Picard as fallback) from the Stokes solution, with the same stopping rule
     as R.
- Forces on the immersed body by the variational (reaction) form, which is consistent with the Nitsche condition;
  the surface integral of the traction is reported next to it.
- Known-answer tests before any case: Poiseuille flow in a channel whose walls cut the grid (exact for Q2),
  manufactured solution with optimal rates (velocity H1 ~ h², L2 ~ h³ for Q2), Stokes drag on a sphere against
  Stokes' law 6πμaU corrected for the box (or the cylinder in a channel at Re → 0 against a fine R run), and
  insensitivity of the condition number to the cut fraction.
- **Total time** = STL / polygon read + SDF and cut-cell quadrature + assembly + solve + forces.

## Cases and published values

Geometry is generated here (no customer data, no private CFD repository). 2-D bodies are closed polylines (the
cylinder as a regular 512-gon, area error 2.5e-5 relative); 3-D bodies are STL (the sphere as an icosphere with
5 subdivisions, 20,480 triangles). The same file feeds R, A and B.

| case | set-up | published values | gate R (reference vs published) |
|---|---|---|---|
| C1 cylinder 2D-1 | Schäfer & Turek (1996), channel 2.2 x 0.41, cylinder D = 0.1 at (0.2, 0.2), parabolic inflow U_max = 0.3, ν = 1e-3, Re = 20 | Cd = 5.5795, Cl = 0.010619, Δp = p(0.15, 0.2) − p(0.25, 0.2) = 0.11752 (the refined values of John & Matthies 2001 / Nabh 1998, inside the benchmark intervals Cd 5.57-5.59, Cl 0.0104-0.0110, Δp 0.1172-0.1176) | Cd ≤ 1 %, Δp ≤ 1 %, Cl ≤ 5 % |
| C2 sphere | uniform inflow U, sphere D, box [−5D, 15D] x [−5D, 5D]², slip / far-field sides, Re = UD/ν = 20, 50, 100 (steady axisymmetric; Re = 100 is below the first bifurcation at 212) | Cd from Clift, Grace & Weber (1978), 20 ≤ Re ≤ 260: Cd = 24/Re (1 + 0.1935 Re^0.6305) (2.735, 1.576, 1.087); Schiller-Naumann 24/Re (1 + 0.15 Re^0.687) reported next to it, not used in the verdict | Cd ≤ 4 % against Clift-Grace-Weber (the two correlations differ by 4.6 % at Re = 20; a finite box adds a few per cent at Re = 20) |
| C3 backward-facing step | Armaly, Durst, Pereira & Schönung (1983): expansion ratio H/h = 1.942 (downstream height H = 10.1 mm, upstream h = 5.2 mm, step S = 4.9 mm); fully developed parabolic inflow 5 S upstream of the step; outlet 30 S downstream; Re = U_mean · 2h / ν = 100, 200, 389 (Armaly's definition: hydraulic diameter of the inlet channel; if the source of x1/S uses another definition, its definition is used and noted) | primary reattachment length x1/S, experimental, from Armaly et al. (1983) (see the note below) | x1/S ≤ 5 % at Re 100 and 200, ≤ 10 % at Re 389 (the 2-D / 3-D discrepancy starts near Re 400) |
| C4 Ahmed body (optional) | the generic Ahmed body already in the library (#926, #982), 25° slant | the drag of the library's own OpenFOAM results and Ahmed et al. (1984) as reported in `docs/guides/ahmed_body_end_to_end.md` | as reported there |

**Note on C3.** No copy of Armaly et al. (1983) is reachable from this container (the open copies' hosts are not
resolvable here, checked on 2026-10-31), and the paper reports x1/S in a figure (x1/S against Re), not in a table.
Rule, fixed now: before any candidate runs C3, the values are taken from the first open tabulation or digitisation of
Armaly's experimental curve that is reachable here, written in the results section with its source and its reading
uncertainty. If none is reachable, gate R and the published-value comparison for C3 are **not tested**, and C3 is
judged against the finest reference run only (the margin below unchanged).

**C4** is attempted only if C1-C3 finish within the budget. At Re ≈ 4e6 it needs a turbulence model on both sides;
R (laminar Taylor-Hood) cannot serve as its reference, so C4 can only be **partial** at best: the candidate against the
library's OpenFOAM result, with no body-fitted run of this campaign.

## Metrics

Per run and case:
- relative error of Cd, Cl, Δp, x1/S against the published value;
- relative L2 error of velocity profiles against the finest reference run: C1 u(y) at x = 0.15, 0.25, 0.4, 1.0;
  C2 u along the axis behind the sphere, 0 to 5D; C3 u(y) at x/S = 2, 5, 10;
- total time and solve-only time (stages as defined above);
- peak memory;
- load average and concurrency.

## Resolution ladders (declared now, not extended after the fact)

Each method runs a fixed ladder of resolutions per case; the run used in the verdict is chosen by the rule below.

| case | R (body-fitted: h at the body / far field) | A (cells per D or per S) | B (background h) |
|---|---|---|---|
| C1 | D/8, D/16, D/32, D/64 at the cylinder; 4x coarser far away | D = 10, 20, 40, 80 cells | D/4, D/8, D/16, D/32 |
| C2 | D/8, D/16, D/24 at the sphere; up to D in the far field | D = 8, 12, 16 (16 if memory allows: 320 x 160 x 160) | D/4, D/6, D/8 (as memory allows) |
| C3 | S/8, S/16, S/32 (uniform in the channel) | S = 10, 20, 40 | S/8, S/16, S/32 |

A rung that does not fit in 14 GB or in 4 h wall is stopped and reported as such.

## Success and kill criteria

Fixed margins on the relative error, in percentage points (pp) of the error against the published value:

| quantity | margin |
|---|---|
| Cd | 2 pp |
| Δp | 2 pp |
| Cl (C1, 0.2 % of Cd, benchmark interval ±3 %) | 10 pp |
| x1/S | 5 pp |
| profile L2 (against the finest R run) | 3 pp |

Rule:
1. **Reference operating point** R*: the cheapest rung of R (in total time) that passes gate R in that case (for a
   multi-Re case, at that Re).
2. **Candidate operating point** X*: the cheapest rung of candidate X whose errors are all ≤ the errors of R* plus the
   margin, quantity by quantity (Cd, Cl, Δp or x1/S, and the profile).
3. **Speed-up** S_X = T_total(R*) / T_total(X*). The solve-only ratio is reported too, but does not enter the verdict.

Verdict of candidate X in a case (in a multi-Re case, at each Re, then the case):
- **validated:** X* exists and S_X ≥ 2;
- **partial:** X* exists and 1 ≤ S_X < 2; or, in a multi-Re case, validated at some Re and not at others (the
  boundary is stated);
- **refuted (kill):** no rung of X meets the accuracy condition, or S_X < 1;
- **not tested:** with the reason measured.

Hypotheses:

| id | hypothesis | kill criterion |
|---|---|---|
| H-A1 | LBM on the voxelised polygon is validated on C1 | no rung within the margins of R*, or S_A < 1 |
| H-A2 | LBM is validated on C2 at Re 20, 50, 100 | as H-A1, at any of the three Re |
| H-A3 | LBM is validated on C3 at Re 100, 200, 389 | as H-A1, at any Re |
| H-B0 | The immersed Stokes / NS solver passes its known-answer tests (Poiseuille to round-off on cut walls; manufactured-solution rates ≥ 1.8 in velocity H1 and ≥ 2.8 in L2 for Q2/Q1; condition number within 10x over cut fractions down to 1e-4; Stokes drag within 2 % of the fine R value at the finest rung) | any of these fails |
| H-B1 | The immersed solver is validated on C1 | as H-A1 |
| H-B2 | The immersed solver is validated on C2 | as H-A1; in 3-D Q2/Q1 may not fit: then **not tested** with the size at which it stopped |
| H-B3 | The immersed solver is validated on C3 | as H-A1 |
| H-G | **Overall: CFD direct from the polygon geometry is faster in total time with similar results** | validated if at least one candidate is validated in C1, C2 and C3; partial if validated in at least one case; refuted if no candidate is validated in any case |

## Figures (planned)

Velocity magnitude and pressure fields of R, A and B for C1 (and the mid-plane for C2, the step for C3); Cd against
resolution for each method with the published band; total time against error (one curve per method and case, the
operating points marked); memory against resolution.

## Results

(Appended after the runs.)

### Deviations from the pre-registration

Recorded before the official runs. The text above is unchanged.

1. **C3: the source of the published values, and Re 400 instead of Re 389.**
   - No copy of Armaly et al. (1983) is reachable from this container. `courses.washington.edu` and `homepage.ntu.edu.tw` do not resolve here (checked on 2026-10-31).
   - Following the rule in the note on C3, x1/S comes from the first open tabulation of Armaly's experimental curve that is reachable here: Table VIII of arXiv:2602.04670 ("Armaly et al." column).
   - Values used: x1/S = 2.96 at Re 100, 4.90 at Re 200 and 8.45 at Re 400. They are given to two decimals, with no reading uncertainty stated. That source also does not restate Armaly's Re definition in the excerpt read; the campaign keeps the pre-registered Re = U_mean 2h / ν.
   - The table has no Re 389, so **C3 runs at Re = 100, 200 and 400** instead of 100, 200 and 389.
   - The pre-registered gate for the highest Re (x1/S within 10 %) applies to Re 400. The margins and the verdict rule are unchanged.
2. **Linear solver.** The pre-registration names SciPy SuperLU / UMFPACK for R. The campaign uses MKL PARDISO (`pypardiso`) for both R and B, through `numerical_solvers.sparse_direct`.
   - The reason is measured: on C2 at Re 20, D/8 (134,501 unknowns), SuperLU used 9 GB and had not finished after more than 10 min.
   - PARDISO finished the same run in 151 s total, with 5.0 GB peak memory.
   - This changes times, not answers. Both sides use the same solver, so the time comparison is not a comparison of linear solvers.

## Change log

- 2026-10-31: pre-registration.
- 2026-10-31: deviations 1 (Armaly source, Re 400 instead of 389) and 2 (PARDISO) recorded, before the official runs.
