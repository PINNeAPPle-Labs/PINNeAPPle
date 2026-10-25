# Falsification campaign for physics condensation — pre-registration

Pre-registered on 2026-10-25, before any of the runs below. This file states each hypothesis, the test, the
metric and the **kill criterion**: the measured result that refutes the hypothesis. It is committed before the
measurements. The results section is appended later; nothing above it changes after the runs, except obvious typos,
which are listed in the change log at the end.

The method under test is the one in [physics_condensation.md](physics_condensation.md): exact static condensation
of repeated blocks, reuse of identical blocks, reduced (POD) interfaces with an error estimate, memoised nested
dissection, learned block operators, and prior plus exception patches. The campaign tries to break it. A refutation
is reported as prominently as a confirmation.

## Scale and references (declared up front)

- **Thermal:** the 3-D multilayer package of the report, up to 128 x 128 blocks (2.4e7 fine cells). That is the
  memory limit of this container (4 cores, 15 GB).
  - The fine finite-volume solve (AMG-CG, `pinneapple_condense.fv`) is the reference wherever it fits, up to 64 x 64
    blocks.
  - Beyond that, the reference is the **exact condensation**: the CG correction on the unreduced interfaces, run to
    1e-10. Condensation is exact algebra, so this equals the fine FV solution. It is verified against the fine FV
    solve where both fit.
- **CFD:** 2-D, steady, laminar, incompressible, on a staggered (MAC) finite-volume grid assembled as one sparse
  system.
  - OpenFOAM is not installed in this container, so the reference is the library FV solve of the same discretisation
    (direct or AMG).
  - That solver is validated first against:
    - plane Poiseuille flow (analytic);
    - the lid-driven cavity at Re = 100, centreline velocities of Ghia, Ghia and Shin, *J. Comput. Phys.* 48
      (1982) 387;
    - the fully developed Nusselt number for parallel plates with one wall at uniform flux and the other adiabatic,
      Nu_Dh = 5.385 (Shah and London, 1978, as used by `pinneapple_simulation.numerical_solvers.thermal_channel_2d`).
  - Its mesh convergence is reported.
- **Meshless:** RBF-FD (PHS r^3 + polynomials) and GFDM stencils of `pinneapple_simulation.meshless.stencils`, on
  point clouds generated from the implicit geometry (SDF).
- **Not measured:**
  - Every number in the results section comes from a run in this container.
  - A hypothesis that could not be tested is reported as **not tested**, with the reason.

Error tolerances are absolute temperatures (K) for the thermal package, as in the report. For the CFD cases they are
relative, against the reference of the same discretisation:
- flow rate or pressure drop;
- the relative L2 norm of the field;
- fin-base / wall temperature rise.

## Hypotheses

### A. Thermal: where the existing method should break

| id | hypothesis | test | metric | kill criterion (refutes) |
|---|---|---|---|---|
| T1 | The number of interface modes m needed for a peak-junction error ≤ 0.1 K grows slowly with the number of blocks N | nb = 16, 32, 64, 128 (N up to 16,384); m ∈ {6, 12, 24, 48}; hotspot power map; reference as above | m*(N): smallest m with \|ΔTmax\| ≤ 0.1 K | m*(16,384) > 2 m*(1,024), or no m ≤ 48 reaches 0.1 K at 16,384 blocks |
| T1b | The exact correction (PCG preconditioned by the reduced model) is a scalable fix | same packages, m = 12 preconditioner | PCG iterations and wall time to \|ΔTmax\| ≤ 1 mK | iterations at 16,384 blocks > 4× those at 1,024, or wall time above one fine AMG solve where that is measurable (≤ 4,096 blocks) |
| T2 | Strong conductivity contrast does not break condensation | underfill k lowered so that k_max/k_min = 1e3, 1e4, 1e5 (the baseline is 8e2); 32 x 32 blocks | exact condensation against fine FV; m=12 error | exact condensation differs from FV by > 1e-6 K, or the m=12 error exceeds 3× its baseline-contrast value |
| T3 | Anisotropic layers are handled like isotropic ones | TIM and interposer with k_xy/k_z = 100 and 0.01 (same k_z); 32 x 32 blocks; interface snapshots regenerated for the stack | as T2 | as T2 |
| T4 | Without repetition, condensation still has a cost advantage | every block distinct (continuous random variants: bump voiding, TIM contact, TSV quality); 32 x 32 blocks; exact per-block condensation and the learned block operator | wall time against one fine AMG solve; error against FV | exact path: build + one solve slower than one fine solve (report the break-even number of power maps). Learned path: slower than one fine solve, or error > 0.1 K |
| T5 | Memoisation degrades gracefully with many distinct variants | V = 1, 4, 16, 64, 256 discrete variants placed at random on 64 x 64 blocks | hierarchy build time against V; hierarchy against flat condensation | at V = 256 the hierarchy build is slower than the flat condensation build (no gain left from memoisation) |
| T6 | Nonlinear k(T) is handled by Picard iteration with re-condensation | silicon and copper k(T) ∝ (300 K / T)^α (α = 1.3 Si, 0.1 Cu), 16 x 16 blocks; blocks condensed with their own mean-temperature k each iteration; reference: fine FV Picard to 1e-8 | peak-junction error; number of iterations; wall time | error > 0.1 K, or > 10 Picard iterations, or wall time above the fine nonlinear solve |
| T7 | Temperature-dependent contact resistance is handled the same way | die/TIM R_c(T) = R_0 (1 + β (T − T_0)), β such that R_c doubles over the range; 16 x 16 blocks | as T6 | as T6 |
| T8 | Reduced interfaces survive power maps unlike the snapshots | checkerboard per cell, checkerboard per block, single-cell spike, white noise, power on block corners only; 16 x 16 blocks, m = 12 | peak-junction error; estimator effectivity | error > 0.1 K on any map, or effectivity outside [0.5, 2] |
| T9 | Very thin layers do not break it | TIM 30 → 3 → 1 µm, die 100 → 20 µm; 16 x 16 blocks | as T2 | as T2 |
| T10 | A non-uniform pitch (column widths alternating 40 / 20 µm) works like the uniform one | 16 x 16 blocks, each pitch a separate block model | as T2 | as T2 |

### B. CFD with a mesh (2-D, steady)

A new generic layer, **algebraic condensation**, condenses any assembled sparse operator over a partition of its
unknowns into blocks:
- the separator consists of the cells on block boundaries;
- for saddle-point systems, the separator also holds one pressure unknown per block;
- block interiors with identical matrices are factorised once.

This is what applies the method to non-symmetric and saddle-point operators.

| id | hypothesis | test | metric | kill criterion |
|---|---|---|---|---|
| F0 | The flow solver is a valid reference | Poiseuille; cavity Re = 100 against Ghia et al.; parallel-plate Nu = 5.385; three-grid convergence | max deviation | Poiseuille error > 1e-6 relative (it is exact for a parabola on a MAC grid), or cavity centreline deviation > 2 % of the lid speed at 128², or Nu off by > 2 % |
| F1 | Exact algebraic condensation stays exact for a non-symmetric operator | steady advection–diffusion through a periodic array of obstacles, velocity from F2; block Péclet Pe_b = U L_b / α ∈ {0.1, 1, 10, 100, 1000}; 8 x 8 blocks | relative difference against the global solve | > 1e-8 relative at any Pe |
| F1r | Reduced interfaces keep at least 4× interface compression at 1 % error up to Pe_b = 100 | same, Galerkin projection onto POD modes per face from local snapshots | smallest m for 1 % L2 error; compression = face unknowns / m | compression < 4 at Pe_b = 100. The expected failure, upwinding across faces, is reported as such |
| F2 | Stokes flow through a periodic pin-fin array condenses exactly and gains from identical blocks | square pin array, 16 x 16 pins, MAC grid, separators with velocity + one pressure per block | flow-rate error; speedup against the global direct/AMG solve of the same system | exactness > 1e-8 relative, or speedup < 2× at 16 x 16 pins (build + solve, one right-hand side) |
| F2r | Reduced interfaces work for Stokes | as F2, POD interface modes | smallest m for 1 % flow-rate error | compression < 4 |
| F3 | Conjugate heat transfer in a cold plate with repeated fins at moderate Re, Picard / Oseen linearised, condensed per block, matches the full solve | channel with a row of fins (solid conduction + fluid advection–diffusion), Re_fin ∈ {1, 10, 50, 100}; reference: full Picard solve of the same discretisation | pressure-drop and fin-base temperature-rise error; wall time | error > 1 % at any Re, or condensed Picard slower than the full Picard solve |
| F3r | Reduced interfaces work for the CHT case at Re_fin = 100 | as F3 | compression at 1 % | compression < 4. The expected cause, recirculation crossing block faces, is checked by measuring the fraction of face flux that is reversed |

### C. CFD without a mesh

| id | hypothesis | test | metric | kill criterion |
|---|---|---|---|---|
| M1 | Meshless advection–diffusion converges on point clouds from the SDF | F1 geometry, RBF-FD and GFDM, three point densities, Pe_b ∈ {1, 10, 100} | error against the fine FV solution interpolated to the points; observed order | error not decreasing at each refinement, or > 10× the FV error at equal unknowns |
| M2 | Meshless Stokes converges | F2 geometry, RBF-FD velocity–pressure with the same stencil family | flow-rate error against FV; convergence | no convergence over three densities |
| M3 | The FV-at-interfaces hybrid removes the meshless interface failure of the report (§4.9) | 2-D multi-material conduction (copper pins in a low-k matrix, a contact resistance on the pin walls): FV cells in a band around every material interface, RBF-FD elsewhere | error against fine FV; convergence | hybrid error > 2× FV at equal unknowns, or non-monotone convergence |
| M4 | Meshless blocks condense exactly to ports, and replicated point clouds give identical-block reuse | F1 and M3 cases, the point cloud of one block replicated, algebraic condensation | difference against the global meshless solve; speedup | > 1e-8 relative, or no speedup from reuse |

### D. Prior + exception points on CFD

| id | hypothesis | test | metric | kill criterion |
|---|---|---|---|---|
| E1 | The fine-operator residual of a cheap prior finds the blocks that need the fine solve (wakes, recirculation) better than random | F1 at Pe_b = 100 and F3 at Re_fin = 50; prior: the solution on a 4× coarser grid, interpolated; patches solved with the prior as Dirichlet data | error against solved fraction, residual vs random selection; overlap of the selected blocks with the blocks of largest prior error | residual selection needs more than half the solved fraction random needs for the same error, or overlap with the top-error blocks < 0.5 |

## Verdicts

Each hypothesis gets one verdict, based only on its kill criterion and the measured numbers:
- **validated**: the kill criterion is not met;
- **refuted**: the kill criterion is met;
- **partially validated**: it holds at some sizes or parameters and is refuted at others; the boundary is stated;
- **not tested**: the reason is given.

## Results

Appended after the runs.

## Change log

- 2026-10-25: pre-registered.
