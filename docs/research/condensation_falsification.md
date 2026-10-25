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

Measured in this container on 25 October 2026 (4 cores, 15 GB). Code: `pinneapple_condense.falsify`, run by
`examples/condensation_falsification/run.py`.
- Raw results: `results/*.json`.
- Verdicts: `results/verdicts.json`, computed by `verdicts()` in the runner from the kill criteria above.
- Figures: `figures/`.

Several runs shared the machine (two runners at a time), so wall times carry that load. Every cost comparison below
was measured inside one process, under the same load on both sides.

### Verdicts

| id | verdict | evidence (measured) |
|---|---|---|
| T1 | validated | smallest m (of 6, 12, 24, 48) with /dTmax/ <= 0.1 K: {'256': 6, '1024': 12, '4096': 12, '16384': 24} blocks -> m; kill if m(16,384) > 2 m(1,024) |
| T1b | validated | PCG iterations to 1 mK: 8 at 1,024 and 32 at 16,384 blocks (limit 4x); time to 1 mK 5.0 s vs fine AMG 481 s at 4,096 blocks |
| T2 | validated | exact vs fine <= 6.5e-09 K; worst m=12 error 1.09x the baseline (16.0 mK) |
| T3 | validated | exact vs fine <= 5.8e-08 K; worst m=12 error 1.00x the baseline (16.0 mK) |
| T9 | validated | exact vs fine <= 6.7e-09 K; worst m=12 error 2.20x the baseline (1.4 mK) |
| T4 | **refuted** | 32 x 32 (pre-registered): killed out of memory at 13 GB (one dense model per distinct block); 16 x 16: fine FV 67 s; exact path (all blocks + PCG) 395 s; learned 27 s (+82 s training), error 0.46 K; m-mode interfaces alone off by 0.46 K on these heterogeneous blocks |
| T5 | **refuted** | V=1: hierarchy 2.7 s / flat 5.8 s; V=4: hierarchy 20.3 s / flat 4.9 s; V=16: hierarchy 21.2 s / flat 5.0 s; V=64: hierarchy 27.1 s / flat 4.6 s; V=256: hierarchy 61.1 s / flat 6.6 s |
| T6 | **refuted** | 16 x 16 (pre-registered): out of memory in the separator factorisation of the generic layer; 8 x 8: error 0.51 mK, 5 Picard iterations, 430 s vs fine 97 s (nonlinear - linear Tmax 1.20 K) |
| T7 | **refuted** | 16 x 16 (pre-registered): out of memory in the separator factorisation of the generic layer; 8 x 8: error 3.2e-06 mK, 4 Picard iterations, 176 s vs fine 39 s (nonlinear - linear Tmax 8.96 K) |
| T8 | **refuted** | worst map 'single-cell spike': 1.66 K; others <= 9.3 mK; effectivity 1.37-1.77 |
| T10 | partially validated | 16 x 16 (pre-registered): out of memory in the generic layer; 8 x 8: exact vs fine 2.9e-09 K; m=12 error 1.4e-06 K (uniform 1.8e-06 K) |
| F0 | partially validated | Poiseuille max rel. error 3.2e-03 (pre-registered as exact, 1e-6: refuted -- the ghost-cell wall is O(h^2), observed order 2); cavity Re 100 vs Ghia 0.49 % of U (limit 2 %); Nu error 0.081 % |
| F1 | validated | max relative difference 1.5e-13 over Pe_b 0.1-1000; blocks distinct by content: 80 of 80 (the computed velocity differs per block) |
| F1r | validated | Pe_b = 100: m = 3 for 1 % (14.2 unknowns per face, compression 4.74); Pe_b 0.1: m 2, Pe_b 1: m 2, Pe_b 10: m 2, Pe_b 100: m 3, Pe_b 1000: m 3 |
| F2 | **refuted** | exact to 1.3e-12; 9 distinct of 288 blocks; direct 2.5 s vs condensed 11.5 s (speedup 0.22, limit 2) |
| F2r | validated | m = 2 for 1 % on the pressure drop (compression 7.21) |
| F3 | **refuted** | error <= 5.5e-11; condensed Picard slower than full at Re [1, 10, 50, 100] Re 1: 9 vs 4 s, Re 10: 11 vs 6 s, Re 50: 16 vs 8 s, Re 100: 19 vs 11 s |
| F3r | validated | Re 100: m = 3 for 1 % (compression 4.86); reversed block-face flux 4.5 % |
| M1 | **refuted** | rbf_fd Pe 1: monotone, up to 31.5x FV; gfdm Pe 1: monotone, up to 2.86e+03x FV; rbf_fd Pe 10: monotone, up to 27.6x FV; gfdm Pe 10: NOT monotone, up to 4.5e+03x FV; rbf_fd Pe 100: monotone, up to 2.28x FV; gfdm Pe 100: NOT monotone, up to 2.02e+11x FV |
| M2 | partially validated | pressure-drop error over densities 8, 16, 32 -- rbf_fd: 46 %, 50 %, 18 % (not converging); gfdm: 103 %, 38 %, 35 % (decreasing); FV: 36 %, 19 %, 7 % |
| M3 | **refuted** | density 16: hybrid 3.91 % vs FV 0.49 %; density 32: hybrid 1.06 % vs FV 0.18 % |
| M4 | **refuted** | exact to 1.5e-09; hybrid: 9 distinct of 16 blocks, speedup 0.16; advection: 24 of 24, speedup 0.22 |
| E1 | **refuted** | advection Pe_b=100: to reach 0.075 residual solves 85%, random 90%; overlap with top-error blocks 0.00-0.67; cold-plate energy Re=50: to reach 0.231 residual solves 73%, random 89%; overlap with top-error blocks 0.00-0.69 |

Summary: 23 hypotheses: 9 validated, 3 partially validated, 11 refuted.

### What holds

- **Exactness everywhere.**
  - Exact condensation, block-wise or algebraic, reproduces the global solve to round-off on every operator tried:
    - symmetric conduction with conductivity contrasts up to 1e5, anisotropy 100 and 0.01, 1-µm layers, alternating
      pitch: 1e-9 to 6e-8 K;
    - non-symmetric advection–diffusion up to Pe_b = 1000: 1e-13;
    - Stokes saddle points with one pressure unknown per block: 1e-12;
    - Picard-linearised Navier–Stokes and conjugate heat transfer up to Re 100: 1e-14;
    - meshless matrices: 1e-9.
  - So the algebra is never the problem.
- **Reduced interfaces degrade gracefully with physics.** Contrast, anisotropy and thin layers change the m = 12
  error by at most 2.2×.
  - For advection–diffusion the modes needed for 1 % grow only from 2 (Pe_b ≤ 10) to 3 (Pe_b 100–1000).
  - For Stokes 2 modes suffice, and for the Re 100 cold plate 3.
  - Recirculation crossing block faces is real but small at these Reynolds numbers: 4.5 % of the face flux at Re 100.
- **Far field (T1).**
  - At fixed m the error grows about 3× for every 4× more blocks: m = 12 gives 1.4 mK at 256 blocks, 16 mK at 1,024,
    52 mK at 4,096 and 0.21 K at 16,384.
  - The m needed for 0.1 K went 6 → 12 → 12 → 24. That is exactly the pre-registered limit of 2×, on a coarse m
    grid.
  - The exact correction fixes it at moderate cost:
    - PCG iterations to 1 mK are 2, 8, 8 and 32;
    - time to 1 mK is 5 s against 481 s for one fine AMG solve at 4,096 blocks.

### What breaks, and why

1. **Cost whenever blocks stop repeating (T4, T5, T6, T7, F1, F3).**
   - The method's advantage is reuse. With random variants the memoised hierarchy is slower than the flat
     condensation from V = 4 on (20 s against 4.9 s; 61 s against 6.6 s at V = 256), because almost no subtree
     repeats.
   - With every block distinct (T4), the exact path costs more than the fine solve. At 32 × 32 it did not fit in
     memory: one dense block model per distinct block.
   - Non-linear material laws make every block distinct at every Picard iteration: 4.4× (T6) and 4.5× (T7) slower
     than the fine non-linear solve, at the same accuracy.
   - A numerically computed velocity field differs bit by bit between "periodic" blocks. Content-identical reuse then
     finds nothing (80 of 80, 44 of 44 blocks distinct), and condensed Picard is about 2× slower than the full one
     (F3).
2. **Cost against a sparse direct solver in 2-D (F2, M4).**
   - Even with 9 distinct of 288 blocks, condensing a 2-D Stokes system was 4.6× slower than one sparse direct
     solve.
   - The factorisation of the separator dominates (its dense blocks fill in).
   - In 2-D, nested dissection is already near-optimal, so there is little left to win for a single right-hand
     side.
3. **Sub-pitch features the modes cannot represent (T8).**
   - A single-cell power spike gives 1.66 K with 12 modes. The other adversarial maps (checkerboards, noise,
     corners) stay below 10 mK.
   - The estimator is not fooled: its effectivity stays in 1.4–1.8. It flags the problem, and refinement or the
     exact correction has to act on it.
4. **Heterogeneous blocks with base-stack modes (T4).**
   - On a package where every block differs (voids, delamination ×30), the m-mode interfaces are off by 0.46 K at
     16 × 16 blocks (0.68 K at 8 × 8). On uniform packages they are off by 2–16 mK.
   - The learned-operator study of the track 1 report compared the learned path with *all-exact blocks on the same
     m-mode interfaces*, not with the fine solution. Its 4 mK agreement did not include this 0.68 K.
5. **The generic algebraic layer in 3-D (T6, T7, T10 at 16 × 16).**
   - The separator factorisation ran out of memory.
   - The port-based block condensation of `pinneapple_condense.substructure` does not have this problem: it keeps
     faces separate.
   - The generic layer is a correctness tool here, not a fast one.
6. **Meshless (M1–M4).**
   - RBF-FD converges for advection–diffusion but is 28–32× less accurate than FV at equal unknowns at low Pe. It is
     only competitive at Pe_b = 100.
   - GFDM is unstable (errors up to 1e9).
   - Collocated meshless Stokes does not converge reliably: RBF-FD goes 46 %, 50 %, 18 %; GFDM decreases but
     stagnates at 35 %.
   - The FV-at-interfaces hybrid removes the interface failure of the report (its error is now spread over the band
     and the points alike), but stays 6–8× worse than FV everywhere. Its largest errors sit at the meshless points
     near the domain boundary and the band edge.
   - Meshless blocks condense exactly and, with deterministic neighbour selection, are found identical. Condensation
     is still slower than one direct solve.
7. **Prior + exception patches on transport (E1).**
   - On advection-dominated fields the residual of a coarse prior is no better than random at choosing where to
     solve, and its selected blocks overlap the true top-error blocks by at most 0.69.
   - The error of a coarse prior is carried downstream, away from where the residual is large.
   - In the thermal case of track 1c (diffusive, local defects) the same method worked.

### Where the idea stands

Condensation is **exact and robust as algebra**. Its accuracy-per-mode survives contrast, anisotropy, thin layers,
moderate Péclet and Reynolds numbers. Its growth with package size is real but controllable (m doubling per 16×
blocks, or the exact correction).

Its **speed comes only from repetition**, and the campaign shows how fragile that is:
- random variants;
- non-linear material laws;
- computed advection fields;
- 2-D problems where sparse direct solvers are already efficient.

Each of these removes the gain, sometimes entirely. A practical method needs:
- identical-up-to-tolerance block matching, not bit-identical (computed velocities, temperature-dependent k);
- parametric block operators (the learned path, with interface modes trained for heterogeneity);
- H-matrix arithmetic for the separators.

None of these is implemented here.

## Deviations from the pre-registration

Listed here instead of changing the hypotheses above.

1. **F0, Poiseuille.** The pre-registered criterion expected an exact parabola (1e-6). The ghost-cell wall treatment
   is O(h²) for curved profiles; the observed order is 2. That sub-criterion is met (refuted) as written.
2. **F2 / F3 metric.** The velocity inlet fixes the flow rate, so the pressure drop at that flow rate (the
   equivalent permeability measure) replaces the flow-rate error.
3. **T7 law.** "β such that R_c doubles over the range" is implemented as a linear rise from R_0 at the coolest to
   2 R_0 at the hottest interface temperature of the linear solution, saturating at 2 R_0.
   - An unsaturated law ran away: R up → T up → R up.
   - A first run took the range from the die-top cells instead of the interface the law is applied to. That made
     the law inert, and it was corrected and re-run.
4. **Reduced scales (memory).**
   - T4 at 16 × 16 instead of 32 × 32 (killed at 13 GB).
   - T6, T7 and T10 at 8 × 8 instead of 16 × 16: the separator factorisation of the generic layer ran out of memory.
     Those failures are kept as results.
   - T1 has no m = 48 point at 16,384 blocks (beyond 15 GB).
5. **T4 "exact path"** is the condensation of every block followed by the PCG correction to the fine solution. The
   all-exact-blocks solve of the learned study still uses m-mode interfaces (item 4 of "What breaks").
6. **Implementation defects found and fixed during the runs; affected studies re-run.**
   - The block count of the algebraic partition counted unknowns on the far boundary as a new block (F1–F3 re-run;
     verdicts unchanged).
   - The meshless k-nearest-neighbour search broke ties differently in different blocks. M4 first measured 16 of 16
     blocks distinct and a speedup of 0.26; after the fix, 9 of 16 distinct and 0.16.
7. **E1 budgets** select a fraction of blocks; with one ring of halo the solved fraction is larger. It is reported
   as such.

## Change log

- 2026-10-25: pre-registered.
- 2026-10-25: results and deviations appended after the runs; the hypotheses and kill criteria above are unchanged.
