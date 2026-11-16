# Preregistered benchmark protocol (draft)

## Questions
RQ1: For matched PINN architectures, does the PINNeAPPle execution path yield comparable accuracy and resource cost to DeepXDE and PhysicsNeMo?
RQ2: Can VeriPhysics detect intentionally invalid or out-of-domain predictions better than residual-only checks?
RQ3: What is the reproducibility variance across hardware, seeds and framework versions?

## Benchmark tiers
1. Analytical: 1D Poisson, 1D Burgers (reference numerical solution), 2D heat equation with manufactured solution.
2. PDE generalization: 2D Darcy with held-out permeability fields, Navier–Stokes/cylinder flow with independently generated reference solutions.
3. Operator learning: Darcy and Burgers with identical train/validation/test splits and FNO/DeepONet where supported.
4. Industrial case study only if mesh, boundary conditions, numerical convergence and rights to publish are independently verified.

## Fair comparison
- Track A: same architecture, optimizer, loss weights, collocation data, training steps, precision and device.
- Track B: framework-native recommended setup under the same total accelerator-time budget.
- Do not compare an operator-learning method to a single-instance PINN as if tasks were identical.
- Five seeds minimum, preregistered train/test splits, explicit failure criteria, and no cherry-picked runs.
- Measure median, IQR and 95% bootstrap CI; disclose hardware, GPU-hours and environment.
- Store each run's exact config, git SHA, package lock, data checksum, random seeds, raw predictions and metrics.
- Residual evaluation must use consistent physical units, coordinate systems and independent test points.
- OOD must be predeclared, e.g. held-out geometry, forcing magnitude, Reynolds number or coefficient field.
- Verify reference solutions by analytical checks or grid convergence, not by treating a solver as infallible.

## Acceptance gates
- All tables trace to raw artifact files.
- Baseline reproduction succeeds in clean environment.
- No unverified accuracy/speed claims.
- External researcher can reproduce a subset within a documented budget.
- Manuscript explains negative results and limits of VeriPhysics.
- Bibliography references primary framework publications and current documentation.

## Official starting points
- DeepXDE: https://github.com/lululxvi/deepxde
- PhysicsNeMo: https://github.com/NVIDIA/physicsnemo
- NeuralOperator: https://github.com/neuraloperator/neuraloperator
- PINNeAPPle: https://github.com/PINNeAPPle-Labs/PINNeAPPle

---

# RQ2 in detail: the verification corpus (design fixed before the run)

RQ2 asks whether a physics check can tell that a prediction is wrong when no reference solution
exists for the case at hand. Measuring that needs cases whose truth is known *and* whose failure mode
is known, so the corpus is built rather than collected.

## Corpus

Six problems: two the checker's equation set can represent exactly (steady conduction with a constant
source; transient 2-D conduction with a manufactured solution) and the four forward benchmarks of
track A (Poisson, Helmholtz, Burgers, Allen–Cahn). For each problem, twelve predictions on the
benchmark's own test grid:

- **1 exact reference.** The negative control that must never be flagged.
- **5 valid approximations**, each the reference plus a smooth perturbation that vanishes on every
  declared boundary face, scaled to a prescribed relative L2 error from 1e-4 to 1e-1. These are the
  calibration ladder: they establish, per rule, the error level at which it starts to fire.
- **6 invalid predictions**, each carrying exactly one named defect:
  `bc_offset` (offset confined to a boundary layer), `wrong_coefficient` (the reference solution of
  the same problem with one physical coefficient changed), `over_smoothed` (Gaussian-filtered with the
  declared boundary data restored), `unit_scale` (×1000), `nonfinite` (five NaN), `zero_field`
  (identically zero).

72 cases. They are analytic functions on fixed grids, so the corpus rebuilds byte-identically on any
machine and needs no GPU, dataset or network.

## Declared expectations

Fixed before the run, and reported as "expected vs measured":

| defect | rule expected to catch it |
|---|---|
| `bc_offset` | PC201 (boundary condition) |
| `wrong_coefficient` | PC301 (PDE residual) |
| `over_smoothed` | PC704 (spectral bias against the reference) |
| `unit_scale` | PC401 (declared bounds) |
| `nonfinite` | PC403 (non-finite values) |
| `zero_field` | PC202 (initial condition) or PC301 |

Declared tolerances, also before the run: reference regression (PC701) 5%; the checker's own defaults
for everything else, except Allen–Cahn's boundary condition, declared at 5e-3 because the reference
itself only satisfies `u(±1) = -1` to 3.1e-3 of the boundary scale (the true condition is periodicity,
which the checker cannot express).

## Two detector families, reported apart

- **Reference-free**: PC201, PC202, PC301, PC401, PC403. These work on a case never seen before.
- **Reference-based**: PC701, PC704. These need a trusted solution for the case at hand — exactly what
  a surrogate model is built to avoid needing.

A check that fires on a valid approximation of a problem whose equation the checker cannot express is
a failure of the *declaration*, not of the check, and the two are counted separately.

## Measurement

Every rule outcome for every case is recorded (`experiments/verification/results/verification.json`),
including the measured value and the threshold, so detection counts can be recomputed with any
tolerance. Alongside the tool's rules the run computes a tool-independent statistic: the PDE residual
of the prediction by second-order finite differences on the case grid, normalised by the size of the
equation's terms at the reference. Its value on the exact reference is the **floor**, and it is
reported with the grid step, because a residual check whose tolerance is not well above the floor
cannot discriminate anything on that grid. A refinement study (101 to 801 points per axis) reports how
the floor falls.

## Acceptance gates for RQ2

- Every number in the RQ2 section maps to a case record.
- The floor of every problem is reported next to the check's tolerance.
- Detection and false-positive counts are given per rule, per problem, and split by whether the
  equation was expressible.
- What the checker cannot express is stated as a result, with the measured consequence (the false
  positive it produces on a correct answer).

