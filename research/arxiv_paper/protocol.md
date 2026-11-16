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
