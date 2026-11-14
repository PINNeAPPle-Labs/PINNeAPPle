# Structural Bracket Surrogate — Phase 1

Reproduction-oriented benchmark inspired by Danilo Rizzo de Oliveira, *A DeepONet Surrogate Model for Stress Fields in Parametric Bracket Families: The Cost of Geometric Generalization* (ICMC-USP).

## Source-supported facts
- Families shown: ribbed G and clevis U (fixed geometry); L and Z (unseen geometry).
- Illustrated median-validation-case relative L2 errors: 0.10, 0.08, 0.16, 0.15 respectively. These are **not** benchmark-wide aggregate scores.
- Branch: 31 values: six load direction terms, log(E), nu, twelve family one-hot features, eleven geometric parameters.
- Trunk: eight values: normalized xyz, two distances to fixation/loading, one concave-edge distance, two proximity features.
- Branch/trunk MLPs: four 128-unit tanh layers each; elementwise product, nonlinear 128→64→1 head; 145,666 parameters reported.
- Output: von Mises stress; stress-weighted loss indicated in source.

## Reproduction boundaries
The supplied six-page presentation does not specify full CAD dimensions, mesh convergence, constitutive/loading magnitudes, training data counts, split protocol, optimizer, or full normalization. Do not claim exact numerical reproduction until these are sourced.

## Implementation plan
1. Parameterized CAD for G, U, L, Z with named physical boundary regions, explicit units, and mesh checks.
2. 3D small-strain isotropic linear-elastic FEM reference, stress tensor, displacement and von Mises outputs.
3. Dataset manifest: geometry family, geometry parameters, material, load vectors, BC tags, mesh provenance, split group.
4. Baseline DeepONet matching 31/8 feature schemas, with per-point weighted stress loss.
5. Evaluation: global relL2, hotspot error, max-stress error, equilibrium/BC residuals, mesh convergence, and inference latency.
6. Holdout protocols: case, parametric geometry, and unseen family. Fit normalizers on train only.
7. VeriPhysics domain-of-validity and uncertainty gates; no unverified safety-factor claims.
8. Engineering API + interactive studio integration after numerical baseline passes.

## Acceptance criteria
- [ ] CAD and FEM cases execute end-to-end on clean environment
- [ ] Physical units and BC tags validated
- [ ] Deterministic dataset and leakage-free splits
- [ ] DeepONet trains and predicts on held-out cases
- [ ] Comparison plots use identical color limits
- [ ] Metrics and artifacts have provenance and reproducible commands
- [ ] VeriPhysics rejects out-of-domain requests
- [ ] CI smoke tests pass

## Reference
User-provided six-page presentation, pages 1–6. Original results are references, not reproduced outcomes.
