# Danilo Rizzo de Oliveira — reproduction fidelity ledger

Source: six-page user-provided ICMC-USP presentation,
*A DeepONet Surrogate Model for Stress Fields in Parametric Bracket Families:
The Cost of Geometric Generalization*.

## Source-observed and implemented

| Detail | Presentation | Current implementation |
|---|---|---|
| Families | G, U seen; L, Z unseen | Four original CAD approximations, **not** original geometries |
| Branch | 31 features: 6 loads, 2 material, 12 family, 11 geometry | Explicit reconstructed 31-feature encoding |
| Trunk | 8 features: xyz, two surface distances, concave edge distance, two proximity values | 8 features; concave edge currently approximate mesh-neighbour proxy |
| Branch/trunk | 4 × 128 tanh each | Implemented |
| Fusion | Elementwise product | Implemented |
| Head | 128 → 64 → 1 | Implemented |
| Output | s × head prediction | Optional per-case scale supported; current training uses global stress normalization |
| Loss | w_i = 0.3 + 0.7 t_i / mean(t) | Implemented with per-case mean |
| Parameters | 145,666 reported | **112,641** for the literal stated layer dimensions; mismatch unresolved |
| Error shown | G 0.10; U 0.08; L 0.16; Z 0.15 representative relative L2 | Not reproduced; no training results yet |

## Missing original artifacts for exact replication

- STEP/CAD files or precise dimensions for each family, including fillets,
  hole positions, support and load surface tags.
- All 12 family one-hot definitions and precise ordering of 11 geometry values.
- Full normalization of F, M, length reference, stress scale s and trunk coordinates.
- Definition of t_i in the loss and reduction details.
- Data generation DOE, sample counts, splits and random seeds.
- FEM element type, mesh density, solver, materials, loads, BCs and convergence study.
- Optimizer, schedule, batch size, epochs, checkpoint selection and full model definition
  resolving the 145,666 vs 112,641 parameter discrepancy.

## Reproducibility levels

- **Level A — architectural reconstruction:** implement the reported features
  and model blocks; note explicit assumptions and unresolved parameter mismatch.
- **Level B — independently reproducible experiment:** generate and validate
  CAD/FEM data, split by geometry, train, compare errors and publish manifests.
- **Level C — exact source replication:** requires original artifacts and
  matching numerical protocol. **Not currently achievable from the presentation alone.**

Do not label Level A/B as exact source replication. No certification, safety factor,
or engineering deployment approval may be inferred from a low L2 error alone.
