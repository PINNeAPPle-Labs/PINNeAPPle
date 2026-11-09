# PINNeAPPLe Large Physics Models (LPM)

The LPM workflow is integrated as `pinneapple.lpm` in the existing
PINNeAPPLe repository. It is a **reference campaign system**, not a pretrained
large foundation model or a production-validated physical simulator.

## Supported workflows

| Solver | Physics | Generator | Field extraction |
| --- | --- | --- | --- |
| OpenFOAM | 2D laminar lid-driven cavity | blockMesh + icoFoam | ASCII cell-centred U/p |
| CalculiX | 3D linear-elastic cantilever | C3D8 deck + ccx | nodal displacement .dat |
| Legacy FEniCS (dolfin) | 3D linear-elastic cantilever | generated dolfin script | vertex displacement .npz |

Campaigns sample a deterministic Latin hypercube and split by complete
simulation case (not mesh node). The reference trainers support pointwise MLP,
2D FNO, and small k-NN graph message passing. Data artifacts are compressed
NPZ files, with train-only normalization, held-out metrics, and checkpoints.

## Run

```bash
python -m pinneapple.lpm examples/lpm/cfd01_cavity_lowRe.json --out runs/lpm-cfd01 --stage prepare
python -m pinneapple.lpm examples/lpm/cfd01_cavity_lowRe.json --out runs/lpm-cfd01 --stage simulate
python -m pinneapple.lpm examples/lpm/cfd01_cavity_lowRe.json --out runs/lpm-cfd01 --stage train
# Or simulate and train in a single run:
python -m pinneapple.lpm examples/lpm/fea06_beam_force.json --out runs/lpm-fea06 --stage all
```

The ten JSON campaigns are under `examples/lpm/`, each with 24 planned
samples: 5 CFD, 3 CalculiX, 2 FEniCS. `prepare` writes input decks without
running the physical solver. `simulate` requires solver executables installed
on the machine. `train` rejects incomplete simulations. `all` does both.

## Validation and deployment

- The data audit rejects missing, nonfinite or shape-inconsistent samples.
- Holdout validation checks relative L2 against the configured threshold.
- The optional `gates.veriphysics_adapter` uses the explicit contract
  `module:function(root, records, metrics) -> {"passed": bool, ...}`.
- The native `pinneapple_veriphysics` package is available in the same
  repository, **but a verified case-specific native validation adapter is
  still required** to supply physics and independent reference evidence.
  A user-provided hook is not a substitute for physical validation.
- `require_veriphysics` defaults to true and publication fails closed
  without an executed, passing adapter.
- The local FastAPI service can be constructed with
  `pinneapple.lpm.service.create_app("runs/lpm-cfd01/deployment")`.
  Only gate-approved checkpoints are served. Do not expose this prototype
  publicly without authentication, limits and hardened checkpoint handling.

## Evidence and limitations

Generated case inputs are not completed simulations. The uploaded prototype
reported 240 planned case inputs, not 240 verified physical solutions.
No OpenFOAM/CalculiX/dolfin solver runs or complete-repository CI execution
have been performed in this integration session.

The physical families are deliberately narrow. No CAD-generalized geometry,
turbulent CFD, plasticity, contact, DOLFINx, arbitrary mesh topology, distributed
HPC scheduling, physics-residual PINO training or native registry publishing is
claimed. The graph implementation uses O(N²) neighbor searches. Numerical
convergence, conservation, mesh/time refinement, expert review, and independent
physical reference comparisons remain required for a validated release.

## Native training and engineering review (integration branch)

The campaign pipeline now delegates optimization and best-checkpoint selection
to `pinneapple_neural.trainer.Trainer`. FNO and graph architectures delegate
to existing `pinneapple_neural` implementations. OpenFOAM field parsing
delegates to the native binary/ASCII reader.

A native `ValidationCase` is persisted at `veriphysics_case.json`.
An independent reference and authorized human decision must be attached
before the checkpoint can be promoted:

```bash
python -m pinneapple.lpm.review runs/lpm-cfd01 reference \
  --artifact /path/to/independent-measurements.csv \
  --kind "experimental data" --metric relative_l2 \
  --value-pct 4.0 --tolerance-pct 5.0 \
  --description "Independently measured cavity velocity profile"
python -m pinneapple.lpm.review runs/lpm-cfd01 review \
  --reviewer "Engineer Name" --role "Responsible Engineer" \
  --decision approved --scope "engineering screening, subject to independent verification" \
  --justification "Reviewed the evidence and applicability limits"
```

**Important:** The CLI records supplied claims; it does not verify that the
reference is truly independent, that the reviewer is authorized, or that the
reported comparison was computed correctly. It must be placed behind an
authenticated, audited review service before any production deployment.
A local `approved_model.pt` file is not a production authorization token.

## Production release gates

- Run actual OpenFOAM, CalculiX and legacy FEniCS cases in pinned containers.
- Verify field ordering, mesh connectivity, dimensions, solver convergence and
  independent conservation/equilibrium checks.
- Execute unit, integration, security and performance tests in CI.
- Authenticate and authorize expert reviewers; audit evidence ingestion and
  independent metric computation rather than trusting CLI-supplied numbers.
- Harden inference with identity, authorization, resource limits, request-size
  limits, checkpoint provenance/signature, observability and rollback.
- Resolve the remaining duplicated solver invocation paths, version
  compatibility and numerical-convergence validation.

This PR is an integration implementation, **not a certified production release**.
