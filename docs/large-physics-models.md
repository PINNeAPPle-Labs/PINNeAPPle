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

## Production hardening changes (latest)

- `pinneapple.lpm.service.create_app` now refuses to start without a
  production-approved model card, matching SHA-256 checkpoint digest, and a
  configured `PINNEAPPLE_LPM_API_KEY` (minimum 32 characters). Prediction
  requests require `X-API-Key`; node count is bounded by
  `PINNEAPPLE_LPM_MAX_NODES` (default 50,000).
- Model checkpoint inference uses tensor-only deserialization.
- **The local expert-review CLI is explicitly NOT an authorization authority.**
  Its self-attested review identity and supplied metric values can never
  promote a checkpoint; the model card stays blocked.
- The `lpm-integration.yml` CI workflow adds Python compilation, schema
  checks, preparation of all 240 planned cases, and an OpenFOAM solver smoke
  job. CI results must be inspected; creating a workflow is not evidence
  that it passed.

### Still required for an actual production release

A trusted authenticated review backend with independently computed metrics
and immutable evidence is not implemented. Solver backends must be executed
against pinned versions, validated numerically, and reviewed for mesh and
field consistency. A real deployment additionally needs TLS at ingress,
network controls, audit logs, rate limiting, resource budgets, observability,
model rollback, and documented operating procedures. Until then the
approval pathway is deliberately fail-closed and the PR must remain draft.

## Authenticated approval service (experimental)

`pinneapple.lpm.approval_service.create_app(campaign_root)` provides an
OIDC-protected approval endpoint. Configure:
`PINNEAPPLE_LPM_OIDC_ISSUER`,
`PINNEAPPLE_LPM_OIDC_AUDIENCE`,
`PINNEAPPLE_LPM_OIDC_JWKS_URL` (HTTPS),
and `PINNEAPPLE_LPM_REVIEWER_SUBJECTS` (comma-separated allowlist of
authorized OIDC subject identifiers). Install `PyJWT[crypto]` and FastAPI
in the approval-service environment.

`POST /approve` requires a Bearer JWT and a JSON object containing
`reference_path`, `reference_sha256`, `justification` and `scope`.
The independent evidence NPZ must contain exactly `features` and
`reference` arrays. The service runs the checkpoint on the supplied
features and computes RMSE, MAE and relative L2 itself. It checks
checkpoint/evidence SHA-256 and the native VeriPhysics decision. It never
accepts a caller-supplied measured error.

**Deployment security note:** The approval service currently accepts a
server-side file path. Do not expose it to untrusted tenants or deploy it
as a public service without an immutable evidence store, path allowlist,
audited upload workflow, reviewer-to-campaign authorization, transaction
locking, and a second independent evidence provenance check. An authenticated
reviewer can still submit non-independent or fabricated reference arrays;
OIDC authentication does not certify scientific independence. This is an
integration foundation, not production certification.
