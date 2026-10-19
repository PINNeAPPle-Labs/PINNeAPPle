# Physics Validation Lab

**Is this model or result physically and numerically consistent? Every check, with the record that it ran.**

Day 24 of the PINNeAPPle app series (validate.pinneapple.org, port 8107).

**The pain.** A surrogate, a PINN or a new solver version produces numbers that look plausible. Before anyone uses
them, someone has to ask several questions:

- Are the equations satisfied?
- Is mass or energy conserved?
- Do the boundary conditions hold, and are the units right?
- Does it match the reference?
- Is it being queried where it was trained?

The answers usually live in ad-hoc scripts. The record that they were asked at all usually lives nowhere.

| You give it | You get |
|---|---|
| Predictions or results: OpenFOAM case (zip), VTU/VTK, NPZ, HDF5, CSV, Parquet or JSON tables. Reference data. The equations, parameters, units, bounds and boundary conditions, written in a form or in a `pinneapple check` configuration (`pinneapple-check.toml`). Or a ModelRef to a zoo model plus query points. | Residuals, conservation errors, boundary-condition errors, constraint violations, dimensional consistency, reference error and out-of-range use, **in three separate groups**. Per target, the groups where nothing ran. An evidence record per check (SHA-256 of its inputs, outputs, start and finish time). The report as JSON and HTML, the configuration that reproduces it, and the evidence ledger. |

## The three groups

| Group | Question | Checks (`pinneapple check` rules) |
|---|---|---|
| Numerical verification | Were the equations solved right? | PDE residual of stored fields and of models (PC301, PC302), finite values (PC403), readable inputs (PC001), artefact integrity: hashes, signatures, unsafe pickles (PC8xx), method status in the catalog (PC901) |
| Validation against data | Does it match trusted data, inside the range the data covers? | Field and QoI regression against a reference (PC701–PC704), out-of-range queries and sparse training regions (PC601, PC602), dataset leakage and diversity (PC452, PC453) |
| Physical consistency | Does it obey the physics? | Mass, energy, momentum, divergence, conserved totals (PC101–PC105); boundary and initial conditions, walls (PC201–PC203); bounds and constraints (PC401, PC402); units and magnitudes (PC501–PC504); dataset physics (PC451) |

No single test guarantees that the model represents reality, and neither does the whole set. The app and the
report both say so. A pass means only that the listed checks ran on these inputs, with these tolerances, and held.
A check that did not run is shown as **not run**, never as a pass.

## On the mesh: 3-D results

When a result is 3-D and lives on a mesh (an OpenFOAM case zip, a VTU/VTK file with cells) and has a reference, the
report adds a 3-D view of what the reference check (PC701) compared: the prediction, the reference and
|prediction − reference| on the mesh, each with a colour bar. A displacement field deforms the body (scale factor,
true scale). Declared boundary conditions are marked (OpenFOAM patches, planes such as `x=min`). A list gives the
regions with the largest error: connected points in the top 10 % of the error, with their maximum and mean, size,
share of the squared error and the faces of the domain they touch. Click a row to zoom to it. The relative L2 shown
is the PC701 number. 2-D results, tables and models keep the current views. Library:
`pinneapple_veriphysics.mesh_views`, `pinneapple_analysis.validation.error_map`, `pp.viz.field_view`.

![A reduced model against the FEM it replaces: |prediction − reference| of the von Mises stress on the deformed cantilever](../../docs/assets/apps/d24-validate-3d-stress.jpg)

## Integrations

- **REST API** (`/docs`): `POST /api/validate` takes files plus a `config` (TOML) or a `spec` (JSON). It returns
  the report (`pinneapple-validation-lab/v1`), plus `views3d` for 3-D results with a reference (not part of the hashed
  report; the glTF is at `/api/views/{id}.glb`). `POST /api/report.html` renders it; `POST /api/config` turns a spec
  into a `pinneapple-check.toml`.
- **CI:** the GitHub Action at the root of this repository (`uses: PINNeAPPle-Labs/PINNeAPPle@main`) runs the same
  checks on every pull request, with annotations, SARIF and a job summary. The API tab gives a workflow that also
  stores the lab's report for each model version.
- **Other apps:** the report JSON, with its evidence ledger, is an input of the
  Engineering Evidence Report (day 30). Zoo models are referenced by name, like the
  ModelRef of the shared data contracts.

## Methods

The library does the work; this app only handles uploads, routes and UI:

- `pinneapple_check.run` runs the physics checks: finite-volume fluxes on OpenFOAM meshes, finite differences on
  grids, least squares on point clouds, unit parsing, the training envelope and regression against a reference;
- `pinneapple_veriphysics.validation_lab` adds the groups, the not-run lists, the ledger and the provenance;
- `pinneapple_veriphysics.ledger` keeps the records.

The validation of each rule is in `tests/test_physics_check.py`: on the fixtures in `examples/physics_check`, every
rule fails exactly where a defect was planted and passes on the clean twin. The ledger and grouping are tested in
`tests/pinneapple_veriphysics/test_evidence_ledger.py`, the app in `tests/test_app_validation_lab.py`.

## Try with example (offline, seconds each)

| Example | What it shows |
|---|---|
| Fifteen results and models that hold | The 135 checks of the clean fixtures: OpenFOAM cavity and channel, conduction, elasticity, a 1-D conservation law, a PINN, a surrogate, a Darcy dataset, signed artefacts |
| The same targets, one defect each | 32 errors and 9 warnings, each in its group (leaking outlet, wrong BC, residual, unit mismatch, out-of-range query, regression, tampered file...) |
| Temperature predictions from a form | `heat_plate.npz` against k∇²T + q = 0, its boundary value, units and bounds: 12 checks; the validation group reads "not run" because no reference was given |
| A reduced model at its worst design, against the FEM, in 3-D | Chosen to fail. A solid cantilever (H 0.289 m, W 0.173 m, load 4.8° from vertical): the POD + GPR model of the ROM Lab (rank 8, trained on 80 FEM designs; 4.4 % mean error over its 20 held-out designs) at the held-out design where it is worst, near the edge of its training box, against a fresh FEM solve of that design (`examples/make_cantilever_3d.py`). It shows a failing reference check, not the ROM's typical quality. The clamp condition holds (PC201); the reference check fails, 21.2 % on the displacement and 27.8 % on the von Mises stress (PC701). The 3-D view shows the stress error at the clamp corners and the displacement error at the tip |
| Zoo heat-sink model at three operating points | `heatsink_forced_mlp` at three designs: the third (20 m/s air, trained on 0.5–6 m/s) fails PC601 |

## Run

```bash
pip install -e . -r apps/validation_lab/requirements.txt              # from the repository root
cd apps/validation_lab && uvicorn validation_lab.api:app --port 8107
```

Environment variables:

- `VAL_USER` / `VAL_PASSWORD`: HTTP Basic login on everything except `/health`;
- `VAL_MAX_MB`: largest upload, default 200;
- `VAL_MAX_HEAVY`: validations at once per worker, default 2; the excess gets HTTP 429;
- `VAL_VIEW_MAX_FACES`: triangles of a 3-D view before it is decimated, default 250,000;
- `VAL_VIEWS_KEPT`: 3-D views (glTF) kept on disk, newest first, default 300; `VAL_DATA_DIR`: where.

Docker: `docker build -f apps/validation_lab/Dockerfile -t validation-lab .`. Deployment with the other apps is in
[`apps/deploy`](../deploy) (service `validate`).

## Limits

- **Coverage depends on what you declare.** No reference means no validation against data. No physics means no
  residual or conservation check. The report lists these gaps per target.
- **Your own model code is not run here.** Python callables and checkpoints would execute uploaded code, so they
  are refused. Upload predictions, or run `pinneapple check` locally or in CI. Zoo models are allowed.
- **Tolerances decide the verdict.** The defaults of `pinneapple check` apply unless you set `tolerances` per
  target (mass imbalance 0.1 %, relative residual 5 %, relative L2 1 %, ...). They are written into the report.
- **Report format.** The format is `pinneapple-validation-lab/v1`. The `validation-report/v1` artifact of
  `pp.validate`, its credibility levels and the fault-injection, metamorphic, residual-bound, rollout and PINN
  checks come from PRs that are not merged yet (#734, #738, #735, #719, #728, #741, #747). They will be added here
  once they are on `main`.
- **No asynchronous job queue.** A validation runs inside the request, which is fine for files of up to a few
  hundred MB. The shared job queue and usage metering of the app runtime will replace the per-worker limit once
  they land.

Monetization: validations per model, dataset or version (a free monthly allowance, then per validation). Teams pay
for private projects, stored reports and CI integration with history per model version.
