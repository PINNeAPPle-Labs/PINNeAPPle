# Physics Validation Lab (day 24)

**Is this model or result physically and numerically consistent?** Live at <https://validate.pinneapple.org>.
Source: [`apps/validation_lab`](https://github.com/PINNeAPPle-Labs/PINNeAPPle/tree/main/apps/validation_lab).

![Report of the fixtures with one defect each: three groups, per-target status](../assets/apps/d24-validate-report.jpg)

Upload predictions or simulation results. Describe the equations, parameters, units, bounds, boundary conditions
and reference data in a form, or upload a [`pinneapple check`](../guides/physics_checks_in_ci.md) configuration,
or reference a zoo model with query points. The lab runs the physics checks and reports them in three groups that
answer different questions:

| Group | Question |
|---|---|
| Numerical verification | Were the equations solved right? Residuals, finite values, intact artefacts, method status |
| Validation against data | Does it match trusted data, inside the range the data covers? Reference error, QoIs, out-of-range queries |
| Physical consistency | Conservation, boundary and initial conditions, bounds and constraints, units |

For each target, the report says which groups did not run (for example, no reference means no validation against
data). Each check that ran has an evidence record: the SHA-256 of its inputs, its outputs, and start and finish
times ([evidence ledger](../guides/evidence_ledger.md)). The report also holds the configuration and the hash of
every input file, so `pinneapple check --config pinneapple-check.toml` reproduces it.

No single test guarantees that a model represents reality, and neither does this set of tests. A pass means only
that the listed checks ran and held on these inputs, with these tolerances.

![A zoo model queried outside its training range: the validation group fails](../assets/apps/d24-validate-zoo-ood.jpg)

## On the mesh (3-D results)

When a result is 3-D and lives on a mesh (an OpenFOAM case zip, a VTU/VTK file with cells) and has a reference, the
report adds a 3-D view of what the reference check (PC701) compared: the prediction, the reference and
|prediction − reference| on the mesh, each with a colour bar. A displacement field deforms the body (scale factor,
true scale). Declared boundary conditions are marked (OpenFOAM patches, planes such as `x=min`). A list gives the
regions with the largest error: connected points in the top 10 % of the error, with their maximum and mean, size,
share of the squared error and the faces of the domain they touch. Click a row to zoom to it. The relative L2 shown
is the PC701 number. 2-D results, tables and models keep the current views. Library:
`pinneapple_veriphysics.mesh_views`, `pinneapple_analysis.validation.error_map`, `pp.viz.field_view`.

The 3-D example is chosen to fail: the ROM Lab's POD + GPR model (4.4 % mean held-out error) at its worst held-out
design, near the edge of its training box. It shows a failing reference check, not the ROM's typical quality.

![The ROM Lab's model at its worst held-out design against the FEM: |prediction − reference| of the von Mises stress on the deformed cantilever, with the regions of largest error at the clamp](../assets/apps/d24-validate-3d-stress.jpg)

![The displacement error of the same model: one region, at the loaded tip](../assets/apps/d24-validate-3d-displacement.jpg)

## Automate it

- `POST /api/validate` takes files plus `config` (TOML) or `spec` (JSON) and returns the report. The interactive
  docs are at `/docs`.
- In CI, the GitHub Action `PINNeAPPle-Labs/PINNeAPPle@main` runs the same checks on every pull request.
- The report JSON, with its ledger, feeds the Engineering Evidence Report (day 30).

![The form: a temperature field checked against the heat equation, its boundary value, units and bounds](../assets/apps/d24-validate-form.jpg)

## Limits

- Python models and checkpoints are not run on the server. Upload their predictions, or use the Action.
- The default tolerances of `pinneapple check` decide pass or fail unless you set them per target.
- The report is `pinneapple-validation-lab/v1`. Export to `validation-report/v1` (`pp.validate`) and its credibility
  levels will follow once those PRs are merged.

Monetization: validations per model, dataset or version, with private projects and CI history for teams.
