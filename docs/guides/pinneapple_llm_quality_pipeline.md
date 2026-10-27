# Quality pipeline for LLM-generated physics and twin artifacts

`pinneapple_llm.qa` answers one question about a generated artifact: **what has actually been
verified, and what has not**. It does not ask a language model whether its own output is correct.

Everything below is implemented in `pinneapple_llm/qa/` and covered by
`tests/test_qa_pipeline.py` and `tests/test_qa_sandbox.py`.

## Architecture

| Stage | Module | What it does |
|---|---|---|
| A | `spec.py` | Typed, serializable `PhysicalSpecification`: geometry, parameters with units, equations, boundary conditions, invariants, acceptance criteria, and the provenance of every value (`user` / `source` / `assumption` / `estimated`). |
| B | *(outside)* | Artifact generation. The pipeline never generates, so "the model wrote it" can never be confused with "the pipeline checked it". |
| C | `static.py` | AST checks plus real introspection: syntax, import allowlist, `pinneapple_*` attribute existence via `getattr`, call signatures via `inspect.signature`, forbidden calls, out-of-scope paths. |
| D | `sandbox.py` | Subprocess execution with rlimits, scrubbed environment, timeout and an in-process audit hook (no network, no process spawn, no file access outside the workdir). |
| E | `validators/` | `units`, `numeric`, `physics` (declarative invariants), `geometry` (including the hollow-body check), `temporal`, `sensors`, and an optional bridge to the existing `PhysicsGuardrail`. |
| F | `references.py` | Comparison against independent reference data with caller-declared metrics, in-sample detection, baseline and per-group breakdowns. |
| G | `repair.py` | Structured diagnosis, bounded repair loop (default 3 attempts), criteria-tamper detection, identical-diff detection. |
| — | `levels.py` | Q0–Q4 computed from the evidence that ran. |
| — | `report.py` | Self-contained JSON report: statuses, hashes, versions, timings, limitations, reproduce string. |

### Generation and validation flow

```
spec (A)  ->  artifact (B, yours or an LLM's)
                    |
                    +-> static checks (C)  -----------\
                    +-> sandboxed run (D, opt-in) ------> report (JSON)
                    +-> scientific validators (E) -----/        |
                    +-> reference comparison (F) ---------------+
                    +-> bounded repair (G, opt-in) -------------+
                                                          quality level Q0-Q4
```

The pipeline calls stage G only if you pass `max_repair_attempts` and a `repair` callable. Nothing
is published, deployed or committed by any stage.

### Stage B: the model's structured claims

`qa.generation.generate_artifact(spec, provider="ollama", model="llama3.1")` asks any already
supported provider for an envelope containing the code plus the assumptions, dependencies, units,
equations and limitations the model claims. `qa.generation.check_claims()` then confronts each
declaration with the specification: units must parse and match, dependencies must be in the static
allowlist, assumptions the spec does not contain are flagged as unverified extras.

The envelope is **not evidence**. It is recorded in the report (`generation` step and a
`generation_metadata` artifact) and every mechanically checkable claim in it is checked; the code
still has to pass stages C–F.

```python
artifact = qa.generate_artifact(spec, provider="ollama", model="llama3.1")
report = qa.QualityPipeline(spec).run(generated=artifact, meshes=..., fields=..., times=...)
```

Providers are the ones `pinneapple_llm` already supports: `anthropic` and `openai` (bring your own
key, `pip install "pinneapple[llm]"`), and `ollama` for a fully local model with no data leaving the
machine. Nothing in stages C–G needs a provider at all.

### Status vocabulary

`PASS`, `FAIL`, `BLOCKED`, `NOT_RUN`, `NOT_APPLICABLE`. They are never folded into each other:

- `BLOCKED` — the check applies but could not run (missing data, missing permission).
- `NOT_RUN` — applicable but not executed in this run.
- `NOT_APPLICABLE` — genuinely does not apply (e.g. no time axis for a steady artifact).

A validation that did not run never contributes a pass to a quality level.

## Defining a specification

```python
import pinneapple_llm.qa as qa

spec = qa.PhysicalSpecification(
    objective="Track pipe wall wear over five years",
    phenomenon="erosive wall thinning",
    geometry=qa.GeometrySpec(
        kind="hollow_pipe",
        dimensions={"outer_radius": qa.Quantity(0.10, "m"),
                    "length": qa.Quantity(2.0, "m")},
        hollow=True,
        wall_thickness=qa.Quantity(0.010, "m"),
    ),
    independent_variables=["t"],
    dependent_variables=["wear_depth", "residual_thickness"],
    parameters={
        "wear_depth": qa.Quantity(0.0, "mm"),
        "minimum_wall": qa.Quantity(2.0, "mm", qa.Provenance.SOURCE, source="ASME B31.3 t_min"),
    },
    governing_equations=["dw/dt = rate", "e(t) = e0 - w(t)"],
    temporal_validity=(qa.Quantity(0.0, "year"), qa.Quantity(5.0, "year")),
    invariants=[
        qa.Invariant("wear_nonnegative", "nonnegative", "wear_depth"),
        qa.Invariant("residual_min", "min", "residual_thickness", params={"limit": 2.0}),
        qa.Invariant("wear_units", "unit_compatible", "wear_depth", params={"unit": "mm"}),
    ],
    validation_metrics={"max_wear_mm": {"limit": 8.0}},
    required_validations=["units", "numeric", "physics", "geometry", "temporal"],
)
```

Invariant kinds accepted by `PhysicalSpecification`: `nonnegative`, `max`, `min`, `range`,
`monotonic_nondecreasing`, `conservation`, `unit_compatible`, `finite`. Arbitrary expressions are
refused on purpose: the specification may come from an LLM, and a spec able to execute code would be
a security hole.

`spec.validate()` reports structural problems (duplicate invariants, incompatible validity bounds,
a boundary condition naming an undeclared variable, a hollow body without a wall thickness).
`spec.missing_information()` lists what was left unspecified — it is reported, never filled in.

## Running the pipeline

```python
import numpy as np
import pinneapple_llm.qa as qa

pipeline = qa.QualityPipeline(
    spec,
    validators=("units", "numeric", "physics", "geometry", "temporal", "sensors"),
    reference=qa.ReferenceData(
        name="analytical-residual",
        origin="analytical",          # experiment | analytical | solver | benchmark | field_measurement
        version="1.0.0",
        values=np.array([...]),        # same length as the compared field
        unit="mm",
        limitations=["closed-form, uniform wear rate"],
    ),
    reference_criteria={"rmse": 0.5, "max_abs": 1.0},
    reference_field="residual_thickness",
)

report = pipeline.run(
    meshes={"wall": (vertices, faces)},
    fields={"wear_depth": wear},       # (V,) or (T, V)
    field_units={"wear_depth": "mm"},
    times=times, time_unit="year",
    sensors=[qa.SensorReading("UT-101", (0.1, 0.5, 0.0), "mm", "residual_thickness",
                              series=series, holdout=True)],
    output_manifest="out/quality_report.json",
)
print(report.summary())
```

Adding generated source code and enabling execution:

```python
from pinneapple_llm.qa import SandboxExecutor, SandboxLimits

report = qa.QualityPipeline(spec, run_artifacts=True,
                            sandbox=SandboxExecutor(allow_execution=True,
                                                    limits=SandboxLimits(timeout_s=30))).run(
    source=generated_code,
    fields=..., meshes=..., times=...,
    exporter=lambda out: build_and_export(out),   # failure is recorded, never swallowed
)
```

## Quality levels

| Level | Granted when |
|---|---|
| Q0 | an artifact exists and nothing was verified |
| Q1 | for source artifacts: syntax, imports, API contracts pass. For data artifacts: at least one structural validator (`numeric`, `geometry`, `temporal`) passes and none failed. |
| Q2 | every applicable validator passed **and** every validator named in `spec.required_validations` passed. A required validator that is `NOT_APPLICABLE` blocks the level. |
| Q3 | Q2 plus a reference comparison that passed the caller's metrics and recorded the reference's origin and version |
| Q4 | Q3 plus `operational_evidence` naming `environment`, `reviewer`, `reviewed_at` and `monitoring` |

Q4 cannot be granted by tests, simulations or an LLM. `report.level_reasons` always states why the
run stopped at its level.

A specification that names no `required_validations` caps at Q1: a level has to rest on evidence the
spec itself demanded, not on whichever validators happened to run.

## Running the reference example

```bash
python -m pinneapple_llm.qa.examples.pipe_wear_twin_main /tmp/pipe_twin
python -m pinneapple_llm.qa.examples.pipe_wear_twin_main /tmp/pipe_twin_anomaly --anomaly
```

It builds a hollow pipe mesh, a wear zone over five years, four thickness sensors (one holdout),
exports `scene.json`, `geometry.glb`, `fields.bin`, `wear_dataset.json` and `quality_report.json`,
and reaches **Q3** on the nominal condition. The `--anomaly` condition breaches the minimum wall and
is expected to fail `validator:physics` and the reference comparison — that failure is the point of
the example.

Open the twin with the bundled viewer (`pinneapple_twin3d.serve`) or import `geometry.glb` into
Blender; the field arrays and units are described in `scene.json`.

The example is **verification**, not validation of a real pipe: the dataset is synthetic, the
reference is the closed-form uniform-wear solution, and the report says so.

## Reference data

Checks that need it: `reference_comparison`, and therefore Q3. Without it a run can demonstrate
internal consistency (Q2) and nothing more. `ReferenceData` refuses to be constructed without an
origin and a version, and marks comparisons computed on calibration data as `in_sample`, which
always fails the level even if the metric passes.

## Adding a scientific validator

```python
from pinneapple_llm.qa.validators import ValidationContext, ValidatorResult, register
from pinneapple_llm.qa.report import Finding, Status

@register("thermal_gradient")
def check_thermal_gradient(ctx: ValidationContext) -> ValidatorResult:
    field = ctx.fields.get("temperature")
    if field is None:
        return ValidatorResult("thermal_gradient", Status.NOT_APPLICABLE,
                               evidence={"reason": "no temperature field"})
    peak = float(field.max() - field.min())
    limit = ctx.spec.validation_metrics.get("max_delta_T", {}).get("limit")
    if limit is None:
        return ValidatorResult("thermal_gradient", Status.BLOCKED,
                               evidence={"reason": "spec declares no max_delta_T"})
    ok = peak <= limit
    return ValidatorResult(
        "thermal_gradient", Status.PASS if ok else Status.FAIL,
        findings=[Finding("thermal_gradient.peak", Status.PASS if ok else Status.FAIL,
                          f"peak delta T = {peak:g} K (limit {limit:g})")],
        evidence={"peak": peak, "limit": limit},
    )
```

Then add `"thermal_gradient"` to `spec.required_validations` and to the pipeline's `validators`.
Import your module before constructing the pipeline so the `@register` decorator runs.

## Security: what the sandbox does and does not do

Implemented: separate process, own temporary directory, scrubbed environment, POSIX rlimits (CPU,
address space, file size, open files, no core dumps), wall-clock timeout, and a Python audit hook
that refuses network connections, process creation and file access outside the artifact directory
and the interpreter's library paths. Produced files are hashed.

**Not** implemented, and stated in every `SandboxResult.limitations`: no container, namespace,
seccomp filter or chroot; the audit hook is enforced in-process and can be bypassed by native code
(`ctypes`, C extensions); rlimits are advisory on some platforms. This is a policy layer against the
ordinary failure modes of generated code, not a boundary between hostile code and the host.

Execution is refused unless `allow_execution=True` (directly or via `run_artifacts=True`). When it
is not enabled the `execution` step does not appear as a pass — it is `BLOCKED`.

## Reproducing a run

Every report carries `spec_id`, `spec_hash`, `criteria_hash`, the SHA-256 of every artifact, the
providers/models involved, the Python and package versions, and a `reproduce` field. To reproduce:

1. load the report (`qa.QualityReport.read(path)`);
2. rebuild the specification from `spec` (serialize it with `spec.to_json()` alongside the report
   when you need exact reproduction);
3. confirm the artifact hashes match `report.artifacts`;
4. re-run the same pipeline call and compare `to_dict()`.

No part of the diagnosis depends on the conversation that produced the artifact.

## Running the tests

```bash
python -m pytest tests/test_qa_pipeline.py tests/test_qa_sandbox.py -q
```

Everything there is deterministic and offline. The existing LLM tests keep their own convention: the
real-model tests skip when Ollama is not reachable.

## Known limitations

- Static validation is source-text analysis. `*args`, `**kwargs` and dynamically built calls escape
  the signature check.
- The geometry hollow-body check needs a closed mesh and a cylinder-like specification; other shapes
  fall back to reporting watertightness only.
- Automatic repair in `QualityPipeline` re-runs validators against the same artifact bundle; it does
  not re-execute repaired code to rebuild data. Use `RepairLoop` with your own `evaluate()` for that.
- Unit analysis checks dimensions, not equations. A dimensionally consistent wrong equation passes.
- Nothing here establishes agreement with a real physical system. That requires real reference data,
  and Q4 requires operational evidence and human review.
