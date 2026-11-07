# The `.ppx` container: a result that can be reproduced elsewhere

`pp.save(artifact, "experiment.ppx")` writes one file holding a result and everything needed to reproduce it: the
problem, the model and its weights, the mesh and any other array, the units, the solver and its settings, the
configuration, the measured metrics, the validation checks, the environment and the provenance. `pp.load` reads it
back, and `pp.verify_artifact` checks it without trusting it.

```python
import pinneapple as pp

solution = pp.solve(problem, "fem", ctx=ctx, n=32)
pp.save(solution, "poisson.ppx", samples=64)        # 64 probe points and the values given there

art = pp.load("poisson.ppx")                        # IntegrityError if a member changed
art.problem, art.units, art.solver["method"], art.metrics
art.reproduce(lambda X: pp.solve(problem, "fem", ctx=ctx, n=32).predict(X))   # same result, on this machine

pp.verify_artifact("poisson.ppx")                   # Report(passed=True, 6 checks, 0 failed)
```

## What is inside

A `.ppx` is a zip of three things:

| Member | What it holds |
|---|---|
| `manifest.json` | schema `pinneapple.ppx/1`, the problem (as `PhysicalProblem.to_dict`), units per coordinate and field, solver (method, kind, wall time, backend info), config, metrics, validation checks, datasets, environment, provenance, and the `integrity` block |
| `arrays/*.npy` | every array the manifest names: mesh points and cells, fields, dataset arrays, probe points and their expected outputs |
| `model/state.pt` | the weights of the model the artifact carries, when it carries one |

The manifest never states a number the file does not back: `integrity` lists the sha256 and size of every member, and
`pp.load` refuses a file whose members do not match (`IntegrityError`) or that carries a member the manifest does not
list. An artifact is not a place to hide a number.

## Provenance, and why a file without it is refused

`provenance` records where the result came from: `source` (which call wrote it), `created_at`, the git commit, branch
and dirty flag of the checkout, and the environment (interpreter, OS, architecture, versions of numpy, torch and the
library). When the checkout is not a git repository the block says so (`git.commit: null` and a `reason`) instead of
leaving the field out.

`pp.load(..., strict=True)` — the default — raises `MissingProvenance` when `source`, `created_at`, `environment` or
`git` is absent or empty. That is the point of the format: an anonymous result is not a result, and `strict=False`
exists only for a reader who is inspecting a file rather than trusting it.

## Reproducing on another machine

`pp.verify_artifact` runs the checks it can:

- `container`: the file is a `.ppx` this version reads;
- `integrity`: every member matches its sha256, and nothing unlisted is inside;
- `provenance`: the block above is complete;
- `sections`: problem, units, solver, environment and provenance are not empty;
- `measurements`: how many metrics and validation checks were recorded (reported as `None`, not passed, when there are none);
- `units`: every field of the problem has a unit;
- `reproduction`: see below.

There are two kinds of reproduction, and the difference is worth knowing.

**A model.** The artifact records the class (`module:Class`) and the keyword arguments that built it next to the
weights, so the model is rebuilt from the file alone and its outputs are compared to the probe outputs stored with
it, under `rtol`/`atol`:

```python
pp.save(pp.Artifact.from_model(model, init_kwargs={"width": 64}, problem=problem,
                               lo=(0, 0), hi=(1, 1), samples=32), "model.ppx")
pp.verify_artifact("model.ppx")     # rebuilds the model and compares 32 predictions
```

Without `init_kwargs` the weights are still stored, but nothing can rebuild the model: `build_model()` then says so
instead of guessing, and the reproduction is reported as skipped with that reason.

**A solved problem.** The classical backends take a callable source (`ctx={"source_fn": ...}`) that no container can
carry, so the artifact records the problem, the method and the settings, and `reproduce` takes the callable that
produces the result again:

```python
report = pp.load("poisson.ppx").reproduce(lambda X: pp.solve(problem, "fem", ctx=ctx, n=32).predict(X))
report.passed, report.checks[0]["detail"]        # True, 'max |difference| = 0 (rtol 1e-09, atol 1e-12)'
```

A reproduction whose probe arrays are missing from the file is a **failed** check, not a skipped one: a container that
cannot be checked is not evidence.

## What it is not

- not a registry: `pinneapple_registry` stores artifacts for deployment, this is the file a result travels in;
- not a replacement for `experiment.json` (`pinneapple_lab.manifest`), which describes a study for the site and the
  Lab; a `.ppx` is the result itself, and `Solution.to_experiment` still writes the manifest;
- not a signature: provenance says where a file came from, not that it is authentic — signing is `pinneapple_security`.
