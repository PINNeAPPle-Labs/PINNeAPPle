# The physical data graph: from a prediction back to the problem

A prediction on its own is a number nobody can defend. The question a reviewer asks is the chain behind it: which
problem, on which geometry, over which simulation runs, from which dataset version, with which model version, and
what validation said about it. `pinneapple_data.lineage` answers that question with a graph of artifacts and
"derived from" edges, and — for the four stores of `pinneapple_registry` — builds it without anyone typing the edges
twice.

```python
from pinneapple_data.lineage import from_registry
from pinneapple_registry import ArtifactRegistry

reg = ArtifactRegistry("artifacts")
g = from_registry(reg, "pipe_flow_3d", prediction="PRED-001", model_version="v20261124_101530")
g.trace("PRED-001")
```

## What a trace returns

`Lineage.trace(artifact)` groups the artifacts upstream of the one you name by the question they answer:

| Group | Kinds that answer it |
|---|---|
| `problem` | `problem` — the stored problem spec, with its version |
| `geometry` | `geometry`, `mesh` |
| `simulations` | `simulation`, and a stored `dataset` when it is one simulation run's arrays |
| `datasets` | `dataset`, with the version the store recorded |
| `models` | `model`, with the version that produced the prediction |
| `validation` | `validation` — the record that graded that model version |

plus `prediction` (the artifact you asked about), `missing` (the groups nothing answers) and `complete` (true only
when `missing` is empty):

```python
answer = g.trace("PRED-001")
answer["missing"]          # ['geometry'] — nobody recorded the geometry this prediction came from
answer["complete"]         # False
answer["validation"][0]    # {'id': 'validation:pipe_flow_3d@v2026…', 'kind': 'validation', 'version': 'v2026…',
                           #  'derived_from': ['model:pipe_flow_3d@v2026…'], …}
```

Each entry is the short form of the artifact — id, kind, name, version, file, software, timestamp, hash, origin, and
what it was derived from — so the answer is JSON, and a report can quote a model version instead of a label.

## What the graph builds from the stores, and what it refuses to build

`from_registry` reads what the stores already write, and writes only those edges:

| Edge | Where it comes from |
|---|---|
| problem versions → the records filed under them | `ProblemStore.versions(problem_id)`; the newest version is the one the other records link to, and `preset.json`/`snapshot.json` is the node's content |
| dataset → problem | the `problem_id` every record is filed under |
| dataset → geometry | the `geometry`, `geometry_id`, `cad` or `mesh` key in the dataset metadata |
| model → dataset | the `datasets`, `dataset`, `dataset_id`, `scenarios` or `trained_on` key in the model metadata |
| model → experiment run | the `run_id` (or `experiment_run`) key in the model metadata |
| validation → model | the `trust_report` or `validation` block stored with the model version |
| prediction → model | `model_version=`, or the newest stored version when you do not pin one |

Everything else is a hole, and a hole is reported as one. A model whose metadata never named a dataset is linked to
the problem and to nothing else; its node carries the note *"the model metadata names no dataset: the dataset version
behind it is unknown"*, and `trace()` puts `datasets` in `missing`. The module does not guess a dataset by "the only
one stored", because a guess in a lineage is worse than a gap: the gap is visible, the guess is not.

## The validation record: two shapes, one verdict

Two records are written in this library, and both are read:

- a `TrustReport` (or its `to_dict()`) under `trust_report`, through `ModelStore.save(trust_report=...)` — it carries
  `status`;
- a `ValidationReport` from `pinneapple_analysis.validation` — the physics checks — whose `to_dict()` carries
  `passed`, `n_passed`, `n_failed`, `n_not_run` and `timestamp`.

The node takes the verdict from whichever is there (`status`, `verdict`, `grade`, or `passed` as PASS/FAIL) and keeps
the counts, so `trace()["validation"][0]` says *PASS with 4 checks* rather than only "a validation record exists". A
record that carries no verdict at all is stored with the note that says so, instead of an empty field that reads as
fine.

## Limits, said plainly

- **The validation node is a record, not a re-run.** It reports the verdict stored with the model version; nothing
  re-runs the checks at trace time, so a record written a month ago still reads as it was written.
- **A dataset is one simulation run.** `DatasetStore` keeps one run's arrays per `problem/scenario`, so it answers for
  both `simulations` and `datasets`. A campaign of several runs appears as several datasets, not as one dataset with
  many runs.
- **The stores record `problem_id`, not which problem version a record used.** A dataset or model made before the
  problem's last version still links to the newest version, and the node says the store never recorded the version it
  was built from.
- **A stored dataset has no version field.** When the metadata does not carry one, the node's version is the creation
  timestamp and the node says so.
- **The experiment store gives runs and metrics, not datasets.** `run_id` links a model to its training run only when
  the caller wrote that key into the model metadata; the store does not record it by itself.
- **The chain form is a straight line.** `Lineage.from_json({"problem": …, "geometry": …, "prediction": …})` has no
  place for a validation record, which is a branch off the model: write it as an artifact
  (`{"artifacts": [{"kind": "validation", "inputs": ["MODEL"]}]}`) or let `from_registry` build it.

## Checks and exports

The graph is the same object the rest of the lineage module already had, so everything else works on it: `checks()`
(broken references, cycles, changed files, stale derivations, mixed revisions), `report()` for a viewer,
`to_prov()` for W3C PROV-JSON, `to_markdown()`, and the questions `upstream()` / `downstream()` /
`sources_of_kind()`.
