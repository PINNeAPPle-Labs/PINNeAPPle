# FluidsBench: scoring a surrogate on the benchmark's own terms

[FluidsBench](https://fluidsbench.org) is an open benchmark for machine-learned fluid dynamics. It publishes, per
dataset, a fixed scoring support (coordinates, weights, ground truth), a set of metric definitions, a composite that the
leaderboard ranks on, and a validator that reviews submitted result packages. AirfRANS, AhmedML, DrivAerML,
DrivAerNet++ and WindsorML accept submissions; BlendedNet, HiLiftAeroML, Rotor37 and VKI-LS59 are announced.

`pinneapple_arena.fluidsbench` does the part a participant owns — read the contract, map your model's output onto the
benchmark's entities, compute the published reductions, write and check the result package — and nothing of the part
the benchmark owns. It never restates a weight, a cap or a ground-truth value: those are read from the benchmark's own
repository at run time, with their digests.

```python
from pinneapple_arena.fluidsbench import Checkout, load_support_release, map_to_support, score_support

checkout = Checkout.find()                                   # $FLUIDSBENCH_REPO or an explicit path
contract = checkout.dataset("airfrans")                      # status, splits, composite, required fields
print(contract.status, contract.open_splits(), contract.support_manifest_sha256)

release = load_support_release(contract.support_manifest, "standard")   # digest chain verified
support = release.support(case_id, "two-dimensional-domain-native")

mapped = map_to_support(support, points=my_points, fields={"velocity_x_pred": u, "velocity_y_pred": v,
                                                           "pressure_pred": p}, method="nearest")
score = score_support(support, mapped.values, coverage=mapped.coverage.as_dict())
print(score.metric_values, score.statistics["flow_domain_velocity_rel_l2"].as_dict())
```

## The division of responsibility

| the benchmark owns | this package does |
|---|---|
| the case set, the coordinates, the weights, the ground truth | reads them, verifies their digests, and joins predictions by `support_id` — never interpolates the benchmark side |
| the metric equations and the weighting tokens | implements the reductions, from the definitions, and recomputes them from the package's own numbers |
| the composite and its caps or physics-null baselines | reads them from `overall_score_composite` and re-derives `overall_score` from the component values |
| the JSON schemas and the validator | writes the package and runs **their** `scripts/validate_submission.py` when asked (`--official`) |

The last row is the important one. A number computed here is not the official score: the official evaluator writes that,
and the maintainers verify the package. What this adapter guarantees is that what leaves the machine is complete,
declared and internally consistent — and it refuses to write a package that is not.

## Reading the contract

```bash
git clone https://github.com/FluidsBench/fluidsbench-submission
export FLUIDSBENCH_REPO=$PWD/fluidsbench-submission

python -m pinneapple_arena.fluidsbench datasets            # what this checkout publishes, and its status
python -m pinneapple_arena.fluidsbench dataset airfrans     # splits with their digests, composite, required fields
python -m pinneapple_arena.fluidsbench support <manifest.json> -c standard
```

`Checkout.dataset(id)` gives a `DatasetContract` with `status`, `submissions_open`, `splits`, `open_splits()`,
`split_case_ids(split_id)` (verified against the published SHA-256), `support_release_id`, `required_fields`,
`composite`, `ranking`, and `digests()` — the SHA-256 of every file it read. A result package built on it can say which
exact contract it was scored against.

## The scoring support

A support release is the benchmark's hash chain: `manifest.json` → `case-sets/<set>/index.json` → `chunk-000.json` → one
ground-truth table per case and support. `load_support_release` verifies the index and chunk digests at load and each
table's digest when that table is read, so a 200-case release is not loaded whole to answer a question about its shape.

Only the `materialized_table` location mode (JSON, CSV and NPZ tables, local or extracted from the release archive) is
read generically, which is what the contract's own generic evaluator does. AirfRANS scores on `materialized_table` NPZ
tables; a dataset that declares `native_entities`, `structured_grid` or `reference_generator` is evaluated by its own
reference code, and this loader raises `UnsupportedLocationMode` rather than inventing a reading of it.

`write_support_release` writes a release **in the same layout** for rehearsal and for this repository's tests. It is
labelled `synthetic_rehearsal` and is not a FluidsBench release.

## Mapping: getting onto the entities, and declaring how far you reached

The benchmark scores on its own coordinates. Everything a model needs to meet them happens on the submitter's side:

| method | what it does | when it is the right one |
|---|---|---|
| `identity` | the predictions are already on the entities, in `support_id_ascending` order | the model is queried at the canonical points — the cleanest submission |
| `by_id` | the predictions carry `support_id` and are joined exactly; missing ids count as unmapped, unknown or duplicate ids are refused | the predictions come from a case directory keyed by the support ids |
| `nearest` | nearest native point, with the distance of every entity reported and an optional `max_distance` | a point-based model on its own mesh |
| `linear` | linear interpolation in the affine space the native points span (a surface cloud is interpolated on its plane), NaN outside the hull | a model on its own structured grid |

A transfer that does not meet the support's `required_coverage` raises `CoverageError` unless `allow_partial=True`, and
an `extrapolation_policy` of `forbidden` is enforced the same way. The coverage that was reached (`scored/total`,
weight fraction, unmapped, extrapolated) goes into `metrics/cases.json` and `discretization/cases.jsonl` — it is part of
the package, not a footnote.

## The reductions

The published equations, implemented from the definitions:

| reduction | equation |
|---|---|
| `relative_l2_percent` | `100 · √(Σ w (ŷ−y)²) / √(Σ w y²)` |
| `relative_l1_percent` | `100 · Σ w |ŷ−y| / Σ w |y|` |
| `mae` / `mse` / `rmse` | the weighted versions, `Σ w · / Σ w` |
| `r2` | `1 − Σ w (ŷ−y)² / Σ w (y−ȳ_w)²`, undefined for constant truth |

Two rules are enforced rather than assumed:

* **a chunk-level L2 is never averaged.** For each case and metric, the additive numerator `N = Σ w‖ŷ−y‖²` and
  denominator `D = Σ w‖y‖²` are accumulated; the case value is `100√(ΣN/ΣD)` once (`relative_l2_from_statistics`).
* **cases are averaged, not points.** `per_geometry_then_macro_average` gives every test geometry equal influence, so a
  fine mesh cannot outvote a coarse one (`macro_average`, `aggregate`).

A spatial weight multiplies the complete squared vector magnitude — one weight per entity, repeated over the components.
`weights` come from the support, never from the submitter.

Empty arrays, non-finite values, negative or zero-total weights, a zero truth norm and a constant-ground-truth R² all
raise; none of them returns a plausible-looking number.

## The composite

Every dataset ranks on `overall_score`, computed from component values with the published transforms and weights:

| transform | component contribution |
|---|---|
| `bounded_error` | `clip(100(1 − e/cap), 0, 100)` |
| `bounded_quality` | `100 · clip(q, 0, 1)` |
| `physics_null_skill` | `100(1 − e/baseline)` — unclipped, so a method worse than the published null keeps negative skill |

`ScoreComposite.score(values)` re-derives it and `group_scores` gives the intermediate `field_score` / `force_score` /
`diagnostic_score` reductions. `competition_rank` rounds half-up at the declared decimal places and gives equal values
the same rank (1, 2, 2, 4) — the leaderboard's own rule.

## The result package

`build_package` writes the schema-v3 directory:

```
submissions/<dataset-id>/<submission-id>/
  submission.json            model, method, dataset, split, hashes, aggregate metric values
  evaluation-evidence.json   the command, the release it scored against, the values it produced
  discretization.json        what the model took in, what it produced, how it reached the support
  discretization/cases.jsonl one line per case: counts, mappings, coverage
  metrics/cases.json         per case and support: values, coverage, sufficient statistics
```

Facts only the submitter knows — architecture, parameter counts, licences, hardware, what the model ingested — are
arguments, never defaults. This module does not invent a methodology record, an SPDX identifier or a compute
measurement; `METHODOLOGY.md` in the benchmark's repository says what those fields must contain, including the
zero-parameter case.

`validate_package` recomputes, from the package's own files, the identities the maintainers check: the digest chain,
`scored + unmapped = support_count`, `coverage_fraction` from the counts, every relative L2 against its sufficient
statistics, every aggregate against the macro-average of its case values, `overall_score` against its components, and
the `<series-id>-vN` revision rule. With a checkout and `run_official=True` it then runs the benchmark's own validator.

```bash
python -m pinneapple_arena.fluidsbench validate submissions/airfrans/my-model-v1 --dataset-id airfrans --official
```

## Before submitting: the trust decision

The benchmark ranks accuracy and trusts the submitter for the rest, so three questions are worth answering about your own
result. `assess_submission` answers them and records each one in a
`pinneapple_veriphysics.ledger.EvidenceLedger`, so a check reads `passed` only when an intact record shows it ran:

| check | question |
|---|---|
| scope | were the cases scored exactly the official split's cases? |
| integrity | does the support hash to what the specification publishes, and does every relative L2 agree with its statistics? |
| coverage | is every entity scored, with no unmapped and no extrapolated entities? |
| applicability | is the model inside its training range for each case? |

The decision is `submit`, `review` (the applicability check did not run, or a case is outside the training range — which
is exactly what an extrapolation split measures: report it) or `hold` (a hard check failed). A check that did not run
stays `not run`; it never becomes a neutral pass.

## Worked example

[`examples/fluidsbench_airfrans`](../../examples/fluidsbench_airfrans/README.md) runs the whole path on potential flow
around a cylinder: seven model families at equal budget, a hold-out angle inside the training range and two outside it,
one validated package per model and split. The reference field scored against itself comes out at relative L2 = 0 and
`overall_score` = 100 — that is the check that says the chain is consistent, independently of model quality.

## Limits, stated

* Profile data (`profiles/`) is part of the schema-v3 package and is extracted by each dataset's own tooling; this
  adapter writes the metrics, evidence and discretization files and leaves profile extraction to the dataset contract.
* Only `materialized_table` supports are read directly (see above).
* Schema validation is delegated: `--official` runs the benchmark's validator, which is the authority.
* The AirfRANS data release is ~10 GB and its support tables ship separately; this package needs a local checkout and,
  for archive-backed supports, the extracted tables (`archive_root=`).

## References

* FluidsBench website: <https://fluidsbench.org>, submission workflow <https://fluidsbench.org/run/>
* Submission repository (Apache-2.0): <https://github.com/FluidsBench/fluidsbench-submission> — `METHODOLOGY.md`,
  `docs/RESULT_FORMAT.md`, `reference/README.md`, `schemas/v3/`
* Metric reference implementation and equations: `reference/metrics.py`, `reference/scores.py`,
  `reference/evaluate_predictions.py` in that repository
* AirfRANS: Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating
  Reynolds-Averaged Navier–Stokes Solutions*, NeurIPS 2022 datasets track
* The cylinder solution used by the example: potential flow, `Cp = 1 − 4 sin²(θ−α)`; d'Alembert's paradox for its zero
  drag
