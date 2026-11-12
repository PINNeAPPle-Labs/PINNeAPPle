# FluidsBench, rehearsed end to end on an exact reference

[FluidsBench](https://fluidsbench.org) evaluates machine-learned surrogates on realistic fluid dynamics — AirfRANS,
AhmedML, DrivAerML, DrivAerNet++, WindsorML and more — with published splits, a benchmark-owned scoring support and a
leaderboard ranked on a composite of field errors, integrated forces and flow profiles. This folder walks the whole
**participant path** of that benchmark without a multi-gigabyte download: the reference is potential flow around a
cylinder, whose solution is closed-form, so every number on the way can be checked.

```bash
python examples/fluidsbench_airfrans/run.py            # 7 model families, two splits, ~1-2 min
python examples/fluidsbench_airfrans/run.py --smoke    # one model, a few seconds
```

## What the run does

| step | what happens | where |
|---|---|---|
| contract | writes a rehearsal `benchmark-specs/` tree and reads it the way a participant reads the real one: status, splits with their SHA-256, the scoring-support release, the composite and its caps, the required predicted fields | `Checkout`, `DatasetContract` |
| support | writes and then verifies the benchmark's own chain — `manifest.json` → `case-sets/standard/index.json` → `chunk-000.json` → one ground-truth table per case and support | `write_support_release`, `load_support_release` |
| model | trains every family of `pinneapple_arena.fieldbench` at the same budget on three angles of attack (−4°, 0°, 8°), the angle of attack entering as a per-node channel | `train_field_model`, `predict_field_model` |
| mapping | the model is queried at the canonical entities themselves, so the mapping is the identity — and it is still declared, with its coverage | `map_to_support` |
| metrics | the published reductions: relative L2 per case with additive sufficient statistics, macro-averaged across cases, then the composite | `score_support`, `aggregate`, `ScoreComposite` |
| package | the schema-v3 result directory (`submission.json`, `evaluation-evidence.json`, `discretization.json` + its case lines, `metrics/cases.json`) with its digest chain | `build_package` |
| validation | every identity recomputed from the package's own numbers: hashes, per-case statistics, macro-average, composite | `validate_package` |
| trust | scope, integrity and applicability as evidence-ledger records, and a decision | `assess_submission` |

## What the numbers mean

Two splits are scored, and they are the point of the benchmark's design:

* **hold-out** — the angle of attack 4°, held out *inside* the training range. The trust decision is `submit`.
* **AoA extrapolation** — 12° and 20°, held out *outside* it. The trust decision is `review`, with the cases and the
  bound they exceed named: the score is still reported, and the fact that the model is extrapolating is reported
  next to it.

The composite caps each error component, so a method can score `0` — see `pointnet_wide` below.

## What it produced on this machine

Hold-out split (composite of a 30 % cap on each error component):

| rank | submission | overall_score |
|---|---|---|
| 1 | transolver-holdout-v1 | 67.53 |
| 2 | edgeconv-holdout-v1 | 21.63 |
| 3 | transolver-lite-holdout-v1 | 9.32 |
| 4 | pointnet-holdout-v1 | 6.93 |
| 5 | crash-solver-holdout-v1 | 5.99 |
| 6 | part-pointnet-holdout-v1 | 3.63 |
| 7 | pointnet-wide-holdout-v1 | 0.00 |

AoA-extrapolation split, same models and budget:

| rank | submission | overall_score |
|---|---|---|
| 1 | transolver-lite-aoa-extrapolation-v1 | 7.63 |
| 2 | transolver-aoa-extrapolation-v1 | 5.91 |
| 3 | crash-solver-aoa-extrapolation-v1 | 4.79 |
| 4 | pointnet / pointnet-wide / edgeconv / part-pointnet | 0.00 |

Two things are worth reading out of those tables. Every score collapses when the split leaves the training range — which
is what an extrapolation split is for — and the family that ranks first inside the range is *not* the one that ranks
first outside it, which is the reason the benchmark scores the two splits separately instead of pooling them.

The reference-oracle row is the pipeline's own test: the reference field scored against itself comes out at relative
L2 = 0 and `overall_score` = 100, which is what says the mapping, the reductions and the composite agree with the ground
truth they were given. It is not ranked. Only 60 epochs per model at batch 2 are spent here: those scores compare
families at equal budget, they are not a claim about what any of them can do, and they move a few points between runs on
the same machine (the last run took 48 s end to end).

Reference values asserted by the run: stagnation pressure `½U² = 0.5 m²/s²`, maximum wall speed `2U = 2 m/s`. Its drag
is zero — d'Alembert — which makes the cylinder a legitimate *reference field* and not a legitimate drag benchmark,
and is why no pressure-integral force metric is scored here.

## What this folder is not

It is not a FluidsBench submission and it never touches the real leaderboard: the support is a rehearsal this example
writes, the contract says `status: rehearsal`, and every package notes that it was written on a synthetic support. To
rehearse against a real release, clone the benchmark and point the adapter at it:

```bash
git clone https://github.com/FluidsBench/fluidsbench-submission
export FLUIDSBENCH_REPO=$PWD/fluidsbench-submission
python -m pinneapple_arena.fluidsbench datasets
python -m pinneapple_arena.fluidsbench dataset airfrans
```

See [`docs/guides/fluidsbench.md`](../../docs/guides/fluidsbench.md) for the API, the metric definitions and the result
package contract.

## Files

| file | what it is |
|---|---|
| `run.py` | the rehearsal; `run(smoke=False)` is its programmatic entry point |
| `experiment.json` | the Lab manifest: what ran, on what budget, and every measured number above |
| `_out/` | generated (`rehearsal.json`, `leaderboard.md`, `support/`, `checkout/`, `submissions/`); git-ignored |
