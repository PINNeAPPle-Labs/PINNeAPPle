# Data readiness: may this dataset start a training run?

Between "we have engineering data" and "a model trains on it" there used to be nothing: `Trainer` trained on whatever
arrays it was given, and the checks that existed (`data_health`, `cae.quality`, `preflight`, `pinneapple_veriphysics`)
did not gate anything. `pinneapple_data.readiness` is that stage, and `pinneapple_data.review` is the record of a
person accepting what the automated checks could not decide (#1491, #1493, part of the PDI program, #1497).

```python
import pandas as pd, pinneapple as pp

report = pp.readiness("examples/data_readiness/manifest.json", table=pd.read_csv("examples/data_readiness/sweep.csv"))
print(report.status, report.score["overall"], report.digest)
pp.require_readiness(report)            # raises ReadinessError when the dataset may not train
```

## The verdict

| status | when | what it does to a run |
|---|---|---|
| `PASS` | no finding | trains, and carries the verdict into the run |
| `WARNING` | findings, none critical | trains **only** if a review accepts exactly those findings for this digest |
| `FAIL` | at least one critical finding | never trains |
| `NOT_ASSESSED` | nothing was assessed | never trains — a dataset with no verdict is not a dataset with a good verdict |

**The verdict is not the score.** `status` comes from the severities; the score is reported next to it with its
formula in the report. Two datasets can both score 70 and be `FAIL` and `WARNING` respectively —
`tests/test_data_readiness.py::test_the_verdict_is_not_derived_from_the_score` asserts exactly that, because a number
used *as* a verdict is how a critical units mismatch becomes "87 %, ship it".

## The checks, and where they come from

| group | what runs |
|---|---|
| `schema` | the manifest: required fields, roles, duplicate columns, files that exist |
| `units` | `physical_units`: declared units parse, and the declared unit agrees with the one in the table header |
| `conditions` | every output field has a boundary or initial condition behind it |
| `validity` | the operating point inside the range the data is declared to cover |
| `table` | `data_health.analyze` — units, completeness, time axis, validity, stuck sensors, physics consistency |
| `case` | `preflight` — the case setup findings, when a case is attached |

Each finding carries an `id`, a `group`, a `severity`, the evidence and a **fix**. `ReadinessError` prints all three
for every finding that stopped the run.

## The manifest

```json
{"schema": "pinneapple.readiness.dataset/1",
 "name": "plate_sweep", "version": "1.0", "temporal": false,
 "columns": [{"name": "Re", "role": "input", "unit": "1"},
             {"name": "T_max", "role": "output", "unit": "K"}],
 "boundary_conditions": [{"field": "T_max", "kind": "dirichlet", "value": 300.0}],
 "validity": {"Re": [10000.0, 100000.0]},
 "operating_point": {"Re": 50000.0}}
```

`"temporal": true` keeps the time-axis checks of `data_health`; a parameter sweep is not a time series and the report
does not pretend otherwise. A policy (`ReadinessPolicy`) can promote a check to critical, demote it to a warning, or
disable a whole group — for the problem at hand, not for everyone.

## The review record

Approval is a **record**, never a flag:

```python
store = pp.ReviewStore("runs/reviews.jsonl")
record = pp.ReviewRecord(subject=report.digest, decision="accept_exception", reviewer="A. Ribeiro", role="thermal",
                         scope=[{"finding": "units.missing", "reason": "unit stated in the case sheet"}])
store.save(record)

trainer = Trainer(model, loss_fn, readiness=report, review=store)
run = trainer.fit(loader, cfg)      # run["readiness"] carries the verdict, the digest and who accepted what
```

* a record needs a **reviewer** and a **decision** — nothing about the metrics can produce them;
* `accept_exception` has to name the **finding ids** it overrides, each with a reason;
* the record is bound to a **digest** of the manifest *and the data*, so approving one version does not approve the
  next one (change one value and the approval stops applying);
* the store is **append-only and chained**: each record carries the digest of the previous one, so
  `store.verify()` reports an edited or removed record instead of trusting it. A new decision supersedes an old one and
  both stay readable.

## What it does not do

It does not decide whether the physics is right (`pinneapple_veriphysics`), and it is not model release approval
(#1267). It answers one question: *may this data start a training run* — and when the answer is "only if someone says
so", it records who said so, about which evidence.
