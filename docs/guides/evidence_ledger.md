# Evidence ledger: a test passed only if it ran

A report that says "all checks passed" is only as good as the record behind it. `pinneapple_veriphysics.ledger`
keeps that record. Each entry holds:

- the check that ran;
- the SHA-256 of its exact inputs;
- its outputs;
- start and finish times (UTC);
- the verdict.

The entry is sealed with the SHA-256 of its own content. `EvidenceLedger.status(check)` returns `"passed"` or
`"failed"` only for a complete entry that is still intact. Every other case returns `"not run"`: no entry, a
missing input hash, outputs or time, a crash, or an entry edited after it was sealed.

```python
from pinneapple_veriphysics.ledger import EvidenceLedger, sha256_bytes

ledger = EvidenceLedger()
with ledger.run("energy-balance", inputs_sha256=sha256_bytes(b"field bytes"), tool="my-check") as rec:
    rec.outputs = {"imbalance": 2e-4}
    rec.passed = 2e-4 < 1e-3
assert ledger.status("energy-balance") == "passed"
assert ledger.status("mass-balance") == "not run"
```

## Validation Lab: three groups, one record per check

`pinneapple_veriphysics.validation_lab.run_validation(folder)` runs [`pinneapple check`](physics_checks_in_ci.md)
on the inputs of a folder. It reports the outcomes in three separate groups:

| Group | Question | Rules |
|---|---|---|
| Numerical verification | Were the equations solved right? | PC3xx residuals, PC403 finite values, PC8xx artefact integrity, PC901 method status, PC001 readable inputs |
| Validation against data | Does it match trusted reference data, inside the range the data covers? | PC7xx reference regression and QoIs, PC6xx out-of-range use, PC452/PC453 dataset diversity and leakage |
| Physical consistency | Conservation, boundary and initial conditions, constraints, units | PC1xx, PC2xx, PC401/PC402 bounds, PC5xx units, PC451 dataset physics |

The report lists, per target, the groups in which no check ran. Each outcome becomes a ledger entry. The inputs
hash of an entry combines the target's files, its reference and the configuration. The report also holds the
manifest of every input file with its hash, and the configuration itself, so
`pinneapple check --config pinneapple-check.toml` reproduces the run.

`config_from_spec` writes that configuration from a plain description of predictions, physics, units, bounds,
boundary conditions, reference and tolerances. `assert_no_code` refuses models that would import or unpickle
uploaded code; zoo models are allowed.

No single test guarantees that a model represents reality, and neither does this set of tests. The report says so.

## Evidence report and package

`pinneapple_veriphysics.evidence_package` reads other tools' outputs and builds a report:

- Validation Lab reports;
- ledgers;
- `pinneapple check --json`;
- Simulation Comparator results;
- StudyResult / ModelRef / DatasetSpec JSON;
- metrics tables;
- any other file, by hash.

The report has a methodology, conditions, a baseline comparison, tests, limitations and references to every
artifact by SHA-256. It renders as HTML or PDF (with `pinneapple_tools.reporting.paperkit`), and
`build_package` writes a zip whose `MANIFEST.json` holds the hash of every file. `verify_package` checks the
zip against that manifest.

Records are matched per source. A test from a source that only *claims* a pass (a `pinneapple check` JSON, a
StudyResult with `tests` but no `evidence`) is printed as "not run", with the claim beside it. The same holds
when another run used the same check id. The tests in `tests/pinneapple_veriphysics/test_evidence_ledger.py`
enforce this.

The [Engineering Evidence Report](../apps/evidence_report.md) (day 30) is built on these modules.
The [Physics Validation Lab](../apps/validation_lab.md) (day 24) is built on these modules.
