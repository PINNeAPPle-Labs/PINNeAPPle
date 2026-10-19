# Engineering Evidence Report

**Turn experiments and simulations into reproducible technical evidence. A test is passed only if it ran.**

Day 30 of the PINNeAPPle app series (evidence.pinneapple.org, port 8113).

**The pain.** At the end of a study, the evidence is scattered: solver logs, a CI badge, a spreadsheet of
metrics, a model file, a validation report. The write-up says "all checks passed". Nobody can tell which checks
ran, on which inputs, or when. A month later, nobody can reproduce it either.

| You give it | You get |
|---|---|
| The study's outputs, as files from any app of the series or from your own tools: Validation Lab reports (with their evidence ledger), evidence ledgers, `pinneapple check --json`, Simulation Comparator results, StudyResult / ModelRef / DatasetSpec JSON, metrics tables (CSV, Parquet, JSON rows), and any other file (meshes, cases, weights). Add a title, purpose, methodology, conditions, the baseline and candidate run names, limitations, references, and where the originals live (URLs). | A **report** (HTML preview, PDF) with: methodology; conditions, including those read from the configurations and parameters inside the sources; a baseline comparison (every metric reported for both runs, with the change and a figure); tests and evidence; limitations; and references to every original artifact by SHA-256 and URL. An **evidence package** (zip): the report (JSON, HTML, PDF), the merged evidence ledger, the figures, the original files unchanged, and a `MANIFEST.json` with the SHA-256 of every file. |

## The central rule, enforced in code

A test is reported as passed or failed **only when an evidence record shows that it ran**. The record holds the
SHA-256 of the inputs, the outputs, and the start and finish time, and it is sealed with its own hash. Everything
else is printed as **not run**, with the source's claim shown next to it:

- a source that only *says* "passed" (a `pinneapple check` JSON, a StudyResult `tests` list without `evidence`, a
  badge);
- a record edited after it was sealed;
- a record from another run that uses the same check id.

The rule lives in `pinneapple_veriphysics.evidence_package` and `pinneapple_veriphysics.ledger`. It is tested in
`tests/pinneapple_veriphysics/test_evidence_ledger.py` (report, HTML and PDF) and, through the HTTP API, in
`tests/test_app_evidence_report.py`.

## Integrations

- **Physics Validation Lab** (day 24): its report carries one record per check, so its passes count here.
- **Simulation Comparator** (day 10): global and per-field errors become metrics; per-field verdicts become claims.
  They count only with a record.
- **`pinneapple check` in CI:** its JSON is read, but it has no records, so its checks print "not run". Run the checks
  through the Validation Lab, or record them with `pinneapple_veriphysics.ledger`, to make them count.
- **Shared data contracts** (`pinneapple_data.contracts`): StudyResult, ModelRef, DatasetSpec and ParameterSpace
  JSON are recognised by their `kind` and validated against the contract schema (problems are listed next to the
  artifact; the file is still read). A StudyResult's `checks` are claims: `PASS` without a ledger record prints "not
  run". Its `evidence` (a ledger) backs them, its `parameters` become conditions, its `metrics` comparable runs and
  its `limits` limitations. Every report is also exported as a StudyResult (study type `evidence`, one check per
  test, artifacts by SHA-256): `study_result` in `/api/report`, `study_result.json` in the package.
- **Your pipeline:** the REST API (`/docs`) offers `POST /api/report`, `/api/report.pdf` and `/api/package`. In
  Python, use `pinneapple_veriphysics.evidence_package` and record tests where they run with `EvidenceLedger.run`.

## Methods

The library does the work:

- `load_source` reads and classifies each file;
- `build_report` merges the evidence per source and builds the report;
- `make_figures` draws the figures with the paperkit style;
- `render_html` and `render_pdf` render the report (`pinneapple_tools.reporting.paperkit.Paper`);
- `build_package` and `verify_package` write and check the zip.

The app handles uploads, routes and the UI. Numbers are copied from the sources; nothing is estimated or recomputed.

## Try with example (about 5 s, real runs)

The example runs a real study on the server at the first request:

- `pp.solve` solves the same Poisson problem with linear finite elements (24×24 cells, the baseline) and with a PINN
  (200 epochs, 2×16, seed 0, the candidate);
- `pp.compare` scores both against the exact solution on 4,096 points;
- a trust card per method is recorded in an evidence ledger;
- a temperature field goes through the Validation Lab.

Results on this run: FEM relative L2 7.5e-3 and trust card APPROVED. The PINN has relative L2 5.6e-2 and fails its
boundary-condition and reference checks, because 200 epochs is too small a budget. The StudyResult also claims a
grid-convergence study that never ran, planted to show the rule: it is printed "not run". The example's source files
can be downloaded and uploaded again to rebuild the report.

## Run

```bash
pip install -e . -r apps/evidence_report/requirements.txt              # from the repository root
cd apps/evidence_report && uvicorn evidence_report.api:app --port 8113
```

Environment variables:

- `EVR_USER` / `EVR_PASSWORD`: HTTP Basic login on everything except `/health`;
- `EVR_MAX_MB`: largest upload, default 200;
- `EVR_MAX_HEAVY`: reports built at once, default 2; the excess gets HTTP 429;
- `EVR_FREE_REPORTS` / `EVR_FREE_DOWNLOADS`: reports and downloads (PDF or package) per caller and month before
  HTTP 429, default 30 each (`unlimited` removes the limit); `EVR_ADMIN_TOKEN` manages API keys (`/api/keys`).
  API keys, usage (`/api/usage`) and the runtime description (`/api/runtime`) come from the shared app runtime
  (`apps/_shared/README.md`); the example is not metered.

Docker: `docker build -f apps/evidence_report/Dockerfile -t evidence-report .` (one worker, 1 GB). Deployment with
the other apps is in [`apps/deploy`](../deploy) (service `evidence`).

## Limits

- **The report checks records; it does not re-run tests.** It verifies that a record exists, is complete and is
  intact. It cannot know whether the hashed inputs were the right ones. Records are not signed yet
  (`pinneapple_security.signing` is the planned path).
- **Baseline matching is by name.** A run belongs to the baseline when its label is or contains the baseline name,
  and does not also contain the candidate name. Metrics are compared as given, and units are not converted.
- **The campaign report code of PINNeAPPle-CFD is not in the library.** Only `pinneapple_tools.reporting.paperkit` is
  reused.
- **Nothing is stored.** Keep the evidence package. Private projects with stored packages are the paid tier.
- **Reports are built inside the request** (seconds), not in the shared job queue; usage is metered per caller
  and month by the shared runtime.

Monetization: reports (a free monthly allowance, then per report or package), campaigns (one package per run or
benchmark of a campaign, with a campaign summary), and private projects (stored packages, signed records, team
access).
