# External falsification and reproduction: cold plate / Darcy

This is an **additive independent-evaluation package**, not a new model family.
It uses the existing PINNeAPPle finite-volume solver, `darcy_checks` physics
checks, applicability envelope, and the existing cold-plate FNO training script.

## Run

From the repository root, with `pip install -e .`:

```bash
python examples/fno_coldplate/external_validation.py --out artifacts/external_validation/coldplate
python examples/fno_coldplate/external_validation.py --fno-smoke --out artifacts/external_validation/coldplate
pytest tests/test_external_validation_coldplate.py -q
```

The first command uses **no training**, and creates `controls.json` and
nine compressed arrays, with SHA-256 checksums. The second additionally runs
the pre-existing FNO **six-step smoke test** (not an accuracy benchmark).

## Pre-registered controls

All use the PINNeAPPle cell-centered finite-volume `-div(a grad u) = f`
solver on a 32 × 32 grid (three source levels: 0.5, 1, 2).

| Control | Construction | Required observation |
| --- | --- | --- |
| Clean finite-volume reference | Exact discrete solve, coefficient `a = 6` | PASS |
| Known broken prediction | Multiply reference solution by 0.5 | FAIL on physics |
| Out-of-envelope reference | Exact discrete solve, `a = 24`, vs training coefficients 3, 6, 12 | FAIL on applicability |
| Same OOD field, physics only | Recheck with no fitted envelope | PASS |

The third/fourth pair distinguishes domain-of-applicability rejection from
incorrect physics. These are artificial controls, **not evidence of FNO
accuracy**, and the training envelope is intentionally narrow.

## Evidence rules

- Do not call an internally produced benchmark an independent reproduction.
- Do not equate a low prediction error with physical acceptance. The FNO
  cold-plate case in `README.md` is currently **REJECT** under physics checks.
- Do not compare FNO inference cost to solver wall time without reporting
  preprocessing, training cost, grid size, hardware, repetitions and uncertainty.
- Treat `fno_smoke.json` as smoke evidence only. A production benchmark
  requires repeated independent held-out seeds and an unbiased baseline.
- An external evaluator should execute in a fresh environment and publish a
  run URL, commit SHA, hardware, report and any failure modes.

## Next experiment

The full FNO experiment is already at `examples/fno_coldplate/run.py`.
To establish scientific generalization, add 5 repeated seeds, a declared
budget, OOD coefficient controls, bootstrapped uncertainty and strong
non-ML baselines. Keep this separate from the quick falsification controls.

## Five-seed learned-model study

The existing `run.py` now accepts `--train-seed`, `--test-seed` and
`--model-seed`. To train and evaluate five independent runs with distinct
training/test GRF seeds and model initializations:

```bash
python examples/fno_coldplate/multiseed_study.py --seeds 0 1 2 3 4
```

Each run writes its original PINNeAPPle `results/summary.json`, containing
in-distribution model accuracy, trust decision, measured training/inference
fields and an **out-of-distribution** evaluation where conductivity is doubled.
The final `study.json` records median, minimum and maximum, not an
unsupported confidence interval. Full runs are not included in the quick CI.

To check integration without making accuracy claims:

```bash
python examples/fno_coldplate/multiseed_study.py --smoke --seeds 0 1
```

**Interpretation:** The FNO's learned predictions, even in this multi-seed
study, remain *author-run* measurements until an independent team reproduces
them. An OOD check rejecting a correct field is a safeguard, not proof that a
learned model extrapolates well. Data-generation and training costs must
be reported before making end-to-end speedup claims.
