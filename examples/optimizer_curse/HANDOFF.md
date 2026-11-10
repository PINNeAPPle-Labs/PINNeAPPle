# Optimizer's curse experiment: handoff and reproducibility record

## Persistent locations

- Implementation PR: https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/1599 (draft until validated)
- Physical research umbrella: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1600
- CI, exact-checkout validation and merge: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1618
- Statistical controls and ablations: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1619
- PINN/FNO/DeepONet integrations: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1620
- Independent CFD and engineer approval: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1621

## Implementation committed in the PR branch

- `pinneapple_veriphysics/design_harness.py`: candidate gate, OOD and bounds screening, optional physics callback, optional independent reference, sealed evidence ledger.
- `pinneapple_veriphysics/design_evaluation.py`: metrics on frozen selections and paired bootstrap confidence intervals.
- `pinneapple_veriphysics/design_adapter.py`: model-agnostic surrogate adapter and frozen reference-blind selection. Adapters for actual trained PINN/FNO/DeepONet models still require integration and validation.
- `examples/optimizer_curse/run.py`: manufactured synthetic reference objective and RBF surrogate, independent post-selection adjudication, JSON, plot and report generation.
- `examples/optimizer_curse/README.md`: scope and reproducibility notes.
- `tests/test_design_harness.py`, `tests/test_design_evaluation.py`, `tests/test_design_adapter.py`, `tests/test_optimizer_curse_integration.py`: written tests, not yet confirmed against the exact GitHub checkout.

## Previously observed local experimental result (NOT validated CI)

A locally executed equivalent prototype (not identical to the final committed GitHub code) used five seeds, 512 candidates per seed, 48 training samples and a manufactured analytic reference, not CFD.

- Baseline average promise confirmation: approximately 87.1%.
- Screened average promise confirmation: approximately 90.0%.
- Paired difference: approximately +2.9 percentage points.
- Exploratory 95% paired-bootstrap interval: approximately 0 to +8.6 percentage points, including zero.
- Average best actual gain: baseline approximately 0.1781; screened approximately 0.1495 (worse).
- Blind test nominal 95% coverage: approximately 92.8%.
- 139 post-selection independent reference adjudication calls across the five seeds.
- The screening uncertainty multiplier is heuristic, not calibrated prediction-interval coverage.
- This does not reproduce the reported 43%, 88% or +37% results and does not establish physical superiority.

These numbers are a historical experimental record; rerun the *exact committed code* before publishing any numerical claim. Locally generated per-seed JSON evidence, image and reports were not committed as canonical reproducibility artifacts. The repository script is the source of truth for regenerating them.

## Reproduction / validation steps

From a clean repository checkout with dependencies installed:

```bash
python -m pytest tests/test_design_harness.py tests/test_design_evaluation.py tests/test_design_adapter.py tests/test_optimizer_curse_integration.py -q
python examples/optimizer_curse/run.py --seeds 5 --budget 512 --train 48 --out /tmp/optimizer-curse
```

Inspect `/tmp/optimizer-curse/results.json`, `REPORT.md`, `confirmation.png` and `evidence/seed_*.json`.

## Honest completion status

Implementation and tests are committed to the PR branch. Full-repository tests, CI approval, solver-backed CFD validation, uncertainty calibration, trained PINN/FNO/DeepONet benchmarking, expert signoff and merge are **NOT confirmed complete**. The issues linked above preserve the remaining work.

The goal of this handoff is to make the work resumable from GitHub without relying on a chat session.
