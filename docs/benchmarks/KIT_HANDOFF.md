# KIT real thermal benchmark: handoff and continuation

Updated: 2026-11-09. **This is a work-in-progress, not a validated result.**

## Durable GitHub locations

- Repository: https://github.com/PINNeAPPle-Labs/PINNeAPPle
- Draft PR: https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/1598
- Working branch: `experiment/kit-real-thermal-benchmark`
- Main tracking issue: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1592
- Follow-up execution/validation issues:
  - #1622 CI, unit tests, physics/security checks
  - #1623 verified KIT download and real-data four-model run
  - #1624 cross-campaign generalization, >=5 seeds, fairness
  - #1625 rollouts, physical conservation, final scientific report
  - #1626 review and merge PR

## Code implemented on branch

- `pinneapple_lab/experiments/real_world_thermal.py`: registered KIT experiment, real CSV parser, chronological split, RC numerical calibration, data-only and discrete RC-regularized PINNeAPPle models, optional continuous ODE PINN, plots and metrics.
- `pinneapple_lab/experiments/kit_continuous_thermal.py`: autograd residual for a lumped thermal ODE, positive heat capacity/conductance, training and optional fitted model return.
- `pinneapple_lab/experiments/kit_scientific_metrics.py`: repeated-seed aggregation, empirical across-seed quantiles, finite-difference energy-balance residual.
- `examples/real_world_thermal/download_kit.py`: public Zenodo dataset download with checksum verification.
- `examples/real_world_thermal/kit_b3_r1_mapping.json`: B3 room 1 sensor mapping.
- `examples/real_world_thermal/run_cross_campaign.py`: Exp1-to-Exp2 cross-campaign evaluation, five seeds by default, RC autonomous rollout, numerical and neural inference, CSV/JSON/plot output.
- `examples/real_world_thermal/README.md`: commands, assumptions and interpretation warnings.
- `docs/benchmarks/kit_real_thermal.md`: scientific protocol.
- Tests: `tests/test_kit_real_thermal.py`, `tests/test_kit_csvw_hours.py`, `tests/test_kit_continuous_thermal.py`, `tests/test_kit_scientific_metrics.py`.
- Experiment registration: `pinneapple_lab/experiments/__init__.py`.

## Resume commands

From repository root with project dependencies installed:

```bash
python examples/real_world_thermal/download_kit.py --output data/kit_thermal
pytest -q tests/test_kit_real_thermal.py tests/test_kit_csvw_hours.py tests/test_kit_continuous_thermal.py tests/test_kit_scientific_metrics.py
python -m pinneapple_lab run kit_real_thermal \
  -p csv=data/kit_thermal/Exp1_tot.csv \
  -p mapping_json=examples/real_world_thermal/kit_b3_r1_mapping.json \
  -p epochs=400 -p seed=0
python examples/real_world_thermal/run_cross_campaign.py \
  --train data/kit_thermal/Exp1_tot.csv \
  --test data/kit_thermal/Exp2_tot.csv \
  --mapping examples/real_world_thermal/kit_b3_r1_mapping.json \
  --seeds 0,1,2,3,4 --epochs 400 \
  --output reports/kit_exp1_to_exp2
```

## Scientific caveats and known work

1. **No successful real-data execution is confirmed.** GitHub Actions had been queued; check latest workflow runs and fix any errors. No measured accuracy or superiority claims have been established.
2. Public Zenodo measurements were not successfully downloaded in the implementation environment. Verify the downloader's published checksums and actual file structure before interpreting output.
3. One-step RC/MLP/RC-regularized models consume the previous *observed* room temperature; RC autonomous rollout does not. These are distinct forecasting tasks and must not be ranked as equivalent.
4. Continuous ODE PINN is a lumped thermal model, **not a spatial PDE PINN**. Cross-campaign transfer currently resets elapsed time without a physical initial condition; this is not an autonomous initialized forecast.
5. The continuous PINN uses a learned time-coordinate mapping with observed exogenous signals. Check forcing discontinuities, physical residual consistency, parameter identifiability, normalization, and training stability.
6. The RC energy balance currently assumes heating efficiency=1 to infer a capacity; this is a conditional assumption, not independent identification of capacity and efficiency.
7. Across-seed quantile bands are **not calibrated predictive intervals**. Check empirical coverage.
8. Do not merge until tests, physical validation, data provenance, cross-campaign fairness, and report are reviewed. Track remaining work in #1622–#1626.
9. PR title/body initially described docs-only work; update them to reflect implemented code during final review (#1626).

## Definition of done

Real measurements acquired with provenance; four models trained and scored; independent-campaign and repeated-seed comparisons; clear distinction between one-step, elapsed-time transfer, and autonomous rollouts; physical consistency reviewed; CI green; reproducible report; PR reviewed and merged.

**All implementation is on the GitHub branch, not on main until merge.** No unpublished chat-only implementation is required to resume; use the PR, files and issues above.
