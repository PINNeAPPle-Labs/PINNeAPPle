# KIT measured thermal dynamics: how to run

Source: [KIT LLEC, Zenodo 21157950](https://zenodo.org/records/21157950), DOI `10.5281/zenodo.21157950`. License CC BY 4.0. **The observed CSV files are not committed to this repository.** Download `Exp1_tot.csv`, `Exp2_tot.csv`, and `Metadata.json`.

The benchmark compares a calibrated Euler RC model, a data-only MLP, a discrete RC-regularized PINNeAPPle network, and an optional **continuous-time lumped ODE PINN** with an autograd residual. The first three methods are **one-step observed-lag predictions**, whereas the continuous PINN learns temperature as a function of time and external forcings. Neither method is a spatial PDE solver or an autonomous free-running forecast. See issue #1592 for additional scientific validation. See [tracking issue #1592](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1592).

The official `Metadata.json` defines `time` in decimal hours and identifies a consistent B3 room-1 mapping already committed at `kit_b3_r1_mapping.json`. Use that file for the initial B3/R1 benchmark; other rooms require corresponding mapping and sensor-quality checks. The explicit schema is:

```json
{
  "time": "time",
  "temperature": "b3_r1_tzone___c_",
  "outdoor": "tamb___c_",
  "heat_w": "b3_r1_qth__w_"
}
```

Do not accidentally use **electrical** heating power as **delivered thermal** power. The room temperature and delivered thermal power must refer to a consistent zone/system. Data with irregular timestamps or missing samples is rejected rather than interpolated without documentation.

Run after installing the PINNeAPPle repository (including `torch`, `numpy`, `scipy`, `pandas`):

```bash
python examples/real_world_thermal/download_kit.py --output data/kit_thermal
python -m pinneapple_lab run kit_real_thermal \
  -p csv=data/kit_thermal/Exp1_tot.csv \
  -p mapping_json=examples/real_world_thermal/kit_b3_r1_mapping.json \
  -p epochs=400 -p seed=0
python -m pinneapple_lab report --html
pytest -q tests/test_kit_real_thermal.py tests/test_kit_csvw_hours.py tests/test_kit_continuous_thermal.py
```

Outputs: numerical parameter calibration, SHA256 provenance, held-out RMSE/MAE/max absolute error for every candidate, learning curves, test predictions and tracked pass/fail check. Chronological target splits: train 60%, validation 20%, held-out test 20%; only validation data controls early stopping. A failure to outperform the calibrated numerical baseline is recorded, not hidden. The experiment does not silently substitute synthetic data if the real CSVs cannot be read.

Before scientific or commercial claims, validate the new continuous-time ODE PINN on the actual data, check multi-campaign train/test separation, free-running rollouts, parameter calibration fairness, repeated seeds, uncertainty and energy-conservation metrics described in #1592.

The KIT B3/R1 column mapping is verified against the [official CSVW metadata](https://zenodo.org/records/21157950/files/Metadata.json?download=1). Verify completeness and physical meaning on the actual measurements before interpreting the resulting metrics.

## Independent-campaign, multi-seed evaluation

The cross-campaign runner fits on the first 80% of the training campaign,
selects neural checkpoints using the remaining 20%, and scores all candidates
on a separate held-out campaign. It produces `metrics.csv`,
`predictions.csv`, `seed_summary.json`, `ensemble_intervals.csv`,
`rc_energy_balance.json`, `provenance.json` and a comparison plot.

```bash
python examples/real_world_thermal/run_cross_campaign.py \
  --train data/kit_thermal/Exp1_tot.csv \
  --test data/kit_thermal/Exp2_tot.csv \
  --mapping examples/real_world_thermal/kit_b3_r1_mapping.json \
  --seeds 0,1,2,3,4 --epochs 400 \
  --output reports/kit_exp1_to_exp2
```

**Interpretation constraints:** one-step networks and the RC one-step baseline
use observed lagged temperatures. The RC autonomous rollout does not.
The continuous ODE PINN is evaluated on a reset elapsed-time coordinate, not
an initialized physical rollout. Ensemble intervals are across-seed quantiles,
**not calibrated predictive confidence intervals**. RC energy-balance parameters
assume heating efficiency equals 1; capacity and efficiency cannot generally
be identified independently from these measurements alone. Check campaign
sensor compatibility before interpreting any metric.

Execution and validation follow-ups: #1622 (CI), #1623 (real data),
#1624 (five-seed cross-campaign fairness), #1625 (physical checks and report),
#1626 (review/merge). None of these are considered complete by the existence
of this code alone.
