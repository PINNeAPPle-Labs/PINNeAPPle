# KIT measured thermal dynamics: how to run

Source: [KIT LLEC, Zenodo 21157950](https://zenodo.org/records/21157950), DOI `10.5281/zenodo.21157950`. License CC BY 4.0. **The observed CSV files are not committed to this repository.** Download `Exp1_tot.csv`, `Exp2_tot.csv`, and `Metadata.json`.

The initial benchmark compares a calibrated Euler RC model (trained using only training observations), a data-only MLP and a physics-regularized PINNeAPPle `vanilla_pinn` network. It is a **one-step observed-lag experiment, not an autonomous simulation**. Its physical regularizer is discrete RC: this is not yet a full strong-form differential PINN. See [tracking issue #1592](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1592).

Read `Metadata.json` and create a **real** four-column mapping (placeholders are not column names):

```json
{
  "time": "ACTUAL_TIMESTAMP_COLUMN",
  "temperature": "ACTUAL_B3_MATCHED_ROOM_TEMPERATURE_COLUMN",
  "outdoor": "ACTUAL_OUTDOOR_AIR_TEMPERATURE_COLUMN",
  "heat_w": "ACTUAL_DELIVERED_THERMAL_HEAT_W_COLUMN"
}
```

Do not accidentally use **electrical** heating power as **delivered thermal** power. The room temperature and delivered thermal power must refer to a consistent zone/system. Data with irregular timestamps or missing samples is rejected rather than interpolated without documentation.

Run after installing the PINNeAPPle repository (including `torch`, `numpy`, `scipy`, `pandas`):

```bash
python -m pinneapple_lab run kit_real_thermal \
  -p csv=/data/Exp1_tot.csv \
  -p mapping_json=/data/kit_columns.json \
  -p epochs=400 -p seed=0
python -m pinneapple_lab report --html
pytest -q tests/test_kit_real_thermal.py
```

Outputs: numerical parameter calibration, SHA256 provenance, held-out RMSE/MAE/max absolute error for every candidate, learning curves, test predictions and tracked pass/fail check. Chronological target splits: train 60%, validation 20%, held-out test 20%; only validation data controls early stopping. A failure to outperform the calibrated numerical baseline is recorded, not hidden. The experiment does not silently substitute synthetic data if the real CSVs cannot be read.

Before scientific or commercial claims, finish the full continuous-time PDE/ODE PINN, check multi-campaign train/test separation, free-running rollouts, parameter calibration fairness, repeated seeds, uncertainty and energy-conservation metrics described in #1592.
