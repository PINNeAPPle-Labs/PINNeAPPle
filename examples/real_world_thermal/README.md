# KIT measured thermal dynamics: how to run

Source: [KIT LLEC, Zenodo 21157950](https://zenodo.org/records/21157950), DOI `10.5281/zenodo.21157950`. License CC BY 4.0. **The observed CSV files are not committed to this repository.** Download `Exp1_tot.csv`, `Exp2_tot.csv`, and `Metadata.json`.

The initial benchmark compares a calibrated Euler RC model (trained using only training observations), a data-only MLP and a physics-regularized PINNeAPPle `vanilla_pinn` network. It is a **one-step observed-lag experiment, not an autonomous simulation**. Its physical regularizer is discrete RC: this is not yet a full strong-form differential PINN. See [tracking issue #1592](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1592).

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
python -m pinneapple_lab run kit_real_thermal \
  -p csv=/data/Exp1_tot.csv \
  -p mapping_json=examples/real_world_thermal/kit_b3_r1_mapping.json \
  -p epochs=400 -p seed=0
python -m pinneapple_lab report --html
pytest -q tests/test_kit_real_thermal.py
```

Outputs: numerical parameter calibration, SHA256 provenance, held-out RMSE/MAE/max absolute error for every candidate, learning curves, test predictions and tracked pass/fail check. Chronological target splits: train 60%, validation 20%, held-out test 20%; only validation data controls early stopping. A failure to outperform the calibrated numerical baseline is recorded, not hidden. The experiment does not silently substitute synthetic data if the real CSVs cannot be read.

Before scientific or commercial claims, finish the full continuous-time PDE/ODE PINN, check multi-campaign train/test separation, free-running rollouts, parameter calibration fairness, repeated seeds, uncertainty and energy-conservation metrics described in #1592.

The KIT B3/R1 column mapping is verified against the [official CSVW metadata](https://zenodo.org/records/21157950/files/Metadata.json?download=1). Verify completeness and physical meaning on the actual measurements before interpreting the resulting metrics.
