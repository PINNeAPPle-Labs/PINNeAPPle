# Physics checks during training (Lightning, PhysicsNeMo, Hugging Face)

Models trained outside PINNeAPPle are usually validated once, at the end, after the whole budget is spent. The
callbacks in `pinneapple_neural.callbacks` run the checks of [`pp.validate`](pp_validate.md) on a fixed validation
batch every N steps and log them to the framework's logger. They warn (or stop the run) when a physics metric worsens
for K consecutive checks while the training loss improves, and write the validation report and trust card at the end.
None of the frameworks is a dependency of PINNeAPPle.

```python
from pinneapple_neural.callbacks import PhysicsMonitor

mon = PhysicsMonitor("darcy_2d", reference=(a_val, u_val), every=50, patience=3, action="warn",
                     report_path="runs/validation.json")        # + runs/validation.trust.json at the end

# PyTorch Lightning: logs to every logger of the Trainer (TensorBoard, W&B, MLflow, CSV)
from pinneapple_neural.callbacks import PhysicsValidationCallback
lightning.Trainer(callbacks=[PhysicsValidationCallback(mon)]).fit(module, loader)

# PhysicsNeMo training loop (PhysicsNeMo has no trainer class): one call per optimiser step
from pinneapple_neural.callbacks import PhysicsNeMoHook
hook = PhysicsNeMoHook(mon)
with LaunchLogger("train", epoch=epoch) as log:                 # physicsnemo.utils.logging
    ...
    if hook(model, step, loss, log=log):                        # True when action="stop" fired
        break
hook.finish(model)

# Hugging Face Trainer: metrics go through trainer.log, i.e. to its report_to integrations
from pinneapple_neural.callbacks import PhysicsTrainerCallback
PhysicsTrainerCallback(PhysicsMonitor(..., model_fn=lambda m: m.net)).attach(trainer)
```

- **Checks.** The monitor accepts any problem `pp.validate` takes: `"darcy_2d"`, a zoo card, a PINN problem
  (`{"pinn": "burgers_1d"}`, [PINN validation](pinn_validation.md)), a rollout, a tabular spec. `level` picks the
  checks. Each check's value is logged as `physics/<check id>`, plus `physics/n_fail` and `physics/n_warn`.
- **`model_fn`.** Maps the framework's module to the network to validate, e.g. when a `LightningModule` or a Hugging
  Face model wraps it.
- **Early warning.** For each watched check (`watch=`, default all numeric ones; a lower |value| is better), it counts
  consecutive checks in which the metric worsened while the loss fell. At `patience` it emits a `PhysicsWarning`
  that lands in `monitor.alerts` and in the trust card's `training_alerts`. With `action="stop"` it also stops
  training.
- **Training-time losses.** `pinneapple_analysis.uncertainty.residual_bound.ResidualLoss` is a differentiable
  physics loss `‖f − A u‖² / ‖f‖²` in the discretisation of the reference solver (sparse, PyTorch). The tests train
  with it.

## Results (from `tests/test_training_callbacks.py`, CPU)

A PhysicsNeMo FNO (2 layers, 6 modes) on 64 Darcy samples at 16 × 16, validated every 40 steps on 8 reference
solutions of `operator_benchmarks.darcy_2d`, 400 steps, loss = relative data MSE + w × `ResidualLoss`:

| Run | Result |
|---|---|
| w = 0.01, trained in Lightning and in a PhysicsNeMo loop | identical physics metrics at all 10 checks (to 1e-5); no alert; relative L2 0.09, energy-norm error bound 0.44 |
| w = 0.01, Hugging Face Trainer | 10 checks logged through `trainer.log`; no alert |
| w = −0.01 (sign error: the residual is maximised) | alert at step 160 of 400 on the energy bound, the flux and the reference error; this is the earliest step `every=40, patience=3` allows. With `action="stop"`, training stops there (also in the HF Trainer). |

Observed but not asserted in the tests: with w = 0 (data loss only), the monitor flags `conservation.flux` at step
200. The boundary-flux imbalance rises from 0.22 to 0.37 while the data loss keeps falling. This is the kind of drift
the callback is for.

## Limits

- A check costs one `pp.validate` call on the validation batch. For Darcy at `level="quick"` with 8 samples at 16 × 16
  it takes about 13 ms; at `standard`, which adds the spectral and symmetry tests, about 28 ms. Choose `every` accordingly.
- The rule needs the loss at each check. In the Hugging Face Trainer it is the latest logged `loss`, so set
  `logging_steps` no larger than `every`.
- Identical metrics across frameworks require identical training: the same initial weights, batch order and
  optimiser. The HF Trainer shuffles and uses its own optimiser, so its run is checked for logging and stopping, not
  for equality.
- `pp.validate` could not run a PhysicsNeMo FNO given a field without a channel axis: PhysicsNeMo raises
  `ValueError`, not `RuntimeError`. This is fixed here.
