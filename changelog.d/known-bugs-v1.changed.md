- A bare number answering a design question (`answers={"t_limit": 70}`) is read in the unit the question shows
  (°C here), not in the slot's SI unit; `"70 degC"` and `(343.15, "K")` are also accepted (#953). Code that passed
  kelvin as a bare number must now pass `(value, "K")`.
- `Trainer.fit(train_loader, val_loader=None, cfg=None)`: the validation loader is optional. Without it the history
  reports `val_total` as `None`, and the training loss drives checkpoint selection and the callbacks that monitor
  `val_total`. The result has a new `validated` key, and `fit` is documented (#958).
