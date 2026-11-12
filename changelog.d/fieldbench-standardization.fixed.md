- **Behavior:** `pinneapple_arena.fieldbench.collate_samples` now applies the `standardize` statistics it was given, and
  keeps padded rows at zero. It accepted the argument and ignored it, so `train_field_model` (whose `standardize=True`
  is the default) fitted physical targets while `predict_field_model` un-standardised its output on the way back: every
  field prediction carried the training mean as a constant offset. Measured on the cylinder rehearsal of the FluidsBench
  adapter, the same model and budget went from 39-65 % relative L2 per channel to 0.6-2.1 %.
