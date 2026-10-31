- `pinneapple_neural.trainer`: `DataModule` batches of `x` / `y` items no longer lose their keys (the PINN collate
  is used only for PINN batches). `PreprocessPipeline.fit` fits each step on the output of the steps before it, so a
  normalizer after a NaN filler no longer computes NaN statistics. Forward fill also fills NaNs before the first
  value. `inference.load_checkpoint` reads the `Trainer`'s own checkpoints (`model_state`) as well as
  `ModelCheckpoint`'s (`model`). `pinneapple_tools.visualization.molecular` uses `matplotlib.colormaps`
  (`cm.get_cmap` was removed in Matplotlib 3.9).
- Examples that did not run: `examples/trainer` imports `pinneapple_neural.trainer` (not the `pinneapple_train`
  shim, which never exported `DataModule`), and example 02 takes its target from the clean signal;
  `examples/pinneapple_cosim` imports `pinneapple_systems.cosimulation` (the `pinneapple_cosim` module does not
  exist); `examples/visualizations/viz_02_heat_2d.py` uses `plt.colormaps`.
