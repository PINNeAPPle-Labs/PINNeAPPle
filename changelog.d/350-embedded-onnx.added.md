- **`pinneapple_physics.embedded`: the model, its physics residual and its sensor check as one ONNX graph** (#350):
  `EmbeddedSpec` fixes the contract (state shape, equation, parameters, `dt`/`dx`, the residual and measurement
  tolerances) and refuses an impossible one at construction; `export` writes `(state, measurement) -> (prediction,
  residual, trust, off_physics, suspect, measurement_error)` as a straight-line graph (no `If`/`Loop`/`Scan`), with the
  stencil of `advection-diffusion`, `advection` or `heat` written as convolutions; `graph_summary` reports the op types
  and whether anything can branch; `EmbeddedRuntime.step`/`.latency` call it and measure it. Measured on an ARM core,
  one thread: **1.5-1.8 kHz** (0.55-0.66 ms median across runs, p95 0.64-0.78 ms) on a 25 kB graph for a 12 k-parameter field model; a tolerance
  calibrated on a held-out in-domain sample (0.094) flags a field moving 2.5x faster than the training data 100 % of
  the time and never flags the training distribution. The example and the guide record the two mistakes the
  measurements caught: a residual measured against the continuous solution instead of the discrete one scores the
  *exact* frame at 1.70 (persistence beats the truth), and a `dt` small enough that the model's error exceeds the
  per-step change makes every prediction fail.
