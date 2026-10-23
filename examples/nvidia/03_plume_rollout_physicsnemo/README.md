# 03 Plume forecast: PhysicsNeMo FNO rollout, PINNeAPPle trust horizon

**What.** A positive pollutant concentration on a periodic 1 km x 1 km domain, carried by a steady wind and
diffused: `c_t + w . grad c = kappa lap c`. A PhysicsNeMo FNO learns the one-step map `c(t) -> c(t + dt)` and is
rolled out for 60 steps.

**Why.** An autoregressive surrogate is usually judged on its one-step error; what matters is how many steps it can
be chained before it drifts, and whether it keeps the physics a forecaster must keep without any reference (mass,
variance decay, positivity). PhysicsNeMo only: no Omniverse.

| Step | Who | Ran here |
|---|---|---|
| initial plumes, spectral solver exact in time (`operator_benchmarks.advection_diffusion_2d`) | PINNeAPPle | yes |
| FNO training, autoregressive rollout | PhysicsNeMo `FNO` (no padding: periodic) | yes (CPU) |
| `validate_rollout` (trust horizon: error, high-frequency spectrum, invariants), mass, variance, positivity | PINNeAPPle | yes |

```bash
pip install nvidia-physicsnemo
python examples/nvidia/03_plume_rollout_physicsnemo/run.py           # ~3 min on 4 CPU cores
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU, PhysicsNeMo 2.2.2)

- FNO (0.93 M parameters), 150 s of training on 1920 pairs (96 trajectories x 20 steps), 32 x 32.
- One-step relative L2 on 8 held-out plumes: **1.79 %**; the rollout error stays below 3 % for all 60 steps.
- Variance never grows, concentration stays positive.
- **Decision: REVIEW.** Total mass drifts by up to **2.2 %** over 60 steps (limit 1 %), and the trust horizon is
  **10 steps** for every plume: the first criterion crossed is the high-frequency spectrum bias (the FNO keeps more
  small-scale energy than diffusion leaves), not the pointwise error. Use the forecast for 10 steps, or retrain with
  a mass-conserving output layer.

Figures in `results/`: `architecture.png`, `snapshots.png` (reference vs FNO at t = 0, 1, 2, 3),
`rollout_checks.png` (error, mass, variance, spectrum bias), `checks.png`.
