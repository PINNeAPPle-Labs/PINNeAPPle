# 09 A world model that keeps the physics: PINNeAPPle + PINNeAPPle World Model + PhysicsNeMo

**What.** The same synthetic bouncing-ball videos as example 04 (exact ballistics, event-exact bounces, randomized
camera and appearance), used for their exact labels. A small CNN (`StateEncoder`) reads the ball's position from each
frame; `physicsnemo.models.mlp.FullyConnected` learns the one-step state map `(x, y, vx, vy) -> next` (flight and
bounces) from 9 400 labelled transitions; the rollout is rendered with each test video's own camera and appearance.

**Why.** Example 04's pixel-space world model fails every physics check. Here the latent is a physical state, and the
same video physics checks (they read only the pixels) judge the result.

| Step | Who | Ran here |
|---|---|---|
| 200 training and 20 test videos with exact labels | PINNeAPPle World Model | yes |
| StateEncoder CNN, 1 500 steps (440 s) | PINNeAPPle World Model | yes |
| FullyConnected 4 x 128, 20 000 Adam steps (1 684 s) | PhysicsNeMo 2.2.2 (CPU) | yes |
| render the rollouts, video physics checks, decision | PINNeAPPle | yes |
| NVIDIA Cosmos (same interface, `CosmosAdapter`) | Cosmos NIM | no (needs a GPU or `NVIDIA_API_KEY`) |

```bash
pip install nvidia-physicsnemo
python examples/nvidia/09_worldmodel_physicsnemo_state/run.py     # ~36 min on 4 CPU cores (CPU shared with example 04)
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU)

- Rendered 40-step rollouts from 8 context frames: **18 of 20 pass every video physics check** (example 04's pixel
  model: 0 of 20). Failures: 2 on fitted g, 1 on visibility, 1 on area.
- g fitted from the rollout pixels: median **9.86 m/s2** (0.5 % from 9.81); 18 of 20 within 9.6 to 10.0, two outliers
  at 7.1 and 8.1.
- One-step error (relative RMS): x 8e-6, y 0.35 %, vy **11 %** (the bounces: a jump in vy the network smooths). vx is
  constant in the data, so its update is learned exactly.
- The rollouts lose more energy at each bounce than the truth (lower apexes in `rollouts.png`): plausible physics,
  not the right restitution.
- State encoder: worst error on a test frame **15.8 px** (limit 1.5 px).
- **Decision: REVIEW.** The rollouts are physically plausible, but the encoder's worst case and the bounce error fail
  their limits.

Figures in `results/`: `architecture.png`, `rollouts.png` (state rollouts vs exact, fitted g), `rollout_frames.png`,
`checks.png`.
