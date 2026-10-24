# 04 Physics-consistent synthetic video and a world model judged by physics: PINNeAPPle + PINNeAPPle World Model

**What.** PINNeAPPle World Model is the CPU alternative to NVIDIA Cosmos here.

- **Synthetic videos:** a thrown ball bouncing on the floor, with event-exact bounces. Camera, size, brightness,
  noise, blur and occluders are randomized. Every frame comes with exact labels (position, velocity, bounce times).
- **Physics checks from the pixels alone:**
  - the object is visible in every frame and its area is conserved;
  - free-flight acceleration equals g, fitted between bounces;
  - no speed beyond what the fall height allows (no teleport);
  - apex heights never grow (no energy gain at a bounce).
- **Test of the checks:** non-physical edits of real videos (time warp, teleport, fade, played backwards after a
  bounce).
- **World model:** a small one (conv encoder, GRU, decoder; 548 k parameters) trained on 200 videos with a
  free-running loss, then judged by the same checks on its reconstructions and its 40-frame rollouts.

**Why.** A world model's video can look right and still break physics. The checks do not care who made the video:
simulator, this model, or NVIDIA Cosmos through `CosmosAdapter`, the same interface on the Cosmos NIM `/v1/infer`.

| Step | Who | Ran here |
|---|---|---|
| synthetic videos with exact labels (200 train, 24 test) | PINNeAPPle World Model | yes |
| video physics checks, edits, verdicts | PINNeAPPle | yes |
| world model, 3 000 steps (926 s) | PINNeAPPle World Model (torch, CPU) | yes |
| the same checks on NVIDIA Cosmos output (`CosmosAdapter`) | Cosmos NIM | no (needs a GPU or `NVIDIA_API_KEY`) |

```bash
python examples/nvidia/04_worldmodel_physics_video/run.py      # ~16 min on 4 CPU cores
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU)

- **Ground truth:** all 24 test videos pass every check, including 4 with an occluder.
  - Tracker error against the exact labels: at most 0.24 px.
  - g from the pixels: within 1.9 % in the worst video.
- **Edits detected:** time warp 100 %, teleport 95 % (1 of 20 missed), fade 100 %, played backwards 100 %.
- **World model** (pixel MAE of the rollouts 0.021):
  - **Reconstructions:** 40 % pass every check. The main failure is the object's area (12 of 20).
  - **Rollouts:** **0 of 20** pass.
    - g fitted from the rollout pixels: median **4.8 m/s2** (1.4 to 17.2).
    - The ball fades and grows in all 20 rollouts and vanishes in 10.
- **Decision: REVIEW.** The checks are sound: ground truth passes and edits are caught. The small world model's
  videos look plausible at a glance (`rollout.png`) but fall at half of g, smear and fade. Example 09 keeps the
  physics by learning the dynamics on the exact state labels instead.

Figures in `results/`: `architecture.png`, `physics_verdicts.png` (pass rates by check for ground truth, edits,
reconstructions and rollouts), `rollout.png` (truth vs rollout frames), `checks.png`.
