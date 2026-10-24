# 17 Robot perception from synthetic camera data: PINNeAPPle + Omniverse (OpenUSD) + Robot Sim + World Model

**What.** The two-link arm of examples 03 and 12 makes 180 random point-to-point moves in PINNeAPPle Robot Sim on
NVIDIA Newton (PD + model feedforward).

- **USD timeline per move:** animated links, a side `UsdGeom.Camera`, sometimes an occluder in front, and the joint
  angles time-sampled as `pinneapple:label:q`.
- **Synthetic camera:** rasterizes the stage at every frame with a random zoom, framing, gain, noise, blur, albedo
  and background.
- **Label check:** keeps a frame only if the elbow and the tool point are visible.
- **CNN:** reads both joint angles from a frame.
- **PINNeAPPle checks:** go beyond the angle error. They measure the tool point through the forward kinematics, the
  joint speed implied by the predicted sequence against Newton's, and frame-to-frame jumps the motors cannot make.

| Step | Who | Ran here |
|---|---|---|
| 160 training + 20 test moves, exact `q`, `qd`, link poses | Robot Sim (Newton 1.7) | yes |
| USD timelines (links, occluder, camera, labels), validated | OpenUSD (`usd-core` 26.11) | yes |
| synthetic camera, visibility check, CNN | PINNeAPPle World Model | yes |
| Isaac Sim camera / Replicator on the same stages | Isaac Sim | no (needs an RTX GPU) |
| Cosmos Transfer (photoreal frames) | NVIDIA Cosmos | no (needs a GPU or `NVIDIA_API_KEY`) |

```bash
pip install usd-core newton newton-usd-schemas
python examples/nvidia/17_omniverse_robotsim_worldmodel_perception/run.py   # ~80 min on 4 CPU cores, shared
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU)

- **Labels:** the forward kinematics equal Newton's link poses to **1.4e-7 m**, and the USD timelines validate.
- **Visibility check:** keeps 3 828 of 4 000 training frames. All 500 test frames happen to be visible, so it drops
  no test frame.
- **CNN on 500 test frames:**
  - Joint angle error **0.23 rad RMS** (limit 0.035 rad = 2 deg).
  - Per test move, the RMS error ranges from 0.8 to 25 deg; the median over moves is about 6 deg.
  - The frame figure shows the arm well located in most frames; a block of frames errs by more than 10 deg (`angles.png`).
- **Tool point:** 0.18 m RMS over all frames, 0.07 m median over moves.
- **Speed:** the joint speed implied by the predicted angles is off by 60 % (median relative RMS); the elbow trace in
  `angles.png` shows why.
- **Jumps:** no frame-to-frame jump beyond the 12 rad/s rate limit.
- **Decision: REVIEW.** The data and its labels are exact, but this small CNN (4 000 steps on CPU) is not a usable
  joint-angle estimator. The dynamic checks catch what a per-frame angle error would hide: a jittery elbow sequence.

Figures in `results/`: `architecture.png`, `frames.png` (test frames with the Newton and CNN skeletons),
`angles.png` (one move's angles over time, error histogram), `usd_preview.png`, `checks.png`.
