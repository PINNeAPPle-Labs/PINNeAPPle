# 12 One scene for Omniverse and the robot sim: PINNeAPPle + Omniverse + PINNeAPPle Robot Sim

**What.** The arm of example 03, authored once as an OpenUSD stage with `UsdPhysics` (rigid bodies with mass,
centre of mass and inertia, revolute joints with effort drives, articulation root, physics scene), simulated by the
robot sim (NVIDIA Newton) *from that file*, and written back to OpenUSD as a timeline: every link's pose per frame,
coloured by the link stress PINNeAPPle computes from the joint-torque sensors.

**Why.** A scene shown in Omniverse is only worth as much as its agreement with what was simulated. Here the file
Omniverse opens is the file Newton ran, and PINNeAPPle checks both ends: the physics stage (parser, validators,
masses), the simulation (same motion from USD and from URDF, energy balance), and the stresses on the animation.

| Step | Who | Ran here |
|---|---|---|
| arm masses and inertias, planned motion, feedforward, stress checks | PINNeAPPle | yes |
| UsdPhysics stage, parser + 30 validators; animated timeline stage, read-back GIF | OpenUSD (`usd-core`) | yes |
| simulation from the `.usda` (Newton `add_usd`) and from the URDF | Robot Sim (Newton 1.7) | yes |
| RTX render in Omniverse Kit (`results/kit_render.py`), Isaac Sim replay (`results/isaac_replay.py`) | Omniverse / Isaac Sim | **no** (need an RTX GPU) |

```bash
pip install newton newton-usd-schemas usd-core
python examples/nvidia/12_omniverse_robotsim_arm/run.py      # ~20 s
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU, Newton 1.7.0, usd-core 26.11)

- Physics stage: 3 rigid bodies, 2 revolute joints, 1 fixed joint, 1 articulation, total 28.18 kg; parser and
  validators clean.
- Newton reads the stage's masses, centres of mass and inertias exactly (difference 0), and the motion simulated
  from the USD equals the motion from the URDF **to the last bit** (max difference 0.0 rad).
- Tracking error 0.48 mrad; actuator work = change of energy to 0.14 %; link-root utilisation 0.15 (upper arm) and
  0.05 (forearm) of yield / 3. **Decision: APPROVED.**

Figures in `results/`: `architecture.png`, `robotsim.png` (USD vs URDF, energy balance, utilisation), `checks.png`,
`usd_physics_preview.png`, `usd_timeline_peak.png` (timeline read back at the stress peak), `usd_timeline.gif`.
