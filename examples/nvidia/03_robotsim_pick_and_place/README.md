# 03 Pick-and-place arm: PINNeAPPle + PINNeAPPle Robot Sim (NVIDIA Newton)

**What.** A two-link arm (6061-T6 square tubes 50 x 50 x 3 mm, 0.6 m and 0.5 m, a 1.5 kg elbow motor, a 5 kg part
in the gripper) moves the part in 0.8 s along a rest-to-rest quintic. Stack I only: the robot sim is
`pinneapple_simulation.robot_sim.RobotSim` on NVIDIA Newton's `SolverFeatherstone` (CPU).

**Why.** Before a simulator's torques size a motor or a link, check that the simulator gets the arm right. PINNeAPPle
describes the arm once (`TwoLinkArm`: masses from the material density, lumped motor and payload), hands it to
Newton as URDF and to its own multibody tree, and compares the two. Then the joint-torque sensors of the final run
feed the structural checks.

| Step | Who | Ran here |
|---|---|---|
| arm description, planned motion, feedforward torques (multibody tree, checked against the closed-form two-link equations) | PINNeAPPle | yes |
| free swing, torque replay at three time steps, final tracking run with joint-torque sensors and IMU | Robot Sim (Newton 1.7, Warp 1.18, CPU) | yes |
| learning loop: cross-entropy method over 49 Newton rollouts to tune the PD gains around the feedforward | Robot Sim (`robot_sim.learn.cem`) | yes |
| energy and convergence checks, link-root stress against yield and fatigue, forearm deflection (beam FD), motor limits | PINNeAPPle | yes |
| the same `run()` on Isaac Sim (`IsaacSimBackend`) | Isaac Sim | **no** (needs Isaac Sim and an RTX GPU) |

```bash
pip install newton
python examples/nvidia/03_robotsim_pick_and_place/run.py      # ~1 min (plus ~20 s of one-off Warp kernel compilation)
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU, Newton 1.7.0)

- Simulator checks: a free swing conserves energy to 3.1e-4 at dt = 0.25 ms, with observed order **1.00**; replaying
  PINNeAPPle's torques open loop reproduces the planned motion with errors 25.0, 12.4 and 6.2 mrad at dt = 1, 0.5
  and 0.25 ms (order **1.01**, Richardson-extrapolated error -0.08 mrad). Newton's semi-implicit integrator is first
  order and agrees with the rigid-body equations as the step shrinks.
- Learned gains kp = (346, 110) N m/rad, kd = (4.9, 2.3) N m s/rad; tracking error **0.48 mrad**; actuator work
  equals the change of energy to 0.14 %; no saturation.
- Peak torques 114.3 N m (shoulder, motor 120) and 41.9 N m (elbow, motor 60). Link-root stress 13.7 and 5.0 MPa:
  utilisation 0.15 and 0.05 of yield / 3. Stress amplitude 7.0 MPa per cycle: 0.22 of the 6061-T6 fatigue strength
  / 3 (96.5 MPa at 5e8 cycles, ASM datasheet). Forearm tip deflection 0.27 mm at the IMU's 17.7 m/s2.
- **Decision: APPROVED**, with the shoulder motor at 95 % of its peak torque (the margin to watch).

Figures in `results/`: `architecture.png`, `tracking.png` (plan vs Newton, error, joint-torque sensors),
`verification.png` (convergence, learning curve, link stress), `motion.png`, `checks.png`.
