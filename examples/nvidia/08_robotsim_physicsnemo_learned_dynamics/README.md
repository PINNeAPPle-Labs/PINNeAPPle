# 08 Learned arm dynamics: PINNeAPPle + PINNeAPPle Robot Sim (Newton) + PhysicsNeMo

**What.** The robot sim (NVIDIA Newton) runs the two-link arm of example 03 through 120 random smooth motions (PD
tracking of random multi-sine references, 1 s each, 1 ms steps) and records angles, rates, accelerations and the
delivered torques. `physicsnemo.models.mlp.FullyConnected` learns the inverse dynamics `tau = f(q, qd, qdd)` and
drives the pick-and-place motion as feedforward.

**Why.** A learned dynamics model that fits its data can still be unphysical. PINNeAPPle checks what any valid
inverse-dynamics model must satisfy, independently of the training loss: torques against its own rigid-body model,
gravity torques at rest, the mass matrix the network implies (symmetric, positive definite, torque linear in the
acceleration), and whether a new motion lies inside the training data.

| Step | Who | Ran here |
|---|---|---|
| 120 000 samples from Newton (torque sensors, solver accelerations) | Robot Sim (Newton 1.7) | yes |
| FullyConnected 4 x 128 (51 k parameters), 40 000 Adam steps | PhysicsNeMo 2.2.2 (CPU) | yes |
| rigid-body reference, gravity, mass matrix, linearity, envelope checks; decision | PINNeAPPle | yes |
| tracking with PD only, PD + rigid-body feedforward, PD + PhysicsNeMo feedforward | Robot Sim | yes |

```bash
pip install newton nvidia-physicsnemo
python examples/nvidia/08_robotsim_physicsnemo_learned_dynamics/run.py      # ~3 min on 4 CPU cores
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU)

- Newton's own torques agree with PINNeAPPle's rigid-body model to 9e-6 (relative RMS): the data is right.
- Network on two held-out Newton runs: relative RMS torque error **4.9 %** (shoulder) and 2.3 % (elbow).
- Physics of the network: gravity torques at rest within 0.6 %; implied mass matrix within 0.7 % of the rigid-body
  one, symmetric to 1 %, positive definite (smallest eigenvalue 0.115 kg m2), linear in the acceleration to 0.4 %.
- Tracking the pick-and-place in Newton (same PD gains): PD only 962 mrad, + rigid-body feedforward 1.9 mrad,
  **+ PhysicsNeMo feedforward 4.6 mrad**.
- **Decision: REVIEW.** Every physics check passes, but 0.12 % of the pick-and-place samples (the start pose, at the
  edge of the training data) fall outside the training envelope; a 3x faster version of the motion has 0.37 % of its
  samples outside. The box envelope is a weak detector here: most of a 3x faster motion still lies inside it.

Figures in `results/`: `architecture.png`, `dynamics.png` (held-out torque, gravity torque, tracking), `checks.png`.
