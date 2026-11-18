# MuJoCo 4.0 quickstart

Runs without any model or data file: two minimal MJCF strings, one two-link arm from
`pinneapple_simulation.robot_sim.arms`. Needs `pip install "pinneapple[mujoco]"` (MuJoCo 4.0 on Python ≥ 3.11,
3.17 on 3.10) and, for the robot part, nothing else — the reference torques come from PINNeAPPle's own multibody tree.

```bash
python examples/mujoco/quickstart.py
```

Output with MuJoCo 4.0.0 on Python 3.11 (macOS arm64):

```
MuJoCo 4.0.0
records 301, qpos (301, 7), z(end) 0.55708 m (closed form 0.55855 m)
energy drift 1.47e-03, UPD fields ['kinetic', 'potential', 'qpos', 'qvel', 'total', 'xpos', 'xquat']
  free_fall          2.63e-03 <= 5e-03  True
  pendulum_period    3.50e-04 <= 1e-02  True
  sliding_friction   1.62e-03 <= 5e-02  True
  gravity_torque     0.00e+00 <= 1e-06  True
  torque_free_spin   1.73e-03 <= 1e-02  True
  resting_contact    1.20e-08 <= 1e-04  True
decision: APPROVED
batch (64, 200, 7) on all threads; z(end) 0.0314 m +/- 0.3451 m
arm held to 4.1e-10 rad; energy change 1.0e-08 J; torque [-59.277 -21.783]
```

The four parts:

1. **Trajectory** — 300 steps at `dt = 1e-3`, recorded with the initial state (301 rows), exported as a
   `PhysicalSample` (UPD) with a time coordinate, units and provenance.
2. **Checks** — `validate()` compares six runs against their closed-form answers; the free-fall error is what
   semi-implicit Euler costs at this step, and the `resting_contact` value is the sinking of a settled box.
3. **Batch** — 64 simulations of one model on a thread pool (`mujoco.batch`, MuJoCo ≥ 3.17), each with its own sphere
   mass drawn from 0.5–2.0 kg; the spread in the final height is the parameter spread, not solver noise.
4. **Robot** — MuJoCo holds the arm at `q0` to 4.1e-10 rad with torques computed by
   `pinneapple_simulation.multibody`, an independent model, and the total energy does not move.

See [`docs/guides/mujoco.md`](../../docs/guides/mujoco.md) for the full API and what MuJoCo 4.0 changed.
