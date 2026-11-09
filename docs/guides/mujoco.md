# MuJoCo 4.0: trajectories, checks and the robot backend

`pinneapple_simulation.external_solvers.mujoco` puts [MuJoCo](https://github.com/google-deepmind/mujoco) under the
same workflow PINNeAPPle uses for its other solvers: one configuration, a recorded trajectory that exports to the UPD,
analytic checks of the run before the data is used, parallel rollouts for dataset generation, and — for robots — the
same `run() -> RobotRun` interface NVIDIA Newton has. Everything here was run against **MuJoCo 4.0.0** (released
2026-11-06) on Python 3.11; the 3.17 line is supported where 4.0 cannot install.

## Install

```bash
pip install "pinneapple[mujoco]"        # MuJoCo 4.0 on Python >= 3.11, MuJoCo 3.17 on 3.10
pip install "pinneapple[mujoco_mjx]"    # + mujoco-mjx and JAX, for GPU / TPU rollouts
```

MuJoCo is optional: without it the package imports, `mujoco_available()` returns `False`, and the checks are skipped.

## What MuJoCo 4.0 changed, and what this layer does about it

| Change in 4.0 | Here |
|---|---|
| `mujoco` split into `mujoco-core` + `mujoco-studio` | nothing to do: `pip install "pinneapple[mujoco]"` pulls the right wheels |
| **MJX left the wheel** (`mujoco-mjx` is a separate package, import path still `mujoco.mjx`) | `has_mjx()` detects it; `MJXParallelRunner` raises an install message that names `mujoco-mjx` |
| Python 3.10 no longer supported | the extra installs `mujoco>=3.17,<4` on 3.10 and `mujoco>=4.0` on 3.11+ |
| New physics: speculative contacts, elasto-plastic flexes and springs | reachable through the MJCF you pass; the trajectory records them like any other state |
| `mujoco.sysid`, `mujoco.MjSpec`, `mujoco.batch` (3.17+) | `load_spec()` / `spec_to_xml()` expose `MjSpec`; `BatchRunner` uses `mujoco.batch` |

Two API details this layer handles for you: `MjData` records are taken with `mj_forward` in sync (MuJoCo leaves the
derived fields one substep behind), and the robot backend writes the compiler's `angle="radian"` so joint ranges from
an `ArticulationSpec` are not read as degrees.

## Simulate and export

```python
from pinneapple_simulation.external_solvers.mujoco import MuJoCoConfig, MuJoCoRunner

cfg = MuJoCoConfig(model_path="hopper.xml", n_steps=500, dt=2e-3, keyframe="stand")
traj = MuJoCoRunner(cfg).simulate()

traj.t, traj.qpos, traj.qvel            # recorded with the initial state, so T = n_steps + 1
traj.body("torso")["pos"]               # xpos of one body over time
traj.energy()["total"]                  # potential + kinetic, from mjENBL_ENERGY
traj.ncon                               # active contacts per record
sample = traj.to_upd()                  # PhysicalSample: state, time coordinate, provenance, units
```

`MuJoCoRunner(config, controller=fn)` injects a controller `fn(model, data)` before each step, which is the hook for a
PINN-predicted or RL policy. Both the runner and the batch backend return the same `MuJoCoTrajectory`, and
`trajectory_to_upd` still accepts the dict the earlier runner returned.

## Check the run before you train on it

`validate()` runs six cases whose answers are closed-form. They catch the failures that produce plausible-looking
datasets: a timestep that is too large, a friction model that was not what you thought, contacts sinking, a model
built with the wrong joint frame.

```python
from pinneapple_simulation.external_solvers.mujoco import validate, summary

rows = validate(dt=1e-3)                 # or validate(cases=["free_fall", "sliding_friction"])
summary(rows)["decision"]                # "APPROVED" / "REJECT"
```

Measured with MuJoCo 4.0.0 at `dt = 1e-3` (the value is the relative error against the closed form):

| Case | Value | Limit | Against |
|---|---:|---:|---|
| `free_fall` | 2.6e-3 | 5e-3 | `z = z0 - g t^2 / 2`, `v = -g t` |
| `pendulum_period` | 3.5e-4 | 1e-2 | `T = 2*pi*sqrt(L/g)`, measured from zero crossings |
| `sliding_friction` | 1.6e-3 | 5e-2 | stopping distance `v0^2 / (2 mu g)` |
| `gravity_torque` | 0.0 | 1e-6 | `m g L sin(theta)` against `qfrc_bias` |
| `torque_free_spin` | 1.7e-3 | 1e-2 | drift of kinetic energy and angular momentum |
| `resting_contact` | 1.2e-8 | 1e-4 | sinking of a settled box, with contacts present |

## Datasets: many simulations of one model on a thread pool

`mujoco.batch` steps N independent simulations on a thread pool, with results bit-identical to a loop at any thread
count, and lets every environment carry its own model parameters. That is a design-of-experiments campaign from one
MJCF, on CPU, without JAX:

```python
from pinneapple_simulation.external_solvers.mujoco import BatchRunner

runner = BatchRunner(cfg, n_envs=256, seed=0)
runner.set_state("qpos", [0, 0, 1.2, 1, 0, 0, 0])
runner.randomize("body", "torso", "mass", 0.8, 1.2)
runner.randomize("geom", "foot", "friction", 0.4, 1.0)
batch = runner.rollout(n_steps=500)      # {"qpos": (256, 500, nq), "xpos": (256, 500, nbody, 3), ...}
samples = runner.rollout_to_upd()        # one PhysicalSample per environment, provenance says backend="batch"
```

For closed-loop policies use `roll_loop(n_steps, policy=fn)`, which binds the state arrays and lets
`policy(qpos, qvel, k)` write the controls. For GPU/TPU and differentiable physics, `MJXParallelRunner` does the same
job through `jax.vmap` once `mujoco-mjx` and JAX are installed.

## Robots: the same interface as Newton, checked against an independent model

`MuJoCoRobotSim` returns the same `RobotRun` as `pinneapple_simulation.robot_sim.RobotSim`, so `run.energy()`,
`run.actuator_work()`, `run.imu(body)` and the PINNeAPPle checks work unchanged:

```python
from pinneapple_simulation.robot_sim.arms import TwoLinkArm
from pinneapple_simulation.external_solvers.mujoco import MuJoCoRobotSim
from pinneapple_simulation.multibody import arm as mb

arm = TwoLinkArm()
tree = arm.tree()                        # PINNeAPPle's own multibody tree
sim = MuJoCoRobotSim.from_spec(arm.spec(), ground=False)
q0 = [-0.6, 1.2]
feed = lambda t, q, qd: -mb.fixed_base_inverse_dynamics(tree, -q, [0, 0], [0, 0])[0]
run = sim.run(0.3, 5e-4, q0=q0, controller=feed)
abs(run.q - q0).max()                    # 4.1e-10 rad: two independent engines, same answer
```

The tree's joints turn about `-y`, so its torques are the negative of MuJoCo's — the same convention the Newton
backend uses. Models can also come from `from_mjcf`, `from_urdf` or `from_xml`; `spec_to_mjcf(spec)` writes the MJCF
from an `ArticulationSpec` (boxes, cylinders, hinges, one effort motor per joint, optional ground).

## Limits

- Contacts are recorded as a count, not as a per-contact table; ask for `d.contact` yourself when a study needs it.
- No rendering: this layer is for data and checks. MuJoCo's own viewers see the same model.
- `MJXParallelRunner` runs on CPU too, but it is the GPU path; on CPU `BatchRunner` is simpler and usually faster.
- The analytic checks cover rigid-body mechanics; flexes, tendons and actuators have no closed-form case here yet.
