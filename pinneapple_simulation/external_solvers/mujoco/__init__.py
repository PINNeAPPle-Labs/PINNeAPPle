"""MuJoCo integration for PINNeAPPle.

A workflow layer over the MuJoCo Python bindings (google-deepmind/mujoco) for **MuJoCo 4.0**, with a fallback to the
3.17 line on Python 3.10: structured model configuration, recorded trajectories, analytic checks, CPU thread-pool
rollouts (``mujoco.batch``), GPU rollouts (MJX), and the same robot interface PINNeAPPle already gives NVIDIA Newton.

What changed in MuJoCo 4.0 (released 2026-11-06)
------------------------------------------------
* MJX left the ``mujoco`` wheel: install ``mujoco-mjx`` (the import path is still ``mujoco.mjx``).
* ``mujoco.batch`` (3.17+) steps many simulations of one model on a thread pool, with per-simulation model
  parameters and bit-identical results — the CPU path for dataset generation, no JAX needed.
* Python 3.10 is no longer supported; ``pip install "pinneapple[mujoco]"`` picks the right line per interpreter.
* New physics this layer exposes through the model (nothing to configure here): speculative contacts
  (``<option flag="speculative">``), elasto-plastic flexes and springs, ``mujoco.sysid``.

Sub-modules
-----------
loader          MuJoCoConfig, load_model/make_data/reset_data, MjSpec, version probes
trajectory      MuJoCoTrajectory (states, energy, contacts) and its UPD export
runner          MuJoCoRunner: one ``mj_step`` loop, controller injection, keyframes
batch           BatchRunner: ``mujoco.batch`` thread-pool rollouts and domain randomisation
validation      analytic checks (free fall, pendulum, friction, torque, spin, resting contact)
robot_backend   MuJoCoRobotSim: ``run() -> RobotRun``, the interface Newton's RobotSim uses
mjx_backend     MJXParallelRunner: JAX/GPU rollouts (needs ``mujoco-mjx`` + ``jax``)

Quick start
-----------
>>> from pinneapple_simulation.external_solvers.mujoco import MuJoCoConfig, MuJoCoRunner, validate
>>> cfg = MuJoCoConfig(model_path="robot.xml", n_steps=500, dt=0.002)
>>> traj = MuJoCoRunner(cfg).simulate()
>>> traj.qpos.shape, traj.energy()["total"][:3]
>>> sample = traj.to_upd()                      # → PhysicalSample (UPD)
>>> rows = validate()                           # are these settings physically right?
>>> summary(rows)["decision"]

Parallel rollouts on CPU (no JAX):
>>> from pinneapple_simulation.external_solvers.mujoco import BatchRunner
>>> runner = BatchRunner(cfg, n_envs=256)
>>> runner.randomize("body", "torso", "mass", 0.8, 1.2)
>>> samples = runner.rollout_to_upd()

Robotics, same interface as Newton's ``RobotSim``:
>>> from pinneapple_simulation.external_solvers.mujoco import MuJoCoRobotSim
>>> sim = MuJoCoRobotSim.from_mjcf("arm.xml")
>>> run = sim.run(0.8, 5e-4, controller=lambda t, q, qd: -kp * q - kd * qd)
"""
from .adapter import trajectories_to_upd, trajectory_to_upd
from .batch import DEFAULT_RECORD, BatchRunner
from .loader import (
    MUJOCO_MIN_VERSION,
    MuJoCoConfig,
    has_batch,
    has_mjx,
    load_model,
    load_spec,
    make_data,
    model_summary,
    mujoco_available,
    mujoco_version,
    require_mujoco,
    reset_data,
    spec_to_xml,
)
from .robot_backend import MuJoCoRobotSim, spec_to_mjcf
from .runner import MuJoCoRunner
from .trajectory import MuJoCoTrajectory
from .validation import CASES, summary, validate

try:  # the class itself only imports JAX/MJX when it runs, but keep the import defensive
    from .mjx_backend import MJXParallelRunner
except Exception:  # pragma: no cover - depends on the environment
    MJXParallelRunner = None  # type: ignore

__all__ = [
    "MUJOCO_MIN_VERSION",
    "MuJoCoConfig",
    "mujoco_available",
    "mujoco_version",
    "require_mujoco",
    "has_batch",
    "has_mjx",
    "load_model",
    "make_data",
    "reset_data",
    "load_spec",
    "spec_to_xml",
    "model_summary",
    "MuJoCoTrajectory",
    "MuJoCoRunner",
    "BatchRunner",
    "DEFAULT_RECORD",
    "trajectory_to_upd",
    "trajectories_to_upd",
    "CASES",
    "validate",
    "summary",
    "MuJoCoRobotSim",
    "spec_to_mjcf",
    "MJXParallelRunner",
]
