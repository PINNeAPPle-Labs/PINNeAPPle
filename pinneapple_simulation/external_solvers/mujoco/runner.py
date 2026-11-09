"""MuJoCo simulation runner for PINNeAPPle.

:class:`MuJoCoRunner` executes the ``mj_step`` loop and records the trajectory (``qpos``, ``qvel``, ``ctrl``,
body poses, body velocities, sensors, energy and contact count) into a :class:`~.trajectory.MuJoCoTrajectory`, ready
for surrogate / PINN training or for the checks in :mod:`.validation`.

Controller injection
--------------------
Pass a callable ``controller(model, data) -> None`` that writes ``data.ctrl`` before each step. This closes the loop
with a PINN-predicted or RL-policy controller:

    def my_ctrl(model, data):
        data.ctrl[:] = pinn_policy(data.qpos, data.qvel)

    runner = MuJoCoRunner(config, controller=my_ctrl)
    traj = runner.simulate()

Reading the state
-----------------
Each record is taken with the derived fields in sync with the state (``mj_forward`` runs after every step when
``sync_derived`` is on), so ``xpos``, ``cvel`` and ``sensordata`` in row ``k`` all describe time ``t[k]`` — MuJoCo's
own ``mj_step`` leaves them one substep behind. ``simulate`` records ``n_steps + 1`` rows, the initial state included.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .loader import MuJoCoConfig, load_model, make_data, model_summary, require_mujoco, reset_data
from .trajectory import MuJoCoTrajectory

__all__ = ["MuJoCoRunner"]


def _full_inertia(model, body_index: int) -> np.ndarray:
    """3x3 inertia about the centre of mass, in the body frame (MuJoCo stores the principal diagonal and a rotation)."""
    mujoco = require_mujoco()
    diag = np.asarray(model.body_inertia[body_index], dtype=float)
    quat = np.asarray(model.body_iquat[body_index], dtype=float)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    return rot @ np.diag(diag) @ rot.T


class MuJoCoRunner:
    """Run a MuJoCo model and collect a trajectory.

    Parameters
    ----------
    config : MuJoCoConfig
    controller : optional callable ``(model, data) -> None`` executed before every step.
    sync_derived : run ``mj_forward`` after each step so every record is internally consistent (default ``True``).
        Turn it off for speed when only ``qpos``/``qvel`` are used, and call ``mj_forward`` yourself before reading
        derived fields.
    """

    def __init__(self, config: MuJoCoConfig, controller: Callable | None = None, sync_derived: bool = True) -> None:
        self.config = config
        self.controller = controller
        self.sync_derived = bool(sync_derived)
        self._model = None
        self._data = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.config)
        return self._model

    @property
    def data(self):
        if self._data is None:
            self._data = make_data(self.model)
        return self._data

    def reset(self) -> None:
        """Reset the simulation to the configured keyframe (or the model's ``qpos0``)."""
        reset_data(self.model, self.data, self.config.keyframe)

    # ------------------------------------------------------------------ simulation
    def simulate(
        self,
        q0: np.ndarray | None = None,
        qd0: np.ndarray | None = None,
        n_steps: int | None = None,
        record_every: int = 1,
        reset: bool = True,
    ) -> MuJoCoTrajectory:
        """Run the simulation and return the recorded trajectory.

        Parameters
        ----------
        q0, qd0 : optional initial generalized position / velocity, written after the reset.
        n_steps : override for ``config.n_steps``.
        record_every : keep one record every this many steps (the first and last ones are always kept).
        reset : start from the configured keyframe (default) or continue from the current state.
        """
        mujoco = require_mujoco()

        config = self.config
        model = self.model
        data = self.data
        if reset:
            self.reset()
        if q0 is not None:
            q0 = np.asarray(q0, dtype=float)
            if q0.size != model.nq:
                raise ValueError(f"q0 has {q0.size} values; the model has nq={model.nq}")
            data.qpos[:] = q0.reshape(-1)
        if qd0 is not None:
            qd0 = np.asarray(qd0, dtype=float)
            if qd0.size != model.nv:
                raise ValueError(f"qd0 has {qd0.size} values; the model has nv={model.nv}")
            data.qvel[:] = qd0.reshape(-1)
        mujoco.mj_forward(model, data)

        n = int(config.n_steps if n_steps is None else n_steps)
        if n < 0:
            raise ValueError(f"n_steps must be >= 0, got {n}")
        record_every = max(1, int(record_every))
        record_energy = bool(config.record_energy)
        record_contacts = bool(config.record_contacts)

        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
                      for i in range(model.nbody)]
        joint_names = [_joint_label(model, i) for i in range(model.njnt)]

        nbody = model.nbody
        record = {"t": [], "qpos": [], "qvel": [], "ctrl": [], "xpos": [], "xquat": [], "cvel": [],
                  "sensordata": [], "ncon": [], "energy": []}
        wall0 = time.perf_counter()

        for k in range(n + 1):
            keep = (k % record_every == 0) or (k == n)
            if keep:
                record["t"].append(float(data.time))
                record["qpos"].append(data.qpos.copy())
                record["qvel"].append(data.qvel.copy())
                record["ctrl"].append(data.ctrl.copy())
                record["xpos"].append(data.xpos.copy())
                record["xquat"].append(data.xquat.copy())
                record["cvel"].append(data.cvel.copy())
                record["sensordata"].append(data.sensordata.copy())
                if record_contacts:
                    record["ncon"].append(int(data.ncon))
                if record_energy:
                    record["energy"].append(np.array(data.energy, dtype=float))
            if k == n:
                break
            if self.controller is not None:
                self.controller(model, data)
            mujoco.mj_step(model, data, nstep=config.n_substeps)
            if self.sync_derived or ((k + 1) % record_every == 0) or (k + 1 == n):
                mujoco.mj_forward(model, data)

        wall = time.perf_counter() - wall0
        return MuJoCoTrajectory(
            body_names=body_names,
            joint_names=joint_names,
            t=np.asarray(record["t"], dtype=float),
            qpos=np.asarray(record["qpos"], dtype=float),
            qvel=np.asarray(record["qvel"], dtype=float),
            ctrl=np.asarray(record["ctrl"], dtype=float),
            xpos=np.asarray(record["xpos"], dtype=float).reshape(-1, nbody, 3),
            xquat=np.asarray(record["xquat"], dtype=float).reshape(-1, nbody, 4),
            cvel=np.asarray(record["cvel"], dtype=float).reshape(-1, nbody, 6),
            sensordata=np.asarray(record["sensordata"], dtype=float).reshape(len(record["t"]), model.nsensordata),
            ncon=np.asarray(record["ncon"], dtype=int) if record_contacts else np.empty(0, dtype=int),
            energy_rows=np.asarray(record["energy"], dtype=float).reshape(-1, 2) if record_energy
            else np.empty((0, 2)),
            mass=np.asarray(model.body_mass, dtype=float).copy(),
            com=np.asarray(model.body_ipos, dtype=float).copy(),
            inertia=np.asarray([_full_inertia(model, i) for i in range(nbody)]).reshape(nbody, 3, 3),
            gravity=np.asarray(model.opt.gravity, dtype=float).copy(),
            dt=float(model.opt.timestep * config.n_substeps * record_every),
            backend="mj_step",
            wall_s=wall,
            keyframe=config.keyframe,
            meta=model_summary(model),
        )

    def collect_trajectories(self, n_rollouts: int, reset_between: bool = True) -> list:
        """Run several rollouts and return one trajectory per rollout.

        Parameters
        ----------
        n_rollouts : number of independent rollouts.
        reset_between : reset to the configured initial state before each rollout (the first one is reset anyway).
        """
        trajectories = []
        for _ in range(int(n_rollouts)):
            trajectories.append(self.simulate(reset=reset_between))
        return trajectories


def _joint_label(model, index: int) -> str:
    """Name of a joint, falling back to ``joint{index}`` for unnamed joints (e.g. an anonymous free joint)."""
    mujoco = require_mujoco()
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
    if name:
        return name
    return f"joint{index}_{mujoco.mjtJoint(model.jnt_type[index]).name.replace('mjJNT_', '').lower()}"
