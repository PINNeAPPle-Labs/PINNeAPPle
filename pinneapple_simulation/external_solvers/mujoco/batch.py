"""Thread-parallel MuJoCo rollouts: many simulations of one model, for training data.

``mujoco.batch`` (MuJoCo 3.17+, and 4.0) steps N independent simulations of a model on a thread pool, with results
bit-identical to a loop over one ``MjData`` per simulation at any thread count. It runs on CPU (no JAX, no GPU) and
lets each simulation carry its own model parameters — mass, friction, gains — which is what turns one MJCF into a
design-of-experiments campaign.

    cfg = MuJoCoConfig(model_path="hopper.xml", dt=2e-3)
    runner = BatchRunner(cfg, n_envs=256)
    runner.randomize("body", "torso", "mass", 0.8, 1.2)        # domain randomisation, per environment
    batch = runner.rollout(n_steps=500)                          # {"qpos": (256, 500, nq), ...}
    samples = runner.rollout_to_upd()                             # one PhysicalSample per environment

For closed-loop policies (the control depends on the state) use :meth:`BatchRunner.roll_loop`, which binds the state
arrays and lets the policy write ``ctrl`` between steps. For GPU/TPU and differentiable physics use
:class:`~.mjx_backend.MJXParallelRunner` (``mujoco-mjx`` + JAX).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .loader import MuJoCoConfig, has_batch, load_model, model_summary, require_mujoco
from .runner import _full_inertia, _joint_label
from .trajectory import MuJoCoTrajectory

__all__ = ["BatchRunner", "DEFAULT_RECORD"]

#: Fields recorded by default (all exist in ``mujoco.batch.DATA_FIELDS``).
DEFAULT_RECORD = ("qpos", "qvel", "ctrl", "xpos", "xquat", "cvel", "sensordata", "ncon", "energy")


class BatchRunner:
    """Run ``n_envs`` copies of one MuJoCo model in parallel on a thread pool.

    Parameters
    ----------
    config : MuJoCoConfig (its ``n_steps`` is the default rollout length).
    n_envs : number of independent simulations.
    nthread : worker threads; ``0`` (default) uses every CPU.
    seed : seed for :meth:`randomize`.

    Raises
    ------
    ImportError
        When ``mujoco`` is missing or older than 3.17 (no ``mujoco.batch``).
    """

    def __init__(self, config: MuJoCoConfig, n_envs: int = 256, nthread: int = 0,
                 seed: int | None = None) -> None:
        if not has_batch():
            version = "not installed"
            from .loader import mujoco_version

            if mujoco_version() is not None:
                version = ".".join(map(str, mujoco_version()))
            raise ImportError(
                f"mujoco.batch needs MuJoCo >= 3.17 (found {version}). "
                'Install it with pip install --upgrade "pinneapple[mujoco]".'
            )
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.config = config
        self.n_envs = int(n_envs)
        self.nthread = int(nthread)
        self.seed = seed
        self._batch = None
        self._model = None
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def available() -> bool:
        """Whether ``mujoco.batch`` can be used in this environment."""
        return has_batch()

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.config)
        return self._model

    @property
    def batch(self):
        if self._batch is None:
            from mujoco import batch as mjbatch

            self._batch = mjbatch.Batch(self.model, nsim=self.n_envs, nthread=self.nthread)
            self._batch.reset()
        return self._batch

    def reset(self) -> None:
        """Reset every simulation to the model's default state."""
        self.batch.reset()

    # ------------------------------------------------------------------ parameters
    def set_field(self, kind: str, name: str, field: str, values) -> np.ndarray:
        """Write one model field per environment, e.g. ``set_field("body", "torso", "mass", [...])``.

        ``kind`` is a named view of ``mujoco.batch`` (``"body"``, ``"geom"``, ``"joint"``, ``"actuator"``, ...),
        ``field`` an attribute of that view (``"mass"``, ``"friction"``, ``"damping"``, ...). ``values`` is
        broadcast to the view's shape. Returns what was written.
        """
        view = getattr(self.batch, kind)(name)
        current = getattr(view, field)
        target = np.broadcast_to(np.asarray(values, dtype=float), current.shape).copy()
        setattr(view, field, target)
        return target

    def randomize(self, kind: str, name: str, field: str, low: float, high: float) -> np.ndarray:
        """Draw one uniform value per environment in ``[low, high)`` and write it to a model field."""
        view = getattr(self.batch, kind)(name)
        shape = getattr(view, field).shape
        values = self._rng.uniform(float(low), float(high), size=(self.n_envs, 1))
        target = np.broadcast_to(values, shape).copy()
        setattr(view, field, target)
        return target

    def set_state(self, field: str, values) -> None:
        """Write a state field (``"qpos"``, ``"qvel"``, ``"ctrl"``, ``"mocap_pos"``, ...) for every simulation."""
        bound = self.batch.bind(field)
        bound[...] = np.broadcast_to(np.asarray(values, dtype=float), bound.shape)

    # ------------------------------------------------------------------ rollouts
    def rollout(self, n_steps: int | None = None, control=None,
                record: Sequence[str] = DEFAULT_RECORD) -> dict:
        """Open-loop rollouts: advance every simulation ``n_steps`` and record after each step.

        Parameters
        ----------
        n_steps : steps per simulation (default ``config.n_steps``).
        control : ``(n_envs, n_steps, nu)`` control sequence, or ``None`` to keep the current ``ctrl``.
        record : ``MjData`` fields to record.

        Returns
        -------
        dict mapping each recorded field to an array ``(n_envs, n_steps, ...)``.
        """
        n = int(self.config.n_steps if n_steps is None else n_steps)
        out = self.batch.rollout(n, control=control, record=tuple(record))
        return {key: np.asarray(value) for key, value in out.items()}

    def roll_loop(self, n_steps: int | None = None, policy: Callable | None = None,
                  record: Sequence[str] = DEFAULT_RECORD) -> dict:
        """Closed-loop rollouts: ``policy(qpos, qvel, k)`` returns controls ``(n_envs, nu)`` before each step.

        The policy receives live views of the batch state (they update in place), so it must return a new array
        rather than keeping a reference. Unlike :meth:`rollout`, the initial state is recorded as the first row,
        so the arrays are ``(n_envs, n_steps + 1, ...)``.
        """
        b = self.batch
        n = int(self.config.n_steps if n_steps is None else n_steps)
        ctrl = b.bind("ctrl")
        qpos, qvel = b.bind("qpos"), b.bind("qvel")
        out = {key: [] for key in record}
        for k in range(n + 1):
            for key in record:
                out[key].append(np.array(getattr(b, key), copy=True))
            if k == n:
                break
            if policy is not None:
                ctrl[...] = np.asarray(policy(qpos, qvel, k), dtype=float)
            b.step()
        return {key: np.stack(values, axis=1) for key, values in out.items()}

    # ------------------------------------------------------------------ export
    def rollout_to_upd(self, n_steps: int | None = None, field_map: dict | None = None) -> list:
        """Run a batch rollout and return one ``PhysicalSample`` per environment."""
        data = self.rollout(n_steps=n_steps)
        model = self.model
        n = data["qpos"].shape[1]
        dt = float(model.opt.timestep)
        body_names = [mj_body_name(model, i) for i in range(model.nbody)]
        joint_names = [_joint_label(model, i) for i in range(model.njnt)]
        mass = np.asarray(model.body_mass, dtype=float).copy()
        com = np.asarray(model.body_ipos, dtype=float).copy()
        inertia = np.asarray([_full_inertia(model, i) for i in range(model.nbody)]).reshape(model.nbody, 3, 3)
        summary = model_summary(model)
        samples = []
        for env in range(self.n_envs):
            traj = MuJoCoTrajectory(
                body_names=body_names,
                joint_names=joint_names,
                t=np.arange(1, n + 1, dtype=float) * dt,
                qpos=data["qpos"][env],
                qvel=data["qvel"][env],
                ctrl=data["ctrl"][env],
                xpos=data["xpos"][env],
                xquat=data["xquat"][env],
                cvel=data["cvel"][env],
                sensordata=data.get("sensordata", np.empty((n, 0)))[env],
                ncon=data.get("ncon", np.empty((n, 0)))[env].reshape(-1),
                energy_rows=data.get("energy", np.empty((n, 0)))[env].reshape(-1, 2) if "energy" in data
                else np.empty((0, 2)),
                mass=mass, com=com, inertia=inertia,
                gravity=np.asarray(model.opt.gravity, dtype=float).copy(),
                dt=dt, backend="batch", meta={**summary, "env_idx": env, "n_envs": self.n_envs},
            )
            samples.append(traj.to_upd(field_map=field_map))
        return samples


def mj_body_name(model, index: int) -> str:
    """Name of a body, falling back to ``body{index}`` for the unnamed world body."""
    mujoco = require_mujoco()
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or f"body{index}"
