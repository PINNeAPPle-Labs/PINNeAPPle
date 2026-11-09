"""MJX (JAX/GPU) backend for massively parallel MuJoCo rollouts.

MJX uploads a MjModel to JAX device memory and runs thousands of environments simultaneously using ``jax.vmap``.
This is the GPU path for generating large training datasets for surrogate models and PINNs.

Requirements
------------
MuJoCo 3.x shipped MJX inside the ``mujoco`` wheel; **4.0 moved it to the separate ``mujoco-mjx`` package** (the
import path is still ``mujoco.mjx``):

    pip install mujoco-mjx jax          # CPU
    pip install mujoco-mjx "jax[cuda12]"  # NVIDIA GPU

Usage
-----
    from pinneapple_simulation.external_solvers.mujoco import MJXParallelRunner, MuJoCoConfig

    cfg = MuJoCoConfig(model_path="robot.xml", n_steps=500)
    runner = MJXParallelRunner(cfg, n_envs=1024)
    batched_traj = runner.rollout()              # {"qpos": (1024, 500, nq), ...}
    samples = runner.rollout_to_upd()            # one PhysicalSample per environment
"""
from __future__ import annotations

import numpy as np

from .loader import MuJoCoConfig, has_mjx, load_model

__all__ = ["MJXParallelRunner"]


class MJXParallelRunner:
    """Run N independent MuJoCo environments in parallel via MJX + ``jax.vmap``.

    Parameters
    ----------
    config : MuJoCoConfig
    n_envs : number of parallel environments
    device : JAX device string, e.g. ``"cuda:0"``, ``"cpu"``. ``None`` = JAX default.

    Raises
    ------
    ImportError
        When MJX is not installed (in 4.0 it is the separate ``mujoco-mjx`` package) or JAX is missing.
    """

    def __init__(
        self,
        config: MuJoCoConfig,
        n_envs: int = 256,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.n_envs = n_envs
        self.device = device
        self._mjx_model = None

    @staticmethod
    def available() -> bool:
        """Whether MJX can be imported here (``mujoco.mjx``, from ``mujoco-mjx`` on 4.0)."""
        return has_mjx()

    def _ensure_mjx(self):
        try:
            import jax  # noqa: F401
            from mujoco import mjx
        except ImportError as e:
            raise ImportError(
                "MJX requires JAX and MuJoCo. In MuJoCo 4.0 MJX is a separate package. Install with:\n"
                "  pip install mujoco-mjx jax            # CPU\n"
                '  pip install mujoco-mjx "jax[cuda12]"  # NVIDIA GPU'
            ) from e
        if self._mjx_model is None:
            mj_model = load_model(self.config)
            self._mjx_model = mjx.put_model(mj_model)
        return self._mjx_model

    def rollout(self, n_steps: int | None = None) -> dict:
        """Run batched rollout across ``n_envs`` environments.

        Returns
        -------
        dict with keys ``qpos``, ``qvel``, ``ctrl``, ``xpos``, ``time``.
        Each value is a NumPy array of shape ``(n_envs, T, ...)``.
        """
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        mjx_model = self._ensure_mjx()
        n_envs = self.n_envs
        n_steps = int(self.config.n_steps if n_steps is None else n_steps)

        def make_data(_):
            return mjx.make_data(mjx_model)

        init_data = jax.vmap(make_data)(jnp.arange(n_envs))

        def step_fn(data, _):
            data = mjx.step(mjx_model, data)
            record = {
                "qpos": data.qpos,
                "qvel": data.qvel,
                "ctrl": data.ctrl,
                "xpos": data.xpos,
                "time": data.time,
            }
            return data, record

        batched_step = jax.vmap(
            lambda d: jax.lax.scan(step_fn, d, None, length=n_steps)
        )

        _, traj = batched_step(init_data)

        return {k: np.asarray(v) for k, v in traj.items()}

    def rollout_to_upd(self, n_steps: int | None = None) -> list:
        """Run a batched rollout and return a list of PhysicalSamples (one per environment)."""
        from .adapter import trajectory_to_upd

        batch = self.rollout(n_steps=n_steps)
        samples = []
        for i in range(self.n_envs):
            env_traj = {k: v[i] for k, v in batch.items()}
            samples.append(
                trajectory_to_upd(env_traj, meta_extra={"env_idx": i, "backend": "mjx"})
            )
        return samples
