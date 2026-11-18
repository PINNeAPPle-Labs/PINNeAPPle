"""Recorded MuJoCo trajectories as one object.

:class:`MuJoCoTrajectory` is what :class:`~.runner.MuJoCoRunner` and :class:`~.batch.BatchRunner` return: the state
per recorded time together with the model's mass properties, so a trajectory can be checked (energy, contacts) and
exported to PINNeAPPle's UPD (:meth:`MuJoCoTrajectory.to_upd`) without carrying the ``MjModel`` around.

Times and states follow MuJoCo's conventions: ``qpos``/``qvel`` are generalized coordinates and velocities, body
quaternions are ``(w, x, y, z)``, ``cvel`` is ``(angular, linear)`` in the world frame about each body's centre of
mass. Positions and quaternions are for the body frame origin, as ``MjData.xpos`` / ``xquat`` give them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MuJoCoTrajectory"]

#: Trajectory fields that can be exported as UPD state fields, with their units.
UNITS = {
    "qpos": "mixed (m, rad)",
    "qvel": "mixed (m/s, rad/s)",
    "ctrl": "mixed (N m, N)",
    "xpos": "m",
    "xquat": "1",
    "cvel": "mixed (rad/s, m/s)",
    "sensordata": "model units",
    "ncon": "1",
    "potential": "J",
    "kinetic": "J",
}


@dataclass
class MuJoCoTrajectory:
    """States recorded from one MuJoCo run.

    Attributes
    ----------
    t : (T,) simulation times [s].
    qpos : (T, nq) generalized positions.
    qvel : (T, nv) generalized velocities.
    ctrl : (T, nu) actuator controls.
    xpos : (T, nbody, 3) body frame origins, world frame [m].
    xquat : (T, nbody, 4) body orientations ``(w, x, y, z)``.
    cvel : (T, nbody, 6) world-frame ``(angular, linear)`` velocity about each body's centre of mass.
    sensordata : (T, nsensordata) sensor readings (empty when the model has no sensors).
    ncon : (T,) number of active contacts (empty when contacts were not recorded).
    energy_rows : (T, 2) potential and kinetic energy [J] (empty when energy was not recorded).
    """

    body_names: list[str]
    joint_names: list[str]
    t: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    xpos: np.ndarray
    xquat: np.ndarray
    cvel: np.ndarray
    sensordata: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    ncon: np.ndarray = field(default_factory=lambda: np.empty(0))
    energy_rows: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    mass: np.ndarray = field(default_factory=lambda: np.empty(0))
    com: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    inertia: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    dt: float = 0.0
    backend: str = "mujoco"
    wall_s: float = 0.0
    keyframe: object | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ mapping compatibility
    _ARRAYS = ("t", "qpos", "qvel", "ctrl", "xpos", "xquat", "cvel", "sensordata", "ncon", "energy_rows")

    def __getitem__(self, key: str):
        """Array access by field name, so code written against the older dict-returning runner keeps working."""
        if key in self._ARRAYS:
            return getattr(self, key)
        if key == "sensor":  # old name for sensordata
            return self.sensordata
        if key == "time":  # old name for t
            return self.t
        if key == "energy":
            return self.energy()
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return list(self._ARRAYS)

    def __len__(self) -> int:
        return len(self.t)

    # ------------------------------------------------------------------ lookups
    def index_body(self, name: str) -> int:
        """Index of a body in the recorded arrays (accepts the full MuJoCo name or its last path segment)."""
        if name in self.body_names:
            return self.body_names.index(name)
        short = [n.split("/")[-1] for n in self.body_names]
        if name in short:
            return short.index(name)
        raise KeyError(f"unknown body {name!r}; recorded bodies: {short}")

    def index_joint(self, name: str) -> int:
        """Index of a recorded joint (same naming rule as :meth:`index_body`)."""
        if name in self.joint_names:
            return self.joint_names.index(name)
        short = [n.split("/")[-1] for n in self.joint_names]
        if name in short:
            return short.index(name)
        raise KeyError(f"unknown joint {name!r}; recorded joints: {short}")

    def body(self, name: str) -> dict:
        """Recorded pose and velocity of one body: ``{"pos", "quat", "vel", "omega"}``."""
        i = self.index_body(name)
        return {
            "pos": self.xpos[:, i],
            "quat": self.xquat[:, i],
            "omega": self.cvel[:, i, :3],
            "vel": self.cvel[:, i, 3:],
        }

    # ------------------------------------------------------------------ physics
    def energy(self) -> dict:
        """Potential, kinetic and total mechanical energy per recorded time [J].

        Empty arrays when the run did not record energy (``MuJoCoConfig.record_energy=False``).
        """
        if self.energy_rows.size == 0:
            return {"potential": np.empty(0), "kinetic": np.empty(0), "total": np.empty(0)}
        potential = self.energy_rows[:, 0]
        kinetic = self.energy_rows[:, 1]
        return {"potential": potential, "kinetic": kinetic, "total": potential + kinetic}

    def energy_drift(self) -> float:
        """Largest relative change of the total energy over the run (0 for a conservative system)."""
        total = self.energy()["total"]
        if total.size < 2:
            return 0.0
        scale = max(float(np.abs(total).max()), 1e-12)
        return float(np.abs(total - total[0]).max() / scale)

    def com_world(self) -> np.ndarray:
        """Centre of mass of every body in the world frame, ``(T, nbody, 3)``."""
        if self.com.size == 0:
            return self.xpos
        return np.stack([self.xpos[k] + _rotate(self.xquat[k], self.com) for k in range(len(self.t))])

    # ------------------------------------------------------------------ export
    def to_upd(self, field_map: dict | None = None, fields: list | None = None):
        """The trajectory as a PINNeAPPle :class:`~pinneapple_data.physical_sample.PhysicalSample`.

        Parameters
        ----------
        field_map : optional rename map ``{trajectory_field: upd_field}``.
        fields : subset of the recorded fields to export (default: every non-empty one).
        """
        import torch

        from pinneapple_data.physical_sample import PhysicalSample

        if fields is None:
            fields = ["qpos", "qvel", "ctrl", "xpos", "xquat", "sensordata"]
        rename = field_map or {}
        state = {}
        units = {}
        for key in fields:
            arr = np.asarray(self[key])
            if arr.size == 0:
                continue
            name = rename.get(key, key)
            state[name] = torch.as_tensor(arr.reshape(len(self.t), -1).astype(np.float32))
            if key in UNITS:
                units[name] = UNITS[key]
        for name, values in self.energy().items():
            if values.size:
                state[rename.get(name, name)] = torch.as_tensor(values.astype(np.float32))
                units[rename.get(name, name)] = "J"

        provenance = {
            "version": "0.1",
            "source": "mujoco",
            "solver": f"mujoco.{self.backend}",
            "backend": self.backend,
            "dt": self.dt,
            "bodies": list(self.body_names),
            "joints": list(self.joint_names),
            **self.meta,
        }
        return PhysicalSample(
            state=state,
            domain={"type": "grid", "coords": {"time": self.t.astype(np.float32)}},
            provenance=provenance,
            schema={"units": units},
        )

    def save(self, path) -> None:
        """Write the trajectory to a compressed ``.npz`` (loadable without MuJoCo installed)."""
        np.savez_compressed(
            path,
            body_names=np.array(self.body_names, dtype=object),
            joint_names=np.array(self.joint_names, dtype=object),
            t=self.t, qpos=self.qpos, qvel=self.qvel, ctrl=self.ctrl, xpos=self.xpos, xquat=self.xquat,
            cvel=self.cvel, sensordata=self.sensordata, ncon=self.ncon, energy_rows=self.energy_rows,
            mass=self.mass, com=self.com, inertia=self.inertia, gravity=self.gravity, dt=self.dt,
            backend=self.backend,
        )


def _rotate(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate ``vectors`` (n, 3) by ``quat`` (w, x, y, z), broadcasting over the leading axis."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    w, x, y, z = w[..., None], x[..., None], y[..., None], z[..., None]
    rot = np.stack([
        np.concatenate([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.concatenate([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.concatenate([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
    return np.einsum("...ij,...j->...i", rot, vectors)
