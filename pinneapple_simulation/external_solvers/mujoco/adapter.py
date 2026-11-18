"""Adapters from MuJoCo trajectories to PINNeAPPle's UPD format.

``trajectory_to_upd`` accepts either a :class:`~.trajectory.MuJoCoTrajectory` (what the runner and the batch backend
return) or the plain dict the older runner produced (``time``, ``qpos``, ``qvel``, ``ctrl``, ``xpos``, ``sensor``),
so existing code keeps working.
"""
from __future__ import annotations

import numpy as np

__all__ = ["trajectory_to_upd", "trajectories_to_upd"]


def _from_dict(traj: dict, field_map: dict[str, str] | None, meta_extra: dict | None):
    import torch

    from pinneapple_data.physical_sample import PhysicalSample

    default_fields = ["qpos", "qvel", "ctrl", "sensor"]
    fm = field_map or {}
    fields: dict[str, torch.Tensor] = {}
    for key in default_fields:
        arr = traj.get(key)
        if arr is None or np.asarray(arr).size == 0:
            continue
        name = fm.get(key, key)
        fields[name] = torch.as_tensor(np.asarray(arr, dtype=np.float32))

    if "xpos" in traj and np.asarray(traj["xpos"]).size > 0:
        xpos = np.asarray(traj["xpos"], dtype=np.float32)
        fields[fm.get("xpos", "xpos")] = torch.as_tensor(xpos.reshape(xpos.shape[0], -1))

    time = np.asarray(traj.get("time", traj.get("t")), dtype=np.float32)
    provenance = {"version": "0.1", "source": "mujoco", "solver": "mujoco.mj_step"}
    units: dict[str, str] = {}
    if meta_extra:
        provenance.update(meta_extra.get("provenance", meta_extra))
        units.update(meta_extra.get("units", {}))

    return PhysicalSample(
        state=fields,
        domain={"type": "grid", "coords": {"time": time}},
        provenance=provenance,
        schema={"units": units},
    )


def trajectory_to_upd(
    traj,
    field_map: dict[str, str] | None = None,
    meta_extra: dict | None = None,
):
    """Convert a MuJoCo trajectory to a UPD ``PhysicalSample``.

    Parameters
    ----------
    traj : :class:`~.trajectory.MuJoCoTrajectory` or a trajectory dict (the legacy runner output).
    field_map : optional rename map ``{trajectory_field: upd_field}``.
    meta_extra : extra keys merged into the provenance (and ``{"units": {...}}`` into the units), for dict input.

    Returns
    -------
    PhysicalSample (from ``pinneapple_data``)
    """
    if hasattr(traj, "to_upd") and not isinstance(traj, dict):
        if meta_extra:
            traj.meta.update(meta_extra)
        return traj.to_upd(field_map=field_map)
    return _from_dict(traj, field_map, meta_extra)


def trajectories_to_upd(
    trajs: list,
    field_map: dict[str, str] | None = None,
    meta_extra: dict | None = None,
) -> list:
    """Convert a list of trajectories to a list of PhysicalSamples."""
    return [trajectory_to_upd(t, field_map=field_map, meta_extra=meta_extra) for t in trajs]
