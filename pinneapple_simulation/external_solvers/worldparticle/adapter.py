"""Boundary adapter for WorldParticle's particle-trajectory model.

Source: https://github.com/WorldParticle/WorldParticle (MIT, 2026).
Uses the upstream TrajectoryModel API, not a copied/modified implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


@dataclass
class WorldParticleState:
    """Single (N,3) particle state plus optional, static model features."""

    position: Any
    velocity: Any
    feats: Any
    box: Any = None
    box_feats: Any = None
    neighbors_index: Any = None
    neighbors_row_splits: Any = None
    external_force: Any = None

    def validate(self) -> None:
        p, v = np.asarray(self.position), np.asarray(self.velocity)
        f = np.asarray(self.feats)
        if p.ndim != 2 or p.shape[1] != 3 or v.shape != p:
            raise ValueError("position and velocity must have identical [N,3] shapes")
        if f.ndim != 2 or f.shape[0] != len(p):
            raise ValueError("feats must have shape [N,F]")
        if (self.box is None) != (self.box_feats is None):
            raise ValueError("box and box_feats must both be provided or both omitted")
        if self.box is not None:
            b, bf = np.asarray(self.box), np.asarray(self.box_feats)
            if b.ndim != 2 or b.shape[1] != 3 or bf.ndim != 2 or len(bf) != len(b):
                raise ValueError("box must be [M,3] and box_feats must be [M,Fb]")
        if (self.neighbors_index is None) != (self.neighbors_row_splits is None):
            raise ValueError("topology neighbors require both CSR arrays")
        if self.neighbors_index is not None:
            index = np.asarray(self.neighbors_index)
            splits = np.asarray(self.neighbors_row_splits)
            if index.ndim != 1 or splits.shape != (len(p) + 1,):
                raise ValueError("invalid topology CSR shapes")
            if splits[0] != 0 or splits[-1] != len(index) or np.any(np.diff(splits) < 0):
                raise ValueError("invalid topology CSR row pointers")
            if not np.issubdtype(index.dtype, np.integer) or np.any(index < 0) or np.any(index >= len(p)):
                raise ValueError("topology neighbor indices out of bounds")
        if self.external_force is not None and np.asarray(self.external_force).shape != p.shape:
            raise ValueError("external_force must be [N,3]")


def load_worldparticle_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an upstream-style NPZ without enabling pickle deserialization."""
    allowed = {
        "position", "velocity", "feats", "box", "box_feats",
        "neighbors_index", "neighbors_row_splits", "force",
    }
    with np.load(path, allow_pickle=False) as file:
        arrays = {key: np.array(file[key]) for key in file.files if key in allowed}
    for name in ("position", "velocity"):
        if name not in arrays:
            raise ValueError(f"missing {name!r} in trajectory NPZ")
        if arrays[name].ndim != 3 or arrays[name].shape[-1] != 3:
            raise ValueError(f"{name} must be [T,N,3]")
    if arrays["position"].shape != arrays["velocity"].shape:
        raise ValueError("position and velocity trajectories must match")
    if not np.isfinite(arrays["position"]).all() or not np.isfinite(arrays["velocity"]).all():
        raise ValueError("nonfinite values in trajectory")
    if "feats" in arrays:
        WorldParticleState(
            position=arrays["position"][0], velocity=arrays["velocity"][0],
            feats=arrays["feats"], box=arrays.get("box"),
            box_feats=arrays.get("box_feats"),
            neighbors_index=arrays.get("neighbors_index"),
            neighbors_row_splits=arrays.get("neighbors_row_splits"),
            external_force=arrays["force"][0] if "force" in arrays else None,
        ).validate()
    return arrays


def save_worldparticle_npz(
    path: str | Path, position: Any, velocity: Any, **optional: Any
) -> Path:
    """Save a trajectory as the upstream NPZ schema, not a checkpoint."""
    p, v = np.asarray(position), np.asarray(velocity)
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape != v.shape:
        raise ValueError("position and velocity must have matching [T,N,3] shapes")
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError("nonfinite trajectory")
    allowed = {"feats", "box", "box_feats", "neighbors_index", "neighbors_row_splits", "force"}
    unknown = set(optional) - allowed
    if unknown:
        raise ValueError(f"unknown WorldParticle fields: {sorted(unknown)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, position=p, velocity=v, **{
        k: np.asarray(v) for k, v in optional.items() if v is not None
    })
    return target


def worldparticle_to_upd(
    position: Any, velocity: Any, *, dt: float | None = None,
    provenance: Mapping[str, Any] | None = None,
):
    """Convert [T,N,3] predictions into a point-domain PhysicalSample."""
    from pinneapple_data.physical_sample import PhysicalSample

    p, v = np.asarray(position), np.asarray(velocity)
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape != v.shape:
        raise ValueError("position and velocity must match [T,N,3]")
    if dt is not None and (not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be positive and finite")
    return PhysicalSample(
        state={"position": p, "velocity": v},
        domain={"type": "points", "coords": {
            "time": np.arange(len(p), dtype=float) * dt if dt is not None
            else np.arange(len(p)),
        }, "particle_count": p.shape[1]},
        schema={"units": {"position": "m", "velocity": "m/s"}},
        provenance={"source": "WorldParticle", "upstream": "WorldParticle/WorldParticle",
                    **dict(provenance or {})},
    )


def validate_worldparticle_rollout(
    position: Any, velocity: Any, *, dt: float,
    reference_position: Any = None, reference_velocity: Any = None,
) -> dict[str, float]:
    """Physical kinematic residual and optional held-out MSE, no target leakage."""
    p, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape != v.shape or len(p) < 2:
        raise ValueError("position and velocity must be [T,N,3] with T>=2")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError("positive dt and finite trajectories required")
    # Trapezoidal integration residual, independent of the model implementation.
    residual = p[1:] - p[:-1] - 0.5 * dt * (v[1:] + v[:-1])
    results = {"kinematic_rmse": float(np.sqrt(np.mean(residual ** 2)))}
    for field, pred, target in (
        ("position_mse", p, reference_position),
        ("velocity_mse", v, reference_velocity),
    ):
        if target is not None:
            t = np.asarray(target, dtype=float)
            if t.shape != pred.shape or not np.isfinite(t).all():
                raise ValueError(f"{field} reference must match finite prediction shape")
            results[field] = float(np.mean((pred - t) ** 2))
    return results


class WorldParticleBackend:
    """Roll out a compatible upstream model without importing WorldParticle.

    Inject a callable with the upstream signature:
    model(pos, vel, feats, box, box_feats,
          topology_neighbors_index=None, topology_neighbors_row_splits=None,
          external_force=None) -> (pos, vel).

    An upstream TrajectoryModel may be passed directly after loading its weights.
    The adapter deliberately does not load untrusted pickle-based checkpoints.
    """

    def __init__(self, model: Callable[..., Any], *, device: str = "cpu"):
        self.model = model
        self.device = device

    def rollout(
        self, initial: WorldParticleState, *, steps: int,
        force_trajectory: Any = None, detach: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        initial.validate()
        if steps < 0:
            raise ValueError("steps must be >= 0")
        def tensor(value, *, dtype=torch.float32):
            if value is None:
                return None
            return torch.as_tensor(value, dtype=dtype, device=self.device)

        pos = tensor(initial.position)
        vel = tensor(initial.velocity)
        feats = tensor(initial.feats)
        box, box_feats = tensor(initial.box), tensor(initial.box_feats)
        topology_index = tensor(initial.neighbors_index, dtype=torch.long)
        topology_rows = tensor(initial.neighbors_row_splits, dtype=torch.long)
        force_sequence = tensor(force_trajectory)
        if force_sequence is not None and (
            force_sequence.ndim != 3 or force_sequence.shape[1:] != pos.shape
            or force_sequence.shape[0] < steps
        ):
            raise ValueError("force_trajectory must be [steps,N,3] or longer")
        if force_sequence is not None and initial.external_force is not None:
            raise ValueError("pass either external_force or force_trajectory")
        positions, velocities = [pos.detach().cpu().numpy()], [vel.detach().cpu().numpy()]
        for step in range(steps):
            force = force_sequence[step] if force_sequence is not None else tensor(initial.external_force)
            context = torch.no_grad() if detach else torch.enable_grad()
            with context:
                pos, vel = self.model(
                    pos, vel, feats, box, box_feats,
                    topology_neighbors_index=topology_index,
                    topology_neighbors_row_splits=topology_rows,
                    external_force=force,
                )
            if pos.shape != positions[0].shape or vel.shape != velocities[0].shape:
                raise ValueError("WorldParticle model returned unexpected particle shapes")
            if not torch.isfinite(pos).all() or not torch.isfinite(vel).all():
                raise ValueError("WorldParticle model produced nonfinite state")
            positions.append(pos.detach().cpu().numpy())
            velocities.append(vel.detach().cpu().numpy())
        return np.stack(positions), np.stack(velocities)
