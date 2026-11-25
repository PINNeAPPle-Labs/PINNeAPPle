"""Comparing :class:`~pinneapple_neural.architectures.graphnn.mgn_dynamics.MeshDynamicsMGN` with PhysicsNeMo's MGN (#231).

The Dynamics MGN here reproduces the training/inference recipe of Pfaff et al. (ICLR 2021) and of the PhysicsNeMo
``vortex_shedding_mgn`` example. "Reproduces" is a claim, so this module carries the comparison: both models are driven
through the *same* free-running rollout on the *same* mesh with the *same* field conventions, and the numbers come
back per model plus the agreement between them.

Field conventions of the reference example, which both sides use here:

=========================  ============================================================
node features              ``[velocity_t, one-hot(node type)]``, normalised
edge features              ``[rel. position, |rel. position|]``, normalised, static
node outputs               ``[Δvelocity, pressure_{t+1}]``
free-running rollout       non-NORMAL nodes (walls, inlet, obstacle) clamped each step
=========================  ============================================================

``physicsnemo`` is optional. Everything here works without it: :func:`compare_rollouts` takes the other model as a
plain step callable, so the comparison runs against any implementation, and :func:`native_mgn_step` is the only entry
that needs the package — it raises the bridge's actionable ``ImportError`` when it is missing. When PhysicsNeMo is
absent the comparison still reports the PINNeAPPle side against the reference trajectory, with ``available: False``
and the reason, so "not installed" is never mistaken for "agrees".
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshDynamicsMGN, MeshGraph

from .adapters import is_physicsnemo_available, require_physicsnemo_model

StepFn = Callable[[torch.Tensor, MeshGraph], torch.Tensor]
"""One step of another model: ``(velocity (N, vel_dim), mesh) -> next velocity (N, vel_dim)``."""


def native_availability() -> dict[str, Any]:
    """Whether PhysicsNeMo can be used here, and the reason when it cannot.

    Returns ``{"available": bool, "provider": "physicsnemo" | None, "reason": str}`` — the shape the comparison
    records, so an absent optional dependency is a declared fact in the report and not a silent zero.
    """
    if is_physicsnemo_available():
        return {"available": True, "provider": "physicsnemo", "reason": ""}
    return {"available": False, "provider": None,
            "reason": "physicsnemo is not installed (pip install nvidia-physicsnemo): the comparison reports the "
                      "PINNeAPPle side only"}


def normalised_fields(vel: torch.Tensor, graph: MeshGraph, num_node_types: int,
                      normaliser: Mapping[str, Sequence[float]] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference example's node and edge features for one state: ``(node (N, vel_dim+types), edge (E, F))``.

    ``normaliser`` maps ``"vel"`` and ``"edge"`` to ``(mean, std)`` pairs (the statistics of the training run); without
    it the fields are raw. This is the single place the two implementations' inputs are built, so a difference in the
    comparison cannot come from two slightly different feature definitions.
    """
    def _apply(key: str, x: torch.Tensor) -> torch.Tensor:
        if not normaliser or key not in normaliser:
            return x
        mean, std = (torch.as_tensor(v, dtype=x.dtype, device=x.device) for v in normaliser[key])
        return (x - mean) / std

    node = torch.cat([_apply("vel", vel),
                      torch.nn.functional.one_hot(graph.node_type, num_node_types).to(vel.dtype)], dim=-1)
    return node, _apply("edge", graph.edge_attr)


def native_mgn_step(model: Any, *, num_node_types: int = 4, call: Callable[..., torch.Tensor] | None = None,
                    normaliser: Mapping[str, Sequence[float]] | None = None) -> StepFn:
    """Wrap a trained PhysicsNeMo MeshGraphNet as a step callable for :func:`compare_rollouts`.

    ``call(node_features, edge_features, edge_index) -> delta_velocity`` is the hop into PhysicsNeMo's own forward:
    its signature and its graph object differ between releases (DGL graph in 1.x, PyG/tensor dict later), so the
    default tries the documented ``model(node_features=..., edge_features=..., graph=...)`` and then the positional
    form, and refuses with a clear message if neither is accepted — pass ``call`` for your version rather than
    guessing. ``normaliser`` inverts the training statistics on the way out (``"dvel"``), so the returned tensor is a
    next velocity in physical units, as this library's models produce.
    """
    require_physicsnemo_model(model)

    def _default_call(node: torch.Tensor, edge: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        attempts = (
            lambda: model(node_features=node, edge_features=edge, graph=edge_index),
            lambda: model(node, edge, edge_index),
        )
        last: Exception | None = None
        for attempt in attempts:
            try:
                return attempt()
            except (TypeError, KeyError, AttributeError) as e:      # a signature this release does not have
                last = e
        raise TypeError("the PhysicsNeMo MeshGraphNet in this release does not accept the documented "
                        "(node_features, edge_features, graph) call: pass call=... with your version's signature"
                        f"). Last error: {last}")
    forward = call or _default_call

    @torch.no_grad()
    def step(vel: torch.Tensor, graph: MeshGraph) -> torch.Tensor:
        node, edge = normalised_fields(vel, graph, num_node_types, normaliser)
        out = forward(node[None], edge[None], graph.edge_index)
        out = out.y if hasattr(out, "y") else out
        dv = out[0] if out.dim() == 3 else out
        if normaliser and "dvel" in normaliser:
            mean, std = (torch.as_tensor(v, dtype=dv.dtype, device=dv.device) for v in normaliser["dvel"])
            dv = dv[:, : dv.shape[-1]] * std + mean
        return vel + dv

    return step


def native_rollout(step: StepFn, vel_traj: torch.Tensor, graph: MeshGraph, n_steps: int,
                   free_types: Sequence[int] = (0,)) -> torch.Tensor:
    """Free-running rollout driven by an external ``step`` — the same clamp rule as :meth:`MeshDynamicsMGN.rollout`.

    Nodes whose type is not in ``free_types`` are overwritten each step by the reference value ``vel_traj[t + 1]``, so
    the comparison isolates the learned dynamics from the boundary data both models are given.
    """
    free = torch.zeros_like(graph.node_type, dtype=torch.bool)
    for t in free_types:
        free |= graph.node_type == t
    free = free.unsqueeze(-1)
    vel = vel_traj[0]
    out = [vel]
    for t in range(n_steps):
        vel = torch.where(free, step(vel, graph), vel_traj[t + 1])
        out.append(vel)
    return torch.stack(out)


def _rmse(pred: torch.Tensor, ref: torch.Tensor, free: torch.Tensor) -> float:
    n = min(pred.shape[0], ref.shape[0])
    return float(((pred[1:n][:, free] - ref[1:n][:, free]) ** 2).mean().sqrt().detach())


def compare_rollouts(model: MeshDynamicsMGN, vel_traj: torch.Tensor, graph: MeshGraph, *, n_steps: int,
                     native_step: StepFn | None = None, free_types: Sequence[int] = (0,),
                     normal_type: int = 0) -> dict[str, Any]:
    """Roll both models out from the same state and report each one's error and their agreement.

    ``native_step`` is the other model's one-step callable (:func:`native_mgn_step` for a PhysicsNeMo MGN, or any
    ``(velocity, mesh) -> next velocity``); ``None`` means the other model is not available — the report says so
    through :func:`native_availability` instead of inventing a number.

    Returns ``{"n_steps", "free_nodes", "pinn_rmse", "pinn_agreement_with_native", "native_rmse", "native"}``:
    ``*_rmse`` is the velocity RMSE against the reference trajectory on the free nodes over the horizon (is the model
    good?), and ``pinn_agreement_with_native`` is the relative L2 between the two rollouts (do they implement the same
    recipe?).
    """
    free = (graph.node_type == normal_type)
    n = min(n_steps, vel_traj.shape[0] - 1)
    pred, _p = model.rollout(vel_traj, graph, n, free_types=tuple(free_types))
    report: dict[str, Any] = {
        "n_steps": int(n),
        "free_nodes": int(free.sum()),
        "pinn_rmse": _rmse(pred, vel_traj[: n + 1], free),
        "pinn_agreement_with_native": None,
        "native_rmse": None,
        "native": native_availability(),
    }
    if native_step is None:
        return report
    native = native_rollout(native_step, vel_traj, graph, n, free_types=tuple(free_types))
    report["native_rmse"] = _rmse(native, vel_traj[: n + 1], free)
    denom = float(native[:, free].norm().detach()) or 1.0
    report["pinn_agreement_with_native"] = float((pred - native)[:, free].norm().detach()) / denom
    report["native"]["provider"] = report["native"]["provider"] or "callable"
    return report
