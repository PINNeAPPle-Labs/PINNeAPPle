from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from pinneapple_neural.architectures.registry import ModelSpec


class GNNAdapter:
    """Adapter for graph/mesh-based models.

    Batch conventions
    -----------------
    ``model.forward_batch``
        tried first: a model that knows what its own batch *means* (``MeshDynamicsMGN`` takes a mesh and a
        trajectory) gets the raw dict and does its own batching.
    ``graph`` / ``graphs``
        One graph object the model understands, or a list of them — one per sample, of **different sizes**. A list is
        run as its disjoint union (:func:`~pinneapple_neural.architectures.graphnn.utils.stack_graphs`): node indices
        are offset per sample and no edge crosses a sample boundary, so a single call gives exactly the per-sample
        answers, and the output comes back padded to ``(B, N_max, out)`` with the validity ``mask`` in ``extras``.
    ``x_list`` / ``edge_index_list`` (+ optional ``pos_list``, ``edge_attr_list``)
        The same per-sample variable-size batch in loose tensors instead of graph objects.
    ``x`` + ``edge_index``
        One graph shared by the whole batch (the historical fixed-topology case, what ``MeshGraphNet`` documents).
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return (spec.input_kind or "") in ("graph", "mesh", "graph_trajectory") or spec.family in ("graphnn",)

    def forward_batch(self, model: nn.Module, batch: dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            try:
                return model.forward_batch(batch)  # type: ignore[attr-defined]
            except KeyError:
                # The model has a forward_batch but does not know these keys (a per-sample ``graphs`` list handed to a
                # model that only documents the shared-topology ``x``/``edge_index``): fall through to the generic
                # variable-size path below, which raises its own clear error if the batch is wrong too.
                pass

        graphs = _first(batch, ("graphs", "graph_list"))
        if graphs is not None:
            return self.variable_forward(model, list(graphs), batch)

        x_list = _first(batch, ("x_list", "node_features_list"))
        if x_list is not None:
            n = len(x_list)
            edge_lists = _first(batch, ("edge_index_list", "edge_lists")) or []
            edge_attrs = _first(batch, ("edge_attr_list", "edge_features_list")) or [None] * n
            positions = _first(batch, ("pos_list",)) or [None] * n
            if len(edge_lists) != n:
                raise ValueError(f"got {len(edge_lists)} edge_index_list entries for {n} x_list entries")
            return self.variable_forward(
                model,
                [{"x": x, "edge_index": ei, "edge_attr": ea, "pos": p}
                 for x, ei, ea, p in zip(x_list, edge_lists, edge_attrs, positions, strict=True)],
                batch,
            )

        g = _first(batch, ("graph", "mesh"))
        if g is not None:
            return model(g)

        # Fallback: try common PyG style args
        if "x" in batch and "edge_index" in batch:
            try:
                return model(batch["x"], batch["edge_index"])
            except TypeError:
                pass

        raise KeyError("GNNAdapter requires model.forward_batch, batch['graph'|'graphs'], "
                       "('x_list','edge_index_list') or ('x','edge_index').")

    @staticmethod
    def variable_forward(model: nn.Module, graphs: Sequence[Any], batch: dict[str, Any] | None = None) -> Any:
        """Run per-sample graphs of different sizes as one disjoint union, and split the output back per sample.

        Works for any model in this family that takes a shared-topology :class:`~...graphnn.base.GraphBatch`: the
        union has the same message-passing semantics sample by sample. Returns a
        :class:`~...graphnn.base.GraphOutput` whose ``y`` is padded to ``(B, N_max, out)``, with ``extras["y_list"]``
        holding the exact per-sample blocks, ``extras["ptr"]`` the node prefix sums and ``extras["mask"]`` the
        validity mask; ``losses["mse"]`` is filled when the batch carries ``y_true`` (padded, or one per sample).
        """
        from pinneapple_neural.architectures.graphnn.base import GraphOutput
        from pinneapple_neural.architectures.graphnn.utils import split_flat, stack_graphs

        flat, ptr = stack_graphs(graphs)
        out = model(flat)
        y = out.y if hasattr(out, "y") else out
        padded, blocks = split_flat(y, ptr)
        mask = torch.zeros(padded.shape[:2], device=padded.device, dtype=padded.dtype)
        for b, block in enumerate(blocks):
            mask[b, : block.shape[0]] = 1.0
        losses = dict(getattr(out, "losses", {}) or {})
        y_true = _first(batch or {}, ("y_true", "y"))
        if y_true is not None:
            if torch.is_tensor(y_true) and tuple(y_true.shape) == tuple(padded.shape):
                m = mask[..., None].to(padded.dtype)
                losses["mse"] = (((padded - y_true) ** 2) * m).sum() / (m.sum() * padded.shape[-1]).clamp_min(1.0)
            elif isinstance(y_true, (list, tuple)) and len(y_true) == len(blocks):
                losses["mse"] = padded.new_tensor(sum(float(((block - target) ** 2).mean()) for block, target
                                                      in zip(blocks, y_true, strict=True)) / len(blocks))
            else:
                raise ValueError(f"y_true of shape {tuple(getattr(y_true, 'shape', ())) or '(unknown)'} does not "
                                 f"match the padded prediction {tuple(padded.shape)}: pass one target per sample or "
                                 f"the padded stack")
            losses["total"] = losses["mse"]
        extras = dict(getattr(out, "extras", {}) or {})
        extras.update({"y_list": blocks, "ptr": ptr, "mask": mask})
        return GraphOutput(y=padded, losses=losses, extras=extras)


def _first(batch: dict[str, Any], names: Sequence[str]):
    """The first key of ``names`` present (and not ``None``) in ``batch``."""
    for name in names:
        if batch.get(name) is not None:
            return batch[name]
    return None
