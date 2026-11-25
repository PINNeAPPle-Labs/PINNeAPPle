"""Utility functions for graph neural network operations."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import GraphBatch

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-add with shared-graph batching.

    src: (B, E, F) — batched edge features
    index: (E,)    — destination node indices [0..N-1], shared across batch
    returns: (B, N, F) where out[:, index[e], :] += src[:, e, :]
    """
    B, E, F = src.shape
    out = torch.zeros((B, dim_size, F), device=src.device, dtype=src.dtype)
    out.index_add_(1, index, src)
    return out

# Alias kept for backward compatibility with existing imports
scatter_add_batched = scatter_add


def scatter_add_flat(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Flattened variable-graph batching.
    src: (E_total, F)
    index: (E_total,) destination indices [0..N_total-1]
    returns: (N_total, F) where out[index[e], :] += src[e, :]
    """
    F = src.shape[-1]
    out = torch.zeros((dim_size, F), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def flatten_graph_batch(
    x_list: list[torch.Tensor],
    edge_index_list: list[torch.Tensor],
    pos_list: list[torch.Tensor] | None = None,
    edge_attr_list: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """
    Turns per-sample graphs into a single flattened graph with node index offsets.

    x_list:        list of (N_b, Fin)
    edge_index_list: list of (2, E_b) with indices in [0..N_b-1]
    pos_list:      list of (N_b, P) or None
    edge_attr_list:list of (E_b, Fe) or None

    Returns:
      x_flat: (N_total, Fin)
      edge_index_flat: (2, E_total) with offsets applied
      pos_flat: (N_total, P) or None
      edge_attr_flat: (E_total, Fe) or None
      ptr: (B+1,) prefix sums over nodes (ptr[0]=0, ptr[b+1]=sum_{i<=b} N_i)
    """
    device = x_list[0].device
    dtype = x_list[0].dtype

    n_per = torch.tensor([x.shape[0] for x in x_list], device=device, dtype=torch.long)
    ptr = torch.zeros((len(x_list) + 1,), device=device, dtype=torch.long)
    ptr[1:] = torch.cumsum(n_per, dim=0)

    x_flat = torch.cat(x_list, dim=0)

    if pos_list is not None:
        pos_flat = torch.cat(pos_list, dim=0)
    else:
        pos_flat = None

    # Apply node offsets to each edge_index
    edge_chunks = []
    for b, ei in enumerate(edge_index_list):
        if ei.numel() == 0:
            continue
        off = ptr[b].item()
        edge_chunks.append(ei + off)
    edge_index_flat = torch.cat(edge_chunks, dim=1) if len(edge_chunks) > 0 else torch.zeros((2, 0), device=device, dtype=torch.long)

    if edge_attr_list is not None:
        edge_attr_flat = torch.cat(edge_attr_list, dim=0) if len(edge_attr_list) > 0 else torch.zeros((0, edge_attr_list[0].shape[-1]), device=device, dtype=dtype)
    else:
        edge_attr_flat = None

    return x_flat, edge_index_flat, pos_flat, edge_attr_flat, ptr


_GRAPH_KEYS = {"x": ("x", "node_features"), "edge_index": ("edge_index",), "edge_attr": ("edge_attr", "edge_features"),
               "pos": ("pos",), "mask": ("mask",)}


def _graph_fields(g) -> dict:
    """The tensor fields of one per-sample graph, read from a :class:`~.base.GraphBatch`, a mapping or an object with
    the same attribute names. ``x``/``edge_attr`` lose a leading batch dimension of one, so ``(1, N, F)`` from a
    hand-built ``GraphBatch`` and ``(N, F)`` are both accepted."""
    def read(names):
        for name in names:
            value = g.get(name) if isinstance(g, Mapping) else getattr(g, name, None)
            if value is not None:
                return value
        return None

    out = {key: read(names) for key, names in _GRAPH_KEYS.items()}
    for key in ("x", "pos", "mask", "edge_attr"):
        value = out.get(key)
        if value is not None and value.dim() == 3 and value.shape[0] == 1:
            out[key] = value[0]
    if out["x"] is None or out["edge_index"] is None:
        raise KeyError("every per-sample graph needs 'x' (or 'node_features') and 'edge_index'")
    return out


def stack_graphs(graphs) -> tuple[GraphBatch, torch.Tensor]:
    """One flat graph from per-sample graphs of *different* sizes — the batching a mesh solver needs.

    The graphs are concatenated into a single disjoint union: nodes are offset per sample and the edge indices carry
    the offsets, so message passing on the union is exactly message passing on each graph separately (no edge crosses
    a sample boundary) while the whole batch runs in one call. That is the same trick
    :func:`flatten_graph_batch` does, wrapped around the fields of a graph object.

    Parameters
    ----------
    graphs : sequence of ``GraphBatch`` / mapping / object with the same fields. ``x`` is ``(N, F)`` or
        ``(1, N, F)``; ``edge_index`` is ``(2, E)`` with indices in ``[0, N)``; ``edge_attr`` / ``pos`` / ``mask``
        are optional and follow the same rule.

    Returns
    -------
    (batch, ptr)
        ``batch`` is a :class:`~.base.GraphBatch` with ``x`` reshaped to ``(1, N_total, F)`` and the offsets applied
        to ``edge_index`` — the shared-topology shape every model in this family already takes — and ``ptr`` is the
        ``(B+1,)`` node prefix sum, so sample ``b`` is ``batch.x[0, ptr[b]:ptr[b+1]]``.

    Examples
    --------
    >>> from pinneapple_neural.architectures.graphnn.base import GraphBatch
    >>> g1 = GraphBatch(x=torch.rand(4, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    >>> g2 = GraphBatch(x=torch.rand(6, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    >>> flat, ptr = stack_graphs([g1, g2])
    >>> flat.x.shape, ptr.tolist()
    (torch.Size([1, 10, 2]), [0, 4, 10])
    """
    fields = [_graph_fields(g) for g in graphs]
    if not fields:
        raise ValueError("stack_graphs needs at least one graph")
    has_attr = [f["edge_attr"] for f in fields]
    x_flat, edge_index_flat, pos_flat, edge_attr_flat, ptr = flatten_graph_batch(
        x_list=[f["x"] for f in fields],
        edge_index_list=[f["edge_index"].long() for f in fields],
        pos_list=None if any(p is None for p in (f["pos"] for f in fields)) else [f["pos"] for f in fields],
        edge_attr_list=None if any(a is None for a in has_attr) else has_attr,
    )
    masks = [f["mask"] for f in fields]
    mask_flat = torch.cat(masks, dim=0) if all(m is not None for m in masks) else None
    batch = GraphBatch(
        x=x_flat.unsqueeze(0),
        edge_index=edge_index_flat,
        pos=None if pos_flat is None else pos_flat.unsqueeze(0),
        edge_attr=None if edge_attr_flat is None else edge_attr_flat.unsqueeze(0),
        mask=None if mask_flat is None else mask_flat.unsqueeze(0),
    )
    return batch, ptr


def split_flat(y: torch.Tensor, ptr: torch.Tensor, *, pad: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Undo :func:`stack_graphs` on an output: ``y`` is ``(1, N_total, F)`` (or ``(N_total, F)``).

    Returns ``(padded, list)`` where ``padded`` is ``(B, N_max, F)`` zero-filled past each sample's own node count and
    ``list[b]`` is the ``(N_b, F)`` block of sample ``b``. Zero padding is what the ``mask`` returned by
    :func:`stack_graphs` and :class:`~.base.GraphBatch` already mean, so a padded output can be scored with it.
    """
    if y.dim() == 3:
        y = y[0]
    blocks = [y[ptr[b].item():ptr[b + 1].item()] for b in range(ptr.numel() - 1)]
    if not pad:
        return y, blocks
    n_max = max((b.shape[0] for b in blocks), default=0)
    out = y.new_zeros((len(blocks), n_max, y.shape[-1]))
    for b, block in enumerate(blocks):
        out[b, :block.shape[0]] = block
    return out, blocks