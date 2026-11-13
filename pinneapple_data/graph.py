"""Geometry as a graph: what a mesh-graph network (MeshGraphNet and family) needs, from a bundle.

A grid or a mesh becomes nodes with features and edges with geometry:

* **nodes**: position, and the field values the caller names (or the nodal fields of the bundle), flattened to
  ``(n_nodes, n_features)``;
* **edges**: for a structured grid, the nearest neighbours along each spatial axis at ``t = 0`` (or a chosen step) --
  the stencil a finite-volume mesh has, without inventing long-range links; for an unstructured mesh, the unique
  edges of the element connectivity, which *is* the mesh's own adjacency;
* **edge features**: the displacement vector and its length, which is what a mesh-graph network conditions on.

Everything is returned as numpy arrays in a small :class:`Graph` dataclass: ``torch_geometric`` is optional and is
only touched by :func:`to_torch_geometric`, so a project that already has PyG gets a ``Data`` object and one that does
not still gets the graph.

    >>> from pinneapple_data.graph import grid_graph
    >>> g = grid_graph(bundle, fields=("u",), step=0)
    >>> g.pos.shape, g.edge_index.shape, g.edge_attr.shape
    ((1024, 1), (2, 2046), (2046, 2))
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any

import numpy as np

__all__ = ["Graph", "grid_graph", "mesh_graph", "to_torch_geometric"]


@dataclass
class Graph:
    """Nodes with features, and directed edges with their geometry (both directions, as message passing expects)."""

    pos: np.ndarray                  # (n_nodes, dim)
    x: np.ndarray                    # (n_nodes, n_features): the field values per node
    edge_index: np.ndarray           # (2, n_edges) int, [source, target]
    edge_attr: np.ndarray            # (n_edges, dim + 1): displacement vector and length
    feature_names: tuple[str, ...] = ()
    n_nodes: int = 0
    metadata: dict[str, Any] = _dc_field(default_factory=dict)

    def summary(self) -> str:
        return (f"graph: {self.pos.shape[0]} nodes, {self.edge_index.shape[1]} directed edges, "
                f"{len(self.feature_names)} node feature(s) {self.feature_names}, dimension {self.pos.shape[1]}")

    def to_dict(self) -> dict[str, Any]:
        return {"pos": self.pos, "x": self.x, "edge_index": self.edge_index, "edge_attr": self.edge_attr,
                "feature_names": list(self.feature_names), "n_nodes": self.n_nodes, "metadata": self.metadata}


def _edge_features(pos: np.ndarray, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both directions of every undirected edge, with the displacement and its length as the feature."""
    src_both = np.concatenate([src, dst])
    dst_both = np.concatenate([dst, src])
    delta = pos[dst_both] - pos[src_both]
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.stack([src_both, dst_both]), np.concatenate([delta, length], axis=1)


#: The edges of each VTK element, by its corner nodes. A polyhedron has no fixed connectivity, which is why the
#: face-based meshes go through their own representation; these are the cells with one.
_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "line": ((0, 1),),
    "triangle": ((0, 1), (1, 2), (2, 0)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "tetra": ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    "pyramid": ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)),
    "wedge": ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)),
    "hexahedron": ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                   (0, 4), (1, 5), (2, 6), (3, 7)),
}


def _cell_edges(element: str) -> tuple[tuple[int, int], ...]:
    """The edge list of an element type, falling back to its corner ring for higher-order variants."""
    name = element.lower()
    if name in _EDGES:
        return _EDGES[name]
    for known, edges in _EDGES.items():
        if name.startswith(known):
            return edges
    raise KeyError(f"no edge table for element type '{element}': the graph needs the element's own edges, and "
                   f"guessing them would invent connectivity (known: {', '.join(sorted(_EDGES))})")


def grid_graph(bundle: Any, *, fields: Sequence[str] = (), step: int = 0,
               stencil: int = 1) -> Graph:
    """The graph of a structured bundle at one time step: neighbours along each spatial axis."""
    axes = [a for a in bundle.axes if a != "t"]
    if not axes:
        raise ValueError("this bundle has no spatial axis: a structured thing to make a graph from is a grid")
    if "t" in bundle.axes:
        n_t = int(bundle.axis_array("t").shape[0])
        if not -n_t <= step < n_t:
            raise IndexError(f"step {step} is outside the {n_t} time steps")
    coords = [np.asarray(bundle.axis_array(a), dtype=np.float64) for a in axes]
    mesh = np.meshgrid(*coords, indexing="ij") if len(coords) > 1 else [coords[0]]
    pos = np.stack([m.ravel() for m in mesh], axis=1) if len(coords) > 1 else coords[0][:, None]
    shape = tuple(len(c) for c in coords)

    index = np.arange(pos.shape[0]).reshape(shape)
    src, dst = [], []
    for axis, length in enumerate(shape):
        for offset in range(1, stencil + 1):
            left = np.take(index, np.arange(0, length - offset), axis=axis)
            right = np.take(index, np.arange(offset, length), axis=axis)
            src.append(left.ravel())
            dst.append(right.ravel())
    if not src:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, pos.shape[1] + 1))
    else:
        edge_index, edge_attr = _edge_features(pos, np.concatenate(src), np.concatenate(dst))

    names = list(fields)
    if not names:
        names = [f.name for f in bundle.manifest.fields.variables if f.location == "nodal"]
    columns = []
    for name in names:
        values = np.asarray(bundle.field_array(name))
        values = values[step] if values.ndim > len(shape) - 1 else values
        columns.append(np.asarray(values).reshape(pos.shape[0], -1))
    x = np.concatenate(columns, axis=1) if columns else np.zeros((pos.shape[0], 0))
    return Graph(pos=pos, x=x, edge_index=np.asarray(edge_index, dtype=np.int64), edge_attr=edge_attr,
                 feature_names=tuple(names), n_nodes=int(pos.shape[0]),
                 metadata={"kind": "grid", "axes": axes, "shape": list(shape), "step": step})


def mesh_graph(bundle: Any) -> Graph:
    """The graph of an unstructured bundle: the element connectivity's own edges, once each."""
    spec = bundle.manifest.mesh
    if spec.nodes is None:
        raise ValueError("this bundle has no node coordinates to place a graph on")
    pos = np.asarray(bundle.array(spec.nodes.key), dtype=np.float64)
    edges: set[tuple[int, int]] = set()
    for _name, etype, ref, _boundary in spec.all_cells():
        conn = np.asarray(bundle.array(ref.key), dtype=np.int64)
        table = _cell_edges(etype)
        for row in conn:
            for a, b in table:
                if a < row.shape[0] and b < row.shape[0]:
                    i, j = int(row[a]), int(row[b])
                    edges.add((i, j) if i < j else (j, i))
    if edges:
        pairs = np.asarray(sorted(edges), dtype=np.int64)
        edge_index, edge_attr = _edge_features(pos, pairs[:, 0], pairs[:, 1])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, pos.shape[1] + 1))
    names = [f.name for f in bundle.manifest.fields.variables if f.location == "nodal"]
    columns = [np.asarray(bundle.field_array(n)).reshape(pos.shape[0], -1) for n in names]
    x = np.concatenate(columns, axis=1) if columns else np.zeros((pos.shape[0], 0))
    return Graph(pos=pos, x=x, edge_index=edge_index, edge_attr=edge_attr, feature_names=tuple(names),
                 n_nodes=int(pos.shape[0]), metadata={"kind": "mesh", "dimension": int(spec.dimension)})


def to_torch_geometric(graph: Graph):
    """A ``torch_geometric.data.Data`` of the same graph (needs PyG; nothing else in this module imports it)."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:                                   # pragma: no cover - optional dependency
        raise ImportError("to_torch_geometric needs torch and torch_geometric "
                          "(pip install torch-geometric)") from exc
    return Data(x=torch.as_tensor(graph.x, dtype=torch.float32),
                pos=torch.as_tensor(graph.pos, dtype=torch.float32),
                edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
                edge_attr=torch.as_tensor(graph.edge_attr, dtype=torch.float32),
                feature_names=graph.feature_names)
