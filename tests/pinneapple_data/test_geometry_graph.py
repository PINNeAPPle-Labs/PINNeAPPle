"""Geometry to graph (INT-009): the stencil of a grid and the connectivity of a mesh, as message passing wants."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.cae.model import CellBlock, Mesh
from pinneapple_data.graph import Graph, grid_graph, mesh_graph, to_torch_geometric
from pinneapple_data.schema import AxisSpec, Field, Physics, PhysicsDataBundle
from pinneapple_data.schema.cae import bundle_from_mesh


def grid_bundle(nt: int = 4, nx: int = 6, ny: int = 5) -> PhysicsDataBundle:
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, nt)
    u = np.sin(2 * np.pi * x[None, :, None]) * np.cos(2 * np.pi * y[None, None, :]) * (1 + t[:, None, None])
    return PhysicsDataBundle.from_arrays(
        {"field:u": u.astype(np.float32), "axis:t": t, "axis:x": x, "axis:y": y},
        fields=[Field.from_array("u", u.astype(np.float32), axes=("t", "x", "y"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m"), AxisSpec("y", "m")],
        physics=Physics(), source="grid", license="CC0-1.0", kind="synthetic")


def test_a_grid_becomes_nodes_with_their_stencil():
    graph = grid_graph(grid_bundle(nx=6, ny=5), fields=("u",))
    assert isinstance(graph, Graph)
    assert graph.pos.shape == (30, 2)                     # 6 x 5 nodes
    # 5 edges along x per row of 5 + 4 edges along y per column of 6 = 25 + 24 = 49, both directions each
    assert graph.edge_index.shape == (2, 98)
    assert graph.edge_attr.shape == (98, 3)               # displacement (2) + length (1)
    assert graph.x.shape == (30, 1)
    assert graph.feature_names == ("u",)
    assert "30 nodes, 98 directed edges" in graph.summary()


def test_the_edge_features_carry_a_unit_length_on_a_unit_spaced_grid():
    graph = grid_graph(grid_bundle(nx=3, ny=3))
    lengths = graph.edge_attr[:, -1]
    assert np.allclose(lengths, 0.5)                      # dx = 1/(3-1)


def test_a_wider_stencil_reaches_further():
    narrow = grid_graph(grid_bundle(nx=5, ny=5), step=0, stencil=1)
    wide = grid_graph(grid_bundle(nx=5, ny=5), step=0, stencil=2)
    assert wide.edge_index.shape[1] > narrow.edge_index.shape[1]


def test_the_step_selects_the_snapshot_and_is_checked():
    bundle = grid_bundle(nt=4)
    first, last = grid_graph(bundle, fields=("u",), step=0), grid_graph(bundle, fields=("u",), step=-1)
    assert not np.allclose(first.x, last.x)
    with pytest.raises(IndexError, match="outside the 4 time steps"):
        grid_graph(bundle, step=9)


def test_a_mesh_graph_uses_the_element_connectivity():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                       [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float)
    mesh = Mesh(points=points, blocks=[CellBlock("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))],
                boundary_blocks=[("wall", CellBlock("triangle", np.array([[0, 1, 2]])))],
                point_data={"T": np.linspace(300, 400, 8)}, units={"T": "K"}, source={"format": "hand"})
    bundle = bundle_from_mesh(mesh, source="unit", license="CC0-1.0")
    graph = mesh_graph(bundle)
    assert graph.pos.shape == (8, 3)
    assert graph.edge_index.shape[0] == 2
    # a hexahedron has 12 edges and the boundary triangle adds the one it brings (its other two already exist)
    assert graph.edge_index.shape[1] // 2 == 13
    assert graph.feature_names == ("T",)


def test_pyg_conversion_is_optional_and_named_in_the_error():
    graph = grid_graph(grid_bundle(nx=3, ny=3))
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="torch-geometric"):
            to_torch_geometric(graph)
    else:                                                 # pragma: no cover - only where PyG is installed
        data = to_torch_geometric(graph)
        assert data.num_nodes == 9
