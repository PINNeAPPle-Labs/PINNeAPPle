import pytest
import torch

from pinneapple_neural.architectures.graphnn.mgn_dynamics import (
    MeshDynamicsMGN,
    MeshGraph,
    Normalizer,
    flatten_meshes,
    train_dynamics_batch,
    triangles_to_edges,
)


def _mesh(n=30):
    torch.manual_seed(0)
    pos = torch.rand(n, 2)
    from scipy.spatial import Delaunay
    cells = torch.from_numpy(Delaunay(pos.numpy()).simplices)
    nt = torch.zeros(n, dtype=torch.long)
    nt[:4] = 3
    return MeshGraph.from_mesh(pos, cells, nt)


def _three_meshes():
    """Three meshes of different node counts and triangulations — the point of the variable-size batching."""
    from scipy.spatial import Delaunay
    out = []
    for size, seed in ((24, 1), (40, 2), (18, 3)):
        g = torch.Generator().manual_seed(seed)
        pos = torch.rand(size, 2, generator=g)
        cells = torch.from_numpy(Delaunay(pos.numpy()).simplices)
        nt = torch.zeros(size, dtype=torch.long)
        nt[:3] = 3                                            # a wall: stays clamped in the rollout
        out.append(MeshGraph.from_mesh(pos, cells, nt))
    return out


def test_triangles_to_edges_bidirectional_unique():
    ei = triangles_to_edges(torch.tensor([[0, 1, 2], [1, 2, 3]]))
    pairs = set(map(tuple, ei.t().tolist()))
    assert len(pairs) == ei.shape[1]
    assert all((b, a) in pairs for a, b in pairs)
    assert (1, 2) in pairs and (3, 1) in pairs


def test_normalizer_roundtrip():
    x = torch.randn(100, 3) * 5 + 2
    n = Normalizer(3).fit([x])
    assert torch.allclose(n.inverse(n(x)), x, atol=1e-4)
    assert torch.allclose(n(x).mean(0), torch.zeros(3), atol=1e-4)


def test_loss_backward_and_rollout_clamps_boundary():
    g = _mesh()
    N = g.node_type.numel()
    v = torch.randn(6, N, 2)
    p = torch.randn(6, N, 1)
    m = MeshDynamicsMGN(hidden_dim=16, n_message_passing=2)
    m.fit_stats([v], [p], [g])
    loss = m.loss(v[0], v[1], p[1], g, noise_std=0.02)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(q.grad is not None for q in m.net.parameters())
    pred, ps = m.rollout(v, g, 4)
    assert pred.shape == (5, N, 2) and ps.shape == (4, N, 1)
    assert torch.equal(pred[1:, :4], v[1:5, :4])  # non-NORMAL nodes follow prescribed values


def test_noise_target_consistency():
    """With zero-weight-effect check: v_in + dv_target == v_{t+1} even with noise."""
    g = _mesh()
    N = g.node_type.numel()
    v0, v1 = torch.randn(N, 2), torch.randn(N, 2)
    m = MeshDynamicsMGN(hidden_dim=8, n_message_passing=1, out_p=False)
    m.fit_stats([torch.stack([v0, v1])], None, [g])
    torch.manual_seed(1)
    noise = torch.randn_like(v0) * 0.1
    dv = m.dvel_norm(v1 - (v0 + noise))
    assert torch.allclose(v0 + noise + m.dvel_norm.inverse(dv), v1, atol=1e-5)


def _load_example(name):
    import importlib.util
    import os
    import sys
    d = os.path.join(os.path.dirname(__file__), "..", "examples", "meshgraphnet")
    sys.path.insert(0, os.path.abspath(d))
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reduced_synthetic_diffusion_beats_frozen_baseline():
    """A small run of examples/meshgraphnet/01_synthetic_diffusion.py: the MGN rollout beats copying the state."""
    pytest.importorskip("scipy")
    ex = _load_example("01_synthetic_diffusion")
    m = ex.run(n_train=12, n_test=3, n_pts=100, steps=1200, rollout=10, log_every=10_000)   # ~12 s on CPU
    assert m["rollout_rmse"] < 0.95 * m["frozen_ic_baseline_rmse"], m                           # measured 0.067 vs 0.081


def test_tfrecord_reader_on_a_generated_file(tmp_path):
    """pinneapple_data.datasets.deepmind_mgn reads DeepMind meshgraphnets records (a tiny file written by the test)."""
    import json

    import numpy as np
    tfw = pytest.importorskip("tfrecord.writer")
    from pinneapple_data.datasets import deepmind_mgn as common
    T, N = 3, 4
    traj = {"mesh_pos": np.random.rand(1, N, 2).astype("float32"),
            "cells": np.array([[[0, 1, 2], [1, 2, 3]]], dtype="int32"),
            "node_type": np.array([[[0], [4], [5], [6]]], dtype="int32"),
            "velocity": np.random.rand(T, N, 2).astype("float32"),
            "pressure": np.random.rand(T, N, 1).astype("float32")}
    feats = {k: {"type": "static" if v.shape[0] == 1 else "dynamic", "shape": [v.shape[0], -1, v.shape[-1]],
                 "dtype": str(v.dtype)} for k, v in traj.items()}
    (tmp_path / "meta.json").write_text(json.dumps({"field_names": list(traj), "features": feats}))
    w = tfw.TFRecordWriter(str(tmp_path / "valid.tfrecord"))
    for _ in range(2):
        w.write({k: (v.tobytes(), "byte") for k, v in traj.items()})
    w.close()
    out = common.read_trajectories(tmp_path, "valid", max_traj=5)
    assert len(out) == 2 and out[0]["velocity"].shape == (T, N, 2) and out[0]["cells"].shape == (2, 3)
    g, v, p = common.cylinder_flow_to_tensors(out[0], common.CYLINDER_TYPE_MAP)
    assert v.shape == (T, N, 2) and p.shape == (T, N, 1) and g.node_type.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# #231 — registry entry, variable-size batching at B > 1, Arena/GNNAdapter path
# ---------------------------------------------------------------------------

def test_registry_entry_carries_its_own_capabilities():
    """#231: MeshDynamicsMGN is discoverable, and what it declares is its own, not the graphnn family default."""
    import pinneapple_neural.architectures  # noqa: F401  registers every family into the global registry
    from pinneapple_neural.architectures.graphnn.registry import GraphCatalog
    from pinneapple_neural.architectures.registry import ModelRegistry

    catalog = GraphCatalog()
    for alias in ("mesh_dynamics_mgn", "mesh_dynamics", "mgn_dynamics"):
        assert catalog.get(alias) is MeshDynamicsMGN
    spec = ModelRegistry.spec("mesh_dynamics_mgn")
    assert spec.family == "graphnn"
    assert spec.input_kind == "graph_trajectory"
    assert spec.expects == ["mesh", "trajectory"] and spec.predicts == ["rollout"]
    # the rest of the family keeps the generic graph/u declaration (one canonical name per class in the global
    # registry: the aliases live in the family catalog)
    assert ModelRegistry.spec("mesh_graph_net").predicts == ["u"]
    assert ModelRegistry.spec("mesh_graph_net").expects == ["graph"]
    model = catalog.build("mesh_dynamics_mgn", vel_dim=2, num_node_types=4, edge_dim=3, hidden_dim=8,
                          n_message_passing=1)
    assert isinstance(model, MeshDynamicsMGN) and model.vel_dim == 2


def test_flatten_meshes_offsets_edges_and_reports_ptr():
    graphs = _three_meshes()
    edge_index, edge_attr, node_type, ptr = flatten_meshes(graphs)
    n = [g.node_type.numel() for g in graphs]
    assert ptr.tolist() == [0, n[0], n[0] + n[1], sum(n)]
    assert node_type.shape[0] == sum(n) and edge_attr.shape[0] == edge_index.shape[1]
    assert int(edge_index.max()) < sum(n) and int(edge_index.min()) >= 0
    # every sample's edges stay inside its own node block: no edge crosses a sample boundary
    for b, g in enumerate(graphs):
        block = [e for e in range(edge_index.shape[1])
                 if (ptr[b] <= edge_index[0, e] < ptr[b + 1]) or (ptr[b] <= edge_index[1, e] < ptr[b + 1])]
        assert len(block) == g.edge_index.shape[1]
        for e in block:
            assert ptr[b] <= edge_index[0, e] < ptr[b + 1] and ptr[b] <= edge_index[1, e] < ptr[b + 1]


def test_batched_step_equals_per_sample_step_at_b3():
    """The batching is exact: one union pass at B = 3 gives the per-sample answers of three different meshes."""
    graphs = _three_meshes()
    torch.manual_seed(4)
    vels = [torch.randn(g.node_type.numel(), 2) for g in graphs]
    model = MeshDynamicsMGN(hidden_dim=16, n_message_passing=2).eval()
    with torch.no_grad():
        batch_vels, batch_ps = model.step_batch(vels, graphs)
        singles = [model.step(v, g) for v, g in zip(vels, graphs, strict=True)]
    for b in range(3):
        assert torch.allclose(batch_vels[b], singles[b][0], atol=1e-5)
        assert torch.allclose(batch_ps[b], singles[b][1], atol=1e-5)


def test_loss_batch_is_the_node_weighted_mean_of_the_single_losses():
    """B > 1 is not a different loss: with the noise off, it is the mean the B = 1 losses average to."""
    graphs = _three_meshes()
    torch.manual_seed(5)
    trajs = [torch.randn(3, g.node_type.numel(), 2) for g in graphs]
    pressures = [torch.randn(3, g.node_type.numel(), 1) for g in graphs]
    model = MeshDynamicsMGN(hidden_dim=16, n_message_passing=2)
    model.fit_stats(trajs, pressures, graphs)
    batched = model.loss_batch([t[0] for t in trajs], [t[1] for t in trajs], [p[1] for p in pressures], graphs,
                               noise_std=0.0)
    single = torch.stack([model.loss(t[0], t[1], p[1], g, noise_std=0.0)
                          for t, p, g in zip(trajs, pressures, graphs, strict=True)])
    n = torch.tensor([g.node_type.numel() for g in graphs], dtype=single.dtype)
    weighted = (single * n).sum() / n.sum()          # the batch mean is over all nodes of all samples
    assert torch.allclose(batched, weighted, atol=1e-5), (float(batched.detach()), float(weighted.detach()))


def test_rollout_batch_matches_rollout_and_clamps_each_mesh():
    graphs = _three_meshes()
    torch.manual_seed(6)
    trajs = [torch.randn(5, g.node_type.numel(), 2) for g in graphs]
    model = MeshDynamicsMGN(hidden_dim=16, n_message_passing=2).eval()
    with torch.no_grad():
        vels, ps = model.rollout_batch(trajs, graphs, 4, free_types=(0,))
        single, single_p = model.rollout(trajs[1], graphs[1], 4, free_types=(0,))
    assert torch.allclose(vels[1], single, atol=1e-5) and torch.allclose(ps[1], single_p, atol=1e-5)
    for b, g in enumerate(graphs):
        assert vels[b].shape == (5, g.node_type.numel(), 2)
        assert torch.equal(vels[b][1:, :3], trajs[b][1:5, :3])       # the wall (type 3) follows the reference


def test_train_dynamics_batch_learns_with_b3_in_few_seconds():
    """A short B = 3 run on three different meshes reduces the loss it reports — the batching trains, not only runs."""
    graphs = _three_meshes()
    torch.manual_seed(7)
    data = [(g, torch.randn(4, g.node_type.numel(), 2), torch.randn(4, g.node_type.numel(), 1)) for g in graphs]
    model = MeshDynamicsMGN(hidden_dim=16, n_message_passing=2)
    model.fit_stats([d[1] for d in data], [d[2] for d in data], graphs)
    hist = train_dynamics_batch(model, data, steps=80, lr=3e-3, lr_final=3e-4, batch_size=3, log_every=20)
    assert len(hist) == 4 and hist[-1] < hist[0], hist


def test_gnn_adapter_runs_a_variable_size_batch_and_matches_per_sample():
    """#231 (2) and (3): the Arena's adapter takes per-sample graphs of different sizes, as objects or as tensors."""
    import pinneapple_neural.architectures  # noqa: F401
    from pinneapple_neural.architectures.adapters.base import select_adapter
    from pinneapple_neural.architectures.graphnn.base import GraphBatch
    from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
    from pinneapple_neural.architectures.registry import ModelRegistry

    adapter = select_adapter(ModelRegistry.spec("mesh_graph_net"))
    assert type(adapter).__name__ == "GNNAdapter"
    net = MeshGraphNet(node_in_dim=3, out_dim=1, edge_in_dim=2, hidden_dim=8, n_message_passing=2).eval()
    torch.manual_seed(8)
    graphs = [{"x": torch.randn(5, 3), "edge_index": torch.tensor([[0, 1], [1, 2]]), "edge_attr": torch.randn(2, 2)},
              {"x": torch.randn(9, 3), "edge_index": torch.tensor([[0, 1], [7, 8]]), "edge_attr": torch.randn(2, 2)}]
    out = adapter.forward_batch(net, {"graphs": graphs})
    assert out.y.shape == (2, 9, 1) and out.extras["ptr"].tolist() == [0, 5, 14]
    assert out.extras["mask"].sum().item() == 14
    with torch.no_grad():
        ref = net(GraphBatch(x=graphs[1]["x"][None], edge_index=graphs[1]["edge_index"],
                             edge_attr=graphs[1]["edge_attr"][None])).y[0]
    assert torch.allclose(out.extras["y_list"][1], ref, atol=1e-5)

    loose = adapter.forward_batch(net, {"x_list": [g["x"] for g in graphs],
                                        "edge_index_list": [g["edge_index"] for g in graphs],
                                        "edge_attr_list": [g["edge_attr"] for g in graphs]})
    assert torch.allclose(loose.y, out.y, atol=1e-6)

    scored = adapter.forward_batch(net, {"graphs": graphs, "y_true": out.y.detach()})
    assert float(scored.losses["mse"].detach()) < 1e-10            # the padded output scored against itself

    with pytest.raises((KeyError, ValueError)):
        adapter.forward_batch(net, {"x": torch.randn(1, 4, 3)})


def test_adapter_dispatches_mesh_dynamics_through_the_model_forward_batch():
    """The Dynamics MGN is reached through the Arena the same way, and its rollout comes back padded per sample."""
    import pinneapple_neural.architectures  # noqa: F401
    from pinneapple_neural.architectures.adapters.base import select_adapter
    from pinneapple_neural.architectures.registry import ModelRegistry

    spec = ModelRegistry.spec("mesh_dynamics_mgn")
    adapter = select_adapter(spec)
    assert adapter.can_handle(spec)
    graphs = _three_meshes()
    torch.manual_seed(9)
    trajs = [torch.randn(4, g.node_type.numel(), 2) for g in graphs]
    model = MeshDynamicsMGN(hidden_dim=16, n_message_passing=2).eval()
    out = adapter.forward_batch(model, {"mesh": graphs, "trajectory": trajs, "n_steps": 3})
    assert out.y.shape == (3, 4, 40, 2)                    # (B, n_steps+1, N_max, vel_dim)
    assert out.extras["mask"].sum().item() == sum(g.node_type.numel() for g in graphs)
    assert [v.shape[1] for v in out.extras["vel_list"]] == [24, 40, 18]


def test_physicsnemo_comparison_declares_the_absence_and_still_measures_this_side():
    """#231 (4): without PhysicsNeMo the comparison says so and measures the PINNeAPPle side; with a step callable it
    reports both and their agreement (a self-comparison must agree exactly)."""
    from pinneapple_neural.architectures.graphnn import MeshDynamicsMGN as MGN
    from pinneapple_neural.physicsnemo_bridge.mgn_compare import (
        compare_rollouts,
        native_availability,
        native_mgn_step,
    )

    graphs = _three_meshes()
    graph = graphs[1]
    torch.manual_seed(10)
    traj = torch.randn(5, graph.node_type.numel(), 2)
    model = MGN(hidden_dim=16, n_message_passing=2).eval()

    alone = compare_rollouts(model, traj, graph, n_steps=4)
    assert alone["native_rmse"] is None and alone["pinn_agreement_with_native"] is None
    assert isinstance(alone["pinn_rmse"], float)
    declared = native_availability()
    assert alone["native"]["available"] == declared["available"]
    if declared["available"]:
        assert declared["reason"] == ""
    else:
        assert "physicsnemo is not installed" in declared["reason"]

    both = compare_rollouts(model, traj, graph, n_steps=4,
                            native_step=lambda vel, g: model.step(vel, g)[0])
    assert both["pinn_agreement_with_native"] == pytest.approx(0.0, abs=1e-6)
    assert both["native_rmse"] == pytest.approx(both["pinn_rmse"])

    if not declared["available"]:
        with pytest.raises(ImportError):
            native_mgn_step(object(), num_node_types=4)
