import torch

from pinneapple_neural.architectures.graphnn.mgn_dynamics import (
    MeshDynamicsMGN, MeshGraph, Normalizer, triangles_to_edges)


def _mesh(n=30):
    torch.manual_seed(0)
    pos = torch.rand(n, 2)
    from scipy.spatial import Delaunay
    cells = torch.from_numpy(Delaunay(pos.numpy()).simplices)
    nt = torch.zeros(n, dtype=torch.long)
    nt[:4] = 3
    return MeshGraph.from_mesh(pos, cells, nt)


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
