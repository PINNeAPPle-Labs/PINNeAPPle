"""DeepONet and graph networks: known operators, symmetries and losses."""
import torch

from pinneapple_neural.architectures.graphnn.base import GraphBatch
from pinneapple_neural.architectures.graphnn.equivariant_gnn import EquivariantGNN
from pinneapple_neural.architectures.graphnn.gnn import GraphNeuralNetwork
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
from pinneapple_neural.architectures.neural_operators.ms_deeponet import MultiScaleDeepONet


def test_deeponet_shapes_and_antiderivative():
    torch.manual_seed(0)
    m = 40
    x = torch.linspace(0, 1, m)
    k = torch.arange(1, 4).float()
    def batch(n):
        c = torch.randn(n, 3)
        u = (c[:, :, None] * torch.cos(k[None, :, None] * torch.pi * x)).sum(1)               # sum c_k cos(k pi x)
        g = (c[:, :, None] * torch.sin(k[None, :, None] * torch.pi * x) / (k[None, :, None] * torch.pi)).sum(1)
        return u, g
    net = DeepONet(m, 1, 1, hidden=64, modes=32, depth=3)
    assert net(torch.randn(2, m), torch.rand(7, 1)).y.shape == (2, 7, 1)
    assert net(torch.randn(3, m), torch.rand(3, 5, 1)).y.shape == (3, 5, 1)          # per-sample query points
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(1500):
        u, g = batch(64)
        loss = ((net(u, x[:, None]).y[..., 0] - g) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    u, g = batch(200)
    with torch.no_grad():
        err = torch.linalg.norm(net(u, x[:, None]).y[..., 0] - g) / torch.linalg.norm(g)
    assert err < 0.05
    ms = MultiScaleDeepONet(m, 1, 2, hidden=32, scales=(8, 8), scale_factors=(1.0, 8.0))
    assert ms(torch.randn(2, m), torch.rand(9, 1)).y.shape == (2, 9, 2)
    assert not hasattr(ms, "trunk")                    # no unused base trunk


def _graph(N=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, N, (2, 70), generator=g)
    ei = torch.cat([ei, ei.flip(0)], 1)
    return torch.randn(1, N, 4, generator=g), torch.randn(1, N, 3, generator=g), ei


def test_message_passing_is_permutation_equivariant():
    x, pos, ei = _graph()
    perm = torch.randperm(x.shape[1])
    inv = torch.argsort(perm)
    for m, use_pos in ((MeshGraphNet(4, 2, pos_dim=3, use_pos=True), True), (GraphNeuralNetwork(4, 2), False)):
        m.eval()
        g1 = GraphBatch(x=x, edge_index=ei, pos=pos if use_pos else None)
        g2 = GraphBatch(x=x[:, perm], edge_index=inv[ei], pos=pos[:, perm] if use_pos else None)
        with torch.no_grad():
            assert torch.allclose(m(g2).y, m(g1).y[:, perm], atol=1e-5)


def test_meshgraphnet_relative_positions_are_translation_invariant():
    x, pos, ei = _graph()
    m = MeshGraphNet(4, 2, pos_dim=3, use_pos=True, absolute_pos=False, decoder_layers=2).eval()
    with torch.no_grad():
        assert torch.allclose(m(GraphBatch(x=x, edge_index=ei, pos=pos)).y,
                              m(GraphBatch(x=x, edge_index=ei, pos=pos + 7.0)).y, atol=1e-4)


def test_egnn_is_e3_equivariant():
    x, pos, ei = _graph()
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(3)
    m = EquivariantGNN(4, 3, 2).eval()
    with torch.no_grad():
        o1 = m(GraphBatch(x=x, edge_index=ei, pos=pos))
        o2 = m(GraphBatch(x=x, edge_index=ei, pos=pos @ R.T + t))
    assert torch.allclose(o1.y, o2.y, atol=1e-4)
    assert torch.allclose(o1.extras["pos"] @ R.T + t, o2.extras["pos"], atol=1e-4)


def test_masked_loss_is_the_mean_over_valid_nodes():
    x, pos, ei = _graph()
    N = x.shape[1]
    mask = torch.ones(1, N)
    mask[:, N // 2:] = 0
    for m in (GraphNeuralNetwork(4, 2), MeshGraphNet(4, 2), EquivariantGNN(4, 3, 2)):
        out = m(GraphBatch(x=x, edge_index=ei, pos=pos, mask=mask), y_true=torch.zeros(1, N, 2), return_loss=True)
        assert torch.allclose(out.losses["mse"], (out.y[:, : N // 2] ** 2).mean(), rtol=1e-5)
