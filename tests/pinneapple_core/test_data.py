import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import pinneapple as pp
from pinneapple_core import Domain, Geometry, Mesh, loss as L
from pinneapple_core.data import (
    ActiveSampler, AdaptiveSampler, Batch, BoundarySampler, CollocationSampler, DataLoader, MeshSampler, PhysicsDataset,
    TrajectorySampler,
)

PI = np.pi


def test_collocation_strategies_stay_in_domain_and_are_reproducible():
    box = Domain.box([0, -1, 0], [2, 1, 3])
    for strategy in ("uniform", "lhs", "sobol"):
        s = CollocationSampler(box, strategy, seed=3)
        a, b = s.sample(64)["x"], s.sample(64)["x"]
        assert a.shape == (64, 3) and box.contains(a.numpy()).all()
        assert not torch.equal(a, b)                                        # fresh points each call
        assert torch.equal(CollocationSampler(box, strategy, seed=3).sample(64)["x"], a)   # same seed, same stream
    lhs = CollocationSampler({"x": (0, 1)}, "lhs").sample(100)["x"][:, 0]
    assert sorted((lhs * 100).floor().int().tolist()) == list(range(100))      # one point per stratum
    assert CollocationSampler({"x": (0, 1), "t": (0, 2)}).coords == ("x", "t")
    with pytest.raises(ValueError):
        CollocationSampler(box, "bogus")


def test_collocation_in_a_csg_domain():
    from pinneapple_design.geometry.csg import annulus

    dom = Domain.from_sdf(annulus(0.0, 0.0, 0.5, 1.0))
    x = CollocationSampler(dom, "lhs").sample(200)["x"].numpy()
    r = np.linalg.norm(x, axis=1)
    assert x.shape == (200, 2) and (r >= 0.5 - 1e-9).all() and (r <= 1 + 1e-9).all()


def test_boundary_sampler_points_normals_and_named_parts():
    geom = Geometry.box([0, 0], [2, 1])
    out = BoundarySampler(geom).sample(300)
    assert torch.allclose(torch.as_tensor(geom.domain.sdf(out["x"].numpy())).abs(), torch.zeros(300, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(out["normal"].norm(dim=1), torch.ones(300))
    part = BoundarySampler(geom, "x_max").sample(50)
    assert torch.allclose(part["x"][:, 0], torch.full((50,), 2.0)) and torch.allclose(part["normal"][:, 0], torch.ones(50))
    with pytest.raises(KeyError):
        BoundarySampler(geom, "nope")
    with pytest.raises(KeyError):
        BoundarySampler(Domain.box([0], [1]), "x_min")


def test_mesh_sampler_modes():
    m = Mesh.structured([0, 0], [1, 1], (4, 4))
    nodes = MeshSampler(m, "nodes").sample(1000)
    assert nodes["x"].shape == (m.n_points, 2) and sorted(nodes["index"].tolist()) == list(range(m.n_points))
    assert MeshSampler(m, "nodes").sample(5)["x"].shape == (5, 2)
    cells = MeshSampler(m, "cells").sample(m.n_cells)
    assert torch.allclose(cells["x"], torch.as_tensor(m.cell_centroids, dtype=torch.float32))
    rnd = MeshSampler(m, "random").sample(500)
    assert rnd["x"].shape == (500, 2) and Domain.box([0, 0], [1, 1]).contains(rnd["x"].numpy()).all()
    located, _ = m.locate(rnd["x"].numpy())
    assert (located >= 0).all()
    with pytest.raises(ValueError):
        MeshSampler(m, "bogus")


def test_trajectory_sampler_windows():
    t = torch.arange(10.0)
    states = torch.stack([t, 10 * t], dim=1)[None].repeat(3, 1, 1) + torch.arange(3.0)[:, None, None]
    out = TrajectorySampler(states, n_in=2, n_out=3).sample(40)
    assert out["x"].shape == (40, 2, 2) and out["y"].shape == (40, 3, 2)
    assert torch.allclose(out["y"][:, 0, 0], out["x"][:, -1, 0] + 1)         # y continues right after x
    assert (out["t0"] >= 0).all() and (out["t0"] <= 10 - 5).all()
    assert TrajectorySampler(states, stride=2).sample(50)["t0"].remainder(2).eq(0).all()
    with pytest.raises(ValueError):
        TrajectorySampler(states, n_in=8, n_out=8)
    with pytest.raises(ValueError):
        TrajectorySampler(torch.zeros(5, 3))


def test_adaptive_sampler_concentrates_where_the_residual_is_large():
    base = CollocationSampler({"x": (0.0, 1.0)}, seed=1)
    resid = lambda model, x: ((x[:, 0] > 0.8).float() * 10 + 0.01)           # large residual on x > 0.8
    ad = AdaptiveSampler(base, resid, pool_size=5000, uniform_fraction=0.1)
    before = (ad.sample(500)["x"][:, 0] > 0.8).float().mean().item()
    ad.update(model=lambda x: x)
    after = (ad.sample(500)["x"][:, 0] > 0.8).float().mean().item()
    assert before == pytest.approx(0.2, abs=0.06) and after > 0.7
    assert ad.sample(10)["x"].requires_grad is False and AdaptiveSampler.needs_grad
    with pytest.raises(ValueError):
        ad.update()


def test_active_sampler_picks_the_highest_scores():
    base = CollocationSampler({"x": (0.0, 1.0)}, seed=2)
    act = ActiveSampler(base, lambda model, x: x[:, 0], n_candidates=1000)
    cold = act.sample(5)
    assert (cold["score"] == 0).all()                                         # no model yet
    act.update(model=object())
    out = act.sample(5)
    assert out["x"][:, 0].min() > 0.99 and torch.equal(out["score"], out["x"][:, 0])


def test_dataset_split_and_validation():
    ds = PhysicsDataset(np.arange(10.0)[:, None], np.arange(10.0)[:, None] * 2, coords=["x"])
    a, b = ds.split([0.7, 0.3], seed=0)
    assert (len(a), len(b)) == (7, 3) and sorted(a.inputs[:, 0].tolist() + b.inputs[:, 0].tolist()) == list(range(10))
    assert torch.allclose(a.targets, 2 * a.inputs)                           # pairs stay together
    with pytest.raises(ValueError):
        ds.split([0.5, 0.2])
    with pytest.raises(ValueError):
        PhysicsDataset(np.zeros((4, 1)), np.zeros((3, 1)))


def test_loader_epoch_mode_covers_the_dataset_once_and_reshuffles():
    ds = PhysicsDataset(np.arange(10.0)[:, None], np.arange(10.0)[:, None])
    loader = DataLoader(ds, batch_size=4, seed=5)
    assert len(loader) == 3
    first = torch.cat([b.x for b in loader])[:, 0]
    second = torch.cat([b.x for b in loader])[:, 0]
    assert sorted(first.tolist()) == list(range(10)) == sorted(second.tolist())
    assert not torch.equal(first, second)
    assert [len(b.x) for b in DataLoader(ds, batch_size=4, shuffle=False)] == [4, 4, 2]
    assert len(DataLoader(ds, batch_size=4, drop_last=True)) == 2
    b = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert isinstance(b, Batch) and torch.equal(b.x, b.y)


def test_loader_mixed_sources_physics_aware_and_options():
    geom = Geometry.box([0.0], [1.0])
    obs = PhysicsDataset(np.linspace(0, 1, 7)[:, None], np.zeros((7, 1)))
    loader = DataLoader({"col": CollocationSampler(geom), "bc": BoundarySampler(geom), "obs": obs},
                        batch_size={"col": 32, "bc": 2, "obs": 3}, steps=5)
    batches = list(loader)
    assert len(batches) == 5
    b = batches[0]
    assert b["col"]["x"].requires_grad and b["bc"]["x"].requires_grad and not b["obs"]["x"].requires_grad
    assert b["col"]["x"].shape == (32, 1) and b["obs"]["x"].shape == (3, 1) and b.meta["coords"]["col"] is None
    seen = torch.cat([bb["obs"]["x"] for bb in batches])[:, 0]
    assert len(seen) == 15 and set(np.round(seen.numpy(), 6)) <= set(np.round(np.linspace(0, 1, 7), 6))   # cycles through
    plain = next(iter(DataLoader({"col": CollocationSampler(geom)}, batch_size=8, steps=1, physics_aware=False)))
    assert not plain.x.requires_grad
    assert next(iter(DataLoader({"col": CollocationSampler(geom)}, 8, steps=1, dtype=torch.float64))).x.dtype == torch.float64
    with pytest.raises(AttributeError):
        batches[0].x
    with pytest.raises(ValueError, match="steps"):
        DataLoader(CollocationSampler(geom))
    with pytest.raises(KeyError):
        DataLoader({"col": CollocationSampler(geom), "obs": obs}, batch_size={"col": 4}, steps=1)
    with pytest.raises(TypeError):
        DataLoader({"x": np.zeros(3)})


def test_loader_update_reaches_adaptive_samplers():
    base = CollocationSampler({"x": (0.0, 1.0)})
    ad = AdaptiveSampler(base, lambda m, x: (x[:, 0] > 0.9).float() + 1e-3, pool_size=2000, uniform_fraction=0.05)
    loader = DataLoader({"col": ad}, batch_size=200, steps=2)
    uniform = next(iter(loader))["col"]["x"]
    loader.update(model=lambda x: x)
    focused = next(iter(loader))["col"]["x"]
    assert (focused > 0.9).float().mean() > 3 * (uniform > 0.9).float().mean()


# -- the done-when: a PINN, an operator and an inverse problem from the same loader API ----------
def _second(u, x):
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return torch.autograd.grad(g.sum(), x, create_graph=True)[0]


def _mlp(i, o, w=24):
    return nn.Sequential(nn.Linear(i, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(), nn.Linear(w, o))


def test_pinn_operator_and_inverse_problem_train_from_the_same_loader_api():
    torch.manual_seed(0)
    geom = Geometry.box([0.0], [1.0])
    xt = torch.linspace(0, 1, 101)[:, None]

    # PINN
    net = _mlp(1, 1)
    loader = DataLoader({"col": CollocationSampler(geom, "lhs"), "bc": BoundarySampler(geom)},
                        batch_size={"col": 128, "bc": 2}, steps=1500)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for batch in loader:
        x = batch["col"]["x"]
        r = -_second(net(x), x) - PI ** 2 * torch.sin(PI * x)
        loss = L.combine({"pde": L.pde(r), "bc": L.boundary(net(batch["bc"]["x"]))}, {"bc": 10.0})
        opt.zero_grad(); loss.backward(); opt.step()
    assert L.supervised(net(xt), torch.sin(PI * xt), "relative_l2").item() < 0.05

    # operator: f -> u for -u'' = f
    n, K = 32, np.arange(1, 4)
    a = np.random.default_rng(0).normal(size=(300, K.size)) / K ** 1.5
    modes = np.sin(PI * np.outer(K, np.linspace(0, 1, n)))
    train, test = PhysicsDataset((a * (PI * K) ** 2) @ modes, a @ modes).split([0.8, 0.2])
    op, scale = _mlp(n, n, 64), train.inputs.std()
    opt = torch.optim.Adam(op.parameters(), lr=3e-3)
    for _ in range(60):
        for batch in DataLoader(train, batch_size=48):
            loss = L.supervised(op(batch.x.float() / scale), batch.y.float())
            opt.zero_grad(); loss.backward(); opt.step()
    tb = next(iter(DataLoader(test, batch_size=len(test), shuffle=False)))
    assert L.supervised(op(tb.x.float() / scale), tb.y.float(), "relative_l2").item() < 0.15

    # inverse problem: kappa in -kappa u'' = 2 pi^2 sin(pi x)
    xo = torch.rand(12, 1)
    obs = PhysicsDataset(xo, torch.sin(PI * xo) + 0.005 * torch.randn(12, 1))
    net, log_kappa = _mlp(1, 1), nn.Parameter(torch.zeros(()))
    loader = DataLoader({"col": CollocationSampler(geom), "obs": obs}, batch_size={"col": 128, "obs": 12}, steps=1800)
    opt = torch.optim.Adam([*net.parameters(), log_kappa], lr=4e-3)
    for batch in loader:
        x = batch["col"]["x"]
        r = -log_kappa.exp() * _second(net(x), x) - 2 * PI ** 2 * torch.sin(PI * x)
        loss = L.combine({"pde": L.pde(r), "data": L.supervised(net(batch["obs"]["x"]), batch["obs"]["y"]),
                          "bc": L.boundary(net(torch.tensor([[0.0], [1.0]])))}, {"data": 10.0, "bc": 10.0})
        opt.zero_grad(); loss.backward(); opt.step()
    assert log_kappa.exp().item() == pytest.approx(2.0, rel=0.1)
