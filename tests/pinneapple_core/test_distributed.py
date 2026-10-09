import numpy as np
import pytest

import pinneapple as pp
import pinneapple_core.distributed as dist
from pinneapple_core import Mesh


def _square(x):
    return x * x


def test_map_orders_results_and_matches_serial_across_backends():
    items = list(range(20))
    serial = dist.map(_square, items, backend="serial")
    assert serial == [i * i for i in items]
    assert dist.map(_square, items, workers=3) == serial
    assert dist.map(lambda k: k + 1, items, workers=3) == [i + 1 for i in items]      # closures work in processes
    assert dist.map(_square, items, workers=3, backend="thread") == serial
    assert dist.map(_square, [], workers=2) == []
    with pytest.raises(ValueError):
        dist.map(_square, items, backend="bogus")


def _flaky(x):
    if x == 3:
        raise ValueError("three is bad")
    return x


def test_errors_raise_or_are_collected_with_their_index():
    with pytest.raises(ValueError, match="three"):
        dist.map(_flaky, range(6), workers=2)
    out = dist.map(_flaky, range(6), workers=2, errors="collect")
    assert out[:3] == [0, 1, 2] and out[4:] == [4, 5]
    assert isinstance(out[3], dist.TaskError) and out[3].index == 3 and out[3].exception == "ValueError"
    with pytest.raises(ValueError):
        dist.map(_flaky, range(2), errors="bogus")


def _rect(a, b):
    torch = pytest.importorskip("torch")
    from pinneapple_core.fem import integrate_p1, solve_poisson

    ref = Mesh.structured([0, 0], [1, 1], (6, 6))
    pts = torch.tensor(ref.points, dtype=torch.float64) * torch.tensor([a, b], dtype=torch.float64)
    return float(integrate_p1(pts, ref.cells, solve_poisson(pts, ref.cells, 1.0, ref.boundary_nodes())))


def test_64_way_simulation_sweep_equals_the_serial_run():
    pytest.importorskip("torch")
    grid = {"a": np.linspace(0.5, 2.0, 8), "b": np.linspace(0.5, 2.0, 8)}          # 64 FEM simulations
    par = dist.sweep(_rect, grid, workers=64)
    ser = dist.sweep(_rect, grid, backend="serial")
    assert len(par) == 64 and par.workers <= 64 and not par.failed
    assert par.results() == pytest.approx(ser.results(), rel=1e-12)
    best = par.best(lambda r: -r)                                                   # largest rigidity: the biggest rectangle
    assert best["params"]["a"] == pytest.approx(2.0) and best["params"]["b"] == pytest.approx(2.0)
    assert par.table()[0]["a"] == pytest.approx(0.5) and "time_s" in par.table()[0]


def _sometimes_fails(x, y):
    if x == 2:
        raise RuntimeError("no")
    return x + y


def test_sweep_records_failures_and_accepts_explicit_configs():
    res = dist.sweep(_sometimes_fails, {"x": [1, 2, 3], "y": [10]}, workers=2)
    assert [r["result"] for r in res.records] == [11, None, 13] and len(res.failed) == 1
    assert res.failed[0]["error"].exception == "RuntimeError" and res.best()["params"]["x"] == 1
    assert dist.sweep(_sometimes_fails, [{"x": 5, "y": 1}], workers=1).results() == [6]
    with pytest.raises(ValueError):
        dist.sweep(_sometimes_fails, {})
    with pytest.raises(ValueError, match="succeeded"):
        dist.sweep(_sometimes_fails, [{"x": 2, "y": 0}], workers=1).best()


def _draw(seed):
    return float(np.random.default_rng(seed).random())


def test_ensemble_seeds_are_independent_and_independent_of_worker_count():
    a = dist.ensemble(_draw, 8, workers=1, seed=7)
    b = dist.ensemble(_draw, 8, workers=4, seed=7)
    assert a == b and len(set(a)) == 8 and dist.ensemble(_draw, 8, workers=2, seed=8) != a


def _nodal_gradient(mesh, values):
    return mesh.gradient(values)[:, 0, :]


def test_partitioned_mesh_gradient_equals_the_global_gradient():
    m = Mesh.structured([0, 0], [2, 1], (14, 9))
    u = np.sin(m.points[:, 0]) * m.points[:, 1] ** 2
    ref = m.gradient(u)[:, 0, :]
    for n_parts in (1, 3, 6):
        assert np.allclose(dist.map_mesh(_nodal_gradient, m, u, n_parts, workers=2), ref, atol=1e-12)
    parts = dist.partition_mesh(m, 5)
    owned = np.concatenate([p.node_ids[p.owned_nodes] for p in parts])
    assert sorted(owned.tolist()) == list(range(m.n_points))                      # every node owned exactly once
    cells = np.concatenate([p.cell_ids[p.owned_cells] for p in parts])
    assert sorted(cells.tolist()) == list(range(m.n_cells))
    sizes = [len(p.owned_cells) for p in parts]
    assert max(sizes) - min(sizes) <= 2                                           # balanced
    with pytest.raises(ValueError):
        dist.partition_mesh(m, 0)


def test_map_mesh_3d_integral_per_part():
    m = Mesh.structured([0, 0, 0], [1, 1, 1], (3, 3, 3))
    u = m.points.sum(axis=1)
    out = dist.map_mesh(lambda mesh, v: mesh.gradient(v)[:, 0, :], m, u, 4, workers=2, backend="serial")
    assert np.allclose(out, 1.0)


def test_decompose_box_geometry():
    subs = dist.decompose_box([0, 0], [4, 1], parts=4, overlap=0.5)
    assert [s.index for s in subs] == [0, 1, 2, 3] and subs[0].left is None and subs[3].right is None and subs[1].left == 0
    assert subs[0].lo[0] == 0 and subs[0].hi[0] == pytest.approx(1.5) and subs[1].lo[0] == pytest.approx(0.5)
    assert subs[3].hi[0] == 4 and all(s.core_hi[0] - s.core_lo[0] == pytest.approx(1.0) for s in subs)
    with pytest.raises(ValueError):
        dist.decompose_box([0], [1], parts=0)


def test_domain_decomposed_pinn_training_converges_through_the_module():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    pi = np.pi
    make = lambda: nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))

    def residual(model, x):                                                       # -u'' = pi^2 sin(pi x)
        u = model(x)
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return -torch.autograd.grad(g.sum(), x, create_graph=True)[0] - pi ** 2 * torch.sin(pi * x)

    res = dist.train_decomposed(residual, lambda X: np.zeros((len(X), 1)), make, [0.0], [1.0], parts=2, overlap=0.4,
                                iterations=8, steps=400, workers=2, reference=lambda X: np.sin(pi * X[:, :1]))
    errs = [h["rel_l2"] for h in res.history]
    assert errs[-1] < 0.02 and errs[-1] < errs[0] / 10                            # Schwarz iterations contract the error
    x = np.linspace(0, 1, 11)[:, None]
    assert np.abs(res.predict(x)[:, 0] - np.sin(pi * x[:, 0])).max() < 0.03
    assert len(res.subdomains) == 2 and res.wall_time_s > 0


def test_namespace():
    assert pp.distributed.map is dist.map
