"""PhysicsNeMo FNO / MeshGraphNet on a grid as PINNeAPPle predictors (pinneapple_neural.physicsnemo_bridge)."""
import importlib.util

import numpy as np
import pytest

from pinneapple_neural.physicsnemo_bridge.field_operators import grid_graph

HAS_PNEMO = importlib.util.find_spec("physicsnemo") is not None
HAS_PYG = HAS_PNEMO and all(importlib.util.find_spec(m) for m in ("torch_geometric", "torch_scatter"))


def test_grid_graph():
    ei, ea = grid_graph(4)
    assert ei.shape == (2, 2 * 2 * 4 * 3) and ea.shape == (ei.shape[1], 3)
    pairs = {tuple(p) for p in ei.T}
    assert all((j, i) in pairs for i, j in pairs)           # undirected
    i, j = ei
    di, dj = j // 4 - i // 4, j % 4 - i % 4
    assert np.array_equal(np.stack([di, dj], 1), ea[:, :2]) and np.allclose(ea[:, 2], 1.0)


def _data(n=16, m=6):
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    a = ob.threshold_coefficient(ob.periodic_grf(m, (n, n), alpha=2.0, tau=3.0, seed=0))
    return a, np.stack([ob.darcy_2d(x) for x in a])


@pytest.mark.skipif(not HAS_PNEMO, reason="nvidia-physicsnemo not installed")
def test_fno_trains_under_a_step_budget_and_round_trips(tmp_path):
    import physicsnemo.models.fno as pfno

    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_neural.physicsnemo_bridge.field_operators import (
        GridFieldPredictor,
        physicsnemo_fno,
        train_grid_operator,
    )
    a, u = _data()
    p = physicsnemo_fno(16, latent_channels=8, num_fno_layers=2, num_fno_modes=4, decoder_layer_size=16)
    assert isinstance(p.physicsnemo_model, pfno.FNO)
    res = train_grid_operator(p, a, u, ComputeBudget(steps=4), batch_size=3)
    assert res.run.steps == 4 and res.run.stopped_by == "steps" and res.epochs == pytest.approx(2.0)
    out = p.predict(a)
    assert out.shape == u.shape and np.isfinite(out).all() and p.predict(a[0]).shape == (16, 16)
    q = GridFieldPredictor.load(p.save(tmp_path / "fno"))
    assert np.allclose(q.predict(a), out, atol=1e-6)
    with pytest.raises(ValueError):
        p.predict(np.ones((1, 8, 8)))


@pytest.mark.skipif(not HAS_PNEMO, reason="nvidia-physicsnemo not installed")
def test_fno_takes_several_input_channels(tmp_path):
    """Two input fields (coefficient and coolant coupling) -> one solution field, and the state dict round-trips."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_neural.physicsnemo_bridge.field_operators import (
        GridFieldPredictor,
        physicsnemo_fno,
        train_grid_operator,
    )
    from pinneapple_physics import darcy_checks as dc
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    n, m = 16, 4
    a, _ = _data(n, m)
    band = 8.0 * (np.abs(np.arange(n) - n / 2) < n / 5).astype(float)
    sink = np.stack([np.tile(band[:, None], (1, n)) for _ in range(m)])
    u = np.stack([spla.spsolve(ob.darcy_matrix(ai) + sp.diags(si.ravel()), np.ones(n * n)).reshape(n, n)
                  for ai, si in zip(a, sink, strict=True)])
    assert np.abs(dc.darcy_residual(u, a, 1.0, sink=sink)).max() < 1e-8      # data of -div(a grad u) + s u = 1
    x = np.stack([a, sink], axis=1)

    p = physicsnemo_fno(n, in_channels=2, latent_channels=8, num_fno_layers=2, num_fno_modes=4,
                        decoder_layer_size=16)
    assert p.in_channels == 2
    res = train_grid_operator(p, x, u, ComputeBudget(steps=4), batch_size=3)
    assert res.run.steps == 4
    assert p.predict(x).shape == u.shape and p.predict(x[0:1]).shape == (1, n, n)
    q = GridFieldPredictor.load(p.save(tmp_path / "fno2"))
    assert q.in_channels == 2 and np.allclose(q.predict(x), p.predict(x), atol=1e-6)
    with pytest.raises(ValueError):                       # one channel where the model wants two
        p.predict(a)
    p1 = physicsnemo_fno(n, latent_channels=8, num_fno_layers=2, num_fno_modes=4, decoder_layer_size=16)
    assert p1.module.a_mean.shape == () and p1.in_channels == 1     # single-channel buffers stay scalar


@pytest.mark.skipif(not HAS_PYG, reason="PhysicsNeMo MeshGraphNet needs torch_geometric and torch_scatter")
def test_meshgraphnet_on_the_grid_graph(tmp_path):
    import physicsnemo.models.meshgraphnet as pmgn

    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_neural.physicsnemo_bridge.field_operators import (
        GridFieldPredictor,
        physicsnemo_meshgraphnet,
        train_grid_operator,
    )
    a, u = _data(n=8, m=4)
    p = physicsnemo_meshgraphnet(8, processor_size=2, hidden_dim=16)
    assert isinstance(p.physicsnemo_model, pmgn.MeshGraphNet)
    before = p.predict(a)
    res = train_grid_operator(p, a, u, ComputeBudget(steps=3), batch_size=2)
    assert res.run.steps == 3
    out = p.predict(a)
    assert out.shape == u.shape and not np.allclose(out, before)
    # graphs in a batch do not interact: one sample alone gives the same field as inside a batch
    assert np.allclose(p.predict(a[:1]), out[:1], atol=1e-5)
    q = GridFieldPredictor.load(p.save(tmp_path / "mgn"))
    assert np.allclose(q.predict(a), out, atol=1e-6)
