"""Contract and isolation tests for hybrid surrogate components.

No real-data/industrial benchmark is implied by these synthetic checks.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_analysis.surrogate.data import SurrogateData
from pinneapple_analysis.surrogate.hybrid import ENCODERS, HybridDeepONet, RBFKANLayer
from pinneapple_analysis.surrogate.search import real_data_splits
from pinneapple_analysis.surrogate.methods import CANDIDATES


@pytest.mark.parametrize("branch,trunk", [("rbf_kan", "mlp"), ("mlp", "rbf_kan"),
                                          ("fourier", "rbf_kan"), ("rbf_kan", "fourier")])
def test_deeponet_independent_encoders_shapes_and_gradients(branch, trunk):
    net = HybridDeepONet(3, 2, 1, branch=branch, trunk=trunk, hidden=12, modes=8,
                         depth=2, grid_size=6, fourier_features=4)
    x = torch.randn(5, 3)
    c = torch.rand(7, 2)
    y = net(x, c).y
    assert y.shape == (5, 7, 1)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert net.branch is not None and net.trunk is not None
    assert any(p.grad is not None for p in net.branch.parameters())
    assert any(p.grad is not None for p in net.trunk.parameters())


def test_per_sample_trunk_contract():
    net = HybridDeepONet(2, 1, branch="mlp", trunk="rbf_kan", hidden=8, modes=4, depth=1)
    y = net(torch.randn(3, 2), torch.rand(3, 11, 1)).y
    assert y.shape == (3, 11, 1)


def test_invalid_encoders_rejected():
    with pytest.raises(ValueError, match="unknown encoder"):
        HybridDeepONet(2, 1, branch="unknown")
    with pytest.raises(ValueError):
        RBFKANLayer(2, 3, grid_size=0)
    assert {"mlp", "rbf_kan", "fourier"} <= set(ENCODERS)
    assert "deeponet_hybrid" in CANDIDATES


def test_grouped_real_data_disallows_case_leakage():
    rng = np.random.default_rng(10)
    x = rng.random((80, 2))
    y = rng.random((80, 12))
    groups = np.repeat(np.arange(16), 5)
    data = SurrogateData(x, y, ["u", "v"], ["temperature"], groups=groups)
    split = real_data_splits(data, seed=4)
    sets = [set(groups[getattr(split, k)]) for k in ("train", "val", "test", "shift")]
    assert all(not sets[i].intersection(sets[j]) for i in range(4) for j in range(i + 1, 4))
    assert len(set(np.concatenate([split.train, split.val, split.test, split.shift]))) == 80


def test_few_groups_fail_closed():
    data = SurrogateData(np.ones((12, 1)), np.ones((12, 5)), ["p"], ["field"],
                         groups=np.repeat(np.arange(4), 3))
    with pytest.raises(ValueError, match="at least eight"):
        real_data_splits(data)
