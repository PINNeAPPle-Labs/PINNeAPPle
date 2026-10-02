import torch

from pinneapple_neural.architectures.neural_operators import NeuralOperatorCatalog, Transolver, TransolverLite


def test_transolver_shapes_and_permutation_equivariance():
    torch.manual_seed(0)
    for m in (Transolver(5, 7, dim=32, depth=2, heads=4, slices=8), TransolverLite(5, 7)):
        m.eval()
        x = torch.randn(2, 50, 5)
        y = m(x).y
        assert y.shape == (2, 50, 7)
        perm = torch.randperm(50)
        assert torch.allclose(m(x[:, perm]).y, y[:, perm], atol=1e-5)


def test_catalog_builds_native_transolver():
    m = NeuralOperatorCatalog().build("transolver_native", in_dim=3, out_dim=1, dim=16, depth=1, heads=2, slices=4)
    assert m(torch.randn(1, 10, 3)).y.shape == (1, 10, 1)
