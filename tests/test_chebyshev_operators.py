"""Contract tests for the Chebyshev-basis operators and the Hadamard branch/trunk family.

Two things are worth testing here beyond shapes. First, the parameter counts of the
published configurations: they are the cheapest evidence that the architecture matches
the paper it cites, and they catch a missing bias or a doubled modulation head without a
dataset. Second, that the registry can build every one of them, since a model that exists
but is unreachable from `instantiate` may as well not exist.
"""
from __future__ import annotations

import pytest
import torch

from pinneapple_neural.architectures.neural_operators import (
    ChebyshevBranchOperator,
    ChebyshevPolynomialNeuralOperator,
    HadamardDeepONet,
    chebyshev_basis,
    chebyshev_features,
)
from pinneapple_neural.architectures.neural_operators.base import OperatorOutput
from pinneapple_neural.architectures.neural_operators.registry import NeuralOperatorCatalog

N_TIMES = 61
GRID = (81, 58)


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def test_chebyshev_basis_matches_numpy():
    import numpy as np

    z = torch.linspace(-1.0, 1.0, 7).unsqueeze(-1)
    basis = chebyshev_basis(z, 6)
    for order in range(7):
        expected = np.polynomial.chebyshev.chebval(z.squeeze(-1).numpy(), [0] * order + [1])
        assert torch.allclose(basis[:, order], torch.as_tensor(expected, dtype=basis.dtype), atol=1e-6)


def test_chebyshev_basis_rejects_negative_order():
    with pytest.raises(ValueError, match="non-negative"):
        chebyshev_basis(torch.zeros(3, 1), -1)


def test_chebyshev_features_expands_every_coordinate():
    coords = torch.rand(5, 3) * 2 - 1
    features = chebyshev_features(coords, 6)
    assert features.shape == (5, 3 * 7)


def test_cpno_parameter_count_matches_the_paper_configuration():
    """Chen et al. (2026), order 6, depth 6: 35,906 parameters."""
    model = ChebyshevPolynomialNeuralOperator(coordinate_dim=3, parameter_dim=2, out_dim=2,
                                              order=6, depth=6)
    assert _count(model) == 35906


def test_cpno_is_a_point_operator():
    model = ChebyshevPolynomialNeuralOperator()
    coords = torch.rand(2, 11, 3) * 2 - 1
    theta = torch.rand(2, 2) * 2 - 1
    output = model(coords, theta)
    assert isinstance(output, OperatorOutput)
    assert output.y.shape == (2, 11, 2)
    assert "total" in output.losses


def test_cpno_broadcasts_a_single_scenario():
    model = ChebyshevPolynomialNeuralOperator()
    output = model(torch.rand(3, 5, 3) * 2 - 1, torch.rand(1, 2) * 2 - 1)
    assert output.y.shape == (3, 5, 2)


def test_cpno_rejects_a_mismatched_coordinate_count():
    model = ChebyshevPolynomialNeuralOperator(coordinate_dim=3)
    with pytest.raises(ValueError, match="coordinates"):
        model(torch.rand(1, 4, 2), torch.rand(1, 2))


def test_hybrid_parameter_counts_match_the_two_published_variants():
    """Santos et al. (2025) decomposition with Chen et al.'s branch: 41,025 at depth 6, 35,681 at depth 5."""
    deep = ChebyshevBranchOperator(n_steps=N_TIMES, grid=GRID, branch_depth=6)
    shallow = ChebyshevBranchOperator(n_steps=N_TIMES, grid=GRID, branch_depth=5)
    assert _count(deep) == 41025
    assert _count(shallow) == 35681


def test_hybrid_emits_the_whole_field_and_bounds_saturation():
    model = ChebyshevBranchOperator(n_steps=N_TIMES, grid=GRID, branch_depth=2)
    coords = torch.rand(GRID[0] * GRID[1], 3)
    coords[:, :2] = coords[:, :2] * 2 - 1
    output = model(coords, torch.rand(2, 2) * 2 - 1, torch.rand(1, N_TIMES))
    assert output.y.shape == (2, 2, N_TIMES, GRID[0], GRID[1])
    saturation = output.y[:, 0].detach()
    assert float(saturation.min()) >= 0.0 and float(saturation.max()) <= 1.0


def test_hadamard_deeponet_parameter_count_matches_the_reference():
    """Branch 18,650 + trunk 5,053 = 23,703 at 61 instants, 5 input channels."""
    model = HadamardDeepONet(in_channels=5, n_steps=N_TIMES, grid=GRID)
    assert _count(model) == 23703


def test_hadamard_deeponet_bounds_saturation_under_a_wild_trunk():
    model = HadamardDeepONet(in_channels=5, n_steps=4, grid=(3, 2))
    with torch.no_grad():
        model.trunk[-1].bias.fill_(40.0)
        model.branch[-1].weight.fill_(0.0)
        model.branch[-1].bias.fill_(20.0)
    saturation = model(torch.zeros(1, 5, 3, 2), torch.zeros(1, 4)).y[:, 0].detach()
    assert float(saturation.min()) >= 0.0 and float(saturation.max()) <= 1.0


def test_hadamard_deeponet_rejects_a_flat_input():
    model = HadamardDeepONet(in_channels=5, n_steps=4, grid=(3, 2))
    with pytest.raises(ValueError, match="branch_fields"):
        model(torch.zeros(1, 5, 6), torch.zeros(1, 4))


@pytest.mark.parametrize("module_name", ["cpno", "chebyshev_branch_trunk", "hadamard_deeponet"])
def test_every_new_operator_is_in_the_catalogue(module_name):
    catalogue = NeuralOperatorCatalog()
    assert module_name in catalogue.list()


def test_catalogue_builds_the_cpno_with_its_arguments():
    catalogue = NeuralOperatorCatalog()
    model = catalogue.get("cpno")(coordinate_dim=2, out_dim=1, order=4, depth=2)
    assert isinstance(model, ChebyshevPolynomialNeuralOperator)
    assert _count(model) < 35906


def test_gradients_reach_the_branch_and_the_trunk():
    model = ChebyshevBranchOperator(n_steps=5, grid=(4, 3), branch_depth=2)
    coords = torch.rand(12, 3)
    coords[:, :2] = coords[:, :2] * 2 - 1
    output = model(coords, torch.rand(2, 2), torch.rand(1, 5))
    output.y.mean().backward()
    assert model.branch.layers[0].poly.weight.grad is not None
    assert model.trunk.coefficients.grad is not None


def test_loss_is_returned_only_when_asked():
    model = HadamardDeepONet(in_channels=5, n_steps=3, grid=(3, 2))
    inputs = torch.rand(2, 5, 3, 2)
    times = torch.rand(1, 3)
    truth = torch.rand(2, 2, 3, 3, 2)
    plain = model(inputs, times)
    assert set(plain.losses) == {"total"}
    trained = model(inputs, times, y_true=truth, return_loss=True)
    assert "mse" in trained.losses and float(trained.losses["total"].detach()) > 0.0
