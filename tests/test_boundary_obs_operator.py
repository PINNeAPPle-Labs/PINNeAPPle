"""Regression tests for boundary observation operators."""

import pytest
import torch
import torch.nn as nn

from pinneapple_analysis.inverse_problems import BoundaryObsOperator
from pinneapple_analysis.inverse_problems.obs_operator import PointObsOperator


class LinearField(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x):
        return x @ self.coefficients.unsqueeze(-1)


def test_boundary_operator_without_normals_matches_point_observations():
    points = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    model = LinearField()

    actual = BoundaryObsOperator(points)(model)
    expected = PointObsOperator(points)(model)

    assert torch.equal(actual, expected)
    assert torch.allclose(actual, torch.tensor([[3.0], [13.0]]))


def test_boundary_operator_projects_gradient_onto_outward_normals():
    points = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    normals = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    model = LinearField()

    actual = BoundaryObsOperator(points, normals)(model)

    assert torch.allclose(actual, torch.tensor([[2.0], [-3.0]]))


def test_boundary_operator_supports_output_selection():
    class TwoField(nn.Module):
        def forward(self, x):
            return torch.stack((2.0 * x[:, 0], 3.0 * x[:, 1]), dim=-1)

    points = torch.tensor([[1.0, 2.0]])
    normals = torch.tensor([[0.0, 1.0]])

    actual = BoundaryObsOperator(points, normals, output_indices=[1])(TwoField())

    assert torch.allclose(actual, torch.tensor([[3.0]]))


def test_boundary_operator_preserves_parameter_gradients():
    points = torch.tensor([[1.0, 2.0]])
    normals = torch.tensor([[1.0, 0.0]])
    model = LinearField()

    BoundaryObsOperator(points, normals)(model).sum().backward()

    assert torch.allclose(model.coefficients.grad, torch.tensor([1.0, 0.0]))


def test_boundary_operator_rejects_mismatched_normals():
    points = torch.zeros((2, 2))
    normals = torch.zeros((1, 2))

    with pytest.raises(ValueError, match='same shape'):
        BoundaryObsOperator(points, normals)


def test_boundary_operator_rejects_normals_for_different_evaluation_points():
    points = torch.zeros((2, 2))
    normals = torch.ones((2, 2))
    operator = BoundaryObsOperator(points, normals)

    with pytest.raises(ValueError, match='match the evaluated'):
        operator(LinearField(), torch.zeros((3, 2)))

def test_boundary_operator_returns_zero_for_constant_fields():
    class ConstantField(nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

    points = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    actual = BoundaryObsOperator(points, normals)(ConstantField())

    assert torch.equal(actual, torch.zeros((2, 1)))
