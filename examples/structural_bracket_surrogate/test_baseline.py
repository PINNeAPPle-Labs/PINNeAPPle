"""Smoke tests for the simplified L-bracket baseline.

Run from this directory: python -m pytest -q
Requires numpy, scipy, torch and pytest.
"""
import numpy as np
import torch
from deeponet import BracketDeepONet, relative_l2, stress_weighted_mse
from fem_reference import constitutive_matrix, tetra_b_matrix, solve_elasticity
from geometry import demo_case


def test_network_shapes_and_gradients():
    model = BracketDeepONet()
    y = model(torch.zeros(2, 31), torch.zeros(2, 5, 8))
    assert y.shape == (2, 5)
    y.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_metrics():
    ref = torch.ones(2, 4)
    assert torch.allclose(relative_l2(ref, ref), torch.tensor(0.0))
    assert torch.allclose(stress_weighted_mse(ref, ref), torch.tensor(0.0))


def test_constitutive_symmetry():
    d = constitutive_matrix(70e9, 0.33)
    assert np.allclose(d, d.T)
    assert np.linalg.eigvalsh(d).min() > 0


def test_tetra_linear_field_gradient():
    x = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    b, volume = tetra_b_matrix(x)
    assert np.isclose(volume, 1/6)
    u = np.column_stack((0.01*x[:,0], np.zeros(4), np.zeros(4))).ravel()
    assert np.allclose((b @ u)[0], 0.01)


def test_l_bracket_force_balance():
    nodes, tetra, fixed, forces = demo_case()
    result = solve_elasticity(nodes, tetra, fixed, forces, 70e9, 0.33)
    assert np.isfinite(result["von_mises_pa"]).all()
    assert result["von_mises_pa"].max() > 0
    assert result["force_balance_n"] < 1e-5
    assert np.allclose(result["displacement_m"][fixed], 0)
