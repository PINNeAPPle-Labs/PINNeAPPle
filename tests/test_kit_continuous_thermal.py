"""Unit tests for continuous-time KIT thermal physics (not real-data results)."""
import numpy as np
import pytest
import torch
from pinneapple_lab.experiments.kit_continuous_thermal import (
    LumpedThermalPINN, train_continuous_pinn,
)


def test_autograd_residual_propagates_to_physical_parameters():
    net = LumpedThermalPINN(time_scale_seconds=3600, width=8)
    t = torch.tensor([[0.0], [60.0], [120.0]], requires_grad=True)
    ambient = torch.full_like(t, 5.0)
    q = torch.full_like(t, 1000.0)
    r = net.residual(t, ambient, q)
    assert r.shape == (3, 1)
    assert torch.isfinite(r).all()
    (r.square().sum()).backward()
    assert net.log_c.grad is not None
    assert torch.isfinite(net.log_c.grad).all()


def test_physical_coefficients_are_constrained():
    net = LumpedThermalPINN(3600)
    p = net.physical_parameters()
    assert p["capacity_j_per_k"].item() > 0
    assert p["conductance_w_per_k"].item() > 0
    assert 0 < p["heating_efficiency"].item() < 1


def test_short_continuous_training_unit_fixture():
    n = 45
    t = np.arange(n, dtype=float)*60
    temp = 19 + .1*np.sin(np.arange(n)/8)
    pred, hist, p = train_continuous_pinn(
        t, np.full(n,5.0), np.full(n,1000.0), temp,
        np.arange(25), np.arange(25,35), epochs=3, seed=0)
    assert pred.shape == (n,)
    assert hist.shape == (3,3)
    assert np.isfinite(pred).all()
    assert p["capacity_j_per_k"] > 0


def test_time_coordinate_must_be_monotonic():
    with pytest.raises(ValueError,match="time must increase"):
        train_continuous_pinn([0,60,60],[5]*3,[1000]*3,[20]*3,
                              [0],[1],epochs=1)
