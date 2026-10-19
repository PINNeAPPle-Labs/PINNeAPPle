"""Pinball loss and QuantileLoss against independent references, and their input validation."""
import numpy as np
import pytest
import torch

from pinneapple_analysis.uncertainty.quantile import (
    QuantileConfig,
    QuantileLoss,
    pinball_loss_torch,
)


def test_pinball_matches_hand_computed_values():
    y = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([2.0, 2.0, 2.0])
    # q = 0.9: errors -1, 0, 1 -> max(0.9 e, -0.1 e) = 0.1, 0, 0.9
    assert pinball_loss_torch(pred, y, 0.9).item() == pytest.approx(1.0 / 3.0)
    # q = 0.5 is half the mean absolute error
    assert pinball_loss_torch(pred, y, 0.5).item() == pytest.approx(0.5 * 2.0 / 3.0)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_pinball_minimiser_is_the_empirical_quantile(q):
    y = np.random.default_rng(0).standard_normal(1001)
    grid = np.linspace(-3, 3, 6001)
    losses = [pinball_loss_torch(torch.full((y.size,), c, dtype=torch.float64), torch.tensor(y), q).item()
              for c in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(np.quantile(y, q, method="inverted_cdf"), abs=1e-3)


def test_quantile_loss_averages_the_levels():
    y = torch.zeros(2, 3)
    pred = torch.stack([torch.full((2, 3), v) for v in (-1.0, 0.0, 1.0)], dim=-1)
    # q = 0.1 with e = 1: 0.1; q = 0.5 with e = 0: 0; q = 0.9 with e = -1: 0.1
    assert QuantileLoss([0.1, 0.5, 0.9])(None, pred, {"y": y})["total"].item() == pytest.approx(0.2 / 3)


@pytest.mark.parametrize("q_out", [2, 5])
def test_quantile_loss_rejects_a_mismatched_number_of_outputs(q_out):
    # regression: 5 outputs for 3 levels silently scored the first 3, and 2 outputs raised a bare IndexError
    with pytest.raises(ValueError, match="quantile outputs"):
        QuantileLoss([0.1, 0.5, 0.9])(None, torch.zeros(2, 3, q_out), {"y": torch.zeros(2, 3)})


def test_quantile_loss_rejects_a_broadcastable_target():
    # regression: a (B, 1) target was broadcast against (B, H) predictions
    with pytest.raises(ValueError, match="same shape"):
        QuantileLoss([0.5])(None, torch.zeros(2, 3, 1), {"y": torch.zeros(2, 1)})


@pytest.mark.parametrize("levels", [(0.0, 0.5), (0.5, 1.0), (1.5,), (), (0.5, 0.5)])
def test_invalid_levels_are_rejected(levels):
    # regression: q = 1.5 gave a loss that rewards over-prediction without bound
    with pytest.raises(ValueError):
        QuantileConfig(quantiles=levels)
    with pytest.raises(ValueError):
        QuantileLoss(levels)
    if len(levels) == 1:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            pinball_loss_torch(torch.zeros(3), torch.ones(3), levels[0])
