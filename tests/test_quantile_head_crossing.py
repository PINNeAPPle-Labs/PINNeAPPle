"""QuantileHead: the predicted quantile levels never cross, and crossing_rate measures it."""
import pytest
import torch
import torch.nn as nn

from pinneapple_analysis.uncertainty import crossing_rate
from pinneapple_analysis.uncertainty.quantile import QuantileConfig, QuantileHead, QuantileLoss


def _head(quantiles=(0.1, 0.5, 0.9), **kw):
    torch.manual_seed(0)
    return QuantileHead(nn.Linear(2, 1), QuantileConfig(quantiles=quantiles), hidden_dim=8, **kw)


def test_crossing_rate_counts_points_with_a_higher_level_below_a_lower_one():
    y = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [3.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    assert crossing_rate(y, (0.1, 0.5, 0.9)) == pytest.approx(0.5)
    assert crossing_rate(y[:1].flip(-1), (0.9, 0.5, 0.1)) == 0.0          # levels given in descending order
    with pytest.raises(ValueError):
        crossing_rate(y, (0.1, 0.9))


def test_raw_head_crosses_and_the_default_head_does_not():
    x = 10 * torch.randn(512, 2)
    raw = _head(rearrange=False)
    assert crossing_rate(raw(x), raw.cfg.quantiles) > 0.1                # the defect in the issue
    fixed = _head()
    out = fixed(x)
    assert crossing_rate(out, fixed.cfg.quantiles) == 0.0
    assert (out[..., 1:] >= out[..., :-1]).all()


def test_rearrangement_keeps_the_values_and_follows_the_order_the_levels_were_given_in():
    x = torch.randn(64, 2)
    raw, fixed = _head((0.9, 0.1, 0.5), rearrange=False), _head((0.9, 0.1, 0.5))
    a, b = raw(x), fixed(x)
    assert torch.equal(a.sort(dim=-1).values, b.sort(dim=-1).values)       # same numbers, reordered per point
    assert (b[..., 0] >= b[..., 2]).all() and (b[..., 2] >= b[..., 1]).all()   # 0.9 >= 0.5 >= 0.1
    assert crossing_rate(b, fixed.cfg.quantiles) == 0.0


def test_gradients_still_reach_every_parameter_through_the_sort():
    head = _head()
    x = torch.randn(16, 2)
    y_pred = head(x)
    loss = QuantileLoss(head.cfg.quantiles)(head, y_pred, {"y": torch.randn(16, 1)})["total"]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
