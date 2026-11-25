import pytest
import torch

from pinneapple_analysis.trust.trust_gate import TrustGate


@pytest.mark.parametrize("point,residual,allowed", [
    (0., 0., True),
    (100., 0., False),
    (0., 2., False),
    (0., float("nan"), False),
    (float("nan"), 0., False),
])
def test_hard_veto(point, residual, allowed):
    gate = TrustGate().fit(torch.tensor([[-1.], [0.], [1.]]))
    ok, reasons = gate.authorize(
        None, torch.tensor([[point]]),
        residual_fn=lambda model, x: torch.full_like(x, residual),
        max_residual_rmse=0.1,
    )
    assert ok == allowed
    assert bool(reasons) != allowed


def test_missing_ood_reference_blocks():
    ok, _ = TrustGate().authorize(
        None, torch.zeros(1, 1),
        residual_fn=lambda model, x: x,
        max_residual_rmse=0.1,
    )
    assert not ok
