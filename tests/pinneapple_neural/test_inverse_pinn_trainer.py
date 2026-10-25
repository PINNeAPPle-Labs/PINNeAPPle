"""InversePINN trainer: early stopping on a loss comparable across steps, and no silent residual fallback.

Both were found by the cross-framework benchmark (benchmarks/vs_frameworks/REPORT.md, track C).
"""
from __future__ import annotations

import pytest
import torch

from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN


def _decay_problem(n_sensors=32, k_true=1.5):
    """dy/dx = -k y on [0, 2], y(0) = 1: sensors of exp(-k x)."""
    x = torch.linspace(0.0, 2.0, n_sensors).unsqueeze(1)
    return x, torch.exp(-k_true * x)


def _residual(model, x, params):
    x.requires_grad_(True)
    y = model(x)
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy + params["k"] * y


def _net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def test_residual_exception_is_raised_not_swallowed():
    x, y = _decay_problem()

    def broken(model, x_col, params):
        raise RuntimeError("bad residual")

    trainer = InversePINN(_net(), [InverseParam(1.0, name="k")], broken, x, y, domain_bounds=[(0.0, 2.0)], n_col=16)
    with pytest.raises(RuntimeError, match="bad residual"):
        trainer.train(n_epochs=3)


class _Frozen(torch.nn.Module):
    """Outputs 0 whatever the input: the data loss is a constant."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros_like(x[:, :1]) + 0.0 * self.w


def test_early_stopping_is_not_fooled_by_the_adaptive_weights():
    """The adaptive PDE weight changes every step, so the weighted total can rise while no loss term gets worse.

    Constant data loss 4 (the network outputs 0, the sensors read 2) and a PDE loss that falls every step. The EMA of
    the data loss starts at 1 and climbs to 4, so the PDE weight ema_data / ema_pde grows and the weighted total rises
    for many steps although the PDE loss only improves. The old rule watched the weighted total and stopped after
    ``patience`` steps; the unweighted sum improves every step, so the run goes on.
    """
    x = torch.linspace(0.0, 2.0, 32).unsqueeze(1)
    y = torch.full_like(x, 2.0)
    step = {"n": 0}

    def shrinking_residual(model, x_col, params):
        step["n"] += 1
        return torch.full((x_col.shape[0], 1), 0.999 ** step["n"]) + 0.0 * model(x_col) * params["k"]

    trainer = InversePINN(_Frozen(), [InverseParam(1.0, name="k")], shrinking_residual, x, y,
                          domain_bounds=[(0.0, 2.0)], n_col=16, patience=20, weighting="adaptive")
    hist = trainer.train(n_epochs=150)
    assert hist["total"][100] > hist["total"][0]          # the weighted total did rise: the trap is real
    assert len(hist["total"]) == 150


def test_early_stopping_still_stops_on_a_plateau_and_can_be_disabled():
    x, y = _decay_problem()

    def const_residual(model, x_col, params):
        # nothing can improve: residual and data loss are independent of every trainable parameter
        return torch.ones(x_col.shape[0], 1) + 0.0 * params["k"]

    trainer = InversePINN(_Frozen(), [InverseParam(1.0, name="k")], const_residual, x, y,
                          domain_bounds=[(0.0, 2.0)], n_col=16, patience=10)
    hist = trainer.train(n_epochs=200)
    assert len(hist["total"]) == 11            # the first step sets the best value, then 10 without improvement
    off = InversePINN(_Frozen(), [InverseParam(1.0, name="k")], const_residual, x, y,
                      domain_bounds=[(0.0, 2.0)], n_col=16, patience=None)
    assert len(off.train(n_epochs=50)["total"]) == 50


@pytest.mark.parametrize("weighting", ["equal", "adaptive"])
def test_recovers_a_decay_rate(weighting):
    x, y = _decay_problem(k_true=1.5)
    k = InverseParam(0.5, log_space=True, name="k")
    # lr 3e-3: at 1e-2 the estimate of k oscillates by about 10 % with either weighting
    trainer = InversePINN(_net(), [k], _residual, x, y, domain_bounds=[(0.0, 2.0)], n_col=128, lr=3e-3,
                          weighting=weighting)
    trainer.train(n_epochs=1500)
    assert abs(trainer.get_identified_params()["k"] - 1.5) / 1.5 < 0.1


def test_default_weighting_is_equal_and_fixed():
    """Default: total = 100 * (L_pde + L_data) at every step (benchmark track C: the adaptive rule left nu 12x off)."""
    x, y = _decay_problem()
    trainer = InversePINN(_net(), [InverseParam(1.0, name="k")], _residual, x, y, domain_bounds=[(0.0, 2.0)],
                          n_col=32, patience=None)
    assert trainer.weighting == "equal"
    h = trainer.train(n_epochs=20)
    for tot, pde, data in zip(h["total"], h["pde"], h["data"], strict=True):
        assert tot == pytest.approx(100.0 * (pde + data), rel=1e-5)


def test_adaptive_weighting_is_still_available():
    x, y = _decay_problem()
    trainer = InversePINN(_net(), [InverseParam(1.0, name="k")], _residual, x, y, domain_bounds=[(0.0, 2.0)],
                          n_col=32, patience=None, weighting="adaptive")
    h = trainer.train(n_epochs=20)
    ema_d = ema_p = 1.0
    for tot, pde, data in zip(h["total"], h["pde"], h["data"], strict=True):
        ema_d = 0.98 * ema_d + 0.02 * (data + 1e-12)
        ema_p = 0.98 * ema_p + 0.02 * (pde + 1e-12)
        w = min(max(ema_d / (ema_p + 1e-12), 0.01), 100.0) * 100.0
        assert tot == pytest.approx(w * pde + 100.0 * data, rel=1e-4)
    assert any(abs(t - 100.0 * (p + d)) > 1e-3 * t for t, p, d in zip(h["total"], h["pde"], h["data"], strict=True))
    with pytest.raises(ValueError, match="weighting"):
        InversePINN(_net(), [InverseParam(1.0, name="k")], _residual, x, y, domain_bounds=[(0.0, 2.0)],
                    weighting="gradnorm")
