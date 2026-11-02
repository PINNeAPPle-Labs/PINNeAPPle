"""Core coverage (#1129, toward 1.0) for the trainer's losses, heteroscedastic UQ and adaptive collocation.

* Losses: values computed by hand, and when the physics hook is switched on.
* Heteroscedastic NLL: its value and minimiser in closed form (the best log-variance is log r^2). The MC-dropout
  decomposition is checked on a network whose mean is a Bernoulli(1/2) * 2 variable, so the epistemic variance is
  exactly 1 and the aleatoric variance is the fixed exp(log-variance).
* Adaptive collocation: when all residual is in one region and residual_frac = 1, every resampled point is there.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.trainer.collocation import (  # noqa: E402
    AdaptiveCollocationConfig,
    AdaptiveCollocationSampler,
    estimate_channel_scales,
    filter_nan_rows,
)
from pinneapple_neural.trainer.heteroscedastic import (  # noqa: E402
    HeteroscedasticWrapper,
    compute_aleatoric_epistemic_uq,
    heteroscedastic_nll_loss,
    init_logvar_bias,
)
from pinneapple_neural.trainer.losses import (  # noqa: E402
    CombinedLoss,
    PhysicsLossHook,
    SupervisedLoss,
    build_loss,
)


# ------------------------------------------------------------------------------------------------ losses
def test_supervised_and_combined_losses_by_hand():
    yh, y = torch.tensor([[1.0], [3.0]]), torch.tensor([[0.0], [0.0]])
    assert SupervisedLoss("mse")(yh, y).item() == pytest.approx(5.0)
    assert SupervisedLoss("mae")(yh, y).item() == pytest.approx(2.0)

    def phys(model, batch):                                            # (model, batch) -> (loss, components)
        return torch.tensor(0.25), {"pde": 0.2, "bc": 0.05}
    out = CombinedLoss(physics=PhysicsLossHook(phys), w_supervised=2.0, w_physics=4.0)(None, yh, {"y": y})
    assert out["total"].item() == pytest.approx(2.0 * 5.0 + 4.0 * 0.25)
    assert out["physics_pde"].item() == pytest.approx(0.2) and out["supervised"].item() == pytest.approx(5.0)

    def phys_pred(pred, batch):                     # the (pred, batch) form: tried second, after a TypeError
        return torch.abs(pred).mean()
    hook = PhysicsLossHook(phys_pred)
    assert hook(None, yh, {})["physics"].item() == pytest.approx(2.0)


def test_build_loss_enables_physics_only_when_everything_agrees():
    fn = lambda m, b: (torch.tensor(1.0), {})  # noqa: E731
    on = build_loss(problem_spec={"pde": "u_xx"}, model_capabilities={"supports_physics_loss": True},
                    weights={"data": 3.0, "pde": 0.5}, supervised_kind="mae", physics_loss_fn=fn)
    assert on.physics is not None and (on.w_supervised, on.w_physics) == (3.0, 0.5) and on.supervised.kind == "mae"
    assert build_loss(problem_spec={"pde": "u_xx"}, model_capabilities={}, physics_loss_fn=fn).physics is None
    assert build_loss(problem_spec=None, model_capabilities={"supports_physics_loss": True},
                      physics_loss_fn=fn).physics is None
    assert build_loss(problem_spec={"bc": 1}, model_capabilities={"supports_physics_loss": True}).physics is None


# ------------------------------------------------------------------------------------------------ heteroscedastic
def test_nll_value_and_its_minimiser():
    mu, y = torch.zeros(4, 1), torch.full((4, 1), 2.0)
    assert heteroscedastic_nll_loss(mu, torch.zeros(4, 1), y).item() == pytest.approx(0.5 * 4.0)
    lv = torch.zeros(4, 1, requires_grad=True)
    opt = torch.optim.SGD([lv], lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        heteroscedastic_nll_loss(mu, lv, y).backward()
        opt.step()
    assert torch.allclose(lv, torch.full((4, 1), math.log(4.0)), atol=1e-4)      # exact optimum: log r^2


def test_wrapper_split_and_bias_init():
    lin = torch.nn.Linear(3, 4)
    init_logvar_bias(lin, 2, bias_value=-4.6)
    assert torch.equal(lin.bias[2:], torch.full((2,), -4.6))
    w = HeteroscedasticWrapper(lin, 2)
    x = torch.randn(5, 3)
    mean, logvar = w.forward_uq(x)
    full = lin(x)
    assert torch.equal(mean, full[:, :2]) and torch.equal(logvar, full[:, 2:]) and torch.equal(w(x), mean)
    assert w.dropout_p == 0.0


class _BernoulliNet(torch.nn.Module):
    """Mean = dropout(1) in {0, 2} (variance 1); log-variance fixed at log 0.09."""
    dropout_p = 0.5

    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        n = x.shape[0]
        return torch.cat([self.drop(torch.ones(n, 1)), torch.full((n, 1), math.log(0.09))], dim=1)


def test_total_variance_splits_into_aleatoric_and_epistemic():
    with pytest.raises(ValueError):
        compute_aleatoric_epistemic_uq(HeteroscedasticWrapper(torch.nn.Linear(1, 2), 1), torch.zeros(3, 1))
    torch.manual_seed(0)
    uq = compute_aleatoric_epistemic_uq(HeteroscedasticWrapper(_BernoulliNet(), 1), torch.zeros(400, 1), n_mc=400)
    assert torch.allclose(uq["aleatoric_std"], torch.full((400, 1), 0.3), atol=1e-6)    # exact: sqrt(0.09)
    assert abs(uq["epistemic_std"].pow(2).mean().item() - 1.0) < 0.02                    # Bernoulli: var = 1
    assert torch.allclose(uq["total_std"] ** 2, uq["aleatoric_std"] ** 2 + uq["epistemic_std"] ** 2, atol=1e-6)
    assert abs(uq["mean"].mean().item() - 1.0) < 0.02


# ------------------------------------------------------------------------------------------------ collocation
def test_resampling_follows_the_residual():
    np.random.seed(0)
    pool = torch.linspace(0.0, 1.0, 1000)[:, None]
    hot = lambda model, pts: (pts[:, 0] > 0.9).float()  # noqa: E731
    cfg = AdaptiveCollocationConfig(n_active=50, n_pool=1000, resample_every=10, residual_frac=1.0, residual_batch=300)
    s = AdaptiveCollocationSampler(pool, cfg, residual_fn=hot)
    assert s.active_points().shape == (50, 1) and s.residual_stats() == {}
    model = torch.nn.Linear(1, 1)
    assert not s.step(5, model) and s.step(10, model) and not s.step(10, model)          # once per epoch multiple
    assert torch.all(s.active_points() > 0.9) and s.resample_count == 1                 # all mass is there
    assert s.resample_history()[0]["mean_residual_active"] == 1.0
    assert s.residual_stats()["mean"] == pytest.approx(0.1, abs=1e-3)

    s.cfg.residual_frac = 0.7                       # 35 draws from 0.7 p_res + 0.3 uniform (73 % hot), 15 uniform
    hot_counts = []
    for _ in range(40):
        s.resample_now(model)
        hot_counts.append(int((s.active_points() > 0.9).sum()))
    assert 22 < np.mean(hot_counts) < 31                                                 # vs 5 for uniform sampling
    s.replace_pool(torch.rand(200, 2))
    assert s.active_points().shape == (50, 2) and s.residual_stats() == {}


def test_manual_residuals_and_uniform_fallback():
    np.random.seed(1)
    s = AdaptiveCollocationSampler(torch.arange(100.0)[:, None],
                                   AdaptiveCollocationConfig(n_active=10, n_pool=100, residual_frac=0.5))
    s.record_residuals(torch.zeros(10))
    s.resample_now()                                                                     # zeros: uniform
    assert len(set(s.active_points()[:, 0].tolist())) == 10


def test_channel_scales_and_nan_filter():
    Y = torch.stack([torch.linspace(-1e-3, 1e-3, 101), torch.linspace(0, 300.0, 101), torch.zeros(101)], dim=1)
    s = estimate_channel_scales(Y, quantile=0.5, channel_names=["u", "T", "z"])
    assert s[0] == pytest.approx(5e-4, rel=1e-3) and s[1] == pytest.approx(150.0) and s[2] == 1e-6
    a = torch.tensor([[1.0], [float("nan")], [3.0], [4.0]])
    b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [float("inf"), 0.0], [4.0, 4.0]])
    fa, fb = filter_nan_rows(a, b)
    assert fa[:, 0].tolist() == [1.0, 4.0] and fb.shape == (2, 2)
    assert filter_nan_rows() == ()
