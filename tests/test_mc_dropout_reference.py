"""MC dropout against the closed-form moments of a two-layer linear network, and its side effects on the model."""
import pytest
import torch
import torch.nn as nn

from pinneapple_analysis.uncertainty.mc_dropout import MCDropout, MCDropoutConfig, MCDropoutWrapper


def _two_layer(seed=0, hidden=32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, hidden), nn.Linear(hidden, 1)).double()


def test_moments_match_dropout_on_the_hidden_layer():
    # y = w2 . (h * m / (1 - p)) + b2 with m ~ Bernoulli(1 - p): mean w2 . h + b2, variance p / (1 - p) sum (w2_i h_i)^2
    model, p, n = _two_layer(), 0.2, 20000
    x = torch.randn(4, 2, dtype=torch.float64)
    with torch.no_grad():
        h = model[0](x)
        w2, b2 = model[1].weight[0], model[1].bias[0]
        mean_ref = h @ w2 + b2
        std_ref = (p / (1 - p) * ((h * w2) ** 2).sum(dim=1)).sqrt()
    res = MCDropoutWrapper(model, MCDropoutConfig(n_samples=n, dropout_p=p, seed=0)).predict_with_uncertainty(x)
    se = std_ref / n ** 0.5
    assert torch.all((res.mean[:, 0] - mean_ref).abs() < 5 * se)
    assert torch.allclose(res.std[:, 0], std_ref, rtol=0.03)


def test_output_is_never_dropped():
    # regression: dropout was applied after every Linear, including the last one, so about p of the samples were
    # exactly zero and the std was |mean| sqrt(p / (1 - p)) whatever the model
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    res = MCDropoutWrapper(model, MCDropoutConfig(n_samples=2000, dropout_p=0.1, seed=0)).predict_with_uncertainty(
        torch.randn(5, 2))
    assert (res.samples == 0).float().mean().item() == 0.0


def test_train_eval_flags_and_global_rng_are_restored():
    # regression: the model was left in eval mode, and a seeded call reset the caller's global random stream
    model = _two_layer()
    model.train()
    model[0].eval()
    before = [m.training for m in model.modules()]
    x = torch.randn(3, 2, dtype=torch.float64)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    MCDropoutWrapper(model, MCDropoutConfig(n_samples=4, seed=7)).predict_with_uncertainty(x)
    MCDropout(model, MCDropoutConfig(n_samples=4, seed=7)).predict_with_uncertainty(x)
    assert torch.equal(torch.rand(3), expected)
    assert [m.training for m in model.modules()] == before


def test_seed_makes_sampling_reproducible():
    x = torch.randn(3, 2, dtype=torch.float64)
    a = MCDropoutWrapper(_two_layer(), MCDropoutConfig(n_samples=5, seed=3)).predict_with_uncertainty(x)
    b = MCDropoutWrapper(_two_layer(), MCDropoutConfig(n_samples=5, seed=3)).predict_with_uncertainty(x)
    assert torch.equal(a.samples, b.samples)


def test_a_single_sample_is_rejected():
    # regression: n_samples = 1 gave a NaN standard deviation
    with pytest.raises(ValueError, match="n_samples"):
        MCDropoutConfig(n_samples=1)
    w = MCDropoutWrapper(_two_layer(), MCDropoutConfig(n_samples=4))
    with pytest.raises(ValueError, match="n_samples"):
        w.predict_with_uncertainty(torch.randn(3, 2, dtype=torch.float64), n_samples=1)


def test_a_model_without_hidden_linear_layers_warns():
    with pytest.warns(UserWarning, match="no hidden"):
        MCDropoutWrapper(nn.Linear(2, 1), MCDropoutConfig(n_samples=4))
