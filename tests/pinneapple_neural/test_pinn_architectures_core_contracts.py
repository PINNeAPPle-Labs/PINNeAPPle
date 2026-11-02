"""Core coverage (#1129, toward 1.0) for the PINN architectures: X-TFC, PIELM, Bayesian PINN, XPINN, VPINN,
PINNsFormer pieces, inverse PINN, PINN-LSTM, the base class and the catalog.

Exact references:
* X-TFC's constrained expression g + B * N meets the boundary values whatever the network;
* a ridge fit solves its normal equations (PIELM, X-TFC);
* a Bayesian layer's KL equals torch's closed-form Gaussian KL, and its sampled outputs have the mean and variance of
  a linear map of Gaussian weights;
* the base class's derivatives are exact on a polynomial; XPINN's interface loss is zero for one shared network.
"""
import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.architectures.pinns import base as PB  # noqa: E402
from pinneapple_neural.architectures.pinns import bpinn as BP  # noqa: E402
from pinneapple_neural.architectures.pinns import inverse as INV  # noqa: E402
from pinneapple_neural.architectures.pinns import pielm as EL  # noqa: E402
from pinneapple_neural.architectures.pinns import pinn_lstm as PL  # noqa: E402
from pinneapple_neural.architectures.pinns import pinnsformer_paper as PF  # noqa: E402
from pinneapple_neural.architectures.pinns import registry as RG  # noqa: E402
from pinneapple_neural.architectures.pinns import vpinn as VP  # noqa: E402
from pinneapple_neural.architectures.pinns import xpinn as XP  # noqa: E402
from pinneapple_neural.architectures.pinns import xtfc as XT  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ------------------------------------------------------------------------------------------------ X-TFC
def test_xtfc_constrained_expression_meets_the_boundary_values_exactly():
    g = lambda x: 1.0 + x                                   # noqa: E731  u(0) = 1, u(1) = 2
    B = lambda x: x * (1.0 - x)                             # noqa: E731  vanishes on the boundary
    m = XT.build_xtfc(1, 1, rf_dim=64, g_fn=g, B_fn=B, use_tfc=False)
    fm = XT.XTFCFactoryModel(m, inverse_params_names=["k"], initial_guesses={"k": 0.3})
    xb = torch.tensor([[0.0], [1.0]])
    assert torch.allclose(fm(xb), torch.tensor([[1.0], [2.0]]), atol=1e-6)          # exact, any weights
    assert float(fm.inverse_params["k"]) == pytest.approx(0.3)
    x = torch.linspace(0, 1, 50)[:, None]
    target = torch.sin(3 * x)
    m.fit_ridge(x, target, l2=1e-8)
    assert torch.allclose(fm(x), m.forward_tensor(x))
    cfg = XT.XTFCConfig(in_dim=1, out_dim=1)
    assert cfg.rf_kind == "rff" and cfg.freeze_random


def test_tfc_availability_is_reported_honestly():
    if XT.tfc_available():
        pytest.skip("tfc is installed")
    with pytest.raises(ImportError, match="tfc"):
        XT.TFCConstrainedExpression(object())


# ------------------------------------------------------------------------------------------------ PIELM
def test_pielm_ridge_fit_solves_the_normal_equations():
    el = EL.PIELM(1, 1, hidden_dim=40)
    x = torch.linspace(-1, 1, 100)[:, None]
    y = x ** 3 - x
    el.fit_ridge(x, y, l2=1e-6)
    H = el.hidden(x)
    lhs = (H.T @ H + 1e-6 * torch.eye(40)) @ el.Beta
    assert torch.allclose(lhs, H.T @ y, rtol=1e-3, atol=1e-4)                         # (H'H + l2 I) Beta = H'y
    assert float(((el.forward_tensor(x) - y) ** 2).mean()) < 1e-4
    ad = EL.PIELMFactoryAdapter(EL.PIELM(2, 1, hidden_dim=30), inverse_params_names=["c"])
    xs, ts = torch.rand(60, 1), torch.rand(60, 1)
    ad.fit_ridge_from_inputs((xs, ts), xs * ts)
    out = ad(xs, ts)
    assert (out.y if hasattr(out, "y") else out).shape == (60, 1) and "c" in ad.inverse_params


# ------------------------------------------------------------------------------------------------ Bayesian PINN
def test_bayesian_layer_kl_and_predictive_moments():
    lay = BP.BayesianLinear(3, 2, prior_std=0.5, init_rho=-1.0)
    q = torch.distributions.Normal(lay.weight_mu, torch.nn.functional.softplus(lay.weight_rho))
    qb = torch.distributions.Normal(lay.bias_mu, torch.nn.functional.softplus(lay.bias_rho))
    p = torch.distributions.Normal(0.0, 0.5)
    ref = torch.distributions.kl_divergence(q, p).sum() + torch.distributions.kl_divergence(qb, p).sum()
    assert torch.allclose(lay.kl(), ref, rtol=1e-5)                                  # closed form
    x = torch.tensor([[1.0, -2.0, 0.5]])
    with torch.no_grad():
        s = torch.stack([lay(x) for _ in range(20000)])
        mean = lay.deterministic(x)
        var = (x ** 2) @ (torch.nn.functional.softplus(lay.weight_rho) ** 2).T \
            + torch.nn.functional.softplus(lay.bias_rho) ** 2
    assert torch.allclose(s.mean(0), mean, atol=0.05) and torch.allclose(s.var(0), var, rtol=0.05)


def test_bayesian_pinn_elbo_and_posterior_predictive():
    cfg = BP.BPINNConfig(in_dim=1, out_dim=1, hidden=[16, 16])
    m = BP.BayesianPINN(config=cfg)
    x = torch.rand(20, 1)
    out = m.elbo_loss((x,), pde_residual_fn=lambda u, xx: u - xx, obs_inputs=(x,), obs_targets=x)
    assert set(out) == {"total", "elbo", "kl", "nll_pde", "nll_obs"}
    assert torch.allclose(out["kl"], m.kl_divergence())
    assert torch.allclose(out["total"], out["nll_pde"] + out["nll_obs"] + cfg.kl_weight * out["kl"])
    mean, std = m.predict_ensemble((x,), n_samples=30)
    assert mean.shape == (20, 1) and torch.all(std > 0)
    assert torch.equal(m.predict_mean((x,)), m(x, deterministic=True).y)


# ------------------------------------------------------------------------------------------------ base class
def test_base_class_derivatives_are_exact_on_a_polynomial():
    x = PB.PINNBase.ensure_requires_grad(torch.rand(10, 2))
    y = x[:, :1] ** 2 * x[:, 1:]                                                       # u = x^2 y
    g = PB.PINNBase.grad(y, x)
    assert torch.allclose(g, torch.cat([2 * x[:, :1] * x[:, 1:], x[:, :1] ** 2], dim=1))
    n = torch.tensor([0.6, 0.8])
    dn = PB.PINNBase.normal_derivative(y, x, n)
    assert torch.allclose(dn.view(-1), (g @ n).view(-1))
    assert isinstance(XP.SubnetWrapper(2, 1, hidden=(8,)), PB.PINNBase)


# ------------------------------------------------------------------------------------------------ XPINN, VPINN
def test_xpinn_interface_loss_vanishes_for_identical_subnets():
    xp = XP.XPINN(2, 1, 1, hidden=(8, 8))
    xp.subnets[1].load_state_dict(xp.subnets[0].state_dict())
    xi = torch.rand(15, 1)
    out = xp([torch.rand(5, 1), torch.rand(7, 1)], interface_pairs=[(0, 1, xi, xi, torch.ones(1), torch.ones(1))])
    assert float(out.losses["interface"]) == 0.0 and float(out.losses["interface_flux"]) == 0.0
    assert [tuple(y.shape) for y in out.y] == [(5, 1), (7, 1)]                     # one output per subdomain
    xp.subnets[1].net.net[0].weight.data += 0.5
    out2 = xp([xi, xi], interface_pairs=[(0, 1, xi, xi)])
    assert float(out2.losses["interface"]) > 0


def test_vpinn_forward_and_weak_loss_adapter():
    v = VP.VPINN(2, 1, hidden=(8, 8), inverse_params_names=["nu"], initial_guesses={"nu": 0.1})
    xs, ts = torch.rand(6, 1), torch.rand(6, 1)
    assert v(xs, ts).y.shape == (6, 1) and float(v.inverse_params["nu"]) == pytest.approx(0.1)

    def weak(model, data):
        return torch.tensor(0.5), {"weak": 0.5}
    fn = VP.pinn_factory_adapter(weak)
    loss, comps = fn(v, {"a": 1}, extra=3)                                            # extra kwargs dropped
    assert float(loss) == 0.5 and comps == {"weak": 0.5}


# ------------------------------------------------------------------------------------------------ others
def test_wave_activation_and_official_init():
    wa = PF.WaveAct()
    x = torch.linspace(-3, 3, 7)
    assert torch.allclose(wa(x), torch.sin(x) + torch.cos(x))                         # w1 = w2 = 1 at start
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
    PF.init_official(net)
    assert all(torch.all(m.bias == 0.01) for m in net if isinstance(m, torch.nn.Linear))


def test_inverse_pinn_and_lstm_shapes():
    ip = INV.InversePINN(2, 1, hidden=[8, 8], inverse_params=["k"], initial_guesses={"k": 2.0})
    assert float(ip.inverse_params["k"]) == pytest.approx(2.0)
    assert ip(torch.rand(4, 2)).y.shape == (4, 1)
    lstm = PL.PINNLSTM(2, 1, seq_len=5, hidden_dim=8, num_layers=1)
    y = lstm(torch.rand(10, 1), torch.rand(10, 1))        # a sequence model: returns the tensor itself
    assert isinstance(y, torch.Tensor) and y.shape == (10, 1)


def test_pinn_catalog_and_registry(monkeypatch):
    cat = RG.PINNCatalog()
    assert cat.list() == sorted(cat.list()) and cat.list()
    name = cat.list()[0]
    assert cat.get(name) is cat.registry[name]
    with pytest.raises(KeyError):
        cat.get("nope")
    from pinneapple_neural.architectures.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "_REGISTRY", {})
    RG.register_into_global()
    assert set(ModelRegistry._REGISTRY) and all(s.family for s in ModelRegistry._REGISTRY.values())
