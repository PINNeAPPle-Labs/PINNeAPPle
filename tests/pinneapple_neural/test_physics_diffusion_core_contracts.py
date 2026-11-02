"""Core coverage for the physics-informed diffusion model (#1129, toward 1.0).

The schedulers are checked against their definitions (VP: alpha^2 + sigma^2 = 1, alpha_bar = prod(1 - beta);
EDM: the Karras schedule runs from sigma_max to sigma_min). The samplers are checked with an exact (oracle)
denoiser for a one-point data distribution x_0 = c, where the reverse process has a closed form:
* EDM Euler: x - c is multiplied by sigma_next / sigma at every step, so after the whole schedule
  x - c = (x_T - c) * sigma_min / sigma_max exactly;
* DDIM (eta = 0): every step lands on alpha_n * c + sigma_n * eps, so the last state is c up to sigma at t = 0.
The guidance gradients are compared with the gradients of their losses written by hand.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.architectures.pinns.physics_diffusion import (  # noqa: E402
    ComposedGuidance,
    DataConsistencyGuidance,
    DSMLoss,
    EDMNoiseScheduler,
    FiLMLayer,
    GuidanceBase,
    MLPDenoiser,
    NoiseSchedulerBase,
    PDEResidualGuidance,
    PhysicsInformedDiffusion,
    PIDiffConfig,
    SinusoidalEmbedding,
    VPNoiseScheduler,
)

C = torch.linspace(-1.0, 1.0, 8)                     # the one point of the data distribution


class _OracleEps(torch.nn.Module):
    """The exact noise predictor for x_0 = C under a VP scheduler."""

    def __init__(self, sched):
        super().__init__()
        self.sched, self.w = sched, torch.nn.Parameter(torch.zeros(1))

    def forward(self, xt, t, cond=None):
        a = self.sched.alpha(t).view(-1, 1)
        s = self.sched.sigma(t).view(-1, 1)
        return (xt - a * C) / s


class _OracleX0(torch.nn.Module):
    """The exact x_0 predictor for x_0 = C (EDM networks predict x_0)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, xt, t, cond=None):
        return C.expand_as(xt).clone()


# ------------------------------------------------------------------------------------------------ schedulers
def test_vp_scheduler_matches_its_definition():
    s = VPNoiseScheduler(T=100, beta_min=1e-4, beta_max=0.02)
    t = torch.arange(100)
    ab = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, 100), 0)
    assert torch.allclose(s.alpha(t) ** 2, ab) and torch.allclose(s.alpha(t) ** 2 + s.sigma(t) ** 2, torch.ones(100))
    assert s.timesteps(5).tolist() == [99, 74, 49, 24, 0]                       # reverse order, T-1 down to 0
    torch.manual_seed(0)
    x0 = torch.randn(4, 8)
    tt = torch.tensor([0, 10, 50, 99])
    xt, eps = s.add_noise(x0, tt)
    assert torch.allclose(xt, s.alpha(tt)[:, None] * x0 + s.sigma(tt)[:, None] * eps)
    assert torch.allclose(s.x0_from_pred(eps, xt, tt), x0, atol=1e-5)           # the true noise gives x_0 back


def test_edm_schedule_runs_from_sigma_max_to_sigma_min():
    s = EDMNoiseScheduler(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    sig = s.sigma_schedule(18)
    assert sig[0].item() == pytest.approx(80.0) and sig[-1].item() == pytest.approx(0.002)
    assert torch.all(sig[1:] < sig[:-1])
    mid = (80.0 ** (1 / 7) + 0.5 * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7     # Karras et al. (2022), eq. 5
    assert s.sigma_schedule(3)[1].item() == pytest.approx(mid, rel=1e-6)
    t = torch.tensor([0.5, 2.0])
    assert torch.equal(s.alpha(t), torch.ones(2)) and torch.equal(s.sigma(t), t) and s.timesteps(4).tolist() == [0, 1, 2, 3]
    x0 = torch.zeros(2, 3)
    xt, eps = s.add_noise(x0, t)
    assert torch.allclose(xt, t[:, None] * eps) and torch.equal(s.x0_from_pred(x0, xt, t), x0)


def test_the_protocols_must_be_implemented():
    with pytest.raises(NotImplementedError):
        NoiseSchedulerBase().alpha(torch.zeros(1))
    with pytest.raises(NotImplementedError):
        GuidanceBase()(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1))


# ------------------------------------------------------------------------------------------------ the network
def test_embedding_film_and_denoiser():
    e = SinusoidalEmbedding(16)(torch.tensor([0.0, 3.0]))
    assert e.shape == (2, 16) and torch.equal(e[0, :8], torch.zeros(8)) and torch.equal(e[0, 8:], torch.ones(8))
    film = FiLMLayer(4, 3)
    for p in film.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(5, 4)
    assert torch.equal(film(x, torch.randn(5, 3)), x)                             # zero modulation: identity
    d = MLPDenoiser(x_dim=8, t_emb_dim=16, hidden=[32, 32])
    assert d(torch.randn(3, 8), torch.tensor([1.0, 5.0, 9.0])).shape == (3, 8)


def test_dsm_loss_is_zero_for_the_exact_x0_predictor_and_one_for_a_zero_eps_predictor():
    torch.manual_seed(0)
    x0 = C.expand(64, 8)
    assert DSMLoss(_OracleX0(), EDMNoiseScheduler())(x0).item() == 0.0
    zero = MLPDenoiser(x_dim=8, t_emb_dim=8, hidden=[8])
    torch.nn.init.zeros_(zero.out_proj.weight)
    torch.nn.init.zeros_(zero.out_proj.bias)
    loss = DSMLoss(zero, VPNoiseScheduler(T=50))(torch.zeros(4096, 8)).item()
    assert abs(loss - 1.0) < 0.05                                                 # E[eps^2] = 1


# ------------------------------------------------------------------------------------------------ guidance
def test_guidance_gradients_match_their_losses():
    x = torch.randn(2, 6)
    g = PDEResidualGuidance(lambda u: u, weight=0.3)(x, x, torch.zeros(2))
    assert torch.allclose(g, -0.3 * 2 * x / x.numel())                            # d/dx mean(x^2)
    mask = torch.tensor([True, False, True, False, False, True])
    y = torch.tensor([1.0, 2.0, 3.0])
    d = DataConsistencyGuidance(mask, y, weight=0.5)(x, x, torch.zeros(2))
    ref = torch.zeros_like(x)
    ref[:, mask] = -0.5 * 2 * (x[:, mask] - y) / (2 * 3)
    assert torch.allclose(d, ref) and torch.all(d[:, ~mask] == 0)
    both = ComposedGuidance(PDEResidualGuidance(lambda u: u, 0.3), DataConsistencyGuidance(mask, y, 0.5))
    assert torch.allclose(both(x, x, torch.zeros(2)), g + d)


# ------------------------------------------------------------------------------------------------ sampling
def test_edm_sampler_contracts_toward_the_data_by_the_exact_sigma_ratio():
    sched = EDMNoiseScheduler(sigma_min=0.002, sigma_max=80.0)
    model = PhysicsInformedDiffusion(x_dim=8, denoiser=_OracleX0(), scheduler=sched)
    torch.manual_seed(1)
    x_T = torch.randn(3, 8) * 80.0
    torch.manual_seed(1)
    out = model.sample(n_samples=3, n_steps=12)
    assert torch.allclose(out - C, (x_T - C) * (0.002 / 80.0), rtol=1e-4, atol=1e-6)   # analytic


def test_ddim_sampler_with_the_exact_noise_predictor_lands_on_the_data():
    sched = VPNoiseScheduler(T=200)
    model = PhysicsInformedDiffusion(x_dim=8, denoiser=_OracleEps(sched), scheduler=sched)
    torch.manual_seed(2)
    out = model.sample(n_samples=4, n_steps=25)
    assert torch.allclose(out, C.expand(4, 8), atol=5 * sched.sigma(torch.tensor([0])).item())   # sigma(0) = 0.01
    traj = model.sample(n_samples=2, n_steps=6, return_trajectory=True)
    assert len(traj) == 5 and all(s.shape == (2, 8) for s in traj)
    mean, std = model.sample_ensemble(n_samples=16, n_steps=25)
    assert torch.allclose(mean, C, atol=0.05) and torch.all(std < 0.05)          # every sample at the data


def test_model_construction_forward_loss_and_predict():
    cfg = PIDiffConfig(x_dim=8, hidden=None, scheduler_type="edm")
    assert cfg.hidden == [256, 256, 256]
    m = PhysicsInformedDiffusion(config=PIDiffConfig(x_dim=8, hidden=[16, 16], scheduler_type="edm"))
    assert isinstance(m.scheduler, EDMNoiseScheduler) and m.denoiser.predict_x0
    v = PhysicsInformedDiffusion(x_dim=8, hidden=[16, 16], T=50)
    assert isinstance(v.scheduler, VPNoiseScheduler) and not v.denoiser.predict_x0
    out = v(torch.randn(3, 8), torch.tensor([1.0, 2.0, 3.0]))
    assert out.y.shape == (3, 8) and float(out.losses["total"]) == 0.0
    with pytest.raises(ValueError):
        v(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    loss = v.make_loss()
    assert isinstance(loss, DSMLoss) and math.isfinite(loss(torch.randn(4, 8)).item())
    assert v.predict(torch.zeros(2, 8)).shape == (2, 8)
