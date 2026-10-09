"""4D-Var through autograd and exact conservation projection for surrogates."""
import pytest

torch = pytest.importorskip("torch")

from pinneapple_analysis.data_assimilation import (  # noqa: E402
    Observation,
    Var4D,
    gradient_check,
    lorenz96_step,
    twin_experiment,
)
from pinneapple_physics.conservation import (  # noqa: E402
    MODES,
    ConservationProjection,
    integral,
    project_integral,
)


def test_autograd_adjoint_matches_finite_differences():
    var = Var4D(lambda x: lorenz96_step(x), 1.0)
    g = torch.Generator().manual_seed(0)
    xb = 8 + torch.randn(40, generator=g, dtype=torch.float64)
    obs = [Observation(5, 8 + torch.randn(40, generator=g, dtype=torch.float64), R=0.5),
           Observation(10, 8 + torch.randn(20, generator=g, dtype=torch.float64), H=lambda s: s[::2], R=torch.full((20,), 2.0, dtype=torch.float64))]
    assert gradient_check(var, xb + 0.3, xb, obs, 10)["relative_error"] < 1e-6


def test_lorenz96_twin_experiment_analysis_beats_background():
    r = twin_experiment(n=20, window_steps=10, obs_every=2, seed=3)
    res = r["result"]
    assert res.cost_final < 0.5 * res.cost_initial
    assert r["rmse_analysis_end"] < 0.5 * r["rmse_background_end"]
    assert r["rmse_analysis_end"] < 1.0                  # below the observation error std


def test_full_covariance_background_is_supported():
    n = 8
    L = torch.eye(n, dtype=torch.float64) + 0.3 * torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
    var = Var4D(lambda x: x, L)
    xb = torch.zeros(n, dtype=torch.float64)
    y = torch.ones(n, dtype=torch.float64)
    res = var.analyse(xb, [Observation(0, y, R=1.0)], 0)
    B = L @ L.T
    expected = B @ torch.linalg.solve(B + torch.eye(n, dtype=torch.float64), y)    # BLUE for identity model
    assert torch.allclose(res.x0, expected, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_projection_hits_the_target_exactly_and_is_differentiable(mode):
    torch.manual_seed(0)
    w = (torch.rand(16, 32) + 0.1).double()
    f = torch.randn(4, 16, 32, dtype=torch.float64)
    f = (f.abs() if mode != "additive" else f).requires_grad_(True)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    g = project_integral(f, w, target, mode)
    assert torch.allclose(integral(g, w, 2), target, atol=1e-12)
    g.square().sum().backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()
    if mode == "positive":
        assert (g >= 0).all()


def test_projection_wrapper_keeps_a_rollout_mass_constant():
    torch.manual_seed(0)
    w = torch.rand(64).double() + 0.5
    net = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64)).double()
    model = ConservationProjection(net, w, mode="additive")
    x = torch.rand(3, 64, dtype=torch.float64)
    m0 = integral(x, w)
    for _ in range(50):
        x = model(x)
    assert torch.allclose(integral(x, w), m0, atol=1e-10)
    assert not torch.allclose(integral(net(x), w), m0, atol=1e-3)    # the bare network does not conserve
