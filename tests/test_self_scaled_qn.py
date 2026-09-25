"""SSBFGS / SSBroyden (Urban, Stefanou & Pons, JCP 2025, arXiv:2405.04230): formulas and convergence."""
import math

import pytest
import torch

from pinneapple_neural.trainer.self_scaled_qn import SelfScaledQuasiNewton, self_scaled_parameters


def _pair(seed=0, n=6):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    A = A @ A.T + n * torch.eye(n, dtype=torch.float64)  # SPD Hessian of a quadratic
    H = torch.eye(n, dtype=torch.float64)
    grad = torch.randn(n, generator=g, dtype=torch.float64)
    alpha = 0.3
    s = -alpha * H.mv(grad)
    y = A.mv(s)
    return H, grad, alpha, s, y


def _update(H, s, y, tau, phi):
    Hy = H.mv(y)
    yHy, ys = y @ Hy, y @ s
    v = torch.sqrt(yHy) * (s / ys - Hy / yHy)
    return (H - torch.outer(Hy, Hy) / yHy + phi * torch.outer(v, v)) / tau + torch.outer(s, s) / ys


@pytest.mark.parametrize("variant", ["bfgs", "ssbfgs", "ssbroyden"])
def test_update_keeps_secant_equation_and_positive_definiteness(variant):
    H, grad, alpha, s, y = _pair()
    tau, phi = self_scaled_parameters(s, y, grad, alpha, H.mv(y), H.shape[0], variant)
    H1 = _update(H, s, y, tau, phi)
    assert torch.allclose(H1.mv(y), s, atol=1e-10)  # v.y = 0, so every member satisfies H y = s
    assert torch.linalg.eigvalsh(0.5 * (H1 + H1.T)).min() > 0


def test_bfgs_is_tau_one_phi_one_and_ssbfgs_matches_eq_11():
    H, grad, alpha, s, y = _pair(seed=1)
    assert self_scaled_parameters(s, y, grad, alpha, H.mv(y), 6, "bfgs") == (1.0, 1.0)
    tau, phi = self_scaled_parameters(s, y, grad, alpha, H.mv(y), 6, "ssbfgs")
    s_Hinv_s = s @ torch.linalg.solve(H, s)  # the explicit inverse Appendix B avoids
    assert tau == pytest.approx(min(1.0, float((y @ s) / s_Hinv_s)), rel=1e-12)
    assert phi == 1.0


@pytest.mark.parametrize("variant", ["bfgs", "ssbfgs", "ssbroyden"])
def test_rosenbrock_converges(variant):
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64, requires_grad=True)
    opt = SelfScaledQuasiNewton([x], variant=variant, max_iter=300)

    def closure():
        opt.zero_grad()
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        f.backward()
        return f

    opt.step(closure)
    assert torch.allclose(x.detach(), torch.ones(2, dtype=torch.float64), atol=1e-6)


def test_small_pinn_beats_tolerance_quickly():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.Tanh(), torch.nn.Linear(10, 1)).double()
    x = torch.linspace(-1, 1, 64, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    xb = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    f = (-(math.pi ** 2) * torch.sin(math.pi * x)).detach()
    opt = SelfScaledQuasiNewton(net.parameters(), variant="ssbroyden", max_iter=300)

    def closure():
        opt.zero_grad()
        u = net(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        loss = ((d2u - f) ** 2).mean() + (net(xb) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    xt = torch.linspace(-1, 1, 501, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        ref = torch.sin(math.pi * xt)  # closed-form solution of u'' = -pi^2 sin(pi x), u(+-1) = 0
        rel = torch.linalg.norm(net(xt) - ref) / torch.linalg.norm(ref)
    assert rel < 1e-3
