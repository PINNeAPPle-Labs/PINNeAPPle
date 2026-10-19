"""MultiAdam (Yao et al., ICML 2023, arXiv:2306.02816): per-term Adam moments, scale invariance, Poisson."""

import math

import pytest
import torch

from pinneapple_neural.trainer.multiadam import MultiAdam


def test_single_term_matches_adam():
    torch.manual_seed(0)
    a = torch.randn(5, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    multi = MultiAdam([a], lr=1e-2, betas=(0.9, 0.999))
    adam = torch.optim.Adam([b], lr=1e-2, betas=(0.9, 0.999))
    for _ in range(20):
        multi.step([((a - 1.0) ** 4).sum()])
        adam.zero_grad()
        ((b - 1.0) ** 4).sum().backward()
        adam.step()
    assert torch.allclose(a, b, atol=1e-12)


def test_rescaling_one_term_does_not_change_the_step():
    torch.manual_seed(0)
    a = torch.randn(4, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    opt_a, opt_b = MultiAdam([a], eps=0.0), MultiAdam([b], eps=0.0)
    for _ in range(10):
        opt_a.step([(a**2).sum(), ((a - 2.0) ** 2).sum()])
        opt_b.step([(b**2).sum(), 1e6 * ((b - 2.0) ** 2).sum()])
    assert torch.allclose(a, b, atol=1e-10)


def test_changing_the_number_of_terms_raises():
    p = torch.zeros(2, requires_grad=True)
    opt = MultiAdam([p])
    opt.step([p.sum() ** 2, (p - 1).sum() ** 2])
    with pytest.raises(ValueError, match="expected 2"):
        opt.step([p.sum() ** 2])


def _poisson_rel_error(optimizer, seed, steps=2000, w_bc=1e-3):
    """u'' = -pi^2 sin(pi x) on [-1, 1], u(+-1) = 0, with the boundary term weighted down by ``w_bc``."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1),
    ).double()
    x = torch.linspace(-1, 1, 64, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    xb = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    f = (-(math.pi**2) * torch.sin(math.pi * x)).detach()
    opt = (
        MultiAdam(net.parameters(), lr=1e-3)
        if optimizer == "multiadam"
        else torch.optim.Adam(net.parameters(), lr=1e-3)
    )
    for _ in range(steps):
        u = net(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        loss_pde = ((d2u - f) ** 2).mean()
        loss_bc = w_bc * (net(xb) ** 2).mean()
        if optimizer == "multiadam":
            opt.step([loss_pde, loss_bc])
        else:
            opt.zero_grad()
            (loss_pde + loss_bc).backward()
            opt.step()
    xt = torch.linspace(-1, 1, 501, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        ref = torch.sin(math.pi * xt)  # closed-form solution
        return float(torch.linalg.norm(net(xt) - ref) / torch.linalg.norm(ref))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_poisson_with_badly_balanced_boundary_term_beats_adam(seed):
    # relative L2 errors against the closed-form solution u = sin(pi x)
    adam = _poisson_rel_error("adam", seed)
    multi = _poisson_rel_error("multiadam", seed)
    assert multi < 0.05
    assert multi < 0.1 * adam  # plain Adam fits the PDE term and gets the boundary values wrong
