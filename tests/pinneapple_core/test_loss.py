import math

import pytest

torch = pytest.importorskip("torch")

import pinneapple as pp
from pinneapple_core import loss as L


def test_terms_match_their_definitions():
    r = torch.tensor([[1.0], [-2.0], [3.0]])
    assert L.pde(r).item() == pytest.approx((1 + 4 + 9) / 3)
    assert L.pde(r, weights=torch.tensor([1.0, 0.0, 1.0])).item() == pytest.approx((1 + 9) / 2)
    assert L.boundary(torch.tensor([1.0, 3.0]), 2.0).item() == pytest.approx(1.0)
    p, t = torch.tensor([1.0, 2.0, 4.0]), torch.tensor([1.0, 3.0, 2.0])
    assert L.supervised(p, t).item() == pytest.approx((0 + 1 + 4) / 3)
    assert L.supervised(p, t, "mae").item() == pytest.approx(1.0)
    assert L.supervised(p, t, "relative_l2").item() == pytest.approx(math.sqrt(5) / math.sqrt(14))
    assert L.supervised(p, t, "huber", delta=1.0).item() == pytest.approx((0 + 0.5 + 1.5) / 3)
    assert L.conservation(torch.tensor([2.0, 2.0]), 1.0, scale=2.0).item() == pytest.approx(0.25)
    with pytest.raises(ValueError):
        L.supervised(p, t, "bogus")


def test_energy_modes():
    e = torch.tensor([1.0, 1.0, 1.5, 0.5])
    assert L.energy(e).item() == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)
    assert L.energy(e, mode="nonincrease").item() == pytest.approx(0.25 / 3)       # only the rise 1.0 -> 1.5
    assert L.energy(e, mode="nondecrease").item() == pytest.approx((0 + 0 + 1.0) / 3)  # only the drop 1.5 -> 0.5
    with pytest.raises(ValueError):
        L.energy(e, mode="x")


def test_symmetry_invariance_and_equivariance():
    f_even = lambda x: (x ** 2).sum(dim=1, keepdim=True)
    f_odd = lambda x: x ** 3
    x = torch.randn(10, 2)
    assert L.symmetry(f_even, x, lambda z: -z).item() == pytest.approx(0.0, abs=1e-12)
    assert L.symmetry(f_odd, x, lambda z: -z).item() > 0                        # odd function is not invariant
    assert L.symmetry(f_odd, x, lambda z: -z, output_transform=lambda y: -y).item() == pytest.approx(0.0, abs=1e-12)


def test_inverse_with_prior():
    pred, obs = torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0])
    theta = torch.tensor([3.0, 5.0])
    assert L.inverse(pred, obs).item() == pytest.approx(2.0)
    assert L.inverse(pred, obs, theta, prior=1.0, reg=0.5).item() == pytest.approx(2.0 + 0.5 * (4 + 16) / 2)
    assert L.inverse(pred, obs, [theta], reg=0.0).item() == pytest.approx(2.0)


def test_causal_pde_downweights_late_times():
    t = torch.linspace(0, 1, 200)[:, None]
    r = torch.where(t < 0.2, torch.tensor(5.0), torch.tensor(1.0))      # large early residual
    w = L.causal_weights(r, t, n_chunks=10, epsilon=1.0)
    assert w[0].item() == pytest.approx(1.0) and w[-1].item() < 1e-3 and (w[1:] <= w[:-1] + 1e-12).all()
    plain, causal = L.pde(r), L.pde(r, t=t, causal_epsilon=1.0, n_chunks=10)
    assert causal.item() > plain.item()                                  # early (large) residuals dominate the mean
    with pytest.raises(ValueError):
        L.pde(r, causal_epsilon=1.0)
    with pytest.raises(ValueError):
        L.pde(r, t=t, causal_epsilon=1.0, weights=w)


def test_combine_and_errors():
    terms = {"pde": torch.tensor(2.0), "bc": torch.tensor(3.0)}
    assert L.combine(terms, {"pde": 1.0, "bc": 10.0}).item() == pytest.approx(32.0)
    assert L.combine(terms).item() == pytest.approx(5.0)
    with pytest.raises(KeyError, match="not in the loss"):
        L.combine(terms, {"bcc": 1.0})
    with pytest.raises(ValueError):
        L.combine({})


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def _terms(net, x):
    u = net(x)
    return {"pde": (u ** 2).mean(), "bc": ((u - 1.0) ** 2).mean()}


def test_every_strategy_is_selectable_by_name_and_trains():
    x = torch.rand(32, 2)
    assert set(L.list_strategies()) >= {"fixed", "self_adaptive", "gradnorm", "ntk", "relobralo", "softadapt",
                                        "augmented_lagrangian", "lr_annealing", "pcgrad", "curriculum", "auto"}
    for name in L.Balancer.strategies():
        net = _net()
        kw = {"model": net}
        if name == "curriculum":
            kw.update(schedule={"bc": (0.0, 10.0)}, steps=5)
        bal = L.Balancer(name, names=["pde", "bc"], update_every=1, **kw)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        first = last = None
        for step in range(6):
            opt.zero_grad()
            terms = _terms(net, x)
            total = bal(terms, step=step, optimizer=opt)
            if name != "pcgrad":                        # PCGrad writes the projected gradients itself
                total.backward()
            opt.step()
            first = first or float(sum(v.detach() for v in terms.values()))
            last = float(sum(v.detach() for v in terms.values()))
            assert torch.isfinite(torch.as_tensor(float(total.detach()) if hasattr(total, "detach") else float(total)))
        assert last < first * 1.5, name                 # sanity: not diverging


def test_fixed_balancer_equals_combine_and_curriculum_ramps():
    net, x = _net(), torch.rand(16, 2)
    terms = _terms(net, x)
    w = {"pde": 1.0, "bc": 10.0}
    assert torch.equal(L.Balancer("fixed", names=["pde", "bc"], weights=w)(terms), L.combine(terms, w))
    cur = L.Balancer("curriculum", names=["pde", "bc"], weights={"pde": 1.0}, schedule={"bc": (0.0, 10.0)}, steps=10)
    seen = []
    for s in (0, 5, 10, 50):
        cur(terms, step=s)
        seen.append(cur.weights["bc"])
    assert seen == pytest.approx([0.0, 5.0, 10.0, 10.0])
    cos = L.Balancer("curriculum", names=["a"], schedule={"a": (1.0, 3.0)}, steps=10, shape="cosine")
    cos({"a": torch.tensor(1.0)}, step=5)
    assert cos.weights["a"] == pytest.approx(2.0)
    assert cur.history()["bc"][:2] == [0.0, 5.0]


def test_self_adaptive_exposes_weight_parameters_and_balancer_validates():
    bal = L.Balancer("self_adaptive", names=["pde", "bc"], weights={"bc": 10.0})
    assert len(bal.weight_parameters()) == 2 and L.Balancer("fixed", names=["a"]).weight_parameters() == []
    with pytest.raises(ValueError, match="pass model"):
        L.Balancer("gradnorm", names=["pde", "bc"])
    with pytest.raises(ValueError, match="unknown strategy"):
        L.Balancer("magic", names=["pde"])
    with pytest.raises(KeyError):
        L.Balancer("fixed", names=["pde"], weights={"bc": 1.0})
    with pytest.raises(KeyError, match="do not match"):
        bal({"pde": torch.tensor(1.0)})
    with pytest.raises(ValueError):
        L.Balancer("fixed", names=[])


def test_namespace_pp_loss():
    assert pp.loss.combine is L.combine and pp.loss.Balancer is L.Balancer
