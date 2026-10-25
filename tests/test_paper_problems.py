"""References of pinneapple_physics.paper_problems satisfy their equations, initial and boundary data."""
import numpy as np
import pytest

from pinneapple_physics import paper_problems as P

H = 1e-4


def _d(f, x, t, wrt, order=1):
    if wrt == "x":
        g = lambda s: f(x + s, t)  # noqa: E731
    else:
        g = lambda s: f(x, t + s)  # noqa: E731
    if order == 1:
        return (g(H) - g(-H)) / (2 * H)
    return (g(H) - 2 * g(0.0) + g(-H)) / H ** 2


def _pts(case, n=40):
    rng = np.random.default_rng(1)
    x = rng.uniform(*case.x_range, n)
    t = rng.uniform(case.t_range[0] + 0.01, case.t_range[1], n)
    return x, t


def test_convection_reference():
    c = P.pinnsformer_case("convection")
    x, t = _pts(c)
    r = _d(c.exact, x, t, "t") + 50 * _d(c.exact, x, t, "x")
    assert np.max(np.abs(r)) < 1e-5 * 50 * 10                 # FD truncation ~ beta^3 h^2 / 6
    assert np.allclose(c.ic(x), np.sin(x))
    tt = np.linspace(0, 1, 7)
    assert np.allclose(c.exact(0 * tt, tt), c.exact(0 * tt + 2 * np.pi, tt))       # periodic


def test_reaction_reference():
    c = P.pinnsformer_case("reaction")
    x, t = _pts(c)
    u = c.exact(x, t)
    r = _d(c.exact, x, t, "t") - 5 * u * (1 - u)
    assert np.max(np.abs(r)) < 1e-6
    assert np.allclose(c.ic(x), np.exp(-(x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2)))
    tt = np.linspace(0, 1, 7)
    assert np.allclose(c.exact(0 * tt, tt), c.exact(0 * tt + 2 * np.pi, tt))


def test_wave_reference():
    c = P.pinnsformer_case("wave")
    x, t = _pts(c)
    r = _d(c.exact, x, t, "t", 2) - 4 * _d(c.exact, x, t, "x", 2)
    assert np.max(np.abs(r)) < 1e-3 * np.max(np.abs(_d(c.exact, x, t, "t", 2)))
    assert np.allclose(c.ic(x), np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x))
    assert np.max(np.abs(_d(c.exact, x, 0 * x, "t"))) < 1e-8                           # u_t(x, 0) = 0
    tt = np.linspace(0, 1, 7)
    assert np.allclose(c.exact(0 * tt, tt), 0) and np.allclose(c.exact(0 * tt + 1, tt), 0, atol=1e-12)


def test_grid_matches_official_ordering():
    g = P.tensor_grid((0, 1), (0, 2), 3, 4)
    assert g["res"].shape == (12, 2)
    assert np.allclose(g["res"][:3, 1], 0) and np.allclose(g["res"][:3, 0], [0, 0.5, 1])  # rows are times
    assert np.allclose(g["left"][:, 1], 0) and np.allclose(g["lower"][:, 0], 0) and np.allclose(g["upper"][:, 0], 1)
    s = P.time_sequence(g["res"], k=5, dt=1e-4)
    assert s.shape == (12, 5, 2) and np.allclose(s[:, 4, 1] - s[:, 0, 1], 4e-4) and np.allclose(s[:, :, 0], s[:, :1, 0])


@pytest.mark.parametrize("name", sorted(P.PINNSFORMER_CASES))
def test_test_grid(name):
    X, u = P.pinnsformer_case(name).test()
    assert X.shape == (101 * 101, 2) and u.shape == (101 * 101,) and np.all(np.isfinite(u))
