"""pinneapple_physics.relativity.warp: the Alcubierre metric, its expansion, its curvature, and the energy
conditions it violates -- checked against the closed form, symbolically and numerically."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.relativity import warp

#: the wall is where ``df/dr`` is largest: ``|r - R| < wall``
BUBBLE = warp.WarpMetric(velocity=1.0, radius=10.0, wall=2.0)


def wall_points(count: int = 40, seed: int = 0) -> np.ndarray:
    """Random points inside the wall (``8 < r < 12`` for the default bubble), away from the shape's flat parts."""
    rng = np.random.default_rng(seed)
    radius = rng.uniform(8.0, 12.0, count)
    cosine = rng.uniform(-1.0, 1.0, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    sine = np.sqrt(1.0 - cosine ** 2)
    xyz = np.stack([radius * cosine, radius * sine * np.cos(angle), radius * sine * np.sin(angle)], axis=1)
    return np.concatenate([np.zeros((count, 1)), xyz], axis=1)


def closed_form(metric: warp.WarpMetric, points: np.ndarray, time: float = 0.0) -> np.ndarray:
    """Alcubierre's ``T^{00} = -(v_s^2/32 pi) (y^2 + z^2)/r_s^2 (df/dr_s)^2``."""
    r = metric.radius_at(points, time)
    derivative = warp.shape_derivative(r, metric.radius, metric.wall, metric.kind)
    return -(metric.velocity ** 2 / (32.0 * np.pi)) * (points[:, 2] ** 2 + points[:, 3] ** 2) / r ** 2 * derivative ** 2


# ------------------------------------------------------------------------------------------ the shape function

def test_shape_function_limits():
    """``f`` is 1 at the centre, falls monotonically through the wall, and is 0 outside."""
    r = np.linspace(0.0, 60.0, 601)
    f = warp.shape(r, 10.0, 2.0)
    assert f[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(f) <= 1e-15)
    assert f[np.searchsorted(r, 50.0)] == 0.0            # tanh saturates exactly in double precision
    assert np.all(f >= 0.0) and np.all(f <= 1.0)


def test_shape_derivative_matches_a_finite_difference():
    r = np.linspace(5.0, 15.0, 41)
    step = 1e-6
    numeric = (warp.shape(r + step, 10.0, 2.0) - warp.shape(r - step, 10.0, 2.0)) / (2.0 * step)
    analytic = warp.shape_derivative(r, 10.0, 2.0)
    assert np.allclose(numeric, analytic, atol=1e-8)


def test_the_wall_thickness_only_bounds_the_alcubierre_shape():
    """``wall`` is a parameter of the tanh profile; the top-hat and Gaussian shapes do not use it."""
    r = np.array([3.0, 10.0, 17.0])
    assert np.allclose(warp.shape(r, 10.0, 0.5, "gaussian"), warp.shape(r, 10.0, 7.0, "gaussian"))
    assert np.allclose(warp.shape(r, 10.0, 0.5, "tophat"), warp.shape(r, 10.0, 7.0, "tophat"))
    assert not np.allclose(warp.shape(r, 10.0, 0.5), warp.shape(r, 10.0, 7.0))


def test_unknown_shape_and_bad_parameters_are_rejected():
    with pytest.raises(ValueError, match="unknown shape"):
        warp.shape(np.array([1.0]), 10.0, 2.0, "sombrero")
    with pytest.raises(ValueError, match="unknown shape"):
        warp.shape_derivative(np.array([1.0]), 10.0, 2.0, "sombrero")
    with pytest.raises(ValueError, match="radius"):
        warp.WarpMetric(radius=0.0)
    with pytest.raises(ValueError, match="wall"):
        warp.WarpMetric(wall=-1.0)
    with pytest.raises(ValueError, match="moving along"):
        warp.WarpMetric(velocity=-1.0)


# ------------------------------------------------------------------------------------------ the metric

def test_metric_has_the_alcubierre_form():
    points = wall_points(5)
    g = BUBBLE.metric(points)
    f = BUBBLE.f(points)
    assert np.allclose(g[:, 0, 0], -1.0 + f ** 2)
    assert np.allclose(g[:, 0, 1], -f)
    assert np.allclose(g[:, 1, 0], -f)
    for i in (1, 2, 3):
        assert np.allclose(g[:, i, i], 1.0)
    assert np.allclose(g, np.swapaxes(g, 1, 2))


def test_the_lapse_is_one_and_the_shift_is_minus_the_velocity_times_f():
    """The whole ADM structure the closed form is derived from: flat spatial metric, ``N = 1``, ``beta_x = -v f``."""
    points = wall_points(6)
    g = BUBBLE.metric(points)
    g_inv = np.linalg.inv(g)
    f = BUBBLE.f(points)
    assert np.allclose(g_inv[:, 0, 0], -1.0)                     # 1 / g^{00} = -N^2 = -1
    assert np.allclose(g[:, 0, 1], -BUBBLE.velocity * f)          # beta_x = g_{0x}
    assert np.allclose(g[:, 1:, 1:], np.einsum("nij->nij", np.broadcast_to(np.eye(3), (len(points), 3, 3))))


def test_the_bubble_moves_at_the_bubble_velocity():
    """At ``t`` the centre is at ``x_s = v_s t`` and the geometry is otherwise unchanged."""
    moving = warp.WarpMetric(velocity=0.5, radius=10.0, wall=2.0)
    near = np.array([[0.0, 100.0, 0.0, 0.0]])
    assert float(moving.f(near, time=200.0)[0]) == pytest.approx(1.0, abs=1e-9)
    assert float(moving.f(near, time=0.0)[0]) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------------------------ the expansion

def test_expansion_contracts_ahead_and_expands_behind():
    """``theta = v_s (x - x_s)/r_s f'(r_s)``: negative in front of the bubble, positive behind it."""
    x = np.array([-10.0, 0.0, 10.0])
    points = np.stack([np.zeros(3), x, np.zeros(3), np.zeros(3)], axis=1)
    theta = warp.expansion(BUBBLE, points)
    assert theta[2] == pytest.approx(-0.25, abs=1e-3)             # at x = +R: contraction ahead
    assert theta[0] == pytest.approx(+0.25, abs=1e-3)             # at x = -R: expansion behind
    assert theta[1] == pytest.approx(0.0, abs=1e-15)              # nothing at the centre plane
    assert theta[2] == pytest.approx(-theta[0], abs=1e-12)        # antisymmetric
    assert np.abs(warp.expansion(BUBBLE, points)).max() == pytest.approx(0.25, rel=0.01)


def test_expansion_matches_the_york_time_taken_from_the_metric():
    """The analytic ``theta`` is the same object the curvature routine reads off ``d_i g_{0i}``."""
    points = wall_points(30, seed=4)
    curvature = warp.einstein_tensor(BUBBLE, points)
    assert np.allclose(curvature.expansion, warp.expansion(BUBBLE, points), atol=1e-9)


# ------------------------------------------------------------------------------------------ the curvature

def test_energy_density_reproduces_the_closed_form():
    points = wall_points(40)
    numeric = warp.energy_density(BUBBLE, points)
    expected = closed_form(BUBBLE, points)
    reference = np.abs(expected).max()
    assert np.all(np.abs(numeric - expected) < 1e-6 * reference)


def test_energy_density_is_zero_on_the_axis():
    """``T^{00}`` carries ``(y^2 + z^2)``, so the axis is not exotic at all -- and a wrong index permutation
    in the Christoffel assembly breaks exactly this."""
    on_axis = np.array([[0.0, x, 0.0, 0.0] for x in (2.0, 10.0, 10.5, 12.0, 20.0)])
    density = warp.energy_density(BUBBLE, on_axis)
    assert np.all(np.abs(density) < 1e-9 * 6.2e-4)


def test_energy_density_is_negative_in_the_wall():
    points = wall_points(40, seed=2)
    density = warp.energy_density(BUBBLE, points)
    off_axis = (points[:, 2] ** 2 + points[:, 3] ** 2) > 1.0
    assert np.any(off_axis)
    assert np.all(density[off_axis] < 0.0)


def test_the_einstein_tensor_is_symmetric():
    curvature = warp.einstein_tensor(BUBBLE, wall_points(8))
    assert np.allclose(curvature.einstein, np.swapaxes(curvature.einstein, 1, 2), atol=1e-12)


def test_the_metric_is_flat_where_the_shape_function_is_zero():
    """Well outside the bubble ``f`` is exactly 0, so the Riemann tensor vanishes there exactly."""
    far = np.array([[0.0, x, y, z] for x, y, z in ((60.0, 3.0, 2.0), (-70.0, 10.0, 8.0), (0.0, 80.0, 0.0))])
    curvature = warp.einstein_tensor(BUBBLE, far)
    assert np.all(curvature.is_flat)
    assert np.all(np.abs(curvature.einstein) < 1e-9)


def test_the_curvature_decays_outside_the_wall():
    """The tanh tail never reaches exactly 0 until ``r - R > 36 wall``, but the curvature has already fallen by
    four orders of magnitude at ``r = 20``."""
    rng = np.random.default_rng(5)
    direction = rng.normal(size=(20, 3))
    direction = 20.0 * direction / np.linalg.norm(direction, axis=1, keepdims=True)
    far = np.concatenate([np.zeros((20, 1)), direction], axis=1)
    peak_wall = np.abs(warp.einstein_tensor(BUBBLE, wall_points(20)).einstein).max()
    assert np.abs(warp.einstein_tensor(BUBBLE, far).einstein).max() < 1e-3 * peak_wall


def shell(radius: float, count: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count, 3))
    direction = radius * direction / np.linalg.norm(direction, axis=1, keepdims=True)
    return np.concatenate([np.zeros((count, 1)), direction], axis=1)


def test_the_tanh_wall_leaks_curvature_into_the_bubble():
    """A measurement that contradicts the usual picture: with a smooth wall the interior is **not** flat.

    ``f`` reaches 1 only at the centre, so the bubble carries residual curvature, and the leak grows with radius:
    0.3 % of the wall's peak at ``r = 0.1 R``, 1.6 % at ``0.3 R``, 10 % at ``0.5 R``. Only a compactly supported
    shape function has a genuinely flat interior.
    """
    peak_wall = np.abs(warp.einstein_tensor(BUBBLE, wall_points(60)).einstein).max()
    leak = []
    for radius in (1.0, 2.0, 3.0, 4.0, 5.0):
        peak = np.abs(warp.einstein_tensor(BUBBLE, shell(radius)).einstein).max()
        leak.append(peak / peak_wall)
    assert 1e-3 < leak[0] < 1e-2          # 0.26 % at r = 1
    assert 3e-3 < leak[2] < 3e-2          # 1.6 % at r = 3
    assert np.all(np.diff(leak) > 0.0)    # and it grows towards the wall
    assert float(BUBBLE.f(np.array([[0.0, 2.0, 0.0, 0.0]]))[0]) == pytest.approx(1.0, abs=1e-3)


def test_sympy_derives_the_same_einstein_tensor_from_the_metric():
    """The strongest check available: build ``g`` symbolically with an abstract ``f(r_s)``, let sympy differentiate
    it, and subtract the closed form. The difference simplifies to **0**, with no undifferentiated ``f`` and no
    ``f''`` term left over.
    """
    sp = pytest.importorskip("sympy")

    t, x, y, z = sp.symbols("t x y z", real=True)
    v = sp.symbols("v", positive=True)
    xi = sp.Symbol("xi", real=True)
    f = sp.Function("f", real=True)
    radius = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    F = f(radius)

    g = sp.Matrix(4, 4, lambda a, b: sp.S.Zero)
    g[0, 0] = -1 + v ** 2 * F ** 2
    g[0, 1] = g[1, 0] = -v * F
    g[1, 1] = g[2, 2] = g[3, 3] = 1
    coords = [t, x, y, z]
    g_inv = sp.simplify(g.inv())

    christoffel = [[[sp.S.Zero] * 4 for _ in range(4)] for _ in range(4)]
    for upper in range(4):
        for a in range(4):
            for b in range(4):
                christoffel[upper][a][b] = sp.expand(sp.Rational(1, 2) * sum(
                    g_inv[upper, s] * (sp.diff(g[a, s], coords[b]) + sp.diff(g[b, s], coords[a])
                                       - sp.diff(g[a, b], coords[s])) for s in range(4)))

    def ricci(sigma: int, nu: int):
        total = sp.S.Zero
        for rho in range(4):
            total += (sp.diff(christoffel[rho][nu][sigma], coords[rho])
                      - sp.diff(christoffel[rho][rho][sigma], coords[nu]))
            total += sum(christoffel[rho][rho][lam] * christoffel[lam][nu][sigma]
                         - christoffel[rho][nu][lam] * christoffel[lam][rho][sigma] for lam in range(4))
        return sp.expand(total)

    ricci_tensor = sp.Matrix(4, 4, ricci)
    scalar = sp.expand(sum(g_inv[i, j] * ricci_tensor[i, j] for i in range(4) for j in range(4)))
    einstein = ricci_tensor - sp.Rational(1, 2) * g * scalar
    density = sum(g_inv[0, i] * g_inv[0, j] * einstein[i, j] for i in range(4) for j in range(4)) / (8 * sp.pi)

    derivative = sp.Subs(sp.Derivative(f(xi), xi), xi, radius)
    closed = -v ** 2 / (32 * sp.pi) * (y ** 2 + z ** 2) / radius ** 2 * derivative ** 2
    assert sp.simplify(sp.expand(density - closed)) == 0


# ------------------------------------------------------------------------------------------ the energy conditions

def test_null_vectors_are_actually_null():
    """``k = n + s`` with ``n = (1, v f, 0, 0)``; ``k = (1, n)`` -- the obvious guess -- is not null here."""
    points = wall_points(10)
    for direction in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.3, -0.5, 0.81)):
        k = warp.null_vector(BUBBLE, points, direction)
        norm = np.einsum("pa,pab,pb->p", k, BUBBLE.metric(points), k)
        assert np.all(np.abs(norm) < 1e-12)
    naive = np.tile(np.array([1.0, 0.0, 1.0, 0.0]), (len(points), 1))
    assert np.any(np.abs(np.einsum("pa,pab,pb->p", naive, BUBBLE.metric(points), naive)) > 1e-3)


def test_null_energy_condition_is_violated_in_the_wall():
    points = wall_points(25, seed=8)
    result = warp.worst_null_energy(BUBBLE, points)
    assert result["violated"] is True
    assert result["points_violating"] == result["points"]
    assert result["minimum"] < -1e-3


def test_null_energy_is_the_contraction_of_the_einstein_tensor_with_the_null_vector():
    """``null_energy`` is ``G_{mu nu} k^mu k^nu / 8 pi``, computed the long way, in every direction."""
    points = wall_points(6, seed=9)
    einstein = warp.einstein_tensor(BUBBLE, points).einstein
    for direction in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)):
        k = warp.null_vector(BUBBLE, points, direction)
        expected = np.einsum("pa,pab,pb->p", k, einstein, k) / (8.0 * np.pi)
        assert np.allclose(warp.null_energy(BUBBLE, points, direction), expected, atol=1e-15)


# ------------------------------------------------------------------------------------------ the bill

def test_the_total_energy_scales_like_velocity_squared():
    slow = warp.integrated_energy(warp.WarpMetric(velocity=0.1, radius=10.0, wall=2.0), samples=31)
    fast = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=10.0, wall=2.0), samples=31)
    assert fast["total_geometric"] < 0.0
    assert fast["total_geometric"] / slow["total_geometric"] == pytest.approx(100.0, rel=0.02)


def test_a_thinner_wall_costs_more():
    """``E ~ v^2 R^2 / wall``: the negative energy is concentrated in the wall, and squeezing the wall raises
    ``f'`` faster than it shrinks the volume."""
    energies = []
    for thickness in (1.0, 2.0, 4.0):
        total = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=10.0, wall=thickness), samples=31)
        energies.append(total["total_geometric"] * thickness)
    assert np.allclose(energies, energies[0], rtol=0.12)
    thin = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=10.0, wall=1.0), samples=31)
    thick = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=10.0, wall=4.0), samples=31)
    assert thin["kilograms"] < thick["kilograms"] < 0.0


def test_the_bill_for_a_ten_metre_bubble_at_the_speed_of_light():
    """The headline number, and it is not small: roughly one Jupiter mass of *negative* energy."""
    total = warp.integrated_energy(BUBBLE, samples=61)
    assert total["kilograms"] == pytest.approx(-1.89e27, rel=0.05)
    assert total["solar_masses"] == pytest.approx(-9.5e-4, rel=0.05)
    assert total["solar_masses"] == pytest.approx(warp.solar_masses(total["kilograms"]))


def test_the_interior_of_the_bubble_still_costs_nothing():
    """Inside, ``T^{00}`` is negative but tiny compared with the wall -- the exotic matter is in the shell."""
    rng = np.random.default_rng(11)
    inside = rng.normal(size=(50, 3))
    inside = 3.0 * inside / np.linalg.norm(inside, axis=1, keepdims=True)
    points = np.concatenate([np.zeros((50, 1)), inside], axis=1)
    peak_wall = np.abs(warp.energy_density(BUBBLE, wall_points(50))).max()
    assert np.abs(warp.energy_density(BUBBLE, points)).max() < 0.02 * peak_wall


def test_curvature_to_dict():
    summary = warp.einstein_tensor(BUBBLE, wall_points(12)).to_dict()
    assert summary["points"] == 12
    assert summary["min_energy_density"] < summary["max_energy_density"]
    assert summary["min_expansion"] < 0.0 < summary["max_expansion"]
    assert 0 <= summary["flat_points"] <= summary["points"]


def test_the_tophat_shape_is_out_of_scope_and_the_module_says_so():
    """An honest limitation, measured: the original top-hat has ``df/dr = 0`` everywhere except the shell, where it
    is a delta function. Its analytic derivative is 0, so the closed form says 0, while the numerical curvature
    depends on whether a grid point happens to land on ``r = R`` -- the totals below differ by 32 orders of
    magnitude. A discontinuous profile is a surface density this representation cannot resolve, which is why the
    module's own guide uses only smooth shapes.
    """
    top_hat = warp.WarpMetric(velocity=1.0, radius=10.0, wall=2.0, kind="tophat")
    totals = [warp.integrated_energy(top_hat, samples=samples)["kilograms"] for samples in (25, 31, 41, 61, 81)]
    assert 0.0 in totals                       # one grid misses the shell entirely
    assert max(abs(value) for value in totals) > 1e30
    assert np.allclose(warp.expansion(top_hat, wall_points(10)), 0.0)     # the analytic derivative is 0
    assert math.isclose(float(top_hat.f(np.array([[0.0, 5.0, 0.0, 0.0]]))[0]), 1.0)
    assert math.isclose(float(top_hat.f(np.array([[0.0, 15.0, 0.0, 0.0]]))[0]), 0.0)
