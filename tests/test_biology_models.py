"""Biological models, ODE identifiability, excitable-media references and the fixed presets."""
import numpy as np
import pytest

from pinneapple_analysis.inverse_problems.forward import (
    Observations,
    Parameter,
    ParameterSet,
    jacobian,
)
from pinneapple_analysis.inverse_problems.identifiability import analyze_identifiability
from pinneapple_analysis.inverse_problems.ode_model import ODEForwardModel
from pinneapple_physics.biology import ODESystem
from pinneapple_physics.biology.excitable import (
    activation_times,
    aliev_panfilov_2d,
    conduction_velocity,
    fitzhugh_nagumo_1d,
)
from pinneapple_physics.biology.models import (
    minimal_pbpk,
    pk_one_compartment_mm,
    ultradian_glucose_insulin,
)


def _fisher(system, names, t, observable, sigma=1.0):
    m = ODEForwardModel(system, [observable], names, max_step=1.0)
    ps = ParameterSet(m.default_parameters())
    obs = Observations({"t": t}, sigma, observable=np.array([observable] * len(t), object))
    _, J, how = jacobian(m, ps, ps.z0(), obs)
    assert how == "model sensitivities"
    return J.T @ J


def test_ultradian_scaling_invariance_is_a_null_direction():
    s = ultradian_glucose_insulin()
    t = np.arange(0, 1801, 10.0)
    F = _fisher(s, ["U0", "Um", "C3"], t, "G")
    w, V = np.linalg.eigh(F)
    assert w[0] / w[-1] < 1e-10                                  # U0, Um, C3 scale together (Eq 7)
    np.testing.assert_allclose(np.abs(V[:, 0]), 1 / np.sqrt(3), atol=1e-6)
    F2 = _fisher(s, ["alpha", "C5"], t, "G")
    w2, _ = np.linalg.eigh(F2)
    assert w2[0] / w2[-1] > 1e-3                                 # broken in the original system


def test_ode_sensitivities_match_finite_differences():
    s = pk_one_compartment_mm()
    m = ODEForwardModel(s, ["C"], ["Vmax", "Km", "ka"])
    ps = ParameterSet(m.default_parameters())
    obs = Observations({"t": np.linspace(0.5, 24, 12)}, 0.05, observable=np.array(["C"] * 12, object))
    _, Jm, _ = jacobian(m, ps, ps.z0(), obs, method="model")
    _, Jf, _ = jacobian(m, ps, ps.z0(), obs, method="fd")
    np.testing.assert_allclose(Jm, Jf, rtol=2e-3, atol=1e-6)


def test_mm_pk_identifiability_and_profile():
    s = pk_one_compartment_mm()
    t = np.array([0.5, 1, 2, 4, 6, 8, 12, 24.0])
    true = s.simulate(t)
    obs = Observations({"t": t}, 0.02, values=true["C"], observable=np.array(["C"] * len(t), object))
    m = ODEForwardModel(s, ["C"], ["Vmax", "Km", "V"])
    rep = analyze_identifiability(m, obs, profile=True, posterior=False, n_profile=9)
    assert rep.rank == 3
    assert set(rep.profiles) == {"Vmax", "Km", "V"}


def test_pbpk_mass_balance_and_auc():
    pb = minimal_pbpk(dose=100.0)
    t = np.linspace(0, 300, 30001)
    r = pb.simulate(t)
    vol = {k: pb.parameters[v] for k, v in pb.meta["volumes"].items()}
    total = sum(vol[k] * r[k] for k in vol) + r["A_elim"]
    np.testing.assert_allclose(total, 100.0, rtol=1e-6)
    auc = np.trapezoid(r["C_plasma_art"], t)
    assert auc == pytest.approx(100.0 / (r["CL_ws"][0] + pb.parameters["CLr"]), rel=2e-3)


def test_breakpoints_and_meal_response():
    s = ultradian_glucose_insulin()
    assert s.breakpoints() == [300.0, 650.0, 1100.0]
    r = s.simulate(np.arange(0, 1800, 1.0), max_step=1.0)
    g = r["G_conc"]
    assert g[0] == pytest.approx(110.0)
    assert g[300:420].max() > g[299] + 30                         # the first meal raises glucose


def test_fitzhugh_nagumo_reference_converges():
    a, b = fitzhugh_nagumo_1d(nx=201), fitzhugh_nagumo_1d(nx=401)
    xq, tq = np.random.default_rng(0).uniform(0, 1, 500), np.random.default_rng(1).uniform(0, 2, 500)
    A, B = a.interpolate(xq, tq), b.interpolate(xq, tq)
    assert np.linalg.norm(A["v"] - B["v"]) / np.linalg.norm(B["v"]) < 1e-5


def test_arena_fitzhugh_nagumo_uses_the_numerical_reference():
    from pinneapple_arena.problems import FitzHughNagumo1D

    p = FitzHughNagumo1D()
    out = p.analytical(np.array([0.2, 0.8]), np.array([1.0, 1.0]))
    ref = fitzhugh_nagumo_1d(nx=801).interpolate(np.array([0.2, 0.8]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(out["v"], ref["v"], atol=1e-5)


def test_aliev_panfilov_planar_wave_speed_is_converged():
    speeds = []
    for n in (241, 481):                     # dx = 0.25, 0.125: speeds 1.383, 1.393 (dx = 1 gives 1.28)
        sol = aliev_panfilov_2d(Lx=60, Ly=2, nx=n, ny=5, t_end=40, save_every=0.1)
        T = sol.activation_times()
        dx = sol.x[1] - sol.x[0]
        cv = conduction_velocity(T, dx, sol.y[1] - sol.y[0])
        mid = cv[:, (sol.x > 20) & (sol.x < 40)]
        speeds.append(float(np.nanmedian(mid)))
        assert np.all(np.isfinite(T[:, sol.x < 50]))               # the wave crosses the strip
    assert abs(speeds[0] - speeds[1]) / speeds[1] < 0.01


def test_activation_times_interpolates_crossing():
    t = np.array([0.0, 1.0, 2.0])
    u = np.array([[0.0], [0.25], [0.75]])
    assert activation_times(t, u, 0.5)[0] == pytest.approx(1.5)


def test_reaction_diffusion_2d_preset_has_the_reaction():
    from pinneapple_physics.pde_environment.presets.industry import (
        reaction_diffusion_2d_default,
        reaction_diffusion_2d_exact,
    )

    spec = reaction_diffusion_2d_default()
    assert spec.pde.kind == "reaction_diffusion_2d" and spec.pde.params["lambda"] == 0.5
    X = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]])
    u = reaction_diffusion_2d_exact(X)
    assert u[0, 0] == pytest.approx(1.0)
    assert u[1, 0] == pytest.approx(np.exp(-(2 * np.pi**2 * 1e-2 + 0.5)))


def test_reaction_kinetics_problem_spec_from_ode_system():
    torch = pytest.importorskip("torch")
    s = ODESystem.from_strings("decay", {"x": "-k*x"}, {"x": 1.0}, {"k": 0.5})
    spec = s.to_problem_spec(4.0)
    assert spec.pde.kind == "reaction_kinetics_network"
    t = torch.linspace(0, 4, 5)[:, None]
    rates = spec.pde.meta["rate_fn"]({"x": torch.exp(-0.5 * t)}, {"t": t}, {})
    np.testing.assert_allclose(rates["x"].numpy(), (-0.5 * torch.exp(-0.5 * t)).numpy(), rtol=1e-6)


def test_parameter_rejects_bad_bounds_for_ode_model():
    s = pk_one_compartment_mm()
    with pytest.raises(KeyError):
        ODEForwardModel(s, ["nope"], ["Vmax"])
    assert isinstance(ODEForwardModel(s, ["C"], ["Vmax"]).default_parameters()[0], Parameter)
