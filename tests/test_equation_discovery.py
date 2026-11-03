"""Equation discovery (#1192): Lorenz and a damped oscillator recovered with intervals that hold the truth, Burgers
by PDE-FIND (and its weak form), a noisy field whose discovered PDE fails its rollout instead of being reported, and
a slightly noisy field whose spurious term the weak-form check names (#1224)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pinneapple_analysis.inverse_problems.discovery import discover_ode, discover_pde


def _terms(r, state):
    return {k: v for k, v in r["per_state"][state]["terms"].items()}


def test_lorenz_with_noise_recovers_its_seven_terms():
    def f(_t, s):
        x, y, z = s
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]
    t = np.arange(0, 20, 0.002)
    Y = solve_ivp(f, (0, 20), [-8, 8, 27], t_eval=t, rtol=1e-10, atol=1e-10).y.T
    Y = Y + 0.01 * Y.std(0) * np.random.default_rng(0).standard_normal(Y.shape)
    r = discover_ode(t, Y, ["x", "y", "z"], n_bootstrap=20)
    assert set(_terms(r, "x")) == {"x", "y"} and set(_terms(r, "y")) == {"x", "y", "x*z"}
    assert set(_terms(r, "z")) == {"z", "x*y"}
    assert _terms(r, "z")["z"]["coefficient"] == pytest.approx(-8 / 3, rel=0.01)
    assert all(c["status"] == "pass" for c in r["checks"])


def test_damped_oscillator_intervals_hold_the_truth():
    t = np.arange(0, 30, 0.01)
    Y = solve_ivp(lambda _t, s: [s[1], -4 * s[0] - 0.4 * s[1]], (0, 30), [1, 0], t_eval=t, rtol=1e-10).y.T
    Y = Y + 0.005 * np.random.default_rng(1).standard_normal(Y.shape)
    r = discover_ode(t, Y, ["x", "v"], n_bootstrap=40)
    v = _terms(r, "v")
    assert set(v) == {"x", "v"}
    for k, true in (("x", -4.0), ("v", -0.4)):
        assert abs(v[k]["coefficient"] - true) < 0.02 * abs(true) + 0.005


def test_burgers_by_pde_find_and_a_noisy_field_is_refused():
    from pinneapple_analysis.validation.pinn import cole_hopf
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t, indexing="ij")
    U = cole_hopf(X, T, 0.1)
    r = discover_pde(U, x, t, n_bootstrap=20)
    assert set(r["terms"]) == {"u_xx", "u*u_x"}
    assert r["terms"]["u_xx"]["coefficient"] == pytest.approx(0.1, rel=0.01)
    assert r["terms"]["u*u_x"]["coefficient"] == pytest.approx(-1.0, rel=0.01)
    noisy = discover_pde(U + 0.01 * np.random.default_rng(0).standard_normal(U.shape), x, t, n_bootstrap=10)
    assert next(c for c in noisy["checks"] if c["code"] == "discover.rollout")["status"] == "fail"
    assert all(c["status"] == "pass" for c in r["checks"])                   # the weak form agrees
    weak = discover_pde(U, x, t, n_bootstrap=20, form="weak")
    assert set(weak["terms"]) == {"u_xx", "u*u_x"} and all(c["status"] == "pass" for c in weak["checks"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_burgers_with_small_noise_the_spurious_term_is_named_and_the_weak_form_drops_it(seed):
    """#1224: with 0.1 % noise the strong form adds a decay term -c*u (c about 0.07 to 0.13) that the held-out fit,
    the bootstrap and the rollout all accept; the weak-form check names it, and the weak form finds the true
    equation."""
    from pinneapple_analysis.validation.pinn import cole_hopf
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t, indexing="ij")
    U = cole_hopf(X, T, 0.1) + 0.001 * np.random.default_rng(seed).standard_normal((256, 101))
    strong = discover_pde(U, x, t, n_bootstrap=20)
    st = {c["code"]: c for c in strong["checks"]}
    assert "u" in strong["terms"]                                            # the bias is still there in PDE-FIND
    assert st["discover.weak_form"]["status"] == "fail" and "u" in st["discover.weak_form"]["value"]["extra"]
    assert "only the strong form selects ['u'" in st["discover.weak_form"]["detail"]
    weak = discover_pde(U, x, t, n_bootstrap=20, form="weak")
    assert set(weak["terms"]) == {"u_xx", "u*u_x"}
    assert weak["terms"]["u_xx"]["coefficient"] == pytest.approx(0.1, rel=0.01)
    assert weak["terms"]["u*u_x"]["coefficient"] == pytest.approx(-1.0, rel=0.01)
    assert all(c["status"] == "pass" for c in weak["checks"]), weak["checks"]


def test_rollout_reads_powers_of_u():
    """A library label with a power (``u**2*u_x``) is split on the product only (it used to raise KeyError)."""
    from pinneapple_analysis.inverse_problems.discovery import _factors, _label_parts
    assert _factors("u**2*u_x") == ["u**2", "u_x"] and _factors("1") == []
    assert _label_parts("u**2*u_xx") == (2, 2) and _label_parts("u_xxx") == (0, 3) and _label_parts("u") == (1, 0)


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "discover.equations" in REGISTRY.types()
    assert match("descobrir a equação governante a partir dos dados (sindy)")[0][1].id == "discover.equations"
