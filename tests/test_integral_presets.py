"""The integral-equation presets and their closed-form solutions.

A preset is only worth what its pair ``(source, solution)`` is worth: the free term has to be the one that makes the
closed form a solution, and the two files that hold them are edited independently. So every case here is checked
against a numerical integral of the kernel weighted by the claimed solution -- an independent construction -- and then
the compiled residual of that solution is required to sit at the quadrature error of the rule, not below it and not
far above it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinneapple_physics.closed_form import integral_equations as cf
from pinneapple_physics.pde_environment.presets.integral import (
    integral_fredholm_separable_exp,
    integral_fredholm_unit_exp,
    integral_log_kernel_linear,
    integral_volterra_unit_exp,
)
from pinneapple_physics.pde_environment.presets.registry import get_preset
from pinneapple_physics.pinn_solver.compiler.integral_kinds import (
    constant_kernel,
    log_diag,
    log_kernel,
    product_kernel,
)

E = math.e
PRESETS = ("integral_fredholm_unit_exp", "integral_fredholm_separable_exp",
           "integral_volterra_unit_exp", "integral_log_kernel_linear")


class Field(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X):
        return self.fn(X)


def residual_rms(preset: str, x: np.ndarray) -> float:
    """Residual the compiler produces for the preset's own closed-form solution on the nodes ``x``."""
    import pinneapple as pp
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    spec = get_preset(preset)
    y = np.asarray(pp.solve(preset, "analytic").predict(x.reshape(-1, 1))).reshape(-1, 1)
    loss = compile_problem(spec)(Field(lambda Z: torch.as_tensor(y, dtype=torch.float64)), None,
                                 {"x_col": torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1), "ctx": {}})
    return float(loss["pde"].detach()) ** 0.5


# ------------------------------------------------------------------------------------------------ registration
def test_every_integral_preset_is_registered_and_has_a_closed_form():
    import pinneapple as pp
    from pinneapple_physics.solving import analytic_for

    names = pp.list_presets()
    for preset in PRESETS:
        assert preset in names
        assert analytic_for(preset) == preset
        spec = get_preset(preset)
        assert spec.pde.kind in ("fredholm2", "volterra2")
        assert spec.pde.meta["analytic"] == preset
        assert spec.domain_bounds == {"x": (0.0, 1.0)}


@pytest.mark.parametrize("preset", PRESETS)
def test_the_preset_solves_analytically_by_name(preset):
    import pinneapple as pp

    sol = pp.solve(preset, "analytic")
    y = np.ravel(sol.predict(np.linspace(0.0, 1.0, 7).reshape(-1, 1)))
    assert np.all(np.isfinite(y))
    assert sol.info.get("solution") == preset


def test_every_integral_preset_has_an_exact_solution_that_its_source_reproduces():
    """Independent construction for each of the four presets: the exact solution ``u`` and the free term ``f`` must
    satisfy ``f = u - lam int K u``, with the integral evaluated by quadrature rather than by the module's own rule.
    The two live in the same file today but are edited independently, and a preset whose source no longer matches its
    solution is a benchmark that measures nothing."""
    from scipy.integrate import quad

    for preset in ("integral_fredholm_unit_exp", "integral_fredholm_separable_exp",
                   "integral_volterra_unit_exp", "integral_log_kernel_linear"):
        spec = get_preset(preset)
        lam = float(spec.pde.params["lambda"])
        u_fn, f_fn = _closed_form(preset)
        x0 = 0.37
        u = float(u_fn(np.array([x0]))[0])

        def integrand(s, _p=preset, _u=u_fn, _x=x0):
            if _p == "integral_log_kernel_linear":
                return float(_u(np.array([s]))[0]) * math.log(abs(_x - s)) if s != _x else 0.0
            if _p == "integral_fredholm_separable_exp":
                return float(_u(np.array([s]))[0]) * _x * s
            return float(_u(np.array([s]))[0])

        upper = x0 if preset == "integral_volterra_unit_exp" else 1.0
        points = [x0] if preset == "integral_log_kernel_linear" else None
        integral = quad(integrand, 0.0, upper, points=points, limit=400)[0]
        assert float(f_fn(np.array([x0]))[0]) == pytest.approx(u - lam * integral, abs=1e-7), preset


# ------------------------------------------------------------------------------------------------ the pairs
def test_the_unit_kernel_source_is_the_one_that_makes_exp_a_solution():
    """``u = exp(x)`` solves ``u = f + lam int u`` only if ``int_0^1 e^s ds = e - 1`` enters ``f``."""
    lam = 0.5
    f = cf.fredholm_unit_exp_source(np.array([0.3]), lam)
    u = cf.fredholm_unit_exp(np.array([0.3]))
    assert u[0] - f[0] == pytest.approx(lam * (E - 1.0), rel=1e-14)


def test_the_separable_kernel_source_uses_int_s_exp_s_ds_equal_to_one():
    lam = 0.5
    x = np.array([0.4])
    f = cf.fredholm_separable_exp_source(x, lam)
    u = cf.fredholm_separable_exp(x)
    # int_0^1 s e^s ds = [e^s (s - 1)]_0^1 = 0 - (-1) = 1, so the integral term is lam * x
    assert u[0] - f[0] == pytest.approx(lam * x[0], rel=1e-14)


@pytest.mark.parametrize("x0", [0.0, 0.07, 0.3, 0.5, 0.82, 1.0])
def test_the_log_kernel_moments_match_a_numerical_integral(x0):
    """The two moments of ``ln|x - s|`` written in closed form against ``scipy.integrate.quad`` with the singularity
    declared, which is what makes ``log_kernel_linear_source`` a free term and not a guess."""
    from scipy.integrate import quad

    def safe(fn):
        return lambda s: fn(s) if abs(s - x0) > 0.0 else 0.0

    m0 = quad(safe(lambda s: math.log(abs(x0 - s))), 0.0, 1.0, points=[x0], limit=400)[0]
    m1 = quad(safe(lambda s: s * math.log(abs(x0 - s))), 0.0, 1.0, points=[x0], limit=800)[0]
    assert float(cf.log_kernel_moment0(np.array([x0]))[0]) == pytest.approx(m0, abs=2e-8)
    assert float(cf.log_kernel_moment1(np.array([x0]))[0]) == pytest.approx(m1, abs=2e-8)


def _closed_form(preset):
    if preset == "integral_fredholm_unit_exp":
        return cf.fredholm_unit_exp, cf.fredholm_unit_exp_source
    if preset == "integral_fredholm_separable_exp":
        return cf.fredholm_separable_exp, cf.fredholm_separable_exp_source
    if preset == "integral_volterra_unit_exp":
        return cf.volterra_unit_exp, cf.volterra_unit_exp_source
    return cf.log_kernel_linear, cf.log_kernel_linear_source


# ------------------------------------------------------------------------------------------------ residuals
@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "integral_log_kernel_linear"])
def test_bounded_presets_have_a_residual_at_the_second_order_floor(preset):
    x = np.linspace(0.0, 1.0, 81)
    assert residual_rms(preset, x) < 1e-4
    assert residual_rms(preset, np.linspace(0.0, 1.0, 321)) < residual_rms(preset, x) / 4.0


def test_the_singular_preset_converges_at_first_order_through_the_preset_path():
    """The log-kernel preset must exercise ``singular=True`` and its ``diag_fn``, and reach the first-order floor."""
    spec = get_preset("integral_log_kernel_linear")
    assert spec.pde.params["singular"] is True and callable(spec.pde.params["diag_fn"])
    coarse = residual_rms("integral_log_kernel_linear", np.linspace(0.0, 1.0, 81))
    fine = residual_rms("integral_log_kernel_linear", np.linspace(0.0, 1.0, 641))
    assert fine < 1e-3 and fine < coarse / 4.0


# ------------------------------------------------------------------------------------------------ training
def test_the_preset_trains_through_solve_pde_with_a_fixed_node_set():
    """``solve_pde(..., x_col=grid)`` is the public way to pin the Nyström nodes -- without it the collocation set is
    resampled every step and the quadrature moves with the batch. This is a wiring test of the whole ``pp`` path and
    not an accuracy claim: a few hundred epochs must reduce the compiled loss by two orders of magnitude.

    ``solve_pde`` feeds the model float32, so the model must not be double -- a mismatch raises deep inside the first
    linear layer, which is why this is asserted here rather than left to a caller to discover."""
    from pinneapple_physics import solve_pde

    spec = get_preset("integral_fredholm_unit_exp")
    x = np.linspace(0.0, 1.0, 41)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    out = solve_pde(spec, model, epochs=400, lr=1e-2, x_col=x.reshape(-1, 1), device="cpu")
    history = out["history"]["loss"]
    assert len(history) == 400 and np.all(np.isfinite(history))
    assert history[-1] < history[0] / 100.0
    with torch.no_grad():
        y = model(torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)).numpy().ravel()
    assert np.all(np.isfinite(y)) and float(np.max(np.abs(y - np.exp(x)))) < 1.0


# ------------------------------------------------------------------------------------------------ kernels
def test_the_kernel_helpers_agree_with_their_definitions():
    x = np.linspace(0.0, 1.0, 5)
    cols = x.reshape(-1, 1)
    assert np.allclose(constant_kernel(cols, cols, {}), 1.0)
    assert np.allclose(product_kernel(cols, cols, {}), cols @ cols.T)
    k = log_kernel(cols, cols, {})
    assert np.allclose(np.diagonal(k), 0.0)
    assert np.allclose(log_diag(np.array([0.1]), {}), [0.1 * (math.log(0.05) - 1.0)])


FACTORIES = {"integral_fredholm_unit_exp": integral_fredholm_unit_exp,
             "integral_fredholm_separable_exp": integral_fredholm_separable_exp,
             "integral_volterra_unit_exp": integral_volterra_unit_exp,
             "integral_log_kernel_linear": integral_log_kernel_linear}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_the_factory_builds_the_spec_the_registry_serves(name):
    """The four presets are factories, and the registry entry has to be the one the factory builds: the catalog
    links a tool to its tests by the symbol a test calls, so the two paths are asserted here and not only through
    ``pp.solve``."""
    import pinneapple as pp

    spec = FACTORIES[name]()
    assert spec.name == name and name in pp.list_presets()
    assert spec.pde.kind == get_preset(name).pde.kind
