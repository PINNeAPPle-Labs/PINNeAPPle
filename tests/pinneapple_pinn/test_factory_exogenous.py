"""PINNFactory with exogenous signals (S(t), Tamb(t)), constants and runtime constants (T0).

The motivating spec is the lumped thermal model dT/dt - a*S + b*(T - Tamb) = 0 of the
Open-Meteo climate-twin notebook, which used to fail at loss compilation ("Unsupported by
TorchPrinter: Tamb") and, worse, silently compiled S(t) to t (sympy.S is a SymPy built-in).
"""
import math

import pytest
import sympy as sp
import torch

from pinneapple_physics.pinn_solver.factory import (
    PINN, NeuralNetwork, PINNFactory, PINNProblemSpec, SympyTorchCompiler, TabulatedSignal,
)

NOTEBOOK_SPEC = dict(
    pde_residuals=["Derivative(T(t), t) - a*S(t) + b*(T(t) - Tamb(t))"],
    conditions=[{"name": "initial_condition", "equation": "T(t) - T0", "weight": 1.0}],
    independent_vars=["t"],
    dependent_vars=["T"],
    inverse_params=["a", "b"],
    loss_weights={"pde": 1.0, "conditions": 10.0, "data": 10.0},
)


class Known(torch.nn.Module):
    """T(t) = sin(t), with trainable-looking inverse params a, b."""

    def __init__(self, a=0.7, b=0.3):
        super().__init__()
        self.inverse_params = torch.nn.ParameterDict({
            "a": torch.nn.Parameter(torch.tensor(a)), "b": torch.nn.Parameter(torch.tensor(b))})

    def forward(self, t):
        return torch.sin(t)


def test_user_names_never_resolve_to_sympy_builtins():
    c = SympyTorchCompiler(["t"], ["T"], ["a"])
    ce = c.compile("a*S(t) + E + I", strict=False)
    assert ce.exogenous == ["S"]
    assert sorted(ce.runtime_constants) == ["E", "I"]
    assert sp.Symbol("t") not in ce.expr.free_symbols  # S(t) must not collapse to t


def test_notebook_spec_compiles_and_explains_itself(capsys):
    f = PINNFactory(PINNProblemSpec(**NOTEBOOK_SPEC, verbose=True))
    assert f.exogenous_vars == ["S", "Tamb"]
    assert f.runtime_constants == ["T0"]
    out = capsys.readouterr().out
    assert "Exogenous Signals" in out and "T0" in out


def test_residual_matches_hand_computation():
    f = PINNFactory(PINNProblemSpec(**{**NOTEBOOK_SPEC, "conditions": []}))
    loss_fn = f.generate_loss_function()
    t = torch.linspace(0, 1, 40).reshape(-1, 1).requires_grad_(True)
    S = TabulatedSignal([0.0, 0.5, 1.0], [0.0, 2.0, 1.0])
    Tamb = lambda tt: 0.1 * tt  # any callable f(*inputs) works
    m = Known()
    _, comps = loss_fn(m, {"collocation": (t,), "exogenous": {"S": S, "Tamb": Tamb}})
    with torch.no_grad():
        td = t.detach()
        r = torch.cos(td) - 0.7 * S(td) + 0.3 * (torch.sin(td) - 0.1 * td)
    assert comps["pde"] == pytest.approx(float((r ** 2).mean()), rel=1e-5)


def test_runtime_constant_from_batch_and_clear_errors():
    f = PINNFactory(PINNProblemSpec(**NOTEBOOK_SPEC))
    loss_fn = f.generate_loss_function()
    t = torch.rand(10, 1, requires_grad=True)
    t0 = torch.zeros(1, 1, requires_grad=True)
    sig = {"S": TabulatedSignal([0, 1], [0, 1]), "Tamb": TabulatedSignal([0, 1], [0, 0])}
    _, comps = loss_fn(Known(), {"collocation": (t,), "conditions": [(t0,)], "exogenous": sig,
                                 "constants": {"T0": 0.5}})
    assert comps["conditions"] == pytest.approx(0.25, rel=1e-6)  # (sin(0) - 0.5)^2
    with pytest.raises(KeyError, match="batch\\['constants'\\]"):
        loss_fn(Known(), {"collocation": (t,), "conditions": [(t0,)], "exogenous": sig})
    with pytest.raises(KeyError, match="no provider"):
        loss_fn(Known(), {"collocation": (t,)})
    with pytest.raises(ValueError, match="rows"):
        loss_fn(Known(), {"collocation": (t,), "exogenous": {"S": torch.zeros(3, 1), "Tamb": torch.zeros(10, 1)}})


def test_declared_constants_and_tensor_providers():
    f = PINNFactory(PINNProblemSpec(**NOTEBOOK_SPEC, constants={"T0": 0.5}))
    assert f.runtime_constants == []
    loss_fn = f.generate_loss_function()
    t = torch.rand(10, 1, requires_grad=True)
    t0 = torch.zeros(1, 1, requires_grad=True)
    total, comps = loss_fn(Known(), {
        "collocation": (t,),
        "exogenous": {"S": torch.ones(10, 1), "Tamb": torch.zeros(10)},
        "conditions": [{"inputs": (t0,), "exogenous": {}}],
    })
    assert torch.isfinite(total) and comps["conditions"] == pytest.approx(0.25, rel=1e-6)


def test_undeclared_name_still_rejected_by_strict_compiler():
    with pytest.raises(ValueError, match="Undeclared name"):
        SympyTorchCompiler(["t"], ["T"]).compile("T(t) - T0")
    with pytest.raises(ValueError, match="derivatives of exogenous"):
        SympyTorchCompiler(["t"], ["T"]).compile("Derivative(S(t), t) - T(t)")


def test_inverse_identification_recovers_a_and_b():
    """Synthetic twin: exact solution of dT/dt = a S - b (T - Tamb); the factory loss recovers a, b."""
    torch.manual_seed(0)
    a_true, b_true = 1.5, 2.0
    Sfun = lambda t: 1.0 + torch.sin(2 * math.pi * t)
    Tamb_fun = lambda t: 0.2 * torch.ones_like(t)
    # integrate the ODE finely (RK4) to build the "observed" temperature
    n = 2001
    tt = torch.linspace(0, 1, n, dtype=torch.float64)
    T = torch.zeros(n, dtype=torch.float64)
    rhs = lambda t, y: a_true * Sfun(t) - b_true * (y - 0.2)
    h = float(tt[1] - tt[0])
    for i in range(n - 1):
        t_, y = tt[i], T[i]
        k1 = rhs(t_, y); k2 = rhs(t_ + h / 2, y + h * k1 / 2); k3 = rhs(t_ + h / 2, y + h * k2 / 2); k4 = rhs(t_ + h, y + h * k3)
        T[i + 1] = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    idx = torch.arange(0, n, 40)
    t_obs = tt[idx].float().reshape(-1, 1)
    T_obs = T[idx].float().reshape(-1, 1)

    s_grid = torch.linspace(0, 1, 201)
    spec = PINNProblemSpec(**{**NOTEBOOK_SPEC, "loss_weights": {"pde": 1.0, "conditions": 10.0, "data": 100.0}})
    loss_fn = PINNFactory(spec).generate_loss_function()
    model = PINN(NeuralNetwork(1, 1, 3, 32, torch.nn.Tanh()), ["a", "b"], {"a": 0.1, "b": 0.1})
    batch = {
        "collocation": (torch.rand(256, 1).requires_grad_(True),),
        "conditions": [(torch.zeros(1, 1, requires_grad=True),)],
        "data": ((t_obs,), T_obs),
        "exogenous": {"S": TabulatedSignal(s_grid, Sfun(s_grid)), "Tamb": TabulatedSignal(s_grid, Tamb_fun(s_grid))},
        "constants": {"T0": 0.0},
    }
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(4000):
        opt.zero_grad()
        loss, _ = loss_fn(model, batch)
        loss.backward()
        opt.step()
    a_hat = float(model.inverse_params["a"].detach())
    b_hat = float(model.inverse_params["b"].detach())
    assert a_hat == pytest.approx(a_true, rel=0.05)
    assert b_hat == pytest.approx(b_true, rel=0.05)
