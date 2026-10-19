"""Equation text -> SymPy (safe parser), classification, the finite-difference solver and the PINN problem built from
the same parsed system, checked against exact solutions."""
import numpy as np
import pytest

from pinneapple_physics.symbolic_pde.text import EquationSyntaxError, parse_system
from pinneapple_simulation.numerical_solvers.symbolic_fd import FDError, solve_fd

HEAT_IC = [{"kind": "initial", "equation": "u = sin(pi*x)"},
           {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}]


@pytest.mark.parametrize("pde,coords,family", [
    ("u_t = alpha*lap(u)", ["x", "t"], "heat"),
    ("u_t = alpha*lap(u)", ["x", "y", "z", "t"], "heat"),
    ("u_tt = c^2*u_xx", ["x", "t"], "wave"),
    ("u_t + u*u_x = nu*u_xx", ["x", "t"], "burgers"),
    ("u_t + c*u_x = D*u_xx", ["x", "t"], "advection_diffusion"),
    ("u_t + c*u_x = 0", ["x", "t"], "advection"),
    ("u_t = D*lap(u) + r*u*(1 - u)", ["x", "y", "t"], "reaction_diffusion"),
    ("lap(u) = -2*pi**2*sin(pi*x)*sin(pi*y)", ["x", "y"], "poisson"),
    ("lap(u) = 0", ["x", "y"], "laplace"),
    ("lap(u) + k**2*u = 0", ["x", "y"], "helmholtz"),
])
def test_classification(pde, coords, family):
    conds = [{"kind": "initial", "equation": "u = 0"}] if "t" in coords else []
    s = parse_system([pde], conds, coords=coords, fields=["u"],
                     coefficients={"alpha": 1, "c": 1, "nu": 1, "D": 1, "r": 1, "k": 1})
    assert s.classify()["family"] == family
    assert s.issues() == []


def test_lap_is_dimension_free():
    for coords, n in ((["x", "t"], 1), (["x", "y", "t"], 2), (["x", "y", "z", "t"], 3)):
        s = parse_system(["u_t = lap(u)"], HEAT_IC, coords=coords, fields=["u"])
        assert len(s.pdes[0].derivatives()) == n + 1


@pytest.mark.parametrize("bad", [
    "__import__('os').system('true')", "u.__class__", "u[0]", "open('x')", "eval('1')", "9**9**9",
    "lambda: 1", "x if u else y", "[u for u in x]", "u_t = 'text'", "u = 1 = 2", "getattr(u, 'x')",
    "x" * 700,
])
def test_parser_refuses_anything_but_math(bad):
    with pytest.raises(EquationSyntaxError):
        parse_system([bad], coords=["x", "t"], fields=["u"])


def test_missing_values_and_unknowns_are_issues():
    s = parse_system(["u_t = alpha*u_xx"], HEAT_IC, coords=["x", "t"], fields=["u"])
    assert any("alpha" in i for i in s.issues())
    s = parse_system(["u_t = alpha*u_xx"], HEAT_IC, coords=["x", "t"], fields=["u"], unknowns=["alpha"])
    assert s.issues() == []
    s = parse_system(["u_t = u_xx"], [], coords=["x", "t"], fields=["u"])
    assert any("initial condition" in i for i in s.issues())


def test_fd_heat_1d_against_exact():
    s = parse_system(["u_t = alpha*lap(u)"], HEAT_IC, coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    r = solve_fd(s, {"x": (0, 1), "t": (0, 0.5)}, n=81)
    ex = np.exp(-0.1 * np.pi ** 2 * r.t)[:, None] * np.sin(np.pi * r.axes["x"])[None]
    assert np.linalg.norm(r.fields["u"] - ex) / np.linalg.norm(ex) < 1e-3


def test_fd_poisson_2d_and_neumann():
    s = parse_system(["lap(u) = -2*pi**2*sin(pi*x)*sin(pi*y)"],
                     [{"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}], coords=["x", "y"], fields=["u"])
    r = solve_fd(s, {"x": (0, 1), "y": (0, 1)}, n=33)
    X, Y = np.meshgrid(r.axes["x"], r.axes["y"], indexing="ij")
    ex = np.sin(np.pi * X) * np.sin(np.pi * Y)
    assert np.linalg.norm(r.fields["u"] - ex) / np.linalg.norm(ex) < 2e-3
    s = parse_system(["u_xx = -1"], [{"kind": "dirichlet", "on": "x_min", "equation": "u = 0"},
                                     {"kind": "neumann", "on": "x_max", "equation": "u_x = 0"}],
                     coords=["x"], fields=["u"])
    r = solve_fd(s, {"x": (0, 1)}, n=21)
    x = r.axes["x"]
    assert np.abs(r.fields["u"] - (x - x ** 2 / 2)).max() < 1e-8


def test_fd_wave_and_3d_heat():
    s = parse_system(["u_tt = c**2*u_xx"], HEAT_IC + [{"kind": "initial", "equation": "u_t = 0"}],
                     coords=["x", "t"], fields=["u"], coefficients={"c": 1.0})
    r = solve_fd(s, {"x": (0, 1), "t": (0, 1)}, n=81, n_save=11)
    ex = np.cos(np.pi * r.t)[:, None] * np.sin(np.pi * r.axes["x"])[None]
    assert np.linalg.norm(r.fields["u"] - ex) / np.linalg.norm(ex) < 1e-3
    s = parse_system(["u_t = alpha*lap(u)"], [{"kind": "initial", "equation": "u = sin(pi*x)*sin(pi*y)*sin(pi*z)"},
                                             {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                     coords=["x", "y", "z", "t"], fields=["u"], coefficients={"alpha": 0.05})
    r = solve_fd(s, {"x": (0, 1), "y": (0, 1), "z": (0, 1), "t": (0, 0.5)}, n=13, n_save=3)
    X, Y, Z = np.meshgrid(*[r.axes[c] for c in "xyz"], indexing="ij")
    ex = np.exp(-3 * 0.05 * np.pi ** 2 * 0.5) * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    assert np.linalg.norm(r.fields["u"][-1] - ex) / np.linalg.norm(ex) < 1e-2


def test_fd_refuses_what_it_cannot_do():
    s = parse_system(["u_t = alpha*u_xx"], HEAT_IC, coords=["x", "t"], fields=["u"])
    with pytest.raises(FDError):
        solve_fd(s, {"x": (0, 1), "t": (0, 1)}, n=21)          # alpha has no value
    s = parse_system(["u_t = u_xx"], HEAT_IC, coords=["x", "t"], fields=["u"])
    with pytest.raises(FDError):
        solve_fd(s, {"x": (0, 1)}, n=21)                       # no time bounds


def test_pinn_residual_vanishes_on_the_exact_solution():
    torch = pytest.importorskip("torch")
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    s = parse_system(["u_t = alpha*lap(u)"], HEAT_IC, coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    P, _ = pinn_problem(s, {"x": (0, 1), "t": (0, 1)})

    class Exact(torch.nn.Module):
        def forward(self, X):
            return (torch.exp(-0.1 * np.pi ** 2 * X[:, 1:2]) * torch.sin(np.pi * X[:, 0:1]))

    X = torch.rand(256, 2, dtype=torch.float64, requires_grad=True)
    u = Exact()(X)
    assert float(P.residual(u, X).detach().abs().max()) < 1e-9
    Xb, T = P.boundary(64, np.random.default_rng(0))
    assert np.allclose(T, 0.0) and set(np.unique(Xb[:, 0])) <= {0.0, 1.0}
    Xi, Ti = P.initial(64, np.random.default_rng(0))
    assert np.allclose(Xi[:, 1], 0.0) and np.allclose(Ti[:, 0], np.sin(np.pi * Xi[:, 0]))


def test_neumann_becomes_a_pinn_constraint_and_unknowns_are_trainable():
    pytest.importorskip("torch")
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    s = parse_system(["u_t = alpha*u_xx"], [{"kind": "initial", "equation": "u = sin(pi*x)"},
                                            {"kind": "dirichlet", "on": "x_min", "equation": "u = 0"},
                                            {"kind": "neumann", "on": "x_max", "equation": "u_x = 0"}],
                     coords=["x", "t"], fields=["u"], unknowns=["alpha"])
    P, model = pinn_problem(s, {"x": (0, 1), "t": (0, 1)}, unknowns={"alpha": 0.3})
    assert len(P.constraints) == 1
    assert "coefficients.alpha" in dict(model.named_parameters())
    assert model.estimates() == pytest.approx({"alpha": 0.3})
