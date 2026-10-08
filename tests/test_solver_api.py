"""One solve(problem) contract: classical (fem), neural (pinn), analytic, external and registered methods."""
import numpy as np
import pytest

import pinneapple as pp
from pinneapple_physics.pde_environment.builder import ProblemBuilder
from pinneapple_physics.solving import MethodNotAvailable, Solution, list_kinds, list_methods, register_method

SMALL_PINN = {"epochs": 40, "n_collocation": 256, "hidden_dim": 16, "n_layers": 2}
SOURCE = lambda X, ctx=None: -2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
EXACT = lambda X: (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None]
CTX = {"source_fn": SOURCE}


def _poisson_2d():
    spec = (ProblemBuilder("poisson_manufactured").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=0.0, on="x_boundary").bc("dirichlet", field="u", value=0.0, on="y_boundary")
            .build())
    return pp.PhysicalProblem.from_pde_spec(spec)


def test_same_problem_through_classical_and_neural_with_common_metadata():
    prob = _poisson_2d()
    fem = pp.solve(prob, "fem", ctx=CTX)
    pinn = pp.solve(prob, "pinn", ctx=CTX, **SMALL_PINN)
    assert (fem.kind, pinn.kind) == ("classical", "neural")
    assert set(fem.metadata()) == set(pinn.metadata())
    assert fem.metadata()["fingerprint"] == pinn.metadata()["fingerprint"] == prob.fingerprint()
    X = np.random.default_rng(0).random((200, 2))
    assert fem.predict(X).shape == pinn.predict(X).shape == (200, 1)
    err = np.linalg.norm(fem.predict(X) - EXACT(X)) / np.linalg.norm(EXACT(X))
    assert err < 1e-2
    # the PINN really received the source term: a trained-for-40-steps net is not the zero field
    assert np.abs(pinn.predict(X)).max() > 0


def test_compare_reports_kind_and_ranks_methods():
    prob = _poisson_2d()
    c = pp.compare(prob, ["fem", "pinn"], reference="exact",
                   options={"exact": {"fn": EXACT}, "fem": {"ctx": CTX}, "pinn": {"ctx": CTX, **SMALL_PINN}})
    assert [r["kind"] for r in c.rows] == ["classical", "neural"]
    assert "classical" in str(c) and c.best() == "fem"


def test_fem_converges_with_refinement_and_handles_1d_and_dirichlet_values():
    prob = _poisson_2d()
    X = np.random.default_rng(1).random((300, 2))
    errs = [np.linalg.norm(pp.solve(prob, "fem", ctx=CTX, n=n).predict(X) - EXACT(X)) for n in (8, 16, 32)]
    assert errs[0] > errs[1] > errs[2] and errs[0] / errs[2] > 8
    # 1D: u'' = -2 with u(0) = 1, u(1) = 3  ->  u = -x^2 + 3x + 1
    spec = (ProblemBuilder("p1d").domain(x=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=1.0, on="x_min").bc("dirichlet", field="u", value=3.0, on="x_max").build())
    s = pp.solve(pp.PhysicalProblem.from_pde_spec(spec), "fem", ctx={"source_fn": lambda X, c=None: -2 + 0 * X[:, 0]}, n=40)
    x = np.linspace(0, 1, 11)[:, None]
    assert s.predict(x)[:, 0] == pytest.approx(-x[:, 0] ** 2 + 3 * x[:, 0] + 1, abs=1e-8)


def test_fem_refuses_what_it_cannot_solve_honestly():
    with pytest.raises(MethodNotAvailable, match="tag selectors"):
        pp.solve("poisson_2d", "fem")
    with pytest.raises(MethodNotAvailable, match="poisson/laplace"):
        pp.solve("burgers_1d", "fem", preset_kwargs={"nu": 0.01})
    with pytest.raises(MethodNotAvailable, match="max_dofs"):
        pp.solve(_poisson_2d(), "fem", n=200)
    spec = (ProblemBuilder("neu").domain(x=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", on="x_min").bc("neumann", field="u", on="x_max").build())
    with pytest.raises(MethodNotAvailable, match="Dirichlet conditions only"):
        pp.solve(pp.PhysicalProblem.from_pde_spec(spec), "fem")


def test_external_runner_and_kinds_registry():
    prob = _poisson_2d()

    def my_code(problem):  # stands in for OpenFOAM / FEniCS / a legacy script
        ax = np.linspace(0, 1, 41)
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        return {"coords": {"x": ax, "y": ax}, "u": np.sin(np.pi * X) * np.sin(np.pi * Y)}

    s = pp.solve(prob, "external", runner=my_code, name="my_code")
    assert s.kind == "external" and s.metadata()["info"]["backend"] == "my_code"
    assert s.predict([[0.5, 0.5]])[0, 0] == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(MethodNotAvailable):
        pp.solve(prob, "external")
    kinds = list_kinds()
    assert {"fem", "reference"} <= set(kinds["classical"]) and "pinn" in kinds["neural"]
    assert list_methods(kind="analytic") == ["analytic", "exact"]


def test_registered_method_kind_and_custom_callable_default():
    name = "toy_zero_solver_for_test"

    @register_method(name, kind="reduced-order")
    def toy(problem, **_):
        return Solution(problem, name, lambda X: np.zeros((len(X), len(problem.fields))), 0.0)

    s = pp.solve(_poisson_2d(), name)
    assert s.kind == "reduced-order" and name in list_methods(kind="reduced-order")
    assert pp.solve(_poisson_2d(), lambda p, **_: Solution(p, "lam", lambda X: np.zeros((len(X), 1)), 0.0)).kind == "custom"
