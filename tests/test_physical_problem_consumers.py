"""One ``PhysicalProblem`` feeds every consumer (#27): backends, the benchmark, the dataset layer and the agent.

The issue's "done when" is a property, not an API: *the adapters pass the current tests with no 0.6 API change, and
``PhysicalProblem`` feeds backends, benchmark, dataset and agent*. So this file takes **one** problem, built once,
and hands it to each of the four, plus the three old classes that must keep working as adapters. If a consumer ever
grows its own problem type again, one of these fails.
"""
from __future__ import annotations

import numpy as np
import pytest

import pinneapple as pp

PRESET = "laplace_2d"


def _exact(X: np.ndarray) -> np.ndarray:
    """The manufactured solution of the plate below: laplace(T) = 0, T = sin(pi x) on the top face, 0 elsewhere."""
    return (np.sin(np.pi * X[:, 0]) * np.sinh(np.pi * X[:, 1]) / np.sinh(np.pi))[:, None]


@pytest.fixture(scope="module")
def problem() -> pp.PhysicalProblem:
    """Built once, through the PDE spec, with the faces spelled out so a classical backend can run it."""
    from pinneapple_physics.pde_environment.builder import ProblemBuilder

    spec = (ProblemBuilder("plate").domain(x=(0, 1), y=(0, 1)).fields("T").pde("laplace")
            .bc("dirichlet", field="T", value=0.0, on="x_min")
            .bc("dirichlet", field="T", value=0.0, on="x_max")
            .bc("dirichlet", field="T", value=0.0, on="y_min")
            .bc("dirichlet", field="T", value=lambda X: np.sin(np.pi * X[:, 0]), on="y_max").build())
    return pp.PhysicalProblem.from_pde_spec(spec)


# ------------------------------------------------------------------ the four consumers
def test_a_backend_solves_the_problem_object_with_no_adapter(problem):
    solution = pp.solve(problem, "fem", n=24)
    assert isinstance(solution, pp.Solution) and solution.method == "fem"
    assert solution.problem.fingerprint() == problem.fingerprint(), "the answer names the problem it answered"
    # the top face was held at sin(pi x), so the field has to reach about 1 in the middle of y
    values = solution.predict(np.array([[0.5, 1.0], [0.5, 0.5]]))
    assert values[0, 0] == pytest.approx(1.0, abs=5e-2)
    assert 0.0 < values[1, 0] < 0.5, "half way up the plate the solution is between the cold wall and the hot face"


def test_the_benchmark_grades_the_same_object_against_a_reference(problem):
    solution = pp.solve(problem, "fem", n=24)
    table = pp.compare(problem, [solution], reference="exact", options={"exact": {"fn": _exact}})
    assert isinstance(table, pp.Comparison)
    row = table.rows[0]
    assert row["status"] == "ok"
    assert row["metrics"]["relative_l2"]["T"] < 1e-2, "P1 elements on this plate land near 2e-3; a regression here is a real one"


def test_the_dataset_layer_samples_the_problem_object(problem):
    from pinneapple_data.collocation import CollocationSampler

    sampler = CollocationSampler.from_problem_spec(problem)
    assert tuple(sampler.coord_names) == tuple(problem.coords)
    assert tuple(sampler.fields) == tuple(problem.fields)
    points = sampler.sample(200)
    assert "x_col" in points and len(points["x_col"]) == 200


def test_the_agent_builds_the_same_type_from_its_json(problem):
    """The MCP tools take JSON and hand the solvers a ``PhysicalProblem`` — the same class, not a lookalike."""
    from pinneapple_mcp.problems import build_problem

    built = build_problem({"preset": PRESET})
    assert isinstance(built.problem, pp.PhysicalProblem)
    assert built.problem.fingerprint() == pp.PhysicalProblem.from_preset(PRESET).fingerprint()


# ------------------------------------------------------------------ the three adapters, unchanged
def test_the_pde_spec_goes_through_the_problem_and_comes_back_equal(problem):
    assert pp.PhysicalProblem.from_pde_spec(problem.to_pde_spec()).to_pde_spec() == problem.to_pde_spec()


def test_the_elicited_description_goes_through_the_problem(problem):
    design = problem.to_problem_design()
    back = pp.PhysicalProblem.from_problem_design(design)
    assert back.name == design.title == problem.name
    assert set(design.outputs) == set(problem.fields)


def test_the_physics_case_goes_through_the_problem(problem):
    from pinneapple_data.physics_case import PhysicsCase

    case = PhysicsCase(name=problem.name, physics=problem.to_pde_spec())
    from_case = pp.PhysicalProblem.from_physics_case(case)
    assert from_case.name == problem.name
    assert from_case.fingerprint() == problem.fingerprint()


def test_the_old_classes_are_still_importable_where_they_were():
    """No 0.6 API change: the three classes keep their module paths, their names and their constructors."""
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_physics.pde_environment.spec import ProblemSpec as PdeProblemSpec
    from pinneapple_problemdesign.schema import ProblemSpec as ElicitedProblemSpec

    assert PdeProblemSpec(name="p", dim=1, coords=("x",), fields=("u",), pde=None, conditions=()).name == "p"
    assert ElicitedProblemSpec(title="t", goal="g").title == "t"
    assert PhysicsCase(name="c").name == "c"
