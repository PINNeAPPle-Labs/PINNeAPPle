"""Tests for ``pinneapple_analysis.inverse_problems.gp_symbolic`` -- symbolic regression that is *checked* (#353).

The heart of the issue is the checking, not the search, so most of what is asserted here is a refusal:

* the **case**: a periodic 1-D field obeying ``u_t + a u_x = D u_xx - k u^p``, integrated with spectral derivatives,
  whose truth is known by construction. The solver is held to the library's exact advection-diffusion solution when
  the reaction is switched off (``rate = 0``), so the "known truth" is not an assumption -- it is measured against
  the analytic one to 1e-6 before anything is discovered.
* the **engine**: the k-th order sink really is recovered, at the finite-difference floor of the data, from a fixed
  seed.
* the **comparison**: the same columns and the same target through SINDy's sparse regression over a library -- and
  the number that matters, what each arm cost and whether it found the form.
* the **filter**: an expression with the wrong dimension, the wrong sign, a function of a dimensioned argument,
  terms that cannot be added, a quarter power, or an integral off the conservation statement is refused, and the
  refusal names the reason. The true expression passes.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.inverse_problems.gp_symbolic import (
    DimensionError,
    Expression,
    GeneticProgramming,
    PhysicsConstraints,
    PolicyViolation,
    advection_diffusion_reaction_case,
    compare_gp_and_sindy,
    discover_expression,
    noise_sweep,
    parse_expression,
)
from pinneapple_analysis.inverse_problems.missing_term import CandidateLibrary

U = {"u": "mol/m^3"}
RATE = "mol/(m^3*s)"


@pytest.fixture(scope="module")
def case():
    """The order-2 sink: generated once, used by every test that needs the field."""
    return advection_diffusion_reaction_case(order=2)


@pytest.fixture(scope="module")
def case3():
    return advection_diffusion_reaction_case(order=3)


# --------------------------------------------------------------------------- the case is a known truth
def test_the_solver_reproduces_the_librarys_exact_advection_diffusion_solution(case):
    """With the reaction switched off the integrator has a closed-form answer from the library, and it must match it.

    Without this the "truth" of the case would rest on the same code that generated it; with it, the field is
    pinned to an analytic solution to 1e-6 of its size."""
    check = {c["code"]: c for c in case["checks"]}
    assert check["adr.linear_solver"]["status"] == "pass"
    assert check["adr.linear_solver"]["value"] <= 1e-6
    assert check["adr.positive"]["status"] == "pass"


def test_the_finite_difference_floor_is_measured_not_assumed(case):
    """The column the discovery has to explain differs from the true term by the grid's own error; the case reports
    that number, and it is what the recovered expression's error is compared against."""
    assert 0 < case["residual_floor"] < 0.01
    check = {c["code"]: c for c in case["checks"]}
    assert check["adr.residual_floor"]["value"] == pytest.approx(case["residual_floor"])


def test_the_case_states_the_unit_of_the_rate_constant_of_its_order(case, case3):
    assert case["prefactor"] == "m^3/(mol^1*s)"                  # second order
    assert case3["prefactor"] == "m^6/(mol^2*s)"                 # third order
    # and the filter built from it wants an expression of u^2 and of u^3 respectively
    assert case["constraints"].dimension_ok(case["truth"])[0]
    assert case3["constraints"].dimension_ok(case3["truth"])[0]
    assert not case["constraints"].dimension_ok(case3["truth"])[0]     # a cubic sink is not a quadratic one


# --------------------------------------------------------------------------- the tree and its units
def test_an_expression_round_trips_through_its_text():
    tree = Expression("*", (Expression.constant(-0.7),
                            Expression("^", (Expression.variable("u"),), power=2)))
    assert tree.to_string() == "(-0.7 * (u)**2)"
    again = parse_expression(tree.to_string())
    columns = {"u": np.linspace(0.2, 2.0, 11)}
    assert np.allclose(again.evaluate(columns), tree.evaluate(columns))


def test_the_dimension_of_a_tree_comes_from_the_units_of_its_columns():
    from pinneapple_data.physical_units import parse_unit
    u2 = Expression("^", (Expression.variable("u"),), power=2)
    assert u2.dimension(U) == parse_unit("mol^2/m^6").dim
    # sqrt of a concentration is a legitimate closure shape, and its exponents are halves
    root = Expression("sqrt", (Expression.variable("u"),))
    halves = root.dimension(U)
    assert all(abs(float(p) * 2 - round(float(p) * 2)) < 1e-12 for p in halves)
    assert [(float(p) * 2) for p in halves] == [-3.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_a_quarter_power_of_a_quantity_is_refused():
    """``sqrt(sqrt(u))`` has exponents of a quarter; that is not a unit and the filter says so."""
    tree = Expression("sqrt", (Expression("sqrt", (Expression.variable("u"),)),))
    with pytest.raises(DimensionError, match="square root"):
        tree.dimension(U)


def test_a_unit_this_library_cannot_read_is_a_policy_violation_not_a_guess():
    """A checker that guessed the missing unit would pass exactly the expression it was written to catch."""
    with pytest.raises(PolicyViolation, match="not one this library parses"):
        PhysicsConstraints(units={"u": "furlong"})


# --------------------------------------------------------------------------- the filter refuses
def test_the_true_expression_passes_the_filter(case):
    verdict = case["truth"].check(case["constraints"], {"u": case["columns"]["u"]})
    assert verdict.ok, verdict.reasons
    assert {c["check"] for c in verdict.checks} == {"dimension", "sign", "limit"}


@pytest.mark.parametrize("candidate, reason", [
    ("(0.7 * (u)**2)", "sign"),                     # a source where the physics says sink
    ("(0.7 * (u)**3)", "dimension"),                # right law, wrong order: not the unit of a quadratic rate
    ("((u)**2 + u)", "dimension"),                  # two terms that cannot be added
    ("(u * sin(u))", "dimension"),                  # sine of a concentration
    ("log(u)", "dimension"),                        # logarithm of a concentration
    ("(0.7 * sqrt(u))", "dimension"),               # a half power does not have the unit of a quadratic rate
])
def test_expressions_that_break_the_physics_are_refused_with_the_reason(case, candidate, reason):
    verdict = parse_expression(candidate).check(case["constraints"], {"u": case["columns"]["u"]})
    assert verdict.refused
    assert any(c["detail"] for c in verdict.checks if not c["ok"])
    failed = {c["check"] for c in verdict.checks if not c["ok"]}
    assert reason in failed, f"{candidate} failed {failed}, expected {reason}"


def test_a_closure_that_does_not_vanish_with_the_reactant_is_refused(case):
    """A drag law has to vanish at rest: ``0.5 v^2 + 1.0`` fits the data and is refused at ``v = 0``.

    Written with a dimensioned column the same mistake never reaches the limit check -- ``-0.7u^2 - 0.05`` adds a
    dimensionless constant to a concentration and the dimension check stops it first, which is the cheaper of the
    two. The limit is what catches the version that stays inside one dimension."""
    v = case["columns"]["u"] / 2.0                       # a speed, m/s, on the same samples
    dimensionless = PhysicsConstraints(units={"u": "1"}, limits={"u": 0.0})
    offset = parse_expression("((0.5 * (u)**2) + 1.0)")
    assert dimensionless.check(offset, {"u": np.append(v, 0.0)}).refused
    ok = parse_expression("(0.5 * (u)**2)")
    assert dimensionless.check(ok, {"u": np.append(v, 0.0)}).ok
    # with the unit of a concentration declared, the constant is caught earlier, by the dimension check
    with_unit = PhysicsConstraints(units=U, target=RATE, prefactor="m^3/(mol*s)")
    verdict = parse_expression("((-0.7 * (u)**2) - 0.05)").check(with_unit, {"u": case["columns"]["u"]})
    assert verdict.refused and "dimension" in {c["check"] for c in verdict.checks if not c["ok"]}


def test_a_sink_that_does_not_add_up_to_the_conservation_statement_is_refused(case):
    """The balance the control volume gives: the term's integral over the field. A fit that is 3 % off in the
    coefficient fits the points and destroys mass, and is refused on the integral."""
    u = case["columns"]["u"]
    total = float(np.sum(u ** 2))
    balanced = PhysicsConstraints(units=U, target=RATE, prefactor="m^3/(mol*s)", sign="<=0",
                                  balance=(-0.7 * total, 1e-3 * 0.7 * total))
    right = parse_expression("(-0.7 * (u)**2)")
    assert balanced.check(right, {"u": u}).ok
    off = parse_expression("(-0.72 * (u)**2)")
    verdict = off.check(balanced, {"u": u})
    assert verdict.refused
    assert "integrates to" in "; ".join(verdict.reasons)
    # and the same expression without the balance statement passes: the balance is what caught it
    assert PhysicsConstraints(units=U, target=RATE, prefactor="m^3/(mol*s)").check(off, {"u": u}).ok


def test_a_symmetry_that_does_not_hold_is_refused():
    x = np.linspace(-2.0, 2.0, 41)
    even = Expression("^", (Expression.variable("u"),), power=2)
    odd_bit = Expression("+", (Expression("^", (Expression.variable("u"),), power=2), Expression.variable("u")))
    constraints = PhysicsConstraints(symmetry={"u": "even"})
    assert even.check(constraints, {"u": x}).ok
    verdict = odd_bit.check(constraints, {"u": x})
    assert verdict.refused and "not even" in "; ".join(verdict.reasons)


def test_a_filter_that_refuses_everything_is_reported_as_such(case):
    """An empty front is a result, not a crash: ``best`` is ``None``, the refusals are quoted, and ``equation()``
    says so rather than printing the expression the filter rejected."""
    u = case["columns"]["u"]
    impossible = PhysicsConstraints(units=U, target=RATE, prefactor="m^3/(mol*s)",
                                    balance=(1.0e6, 1e-6))          # no expression can integrate to this
    result = discover_expression({"u": u}, -case["truth"].evaluate({"u": u}), constraints=impossible,
                                 population=40, generations=4, seed=0)
    assert result.best is None
    assert result.refused and result.refused[0]["reasons"]
    assert "refused by the physics filter" in result.equation()
    assert next(c for c in result.checks if c["code"] == "gp.physics")["status"] == "fail"


# --------------------------------------------------------------------------- the engine recovers the truth
def test_the_engine_recovers_the_second_order_sink_from_the_field(case):
    u = case["columns"]["u"]
    result = discover_expression({"u": u}, case["columns"]["residual"], constraints=case["constraints"],
                                 population=120, generations=15, seed=0)
    assert result.best is not None, result.expressions[:1]
    recovered = parse_expression(result.best["expression"]).evaluate({"u": u}) * result.best["scale"]
    assert np.allclose(recovered, -0.7 * u ** 2, rtol=5e-2, atol=1e-9)
    # the reported error is at the data's own floor, not merely small
    assert result.best["relative_error"] < 2 * case["residual_floor"]
    assert result.best["nodes"] <= 6


def test_the_engine_recovers_a_term_no_degree_two_library_holds(case3):
    """A third-order sink against a quadratic squeeze's worth of library: the GP gets it from four nodes."""
    u = case3["columns"]["u"]
    result = discover_expression({"u": u}, case3["columns"]["residual"], constraints=case3["constraints"],
                                 population=120, generations=15, seed=0)
    assert result.best is not None
    recovered = parse_expression(result.best["expression"]).evaluate({"u": u}) * result.best["scale"]
    assert np.allclose(recovered, -0.7 * u ** 3, rtol=5e-2, atol=1e-9)


def test_the_pareto_front_is_non_dominated_and_ordered_by_error(case3):
    """The front keeps the expressions no other one beats on both objectives -- so along the error order, cheaper
    cannot also be more accurate. On a one-term target it collapses to a single form (measured: four nodes), which
    is the honest shape of the front here rather than a curve of a dozen points."""
    u = case3["columns"]["u"]
    result = discover_expression({"u": u}, case3["columns"]["residual"], constraints=case3["constraints"],
                                 population=80, generations=10, seed=5)
    front = result.expressions
    assert len(front) >= 1
    errors = [e["relative_error"] for e in front]
    assert errors == sorted(errors)
    nodes = [e["nodes"] for e in front]
    for i in range(len(front)):
        for j in range(len(front)):
            if i != j:                       # j must not be at least as cheap and at least as accurate as i
                assert not (nodes[j] <= nodes[i] and errors[j] <= errors[i]), (front[i], front[j])
    assert len(front) <= 25                  # max_front: a hundred spellings of one formula are one entry
    assert result.best is not None and result.best["relative_error"] == errors[0]


def test_the_same_seed_gives_the_same_expression(case):
    u = case["columns"]["u"]
    runs = [discover_expression({"u": u}, case["columns"]["residual"], constraints=case["constraints"],
                                population=60, generations=8, seed=7) for _ in range(2)]
    assert runs[0].expression == runs[1].expression


def test_a_different_seed_still_lands_on_a_feasible_expression(case):
    u = case["columns"]["u"]
    for seed in (3, 11):
        result = discover_expression({"u": u}, case["columns"]["residual"], constraints=case["constraints"],
                                     population=80, generations=10, seed=seed)
        assert result.best is not None
        assert result.best["relative_error"] < 0.05


# --------------------------------------------------------------------------- the comparison with SINDy
def test_the_comparison_runs_both_arms_on_the_same_target(case):
    u = case["columns"]["u"]
    out = compare_gp_and_sindy({"u": u}, case["columns"]["residual"], constraints=case["constraints"],
                               truth=case["truth"], population=80, generations=10, seed=0)
    assert set(out) >= {"gp", "sindy", "winner", "comparison", "checks", "cost_ratio"}
    # both arms reach the data's floor on a term the library holds -- and SINDy gets there far cheaper
    assert out["gp"]["relative_error"] < 0.01
    assert out["sindy"]["relative_error"] < 0.01
    assert out["sindy"]["seconds"] < out["gp"]["seconds"]
    assert out["gp"]["recovered"] and out["sindy"]["recovered"]


def test_a_term_outside_the_library_is_recovered_by_the_search_and_not_by_sparse_regression(case3):
    """This is the reason the engine exists: a degree-2 library cannot write ``u^3``, so SINDy fits something else
    (2 % of the term's RMS, three terms, the wrong form) while the search writes it exactly."""
    u = case3["columns"]["u"]
    out = compare_gp_and_sindy({"u": u}, case3["columns"]["residual"], constraints=case3["constraints"],
                               truth=case3["truth"], library=CandidateLibrary(poly_order=2),
                               population=120, generations=15, seed=0)
    assert out["gp"]["recovered"] is True
    assert out["sindy"]["recovered"] is False
    assert out["gp"]["relative_error"] < 0.01 < out["sindy"]["relative_error"]
    assert out["winner"] == "gp"


def test_given_the_right_library_sparse_regression_finds_it_and_costs_a_thousandth_of_the_search(case3):
    """The honest other half: when the term is in the library, SINDy is the right tool -- bounds in microseconds
    against seconds, and with the coefficient to 2 %."""
    u = case3["columns"]["u"]
    out = compare_gp_and_sindy({"u": u}, case3["columns"]["residual"], constraints=case3["constraints"],
                               truth=case3["truth"], library=CandidateLibrary(poly_order=3),
                               population=60, generations=8, seed=0)
    assert out["sindy"]["expression"].count("x0") >= 1
    assert out["sindy"]["relative_error"] < 0.01
    assert out["sindy"]["seconds"] < 0.5
    assert out["cost_ratio"] > 10


def test_a_law_no_polynomial_library_holds_is_reached_by_the_search_only(case):
    """``-0.8 sqrt(u)``: every polynomial library imitates it inside the sampled range (0.1 % at degree 6) and its
    recovered form is not the law. Recovery is judged past the data's range as well as on it, which is what tells
    the two apart."""
    u = case["columns"]["u"]
    truth = parse_expression("(-0.8 * sqrt(u))")
    target = truth.evaluate({"u": u})
    constraints = PhysicsConstraints(units=U, sign="<=0", limits={"u": 0.0})
    out = compare_gp_and_sindy({"u": u}, target, constraints=constraints, truth=truth,
                               library=CandidateLibrary(poly_order=3), population=120, generations=15, seed=0)
    assert out["gp"]["recovered"] is True
    assert float(out["gp"]["relative_error"]) < 0.01
    assert out["sindy"]["recovered"] is False
    # the polynomial does follow the curve over the data: that is why fitting error alone cannot decide
    assert out["sindy"]["relative_error"] < 0.01


def test_the_noise_sweep_reports_a_rate_for_every_arm(case):
    sweep = noise_sweep(case, levels=(0.0, 0.05), seeds=2, library_orders=(2, 3), population=60, generations=8)
    assert sweep["levels"] == [0.0, 0.05]
    for level in sweep["levels"]:
        assert 0.0 <= sweep["gp"][level]["recovered"] <= 1.0
        assert 0.0 <= sweep["sindy"][level][2]["recovered"] <= 1.0
    assert sweep["residual_floor"] == pytest.approx(case["residual_floor"])


# --------------------------------------------------------------------------- the engine's edges
def test_columns_can_be_an_array_with_names_or_a_mapping():
    u = np.linspace(0.2, 2.0, 60)
    by_array = discover_expression(u[:, None], -0.5 * u ** 2, names=["u"], population=40, generations=6, seed=0)
    by_mapping = discover_expression({"u": u}, -0.5 * u ** 2, population=40, generations=6, seed=0)
    for result in (by_array, by_mapping):
        assert result.best is not None
        assert "u" in result.best["expression"]


def test_a_target_with_a_nan_is_refused_before_the_search():
    u = np.linspace(0.2, 2.0, 30)
    y = -0.5 * u ** 2
    y[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        discover_expression({"u": u}, y, population=20, generations=2)


def test_an_unknown_operator_is_refused_at_construction():
    with pytest.raises(PolicyViolation, match="unknown operator"):
        GeneticProgramming(operators=("bessel",))


def test_a_target_unit_without_units_is_refused():
    with pytest.raises(PolicyViolation, match="needs the units"):
        PhysicsConstraints(target="mol/(m^3*s)")


def test_parse_expression_goes_through_the_safe_evaluator():
    """Text that would be code is refused by ``pinneapple_data.expressions`` before it is ever parsed here."""
    with pytest.raises(ValueError):
        parse_expression("__import__('os').system('echo pwned')")


def test_the_result_reports_its_cost_and_its_checks(case):
    u = case["columns"]["u"]
    result = discover_expression({"u": u}, case["columns"]["residual"], constraints=case["constraints"],
                                 population=40, generations=6, seed=0)
    assert result.seconds > 0
    assert result.population == 40 and result.generations == 6
    codes = {c["code"] for c in result.checks}
    assert {"gp.fit", "gp.physics", "gp.pareto"} <= codes
    assert result.equation().startswith("term = ")
    assert "Pareto" in str(result)
