"""A PDE whose terms do not share a dimension is caught before training (#205).

The failure this prevents does not look like a unit bug when it arrives: a source term quoted in W/m³ next to a time
derivative in K/s trains fine, and then the loss floors at a value nobody explains. These tests hold the checker to the
two halves of the acceptance — it flags the seeded inconsistency, it passes the consistent equation — and to the rule
that makes it usable: a name it cannot give a dimension to is refused, never assumed.
"""
from __future__ import annotations

import pytest

import pinneapple as pp
from pinneapple_physics.dimensions import check_dimensions, format_dim, unit_dim

HEAT = {"T": "K", "t": "s", "x": "m", "alpha": "m^2/s", "q": "W/m^3", "rho": "kg/m^3", "cp": "J/(kg*K)"}


def test_the_consistent_heat_equation_passes():
    report = check_dimensions("dT/dt = alpha*laplacian(T) + q/(rho*cp)", units=HEAT)
    assert report.ok, report.problems
    assert [t.unit for t in report.terms] == ["K/s", "K/s", "K/s"]
    assert str(report).endswith("consistent")


def test_a_source_in_wrong_units_is_flagged_with_both_units():
    """The seeded inconsistency: ``q`` in kelvin instead of W/m³."""
    units = {**HEAT, "q": "K"}
    report = check_dimensions("dT/dt = alpha*laplacian(T) + q/(rho*cp)", units=units)
    assert not report.ok
    assert len(report.problems) == 1
    problem = report.problems[0]
    assert "q / (rho * cp)" in problem and "the other terms of the equation are K/s" in problem
    assert str(report).endswith("inconsistent")
    assert "| `dT/dt` | K/s |" in report.to_markdown()


@pytest.mark.parametrize("equation", [
    "dT/dt = alpha*laplacian(T)",
    "∂T/∂t = alpha*∇²T",
    "dT/dt = alpha*d2T/dx2",
    "dT/dt = alpha*lap(T)",
    "dT/dt = alpha*d2_dx2(T)",
])
def test_the_notations_a_paper_uses_all_read_the_same(equation):
    assert check_dimensions(equation, units=HEAT).ok


def test_a_homogeneous_equation_is_consistent():
    """``… = 0`` says the expression vanishes; the zero is not a term that has to carry K/m²."""
    report = check_dimensions("∂²T/∂x² = 0", units=HEAT)
    assert report.ok, report.problems
    assert [t.unit for t in report.terms] == ["K/m^2"]


def test_an_unknown_name_is_refused_by_name():
    report = check_dimensions("dT/dt = alpha*laplacian(T) + source", units=HEAT)
    assert not report.ok
    assert "unknown name 'source'" in report.problems[0]
    assert "units={'source'" in report.problems[0]


def test_an_unknown_function_is_refused():
    report = check_dimensions("dT/dt = alpha*swirl(T)", units=HEAT)
    assert not report.ok
    assert "unknown function 'swirl()'" in report.problems[0]


def test_an_equation_without_an_equals_sign_is_refused():
    report = check_dimensions("dT/dt + alpha*laplacian(T)", units=HEAT)
    assert not report.ok
    assert "exactly one '='" in report.problems[0]


def test_a_spatial_derivative_without_a_length_is_refused_not_assumed():
    report = check_dimensions("dT/dt = alpha*laplacian(T)", units={"T": "K", "t": "s", "alpha": "m^2/s"})
    assert not report.ok
    assert "needs a length" in report.problems[0]
    assert check_dimensions("dT/dt = alpha*laplacian(T)",
                            units={"T": "K", "t": "s", "alpha": "m^2/s"}, space="m").ok


def test_a_time_derivative_without_a_time_is_refused():
    report = check_dimensions("dT/dt = alpha*laplacian(T)", units={"T": "K", "x": "m", "alpha": "m^2/s"})
    assert not report.ok
    assert "needs a time" in report.problems[0]


def test_adding_two_different_units_inside_a_product_is_reported():
    """``2*(T + q)`` cannot be added, so the term has no unit at all — saying which pair broke it is the report."""
    report = check_dimensions("dT/dt = alpha*laplacian(T) + 2*(T + q)", units={**HEAT, "q": "W/m^3"})
    assert not report.ok
    assert "'T' is K but 'q' is W/m^3" in report.problems[0]


def test_a_parenthesised_sum_is_split_into_terms_and_each_one_is_checked():
    report = check_dimensions("dT/dt = alpha*laplacian(T) + (T + q)", units={**HEAT, "q": "K/s"})
    assert [t.text for t in report.terms] == ["dT/dt", "alpha * laplacian(T)", "T", "q"]
    assert report.problems == ["'T' is K, the other terms of the equation are K/s"]


def test_the_report_refuses_to_go_on_with_the_reason_in_the_message():
    report = check_dimensions("dT/dt = alpha*laplacian(T) + q", units={**HEAT, "q": "W/m^3"})
    with pytest.raises(ValueError, match="dimensionally inconsistent"):
        report.raise_if_inconsistent()
    assert report.raise_if_inconsistent.__doc__


def test_the_report_serialises_and_reads_as_a_verdict():
    report = check_dimensions("dT/dt = alpha*laplacian(T)", units=HEAT)
    payload = report.to_dict()
    assert payload["ok"] is True and payload["terms"][0]["unit"] == "K/s"
    assert bool(report) is True


# ── wired into the problem ──────────────────────────────────────────────────
def _heat_problem(**overrides):
    spec = pp.PhysicalProblem.from_preset("transient_heat_3d").to_pde_spec()
    units = {"x": "m", "y": "m", "z": "m", "t": "s", "T": "K", "alpha": "m^2/s"}
    units.update(overrides)
    return pp.PhysicalProblem.from_pde_spec(spec, units=units)


def test_the_problem_carries_its_own_units_into_the_check():
    problem = _heat_problem()
    report = problem.dimensions("dT/dt = alpha*laplacian(T)")
    assert report.ok, report.problems
    assert [t.unit for t in report.terms] == ["K/s", "K/s"]


def test_a_problem_with_a_wrong_parameter_unit_fails_validation():
    problem = _heat_problem(alpha="m^2")
    problems = problem.validate(equations="dT/dt = alpha*laplacian(T)")
    assert any("dimensionally inconsistent" in p for p in problems)
    assert problem.validate(equations="dT/dt = alpha*laplacian(T)", require_units=True)


def test_validation_accepts_one_equation_or_a_list():
    problem = _heat_problem()
    assert not [p for p in problem.validate(equations=["dT/dt = alpha*laplacian(T)"])
                if "dimension" in p]
    assert not [p for p in problem.validate(equations={"dT/dt = alpha*laplacian(T)"}) if "dimension" in p]


def test_the_units_win_over_the_problem_when_both_are_given():
    problem = _heat_problem()
    wrong = problem.dimensions("dT/dt = alpha*laplacian(T)", units={"alpha": "m^2"})
    assert not wrong.ok


# ── the unit text ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(("unit", "shown"), [
    ("K/s", "K/s"),
    ("m^2/s", "m^2/s"),
    ("W/m^3", "W/m^3"),
    ("kg*m/s^2", "N"),
    ("1/s", "Hz"),
    ("-", "dimensionless"),
    ("J/(kg*K)", "m^2/(s^2·K)"),
])
def test_the_unit_a_term_works_out_to_is_written_the_way_a_reader_expects(unit, shown):
    assert format_dim(unit_dim(unit)) == shown


def test_a_unit_it_cannot_read_says_so():
    with pytest.raises(ValueError, match="not a unit this library parses"):
        unit_dim("furlongs per fortnight")
