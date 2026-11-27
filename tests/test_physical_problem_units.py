"""Units as a requirement and as a conversion (#110).

Two halves: a problem can be asked whether it is *complete* with respect to units, and it can be moved to another
unit system with every number attached to a name converted — bounds, field ranges and parameter values.
"""
from __future__ import annotations

import pytest

import pinneapple as pp


def bar():
    """A rod in metres with a displacement field in metres, a parameter with a unit and a range."""
    return pp.PhysicalProblem(
        name="barra", coords=("x",), fields=("u",), domain_bounds={"x": (0.0, 1.0)},
        units={"x": "m", "u": "m"}, field_ranges={"u": (0.0, 0.002)},
        parameters={"E": pp.Parameter(name="E", value=210e9, unit="Pa", role="fixed"),
                    "L": pp.Parameter(name="L", value=1.0, unit="m", role="fixed")})


def test_a_problem_without_units_is_refused_when_they_are_required():
    incomplete = pp.PhysicalProblem.from_preset("poisson_2d")
    assert incomplete.validate() == [], "no unit is not an inconsistency on its own"
    errors = incomplete.validate(require_units=True)
    assert errors and "units are required and missing" in errors[-1]
    for name in incomplete.fields:
        assert f"'{name}'" in errors[-1], errors[-1]


def test_a_problem_with_units_passes_when_they_are_required():
    assert bar().validate(require_units=True) == []
    assert bar().missing_units() == []


def test_missing_units_names_coordinates_fields_and_parameters():
    p = bar()
    p.units.pop("x")
    p.parameters["E"] = pp.Parameter(name="E", value=210e9, unit=None, role="fixed")
    assert p.missing_units() == ["x", "E"]


def test_a_conversion_moves_every_number_attached_to_the_name():
    p = bar()
    mm = p.to_unit_system({"x": "mm", "u": "mm"})
    assert mm.units["x"] == "mm" and mm.units["u"] == "mm"
    assert mm.domain_bounds["x"] == (0.0, 1000.0)
    assert mm.field_ranges["u"] == (0.0, 2.0)
    assert mm.to_unit_system({"x": "m", "u": "m"}).domain_bounds["x"] == (0.0, 1.0)   # round trip


def test_parameters_convert_value_bounds_and_unit_together():
    p = bar().set_role("E", "unknown", bounds=(1e11, 3e11))
    gpa = p.to_unit_system({"E": "GPa"})
    assert gpa.parameters["E"].unit == "GPa"
    assert gpa.parameters["E"].value == pytest.approx(210.0)
    assert gpa.parameters["E"].bounds == (pytest.approx(100.0), pytest.approx(300.0))
    assert gpa.pde is None or gpa.pde.params.get("E", 210.0) == pytest.approx(210.0)


def test_the_original_is_not_modified():
    p = bar()
    before = p.to_dict()
    p.to_unit_system({"x": "km"})
    assert p.to_dict() == before, "the conversion returns a copy; a problem is not edited in place"


def test_an_unknown_name_is_refused_so_a_typo_does_not_pass():
    with pytest.raises(KeyError, match="unknown name"):
        bar().to_unit_system({"xx": "mm"})


def test_a_name_without_a_declared_unit_cannot_be_converted():
    p = bar()
    p.units.pop("u")
    with pytest.raises(ValueError, match="declares no unit"):
        p.to_unit_system({"u": "mm"})


def test_converting_across_dimensions_is_refused_with_both_dimensions_named():
    with pytest.raises(ValueError, match=r"cannot convert 'x' from 'm' to 's'"):
        bar().to_unit_system({"x": "s"})


def test_an_unparseable_target_is_refused():
    with pytest.raises(ValueError, match="not one this library parses"):
        bar().to_unit_system({"x": "furlong per fortnight"})


def test_the_fingerprint_changes_because_the_description_changed():
    p = bar()
    assert p.to_unit_system({"x": "mm"}).fingerprint() != p.fingerprint()
