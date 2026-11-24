"""The design study schema ``pinneapple-design-study/v1``: what it round-trips and what it refuses.

The refusals are the point: a study with an output that has no unit, a variable the tool's input mapping does not
cover, a limit relative to a baseline nobody measured, or an STL with nothing to vary is not a study with a default —
it is a study that cannot be checked, so it never reaches the runner.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_analysis.design_study import (
    SCHEMA,
    SURROGATE_FAMILIES,
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Limit,
    Model,
    Objective,
    Quantity,
    Variable,
    Verifier,
    run_study,
)
from pinneapple_analysis.design_study.examples import (
    EXAMPLES,
    example,
    heatsink_study,
    polar_study,
    wing_study,
)


def toy(**overrides):
    """A minimal valid study: one variable, two outputs, one objective, nothing to fit yet."""
    d = dict(kind="optimize", title="toy",
             base=Base(kind="geometry", ref="pp.geometries:box", variables=[Variable("h", 0.0, 1.0, unit="m")]),
             evaluator=Evaluator(tool="toy.box", inputs={"h": "height"}, outputs=[Quantity("f", "1"),
                                                                                Quantity("g", "s")]),
             objectives=[Objective("f", "min")], budget=Budget(max_evaluations=8, n_doe=4, n_verify=2))
    d.update(overrides)
    return DesignStudy.create(**d)


# ---------------------------------------------------------------------------------------------- round trip
@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_example_round_trips_through_dict_and_json(name):
    spec, _ = example(name)
    back = DesignStudy.from_dict(spec.to_dict())
    assert back.to_dict() == spec.to_dict()
    assert DesignStudy.from_json(spec.to_json()).to_dict() == spec.to_dict()
    assert back.digest() == spec.digest() and len(spec.digest()) == 64
    assert json.loads(spec.to_json())["schema"] == SCHEMA


def test_the_spec_describes_itself_for_a_wizard():
    d = wing_study().describe()
    assert d["schema"] == SCHEMA and d["kind"] == "optimize"
    assert d["base"]["baseline"] == {"aspect_ratio": 7.0, "taper_ratio": 0.5}
    assert [o["unit"] for o in d["objectives"]] == ["1", "kg"]         # objectives carry their output's unit
    assert d["constraints"][1]["text"].startswith("mass <= 1.05 x baseline") and d["constraints"][1]["relative"]
    assert d["budget"]["plan"] == {"baseline": 1, "doe": 10, "verify": 3, "search": 6}


def test_the_baseline_defaults_to_the_middle_of_every_range():
    assert toy().baseline_design() == {"h": 0.5}
    assert toy(base=Base(kind="geometry", ref="x", variables=[Variable("h", 0.0, 1.0, unit="m")],
                         baseline={"h": 0.2})).baseline_design() == {"h": 0.2}


def test_the_surrogate_families_are_the_library_ones():
    from pinneapple_autolab.surrogates import FAMILIES
    assert tuple(FAMILIES) == SURROGATE_FAMILIES


# ---------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("unit", ["", "  ", "?", "unknown", "n/a", "none"])
def test_an_output_without_a_unit_is_refused(unit):
    with pytest.raises(ValueError, match="no unit given"):
        Quantity("drag", unit)
    with pytest.raises(ValueError, match="no unit given"):
        Evaluator(tool="t", inputs={"h": "h"}, outputs=[Quantity("f", "1"), Quantity("g", unit)])


def test_a_variable_without_a_unit_and_a_dimensionless_output_are_both_explicit():
    with pytest.raises(ValueError, match="no unit given"):
        Variable("h", 0.0, 1.0, unit="")
    assert Quantity("cl", "1").unit == "1" and Quantity("cl", "-").unit == "-"


def test_an_unknown_field_is_refused_rather_than_dropped():
    spec = toy().to_dict()
    with pytest.raises(ValueError, match="unknown field"):
        DesignStudy.from_dict(spec | {"optimizer": "nsga2"})
    with pytest.raises(ValueError, match="unknown field"):
        DesignStudy.from_dict(spec | {"budget": spec["budget"] | {"max_runs": 3}})


def test_another_schema_version_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        DesignStudy.from_dict(toy().to_dict() | {"schema": "pinneapple-design-study/v2"})


def test_a_variable_without_a_tool_input_or_a_tool_input_without_a_variable_is_refused():
    with pytest.raises(ValueError, match="do not match"):
        toy(evaluator=Evaluator(tool="t", inputs={"other": "h"}, outputs=[Quantity("f", "1")]))
    with pytest.raises(ValueError, match="do not match"):
        toy(evaluator=Evaluator(tool="t", inputs={"h": "h", "extra": "e"}, outputs=[Quantity("f", "1")]))


def test_an_objective_on_an_undeclared_output_is_refused():
    with pytest.raises(ValueError, match="does not declare"):
        toy(objectives=[Objective("cd", "min")])


def test_a_limit_that_bounds_nothing_is_refused():
    with pytest.raises(ValueError, match="bounds nothing"):
        toy(constraints=[Limit("g")])


def test_a_relative_limit_is_only_a_number_after_the_baseline():
    spec = toy(constraints=[Limit("g", upper_frac_of_baseline=1.1)])
    assert spec.constraints[0].relative
    assert spec.constraints[0].resolved({"g": 10.0}).upper == pytest.approx(11.0)
    with pytest.raises(Exception, match="baseline"):
        spec.constraints[0].resolved(None)


def test_an_stl_base_needs_a_parametrization():
    with pytest.raises(ValueError, match="parametrization"):
        Base(kind="stl", ref="wing.stl", variables=[Variable("thickness", 0.0, 1.0, unit="m")])
    assert Base(kind="stl", ref="wing.stl", variables=[Variable("thickness", 0.0, 1.0, unit="m")],
                parametrization="ffd").parametrization == "ffd"


def test_a_budget_that_cannot_hold_the_cycle_is_refused():
    with pytest.raises(ValueError, match="at least 4"):
        Budget(max_evaluations=3)
    with pytest.raises(ValueError, match="cannot hold"):
        Budget(max_evaluations=8, n_doe=7, n_verify=3).planned(1)


def test_an_unknown_surrogate_family_is_refused():
    with pytest.raises(ValueError, match="unknown surrogate families"):
        Model(families=("gp", "transformer"))


def test_a_pinn_study_needs_the_equation_it_is_informed_by():
    with pytest.raises(ValueError, match="model.pde"):
        toy(kind="pinn")
    ok = toy(kind="pinn", model=Model(pde={"type": "poisson_1d", "coords": {"x": "x"},
                                           "coefficients": {"k": 1.0, "f": 1.0}}))
    assert ok.model.pde["type"] == "poisson_1d"


def test_a_baseline_outside_its_range_is_refused():
    with pytest.raises(ValueError, match="outside its range"):
        Base(kind="geometry", ref="x", variables=[Variable("h", 0.0, 1.0, unit="m")], baseline={"h": 2.0})


def test_a_parameter_cannot_be_varied_and_fixed_at_once():
    with pytest.raises(ValueError, match="both varied and fixed"):
        Base(kind="geometry", ref="x", variables=[Variable("h", 0.0, 1.0, unit="m")], fixed={"h": 0.5})


# ---------------------------------------------------------------------------------------------- what runs today
@pytest.mark.parametrize("kind", ["pinn", "inverse", "twin"])
def test_kinds_beyond_the_two_that_run_are_described_and_refused_with_the_reason(kind):
    spec = toy(kind=kind, model=Model(pde={"type": "laplace_2d", "coords": {"x": "x", "y": "y"}}))
    run = run_study(spec, lambda p: {"f": 1.0, "g": 2.0})
    assert run.status == "refused" and not run.rows
    assert f"a {kind!r} study is described by the schema" in run.reasons[0]
    assert "#1036" in run.reasons[0]
    assert run.to_dict()["spec"]["kind"] == kind


def test_the_examples_name_their_model_and_it_is_not_a_solver():
    for name, (make, evaluator) in EXAMPLES.items():
        spec = make()
        assert spec.evaluator.fidelity == "model", name
        assert "not a" in spec.metadata["model"], name
        assert callable(evaluator)


def test_the_three_examples_are_an_optimisation_pair_and_a_surrogate():
    assert wing_study().kind == "optimize" and heatsink_study().kind == "optimize"
    assert polar_study().kind == "surrogate"
    assert Verifier().min_verified == 1 and polar_study().verifier.min_verified == 0


def test_validating_a_spec_loads_nothing_of_the_study_itself():
    """The wizard refuses a broken study without paying for the surrogate, the optimiser or the lab.

    ``pinneapple_analysis`` itself already imports torch (its uncertainty and inverse-problem modules do), so what is
    checked here is what *this* package brings: importing it and validating a spec pulls neither scikit-learn (the
    surrogate board) nor ``pinneapple_autolab`` (the validated domain), which are imported where they are used.
    """
    import subprocess
    import sys

    code = ("import sys, pinneapple_analysis.design_study as ds\n"
            "spec = ds.DesignStudy.from_dict(ds.examples.wing_study().to_dict())\n"
            "assert spec.digest() and spec.kind == 'optimize'\n"
            "print(sorted(m for m in ('sklearn', 'pinneapple_autolab', 'pinneapple_design') if m in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout
