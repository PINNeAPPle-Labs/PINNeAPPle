"""The façade over the problem (#151): what the answer is for, what limits it, how much physics it needs.

The acceptance criterion of the issue is one sentence — *a problem created in one place compiles into the PDE spec
and into the elicited spec without re-entering data* — so the last test here does exactly that: it sets the façade
once, exports both specs, reads them back, and asserts the four fields arrived on the other side. The rest of the
file guards the properties that make that safe: the façade validates itself where it is set, it survives JSON, and a
preset that predates it does not grow the keys (a fingerprint is how a stored result names its problem).
"""
from __future__ import annotations

import dataclasses

import pytest

import pinneapple as pp

PRESET = "burgers_1d"
FACADE_KEYS = ("objectives", "constraints", "fidelity", "geometry_ref")


def _facade_problem():
    """A preset problem with the whole façade set — the "created in one place" of the acceptance criterion."""
    return dataclasses.replace(
        pp.PhysicalProblem.from_preset(PRESET),
        objectives=(pp.Objective("hold_low", "u", sense="target", target=0.0, tolerance=0.1),),
        constraints=(pp.Constraint("wall_limit", quantity="u", op="<=", value=1.0),
                     pp.Constraint("budget", kind="expression", expression="nu <= 0.02")),
        fidelity=pp.Fidelity(level="rans", mesh_level="L1", tolerance=0.02),
        geometry_ref="cad/duct.step",
    )


# ------------------------------------------------------------------ the three types on their own
def test_an_objective_says_what_the_answer_is_for():
    assert pp.Objective("drag", "Cd").sense == "minimize"          # the default, because it is the common case
    assert pp.Objective("lift", "Cl", sense="maximize").sense == "maximize"
    with pytest.raises(ValueError, match="sense"):
        pp.Objective("wrong", "Cd", sense="minimise")              # spelling is not a synonym
    with pytest.raises(ValueError, match="target value"):
        pp.Objective("no_target", "T", sense="target")             # a target objective without a target
    with pytest.raises(ValueError, match="weight"):
        pp.Objective("weightless", "T", weight=0.0)                # it would vanish from the sum


def test_a_constraint_prints_itself_the_same_way_in_every_spec():
    bound = pp.Constraint("wall", quantity="T", op="<=", value=350.0, unit="K")
    assert bound.text() == "T <= 350.0 K"
    assert pp.Constraint("free", kind="expression", expression="mass <= 12 kg").text() == "mass <= 12 kg"
    with pytest.raises(ValueError, match="op"):
        pp.Constraint("bad_op", quantity="T", op="<", value=1.0)
    with pytest.raises(ValueError, match="value"):
        pp.Constraint("no_value", quantity="T")                    # a bound without a bound
    with pytest.raises(ValueError, match="expression"):
        pp.Constraint("empty", kind="expression")


def test_fidelity_says_how_much_physics_and_how_good():
    assert pp.Fidelity().describe() == "unspecified"
    assert pp.Fidelity(level="les", mesh_level="coarse", tolerance=0.05).describe() == \
        "les, mesh coarse, within 0.05 relative error"
    assert pp.Fidelity(level="closed_form", notes="analytic").describe() == "closed_form, analytic"


# ------------------------------------------------------------------ carried through the problem
def test_the_facade_survives_json():
    problem = _facade_problem()
    d = problem.to_dict()
    for key in FACADE_KEYS:
        assert key in d, f"{key} has to be in the dict or a saved study cannot reproduce the question"
    back = pp.PhysicalProblem.from_dict(d)
    assert back.objectives == problem.objectives
    assert [c.text() for c in back.constraints] == ["u <= 1.0", "nu <= 0.02"]
    assert back.fidelity == problem.fidelity
    assert back.geometry_ref == "cad/duct.step"
    assert back.fingerprint() == problem.fingerprint(), "same problem, same fingerprint"


def test_the_facade_moves_the_fingerprint_when_it_changes():
    """The fingerprint is how a result names its problem: two problems with different constraints are not the same."""
    base = _facade_problem()
    other = dataclasses.replace(base, constraints=(pp.Constraint("wall", quantity="u", op="<=", value=2.0),))
    assert base.fingerprint() != other.fingerprint()


def test_a_preset_carries_no_facade_keys_so_its_fingerprint_does_not_move():
    """Every preset predates these fields; emitting empty ones would re-fingerprint the whole library."""
    preset = pp.PhysicalProblem.from_preset(PRESET)
    d = preset.to_dict()
    assert not [k for k in FACADE_KEYS if k in d], [k for k in FACADE_KEYS if k in d]
    assert not preset.to_pde_spec().meta.get("facade")


# ------------------------------------------------------------------ the acceptance criterion
def test_one_problem_compiles_into_both_specs_without_re_entering_data():
    """#151: the façade is set once, and both compilations carry it — the PDE spec and the elicited design."""
    problem = _facade_problem()

    # (1) into the runnable PDE spec, and back
    spec = problem.to_pde_spec()
    assert spec.meta["facade"]["objectives"][0]["name"] == "hold_low"
    from_spec = pp.PhysicalProblem.from_pde_spec(spec)
    assert from_spec.objectives == problem.objectives
    assert [c.text() for c in from_spec.constraints] == [c.text() for c in problem.constraints]
    assert from_spec.fidelity == problem.fidelity and from_spec.geometry_ref == problem.geometry_ref
    assert from_spec.to_pde_spec() == spec, "the detour through the facade is lossless for the spec"

    # (2) into the elicited description, as the prose a report or the assistant shows
    design = problem.to_problem_design()
    assert design.physics.constraints == ["u <= 1.0", "nu <= 0.02"]
    assert design.geometry.domain == "cad/duct.step", "where the geometry came from, in prose"
    assert design.geometry.representation == "rans, mesh L1, within 0.02 relative error"


def test_the_elicited_side_keeps_its_own_constraints_when_the_facade_is_empty():
    """A problem elicited with prose constraints must not lose them to an empty façade."""
    preset = pp.PhysicalProblem.from_preset(PRESET)
    preset.intent["physics_text"] = {"constraints": ["keep the flow laminar"]}
    assert preset.to_problem_design().physics.constraints == ["keep the flow laminar"]


def test_a_problem_without_a_pde_refuses_to_pretend_it_has_one():
    """The façade makes a problem describable, not solvable: no PDE, no PDE spec."""
    description_only = pp.PhysicalProblem(name="duct", objectives=(pp.Objective("drag", "Cd"),),
                                          fidelity=pp.Fidelity(level="rans"))
    with pytest.raises(ValueError, match="no PDE"):
        description_only.to_pde_spec()
    assert description_only.to_problem_design().title == "duct"      # the elicited side still works
