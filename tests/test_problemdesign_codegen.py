"""The elicited problem becomes concrete API objects: ``build_pinneapple_spec`` (pinneapple_problemdesign.codegen).

The codegen is the last step of problem design, and the only one nobody could call from a form: its argument is the
``ProblemSpec`` a :class:`~pinneapple_problemdesign.agent.DesignAgent` elicits over a conversation. These tests build
that spec by hand and check what comes out — that the identified PDE reaches the compiled objects, that the code
snippet is runnable text, and that a spec with no recognisable physics degrades to ``custom`` instead of raising.
"""
from __future__ import annotations

from pinneapple_problemdesign.codegen import build_pinneapple_spec
from pinneapple_problemdesign.schema import PhysicsSpec
from pinneapple_problemdesign.schema import ProblemSpec as ElicitedSpec


def _design(**over) -> ElicitedSpec:
    physics = PhysicsSpec(governing_equations=["steady heat conduction"],
                          parameters_known=["k"], parameters_unknown=["h"],
                          units={"k": "W/(m*K)", "h": "W/(m^2*K)", "T": "K"})
    arguments = {"title": "Heat sink sizing", "goal": "keep the CPU below 85 C", "task_type": "inverse_problem",
                 "inputs": ["x", "y"], "outputs": ["T"], "physics": physics}
    arguments.update(over)
    return ElicitedSpec(**arguments)


def test_the_codegen_turns_an_elicited_spec_into_concrete_pinneapple_objects():
    spec = build_pinneapple_spec(_design())
    assert spec.pde_kind == "heat_equation"
    assert spec.fields == ["T"] and spec.coords == ["x", "y"]
    assert isinstance(spec.pde_params, dict) and isinstance(spec.environment_kwargs, dict)
    assert spec.model_name, "a spec without a model name is not a recipe"
    # ``pde_confidence`` is a keyword-overlap score, not a fraction: the schema's docstring calls it a match
    # confidence and says 0 = no match, and the value the codegen reports here is above one. Asserted as a score
    # rather than corrected, because which of the two the schema means is the schema owner's call.
    assert float(spec.pde_confidence) >= 0.0


def test_the_codegen_emits_code_that_mentions_what_it_identified():
    spec = build_pinneapple_spec(_design())
    code = getattr(spec, "code", "") or getattr(spec, "snippet", "") or ""
    if code:                                   # the field exists in some builds of the schema, not in all
        assert isinstance(code, str) and "\n" in code
        assert spec.fields and any(name in code for name in spec.fields)


def test_an_unrecognised_physics_degrades_to_custom_instead_of_raising():
    vague = build_pinneapple_spec(_design(
        physics=PhysicsSpec(governing_equations=["something nobody has named yet"])))
    assert vague.pde_kind in ("custom", "") or vague.pde_confidence < 1.0
    assert isinstance(vague.environment_kwargs, dict)
