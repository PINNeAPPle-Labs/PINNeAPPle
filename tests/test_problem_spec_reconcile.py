"""The three problem specifications, reconciled: converters both ways and the fields they share (#150, D1).

The point of these tests is not that the converters run; it is that they are **explicit about what they lose**. So
each direction is checked on the fields the three specifications actually share (the table in
``pinneapple_physics.pde_environment.reconcile.SHARED_FIELDS``), and the loss is asserted too: a converter that
silently dropped the equation, or invented a value function from a string, would pass a "does it convert" test and
fail these.
"""
from __future__ import annotations

import pytest

import pinneapple as pp
from pinneapple_physics.pde_environment.reconcile import (LOSSES, SHARED_FIELDS, design_to_pde,
                                                          pde_condition_to_upd, pde_to_design, upd_condition_to_pde)

PRESET = "burgers_1d"


@pytest.fixture()
def pde_spec():
    return pp.PhysicalProblem.from_preset(PRESET).to_pde_spec()


# ------------------------------------------------------------------ the table itself
def test_every_shared_field_is_documented_and_every_loss_is_written_down():
    assert SHARED_FIELDS, "the table is the decision record's source; it cannot be empty"
    for name, (elicited, pde, upd) in SHARED_FIELDS.items():
        assert elicited or pde or upd, name
    for direction in ("design_to_pde", "pde_to_design", "upd_condition_to_pde", "pde_condition_to_upd"):
        assert direction in LOSSES and len(LOSSES[direction]) > 40, direction


# ------------------------------------------------------------------ PDE <-> elicited
def test_a_pde_problem_becomes_an_elicited_description(pde_spec):
    design = pde_to_design(pde_spec)
    assert design.title == pde_spec.name
    assert list(design.outputs) == list(pde_spec.fields)
    assert pde_spec.pde.kind in design.physics.governing_equations
    assert list(design.inputs) == list(pde_spec.coords)


def test_the_round_trip_recovers_the_shared_fields(pde_spec):
    back = design_to_pde(pde_to_design(pde_spec), pde_kind=pde_spec.pde.kind,
                         domain_bounds=pde_spec.domain_bounds)
    assert back.name == pde_spec.name
    assert tuple(back.fields) == tuple(pde_spec.fields)
    assert tuple(back.coords) == tuple(pde_spec.coords)
    assert back.pde.kind == pde_spec.pde.kind
    assert {k: tuple(v) for k, v in back.domain_bounds.items()} == {k: tuple(v) for k, v in
                                                                    pde_spec.domain_bounds.items()}


def test_the_elicited_side_keeps_its_free_text_when_it_goes_to_a_pde_spec(pde_spec):
    design = pde_to_design(pde_spec)
    design.goal = "know the temperature field"
    back = design_to_pde(design, pde_kind="burgers")
    assert back.meta["elicited"]["goal"] == "know the temperature field", "the prose has to survive somewhere"


def test_the_equation_kind_is_a_decision_not_a_guess(pde_spec):
    """Which ``pde.kind`` is meant is exactly what the elicitation cannot say, so it is required."""
    design = pde_to_design(pde_spec)
    with pytest.raises(TypeError):
        design_to_pde(design)                       # no pde_kind: refused at the signature
    back = design_to_pde(design, pde_kind="heat")
    assert back.pde.kind == "heat"


# ------------------------------------------------------------------ sampling <-> PDE conditions
class _UpdCondition:
    """The shape of ``pinneapple_data.upd_types.ConditionSpec`` (dataclass with name/type/equation/n/options)."""

    def __init__(self, name="ic", type="initial", equation="u = sin(pi*x)", n=512, options=None):
        self.name, self.type, self.equation, self.n = name, type, equation, n
        self.options = dict(options or {})


def test_a_sampling_condition_becomes_a_pde_condition_and_back():
    bridge = upd_condition_to_pde(_UpdCondition(), fields=["u"])
    assert bridge.condition.kind == "initial" and bridge.condition.fields == ("u",)
    assert bridge.carried["n"] == 512 and bridge.carried["equation"] == "u = sin(pi*x)"
    back = pde_condition_to_upd(bridge.condition, carried=bridge.carried)
    assert (back.name, back.type, back.equation, back.n) == ("ic", "initial", "u = sin(pi*x)", 512)


def test_without_a_value_the_conversion_says_what_it_could_not_do():
    """A PDE condition needs a value function; a sampling condition only has a string. Guessing is not an option."""
    bridge = upd_condition_to_pde(_UpdCondition(type="boundary"), fields=["u"])
    assert bridge.condition.value_fn is None
    assert any("value_fn" in line for line in bridge.lost)
    with_value = upd_condition_to_pde(_UpdCondition(type="boundary"), fields=["u"], value=0.0)
    assert with_value.condition.value_fn(None) == 0.0 and not with_value.lost


@pytest.mark.parametrize("upd_type,kind", [("initial", "initial"), ("boundary", "dirichlet"),
                                           ("interior", "data"), ("slice", "dirichlet")])
def test_each_sampling_type_lands_on_a_pde_kind_that_suits_it(upd_type, kind):
    bridge = upd_condition_to_pde(_UpdCondition(type=upd_type), fields=["u"], value=1.0)
    assert bridge.condition.kind == kind


def test_the_upd_condition_is_the_real_class():
    from pinneapple_data.upd_types import ConditionSpec as SamplingCondition
    real = SamplingCondition(name="wall", type="boundary", equation="T = 300", n=64)
    bridge = upd_condition_to_pde(real, fields=["T"], value=300.0)
    back = pde_condition_to_upd(bridge.condition, carried=bridge.carried)
    assert isinstance(back, SamplingCondition) and back.equation == "T = 300" and back.n == 64
