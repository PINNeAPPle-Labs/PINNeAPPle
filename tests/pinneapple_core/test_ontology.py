"""A ontologia física: um conceito, um tipo, e os três consumidores produzindo o mesmo objeto (#204).

O aceite do #204 é que o esquema seja **publicado** e que ``ProblemSpec``, o manifesto do dataset e o cartão de modelo
**reusem** os conceitos em vez de os redefinirem. As duas metades têm teste aqui:

* o esquema publicado (``docs/schemas/physics_ontology.json``) é comparado com o que o código gera hoje — se alguém
  mudar um tipo e não republicar, o teste reprova;
* o **mesmo caso** (Burgers 1-D: ``u_t + u u_x = nu u_xx`` em ``x`` e ``t``, com ``nu = 0.01``) é descrito pelos três
  caminhos, e os conceitos que os três de fato descrevem têm de sair **iguais** — ``Field`` e ``Parameter`` objeto a
  objeto, a condição de contorno entre o dataset e o cartão, a equação governante pela expressão e pelos campos que
  ela cita.

O que os três descrevem de jeitos diferentes é dito e testado como diferente: o ``ProblemSpec`` não declara eixos por
campo nem unidades fora de ``meta["units"]``, o cartão não diz sobre que eixos o campo vive, e a condição de contorno
do ``ProblemSpec`` é um predicado (``value_fn``) que o esquema não tenta adivinhar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pinneapple_core.ontology import (
    Axis,
    BoundaryCondition,
    ConstitutiveLaw,
    Domain,
    Field,
    Geometry,
    GoverningEquation,
    InitialCondition,
    Material,
    Model,
    Observation,
    Parameter,
    PhysicalSystem,
    Solver,
    json_schema,
    publish_schema,
    unit_or_dimensionless,
)

REPO = Path(__file__).resolve().parents[2]
PUBLISHED = REPO / "docs" / "schemas" / "physics_ontology.json"


# --------------------------------------------------------------------------- o caso, descrito pelos três caminhos
def burgers_problem_spec():
    from pinneapple_physics.pde_environment.conditions import ConditionSpec
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec

    def zero(x, ctx):  # noqa: ARG001 -- the compiler's own condition signature
        return np.zeros(len(x))

    return ProblemSpec(
        name="burgers_1d", dim=2, coords=("x", "t"), fields=("u",),
        pde=PDETermSpec(kind="burgers", fields=("u",), coords=("x", "t"), params={"nu": 0.01}),
        conditions=(ConditionSpec(name="x_low", kind="dirichlet", fields=("u",), value_fn=zero),
                    ConditionSpec(name="start", kind="initial", fields=("u",), value_fn=zero)),
        domain_bounds={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        field_ranges={"u": (-1.0, 1.0)},
        meta={"units": {"u": "1", "x": "m", "t": "s", "nu": "1"},
              "equation": "u_t + u u_x = nu u_xx"},
        solver_spec={"name": "fd", "family": "fdm", "library": "numpy"},
    )


def burgers_manifest():
    from pinneapple_data.schema.models import (
        ArrayRef,
        AxisSpec,
        BundleManifest,
        Fields,
        Mesh,
        Physics,
        Provenance,
    )
    from pinneapple_data.schema.models import (
        BoundaryCondition as DSBoundaryCondition,
    )
    from pinneapple_data.schema.models import (
        Field as DSField,
    )
    from pinneapple_data.schema.models import (
        Parameter as DSParameter,
    )

    return BundleManifest(
        id="burgers_1d", title="burgers",
        mesh=Mesh(kind="grid", dimension=2, axes=[
            AxisSpec("x", "m", ArrayRef(key="axis:x", shape=(8,), dtype="float64")),
            AxisSpec("t", "s", ArrayRef(key="axis:t", shape=(4,), dtype="float64")),
        ]),
        fields=Fields(variables=[DSField(name="u", unit="1", axes=["x", "t"],
                                        data=ArrayRef(key="field:u", shape=(8, 4), dtype="float64"))]),
        physics=Physics(governing="u_t + u u_x = nu u_xx",
                        parameters=[DSParameter(name="nu", value=0.01, unit="1")],
                        boundary_conditions=[DSBoundaryCondition(field="u", region="x_low", kind="dirichlet",
                                                                 value=0.0)]),
        provenance=Provenance(source="a run", license="apache-2.0"),
    )


def burgers_card():
    from pinneapple_hub.model_card import ModelCard

    return ModelCard(
        name="burgers", architecture="modified_mlp",
        inputs=[{"name": "x", "unit": "m"}, {"name": "t", "unit": "s"}],
        outputs=[{"name": "u", "unit": "1"}],
        equations=["u_t + u u_x = nu u_xx"],
        domain_bounds={"x": [0.0, 1.0, "m"], "t": [0.0, 1.0, "s"]},
        boundary_conditions={"u": {"region": "x_low", "kind": "dirichlet", "value": 0.0, "unit": "1"}},
    )


def test_the_three_consumers_produce_the_same_field_and_parameter():
    """``Field`` e ``Parameter`` são o mesmo objeto vindo de um ``ProblemSpec`` e de um manifesto de dataset."""
    from_spec = burgers_problem_spec().to_system()
    from_bundle = burgers_manifest().physical_system()

    spec_field, bundle_field = from_spec.field("u"), from_bundle.field("u")
    # a faixa de validade é o único item que o ProblemSpec declara (field_ranges) e o manifesto do dataset não
    assert spec_field.model_dump(exclude={"value_range"}) == bundle_field.model_dump(exclude={"value_range"})
    assert spec_field.value_range == (-1.0, 1.0) and bundle_field.value_range is None
    assert spec_field.axes == ["x", "t"]
    assert from_spec.parameter("nu") == from_bundle.parameter("nu")
    assert from_spec.parameter("nu").value == 0.01


def test_the_dataset_and_the_card_agree_on_the_boundary_condition():
    """A condição de contorno do manifesto e a do cartão de modelo são o mesmo objeto."""
    from_bundle = burgers_manifest().physical_system()
    from_card = burgers_card().to_system()

    assert from_bundle.boundary_conditions[0] == from_card.boundary_conditions[0]
    assert from_bundle.boundary_conditions[0].region == "x_low"


def test_the_equation_is_the_same_expression_with_the_same_cited_fields():
    """A equação governante: mesma expressão e mesmos campos/parâmetros citados pelos três caminhos."""
    systems = [burgers_problem_spec().to_system(), burgers_manifest().physical_system(),
               burgers_card().to_system()]
    for system in systems:
        equation = system.governing_equations[0]
        assert equation.expression == "u_t + u u_x = nu u_xx"
        assert equation.fields == ["u"]
    # o ProblemSpec cita o parâmetro da equação; o cartão não o declara, e por isso não o cita
    assert systems[0].governing_equations[0].parameters == ["nu"]
    assert systems[2].governing_equations[0].parameters == []


def test_the_card_turns_its_coordinates_into_axes_not_fields():
    """Um ``input`` do cartão que é eixo do domínio vira eixo, não campo: a entrada de uma PINN é a coordenada."""
    system = burgers_card().to_system()
    assert [a.name for a in system.domain.axes] == ["x", "t"]
    assert system.domain.axis("x").unit == "m"
    assert [f.name for f in system.fields] == ["u"]


def test_the_problem_spec_carries_units_and_conditions_into_the_system():
    """Do ``ProblemSpec``: unidades de ``meta``, cotas do domínio, condição inicial separada da de contorno."""
    system = burgers_problem_spec().to_system()
    assert system.domain.axis("x").bounds == (0.0, 1.0)
    assert system.domain.axis("x").unit == "m"
    assert [c.field for c in system.boundary_conditions] == ["u"]
    assert all(isinstance(c, BoundaryCondition) for c in system.boundary_conditions)
    assert [c.field for c in system.initial_conditions] == ["u"]
    assert system.solver.family == "fdm" and system.solver.library == "numpy"
    assert system.geometry.units == "m"


def test_a_problem_spec_without_units_records_dimensionless_instead_of_guessing():
    """Sem ``meta["units"]`` a unidade é ``"1"``: não se adivinha ``m/s`` pelo nome do campo."""
    spec = burgers_problem_spec()
    spec.meta.pop("units")
    field = spec.to_system().field("u")
    assert field.unit == "1"


# ------------------------------------------------------------------------------------------ o esquema publicado
def test_the_published_schema_is_the_one_the_code_generates_today():
    """O esquema está publicado no repositório e não saiu de sincronia com os tipos."""
    assert PUBLISHED.is_file(), f"{PUBLISHED} is not published: run publish_schema on it"
    published = json.loads(PUBLISHED.read_text())
    assert published == json_schema()
    assert published["$id"] == "pinneapple.physics-ontology/1"


def test_the_published_schema_covers_the_thirteen_concepts_of_the_issue():
    """Os treze conceitos do #204 estão no esquema, cada um com o seu nome."""
    schema = json_schema()
    defs = set(schema["$defs"]) | {schema["title"]}   # o agregado é o documento, não uma definição dentro dele
    expected = {
        "PhysicalSystem", "Domain", "Geometry", "Material", "Field", "Parameter", "GoverningEquation",
        "ConstitutiveLaw", "BoundaryCondition", "InitialCondition", "Observation", "Solver", "Model",
    }
    assert expected <= defs, f"missing from the schema: {sorted(expected - defs)}"


def test_publish_writes_what_it_returns(tmp_path):
    target = publish_schema(tmp_path / "nested" / "ontology.json")
    assert json.loads(target.read_text()) == json_schema()


# ------------------------------------------------------------------------------------------ o sistema cobra
def _system(**overrides):
    base = dict(
        name="channel",
        domain=Domain(axes=[Axis(name="x", unit="m", bounds=(0.0, 1.0))]),
        fields=[Field(name="u", unit="m/s", axes=["x"]), Field(name="p", unit="Pa", axes=["x"])],
        parameters=[Parameter(name="mu", value=1e-3, unit="Pa*s")],
        governing_equations=[GoverningEquation(name="ns", expression="rho du/dt = -grad p + mu lap u",
                                               fields=["u"], parameters=["mu"])],
    )
    base.update(overrides)
    return PhysicalSystem(**base)


def test_the_thirteen_concepts_are_all_constructible():
    """Cada tipo do esquema aceita os seus próprios campos e recusa os dos outros (``extra="forbid"``)."""
    system = _system(
        geometry=Geometry(name="plate", kind="cad", dimension=1, units="mm", files=["plate.step"]),
        materials=[Material(name="water", properties={"rho": 998.0}, units={"rho": "kg/m^3"})],
        constitutive_laws=[ConstitutiveLaw(name="newtonian", expression="tau = mu du/dy", applies_to="ns",
                                           parameters=["mu"])],
        boundary_conditions=[BoundaryCondition(field="u", region="x_min", kind="inlet", value=1.0, unit="m/s")],
        initial_conditions=[InitialCondition(field="u", value=0.0, unit="m/s")],
        observations=[Observation(name="piv", field="u", unit="m/s", kind="experiment", uncertainty=0.02)],
        solver=Solver(name="simpleFoam", family="fvm", library="OpenFOAM", settings={"n_iter": 500}),
        model=Model(name="rom", family="rom", predicts=["u"], trained_on="piv"),
    )
    assert system.summary()["boundary_conditions"] == 1
    assert system.field("u").is_vector() is False
    with pytest.raises(ValidationError):
        Field(name="u", unit="m/s", colour="blue")  # conceito com campo que não existe


def test_an_equation_that_cites_a_field_the_system_does_not_have_is_refused():
    with pytest.raises(ValidationError, match="cites \\['v'\\]"):
        _system(governing_equations=[GoverningEquation(name="ns", expression="...", fields=["u", "v"])])


def test_a_boundary_condition_on_a_field_the_system_does_not_have_is_refused():
    with pytest.raises(ValidationError, match="cites \\['w'\\]"):
        _system(boundary_conditions=[BoundaryCondition(field="w", region="x_min", value=0.0)])


def test_a_model_that_predicts_an_undeclared_field_is_refused():
    with pytest.raises(ValidationError, match="model 'rom' cites"):
        _system(model=Model(name="rom", predicts=["T"]))


def test_a_field_on_an_axis_the_domain_does_not_have_is_refused():
    with pytest.raises(ValidationError, match="cites \\['y'\\]"):
        _system(fields=[Field(name="u", unit="m/s", axes=["y"])])


def test_a_condition_that_prescribes_nothing_is_refused_by_the_system():
    """Uma "dirichlet" sem valor nem expressão não é condição; a checagem fica no sistema, não na construção."""
    loose = BoundaryCondition(field="u", region="x_min", kind="dirichlet")  # aceita: é um registro ainda vazio
    with pytest.raises(ValidationError, match="prescribes nothing"):
        _system(boundary_conditions=[loose])
    # e uma condição que impõe por natureza (parede, simetria) não precisa de valor
    assert BoundaryCondition(field="u", region="wall", kind="wall").prescribes_something()


def test_duplicated_names_are_refused():
    with pytest.raises(ValidationError, match="duplicated field names"):
        _system(fields=[Field(name="u"), Field(name="u")])


def test_a_material_property_without_a_unit_is_refused():
    with pytest.raises(ValidationError, match="without a unit"):
        Material(name="water", properties={"rho": 998.0}, units={})


def test_a_constitutive_law_that_applies_to_nothing_known_is_refused():
    with pytest.raises(ValidationError, match="neither a field nor an equation"):
        _system(constitutive_laws=[ConstitutiveLaw(name="law", applies_to="energy")])


# ------------------------------------------------------------------------------------------ unidades e JSON
def test_units_are_parsed_at_construction():
    """A mesma regra de unidade do esquema do dataset, agora uma só: adimensional normalizado, unidade inválida é erro."""
    assert Field(name="u", unit="-").unit == "1"
    assert Field(name="u", unit="dimensionless").unit == "1"
    assert Field(name="u", unit="unknown").unit == "unknown"
    assert Field(name="u", unit="kg/m^3").unit == "kg/m^3"
    with pytest.raises(ValueError, match="not a unit this library knows"):
        Field(name="u", unit="furlongs per fortnight")
    assert unit_or_dimensionless("m/s") == "m/s"


def test_the_system_round_trips_through_json():
    """O sistema vai para JSON e volta sendo o mesmo objeto — é o que torna o esquema publicável."""
    system = burgers_problem_spec().to_system()
    document = json.loads(system.to_json())
    assert PhysicalSystem.from_dict(document) == system
    assert document["name"] == "burgers_1d"


def test_the_dataset_schema_imports_the_shared_unit_rule():
    """O ``unit_or_dimensionless`` do esquema do dataset é a função da ontologia, não uma cópia."""
    from pinneapple_data.schema import models

    assert models.unit_or_dimensionless is unit_or_dimensionless
    # e as classes do dataset herdam dos conceitos, em vez de os redefinir
    from pinneapple_data.schema.models import (
        BoundaryCondition as DSBoundaryCondition,
    )
    from pinneapple_data.schema.models import (
        Field as DSField,
    )
    from pinneapple_data.schema.models import (
        InitialCondition as DSInitialCondition,
    )
    from pinneapple_data.schema.models import (
        Material as DSMaterial,
    )
    from pinneapple_data.schema.models import (
        Parameter as DSParameter,
    )
    for dataset_class, concept in ((DSField, Field), (DSParameter, Parameter), (DSBoundaryCondition, BoundaryCondition),
                                   (DSInitialCondition, InitialCondition), (DSMaterial, Material)):
        assert issubclass(dataset_class, concept)
        assert dataset_class is not concept
