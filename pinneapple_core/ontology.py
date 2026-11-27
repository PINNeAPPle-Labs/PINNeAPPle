"""A ontologia física: um tipo formal por conceito, e os três consumidores usando o mesmo (#204).

O mesmo conceito aparecia escrito três vezes na biblioteca, cada vez com o seu formato: um campo era ``spec.fields``
(uma tupla de nomes) no ``ProblemSpec``, um ``Field`` com unidade e localização no manifesto do dataset, e uma linha
de ``inputs``/``outputs`` (um dicionário com ``name``/``unit``) no cartão de modelo. Uma condição de contorno era
``ConditionSpec(kind="dirichlet", value_fn=...)`` no primeiro, ``BoundaryCondition(field, region, kind, value)`` no
segundo e ``boundary_conditions: dict`` no terceiro. Nada disso era falso — o problema é que eram três definições do
mesmo conceito, e três definições divergem.

Este módulo é a definição única. Os treze conceitos do #204 estão aqui como modelos ``pydantic`` com
``extra="forbid"``:

======================  ==========================================================================
conceito                o que fixa
======================  ==========================================================================
``PhysicalSystem``      o agregado: tudo abaixo, com as referências cruzadas conferidas
``Domain``              os eixos do domínio, com unidade e limites (``Axis``)
``Geometry``            a forma e os arquivos que a descrevem, e a unidade em que eles estão
``Material``            um material nomeado com propriedades **com unidade**
``Field``               uma variável: nome, unidade, componentes, eixos e papel
``Parameter``           um parâmetro do caso: valor, unidade, procedência
``GoverningEquation``   a equação que o caso satisfaz, em texto, com os campos e parâmetros que ela usa
``ConstitutiveLaw``     a relação constitutiva, e à qual campo ou equação ela se aplica
``BoundaryCondition``   qual campo, em qual região, de qual tipo, com que valor
``InitialCondition``    o estado inicial de um campo transiente
``Observation``         uma medida do campo, com a incerteza que ela carrega
``Solver``              quem resolve (família e biblioteca), e com que ajustes
``Model``               o que se treinou, o que ele prevê e com que dados
======================  ==========================================================================

As unidades passam pelo mesmo validador do resto da biblioteca (:func:`unit_or_dimensionless`, que é a função que o
esquema do dataset já usava e que agora mora aqui): um ``unit`` que não analisa é erro na construção, não descoberto
quatro camadas depois.

O que a montagem de um :class:`PhysicalSystem` **cobra**, e é o motivo de o agregado existir
---------------------------------------------------------------------------------------------
Um conjunto de objetos soltos sempre parece consistente. Juntá-los faz as perguntas chatas:

* a equação governante cita um campo que não está na lista de campos? Erro.
* a condição de contorno aponta para um campo inexistente, ou para uma região vazia? Erro.
* o modelo diz que prevê um campo que o sistema não declara? Erro.
* dois campos com o mesmo nome? Dois parâmetros com o mesmo nome? Erro.
* o eixo que o campo indexa não está no domínio? Erro.
* uma condição de Dirichlet/Neumann/Robin sem valor nem expressão? Erro.

Reuso pelos três consumidores, e onde ele está
----------------------------------------------
* ``ProblemSpec.to_system()`` (:mod:`pinneapple_physics.pde_environment.spec`) — domínio, campos, parâmetros,
  equação, condições e solver do problema;
* ``PhysicsDataBundle`` / ``BundleManifest.physical_system()`` (:mod:`pinneapple_data.schema.bundle`) — o manifesto do
  dataset, e as classes ``Field``/``Parameter``/``BoundaryCondition``/``InitialCondition``/``Material`` do próprio
  esquema do dataset, que agora **herdam** daqui em vez de redefinir;
* ``ModelCard.to_system()`` (:mod:`pinneapple_hub.model_card`) — os ``inputs``/``outputs`` do cartão viram campos
  com unidade analisada, e ``equations``/``boundary_conditions`` viram equação e condições tipadas.

O teste que responde ao aceite de verdade está em ``tests/pinneapple_core/test_ontology.py``: o **mesmo caso**
descrito pelos três caminhos tem de produzir conceitos **iguais** (``Field``, ``Parameter``, ``GoverningEquation``),
não equivalentes em prosa.

Limite declarado: o ``ProblemSpec`` não carrega unidades nesta versão. Quando ele não tem `meta["units"]`, o sistema
gerado marca a unidade como ``"1"`` — dimensão adimensional — em vez de adivinhar ``"m/s"`` pelo nome do campo.

    >>> from pinneapple_core.ontology import Field, GoverningEquation, Parameter, PhysicalSystem
    >>> u = Field(name="u", unit="1", components=["u"], axes=["x"])
    >>> nu = Parameter(name="nu", value=0.01, unit="1")
    >>> system = PhysicalSystem(
    ...     name="burgers",
    ...     fields=[u],
    ...     parameters=[nu],
    ...     governing_equations=[GoverningEquation(name="burgers", expression="u_t + u u_x = nu u_xx",
    ...                                            fields=["u"], parameters=["nu"])],
    ... )
    >>> system.fields[0].unit, len(system.governing_equations)
    ('1', 1)
    >>> Field(name="p", unit="unknown").unit        # "unknown" é uma unidade que existe e não foi anotada
    'unknown'
    >>> Field(name="p", unit="-").unit              # os dois jeitos de escrever adimensional viram um só
    '1'

A unidade é analisada **na construção**: ``Field(name="p", unit="m/s+")`` levanta ``ValueError`` dizendo que
``m/s+`` não é uma unidade que a biblioteca conhece, em vez de deixar o erro aparecer quatro camadas depois.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic import Field as PField

__all__ = [
    "SCHEMA_ID",
    "unit_or_dimensionless",
    "Axis",
    "Domain",
    "Geometry",
    "Material",
    "Field",
    "Parameter",
    "GoverningEquation",
    "ConstitutiveLaw",
    "BoundaryCondition",
    "InitialCondition",
    "Observation",
    "Solver",
    "Model",
    "PhysicalSystem",
    "json_schema",
    "publish_schema",
]

SCHEMA_ID = "pinneapple.physics-ontology/1"

#: As grafias de "adimensional" que aparecem na prática (NASA CFL3D escreve ``1``, OpenFOAM escreve ``-``).
_DIMENSIONLESS = frozenset({"1", "-", "", "dimensionless", "[-]"})
#: Uma unidade que se sabe existir mas não foi anotada (um deck CalculiX em "unidades de modelo"). Não é adimensional.
UNKNOWN_UNIT = "unknown"

_UNIT_DOC = ("a unidade é analisada na construção; '1' (ou '-', 'dimensionless', '[-]') é adimensional e 'unknown' "
             "é uma unidade que existe mas não foi anotada")


def unit_or_dimensionless(text: str) -> str:
    """Devolve ``text`` se for uma unidade analisável, se for uma grafia de adimensional, ou ``"unknown"``.

    É a mesma regra que o esquema do dataset aplicava por conta própria; agora mora aqui e os dois lados a importam.
    A importação de ``parse_unit`` é feita dentro da função porque ``pinneapple_data`` importa este módulo (pelo
    esquema do dataset) — no topo, isso seria um ciclo de importação.
    """
    raw = (text or "").strip()
    if raw.lower() in _DIMENSIONLESS:
        return "1"
    if raw.lower() == UNKNOWN_UNIT:
        return UNKNOWN_UNIT
    from pinneapple_data.physical_units import UnitError, parse_unit

    try:
        parse_unit(raw)
    except UnitError as exc:
        raise ValueError(f"'{text}' is not a unit this library knows ({exc}); use \"1\" for dimensionless") from exc
    return raw


class _Concept(BaseModel):
    """Base de todo conceito: campo desconhecido é erro, e o nome vazio não existe."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ----------------------------------------------------------------------------------------------------- domínio
class Axis(_Concept):
    """Um eixo do domínio: o nome (``x``, ``t``), a unidade das suas coordenadas e, quando conhecidos, os limites."""

    name: str = PField(min_length=1)
    unit: str = "1"
    bounds: tuple[float, float] | None = None

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))

    @model_validator(mode="after")
    def _ordered(self) -> Axis:
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError(f"axis '{self.name}': bounds must be (min, max), got {self.bounds}")
        return self


class Domain(_Concept):
    """O domínio do sistema: os eixos que o definem, cada um com a sua unidade."""

    name: str = ""
    axes: list[Axis] = PField(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> Domain:
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"axis names must be unique; got {names}")
        return self

    def axis(self, name: str) -> Axis:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(f"no axis '{name}' in {[a.name for a in self.axes]}")

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Os limites por eixo; eixo sem limites fica de fora (não é ``(0, 1)`` por omissão)."""
        return {a.name: a.bounds for a in self.axes if a.bounds is not None}


class Geometry(_Concept):
    """A forma do domínio e os arquivos que a descrevem, com a unidade em que esses arquivos estão.

    ``units`` é a unidade do arquivo, não a da simulação: um STEP em ``mm`` resolvido em ``m`` registra ``"mm"`` aqui
    e ``"m"`` no eixo — é a diferença que some e multiplica um coeficiente de arrasto por ``1e-3``.
    """

    name: str = ""
    kind: Literal["cad", "mesh", "structured-grid", "analytic", "none"] = "analytic"
    dimension: int = PField(default=1, ge=0, le=3)
    units: str = "m"
    files: list[str] = PField(default_factory=list)
    bounding_box: tuple[float, ...] | None = None
    description: str = ""

    _units = field_validator("units")(lambda cls, v: unit_or_dimensionless(v))


class Material(_Concept):
    """Um material nomeado com as suas propriedades e a unidade de cada uma."""

    name: str = PField(min_length=1)
    properties: dict[str, float] = PField(default_factory=dict)
    units: dict[str, str] = PField(default_factory=dict)

    @field_validator("units")
    @classmethod
    def _units(cls, v: dict[str, str]) -> dict[str, str]:
        return {k: unit_or_dimensionless(u) for k, u in v.items()}

    @model_validator(mode="after")
    def _every_property_has_a_unit(self) -> Material:
        missing = sorted(set(self.properties) - set(self.units))
        if missing:
            raise ValueError(
                f"material '{self.name}': properties without a unit: {missing} -- a property without a unit is a "
                "number whose meaning is a guess"
            )
        return self


# ----------------------------------------------------------------------------------------------------- variáveis
class Field(_Concept):
    """Uma variável do sistema: nome, unidade, componentes, eixos que ela indexa e o papel que ela tem.

    ``role`` responde "por que este campo existe no sistema": ``solution`` é o que se resolve, ``input`` é o que se
    impõe, ``derived`` é calculado a partir dos outros e ``observed`` é o que se mede.
    """

    name: str = PField(min_length=1)
    symbol: str = ""
    unit: str = "1"
    components: list[str] = PField(default_factory=list)
    axes: list[str] = PField(default_factory=list)
    role: Literal["solution", "input", "derived", "observed"] = "solution"
    value_range: tuple[float, float] | None = None
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))

    @model_validator(mode="after")
    def _range_ordered(self) -> Field:
        if self.value_range is not None and self.value_range[0] > self.value_range[1]:
            raise ValueError(f"field '{self.name}': value_range must be (min, max), got {self.value_range}")
        return self

    def is_vector(self) -> bool:
        return len(self.components) > 1


class Parameter(_Concept):
    """Um parâmetro do caso (uma viscosidade, uma condutividade), com a unidade e de onde ele veio."""

    name: str = PField(min_length=1)
    value: float | int | None = None
    unit: str = "1"
    source: str = ""
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))


class Observation(_Concept):
    """Uma medida de um campo, e a incerteza que ela carrega (``None`` = não declarada, que não é zero)."""

    name: str = PField(min_length=1)
    field: str = PField(min_length=1)
    unit: str = "1"
    kind: Literal["experiment", "simulation", "field", "scalar"] = "experiment"
    source: str = ""
    uncertainty: float | None = None
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))

    @field_validator("uncertainty")
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and float(v) < 0:
            raise ValueError("an uncertainty is a magnitude: it cannot be negative")
        return v


# ----------------------------------------------------------------------------------------------------- física
class GoverningEquation(_Concept):
    """A equação que o sistema satisfaz, com os campos e parâmetros que ela cita.

    ``expression`` é o texto da equação (``"u_t + u u_x = nu u_xx"``): este esquema não finge analisar EDPs — quem
    faz isso é :mod:`pinneapple_physics.pde_environment` —, ele registra o enunciado para que a checagem a jusante
    saiba contra o que está checando.
    """

    name: str = PField(min_length=1)
    expression: str = ""
    kind: Literal["pde", "ode", "algebraic", "constraint"] = "pde"
    fields: list[str] = PField(default_factory=list)
    parameters: list[str] = PField(default_factory=list)
    description: str = ""


class ConstitutiveLaw(_Concept):
    """A relação constitutiva (``tau = mu du/dy``, ``q = -k grad T``) e a que campo ou equação ela se aplica."""

    name: str = PField(min_length=1)
    expression: str = ""
    applies_to: str = ""
    parameters: list[str] = PField(default_factory=list)
    description: str = ""


class BoundaryCondition(_Concept):
    """Uma condição de contorno: qual campo, em qual região, de qual tipo, com que valor ou expressão."""

    name: str = ""
    field: str = PField(min_length=1)
    region: str = PField(min_length=1)
    kind: Literal["dirichlet", "neumann", "robin", "periodic", "wall", "inlet", "outlet", "symmetry", "open",
                  "interface"] = "dirichlet"
    value: float | int | Sequence[float] | None = None
    expression: str = ""
    unit: str = "1"
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))

    def prescribes_something(self) -> bool:
        """A condição diz o que impõe? ``wall``/``symmetry``/``periodic``/``open`` impõem por si; as demais precisam
        de valor ou expressão. A checagem é feita pelo sistema, não na construção: um registro do dataset pode ter a
        condição ainda não preenchida."""
        if self.value is not None or self.expression.strip():
            return True
        return self.kind in ("periodic", "symmetry", "wall", "open")


class InitialCondition(_Concept):
    """O estado de onde um caso transiente parte."""

    name: str = ""
    field: str = PField(min_length=1)
    value: float | int | Sequence[float] | None = None
    expression: str = ""
    unit: str = "1"
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))


# ----------------------------------------------------------------------------------------------------- quem roda
class Solver(_Concept):
    """Quem resolve o sistema: a família numérica, a biblioteca e os ajustes que de fato foram usados."""

    name: str = ""
    family: Literal["fdm", "fem", "fvm", "lbm", "spectral", "pinn", "neural-operator", "rom", "hybrid",
                    "closed-form", "other"] = "other"
    library: str = ""
    settings: dict[str, Any] = PField(default_factory=dict)
    version: str = ""


class Model(_Concept):
    """O que se treinou: a família, os campos que ele prevê e sobre o que ele foi treinado."""

    name: str = PField(min_length=1)
    family: str = ""
    predicts: list[str] = PField(default_factory=list)
    trained_on: str = ""
    uncertainty_method: str = ""
    settings: dict[str, Any] = PField(default_factory=dict)
    version: str = ""


# ----------------------------------------------------------------------------------------------------- agregado
def _unique(values: Sequence[str], what: str) -> None:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    if duplicated:
        raise ValueError(f"duplicated {what}: {sorted(duplicated)}")


class PhysicalSystem(_Concept):
    """O sistema físico inteiro, e a hora em que as referências cruzadas são conferidas.

    Cada lista é opcional porque um dataset de tabela não tem equação governante e um cartão de modelo não tem malha.
    O que não é opcional é a coerência do que está declarado: tudo o que é citado (campo, parâmetro, eixo) tem de
    existir no próprio sistema.
    """

    name: str = PField(min_length=1)
    domain: Domain | None = None
    geometry: Geometry | None = None
    materials: list[Material] = PField(default_factory=list)
    fields: list[Field] = PField(default_factory=list)
    parameters: list[Parameter] = PField(default_factory=list)
    governing_equations: list[GoverningEquation] = PField(default_factory=list)
    constitutive_laws: list[ConstitutiveLaw] = PField(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = PField(default_factory=list)
    initial_conditions: list[InitialCondition] = PField(default_factory=list)
    observations: list[Observation] = PField(default_factory=list)
    solver: Solver | None = None
    model: Model | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _references_resolve(self) -> PhysicalSystem:
        field_names = [f.name for f in self.fields]
        parameter_names = [p.name for p in self.parameters]
        _unique(field_names, "field names")
        _unique(parameter_names, "parameter names")
        _unique([e.name for e in self.governing_equations], "equation names")
        _unique([m.name for m in self.materials], "material names")

        def _require(names: Sequence[str], known: Sequence[str], where: str) -> None:
            missing = [n for n in names if n not in known]
            if missing:
                raise ValueError(
                    f"{where} cites {missing}, which the system does not declare (it has "
                    f"{list(known) or 'none'})"
                )

        axis_names = [a.name for a in (self.domain.axes if self.domain else [])]
        for equation in self.governing_equations:
            _require(equation.fields, field_names, f"equation '{equation.name}'")
            _require(equation.parameters, parameter_names, f"equation '{equation.name}'")
        for law in self.constitutive_laws:
            _require(law.parameters, parameter_names, f"constitutive law '{law.name}'")
            if law.applies_to and law.applies_to not in (*field_names, *(e.name for e in self.governing_equations)):
                raise ValueError(
                    f"constitutive law '{law.name}' applies to '{law.applies_to}', which is neither a field nor an "
                    f"equation of the system"
                )
        for bc in self.boundary_conditions:
            _require([bc.field], field_names, f"boundary condition on '{bc.region}'")
            if not bc.prescribes_something():
                raise ValueError(
                    f"boundary condition on '{bc.field}' at '{bc.region}' of kind '{bc.kind}' declares neither a "
                    "value nor an expression -- a condition that prescribes nothing is not a condition"
                )
        for ic in self.initial_conditions:
            _require([ic.field], field_names, "initial condition")
        for obs in self.observations:
            _require([obs.field], field_names, f"observation '{obs.name}'")
        if self.model is not None:
            _require(self.model.predicts, field_names, f"model '{self.model.name}'")
        if axis_names:
            for field in self.fields:
                _require(field.axes, axis_names, f"field '{field.name}'")
        return self

    # -- leitura ---------------------------------------------------------------------------------------------
    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"no field '{name}' in {[f.name for f in self.fields]}")

    def parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"no parameter '{name}' in {[p.name for p in self.parameters]}")

    def summary(self) -> dict[str, int]:
        """Quantos objetos de cada conceito o sistema declara — o que se confere antes de comparar dois sistemas."""
        counts = {
            "fields": len(self.fields), "parameters": len(self.parameters),
            "governing_equations": len(self.governing_equations),
            "constitutive_laws": len(self.constitutive_laws),
            "boundary_conditions": len(self.boundary_conditions),
            "initial_conditions": len(self.initial_conditions),
            "observations": len(self.observations), "materials": len(self.materials),
        }
        return counts

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> PhysicalSystem:
        return cls.model_validate(document)

    # -- conversores: os três consumidores que passam a falar esta língua -------------------------------------
    @classmethod
    def from_problem_spec(cls, spec: Any) -> PhysicalSystem:
        """O sistema que um ``ProblemSpec`` descreve. Ver a nota do docstring do módulo sobre unidades: o
        ``ProblemSpec`` não as carrega, e o que não está em ``spec.meta["units"]`` fica ``"1"`` em vez de adivinhado.
        """
        meta = dict(getattr(spec, "meta", {}) or {})
        units = dict(meta.get("units") or {})
        field_ranges = dict(getattr(spec, "field_ranges", {}) or {})
        bounds = dict(getattr(spec, "domain_bounds", {}) or {})

        axes = [Axis(name=str(c), unit=str(units.get(c, units.get(f"axis:{c}", "1"))), bounds=bounds.get(c))
                for c in getattr(spec, "coords", ()) or ()]
        fields = [
            Field(name=str(name), unit=str(units.get(name, "1")),
                  axes=[str(c) for c in (getattr(spec, "coords", ()) or ())],
                  value_range=_range(field_ranges.get(str(name))),
                  role="solution")
            for name in getattr(spec, "fields", ()) or ()
        ]
        pde = getattr(spec, "pde", None)
        params = dict(getattr(pde, "params", {}) or {})
        parameters = [Parameter(name=str(k), value=_number(v), unit=str(units.get(k, "1")))
                      for k, v in params.items()]

        equations: list[GoverningEquation] = []
        if pde is not None:
            equations.append(GoverningEquation(
                name=str(getattr(pde, "kind", "") or "governing"),
                expression=str(meta.get("equation", "")),
                fields=[str(f) for f in (getattr(pde, "fields", ()) or ())],
                parameters=[str(p) for p in params],
                description="transcrito do PDE do ProblemSpec" if not meta.get("equation") else "",
            ))

        conditions = getattr(spec, "conditions", ()) or ()
        bcs: list[BoundaryCondition] = []
        ics: list[InitialCondition] = []
        for i, condition in enumerate(conditions):
            kind = str(getattr(condition, "kind", "dirichlet"))
            names = [str(f) for f in (getattr(condition, "fields", ()) or ())]
            label = str(getattr(condition, "name", "") or f"condition #{i}")
            selector = getattr(condition, "selector", None)
            # onde a condição vale: o seletor do ProblemSpec quando ele diz (um dicionário de tags), senão o nome da
            # própria condição -- e nunca um "qualquer lugar" inventado
            where = str(selector) if isinstance(selector, str) else (str(selector.get("tag", "")) if isinstance(
                selector, dict) else label)
            if kind == "initial":
                ics.extend(InitialCondition(name=label, field=n, expression="condição inicial do ProblemSpec",
                                            unit=str(units.get(n, "1")))
                           for n in names)
                continue
            bcs.extend(
                BoundaryCondition(name=label, field=n, region=where or label,
                                  kind=kind if kind in _BC_KINDS else "dirichlet",
                                  expression=f"condition '{label}' of the ProblemSpec",
                                  unit=str(units.get(n, "1")))
                for n in names
            )

        solver_spec = dict(getattr(spec, "solver_spec", {}) or {})
        solver = (
            Solver(name=str(solver_spec.get("name", "")),
                   family=solver_spec.get("family", "other"),
                   library=str(solver_spec.get("library", "")),
                   settings={k: v for k, v in solver_spec.items() if k not in ("name", "family", "library")})
            if solver_spec else None
        )

        return cls(
            name=str(getattr(spec, "name", "") or getattr(spec, "problem_id", "") or "problem"),
            domain=Domain(name=str(getattr(spec, "name", "") or ""), axes=axes),
            geometry=Geometry(name=str(getattr(spec, "name", "") or ""), kind="analytic",
                              dimension=int(getattr(spec, "dim", len(axes)) or len(axes)),
                              units=str(meta.get("geometry_units", "m")),
                              bounding_box=_bbox(bounds)),
            fields=fields,
            parameters=parameters,
            governing_equations=equations,
            boundary_conditions=bcs,
            initial_conditions=ics,
            solver=solver,
            notes="convertido de um ProblemSpec",
        )

    @classmethod
    def from_bundle_manifest(cls, manifest: Any) -> PhysicalSystem:
        """O sistema que o manifesto de um ``PhysicsDataBundle`` descreve (esquema ``pinneapple.physics-data/1``)."""
        mesh = getattr(manifest, "mesh", None)
        geometry = getattr(manifest, "geometry", None)
        physics = getattr(manifest, "physics", None)
        fields_model = getattr(manifest, "fields", None)

        axes = [Axis(name=str(a.name), unit=str(a.unit), bounds=None)
                for a in (getattr(mesh, "axes", None) or [])]
        fields = [
            Field(name=str(f.name), symbol=str(getattr(f, "symbol", "") or ""), unit=str(f.unit),
                  components=[str(c) for c in (getattr(f, "components", None) or [])],
                  axes=[str(a) for a in (getattr(f, "axes", None) or [])],
                  description=str(getattr(f, "description", "") or ""))
            for f in (getattr(fields_model, "variables", None) or [])
        ]
        parameters = [
            Parameter(name=str(p.name), value=_number(getattr(p, "value", None)), unit=str(p.unit),
                      source=str(getattr(p, "source", "") or ""),
                      description=str(getattr(p, "description", "") or ""))
            for p in (getattr(physics, "parameters", None) or [])
        ]
        bcs = [
            BoundaryCondition(field=str(bc.field), region=str(bc.region), kind=str(bc.kind),
                              value=_plain(getattr(bc, "value", None)), unit=str(bc.unit),
                              description=str(getattr(bc, "description", "") or ""))
            for bc in (getattr(physics, "boundary_conditions", None) or [])
        ]
        ics = [
            InitialCondition(field=str(ic.field), value=_plain(getattr(ic, "value", None)), unit=str(ic.unit),
                             description=str(getattr(ic, "description", "") or ""))
            for ic in (getattr(physics, "initial_conditions", None) or [])
        ]
        materials = [
            Material(name=str(m.name), properties=dict(getattr(m, "properties", {}) or {}),
                     units=dict(getattr(m, "units", {}) or {}))
            for m in (getattr(physics, "materials", None) or [])
        ]
        equations: list[GoverningEquation] = []
        governing = str(getattr(physics, "governing", "") or "")
        if governing:
            equations.append(GoverningEquation(
                name="governing", expression=governing,
                fields=[f.name for f in fields], parameters=[p.name for p in parameters],
            ))

        return cls(
            name=str(getattr(manifest, "title", "") or getattr(manifest, "id", "") or "dataset"),
            domain=Domain(axes=axes) if axes else None,
            geometry=Geometry(
                name=str(getattr(geometry, "description", "") or ""),
                kind=str(getattr(geometry, "kind", "analytic")),
                dimension=int(getattr(geometry, "dimension", len(axes)) or len(axes)),
                units=str(getattr(geometry, "units", "m")),
                files=[str(getattr(f, "path", "")) for f in (getattr(geometry, "files", None) or [])],
                bounding_box=tuple(getattr(geometry, "bounding_box", None) or ()) or None,
                description=str(getattr(geometry, "description", "") or ""),
            ) if geometry is not None else None,
            materials=materials, fields=fields, parameters=parameters, governing_equations=equations,
            boundary_conditions=bcs, initial_conditions=ics,
            notes="convertido de um manifesto de dataset",
        )

    @classmethod
    def from_model_card(cls, card: Any) -> PhysicalSystem:
        """O sistema que um ``ModelCard`` descreve: os ``inputs``/``outputs`` viram campos com unidade analisada."""
        def _fields(entries: Any, role: str) -> list[Field]:
            out: list[Field] = []
            for entry in entries or []:
                if isinstance(entry, str):
                    out.append(Field(name=entry, role=role))  # type: ignore[arg-type]
                    continue
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                out.append(Field(
                    name=str(entry["name"]), unit=str(entry.get("unit", "1") or "1"),
                    description=str(entry.get("description", "") or ""), role=role,  # type: ignore[arg-type]
                ))
            return out

        bounds = dict(getattr(card, "domain_bounds", None) or {})
        axes = [Axis(name=str(k), unit=str(v[2]) if isinstance(v, (list, tuple)) and len(v) > 2 else "1",
                     bounds=(float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) and len(v) >= 2 else None)
                for k, v in bounds.items()]
        axis_names = {a.name for a in axes}
        # um "input" do cartão que é um eixo do domínio (a entrada de uma PINN é a coordenada, não um campo) vira
        # a unidade daquele eixo; o resto é campo de entrada
        for axis in axes:
            for entry in getattr(card, "inputs", None) or []:
                if isinstance(entry, dict) and entry.get("name") == axis.name:
                    axis.unit = unit_or_dimensionless(str(entry.get("unit", axis.unit) or axis.unit))
        input_entries = [e for e in (getattr(card, "inputs", None) or [])
                         if not (isinstance(e, dict) and e.get("name") in axis_names)]
        fields = _fields(input_entries, "input") + _fields(getattr(card, "outputs", None), "solution")
        # a faixa de validade do cartão é a faixa do campo: mesmo conceito, mesmo lugar
        validity = getattr(card, "validity_range", None) or {}
        for field in fields:
            field.value_range = _range(validity.get(field.name))
        names = [f.name for f in fields]
        equations = [GoverningEquation(name=f"equation {i}", expression=str(text), fields=names)
                     for i, text in enumerate(getattr(card, "equations", None) or [])]
        bcs = [
            BoundaryCondition(field=str(k), region=str(v.get("region", "") or "unspecified"),
                              kind=str(v.get("kind", "dirichlet")),
                              value=v.get("value"), expression=str(v.get("expression", "") or ""),
                              unit=str(v.get("unit", "1") or "1"))
            for k, v in (getattr(card, "boundary_conditions", None) or {}).items() if isinstance(v, dict)
        ]
        ics = [
            InitialCondition(field=str(k), value=v.get("value"), unit=str(v.get("unit", "1") or "1"))
            for k, v in (getattr(card, "initial_condition", None) or {}).items() if isinstance(v, dict)
        ]
        architecture = str(getattr(card, "architecture", "") or "")
        model = Model(
            name=str(getattr(card, "name", "") or "model"), family=architecture,
            predicts=[f.name for f in fields if f.role != "input"],
            trained_on=str(getattr(card, "reference_source", "") or ""),
            settings=dict(getattr(card, "architecture_config", None) or {}),
            version=str(getattr(card, "version", "") or ""),
        )
        return cls(
            name=str(getattr(card, "name", "") or "model"),
            domain=Domain(axes=axes) if axes else None,
            fields=fields, governing_equations=equations, boundary_conditions=bcs, initial_conditions=ics,
            model=model, notes="convertido de um cartão de modelo",
        )


_BC_KINDS = ("dirichlet", "neumann", "robin", "periodic", "wall", "inlet", "outlet", "symmetry", "open", "interface")


def _range(value: Any) -> tuple[float, float] | None:
    """Uma faixa declarada (lista, tupla ou dicionário) como ``(min, max)``; o que não for uma faixa fica ``None``."""
    if isinstance(value, dict):
        value = (value.get("min"), value.get("max"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _plain(value: Any) -> Any:
    """Um valor de condição como o esquema o aceita (escalar ou sequência de escalares; o resto vira ``None``)."""
    if value is None or isinstance(value, (int, float, str)):
        return value if not isinstance(value, str) else None
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
        return list(value)
    return None


def _bbox(bounds: dict[str, tuple[float, float]]) -> tuple[float, ...] | None:
    if not bounds:
        return None
    return tuple(float(v) for key in sorted(bounds) for v in bounds[key])


# ----------------------------------------------------------------------------------------------------- publicação
def json_schema() -> dict[str, Any]:
    """O esquema JSON de :class:`PhysicalSystem` — o documento que se publica para outros consumidores."""
    schema = PhysicalSystem.model_json_schema()
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID, **schema}
    schema.setdefault("description", PhysicalSystem.__doc__ or "")
    return schema


def publish_schema(path: str | Path) -> Path:
    """Escreve o esquema JSON em ``path`` (o publicado no repositório é ``docs/schemas/physics_ontology.json``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target
