"""``PhysicalProblem``: one description of a physics problem for every part of the library.

Before this module there were three overlapping classes, each covering part of a problem:

* ``pinneapple_physics.pde_environment.spec.ProblemSpec`` -- the executable PDE definition (coordinates,
  fields, PDE term, boundary/initial/data conditions, domain bounds). The PINN compiler consumes it.
* ``pinneapple_problemdesign.schema.ProblemSpec`` -- the elicited, mostly free-text description of a problem
  (goal, task type, data, validation and deployment needs) produced by the problem-design assistant.
* ``pinneapple_data.physics_case.PhysicsCase`` -- a bundle that references geometry, a PDE ``ProblemSpec``,
  a solver choice, results and a reference benchmark.

``PhysicalProblem`` holds all of that in one place and adds what none of them had: **parameters with units
and a role** (fixed, design, uncertain, unknown), so the same object can drive a forward solve, a design
optimization, an uncertainty study or a parameter-discovery run, plus **quantities of interest** and a
stable **fingerprint** for provenance.

The three existing classes keep working unchanged. They are adapters:

>>> import pinneapple as pp
>>> prob = pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01)
>>> spec = prob.to_pde_spec()                 # what the PINN compiler consumes, same object shape as before
>>> pp.PhysicalProblem.from_pde_spec(spec).to_pde_spec() == spec
True

Units are optional but checked when given (``pinneapple_data.physical_units``); ``validate()`` reports
inconsistencies (unknown fields in a condition, a value outside its bounds, an unparseable unit) as a list
of readable strings instead of failing later inside a solver.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .pde_environment.conditions import ConditionSpec
from .pde_environment.scales import ScaleSpec
from .pde_environment.spec import PDETermSpec, ProblemSpec

__all__ = ["Parameter", "Quantity", "Objective", "Constraint", "Fidelity", "PhysicalProblem",
           "PARAMETER_ROLES", "CONSTRAINT_FAMILIES"]

PARAMETER_ROLES = ("fixed", "design", "uncertain", "unknown")

#: the six kinds of limit #221 names, plus ``""`` for a constraint written before the vocabulary existed. The label
#: travels with the constraint into training, the optimizer and the deployment guard; it never changes the arithmetic.
CONSTRAINT_FAMILIES = ("", "physical", "conservation", "operational", "material", "actuator", "safety")


# --------------------------------------------------------------------------- building blocks
@dataclass(frozen=True)
class Parameter:
    """A named physical parameter.

    role:
      * ``fixed``     -- known value, held constant (the default).
      * ``design``    -- a design variable an optimizer may move within ``bounds``.
      * ``uncertain`` -- known only as a distribution (``distribution``) or interval (``bounds``); used by UQ.
      * ``unknown``   -- to be identified from data (parameter discovery); ``value`` is the initial guess.

    ``unit`` is a unit string such as ``"m^2/s"`` or ``"Pa"`` (``None`` means not stated, ``"1"`` means
    dimensionless). ``distribution`` is a plain dict, e.g. ``{"type": "normal", "mean": 0.01, "std": 0.001}``.
    """
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    role: str = "fixed"
    bounds: Optional[Tuple[float, float]] = None
    distribution: Optional[Dict[str, Any]] = None
    description: str = ""

    def issues(self) -> List[str]:
        out: List[str] = []
        if self.role not in PARAMETER_ROLES:
            out.append(f"parameter '{self.name}': role '{self.role}' is not one of {PARAMETER_ROLES}")
        if self.bounds is not None:
            lo, hi = self.bounds
            if not lo <= hi:
                out.append(f"parameter '{self.name}': bounds {self.bounds} have lower > upper")
            elif _is_number(self.value) and not lo <= float(self.value) <= hi:
                out.append(f"parameter '{self.name}': value {self.value} is outside bounds {self.bounds}")
        if self.role == "design" and self.bounds is None:
            out.append(f"parameter '{self.name}': a design parameter needs bounds")
        if self.role == "uncertain" and self.bounds is None and not self.distribution:
            out.append(f"parameter '{self.name}': an uncertain parameter needs a distribution or bounds")
        if self.role == "fixed" and self.value is None:
            out.append(f"parameter '{self.name}': a fixed parameter needs a value")
        if self.unit is not None and self.unit not in ("", "1"):
            from pinneapple_data.physical_units import try_parse_unit
            if try_parse_unit(self.unit) is None:
                out.append(f"parameter '{self.name}': unit '{self.unit}' is not recognised")
        return out


@dataclass(frozen=True)
class Quantity:
    """A quantity of interest: what the user actually wants out of the solution (a field, a coefficient,
    a peak value). ``field`` names the model output it derives from, when there is one."""
    name: str
    unit: Optional[str] = None
    field: Optional[str] = None
    description: str = ""


@dataclass
class Objective:
    """What the problem is *for*: a quantity to minimise, maximise, or drive to a target (issue #151).

    ``quantity`` names a ``Quantity`` (or a field) of the same problem; ``target`` is required when ``sense`` is
    ``"target"`` and ignored otherwise. ``weight`` lets several objectives be combined without losing which one
    mattered more, and ``unit`` is the unit of ``target``/``tolerance``.
    """
    name: str
    quantity: str = ""
    sense: str = "minimize"          # minimize | maximize | target
    target: Optional[float] = None
    tolerance: Optional[float] = None
    weight: float = 1.0
    unit: Optional[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.sense not in ("minimize", "maximize", "target"):
            raise ValueError(f"objective {self.name!r}: sense must be minimize, maximize or target, got {self.sense!r}")
        if self.sense == "target" and self.target is None:
            raise ValueError(f"objective {self.name!r}: a target objective needs a target value")
        if self.weight <= 0:
            raise ValueError(f"objective {self.name!r}: weight must be positive")


@dataclass
class Constraint:
    """A limit the answer has to satisfy: a bound on a quantity, or an expression over parameters (#151, #221).

    ``kind="bound"`` uses ``quantity``/``op``/``value``; ``kind="expression"`` uses ``expression`` (free text, the
    same form the elicited description and the design-of-experiments ``ParameterSpace`` already accept).

    ``family`` says **what kind of limit** it is -- ``"physical"`` (a physical limit: a temperature or a yield
    stress), ``"conservation"`` (a conserved total), ``"operational"`` (a derating or an operating envelope),
    ``"material"``, ``"actuator"`` (a stroke or a rate) or ``"safety"`` -- and ``""`` leaves it unlabelled for the
    constraints that predate the vocabulary. The family does not change how the limit is *checked*: that is the
    point of #221, one object and one arithmetic applied in training, in an optimizer and in a deployment guard
    (:mod:`pinneapple_physics.constraint_engine`). It is the label the report groups by, the optimizer names its
    constraint with, and the guard puts in its refusal.

    ``weight`` scales the penalty the constraint contributes during training, and ``hard`` says whether a violation
    is fatal: a hard constraint stops a deployment and rejects a design, a soft one is reported and penalised but
    lets the answer through.
    """
    name: str
    kind: str = "bound"              # bound | expression
    quantity: str = ""
    op: str = "<="                   # <= | >= | ==
    value: Optional[float] = None
    expression: str = ""
    unit: Optional[str] = None
    description: str = ""
    # -- #221: what kind of limit this is, and how it is paid for and enforced -------------------------------
    family: str = ""                 # physical | conservation | operational | material | actuator | safety | ""
    weight: float = 1.0              # penalty coefficient during training
    hard: bool = True                # a violation is fatal for a deployment / rejects a design

    def __post_init__(self) -> None:
        if self.kind not in ("bound", "expression"):
            raise ValueError(f"constraint {self.name!r}: kind must be bound or expression, got {self.kind!r}")
        if self.family not in CONSTRAINT_FAMILIES:
            raise ValueError(f"constraint {self.name!r}: family must be one of {CONSTRAINT_FAMILIES} "
                             f"(or '' for unlabelled), got {self.family!r}")
        if not float(self.weight) > 0:
            raise ValueError(f"constraint {self.name!r}: weight must be positive, got {self.weight}")
        if self.kind == "bound":
            if self.op not in ("<=", ">=", "=="):
                raise ValueError(f"constraint {self.name!r}: op must be <=, >= or ==, got {self.op!r}")
            if self.value is None:
                raise ValueError(f"constraint {self.name!r}: a bound needs a value")
        elif not self.expression.strip():
            raise ValueError(f"constraint {self.name!r}: an expression constraint needs an expression")

    def text(self) -> str:
        """The constraint as one line, for the elicited description and for a report."""
        if self.kind == "expression":
            return self.expression
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.quantity} {self.op} {self.value}{unit}"


@dataclass
class Fidelity:
    """How much physics the answer has to have, and how good it has to be (#151).

    ``level`` is the physics: ``"closed_form"``, ``"linear"``, ``"rans"``, ``"les"``, ``"dns"``, ``"surrogate"`` or
    ``"unspecified"``. ``mesh_level`` names a mesh (``"L1"``, ``"coarse"``) when the answer depends on one, and
    ``tolerance`` is the relative error the result has to meet against its reference.
    """
    level: str = "unspecified"
    mesh_level: str = ""
    tolerance: Optional[float] = None
    notes: str = ""

    def describe(self) -> str:
        parts = [self.level] if self.level else []
        if self.mesh_level:
            parts.append(f"mesh {self.mesh_level}")
        if self.tolerance is not None:
            parts.append(f"within {self.tolerance:g} relative error")
        if self.notes:
            parts.append(self.notes)
        return ", ".join(parts) or "unspecified"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _facade_dict(problem: "PhysicalProblem") -> Dict[str, Any]:
    """The façade fields (#151), only when set: an empty tuple here would move the fingerprint of every preset
    that predates them, and a fingerprint is how a stored result names its problem."""
    out: Dict[str, Any] = {}
    if problem.objectives:
        out["objectives"] = [dataclasses.asdict(o) for o in problem.objectives]
    if problem.constraints:
        out["constraints"] = [{"text": c.text(), **dataclasses.asdict(c)} for c in problem.constraints]
    if problem.fidelity is not None:
        out["fidelity"] = dataclasses.asdict(problem.fidelity)
    if problem.geometry_ref:
        out["geometry_ref"] = problem.geometry_ref
    return out


def _pde_meta_with_facade(problem: "PhysicalProblem", saved: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``meta`` of the exported spec: what it already had, plus the façade (#151) when the problem carries one."""
    meta = dict(saved.get("meta", {}) or {})
    facade = _facade_dict(problem)
    if facade:
        meta["facade"] = facade
    return meta


def _facade_from(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Read back what :func:`_facade_dict` wrote (missing keys mean "not set", as in the source problem)."""
    objectives = tuple(Objective(**o) for o in d.get("objectives") or ())
    constraints = tuple(Constraint(**{k: v for k, v in c.items() if k != "text"})
                        for c in d.get("constraints") or ())
    fidelity = Fidelity(**d["fidelity"]) if d.get("fidelity") else None
    return {"objectives": objectives, "constraints": constraints, "fidelity": fidelity,
            "geometry_ref": d.get("geometry_ref", "")}


# --------------------------------------------------------------------------- the problem
@dataclass
class PhysicalProblem:
    """A complete physics problem: geometry, physics, conditions, parameters, solver choice and intent.

    Only ``name`` is required. A problem can be described before its geometry exists, before the PDE is
    known (for example right after the problem-design conversation) or before it has been solved.

    Attributes
    ----------
    coords, fields : coordinate and output-field names, e.g. ``("x", "t")`` and ``("u",)``.
    pde : the governing equation as a ``PDETermSpec`` (``kind`` + ``params``), or ``None``.
    conditions : boundary, initial, interface and data constraints (``ConditionSpec``).
    domain_bounds : axis-aligned bounds per coordinate, e.g. ``{"x": (-1, 1), "t": (0, 1)}``.
    geometry : a geometry object (mesh, CAD, voxel grid, SDF...) when the domain is not a box.
    parameters : ``Parameter`` objects keyed by name. ``pde.params`` values are kept in sync on export.
    units : unit per coordinate or field name, e.g. ``{"x": "m", "u": "m/s"}``.
    quantities : quantities of interest.
    solver, solver_config : the solver/backend chosen for this problem and its options.
    reference_solver : how reference solutions for this problem are produced (a preset's ``solver_spec``).
    task : what is being asked: ``"forward"``, ``"inverse"``, ``"design"``, ``"uq"``, ``"surrogate"`` or free text.
    objectives : ``Objective`` objects -- what the answer is for (minimise drag, hold peak temperature under X).
    constraints : ``Constraint`` objects -- the limits the answer has to satisfy, as bounds or expressions.
    fidelity : how much physics and how good the answer has to be (``Fidelity``), or ``None``.
    geometry_ref : a path or URI to the CAD/mesh the domain came from, when there is one.
    intent : free-text context (goal, data, validation, deployment) carried over from problem design.
    references, metadata : citations and free-form provenance.
    """
    name: str
    coords: Tuple[str, ...] = ()
    fields: Tuple[str, ...] = ()
    pde: Optional[PDETermSpec] = None
    conditions: Tuple[ConditionSpec, ...] = ()
    domain_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    geometry: Any = None
    geometry_kind: Optional[str] = None
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    quantities: Tuple[Quantity, ...] = ()
    field_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scales: ScaleSpec = field(default_factory=ScaleSpec)
    sample_defaults: Dict[str, int] = field(default_factory=dict)
    solver: Optional[str] = None
    solver_config: Dict[str, Any] = field(default_factory=dict)
    reference_solver: Dict[str, Any] = field(default_factory=dict)
    task: str = "forward"
    # -- the façade (#151): what the problem is for, what limits it, and how much physics it needs ------------
    objectives: Tuple[Objective, ...] = ()
    constraints: Tuple[Constraint, ...] = ()
    fidelity: Optional[Fidelity] = None
    geometry_ref: str = ""            # a path or URI to the CAD/mesh the domain came from
    intent: Dict[str, Any] = field(default_factory=dict)
    references: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ convenience
    @property
    def dim(self) -> int:
        return len(self.coords)

    def parameter_values(self) -> Dict[str, Any]:
        """Current value of every parameter that has one (what the PDE residual needs)."""
        return {k: p.value for k, p in self.parameters.items() if p.value is not None}

    def parameters_by_role(self, role: str) -> Dict[str, Parameter]:
        if role not in PARAMETER_ROLES:
            raise ValueError(f"role must be one of {PARAMETER_ROLES}, got {role!r}")
        return {k: p for k, p in self.parameters.items() if p.role == role}

    def with_parameters(self, **values: float) -> "PhysicalProblem":
        """A copy with new parameter values (unknown names raise, so typos do not pass silently).

        The copy in ``pde.params`` moves with them: a stale entry there would make ``to_pde_spec()`` -- which
        writes the current values -- describe a different problem from the one that was just built, so
        ``from_pde_spec(to_pde_spec(problem))`` came back with a different fingerprint and anything reading
        ``pde.params`` directly saw the value the preset shipped instead of the override.
        """
        missing = sorted(set(values) - set(self.parameters))
        if missing:
            raise KeyError(f"unknown parameter(s) {missing}; known: {sorted(self.parameters)}")
        params = dict(self.parameters)
        pde = self.pde
        pde_params = None if pde is None else dict(pde.params)
        for k, v in values.items():
            params[k] = replace(params[k], value=v)
            if pde_params is not None and k in pde_params and _is_number(pde_params[k]):
                pde_params[k] = v
        if pde is not None and pde_params is not None and pde_params != dict(pde.params):
            pde = replace(pde, params=pde_params)
        return replace(self, parameters=params, pde=pde)

    def set_role(self, name: str, role: str, *, bounds: Optional[Tuple[float, float]] = None,
                 distribution: Optional[Dict[str, Any]] = None) -> "PhysicalProblem":
        """A copy where parameter ``name`` has a new role, e.g. ``set_role("nu", "unknown")`` for discovery."""
        if name not in self.parameters:
            raise KeyError(f"unknown parameter {name!r}; known: {sorted(self.parameters)}")
        if role not in PARAMETER_ROLES:
            raise ValueError(f"role must be one of {PARAMETER_ROLES}, got {role!r}")
        p = self.parameters[name]
        p = replace(p, role=role, bounds=bounds if bounds is not None else p.bounds,
                    distribution=distribution if distribution is not None else p.distribution)
        return replace(self, parameters={**self.parameters, name: p})

    # ------------------------------------------------------------------ training backends (#75)
    def backend(self, backend: str = "pytorch", **options: Any) -> Any:
        """The trainer for this problem on ``backend``: ``"pytorch"``, ``"jax"`` or ``"physicsnemo"``.

        The problem is the description; this is the engine that fits it, and the same problem object drives all
        three::

            problem = pp.PhysicalProblem.from_preset("heat_1d")
            jax_run = problem.backend("jax", ...)      # a ProblemBackend; options go to its constructor
            report = jax_run.fit(epochs=300, seed=0)   # a FitReport
            y = jax_run.predict(X)                     # a Prediction, (N, len(fields))
            jax_run.save("heat_1d.npz")                # reloaded by pinneapple_neural.model_provider.load

        Training options (``epochs``, ``n_collocation``, ``lr``, ``seed``, ``hidden_dim``, ``n_layers``) belong to
        ``fit``; ``options`` here are the backend's own constructor arguments (``architecture=`` on pytorch). A
        backend whose package is missing, and a PDE kind or condition the backend does not build a residual for, are
        refused by name -- see :mod:`pinneapple_neural.problem_backends`.
        """
        from pinneapple_neural.problem_backends import problem_backend

        return problem_backend(self, backend, **options)

    # ------------------------------------------------------------------ units (#110)
    def unit_names(self) -> list[str]:
        """Every name in the problem that can carry a unit: coordinates, fields and parameters."""
        return [*self.coords, *self.fields, *self.parameters]

    def missing_units(self) -> list[str]:
        """The names above that declare no unit — what ``validate(require_units=True)`` refuses."""
        declared = {k for k, u in self.units.items() if u not in (None, "")}
        declared |= {k for k, p in self.parameters.items() if p.unit not in (None, "")}
        return [n for n in self.unit_names() if n not in declared]

    def to_unit_system(self, targets: Mapping[str, str]) -> PhysicalProblem:
        """A copy with ``targets`` applied: every number attached to a name converts with its declared unit.

        The numbers a problem carries are the bounds of its coordinates, the ranges of its fields and its parameter
        values: those convert. A name that declares no unit is **refused** (there is nothing to convert from) and so
        is a name the problem does not have (a typo would otherwise be a silent no-op), because a conversion that
        quietly skipped a value would leave a problem half in each system.

        The fingerprint changes when the units do, because it identifies the *description*: a problem in mm and the
        same problem in m are the same physics with two different records, and a record that claimed otherwise would
        match a run it does not describe.
        """
        from pinneapple_data.physical_units import UnitError, convert, try_parse_unit

        known = set(self.unit_names())
        unknown = sorted(set(targets) - known)
        if unknown:
            raise KeyError(f"unknown name(s) {unknown} to convert; the problem has {sorted(known)}")
        units = dict(self.units)
        bounds = dict(self.domain_bounds)
        ranges = dict(self.field_ranges)
        params = dict(self.parameters)
        for name, target in targets.items():
            source = self.units.get(name) or getattr(self.parameters.get(name), "unit", None)
            if source in (None, ""):
                raise ValueError(f"'{name}' declares no unit, so it cannot be converted to {target!r}: a conversion "
                                 "from an unknown unit is a guess")
            if try_parse_unit(target) is None:
                raise ValueError(f"target unit {target!r} for '{name}' is not one this library parses")
            try:
                if name in bounds:
                    lo, hi = bounds[name]
                    bounds[name] = (float(convert(lo, source, target)), float(convert(hi, source, target)))
                if name in ranges:
                    lo, hi = ranges[name]
                    ranges[name] = (float(convert(lo, source, target)), float(convert(hi, source, target)))
                if name in params:
                    p = params[name]
                    params[name] = replace(
                        p, value=None if p.value is None else float(convert(p.value, source, target)),
                        bounds=None if not p.bounds else (float(convert(p.bounds[0], source, target)),
                                                          float(convert(p.bounds[1], source, target))),
                        unit=target)
            except UnitError as exc:
                raise ValueError(f"cannot convert '{name}' from {source!r} to {target!r}: {exc}") from None
            units[name] = target
        converted = replace(self, units=units, domain_bounds=bounds, field_ranges=ranges, parameters=params)
        numbers = {k: v.value for k, v in params.items() if v.value is not None and k in targets}
        return converted.with_parameters(**numbers) if numbers else converted

    # ------------------------------------------------------------------ checks
    def dimensions(self, equation: str, **kwargs: Any) -> Any:
        """Check one equation against the units this problem declares; see :mod:`pinneapple_physics.dimensions`."""
        from .dimensions import check_dimensions

        return check_dimensions(equation, problem=self, **kwargs)

    def validate(self, *, require_units: bool = False,
                 equations: "str | Sequence[str] | None" = None) -> List[str]:
        """Inconsistencies as readable strings; an empty list means none were found.

        ``require_units=True`` adds "a field without a unit" to the list: the problem refuses to be called complete
        when a value has no unit, because nothing downstream can check a number whose unit is unknown.

        ``equations`` adds the dimensional check (#205): each equation is read term by term against the problem's
        units, so a source term in W/m^3 next to a time derivative in K/s is reported here instead of becoming an
        error floor after training."""
        out: List[str] = []
        if len(set(self.coords)) != len(self.coords):
            out.append(f"duplicate coordinate names in {self.coords}")
        if len(set(self.fields)) != len(self.fields):
            out.append(f"duplicate field names in {self.fields}")
        if self.pde is not None:
            for f in self.pde.fields:
                if self.fields and f not in self.fields:
                    out.append(f"PDE field '{f}' is not one of the problem fields {self.fields}")
            for c in self.pde.coords:
                if self.coords and c not in self.coords:
                    out.append(f"PDE coordinate '{c}' is not one of the problem coordinates {self.coords}")
        for c in self.domain_bounds:
            if self.coords and c not in self.coords:
                out.append(f"domain bound given for unknown coordinate '{c}'")
            lo, hi = self.domain_bounds[c]
            if not lo < hi:
                out.append(f"domain bound for '{c}' is empty or reversed: {self.domain_bounds[c]}")
        for cond in self.conditions:
            # Same rule as the PINN compiler: a condition may name non-output quantities (tractions "tx",
            # a heat flux "q_heat", convection "h"/"T_ref") only as a first-order Neumann condition that says
            # how to resolve them (traction_map, normal_stress_field or thermal_bc); otherwise compiling
            # the problem raises a KeyError.
            special = cond.kind == "neumann" and cond.order <= 1 and bool(
                cond.traction_map or cond.normal_stress_field or cond.thermal_bc)
            unknown = [f for f in cond.fields if self.fields and f not in self.fields]
            if unknown and not special:
                out.append(f"condition '{cond.name}' refers to {unknown}, which are not model fields "
                           f"{self.fields}, and has no traction_map/normal_stress_field/thermal_bc to resolve them")
            if cond.deriv_coord and self.coords and cond.deriv_coord not in self.coords:
                out.append(f"condition '{cond.name}' differentiates along unknown coordinate '{cond.deriv_coord}'")
        for name, p in self.parameters.items():
            if p.name != name:
                out.append(f"parameter stored under '{name}' is named '{p.name}'")
            out.extend(p.issues())
        if self.units:
            from pinneapple_data.physical_units import try_parse_unit
            for k, u in self.units.items():
                if self.coords and self.fields and k not in self.coords and k not in self.fields:
                    out.append(f"unit given for '{k}', which is neither a coordinate nor a field")
                if u not in ("", "1") and try_parse_unit(u) is None:
                    out.append(f"unit '{u}' for '{k}' is not recognised")
        for q in self.quantities:
            if q.field is not None and self.fields and q.field not in self.fields:
                out.append(f"quantity of interest '{q.name}' refers to unknown field '{q.field}'")
        if equations:
            for equation in ([equations] if isinstance(equations, str) else list(equations)):
                report = self.dimensions(equation)
                out.extend(f"dimensionally inconsistent: {problem}" for problem in report.problems)
        if require_units:
            missing = self.missing_units()
            if missing:
                out.append(f"units are required and missing for {missing}: a value without a unit cannot be checked "
                           "against anything")
        return out

    # ------------------------------------------------------------------ provenance
    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly description. Callables in conditions and the geometry object are described, not
        inlined (persist them with their own storage); everything else round-trips through ``from_dict``."""
        def cond(c: ConditionSpec) -> Dict[str, Any]:
            d = {}
            for f in dataclasses.fields(c):
                v = getattr(c, f.name)
                d[f.name] = _describe_callable(v) if callable(v) else (list(v) if isinstance(v, tuple) else v)
            return d

        return {
            "name": self.name,
            "coords": list(self.coords),
            "fields": list(self.fields),
            "pde": None if self.pde is None else {"kind": self.pde.kind, "fields": list(self.pde.fields),
                                                  "coords": list(self.pde.coords), "params": dict(self.pde.params),
                                                  "meta": dict(self.pde.meta)},
            "conditions": [cond(c) for c in self.conditions],
            "domain_bounds": {k: list(v) for k, v in self.domain_bounds.items()},
            "geometry": _describe_object(self.geometry),
            "geometry_kind": self.geometry_kind,
            "parameters": {k: {**dataclasses.asdict(p), "bounds": list(p.bounds) if p.bounds else None}
                           for k, p in self.parameters.items()},
            "units": dict(self.units),
            "quantities": [dataclasses.asdict(q) for q in self.quantities],
            "field_ranges": {k: list(v) for k, v in self.field_ranges.items()},
            "scales": dataclasses.asdict(self.scales),
            "sample_defaults": dict(self.sample_defaults),
            "solver": self.solver,
            "solver_config": dict(self.solver_config),
            "reference_solver": dict(self.reference_solver),
            "task": self.task,
            "intent": dict(self.intent),
            **_facade_dict(self),
            "references": list(self.references),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PhysicalProblem":
        """Rebuild from ``to_dict()``. A condition whose selector or value function was a Python callable comes
        back with a placeholder (``to_dict`` only stored its name), which raises a clear error if it is called;
        re-attach the real function (``dataclasses.replace(cond, value_fn=...)``) before compiling."""
        pde_d = d.get("pde")
        pde = None if not pde_d else PDETermSpec(kind=pde_d["kind"], fields=tuple(pde_d["fields"]),
                                                 coords=tuple(pde_d["coords"]), params=dict(pde_d.get("params") or {}),
                                                 meta=dict(pde_d.get("meta") or {}))
        conds = []
        for c in d.get("conditions") or ():
            c = dict(c)
            for k in ("selector", "value_fn"):
                if isinstance(c.get(k), dict) and c[k].get("callable"):
                    # Not None: a missing value_fn would silently mean "no prescribed value". The placeholder
                    # keeps the description (so the fingerprint is unchanged) and fails loudly if used.
                    c[k] = _DetachedCallable(c[k], condition=c.get("name", "?"), slot=k)
            c["fields"] = tuple(c.get("fields") or ())
            conds.append(ConditionSpec(**c))
        params = {}
        for k, p in (d.get("parameters") or {}).items():
            p = dict(p)
            p["bounds"] = tuple(p["bounds"]) if p.get("bounds") else None
            params[k] = Parameter(**p)
        return cls(
            name=d["name"], coords=tuple(d.get("coords") or ()), fields=tuple(d.get("fields") or ()), pde=pde,
            conditions=tuple(conds), domain_bounds={k: tuple(v) for k, v in (d.get("domain_bounds") or {}).items()},
            geometry=None, geometry_kind=d.get("geometry_kind"), parameters=params, units=dict(d.get("units") or {}),
            quantities=tuple(Quantity(**q) for q in d.get("quantities") or ()),
            field_ranges={k: tuple(v) for k, v in (d.get("field_ranges") or {}).items()},
            scales=ScaleSpec(**(d.get("scales") or {})), sample_defaults=dict(d.get("sample_defaults") or {}),
            solver=d.get("solver"), solver_config=dict(d.get("solver_config") or {}),
            reference_solver=dict(d.get("reference_solver") or {}), task=d.get("task", "forward"),
            **_facade_from(d),
            intent=dict(d.get("intent") or {}), references=tuple(d.get("references") or ()),
            metadata=dict(d.get("metadata") or {}),
        )

    def fingerprint(self) -> str:
        """SHA-256 of the canonical description (excluding ``metadata``). Two problems with the same physics,
        conditions, parameters and solver choice share a fingerprint; use it to tie results to their problem."""
        d = self.to_dict()
        d.pop("metadata", None)
        blob = json.dumps(d, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def summary(self) -> str:
        lines = [f"PhysicalProblem '{self.name}' ({self.task})"]
        if self.pde is not None:
            lines.append(f"  PDE: {self.pde.kind} on {', '.join(self.coords)} -> {', '.join(self.fields)}")
        if self.domain_bounds:
            lines.append("  domain: " + ", ".join(f"{k} in [{a:g}, {b:g}]" for k, (a, b) in self.domain_bounds.items()))
        if self.geometry is not None:
            lines.append(f"  geometry: {type(self.geometry).__name__}" + (f" ({self.geometry_kind})" if self.geometry_kind else ""))
        if self.conditions:
            kinds: Dict[str, int] = {}
            for c in self.conditions:
                kinds[c.kind] = kinds.get(c.kind, 0) + 1
            lines.append("  conditions: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
        for p in self.parameters.values():
            u = f" {p.unit}" if p.unit else ""
            extra = f" in {list(p.bounds)}" if p.bounds else ""
            lines.append(f"  {p.role:<9} {p.name} = {p.value}{u}{extra}")
        for q in self.quantities:
            lines.append(f"  output: {q.name}" + (f" [{q.unit}]" if q.unit else ""))
        if self.solver:
            lines.append(f"  solver: {self.solver}")
        if self.reference_solver:
            lines.append(f"  reference solver: {self.reference_solver.get('name', '?')}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # dataclass repr would dump geometry and callables
        return f"PhysicalProblem(name={self.name!r}, task={self.task!r}, pde={getattr(self.pde, 'kind', None)!r}, " \
               f"coords={self.coords}, fields={self.fields}, parameters={sorted(self.parameters)})"

    # ------------------------------------------------------------------ adapter: PDE ProblemSpec (lossless)
    @classmethod
    def from_pde_spec(cls, spec: ProblemSpec, *, units: Optional[Mapping[str, str]] = None,
                      parameter_units: Optional[Mapping[str, str]] = None) -> "PhysicalProblem":
        """From the PDE ``ProblemSpec`` (presets, ``ProblemBuilder``). Lossless: ``to_pde_spec()`` returns an
        equal spec. ``pde.params`` become ``fixed`` parameters (units may be given via ``parameter_units``)."""
        pu = dict(parameter_units or {})
        # Only numeric PDE parameters become Parameters; others (mode strings, arrays, callables) stay in
        # pde.params untouched and are exported exactly as they were.
        params = {k: Parameter(name=k, value=v, unit=pu.get(k)) for k, v in spec.pde.params.items() if _is_number(v)}
        meta = dict(spec.meta)
        return cls(
            name=spec.name, coords=tuple(spec.coords), fields=tuple(spec.fields), pde=spec.pde,
            conditions=tuple(spec.conditions), domain_bounds=dict(spec.domain_bounds), parameters=params,
            units=dict(units or {}), field_ranges=dict(spec.field_ranges), scales=spec.scales,
            sample_defaults=dict(spec.sample_defaults), reference_solver=dict(spec.solver_spec),
            references=tuple(spec.references),
            metadata={"_pde_spec": {"name": spec.name, "dim": spec.dim, "problem_id": spec.problem_id, "meta": meta}},
            **_facade_from(dict(spec.meta.get("facade") or {})),
        )

    def to_pde_spec(self) -> ProblemSpec:
        """The PDE ``ProblemSpec`` the PINN compiler and the solvers consume. Current parameter values are
        written into ``pde.params`` (non-numeric entries already there are kept)."""
        if self.pde is None:
            raise ValueError(f"PhysicalProblem '{self.name}' has no PDE; set .pde before exporting a PDE spec")
        pde_params = dict(self.pde.params)
        for k, p in self.parameters.items():
            if p.value is not None:
                pde_params[k] = p.value
        pde = self.pde if pde_params == dict(self.pde.params) else replace(self.pde, params=pde_params)
        saved = self.metadata.get("_pde_spec", {})
        dim = saved.get("dim", len(self.coords))
        return ProblemSpec(
            name=saved.get("name", self.name), dim=dim, coords=tuple(self.coords), fields=tuple(self.fields), pde=pde,
            conditions=tuple(self.conditions), sample_defaults=dict(self.sample_defaults), scales=self.scales,
            field_ranges=dict(self.field_ranges), references=tuple(self.references),
            domain_bounds=dict(self.domain_bounds), solver_spec=dict(self.reference_solver),
            meta=_pde_meta_with_facade(self, saved), problem_id=saved.get("problem_id", ""),
        )

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "PhysicalProblem":
        """From a named preset (``pp.list_presets()``); keyword overrides go to the preset, e.g. ``nu=0.01``."""
        from .pde_environment import get_preset
        return cls.from_pde_spec(get_preset(name, **overrides))

    # ------------------------------------------------------------------ adapter: PhysicsCase (lossless)
    @classmethod
    def from_physics_case(cls, case: Any) -> "PhysicalProblem":
        """From ``pinneapple_data.physics_case.PhysicsCase``. Results and the reference benchmark are kept in
        ``metadata`` (a problem describes the question; results belong to a run)."""
        base = cls.from_pde_spec(case.physics) if case.physics is not None else cls(name=case.name)
        md = dict(base.metadata)
        md.update(dict(case.metadata))
        md["_physics_case"] = {"name": case.name, "results": case.results,
                               "reference_benchmark": case.reference_benchmark,
                               "has_physics": case.physics is not None}
        return replace(base, name=case.name or base.name, geometry=case.geometry, geometry_kind=case.geometry_kind,
                       solver=case.solver if case.solver is not None else base.solver,
                       solver_config=dict(case.solver_config) if case.solver_config else base.solver_config,
                       metadata=md)

    def to_physics_case(self, *, results: Any = None, reference_benchmark: Optional[str] = None):
        """A ``PhysicsCase`` referencing this problem's PDE spec, geometry and solver."""
        from pinneapple_data.physics_case import PhysicsCase
        saved = self.metadata.get("_physics_case", {})
        md = {k: v for k, v in self.metadata.items() if not k.startswith("_")}
        physics = self.to_pde_spec() if self.pde is not None and saved.get("has_physics", True) else None
        return PhysicsCase(
            name=saved.get("name", self.name), geometry=self.geometry, geometry_kind=self.geometry_kind,
            physics=physics, solver=self.solver, solver_config=dict(self.solver_config) or None,
            results=results if results is not None else saved.get("results"),
            reference_benchmark=reference_benchmark or saved.get("reference_benchmark"), metadata=md,
        )

    # ------------------------------------------------------------------ adapter: problem-design ProblemSpec
    _TASK_FROM_DESIGN = {"pde_solution": "forward", "inverse_problem": "inverse", "optimization": "design",
                         "neural_operator": "surrogate", "forecasting": "forecasting", "control": "control",
                         "anomaly_detection": "anomaly_detection", "other": "other"}

    @classmethod
    def from_problem_design(cls, spec: Any) -> "PhysicalProblem":
        """From the elicited ``pinneapple_problemdesign.schema.ProblemSpec``. Free-text equations and conditions
        cannot be compiled, so they go to ``intent``; outputs become fields and quantities of interest, known and
        unknown parameter names become ``fixed`` (value pending) and ``unknown`` parameters, and units carry over.
        Attach a ``pde`` (for example from a preset) before solving."""
        d = spec.to_dict() if hasattr(spec, "to_dict") else dataclasses.asdict(spec)
        phys = d.get("physics") or {}
        units = dict(phys.get("units") or {})
        params: Dict[str, Parameter] = {}
        for n in phys.get("parameters_known") or ():
            params[n] = Parameter(name=n, unit=units.get(n), role="fixed",
                                  description="known; value not elicited yet")
        for n in phys.get("parameters_unknown") or ():
            params[n] = Parameter(name=n, unit=units.get(n), role="unknown", description="to be identified")
        outputs = tuple(d.get("outputs") or ())
        geom = d.get("geometry") or {}
        intent = {k: d.get(k) for k in ("goal", "inputs", "horizon", "input_window", "frequency", "domain_context",
                                        "data", "validation", "deployment", "constraints", "assumptions", "risks")}
        intent["task_type"] = d.get("task_type")
        intent["physics_text"] = {k: phys.get(k) for k in ("governing_equations", "boundary_conditions",
                                                           "initial_conditions", "constraints",
                                                           "turbulence_model", "numerical_method")}
        intent["geometry_text"] = geom
        return cls(
            name=d.get("title") or "untitled", fields=outputs, parameters=params,
            units={k: v for k, v in units.items() if k not in params},
            quantities=tuple(Quantity(name=o, unit=units.get(o), field=o) for o in outputs),
            solver=phys.get("numerical_method"), task=cls._TASK_FROM_DESIGN.get(d.get("task_type"), "other"),
            intent=intent, geometry_kind=geom.get("representation") or None,
        )

    def to_problem_design(self):
        """A problem-design ``ProblemSpec`` describing this problem (for the assistant and its reports)."""
        from pinneapple_problemdesign.schema import (ConstraintsSpec, DataSpec, DeploymentSpec, GeometrySpec,
                                                     PhysicsSpec, ProblemSpec as DesignSpec, ValidationSpec)
        inv = {v: k for k, v in self._TASK_FROM_DESIGN.items()}
        it = self.intent
        ptxt = dict(it.get("physics_text") or {})
        units = dict(self.units)
        units.update({k: p.unit for k, p in self.parameters.items() if p.unit})
        eqs = list(ptxt.get("governing_equations") or ([self.pde.kind] if self.pde else []))

        def sub(cls_, key):
            raw = it.get(key) or {}
            names = {f.name for f in dataclasses.fields(cls_)}
            return cls_(**{k: v for k, v in raw.items() if k in names}) if isinstance(raw, dict) else cls_()

        gtxt = dict(it.get("geometry_text") or {})
        geometry = GeometrySpec(**{k: v for k, v in gtxt.items() if k in {f.name for f in dataclasses.fields(GeometrySpec)}})
        if not geometry.domain and self.geometry_ref:
            geometry = replace(geometry, domain=self.geometry_ref)      # where the geometry came from, in prose
        if not geometry.representation and self.fidelity is not None and self.fidelity.level:
            geometry = replace(geometry, representation=self.fidelity.describe())
        return DesignSpec(
            title=self.name, goal=it.get("goal") or "", task_type=it.get("task_type") or inv.get(self.task, "other"),
            inputs=list(it.get("inputs") or self.coords), outputs=[q.name for q in self.quantities] or list(self.fields),
            horizon=it.get("horizon") or "", input_window=it.get("input_window") or "",
            frequency=it.get("frequency") or "", domain_context=it.get("domain_context") or "",
            data=sub(DataSpec, "data"),
            physics=PhysicsSpec(
                governing_equations=eqs,
                boundary_conditions=list(ptxt.get("boundary_conditions") or [c.name for c in self.conditions
                                                                             if c.kind != "initial"]),
                initial_conditions=list(ptxt.get("initial_conditions") or [c.name for c in self.conditions
                                                                           if c.kind == "initial"]),
                constraints=[c.text() for c in self.constraints] or list(ptxt.get("constraints") or []),
                parameters_known=[k for k, p in self.parameters.items() if p.role != "unknown"],
                parameters_unknown=[k for k, p in self.parameters.items() if p.role == "unknown"],
                units=units, turbulence_model=ptxt.get("turbulence_model"),
                numerical_method=ptxt.get("numerical_method") or self.solver),
            geometry=geometry, validation=sub(ValidationSpec, "validation"),
            deployment=sub(DeploymentSpec, "deployment"), constraints=sub(ConstraintsSpec, "constraints"),
        )


class _DetachedCallable:
    """Stands in for a condition callable that was not serialized. Calling it raises instead of guessing."""

    def __init__(self, desc: Dict[str, Any], *, condition: str, slot: str):
        self.__qualname__ = desc.get("name") or "callable"
        self.__module__ = desc.get("module")
        self._where = f"condition '{condition}' ({slot})"

    def __call__(self, *args: Any, **kwargs: Any):
        raise RuntimeError(f"{self._where} was loaded from a dict without its Python function "
                           f"'{self.__qualname__}'; re-attach it before compiling or solving")

    def __repr__(self) -> str:
        return f"<detached {self.__qualname__} for {self._where}>"


def _describe_callable(fn: Any) -> Dict[str, Any]:
    return {"callable": True, "name": getattr(fn, "__qualname__", type(fn).__name__),
            "module": getattr(fn, "__module__", None)}


def _describe_object(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    desc: Dict[str, Any] = {"type": type(obj).__name__}
    for attr in ("summary", "domain_type"):
        f = getattr(obj, attr, None)
        if callable(f):
            try:
                desc[attr] = f()
            except Exception:  # noqa: BLE001 - a description must never fail
                pass
    return desc
