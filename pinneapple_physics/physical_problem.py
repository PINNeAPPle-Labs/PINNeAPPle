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
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .pde_environment.conditions import ConditionSpec
from .pde_environment.scales import ScaleSpec
from .pde_environment.spec import PDETermSpec, ProblemSpec

__all__ = ["Parameter", "Quantity", "PhysicalProblem", "PARAMETER_ROLES"]

PARAMETER_ROLES = ("fixed", "design", "uncertain", "unknown")


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


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


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
        """A copy with new parameter values (unknown names raise, so typos do not pass silently)."""
        missing = sorted(set(values) - set(self.parameters))
        if missing:
            raise KeyError(f"unknown parameter(s) {missing}; known: {sorted(self.parameters)}")
        params = dict(self.parameters)
        for k, v in values.items():
            params[k] = replace(params[k], value=v)
        return replace(self, parameters=params)

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

    # ------------------------------------------------------------------ checks
    def validate(self) -> List[str]:
        """Inconsistencies as readable strings; an empty list means none were found."""
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
            meta=dict(saved.get("meta", {})), problem_id=saved.get("problem_id", ""),
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
                constraints=list(ptxt.get("constraints") or []),
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
