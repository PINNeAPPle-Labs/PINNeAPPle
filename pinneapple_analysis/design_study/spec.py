"""The design study specification, schema ``pinneapple-design-study/v1``: one serializable description of a Physics AI
study, whatever produces the numbers.

A study says six things, and nothing else:

* **what kind of study** it is (:data:`KINDS`): a design optimisation, a surrogate, a PINN or operator fit, an inverse
  identification or a digital twin;
* **what is varied**: the base geometry or system (a library entry, the user's STL or a system model) and its
  variables with bounds;
* **who evaluates it**: a tool of the catalog (any id), the mapping from each study variable to that tool's own input
  name, the fixed operating conditions, and the outputs *each with its unit*;
* **what to achieve**: objectives and limits, absolute or relative to the baseline;
* **how much may be spent**: reference evaluations, DOE points, verifications, optional wall clock and the seed;
* **what counts as verified**: the verifier's tolerances.

Why a schema of its own instead of the library's classes
-------------------------------------------------------
The spec is a *wire format*: the Studio wizard, the Engineer and a saved file all speak it, and a stored study has to
stay readable after the library's internal classes move. It is therefore frozen at ``v1`` and validated here, and the
conversions to the classes that do the work (``pinneapple_design.design_optimizer.study.Variable`` / ``Constraint``,
``pinneapple_autolab.mission.Limit``) live in :mod:`.study`, which imports them lazily. This module imports nothing but
the standard library, so a wizard can refuse a broken study without loading the optimiser, the surrogate board or the
lab (the parent ``pinneapple_analysis`` package does import torch for its own reasons; nothing here adds to that).

What it refuses, instead of assuming
------------------------------------
* an output with no unit (``""``, ``"?"``, ``"unknown"``, ``"n/a"``): a drag in "?" is not a number anyone can
  check. A dimensionless output says ``"1"`` (or ``"-"``);
* a variable with no unit;
* a variable the evaluator's input mapping does not cover, and a mapping entry for a variable that does not exist;
* an objective or a limit on an output the evaluator does not declare;
* a limit that bounds nothing, or that is relative to the baseline without the baseline's output;
* an STL base without a parametrization (the morphing has nothing to move);
* a budget too small to hold a baseline, a DOE and one verification;
* an unknown field anywhere: a typo in a spec is a bug in whoever wrote it, not something to drop silently.

    >>> from pinneapple_analysis.design_study import DesignStudy
    >>> spec = DesignStudy.create(kind="optimize", title="Lower drag", base=..., evaluator=...)
    >>> DesignStudy.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    True
    >>> DesignStudy.from_dict(spec.to_dict() | {"schema": "pinneapple-design-study/v2"})
    Traceback (most recent call last):
    ...
    ValueError: schema 'pinneapple-design-study/v2' is not supported; this library reads 'pinneapple-design-study/v1'
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

__all__ = [
    "SCHEMA", "KINDS", "KINDS_BEYOND_OPTIMISATION", "BASE_KINDS", "FIDELITIES", "SURROGATE_FAMILIES", "SENSES",
    "Quantity", "Variable", "Base", "Evaluator", "Objective", "Limit", "Budget", "Verifier", "Model", "DesignStudy",
]

SCHEMA = "pinneapple-design-study/v1"

#: The kinds of study the schema describes. ``optimize`` and ``surrogate`` are the two :func:`.study.run_study`
#: runs today; the other three are described here and refused with the reason that is missing (see ``.study``).
KINDS = ("optimize", "surrogate", "pinn", "inverse", "twin")
KINDS_BEYOND_OPTIMISATION = ("surrogate", "pinn", "inverse", "twin")

#: ``geometry``: an entry of ``pp.geometries``; ``stl``: the user's own mesh with a parametrization; ``system``: a
#: model of a system rather than a shape (a plant, a thermal network, a zoo model).
BASE_KINDS = ("geometry", "stl", "system")

#: What the evaluator is: a learned model, a classical engineering model, or the reference solver.
FIDELITIES = ("surrogate", "model", "reference")

#: The surrogate families of :func:`pinneapple_autolab.surrogates.compare_models`; a test keeps the two in step.
SURROGATE_FAMILIES = ("gp", "mlp_ensemble", "quadratic")

SENSES = ("min", "max")

#: Units that say nothing about the quantity: refused wherever a unit is required.
_NOT_A_UNIT = frozenset({"", "?", "unknown", "n/a", "na", "none", "no unit", "unitless", "unspecified"})

_TOOL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]*$")


# --------------------------------------------------------------------------- strict parsing helpers
def _unit(value: Any, what: str) -> str:
    """The unit as written, or a refusal: a quantity without a unit is not a quantity anyone can check."""
    if not isinstance(value, str):
        raise ValueError(f"{what}: the unit must be a string, got {type(value).__name__}")
    u = value.strip()
    if u.lower() in _NOT_A_UNIT:
        raise ValueError(f"{what}: no unit given ({value!r}); a dimensionless output declares '1' (or '-'). Anything "
                         "else must name its unit, because an output without one is not assumed to have one")
    return u


def _positive(value: Any, what: str) -> float:
    v = float(value)
    if not v > 0 or v != v:
        raise ValueError(f"{what} must be a positive number, got {value!r}")
    return v


def _keys(data: Mapping, cls: type, what: str) -> None:
    known = [f.name for f in fields(cls)]
    if not isinstance(data, Mapping):
        raise TypeError(f"{what}: expected a mapping, got {type(data).__name__}")
    extra = sorted(set(data) - set(known))
    if extra:
        raise ValueError(f"{what}: unknown field(s) {extra}; known: {known}")
    missing = [f.name for f in fields(cls)
               if f.name not in data and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError(f"{what}: missing field(s) {missing}")


def _items(value: Any, cls: type, what: str) -> list:
    """A list of nested objects, each given as the object itself or as its dict form."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{what}: expected a list, got {type(value).__name__}")
    return [_obj(v, cls, f"{what}[{i}]") for i, v in enumerate(value)]


def _obj(value: Any, cls: type, what: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_dict(value)
    raise TypeError(f"{what}: expected a {cls.__name__} or a dict, got {type(value).__name__}")


def _one_of(value: Any, allowed: Sequence[str], what: str) -> str:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {list(allowed)}, got {value!r}")
    return str(value)


# --------------------------------------------------------------------------- a quantity with a unit
@dataclass(frozen=True)
class Quantity:
    """One output of the evaluator: ``name`` is the study's word for it, ``unit`` what it is measured in.

    ``label`` is what a report shows; ``description`` says how the number is defined when the name is not enough
    ("total pressure loss from inlet to outlet, mass-flow averaged").
    """

    name: str
    unit: str
    label: str = ""
    description: str = ""

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("a quantity needs a name")
        object.__setattr__(self, "unit", _unit(self.unit, f"output {self.name!r}"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "label": self.label, "description": self.description}

    @classmethod
    def from_dict(cls, d: Mapping) -> Quantity:
        _keys(d, cls, "output")
        return cls(**dict(d))


# --------------------------------------------------------------------------- what is varied
@dataclass(frozen=True)
class Variable:
    """A design variable in ``[lower, upper]`` with its unit; ``kind`` is ``"continuous"`` or ``"integer"``."""

    name: str
    lower: float
    upper: float
    unit: str
    kind: str = "continuous"
    description: str = ""

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("a variable needs a name")
        lo, hi = float(self.lower), float(self.upper)
        if not hi > lo:
            raise ValueError(f"variable {self.name!r}: upper ({self.upper!r}) must exceed lower ({self.lower!r})")
        if not (lo == lo and hi == hi and abs(lo) != float("inf") and abs(hi) != float("inf")):
            raise ValueError(f"variable {self.name!r}: bounds must be finite, got [{self.lower}, {self.upper}]")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "kind", _one_of(self.kind, ("continuous", "integer"), f"variable {self.name!r} kind"))
        object.__setattr__(self, "unit", _unit(self.unit, f"variable {self.name!r}"))

    @property
    def span(self) -> float:
        return self.upper - self.lower

    def contains(self, x: float) -> bool:
        return self.lower <= float(x) <= self.upper

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lower": self.lower, "upper": self.upper, "unit": self.unit, "kind": self.kind,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: Mapping) -> Variable:
        _keys(d, cls, "variable")
        return cls(**dict(d))


@dataclass(frozen=True)
class Base:
    """What the study varies: a geometry (a library entry, the user's STL) or a system.

    ``ref`` names it — ``"pp.geometries:naca2412"``, the id of a catalog entry, the path of an STL or the id of a
    system model. ``fixed`` holds the parameters that are not varied (and are not design variables); ``baseline``
    holds the values of the variables of the reference design.
    """

    kind: str
    ref: str
    variables: tuple[Variable, ...] = ()
    fixed: Mapping[str, float] = field(default_factory=dict)
    baseline: Mapping[str, float] = field(default_factory=dict)
    parametrization: str = ""
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "kind", _one_of(self.kind, BASE_KINDS, "base kind"))
        if not str(self.ref).strip():
            raise ValueError("the base needs a reference (a geometry id, a catalog entry, a path, a system model)")
        vs = tuple(_obj(v, Variable, "variable") for v in self.variables)
        object.__setattr__(self, "variables", vs)
        names = [v.name for v in vs]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"variable names must be unique; repeated: {dup}")
        fixed = {str(k): float(v) for k, v in dict(self.fixed).items()}
        clash = sorted(set(fixed) & set(names))
        if clash:
            raise ValueError(f"{clash} are both varied and fixed; a parameter is one or the other")
        object.__setattr__(self, "fixed", fixed)
        base = {str(k): float(v) for k, v in dict(self.baseline).items()}
        unknown = sorted(set(base) - set(names))
        if unknown:
            raise ValueError(f"the baseline gives {unknown}, which are not variables; variables: {names}")
        for k, v in base.items():
            var = next(x for x in vs if x.name == k)
            if not var.contains(v):
                raise ValueError(f"baseline {k} = {v} is outside its range [{var.lower}, {var.upper}]")
        object.__setattr__(self, "baseline", base)
        if self.kind == "stl" and not str(self.parametrization).strip():
            raise ValueError("an STL base needs a parametrization (ffd, rbf, ...): without one there is nothing to "
                             "vary on the user's mesh")

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def baseline_design(self) -> dict[str, float]:
        """The reference design: the declared baseline, or the midpoint of every range."""
        return {v.name: float(self.baseline.get(v.name, 0.5 * (v.lower + v.upper))) for v in self.variables}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "variables": [v.to_dict() for v in self.variables],
                "fixed": dict(self.fixed), "baseline": dict(self.baseline),
                "parametrization": self.parametrization, "description": self.description}

    @classmethod
    def from_dict(cls, d: Mapping) -> Base:
        _keys(d, cls, "base")
        return cls(kind=d["kind"], ref=d["ref"], variables=tuple(_items(d.get("variables", []), Variable,
                                                                       "base.variables")),
                   fixed=dict(d.get("fixed") or {}), baseline=dict(d.get("baseline") or {}),
                   parametrization=d.get("parametrization", ""), description=d.get("description", ""))


# --------------------------------------------------------------------------- who evaluates
@dataclass(frozen=True)
class Evaluator:
    """The tool that produces the numbers: a catalog id, the input mapping and the outputs with their units.

    ``inputs`` maps every variable of the base to the name that tool calls it by (``{"h_fin": "fin_height_mm"}``);
    ``conditions`` holds the operating conditions the study never changes (speed, temperature, boundary conditions),
    handed to the tool as they are. ``settings`` is everything else the tool needs to be configured (mesh level,
    turbulence model, iterations) and is declared here so a study is reproducible from its spec alone.
    """

    tool: str
    inputs: Mapping[str, str]
    outputs: tuple[Quantity, ...]
    conditions: Mapping[str, float] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    fidelity: str = "reference"
    description: str = ""

    def __post_init__(self):
        tool = str(self.tool).strip()
        if not _TOOL_ID.match(tool):
            raise ValueError(f"the evaluator's tool id {self.tool!r} is not an id (letters, digits and _.:+/-, e.g. "
                             "'openfoam.rans_airfoil' or 'vlm.wing_planform')")
        object.__setattr__(self, "tool", tool)
        inputs = {str(k): str(v) for k, v in dict(self.inputs).items()}
        if not inputs:
            raise ValueError("the evaluator needs the input mapping: which of its inputs each variable is")
        empty = sorted(k for k, v in inputs.items() if not v.strip())
        if empty:
            raise ValueError(f"{empty} map to an empty tool input name")
        object.__setattr__(self, "inputs", inputs)
        outs = tuple(_obj(q, Quantity, "output") for q in self.outputs)
        if not outs:
            raise ValueError("the evaluator needs at least one output")
        names = [q.name for q in outs]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"output names must be unique; repeated: {dup}")
        object.__setattr__(self, "outputs", outs)
        object.__setattr__(self, "conditions", {str(k): float(v) for k, v in dict(self.conditions).items()})
        object.__setattr__(self, "settings", dict(self.settings))
        object.__setattr__(self, "fidelity", _one_of(self.fidelity, FIDELITIES, "the evaluator's fidelity"))

    @property
    def output_names(self) -> list[str]:
        return [q.name for q in self.outputs]

    def unit_of(self, output: str) -> str:
        for q in self.outputs:
            if q.name == output:
                return q.unit
        raise ValueError(f"no output named {output!r}; outputs: {self.output_names}")

    def tool_params(self, design: Mapping[str, float]) -> dict[str, float]:
        """The design in the tool's own input names, with the fixed conditions the study never changes.

        ``settings`` is not merged here: it configures the tool itself (mesh level, turbulence model), so the adapter
        that wraps the tool for this study reads it from the spec it was built from.
        """
        return {**{self.inputs[k]: float(v) for k, v in design.items()}, **dict(self.conditions)}

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "inputs": dict(self.inputs), "outputs": [q.to_dict() for q in self.outputs],
                "conditions": dict(self.conditions), "settings": dict(self.settings), "fidelity": self.fidelity,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: Mapping) -> Evaluator:
        _keys(d, cls, "evaluator")
        for required in ("tool", "inputs", "outputs"):
            if required not in d:
                raise ValueError(f"evaluator: missing field {required!r}")
        return cls(tool=d["tool"], inputs=dict(d["inputs"]),
                   outputs=tuple(_items(d["outputs"], Quantity, "evaluator.outputs")),
                   conditions=dict(d.get("conditions") or {}), settings=dict(d.get("settings") or {}),
                   fidelity=d.get("fidelity", "reference"), description=d.get("description", ""))


# --------------------------------------------------------------------------- what to achieve
@dataclass(frozen=True)
class Objective:
    """Minimise or maximise one output of the evaluator (its unit is the output's)."""

    output: str
    sense: str = "min"
    label: str = ""

    def __post_init__(self):
        if not str(self.output).strip():
            raise ValueError("an objective needs an output name")
        object.__setattr__(self, "sense", _one_of(self.sense, SENSES, f"objective {self.output!r} sense"))

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output, "sense": self.sense, "label": self.label}

    @classmethod
    def from_dict(cls, d: Mapping) -> Objective:
        _keys(d, cls, "objective")
        return cls(**dict(d))


@dataclass(frozen=True)
class Limit:
    """``lower <= output <= upper``, in the output's units or as a fraction of the baseline's value.

    ``lower_frac_of_baseline=1.0`` keeps the baseline ("no less downforce than the reference design");
    ``upper_frac_of_baseline=1.05`` allows 5 % more mass. A relative limit is only a number once the baseline has
    been evaluated, so :meth:`resolved` refuses before that; the resolution itself is
    :meth:`pinneapple_autolab.mission.Limit.resolved`, not a second implementation of it.

    ``kind`` labels what the limit stands for — ``"physical"``, ``"manufacturing"``, ``"validity"`` — and only
    reaches the feasibility report.
    """

    output: str
    lower: float | None = None
    upper: float | None = None
    lower_frac_of_baseline: float | None = None
    upper_frac_of_baseline: float | None = None
    kind: str = "physical"
    label: str = ""

    def __post_init__(self):
        if not str(self.output).strip():
            raise ValueError("a limit needs an output name")
        if all(v is None for v in (self.lower, self.upper, self.lower_frac_of_baseline,
                                   self.upper_frac_of_baseline)):
            raise ValueError(f"limit on {self.output!r} bounds nothing: give lower/upper or a fraction of the "
                             "baseline")
        for name in ("lower", "upper", "lower_frac_of_baseline", "upper_frac_of_baseline"):
            v = getattr(self, name)
            if v is None:
                continue
            if "frac" in name:
                object.__setattr__(self, name, _positive(v, f"limit {self.output!r} {name}"))
            else:
                f = float(v)
                if f != f:
                    raise ValueError(f"limit {self.output!r}: {name} is not a number")
                object.__setattr__(self, name, f)
        if self.lower is not None and self.upper is not None and self.upper < self.lower:
            raise ValueError(f"limit {self.output!r}: upper {self.upper} is below lower {self.lower}")

    @property
    def relative(self) -> bool:
        return self.lower_frac_of_baseline is not None or self.upper_frac_of_baseline is not None

    def describe(self) -> str:
        parts = []
        if self.lower_frac_of_baseline is not None:
            parts.append(f">= {self.lower_frac_of_baseline:g} x baseline")
        if self.lower is not None:
            parts.append(f">= {self.lower:g}")
        if self.upper_frac_of_baseline is not None:
            parts.append(f"<= {self.upper_frac_of_baseline:g} x baseline")
        if self.upper is not None:
            parts.append(f"<= {self.upper:g}")
        return f"{self.label or self.output} " + " and ".join(parts)

    def resolved(self, baseline: Mapping[str, float] | None):
        """The absolute ``Constraint``, once the baseline's outputs are known (an autolab ``Limit`` does the work)."""
        from pinneapple_autolab.mission import Limit as LabLimit
        return LabLimit(name=self.output, lower=self.lower, upper=self.upper,
                        lower_frac_of_baseline=self.lower_frac_of_baseline,
                        upper_frac_of_baseline=self.upper_frac_of_baseline, label=self.label,
                        kind=self.kind).resolved(None if baseline is None else dict(baseline))

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output, "lower": self.lower, "upper": self.upper,
                "lower_frac_of_baseline": self.lower_frac_of_baseline,
                "upper_frac_of_baseline": self.upper_frac_of_baseline, "kind": self.kind, "label": self.label}

    @classmethod
    def from_dict(cls, d: Mapping) -> Limit:
        _keys(d, cls, "limit")
        return cls(**dict(d))


# --------------------------------------------------------------------------- how much may be spent
@dataclass(frozen=True)
class Budget:
    """What the study may spend.

    ``max_evaluations`` counts **reference** evaluations of the evaluator (the expensive ones: one solver run each);
    ``surrogate_evaluations`` counts the optimiser's calls on the surrogate, which cost nothing by comparison and are
    therefore budgeted separately. ``n_doe`` defaults to the rule the autonomous lab uses
    (``max(2 d + 2, 8)`` capped by what is left); ``n_verify`` is reserved for the end and is never spent on the
    search. ``max_wall_s`` caps the summed evaluator wall time when the tool reports it.
    """

    max_evaluations: int = 40
    n_doe: int | None = None
    n_verify: int = 3
    surrogate_evaluations: int = 2000
    max_wall_s: float | None = None
    seed: int = 0

    def __post_init__(self):
        m = int(self.max_evaluations)
        if m < 4:
            raise ValueError(f"a budget of {m} evaluation(s) cannot hold a baseline, a DOE and a verification "
                             "(at least 4)")
        object.__setattr__(self, "max_evaluations", m)
        if self.n_doe is not None:
            n = int(self.n_doe)
            if n < 2:
                raise ValueError(f"n_doe must be at least 2, got {n}")
            object.__setattr__(self, "n_doe", n)
        v = int(self.n_verify)
        if v < 0:
            raise ValueError(f"n_verify must be >= 0, got {v}")
        object.__setattr__(self, "n_verify", v)
        s = int(self.surrogate_evaluations)
        if s < 10:
            raise ValueError(f"surrogate_evaluations must be at least 10, got {s}")
        object.__setattr__(self, "surrogate_evaluations", s)
        if self.max_wall_s is not None:
            object.__setattr__(self, "max_wall_s", _positive(self.max_wall_s, "max_wall_s"))
        if int(self.seed) < 0:
            raise ValueError("seed must be >= 0")
        object.__setattr__(self, "seed", int(self.seed))
        if v >= m:
            raise ValueError(f"n_verify ({v}) must be smaller than max_evaluations ({m})")

    def planned(self, n_variables: int) -> dict[str, int]:
        """How the reference budget splits: one baseline, the DOE, the verifications, and the search left over."""
        n_doe = self.n_doe if self.n_doe is not None else min(max(2 * n_variables + 2, 8),
                                                             max(self.max_evaluations - self.n_verify - 2, 2))
        if 1 + n_doe + self.n_verify > self.max_evaluations:
            raise ValueError(f"a budget of {self.max_evaluations} evaluations cannot hold 1 baseline + {n_doe} DOE + "
                             f"{self.n_verify} verifications; raise max_evaluations or lower n_doe/n_verify")
        return {"baseline": 1, "doe": n_doe, "verify": self.n_verify,
                "search": self.max_evaluations - 1 - n_doe - self.n_verify}

    def to_dict(self) -> dict[str, Any]:
        return {"max_evaluations": self.max_evaluations, "n_doe": self.n_doe, "n_verify": self.n_verify,
                "surrogate_evaluations": self.surrogate_evaluations, "max_wall_s": self.max_wall_s,
                "seed": self.seed}

    @classmethod
    def from_dict(cls, d: Mapping) -> Budget:
        _keys(d, cls, "budget")
        return cls(**dict(d))


# --------------------------------------------------------------------------- what counts as verified
@dataclass(frozen=True)
class Verifier:
    """The tolerances that decide whether the study's answer can be signed.

    ``surrogate_target``: the held-out nRMSE (RMSE divided by the standard deviation of the measured outputs) a
    surrogate has to reach before it may steer the search. ``agreement_tol``: per output, the relative tolerance
    between the surrogate's prediction and the reference run of the same design (default ``surrogate_target``).
    ``min_verified``: how many designs of the answer must have been measured, not predicted. ``folds``: the
    cross-validation folds of the surrogate.
    """

    surrogate_target: float = 0.10
    agreement_tol: Mapping[str, float] = field(default_factory=dict)
    min_verified: int = 1
    folds: int = 5

    def __post_init__(self):
        t = _positive(self.surrogate_target, "surrogate_target")
        if t >= 1:
            raise ValueError(f"surrogate_target is a relative error; {t} means 'anything' and cannot be a target")
        object.__setattr__(self, "surrogate_target", t)
        tol = {str(k): _positive(v, f"agreement_tol[{k}]") for k, v in dict(self.agreement_tol).items()}
        object.__setattr__(self, "agreement_tol", tol)
        if int(self.min_verified) < 0:
            raise ValueError("min_verified must be >= 0")
        object.__setattr__(self, "min_verified", int(self.min_verified))
        if int(self.folds) < 2:
            raise ValueError("folds must be at least 2")
        object.__setattr__(self, "folds", int(self.folds))

    def tolerance(self, output: str) -> float:
        return float(self.agreement_tol.get(output, self.surrogate_target))

    def to_dict(self) -> dict[str, Any]:
        return {"surrogate_target": self.surrogate_target, "agreement_tol": dict(self.agreement_tol),
                "min_verified": self.min_verified, "folds": self.folds}

    @classmethod
    def from_dict(cls, d: Mapping) -> Verifier:
        _keys(d, cls, "verifier")
        return cls(**dict(d))


@dataclass(frozen=True)
class Model:
    """The learned model of the study: which surrogate families compete, and, for a PINN or operator study, the PDE
    residual the fit is informed by (``pinneapple_analysis.surrogate.pde``)."""

    families: tuple[str, ...] = ("gp", "quadratic")
    pde: Mapping[str, Any] | None = None
    notes: str = ""

    def __post_init__(self):
        fams = tuple(str(f) for f in self.families)
        if not fams:
            raise ValueError("the model needs at least one surrogate family")
        unknown = [f for f in fams if f not in SURROGATE_FAMILIES]
        if unknown:
            raise ValueError(f"unknown surrogate families {unknown}; available: {list(SURROGATE_FAMILIES)}")
        object.__setattr__(self, "families", fams)
        object.__setattr__(self, "pde", None if self.pde is None else dict(self.pde))

    def to_dict(self) -> dict[str, Any]:
        return {"families": list(self.families), "pde": None if self.pde is None else dict(self.pde),
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Mapping) -> Model:
        _keys(d, cls, "model")
        return cls(families=tuple(d.get("families") or ("gp", "quadratic")), pde=d.get("pde"),
                   notes=d.get("notes", ""))


# --------------------------------------------------------------------------- the study
@dataclass(frozen=True)
class DesignStudy:
    """A whole study: kind, base, evaluator, objectives, limits, budget and verifier.

    ``statement`` is the user's sentence, kept as written (it is what the Engineer turned into the rest); ``metadata``
    carries whatever the app wants to keep with the study (owner, client, app id) and never changes how it runs.
    """

    kind: str
    title: str
    base: Base
    evaluator: Evaluator
    objectives: tuple[Objective, ...] = ()
    constraints: tuple[Limit, ...] = ()
    budget: Budget = field(default_factory=Budget)
    verifier: Verifier = field(default_factory=Verifier)
    model: Model = field(default_factory=Model)
    statement: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "kind", _one_of(self.kind, KINDS, "study kind"))
        if not str(self.title).strip():
            raise ValueError("a study needs a title")
        base = _obj(self.base, Base, "base")
        evaluator = _obj(self.evaluator, Evaluator, "evaluator")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "evaluator", evaluator)
        object.__setattr__(self, "objectives", tuple(_obj(o, Objective, "objective") for o in self.objectives))
        object.__setattr__(self, "constraints", tuple(_obj(c, Limit, "limit") for c in self.constraints))
        for name, value, cls in (("budget", self.budget, Budget), ("verifier", self.verifier, Verifier),
                                 ("model", self.model, Model)):
            if not isinstance(value, cls):
                object.__setattr__(self, name, cls.from_dict(value) if isinstance(value, Mapping)
                                   else cls(**value))
        if not base.variables:
            raise ValueError("a study needs at least one variable: a design study that varies nothing is a "
                             "simulation")
        missing = sorted(set(base.names) - set(evaluator.inputs))
        extra = sorted(set(evaluator.inputs) - set(base.names))
        if missing or extra:
            raise ValueError(f"the evaluator's input mapping and the variables do not match: variables without a "
                             f"tool input {missing}, tool inputs without a variable {extra}")
        outs = set(evaluator.output_names)
        bad = sorted({o.output for o in self.objectives if o.output not in outs}
                     | {c.output for c in self.constraints if c.output not in outs})
        if bad:
            raise ValueError(f"objectives or limits name outputs the evaluator does not declare: {bad}; "
                             f"outputs: {sorted(outs)}")
        if not self.objectives:
            raise ValueError(f"a {self.kind} study needs at least one objective: a study exists to answer a "
                             "question, and a surrogate study's objectives are what its user will optimise with it")
        self.budget.planned(len(base.variables))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.kind == "pinn" and not self.model.pde:
            raise ValueError("a pinn study needs model.pde: the residual the physics loss is built from, without "
                             "which the model is an ordinary regressor")

    # ------------------------------------------------------------------ shortcuts
    @property
    def outputs(self) -> list[str]:
        """The outputs the study has to predict: objectives first, then the constrained ones."""
        return list(dict.fromkeys([o.output for o in self.objectives] + [c.output for c in self.constraints]))

    def units(self) -> dict[str, str]:
        return {q.name: q.unit for q in self.evaluator.outputs}

    def baseline_design(self) -> dict[str, float]:
        return self.base.baseline_design()

    def digest(self) -> str:
        """Content hash of the spec: two studies with the same digest are the same study."""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "kind": self.kind, "title": self.title, "statement": self.statement,
                "base": self.base.to_dict(), "evaluator": self.evaluator.to_dict(),
                "objectives": [o.to_dict() for o in self.objectives],
                "constraints": [c.to_dict() for c in self.constraints],
                "budget": self.budget.to_dict(), "verifier": self.verifier.to_dict(),
                "model": self.model.to_dict(), "metadata": dict(self.metadata)}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Mapping) -> DesignStudy:
        """Strict: a missing or unknown field raises, and so does another schema version."""
        if not isinstance(d, Mapping):
            raise TypeError(f"study: expected a mapping, got {type(d).__name__}")
        d = dict(d)
        schema = d.pop("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"schema {schema!r} is not supported; this library reads {SCHEMA!r}")
        _keys(d, cls, "study")
        for required in ("kind", "title", "base", "evaluator"):
            if required not in d:
                raise ValueError(f"study: missing field {required!r}")
        return cls(kind=d["kind"], title=d["title"], statement=d.get("statement", ""),
                   base=_obj(d["base"], Base, "base"), evaluator=_obj(d["evaluator"], Evaluator, "evaluator"),
                   objectives=tuple(_items(d.get("objectives") or [], Objective, "objectives")),
                   constraints=tuple(_items(d.get("constraints") or [], Limit, "constraints")),
                   budget=Budget.from_dict(d["budget"]) if isinstance(d.get("budget"), Mapping)
                   else (d.get("budget") or Budget()),
                   verifier=Verifier.from_dict(d["verifier"]) if isinstance(d.get("verifier"), Mapping)
                   else (d.get("verifier") or Verifier()),
                   model=Model.from_dict(d["model"]) if isinstance(d.get("model"), Mapping)
                   else (d.get("model") or Model()),
                   metadata=dict(d.get("metadata") or {}))

    @classmethod
    def from_json(cls, text: str) -> DesignStudy:
        return cls.from_dict(json.loads(text))

    @classmethod
    def create(cls, *, kind: str, title: str, base: Base | Mapping, evaluator: Evaluator | Mapping,
               objectives: Sequence[Objective | Mapping] = (), constraints: Sequence[Limit | Mapping] = (),
               budget: Budget | Mapping | None = None, verifier: Verifier | Mapping | None = None,
               model: Model | Mapping | None = None, statement: str = "",
               metadata: Mapping[str, Any] | None = None) -> DesignStudy:
        """The same validation, from anything that looks like the parts (dicts, objects, tuples)."""
        return cls(kind=kind, title=title, statement=statement, base=_obj(base, Base, "base"),
                   evaluator=_obj(evaluator, Evaluator, "evaluator"), objectives=tuple(objectives),
                   constraints=tuple(constraints), budget=budget or Budget(),
                   verifier=verifier or Verifier(), model=model or Model(), metadata=dict(metadata or {}))

    def describe(self) -> dict[str, Any]:
        """What the Studio shows before the study runs: kind, what varies, what is measured and what must hold."""
        p = self.budget.planned(len(self.base.variables))
        return {"schema": SCHEMA, "digest": self.digest(), "kind": self.kind, "title": self.title,
                "statement": self.statement, "base": {"kind": self.base.kind, "ref": self.base.ref,
                                                      "variables": [v.to_dict() for v in self.base.variables],
                                                      "fixed": dict(self.base.fixed),
                                                      "baseline": self.base.baseline_design()},
                "evaluator": {"tool": self.evaluator.tool, "fidelity": self.evaluator.fidelity,
                              "outputs": [q.to_dict() for q in self.evaluator.outputs]},
                "objectives": [o.to_dict() | {"unit": self.evaluator.unit_of(o.output)} for o in self.objectives],
                "constraints": [c.to_dict() | {"text": c.describe(), "relative": c.relative}
                                for c in self.constraints],
                "budget": {**self.budget.to_dict(), "plan": p},
                "verifier": self.verifier.to_dict(), "model": self.model.to_dict()}
