"""What the advisor must know about a study before it can suggest a pipeline, and where every value came from.

The brief is a small, closed set of fields (`FIELDS`) with allowed values, because a recommendation is only as
trustworthy as the facts behind it: a value read from the sentence, an answer the user gave, a value a local model
proposed, or a documented default. Each field records its source, so the plan can say "3-D because you said so" and
"100 evaluations because that is the default for a parameter sweep, change it if you know the number".

Nothing here is inferred twice: the rule-based reader (``pinneapple_advisor.extract``) fills what the words say
first, the language model only fills what is still empty, and a value that contradicts a rule is kept with its
source so the disagreement is visible instead of silently resolved.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Family names are the PDE kinds of pinneapple_physics.pde_environment.capabilities (open set: "custom" is allowed
# for a problem no family covers, which is a real answer -- pinneapple_problemdesign.formulate composes one).
PHYSICS_FAMILIES: Sequence[str] = (
    "black_scholes", "burgers", "darcy", "fisher_kpp", "heat_equation", "helmholtz", "inviscid_burgers", "laplace",
    "linear_elasticity", "navier_stokes", "poisson", "reaction_diffusion", "sir", "wave", "custom",
)
GOALS: Sequence[str] = ("forward", "inverse", "optimization", "screening")
OUTPUT_KINDS: Sequence[str] = ("scalar", "grid_field", "mesh_field", "time_series")

# name -> (kind, allowed values / bounds, default, the question that fills it)
FIELDS: dict[str, dict[str, Any]] = {
    "physics_family": {"kind": "choice", "choices": PHYSICS_FAMILIES, "default": None, "question": "physics_family"},
    "dimension": {"kind": "int", "min": 1, "max": 3, "default": None, "question": "dimension"},
    "time_dependent": {"kind": "bool", "default": None, "question": "time_dependent"},
    "nonlinear": {"kind": "bool", "default": None, "question": "nonlinear"},
    "turbulent": {"kind": "bool", "default": None, "question": "turbulent"},
    "geometry_present": {"kind": "bool", "default": None, "question": "geometry_present"},
    "geometry_formats": {"kind": "list", "choices": ("cad", "stl", "step", "analytic"), "default": None,
                         "question": "geometry_formats"},
    "geometry_varies": {"kind": "bool", "default": None, "question": "geometry_varies"},
    "mesh_present": {"kind": "bool", "default": None, "question": "mesh_present"},
    "dataset_present": {"kind": "bool", "default": None, "question": "dataset_present"},
    "n_simulations_present": {"kind": "int", "min": 0, "default": None, "question": "n_simulations_present"},
    "parameter_varies": {"kind": "bool", "default": None, "question": "parameter_varies"},
    "n_parameters": {"kind": "int", "min": 0, "default": None, "question": "n_parameters"},
    "goal": {"kind": "choice", "choices": GOALS, "default": "forward", "question": "goal"},
    "evaluations_needed": {"kind": "int", "min": 1, "default": None, "question": "evaluations_needed"},
    "accuracy_target": {"kind": "number", "min": 1e-6, "max": 1.0, "default": 0.05, "question": "accuracy_target"},
    "output_kind": {"kind": "choice", "choices": OUTPUT_KINDS, "default": None, "question": "output_kind"},
    "budget_hours": {"kind": "number", "min": 0.0, "default": None, "question": "budget_hours"},
    "budget_usd": {"kind": "number", "min": 0.0, "default": None, "question": "budget_usd"},
    "cores": {"kind": "int", "min": 1, "default": 1, "question": "cores"},
    "gpu": {"kind": "bool", "default": False, "question": "gpu"},
    "approx_cells": {"kind": "int", "min": 100, "default": None, "question": "approx_cells"},
    "solve_seconds_measured": {"kind": "number", "min": 0.0, "default": None, "question": "solve_seconds_measured"},
}

# The order an open question is asked in: physics before geometry, geometry before data, money last.
ASK_ORDER: Sequence[str] = (
    "physics_family", "dimension", "time_dependent", "nonlinear", "turbulent", "geometry_present",
    "geometry_varies", "mesh_present", "dataset_present", "n_simulations_present", "parameter_varies",
    "n_parameters", "goal", "evaluations_needed", "output_kind", "accuracy_target", "cores", "gpu",
    "approx_cells", "solve_seconds_measured", "budget_hours", "budget_usd",
)


class BriefError(ValueError):
    """A value that the field does not allow (wrong type, out of range, not in the closed list)."""


@dataclass
class Brief:
    """The facts of a study, each with the source it came from."""
    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)     # field -> "text" | "rule" | "answer" | "llm:<model>" | "default"
    description: str = ""
    notes: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)   # values a model proposed and validation refused

    # ---- writing ------------------------------------------------------------------------------------------------
    def set(self, name: str, value: Any, source: str = "answer") -> Brief:
        """Set a field, validating it against :data:`FIELDS`; raises :class:`BriefError` when it does not fit."""
        if name not in FIELDS:
            raise BriefError(f"unknown field {name!r}; known: {', '.join(FIELDS)}")
        self.values[name] = _validate(name, value)
        self.sources[name] = source
        return self

    def try_set(self, name: str, value: Any, source: str = "answer") -> bool:
        """Set when the value is valid; otherwise record it in ``rejected`` and return False (never raise)."""
        try:
            self.set(name, value, source)
            return True
        except BriefError as exc:
            self.rejected.append({"field": name, "value": value, "source": source, "why": str(exc)})
            return False

    def merge(self, other: dict[str, Any], source: str = "answer", overwrite: bool = False) -> Brief:
        """Fill the fields ``other`` names (a dict of field -> value), keeping what is already known unless
        ``overwrite``."""
        for name, value in dict(other or {}).items():
            if name not in FIELDS:
                self.rejected.append({"field": name, "value": value, "source": source,
                                      "why": f"unknown field; known: {', '.join(FIELDS)}"})
                continue
            if not overwrite and self.values.get(name) is not None:
                continue
            self.try_set(name, value, source)
        return self

    # ---- reading ------------------------------------------------------------------------------------------------
    def get(self, name: str, default: Any = None) -> Any:
        value = self.values.get(name)
        return FIELDS[name]["default"] if value is None and name not in self.values else (
            default if value is None else value)

    def source(self, name: str) -> str:
        return self.sources.get(name, "default")

    def unknown(self, fields: Sequence[str] | None = None) -> list[str]:
        """The fields with no value yet, in the order they should be asked.

        A field's default is what the plan falls back to, not knowledge: ``goal`` defaults to ``forward`` and is
        still asked, because the answer changes which stages the plan has.
        """
        names = list(fields) if fields is not None else list(ASK_ORDER)
        return [n for n in names if n in FIELDS and self.values.get(n) is None]

    def complete(self) -> bool:
        return not self.unknown()

    # ---- serialising --------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "values": dict(self.values), "sources": dict(self.sources),
                "unknown": self.unknown(), "rejected": list(self.rejected), "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Brief:
        brief = cls(description=str((data or {}).get("description", "")))
        for name, value in dict((data or {}).get("values") or {}).items():
            brief.try_set(name, value, "answer")
        for name, source in dict((data or {}).get("sources") or {}).items():
            if name in brief.values:
                brief.sources[name] = str(source)
        return brief


def _validate(name: str, value: Any) -> Any:
    spec = FIELDS[name]
    kind = spec["kind"]
    if value is None:
        return None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "yes", "sim", "1", "y"):
            return True
        if isinstance(value, str) and value.strip().lower() in ("false", "no", "nao", "não", "0", "n"):
            return False
        raise BriefError(f"{name} is yes/no, not {value!r}")
    if kind == "int":
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise BriefError(f"{name} is a whole number, not {value!r}") from exc
        if out < spec.get("min", out) or out > spec.get("max", out):
            raise BriefError(f"{name} must be between {spec.get('min')} and {spec.get('max')}, not {out}")
        return out
    if kind == "number":
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise BriefError(f"{name} is a number, not {value!r}") from exc
        if out < spec.get("min", out) or out > spec.get("max", out):
            raise BriefError(f"{name} must be between {spec.get('min')} and {spec.get('max')}, not {out}")
        return out
    if kind == "choice":
        out = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        if out not in spec["choices"]:
            raise BriefError(f"{name} is one of {', '.join(spec['choices'])}, not {value!r}")
        return out
    if kind == "list":
        items = [value] if isinstance(value, str) else list(value)
        out = [str(i).strip().lower() for i in items]
        bad = [i for i in out if i not in spec["choices"]]
        if bad:
            raise BriefError(f"{name} takes values from {', '.join(spec['choices'])}; refused {bad}")
        return out
    raise BriefError(f"field {name} has an unsupported kind {kind!r}")


__all__ = ["FIELDS", "ASK_ORDER", "PHYSICS_FAMILIES", "GOALS", "OUTPUT_KINDS", "Brief", "BriefError"]
