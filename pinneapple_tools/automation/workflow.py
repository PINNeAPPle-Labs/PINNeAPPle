"""The structured workflow an automation run executes: inputs with bounds, steps with preconditions, the artifacts
each step must produce, the confirmations it needs, and the acceptance criteria of the result.

A workflow is plain data (YAML or JSON), so it can be written by hand, produced by :func:`interpret` from a sentence,
reviewed in a pull request and replayed later::

    id: channel_obstacle
    title: Channel with an obstacle, pressure drop vs obstacle height
    inputs:
      obstacle_height_mm: {default: 40, min: 10, max: 80, unit: mm, description: height of the obstacle}
    steps:
      - id: open_project
        action: freecad.open
        params: {document: "{example}/channel_obstacle.FCStd"}
        expects: [{path: "open_project/project.json"}]
      - id: solve
        action: openfoam.solve
        after: [mesh]
        confirm: [expensive]
        params: {case: "{steps.setup.case}"}

Values in ``params`` and ``expects`` are templates: ``{inputs.<name>}``, ``{steps.<id>.<output>}``, ``{workdir}``
(the run directory) and ``{example}`` (the folder of the workflow file). A template that is a whole value keeps the
type of what it refers to (a number stays a number).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ArtifactSpec", "InputSpec", "Step", "Criterion", "Workflow", "WorkflowError", "CONFIRM_CATEGORIES",
           "render"]

CONFIRM_CATEGORIES = {
    "destructive": "deletes or replaces data",
    "overwrite": "writes over a file that already exists",
    "expensive": "starts a long or costly job (solver, cluster, licence)",
    "external": "has consequences outside this run (network, e-mail, another system)",
}
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TEMPLATE = re.compile(r"\{((?:inputs|steps|workdir|example)(?:\.[A-Za-z0-9_]+)*)\}")


class WorkflowError(ValueError):
    """The workflow file or the run inputs are invalid; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


@dataclass
class ArtifactSpec:
    """A file or folder a step must leave behind. ``path`` is relative to the run directory."""
    path: str
    kind: str = "file"                      # file | dir
    min_bytes: int = 1
    description: str = ""

    @classmethod
    def from_any(cls, v: Any) -> ArtifactSpec:
        return cls(path=v) if isinstance(v, str) else cls(**v)


@dataclass
class InputSpec:
    default: Any
    min: float | None = None
    max: float | None = None
    unit: str = ""
    description: str = ""
    choices: list[Any] | None = None

    def check(self, name: str, value: Any) -> list[str]:
        if self.choices is not None:
            return [] if value in self.choices else [f"input {name}={value!r} is not one of {self.choices}"]
        if isinstance(self.default, (int, float)) and not isinstance(self.default, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
                return [f"input {name} must be a number, got {value!r}"]
            if self.min is not None and value < self.min or self.max is not None and value > self.max:
                lo = "-inf" if self.min is None else f"{self.min:g}"
                hi = "inf" if self.max is None else f"{self.max:g}"
                return [f"input {name}={value:g}{' ' + self.unit if self.unit else ''} is outside [{lo}, {hi}]"]
        return []


@dataclass
class Step:
    id: str
    action: str                              # "<adapter>.<action>", e.g. "freecad.macro"
    title: str = ""
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    after: list[str] = field(default_factory=list)          # steps that must have passed first
    requires: list[str] = field(default_factory=list)       # tools that must be installed ("freecad", "openfoam", ...)
    expects: list[ArtifactSpec] = field(default_factory=list)
    confirm: list[str] = field(default_factory=list)        # CONFIRM_CATEGORIES this step falls into
    timeout_s: float = 1800.0

    @property
    def adapter(self) -> str:
        return self.action.split(".", 1)[0]


@dataclass
class Criterion:
    """An acceptance criterion on the result, evaluated by the ``verify.criteria`` action."""
    id: str
    kind: str                                # mass_balance | pressure_drop | converged | artifact
    title: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    id: str
    title: str
    steps: list[Step]
    description: str = ""
    application: str = ""
    inputs: dict[str, InputSpec] = field(default_factory=dict)
    criteria: list[Criterion] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    source: str | None = None                # file it was loaded from

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str | None = None) -> Workflow:
        if not isinstance(d, dict):
            raise WorkflowError(["a workflow is a mapping with id, title and steps"])
        errors: list[str] = []
        steps = []
        for i, s in enumerate(d.get("steps") or []):
            if not isinstance(s, dict) or "id" not in s or "action" not in s:
                errors.append(f"step #{i + 1} needs an id and an action")
                continue
            s = dict(s)
            s["expects"] = [ArtifactSpec.from_any(a) for a in s.get("expects") or []]
            try:
                steps.append(Step(**s))
            except TypeError as e:
                errors.append(f"step {s.get('id')}: {e}")
        try:
            inputs = {k: InputSpec(**v) if isinstance(v, dict) else InputSpec(default=v)
                      for k, v in (d.get("inputs") or {}).items()}
            criteria = [Criterion(**c) for c in d.get("criteria") or []]
        except TypeError as e:
            errors.append(str(e))
            inputs, criteria = {}, []
        if errors:
            raise WorkflowError(errors)
        wf = cls(id=str(d.get("id", "")), title=str(d.get("title", "")), steps=steps,
                 description=d.get("description", ""), application=d.get("application", ""), inputs=inputs,
                 criteria=criteria, limits=list(d.get("limits") or []), source=source)
        wf.validate(raise_=True)
        return wf

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml
            data = yaml.safe_load(text)
        return cls.from_dict(data, source=str(path.resolve()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source")
        return d

    # ------------------------------------------------------------------ checks
    def validate(self, known_actions: set[str] | None = None, raise_: bool = False) -> list[str]:
        """Every structural problem: ids, unknown actions or confirmation categories, ``after`` pointing at a later
        or missing step, templates that refer to unknown inputs or steps, duplicated criteria."""
        errors = []
        if not _ID.match(self.id or ""):
            errors.append(f"workflow id {self.id!r} must be lower_snake_case")
        if not self.title:
            errors.append("workflow needs a title")
        if not self.steps:
            errors.append("workflow has no steps")
        seen: set[str] = set()
        for s in self.steps:
            if not _ID.match(s.id):
                errors.append(f"step id {s.id!r} must be lower_snake_case")
            if s.id in seen:
                errors.append(f"step id {s.id!r} is used twice")
            if "." not in s.action:
                errors.append(f"step {s.id}: action {s.action!r} must be <adapter>.<action>")
            elif known_actions is not None and s.action not in known_actions:
                errors.append(f"step {s.id}: unknown action {s.action!r}")
            for a in s.after:
                if a not in seen:
                    errors.append(f"step {s.id}: 'after: {a}' must name an earlier step")
            for c in s.confirm:
                if c not in CONFIRM_CATEGORIES:
                    errors.append(f"step {s.id}: confirmation category {c!r} is not one of {sorted(CONFIRM_CATEGORIES)}")
            for ref in _refs(s.params) + _refs([a.path for a in s.expects]):
                root, *rest = ref.split(".")
                if root == "inputs" and (not rest or rest[0] not in self.inputs):
                    errors.append(f"step {s.id}: template {{{ref}}} names an unknown input")
                if root == "steps" and (len(rest) < 2 or rest[0] not in seen):
                    errors.append(f"step {s.id}: template {{{ref}}} must name an earlier step and one of its outputs")
            seen.add(s.id)
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            errors.append("criterion ids must be unique")
        if raise_ and errors:
            raise WorkflowError(errors)
        return errors

    def resolve_inputs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Defaults merged with ``overrides``; unknown names and values out of bounds raise :class:`WorkflowError`."""
        overrides = dict(overrides or {})
        errors = [f"unknown input {k!r}; this workflow takes {sorted(self.inputs) or 'none'}"
                  for k in overrides if k not in self.inputs]
        values = {k: overrides.get(k, spec.default) for k, spec in self.inputs.items()}
        for k, v in values.items():
            errors += self.inputs[k].check(k, v)
        if errors:
            raise WorkflowError(errors)
        return values

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    @property
    def folder(self) -> Path | None:
        return Path(self.source).parent if self.source else None


def _refs(v: Any) -> list[str]:
    if isinstance(v, str):
        return _TEMPLATE.findall(v)
    if isinstance(v, dict):
        return [r for x in v.values() for r in _refs(x)]
    if isinstance(v, (list, tuple)):
        return [r for x in v for r in _refs(x)]
    return []


def _lookup(ref: str, ctx: dict[str, Any]) -> Any:
    cur: Any = ctx
    for part in ref.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"template {{{ref}}} has no value (missing {part!r})")
    return cur


def render(value: Any, ctx: dict[str, Any]) -> Any:
    """Fill the templates of ``value`` (str, list or dict, recursively) from ``ctx`` = {"inputs", "steps",
    "workdir", "example"}. A string that is exactly one template takes the referenced value as is."""
    if isinstance(value, str):
        m = _TEMPLATE.fullmatch(value)
        if m:
            return _lookup(m.group(1), ctx)
        return _TEMPLATE.sub(lambda mm: str(_lookup(mm.group(1), ctx)), value)
    if isinstance(value, dict):
        return {k: render(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, ctx) for v in value]
    return value
