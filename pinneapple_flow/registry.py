"""Node types: typed ports, parameter schemas (rendered as forms by any GUI), and the library call each one makes.

A node type is a :class:`NodeSpec` registered in :data:`REGISTRY`. Only registered types run; a flow document names
types and gives parameter values, never code::

    @node("solver.fd", "Finite differences", category="Solvers",
          inputs=[Port("problem", "problem")], outputs=[Port("solution", "solution"), Port("metrics", "metrics")],
          params=[Param("n", "int", 64, minimum=8, maximum=512, help="nodes per axis")],
          calls="pinneapple_simulation.numerical_solvers.symbolic_fd.solve_fd")
    def fd(ctx, inputs, params):
        ...
        return {"solution": ctx.save_npz("solution", ...), "metrics": {...}}

``params_schema()`` is a JSON Schema per node (the GUI generates its form from it); ``check_params`` validates values.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import PORT_TYPES

PARAM_KINDS = ("int", "float", "bool", "str", "enum", "text", "equations", "list", "dict", "expression")


@dataclass
class Port:
    name: str
    type: str
    required: bool = True
    multiple: bool = False            # an input that accepts several edges (values arrive as a list)
    help: str = ""

    def __post_init__(self):
        if self.type not in PORT_TYPES:
            raise ValueError(f"unknown port type {self.type!r}")


@dataclass
class Param:
    name: str
    kind: str
    default: Any = None
    help: str = ""
    choices: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    max_length: int = 4000
    advanced: bool = False
    label: str = ""

    def __post_init__(self):
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"unknown parameter kind {self.kind!r}")

    def json_schema(self) -> dict[str, Any]:
        k = self.kind
        s: dict[str, Any] = {"title": self.label or self.name.replace("_", " "), "description": self.help,
                             "x-kind": k}
        if self.default is not None:
            s["default"] = self.default
        if self.advanced:
            s["x-advanced"] = True
        if k == "int":
            s["type"] = "integer"
        elif k == "float":
            s["type"] = "number"
        elif k == "bool":
            s["type"] = "boolean"
        elif k in ("str", "text", "expression"):
            s["type"] = "string"
            s["maxLength"] = self.max_length
        elif k == "enum":
            s["enum"] = list(self.choices or [])
        elif k in ("list", "equations"):
            s["type"] = "array"
        elif k == "dict":
            s["type"] = "object"
        if self.minimum is not None:
            s["minimum"] = self.minimum
        if self.maximum is not None:
            s["maximum"] = self.maximum
        return s

    def check(self, v: Any) -> str | None:
        k = self.kind
        if k == "int" and (isinstance(v, bool) or not isinstance(v, int)):
            if not (isinstance(v, float) and v.is_integer()):
                return f"{self.name} must be an integer"
        if k == "float" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            return f"{self.name} must be a number"
        if k == "bool" and not isinstance(v, bool):
            return f"{self.name} must be true or false"
        if k in ("str", "text", "expression") and not isinstance(v, str):
            return f"{self.name} must be a string"
        if k in ("str", "text", "expression") and len(v) > self.max_length:
            return f"{self.name} is longer than {self.max_length} characters"
        if k == "enum" and v not in (self.choices or []):
            return f"{self.name} must be one of {self.choices}"
        if k in ("list", "equations") and not isinstance(v, list):
            return f"{self.name} must be a list"
        if k == "dict" and not isinstance(v, dict):
            return f"{self.name} must be an object"
        if k in ("int", "float") and isinstance(v, (int, float)):
            if self.minimum is not None and v < self.minimum:
                return f"{self.name} must be >= {self.minimum}"
            if self.maximum is not None and v > self.maximum:
                return f"{self.name} must be <= {self.maximum}"
        return None


@dataclass
class NodeSpec:
    type: str
    title: str
    category: str
    fn: Callable[..., dict[str, Any]]
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    description: str = ""
    calls: str = ""                                  # the library function(s) this node wraps
    requires: tuple[str, ...] = ()                   # importable modules it needs (availability check)
    available_fn: Callable[[], tuple[bool, str]] | None = None
    tags: tuple[str, ...] = ()                       # search words; "section:<id>" places it in the palette

    def input(self, name: str) -> Port | None:
        return next((p for p in self.inputs if p.name == name), None)

    def output(self, name: str) -> Port | None:
        return next((p for p in self.outputs if p.name == name), None)

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolved(self, params: dict[str, Any]) -> dict[str, Any]:
        return {**self.defaults(), **(params or {})}

    def check_params(self, params: dict[str, Any]) -> list[tuple[str, str]]:
        out = []
        known = {p.name for p in self.params}
        for k in params:
            if k not in known:
                out.append((k, f"unknown parameter {k!r} for {self.type}"))
        for p in self.params:
            if p.name in params and params[p.name] is not None:
                msg = p.check(params[p.name])
                if msg:
                    out.append((p.name, msg))
        return out

    def available(self) -> tuple[bool, str]:
        for mod in self.requires:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001 - any import failure means unavailable
                return False, f"needs {mod} ({type(exc).__name__}: {exc})"
        if self.available_fn is not None:
            return self.available_fn()
        return True, ""

    def params_schema(self) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": False,
                "properties": {p.name: p.json_schema() for p in self.params}}

    def to_dict(self) -> dict[str, Any]:
        ok, why = self.available()
        return {"type": self.type, "title": self.title, "category": self.category, "description": self.description,
                "calls": self.calls, "inputs": [asdict(p) for p in self.inputs],
                "outputs": [asdict(p) for p in self.outputs], "params": [asdict(p) for p in self.params],
                "params_schema": self.params_schema(), "available": ok, "unavailable_reason": why,
                "tags": list(self.tags), "palette": palette_entry(self, (ok, why))}


def palette_entry(spec: NodeSpec, avail: tuple[bool, str]) -> dict[str, Any]:
    from .palette import entry
    return entry(spec, avail)


class Registry:
    def __init__(self):
        self._specs: dict[str, NodeSpec] = {}

    def register(self, spec: NodeSpec) -> NodeSpec:
        if spec.type in self._specs:
            raise ValueError(f"node type {spec.type!r} is already registered")
        self._specs[spec.type] = spec
        return spec

    def register_custom(self, spec: NodeSpec) -> NodeSpec:
        """Register (or replace) a user-defined node type; only ``custom.*`` types can be replaced."""
        if not spec.type.startswith("custom."):
            raise ValueError("user-defined node types are named custom.<name>")
        _load_builtin()
        self._specs[spec.type] = spec
        return spec

    def unregister_custom(self, type_: str) -> None:
        if type_.startswith("custom."):
            self._specs.pop(type_, None)

    def get(self, type_: str) -> NodeSpec | None:
        _load_builtin()
        return self._specs.get(type_)

    def __getitem__(self, type_: str) -> NodeSpec:
        spec = self.get(type_)
        if spec is None:
            raise KeyError(f"unknown node type {type_!r}")
        return spec

    def __contains__(self, type_: str) -> bool:
        return self.get(type_) is not None

    def types(self) -> list[str]:
        _load_builtin()
        return sorted(self._specs)

    def specs(self) -> list[NodeSpec]:
        _load_builtin()
        return [self._specs[k] for k in sorted(self._specs)]

    def catalog(self) -> list[dict[str, Any]]:
        """Every node type as JSON (ports, params schema, availability) for a GUI palette."""
        return [s.to_dict() for s in self.specs()]


REGISTRY = Registry()
_LOADED = False


def _load_builtin():
    global _LOADED
    if not _LOADED:
        _LOADED = True
        importlib.import_module("pinneapple_flow.nodes")


def node(type_: str, title: str, *, category: str, inputs=(), outputs=(), params=(), description: str = "",
         calls: str = "", requires: tuple[str, ...] = (), available: Callable[[], tuple[bool, str]] | None = None,
         tags: tuple[str, ...] = (), registry: Registry | None = None):
    """Decorator registering ``fn(ctx, inputs, params) -> {output port: value}`` as a node type."""
    def deco(fn):
        (registry or REGISTRY).register(NodeSpec(type=type_, title=title, category=category, fn=fn,
                                                 inputs=list(inputs), outputs=list(outputs), params=list(params),
                                                 description=description or (fn.__doc__ or "").strip(),
                                                 calls=calls, requires=tuple(requires), available_fn=available,
                                                 tags=tuple(tags)))
        return fn
    return deco
