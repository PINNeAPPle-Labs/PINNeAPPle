"""Experiment specification and registry."""
from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any

_REGISTRY: dict[str, type[Experiment]] = {}
_BUILTINS = "pinneapple_lab.experiments"


class Experiment:
    """Subclass, set ``name`` and ``params`` (defaults), and implement ``run(ctx)``.

    Attributes
    ----------
    name: unique id (used in run ids and folders).
    version: bump when the experiment's code changes meaning; it is part of the run id, so old results stay.
    description, tags: for the catalogue.
    params: default parameters (JSON-serialisable values).
    space: optional sampling ranges for ``sweep(samples=...)``: {param: (low, high)} for floats/ints,
           {param: [choices]} for categorical, {param: ("log", low, high)} for log-uniform.
    """

    name: str = ""
    version: str = "1"
    description: str = ""
    tags: list[str] = []
    params: dict[str, Any] = {}
    space: dict[str, Any] = {}

    def run(self, ctx) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    @classmethod
    def resolve_params(cls, params: dict[str, Any] | None = None) -> dict[str, Any]:
        unknown = set(params or {}) - set(cls.params)
        if unknown:
            raise KeyError(f"{cls.name}: unknown parameters {sorted(unknown)}; known: {sorted(cls.params)}")
        out = dict(cls.params)
        out.update(params or {})
        return out

    @classmethod
    def run_id(cls, params: dict[str, Any]) -> str:
        blob = json.dumps({"v": cls.version, "p": params}, sort_keys=True, default=str)
        return f"{cls.name}-{hashlib.sha1(blob.encode()).hexdigest()[:12]}"


def register(cls: type[Experiment]) -> type[Experiment]:
    """Class decorator: make the experiment available by name to the runner and the CLI."""
    if not cls.name:
        raise ValueError("an Experiment needs a name")
    prev = _REGISTRY.get(cls.name)
    if prev is not None and prev.__qualname__ != cls.__qualname__:
        raise ValueError(f"experiment name '{cls.name}' already registered by {prev.__module__}.{prev.__qualname__}")
    _REGISTRY[cls.name] = cls
    return cls


def _load_builtins() -> None:
    importlib.import_module(_BUILTINS)


def get(name: str) -> type[Experiment]:
    if name not in _REGISTRY:
        _load_builtins()
    if name not in _REGISTRY and ":" in name:          # "package.module:Class" for experiments outside the package
        mod, _, attr = name.partition(":")
        cls = getattr(importlib.import_module(mod), attr)
        return register(cls)
    if name not in _REGISTRY:
        raise KeyError(f"unknown experiment '{name}'; available: {available()}")
    return _REGISTRY[name]


def available() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)
