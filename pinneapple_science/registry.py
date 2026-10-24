"""The experiments of :mod:`pinneapple_science`, by name (modules are imported on first use)."""
from __future__ import annotations

import importlib
from typing import Any

from .core import Experiment, Result

#: modules that define experiments (each class with a ``name`` registers itself through :func:`register`)
MODULES = ("pinneapple_science.em.statics", "pinneapple_science.em.induction", "pinneapple_science.em.waves",
           "pinneapple_science.em.bench", "pinneapple_science.em.research")

_REGISTRY: dict[str, type[Experiment]] = {}
_loaded = False

__all__ = ["register", "get", "names", "experiments", "run", "MODULES"]


def register(cls: type[Experiment]) -> type[Experiment]:
    """Class decorator: add an experiment to the catalogue."""
    if not cls.name:
        raise ValueError("an experiment needs a name")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"experiment {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def _load() -> None:
    global _loaded
    if not _loaded:
        for m in MODULES:
            importlib.import_module(m)
        _loaded = True


def get(name: str) -> Experiment:
    _load()
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown experiment {name!r}; known: {', '.join(sorted(_REGISTRY))}") from None


def names() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def experiments(area: str | None = None, *, level: str | None = None, kind: str | None = None,
                dim: str | None = None) -> list[dict[str, Any]]:
    """Cards of the experiments matching every filter given (``area`` matches the area id or its prefix)."""
    _load()
    out = []
    for n in sorted(_REGISTRY):
        e = _REGISTRY[n]
        if area and not (e.area == area or e.area.startswith(area + ".") or e.area.split(".")[-1] == area):
            continue
        if level and e.level != level or kind and e.kind != kind or dim and dim not in e.dims:
            continue
        out.append(e().card())
    return out


def run(name: str, dim: str | None = None, *, quick: bool = False, **params) -> Result:
    """Run one experiment in one dimension (its first by default) with parameters over its defaults."""
    return get(name).run(dim, quick=quick, **params)
