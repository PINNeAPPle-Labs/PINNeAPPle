"""Built-in form templates. Add one by writing a module that builds a ``FormSpec`` and registering it here, or ship it
as JSON and load it with ``FormSpec.load``."""
from __future__ import annotations

from typing import Callable, Dict, List

from ..spec import FormSpec

__all__ = ["get_spec", "list_specs"]


def _load(module: str) -> Callable[[], FormSpec]:
    def get() -> FormSpec:
        import importlib
        return importlib.import_module(f"{__name__}.{module}").SPEC
    return get


# order = order in pickers
_REGISTRY: Dict[str, Callable[[], FormSpec]] = {
    "asme_u-dr-1": _load("asme_udr1"), "psv": _load("psv"), "shell_tube": _load("shell_tube"),
    "tank": _load("tank"), "pump": _load("pump"),
}


def get_spec(spec_id: str) -> FormSpec:
    try:
        return _REGISTRY[spec_id]()
    except KeyError:
        raise KeyError(f"unknown form template {spec_id!r}; built-in: {sorted(_REGISTRY)}") from None


def list_specs() -> List[Dict[str, object]]:
    out = []
    for k, f in _REGISTRY.items():
        s = f()
        out.append({"id": k, "title": s.title, "standard": s.standard, "summary": s.summary, "inputs": list(s.inputs),
                    "output": s.output, "items": len(s.fields), "required": sum(x.required for x in s.fields),
                    "tables": [t.label for t in s.tables]})
    return out
