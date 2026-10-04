"""Built-in form templates. Add one by writing a module that builds a ``FormSpec`` and registering it here, or ship it
as JSON and load it with ``FormSpec.load``."""
from __future__ import annotations

from typing import Callable, Dict, List

from ..spec import FormSpec

__all__ = ["get_spec", "list_specs"]


def _udr1() -> FormSpec:
    from .asme_udr1 import SPEC
    return SPEC


_REGISTRY: Dict[str, Callable[[], FormSpec]] = {"asme_u-dr-1": _udr1}


def get_spec(spec_id: str) -> FormSpec:
    try:
        return _REGISTRY[spec_id]()
    except KeyError:
        raise KeyError(f"unknown form template {spec_id!r}; built-in: {sorted(_REGISTRY)}") from None


def list_specs() -> List[Dict[str, str]]:
    return [{"id": k, "title": f().title} for k, f in _REGISTRY.items()]
