"""``pp.registry``: one registry protocol for every extensible object type.

The library grew a registry per object type, each with its own shape: presets are a module dict behind a decorator,
solvers are ``solving._METHODS`` with a ``kind`` label, models are ``ModelRegistry`` with a family and a spec,
datasets are a ``DatasetRegistry`` of ``DatasetInfo`` plus a loader, materials are the ``MATERIALS`` dict. This module
gives them one protocol, so a user (and a plugin) learns it once::

    import pinneapple as pp

    pp.register(MySolver, kind="solver", name="my_solver")   # or @pp.register(kind="solver")
    pp.registered("solver")                                  # every name, the ones the library ships included
    pp.resolve("solver", "my_solver")                        # the object itself
    pp.unregister("solver", "my_solver")

Nine kinds: ``physics`` (problem presets), ``model``, ``solver``, ``operator``, ``dataset``, ``benchmark``,
``geometry``, ``material`` and ``optimizer``. Each kind declares what it accepts, once, in :data:`KINDS`; registering
something of the wrong type raises ``WrongType`` naming the kind and what it takes, instead of failing later inside a
solve.

Where a kind already had a home in the library, registering through here also reaches that home — a ``solver`` becomes
callable through ``pp.solve(problem, "my_solver")``, a ``physics`` preset through ``pp.get_preset("my_problem")``, a
``model`` through ``pp.list_models()``, a ``dataset`` through ``DatasetRegistry``, a ``material`` through
``MATERIALS``. ``operator``, ``benchmark``, ``geometry`` and ``optimizer`` had no registry of their own: this module is
where they live, and :func:`resolve` is how the rest of the library (and a plugin) finds them.

Inference is deliberately conservative: ``pp.register(obj)`` without ``kind`` only works when the type settles it — a
``torch.nn.Module`` subclass is a model, a ``Material`` is a material, a ``Geometry`` is a geometry, a ``DatasetInfo``
is a dataset. A plain callable could be a solver, an operator or a benchmark, so it raises ``AmbiguousKind`` and asks
for ``kind=``.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Registry", "KINDS", "kinds", "register", "unregister", "registered", "resolve", "resolve_kind", "RegistryError",
           "UnknownKind", "UnknownName", "AlreadyRegistered", "WrongType", "AmbiguousKind"]


class RegistryError(Exception):
    """Anything the registry refuses."""


class UnknownKind(RegistryError, KeyError):
    """``kind`` is not one of :data:`KINDS`."""


class UnknownName(RegistryError, KeyError):
    """No object of that kind is registered under that name."""


class AlreadyRegistered(RegistryError, ValueError):
    """The name is taken in this kind (``replace=True`` overwrites)."""


class WrongType(RegistryError, TypeError):
    """The object does not satisfy what the kind accepts."""


class AmbiguousKind(RegistryError, TypeError):
    """The type does not settle the kind; pass ``kind=`` explicitly."""


# --------------------------------------------------------------------------- the protocol
@dataclass
class Registry:
    """A ``name -> object`` map with one validation rule, one error vocabulary and an optional bridge to a legacy home.

    ``accepts`` is the predicate of the kind, ``accepts_doc`` says in words what it accepts, and ``bridge`` (when the
    library already had a home for the kind) is called on the first registration of a name to give it to that home too.
    """
    kind: str
    accepts: Callable[[Any], bool] | None = None
    accepts_doc: str = "anything"
    entries: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    home: str = ""
    _names: Callable[[], list[str]] | None = None        # names the legacy home already has
    _get: Callable[[str], Any] | None = None             # object from the legacy home
    _bridge: Callable[[str, Any, Mapping[str, Any]], Any] | None = None
    _drop: Callable[[str], Any] | None = None            # remove from the legacy home

    # -- registration -------------------------------------------------------
    def register(self, name: str, obj: Any, *, replace: bool = False, **meta: Any) -> Any:
        name = _clean(name, self.kind)
        if self.accepts is not None and not self.accepts(obj):
            raise WrongType(f"{self.kind} expects {self.accepts_doc}, got {type(obj).__name__}")
        if name in self.entries and not replace:
            raise AlreadyRegistered(f"{self.kind} {name!r} is already registered; pass replace=True to overwrite")
        self.entries[name] = obj
        self.meta[name] = dict(meta)
        if self._bridge is not None:
            try:
                bridged = self._bridge(name, obj, meta)
            except Exception as exc:                    # noqa: BLE001 - a missing optional package is reported, not fatal
                bridged = f"unavailable: {type(exc).__name__}: {exc}"
            if bridged is not None:
                self.meta[name]["bridge"] = bridged
        return obj

    def get(self, name: str) -> Any:
        name = _key(name)
        if name in self.entries:
            return self.entries[name]
        if self._get is not None:
            try:
                return self._get(name)
            except KeyError as exc:
                raise UnknownName(f"{self.kind} {name!r} is not registered; see pp.registered({self.kind!r})") from exc
        raise UnknownName(f"{self.kind} {name!r} is not registered; see pp.registered({self.kind!r})")

    def list(self) -> list[str]:
        names = set(self.entries)
        if self._names is not None:
            try:
                names.update(self._names())
            except Exception:                           # noqa: BLE001 - the home may need an optional package
                pass
        return sorted(names)

    def unregister(self, name: str) -> Any:
        name = _key(name)
        dropped = self.entries.pop(name, _MISSING)
        self.meta.pop(name, None)
        if self._drop is not None:
            try:
                self._drop(name)
            except Exception:                           # noqa: BLE001 - the home may not have it, or may need a package
                pass
        if dropped is _MISSING:
            raise UnknownName(f"{self.kind} {name!r} is not registered")
        return dropped

    def describe(self, name: str) -> dict[str, Any]:
        return {"name": name, "kind": self.kind, "home": self.home, **self.meta.get(_key(name), {})}

    # -- dunder -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _key(name) in self.list()

    def __len__(self) -> int:
        return len(self.list())

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self)} entries, accepts {self.accepts_doc})"


_MISSING = object()


def _key(name: Any) -> str:
    return str(name).strip().lower()


def _clean(name: Any, kind: str) -> str:
    key = _key(name)
    if not key:
        raise RegistryError(f"{kind}: a name is required")
    return key


# --------------------------------------------------------------------------- the kinds
def _is_callable(obj: Any) -> bool:
    return callable(obj)


def _is_model_class(obj: Any) -> bool:
    if not isinstance(obj, type):
        return False
    try:
        from torch import nn

        return issubclass(obj, nn.Module)
    except ImportError:                                 # pragma: no cover - torch is a hard dependency
        return False


def _is_geometry(obj: Any) -> bool:
    from .geometry import Geometry

    return isinstance(obj, Geometry) or (isinstance(obj, type) and issubclass(obj, Geometry))


def _is_material(obj: Any) -> bool:
    try:
        from pinneapple_design.intent.materials import Material
    except ImportError:                                 # pragma: no cover - the package ships with the library
        return False
    return isinstance(obj, Material)


def _is_dataset_info(obj: Any) -> bool:
    try:
        from pinneapple_data.datasets.registry import DatasetInfo
    except ImportError:                                 # pragma: no cover
        return False
    return isinstance(obj, DatasetInfo)


def _is_optimizer(obj: Any) -> bool:
    """A ``PhysicsOptimizer`` subclass, and only that: a plain minimiser callable has to say ``kind="optimizer"``."""
    from .optim import PhysicsOptimizer

    return isinstance(obj, type) and issubclass(obj, PhysicsOptimizer)


# -- bridges to the homes the library already had ----------------------------------------
def _bridge_preset(name: str, obj: Any, meta: Mapping[str, Any]) -> str:
    from pinneapple_physics.pde_environment.presets.registry import _REGISTRY, register_preset

    if name in _REGISTRY:
        return "already in pinneapple_physics presets"
    register_preset(name)(obj)
    return "pinneapple_physics presets (pp.get_preset)"


def _drop_preset(name: str) -> None:
    from pinneapple_physics.pde_environment.presets.registry import _REGISTRY

    _REGISTRY.pop(name, None)


def _preset_names() -> list[str]:
    from pinneapple_physics.pde_environment.presets.registry import _REGISTRY

    return list(_REGISTRY)


def _get_preset_obj(name: str) -> Any:
    from pinneapple_physics.pde_environment.presets.registry import _REGISTRY

    return _REGISTRY[name]


def _bridge_method(name: str, obj: Any, meta: Mapping[str, Any]) -> str:
    from pinneapple_physics.solving import _METHODS, register_method

    if name in _METHODS:
        return "already a method of pp.solve"
    register_method(name, str(meta.get("family", meta.get("method_kind", "custom"))))(obj)
    return "pp.solve methods"


def _drop_method(name: str) -> None:
    from pinneapple_physics.solving import _KINDS, _METHODS

    _METHODS.pop(name, None)
    _KINDS.pop(name, None)


def _method_names() -> list[str]:
    from pinneapple_physics.solving import _METHODS

    return list(_METHODS)


def _get_method(name: str) -> Any:
    from pinneapple_physics.solving import _METHODS

    return _METHODS[name]


def _bridge_model(name: str, obj: Any, meta: Mapping[str, Any]) -> str:
    from pinneapple_neural.architectures.registry import ModelRegistry

    if name in ModelRegistry.list():
        return "already in ModelRegistry"
    ModelRegistry.register(name=name, family=str(meta.get("family", "custom")),
                          description=str(meta.get("description", "")), tags=list(meta.get("tags") or []))(obj)
    return "ModelRegistry (pp.list_models)"


def _drop_model(name: str) -> None:
    from pinneapple_neural.architectures.registry import ModelRegistry

    ModelRegistry._REGISTRY.pop(name, None)


def _model_names() -> list[str]:
    from pinneapple_neural.architectures.registry import ModelRegistry

    return ModelRegistry.list()


def _get_model(name: str) -> Any:
    from pinneapple_neural.architectures.registry import ModelRegistry

    return ModelRegistry.spec(name).cls


def _bridge_dataset(name: str, obj: Any, meta: Mapping[str, Any]) -> str:
    from pinneapple_data.datasets.registry import DatasetRegistry

    loader = meta.get("loader")
    if loader is None:
        raise ValueError("a dataset needs loader=fn: the callable that returns its arrays")
    key = str(getattr(obj, "id", name))
    if key != name:
        raise ValueError(f"a dataset is registered under its DatasetInfo.id ({key!r}), not {name!r}")
    if key in DatasetRegistry.ids():
        return "already in DatasetRegistry"
    DatasetRegistry.register(obj, loader)
    return "DatasetRegistry"


def _drop_dataset(name: str) -> None:
    from pinneapple_data.datasets.registry import DatasetRegistry

    DatasetRegistry._store.pop(name, None)


def _dataset_names() -> list[str]:
    from pinneapple_data.datasets.registry import DatasetRegistry

    return DatasetRegistry.ids()


def _get_dataset(name: str) -> Any:
    from pinneapple_data.datasets.registry import DatasetRegistry

    return DatasetRegistry.get_info(name)


def _bridge_material(name: str, obj: Any, meta: Mapping[str, Any]) -> str:
    from pinneapple_design.intent.materials import MATERIALS

    if name in MATERIALS:
        return "already in MATERIALS"
    MATERIALS[name] = obj
    return "pinneapple_design MATERIALS"


def _drop_material(name: str) -> None:
    from pinneapple_design.intent.materials import MATERIALS

    MATERIALS.pop(name, None)


def _material_names() -> list[str]:
    from pinneapple_design.intent.materials import MATERIALS

    return list(MATERIALS)


def _get_material(name: str) -> Any:
    from pinneapple_design.intent.materials import MATERIALS

    return MATERIALS[name]


KINDS: dict[str, Registry] = {
    "physics": Registry("physics", _is_callable, "a callable preset factory (name -> ProblemSpec)", home="pinneapple_physics presets",
                        _names=_preset_names, _get=_get_preset_obj, _bridge=_bridge_preset, _drop=_drop_preset),
    "model": Registry("model", _is_model_class, "a torch.nn.Module subclass", home="ModelRegistry",
                      _names=_model_names, _get=_get_model, _bridge=_bridge_model, _drop=_drop_model),
    "solver": Registry("solver", _is_callable, "a callable fn(problem, **options) -> Solution", home="pp.solve methods",
                       _names=_method_names, _get=_get_method, _bridge=_bridge_method, _drop=_drop_method),
    "operator": Registry("operator", _is_callable, "a callable (values, ...) -> array", home="this module"),
    "dataset": Registry("dataset", _is_dataset_info, "a DatasetInfo (with loader=fn)", home="DatasetRegistry",
                        _names=_dataset_names, _get=_get_dataset, _bridge=_bridge_dataset, _drop=_drop_dataset),
    "benchmark": Registry("benchmark", _is_callable, "a callable that runs the benchmark", home="this module"),
    "geometry": Registry("geometry", _is_geometry, "a Geometry instance or subclass", home="this module"),
    "material": Registry("material", _is_material, "a Material", home="pinneapple_design MATERIALS",
                         _names=_material_names, _get=_get_material, _bridge=_bridge_material, _drop=_drop_material),
    "optimizer": Registry("optimizer", _is_optimizer, "a PhysicsOptimizer subclass",
                          home="this module"),
}


def kinds() -> list[str]:
    """The kinds this library knows how to register."""
    return sorted(KINDS)


def resolve_kind(kind: str) -> Registry:
    key = _key(kind)
    if key not in KINDS:
        raise UnknownKind(f"unknown kind {kind!r}; known kinds: {kinds()}")
    return KINDS[key]


def _infer(obj: Any) -> str:
    """The kind the object's own type settles, or an honest refusal when it settles nothing."""
    if _is_model_class(obj):
        return "model"
    if _is_material(obj):
        return "material"
    if _is_geometry(obj):
        return "geometry"
    if _is_dataset_info(obj):
        return "dataset"
    if _is_optimizer(obj):
        return "optimizer"
    if _is_callable(obj):
        raise AmbiguousKind(f"{getattr(obj, '__name__', type(obj).__name__)} is a callable, which could be "
                            f"{[k for k in kinds() if KINDS[k].accepts is _is_callable]}; pass kind=...")
    raise UnknownKind(f"cannot tell what kind {type(obj).__name__} is; pass kind=... (known kinds: {kinds()})")


def _name_of(obj: Any, name: str | None) -> str:
    if name:
        return str(name)
    for attr in ("key", "id", "name", "__name__"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    raise RegistryError(f"cannot infer a name for {type(obj).__name__}: pass name=...")


def register(obj: Any = None, *, kind: str | None = None, name: str | None = None, replace: bool = False,
             **meta: Any):
    """Register ``obj`` under ``kind`` (inferred when the type settles it) and return it.

    Works as a decorator too::

        @pp.register(kind="operator", name="my_curl")
        def my_curl(values, mesh=None):
            ...
    """
    if obj is None:
        def decorator(target: Any):
            return register(target, kind=kind, name=name, replace=replace, **meta)
        return decorator
    registry = resolve_kind(kind or _infer(obj))
    return registry.register(_name_of(obj, name), obj, replace=replace, **meta)


def unregister(kind: str, name: str) -> Any:
    """Remove a name from a kind, and from the library home it was bridged to."""
    return resolve_kind(kind).unregister(name)


def registered(kind: str | None = None) -> list[str] | dict[str, list[str]]:
    """Every name of one kind (what the library ships included), or of every kind as ``{kind: names}``."""
    if kind is None:
        return {k: KINDS[k].list() for k in kinds()}
    return resolve_kind(kind).list()


def resolve(kind: str, name: str) -> Any:
    """The object registered under ``name`` for ``kind`` — the bridge reaches what the library already had."""
    return resolve_kind(kind).get(name)


def _module() -> Any:                                   # pragma: no cover - convenience for ``pp.registry``
    return sys.modules[__name__]
