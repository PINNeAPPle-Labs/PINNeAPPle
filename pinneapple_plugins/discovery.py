"""Finding the plugins that are installed: the ``pinneapple.plugins`` entry-point group and nothing else.

Discovery reads distribution *metadata*, not code: walking ``importlib.metadata`` costs the import of nothing, so a
broken plugin cannot break discovery, and ``pinneapple plugins list`` can show a plugin whose module raises on import
(#1474). Importing the object is a separate, per-plugin step (:mod:`pinneapple_plugins.manager`).

The group name is part of the contract: a package declares it in its own ``pyproject.toml``::

    [project.entry-points."pinneapple.plugins"]
    openfoam = "pinneapple_openfoam:PLUGIN"

The value resolves to a :class:`pinneapple_plugins.contract.Plugin` instance, a subclass, or a zero-argument
callable that returns one.

Nothing here raises on bad metadata: a distribution whose entry points cannot be read is reported as an entry with
``error`` set, because the diagnostics command has to be able to say so.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ENTRY_GROUP", "Entry", "discover"]

#: The entry-point group a package declares to be a PINNeAPPle plugin.
ENTRY_GROUP = "pinneapple.plugins"


@dataclass(frozen=True)
class Entry:
    """One discovered plugin: where it comes from, without having imported it.

    ``loader`` is how tests hand over a plugin that is not installed (a module in ``tmp_path``, a class defined in the
    test): everything downstream treats the two the same, which is the point — the loading rules are what is under
    test, not setuptools.
    """

    name: str
    value: str = ""
    distribution: str = ""
    version: str = ""
    group: str = ENTRY_GROUP
    error: str = ""
    loader: Callable[[], Any] | None = field(default=None, compare=False, repr=False)

    def load(self) -> Any:
        """Resolve and import the entry point. Raises whatever the import raises — the manager records it."""
        if self.error:
            raise ImportError(self.error)
        if self.loader is not None:
            return self.loader()
        from importlib.metadata import EntryPoint

        return EntryPoint(name=self.name, value=self.value, group=self.group).load()

    @property
    def source(self) -> str:
        """Where it came from, as ``"distribution version"`` (the docs say the same words)."""
        return " ".join(x for x in (self.distribution, self.version) if x) or "unrecorded distribution"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "distribution": self.distribution, "version": self.version,
                "group": self.group, "error": self.error}


def discover(group: str = ENTRY_GROUP) -> list[Entry]:
    """Every installed plugin, in the deterministic order the rest of the library relies on (by name).

    Two distributions declaring the same entry-point name are both kept: the second is recorded with ``error`` set
    naming the first, so ``doctor`` reports a collision instead of one plugin silently winning.
    """
    found: dict[str, Entry] = {}
    for dist in _distributions():
        for ep in _entry_points_of(dist, group):
            name = str(getattr(ep, "name", "") or "")
            if not name:
                continue
            entry = Entry(name=name, value=str(getattr(ep, "value", "") or ""),
                          distribution=_dist_name(dist), version=str(getattr(dist, "version", "") or ""),
                          group=group)
            if name in found:
                first = found[name]
                found[name] = Entry(name=name, value=entry.value, distribution=entry.distribution,
                                    version=entry.version, group=group,
                                    error=f"the entry point name {name!r} is declared by {first.source} and by "
                                          f"{entry.source}; rename one of them")
            else:
                found[name] = entry
    return [found[k] for k in sorted(found)]


def _distributions() -> Iterable[Any]:
    from importlib.metadata import distributions

    try:
        return list(distributions())
    except Exception:                                   # noqa: BLE001 - unreadable metadata must not break discovery
        return []


def _dist_name(dist: Any) -> str:
    meta = getattr(dist, "metadata", None)
    name = (meta.get("Name") if meta is not None and hasattr(meta, "get") else None) or getattr(dist, "name", "")
    return str(name or "")


def _entry_points_of(dist: Any, group: str) -> list[Any]:
    try:
        return [ep for ep in dist.entry_points if str(getattr(ep, "group", "")) == group]
    except Exception:                                   # noqa: BLE001 - one unreadable distribution is not all of them
        return []
