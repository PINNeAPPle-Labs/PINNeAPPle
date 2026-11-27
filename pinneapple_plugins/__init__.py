"""Plugins: third-party packages that register a solver, a model, a benchmark or an adapter without a change here.

The three parts, and which issue they answer:

=================================  =================================================================================
:mod:`.contract`                   what an installed package must implement, and how its contract version is checked
:mod:`.discovery`                  finding installed plugins by the ``pinneapple.plugins`` entry-point group (#100)
:mod:`.manager`                    loading, recording and diagnosing them: list, inspect, doctor, enable, disable (#1474)
=================================  =================================================================================

    import pinneapple as pp

    pp.plugins.load()                                   # discover + initialize (pp.list_models() does this too)
    pp.plugins.names()                                  # ['openfoam', 'my-solver']
    pp.plugins.find(capability="solver", domain="cfd")  # which installed plugin can do this
    pp.plugins.inspect("openfoam")                      # state, source, contract, why it is broken
    pp.plugins.doctor()                                 # the per-plugin checks, in order

What loading a plugin does and does not guarantee is stated once, in :mod:`pinneapple_plugins.policy`, printed by
``pinneapple plugins policy`` and quoted sentence for sentence by ``docs/core_concepts/plugins.md``: the declarations
are checked (dependencies, contract version, the registrations it makes, where it came from), and there is no
isolation, no resource limit, no enforcement of the declared permissions and no revocation beyond disable or uninstall
(#1475).

``activate()`` is what the library calls the first time a registry is listed. It runs **once per process**, imports
only plugins that are enabled, and records every failure instead of raising: a broken plugin never takes down
``import pinneapple`` or another plugin. Setting ``PINNEAPPLE_PLUGINS=0`` turns discovery off for the process.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from .contract import (
    CONTRACT_HISTORY,
    CONTRACT_VERSION,
    PERMISSIONS,
    IncompatibleContract,
    InvalidDeclaration,
    MissingDependency,
    Plugin,
    PluginContext,
    PluginError,
    check_declaration,
    contract_compatible,
    satisfies,
)
from .discovery import ENTRY_GROUP, Entry, discover
from .manager import (
    STATES,
    AmbiguousCapability,
    ConfigSchema,
    Match,
    PluginManager,
    PluginRecord,
    UnknownPlugin,
    config_path,
    load_config,
    registration_problem,
    save_config,
)
from .policy import CHECKED, NOT_PROVIDED, ONE_SENTENCE, RECORDED, policy_dict, policy_text

__all__ = [
    "CHECKED",
    "CONTRACT_VERSION",
    "CONTRACT_HISTORY",
    "NOT_PROVIDED",
    "ONE_SENTENCE",
    "RECORDED",
    "ENTRY_GROUP",
    "PERMISSIONS",
    "STATES",
    "AmbiguousCapability",
    "ConfigSchema",
    "Entry",
    "IncompatibleContract",
    "InvalidDeclaration",
    "Match",
    "MissingDependency",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginManager",
    "PluginRecord",
    "UnknownPlugin",
    "activate",
    "active",
    "capabilities",
    "check_declaration",
    "config_path",
    "contract_compatible",
    "deactivate",
    "disable",
    "discover",
    "doctor",
    "enable",
    "find",
    "find_one",
    "inspect",
    "load",
    "load_config",
    "names",
    "policy_dict",
    "policy_text",
    "registration_problem",
    "records",
    "save_config",
    "satisfies",
    "shutdown",
]

_ACTIVE: PluginManager | None = None

_OFF = ("0", "false", "no", "off")


def _disabled_by_env() -> bool:
    return os.environ.get("PINNEAPPLE_PLUGINS", "").strip().lower() in _OFF


def activate(*, entries: Iterable[Entry] | None = None, config: str | os.PathLike | None = None,
             context: PluginContext | None = None, log: Any = None, force: bool = False) -> PluginManager:
    """Discover and initialize the plugins, once per process, and return the manager.

    ``PINNEAPPLE_PLUGINS=0`` disables discovery for the process (an empty manager: nothing is imported and nothing is
    listed). Loading never raises for a plugin's sake — every failure is in the returned records.
    """
    global _ACTIVE
    if _ACTIVE is not None and not force:
        return _ACTIVE
    if _disabled_by_env() and entries is None:
        _ACTIVE = PluginManager(entries=[], config=config, context=context, log=log)
        _ACTIVE.load_all()
        return _ACTIVE
    manager = PluginManager(entries=entries, config=config, context=context, log=log)
    try:
        manager.load_all()
    except Exception as exc:                            # noqa: BLE001 - discovery itself must not break the library
        manager.log(f"plugin discovery failed: {type(exc).__name__}: {exc}")
    _ACTIVE = manager
    return manager


def active() -> PluginManager | None:
    """The manager this process is using, if activation happened."""
    return _ACTIVE


def deactivate() -> None:
    """Shut every initialized plugin down and forget the manager (tests, and a process that is finishing)."""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.shutdown_all()
    _ACTIVE = None


def load(**kwargs: Any) -> PluginManager:
    """``pp.plugins.load()``: activate (or return the manager already active)."""
    return activate(**kwargs)


def shutdown() -> None:
    """``pp.plugins.shutdown()``: ``shutdown()`` on every plugin that was initialized. Safe to call twice."""
    if _ACTIVE is not None:
        _ACTIVE.shutdown_all()


# -- the questions, answered by the active manager ---------------------------------------------------------------
def _manager() -> PluginManager:
    return activate()


def names() -> list[str]:
    """The installed plugins, by name (no import of any of them beyond what activation already did)."""
    return _manager().names()


def records() -> dict[str, PluginRecord]:
    return _manager().records()


def inspect(name: str) -> PluginRecord:
    return _manager().inspect(name)


def doctor(name: str | None = None) -> list[PluginRecord]:
    return _manager().doctor(name)


def capabilities() -> dict[str, list[str]]:
    return _manager().capabilities()


def find(capability: str | None = None, **query: Any) -> list[Match]:
    return _manager().find(capability, **query)


def find_one(capability: str | None = None, **query: Any) -> Match:
    return _manager().find_one(capability, **query)


def enable(name: str) -> Any:
    return _manager().enable(name)


def disable(name: str) -> Any:
    return _manager().disable(name)


def provenance() -> list[dict[str, Any]]:
    """The plugin set of this process, for a run's provenance (#1475): name, version, distribution and where the
    entry point pointed, one dict per ready plugin."""
    out = []
    for record in _manager().records().values():
        if record.state == "ready":
            out.append({"name": record.name, "version": record.version, "contract_version": record.contract_version,
                        "source": record.source, "entry_point": record.entry_point})
    return sorted(out, key=lambda d: d["name"])


def summary() -> Mapping[str, int]:
    """How many plugins are in each state, for a one-line report."""
    counts = {state: 0 for state in STATES}
    for record in _manager().records().values():
        counts[record.state] = counts.get(record.state, 0) + 1
    return counts
