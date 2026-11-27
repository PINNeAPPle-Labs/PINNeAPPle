"""Loading the discovered plugins, recording their state, and diagnosing them.

One object does the three things #1474 needs and #100 depends on: :class:`PluginManager` discovers (metadata only),
loads (import, contract, dependencies, configuration, ``initialize``) and answers questions about each plugin —
``records()``, ``inspect()``, ``doctor()``, ``find()``, ``enable()``/``disable()``.

The rules are the contract's (:mod:`pinneapple_plugins.contract`), applied in this order and stopped at the first
failure that can be explained:

1. the entry point resolves and the module imports;
2. ``contract_version`` is compatible with the running contract;
3. declared dependencies — the library's version, other plugins, importable modules, executables on ``PATH``;
4. required configuration keys are present;
5. ``initialize()`` succeeds (``shutdown()`` is exercised by ``doctor`` and called for every initialized plugin when
   the process ends, including when a later plugin fails).

**A failure is a record, never an exception.** Every method here returns data; the only exceptions raised are the
ones a caller asks for by name (``enable`` of a plugin that is not installed raises :class:`UnknownPlugin`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .contract import (
    CONTRACT_VERSION,
    InvalidDeclaration,
    Plugin,
    PluginContext,
    check_declaration,
    contract_compatible,
    satisfies,
)
from .discovery import ENTRY_GROUP, Entry, discover

__all__ = ["ConfigSchema", "PluginManager", "PluginRecord", "UnknownPlugin", "Match", "AmbiguousCapability",
           "config_path", "load_config", "save_config", "registration_problem"]

#: The states a plugin can be in. ``not_installed`` is the answer to "inspect a name nothing provides".
STATES: tuple[str, ...] = ("ready", "broken", "disabled", "not_installed")

CONFIG_SCHEMA = "pinneapple.plugins/1"


class UnknownPlugin(KeyError):
    """``enable``/``disable`` were asked about a name that no installed distribution declares."""


@dataclass
class Match:
    """One candidate of :meth:`PluginManager.find`: who, and what it said it can do."""

    plugin: str
    capability: str
    values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"plugin": self.plugin, "capability": self.capability, "values": list(self.values)}


class AmbiguousCapability(LookupError):
    """Two plugins declare the same capability at the same rank; the message names both, never picks one silently."""

    def __init__(self, candidates: Sequence[Match], query: Mapping[str, Any]):
        self.candidates = list(candidates)
        names = ", ".join(f"{m.plugin}{dict(values=m.values)}" for m in self.candidates)
        super().__init__(f"{len(self.candidates)} plugins match {dict(query)}: {names}. Nothing was chosen: "
                         f"narrow the query (a domain, a format, a flag) or name the plugin.")


@dataclass
class PluginRecord:
    """What is known about one plugin: identity, where it came from, its state, and the checks that got it there."""

    name: str
    state: str = "not_installed"
    version: str = ""
    contract_version: str = ""
    source: str = ""
    entry_point: str = ""
    reason: str = ""
    stage: str = ""
    traceback: str = ""
    capabilities: dict[str, list[str]] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)
    requires: dict[str, Any] = field(default_factory=dict)
    registers: dict[str, list[str]] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    #: How many checks a complete diagnosis has (import, contract, dependencies, config, initialize, registers).
    CHECKS: tuple[str, ...] = ("import", "contract", "dependencies", "config", "initialize", "registers")

    @property
    def ok(self) -> bool:
        return self.state == "ready"

    @property
    def usable(self) -> bool:
        """Whether a caller may use what this plugin provides (a broken plugin is never usable)."""
        return self.state == "ready"

    def check(self, name: str) -> dict[str, Any] | None:
        return next((c for c in self.checks if c.get("check") == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "version": self.version,
                "contract_version": self.contract_version, "source": self.source, "entry_point": self.entry_point,
                "reason": self.reason, "stage": self.stage, "capabilities": self.capabilities,
                "permissions": self.permissions, "requires": self.requires, "registers": self.registers,
                "checks": self.checks,
                "details": self.details}


# ------------------------------------------------------------------------------------------- the user configuration
def config_path(explicit: str | Path | None = None) -> Path:
    """Where the enabled/disabled list lives: ``--config``, else ``$PINNEAPPLE_PLUGINS_CONFIG``, else
    ``~/.pinneapple/plugins.json``. A file, not a package operation: disabling is the reversible action an operator
    reaches for when a plugin misbehaves in production (#1474)."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("PINNEAPPLE_PLUGINS_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".pinneapple" / "plugins.json"


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the configuration, or an empty one when the file is absent.

    A file that is there and unreadable is an error this raises: silently ignoring it would enable a plugin the
    operator disabled.
    """
    p = Path(path)
    if not p.is_file():
        return {"schema": CONFIG_SCHEMA, "disabled": [], "options": {}}
    data = json.loads(p.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"{p}: the plugins configuration must be a JSON object")
    data.setdefault("schema", CONFIG_SCHEMA)
    data.setdefault("disabled", [])
    data.setdefault("options", {})
    if not isinstance(data["disabled"], list):
        raise ValueError(f"{p}: `disabled` must be a list of plugin names")
    if not isinstance(data["options"], dict):
        raise ValueError(f"{p}: `options` must be an object of plugin name -> its settings")
    return data


def save_config(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Write the configuration back atomically (a half-written file would enable what was disabled)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


class ConfigSchema:
    """Reads the ``disabled``/``options`` sections without rewriting them. Kept tiny on purpose."""

    def __init__(self, data: Mapping[str, Any]):
        self.data = dict(data)

    @property
    def disabled(self) -> list[str]:
        return [str(n) for n in self.data.get("disabled") or []]

    def options_for(self, name: str) -> dict[str, Any]:
        return dict((self.data.get("options") or {}).get(name) or {})


# ----------------------------------------------------------------------------------------------- the manager
class PluginManager:
    """Discovers, loads and diagnoses plugins. Construct once per process; :func:`pinneapple_plugins.activate` does.

    ``entries`` is how a caller (a test, ``doctor --from-file``) supplies the plugin list instead of the installed
    distributions: everything downstream is identical, so the loading rules are what gets exercised.
    """

    def __init__(self, *, entries: Iterable[Entry] | None = None, config: str | Path | None = None,
                 context: PluginContext | None = None, log: Callable[[str], None] | None = None,
                 running_version: str | None = None):
        self._entries: dict[str, Entry] = {e.name: e for e in (entries if entries is not None else discover())}
        self.config_file = config_path(config)
        self.config = load_config(self.config_file)
        self.schema = ConfigSchema(self.config)
        self.log = log or (lambda line: None)
        self.context = context or PluginContext(logger=self.log)
        self._running_version = running_version
        self._records: dict[str, PluginRecord] = {}
        self._objects: dict[str, Any] = {}
        self._loaded = False

    # -- discovery -------------------------------------------------------------------------------------------------
    def discovered(self) -> list[Entry]:
        """The installed plugins, by name, without having imported any of them."""
        return [self._entries[k] for k in sorted(self._entries)]

    def names(self) -> list[str]:
        return sorted(self._entries)

    # -- loading ---------------------------------------------------------------------------------------------------
    def load_all(self, *, force: bool = False) -> dict[str, PluginRecord]:
        """Load every discovered plugin, recording each outcome. Idempotent unless ``force`` (which re-runs the
        checks against a fresh import of the entry points' objects)."""
        if self._loaded and not force:
            return dict(self._records)
        for name in self.names():
            self._records[name] = self._load_one(self._entries[name])
        self._loaded = True
        return dict(self._records)

    def load(self, name: str) -> PluginRecord:
        """Load one plugin by name (loading everything else first is not required — ``requires.plugins`` is checked
        against the discovered set, not against what happened to be initialized)."""
        if name not in self._entries:
            return self._not_installed(name)
        if name in self._records and not self._records[name].state == "broken":
            return self._records[name]
        self._records[name] = self._load_one(self._entries[name])
        return self._records[name]

    def records(self) -> dict[str, PluginRecord]:
        if not self._loaded:
            self.load_all()
        return dict(self._records)

    def inspect(self, name: str) -> PluginRecord:
        """Everything known about one plugin, including the ones that failed and the ones nothing provides."""
        if name in self._records:
            return self._records[name]
        if name not in self._entries:
            return self._not_installed(name)
        return self.load(name)

    def object(self, name: str) -> Any:
        """The initialized plugin, or ``None`` — a caller that wants to know why asks :meth:`inspect`."""
        record = self.inspect(name)
        return self._objects.get(name) if record.state == "ready" else None

    def _not_installed(self, name: str) -> PluginRecord:
        return PluginRecord(name=name, state="not_installed",
                            reason=f"no installed distribution declares a {ENTRY_GROUP!r} entry point named {name!r}"
                                   f" (checked {len(self._entries)} plugin(s): {', '.join(self.names()) or 'none'})")

    def _load_one(self, entry: Entry) -> PluginRecord:
        record = PluginRecord(name=entry.name, entry_point=entry.value, source=entry.source,
                              version=entry.version)
        if entry.name in self.schema.disabled:
            record.state = "disabled"
            record.reason = f"disabled in {self.config_file} (`pinneapple plugins enable {entry.name}` turns it back on)"
            return record

        # 1. the entry point resolves and the module imports ------------------------------------------------------
        try:
            obj = entry.load()
        except Exception as exc:                        # noqa: BLE001 - one broken plugin is a record
            return self._broken(record, "import", f"{type(exc).__name__}: {exc}", traceback_text=_tb())
        record.checks.append(_check("import", True, f"{entry.value} imported"))

        # 2. the contract ------------------------------------------------------------------------------------------
        try:
            declared = check_declaration(obj)
        except InvalidDeclaration as exc:
            return self._broken(record, "contract", str(exc))
        record.version = declared["version"]
        record.contract_version = declared["contract_version"]
        record.capabilities = declared["capabilities"]
        record.permissions = declared["permissions"]
        record.requires = declared["requires"]
        record.registers = declared["registers"]
        ok, why = contract_compatible(record.contract_version, CONTRACT_VERSION)
        if not ok:
            return self._broken(record, "contract", why)
        if record.contract_version != CONTRACT_VERSION:
            record.details["contract_note"] = (f"written against {record.contract_version}, running {CONTRACT_VERSION}"
                                               " (compatible)")
        record.checks.append(_check("contract", True, f"contract {record.contract_version} vs running "
                                                     f"{CONTRACT_VERSION}"))

        # 3. declared dependencies ---------------------------------------------------------------------------------
        problems, details = self._dependencies(record)
        record.details.update(details)
        if problems:
            return self._broken(record, "dependencies", "; ".join(problems))
        record.checks.append(_check("dependencies", True, _summary(details)))

        # 4. required configuration --------------------------------------------------------------------------------
        options = self.schema.options_for(record.name)
        missing = [k for k in (record.requires.get("config") or []) if self._config_value(record.name, k, options)
                   is None]
        if missing:
            return self._broken(record, "config", f"missing required configuration key(s) {missing} for plugin "
                                                  f"{record.name!r}: set them under options.{record.name} in "
                                                  f"{self.config_file} or as "
                                                  f"{', '.join(self._env_name(record.name, k) for k in missing)}")
        record.checks.append(_check("config", True, f"{len(record.requires.get('config') or [])} required key(s) "
                                                   "present"))

        # 5. initialize, watching what it registers ----------------------------------------------------------------
        ctx = replace(self.context, config=options, logger=self.log,
                      paths={**dict(self.context.paths), "plugin": self._data_dir(record.name)})
        registered: dict[str, list[str]] = {}
        try:
            with _watch_registry(registered):
                obj.initialize(ctx)
        except Exception as exc:                        # noqa: BLE001 - the plugin's failure, not the library's
            self._roll_back(record, registered)
            self._shutdown_quietly(record.name, obj)
            return self._broken(record, "initialize", f"{type(exc).__name__}: {exc}", traceback_text=_tb())
        record.details["registered"] = registered
        record.checks.append(_check("initialize", True, "initialized"))

        # 6. the registrations it declared, against the ones it made ------------------------------------------------
        problem = _unknown_kinds(record.registers) or registration_problem(record.registers, registered)
        if problem:
            self._roll_back(record, registered)
            self._shutdown_quietly(record.name, obj)
            return self._broken(record, "registers", problem)
        record.checks.append(_check("registers", True, _registered_summary(record.registers, registered)))
        record.state = "ready"
        self._objects[record.name] = obj
        self.log(f"plugin {record.name} {record.version}: ready ({record.source})")
        return record

    def _roll_back(self, record: PluginRecord, registered: Mapping[str, Sequence[str]]) -> None:
        """Undo what a plugin registered: half a plugin left in the registry is the failure this check exists for."""
        if not registered:
            return
        from pinneapple_core.registry import unregister

        for kind, names in registered.items():
            for name in names:
                try:
                    unregister(kind, name)
                except Exception as exc:                # noqa: BLE001 - removing what is not there is not a failure
                    self.log(f"plugin {record.name}: could not roll back {kind} {name!r}: "
                             f"{type(exc).__name__}: {exc}")
        self.log(f"plugin {record.name}: rolled back {_registered_summary({}, registered)}")

    def _broken(self, record: PluginRecord, stage: str, reason: str, *, traceback_text: str = "") -> PluginRecord:
        record.state = "broken"
        record.stage = stage
        record.reason = reason
        record.traceback = traceback_text
        record.checks.append(_check(stage, False, reason))
        self.log(f"plugin {record.name} is broken at {stage}: {reason}")
        return record

    def _dependencies(self, record: PluginRecord) -> tuple[list[str], dict[str, Any]]:
        """Every declared dependency, resolved against the environment. Returns (problems, what was found)."""
        requires = record.requires or {}
        details: dict[str, Any] = {"executables": {}, "python": {}, "plugins": {}}
        problems: list[str] = []

        spec = str(requires.get("pinneapple") or "").strip()
        if spec:
            running = self._library_version()
            if not running:
                details["pinneapple"] = "unknown: this installation does not report its version"
            else:
                details["pinneapple"] = f"{running} {'satisfies' if satisfies(running, spec) else 'does NOT satisfy'} {spec}"
                if not satisfies(running, spec):
                    problems.append(f"pinneapple {running} does not satisfy {spec}")

        for mod in requires.get("python") or []:
            ok = _module_present(str(mod))
            details["python"][str(mod)] = "present" if ok else "missing"
            if not ok:
                problems.append(f"python module {mod} is not installed")

        for exe in requires.get("executables") or []:
            found = shutil.which(str(exe))
            details["executables"][str(exe)] = found or "not found on PATH"
            if not found:
                problems.append(f"executable {exe} is not on PATH")

        for dep in requires.get("plugins") or []:
            present = str(dep) in self._entries
            details["plugins"][str(dep)] = "installed" if present else "not installed"
            if not present:
                problems.append(f"plugin {dep} is not installed")
        return problems, details

    def _library_version(self) -> str:
        if self._running_version:
            return self._running_version
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pinneapple")
        except PackageNotFoundError:
            pass
        loaded = sys.modules.get("pinneapple")
        got = getattr(loaded, "__version__", "") if loaded is not None else ""
        if got:
            return str(got)
        # running from a source tree without installed metadata: read the version the package declares
        candidate = Path(__file__).resolve().parent.parent / "pinneapple" / "__init__.py"
        try:
            m = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', candidate.read_text(encoding="utf-8"), re.M)
            return m.group(1) if m else ""
        except OSError:
            return ""

    def _config_value(self, name: str, key: str, options: Mapping[str, Any]) -> Any:
        if key in options:
            return options[key]
        return os.environ.get(self._env_name(name, key))

    @staticmethod
    def _env_name(name: str, key: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9]+", "_", f"{name}_{key}").upper()
        return f"PINNEAPPLE_PLUGIN_{clean}"

    def _data_dir(self, name: str) -> Path:
        d = Path.home() / ".pinneapple" / "plugins" / name
        return d

    def _shutdown_quietly(self, name: str, obj: Any) -> None:
        try:
            obj.shutdown()
        except Exception as exc:                        # noqa: BLE001 - shutdown must never raise on the caller
            self.log(f"plugin {name}: shutdown raised {type(exc).__name__}: {exc}")

    def shutdown_all(self) -> None:
        """``shutdown()`` for every plugin that was initialized — including the ones loaded before a later failure.

        Safe to call twice: the objects are forgotten once shut down, so the second call is a no-op.
        """
        for name in sorted(self._objects):
            obj = self._objects.pop(name)
            self._shutdown_quietly(name, obj)

    # -- capability lookup -----------------------------------------------------------------------------------------
    def capabilities(self) -> dict[str, list[str]]:
        """Capability -> the plugins that declared it, in name order. Only what is ready is listed."""
        out: dict[str, list[str]] = {}
        for record in self.records().values():
            if record.state != "ready":
                continue
            for cap in record.capabilities:
                out.setdefault(cap, []).append(record.name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def find(self, capability: str | None = None, **query: Any) -> list[Match]:
        """Plugins that declare ``capability`` and every axis of ``query`` (#1473).

        ``find(capability="solver", domain="cfd", steady_state=True, formats=("stl", "step"))`` matches a plugin
        whose ``capabilities["solver"]`` lists ``cfd`` and ``steady_state`` and whose ``capabilities["formats"]``
        lists ``stl`` and ``step``. The order is deterministic (plugin name). Nothing is inferred: a capability no
        plugin declares returns an empty list, and the caller can ask :meth:`capabilities` what does exist.
        """
        if not capability:
            raise ValueError("find needs a capability, e.g. find(capability='solver', domain='cfd'); known "
                             f"capabilities: {sorted(self.capabilities())}")
        matches: list[Match] = []
        for record in self.records().values():
            if record.state != "ready" or capability not in record.capabilities:
                continue
            values = list(record.capabilities[capability])
            if not _matches(values, query, record):
                continue
            matches.append(Match(plugin=record.name, capability=capability, values=values))
        return matches

    def find_one(self, capability: str | None = None, **query: Any) -> Match:
        """Exactly one match, or :class:`AmbiguousCapability` naming every candidate — never a silent pick."""
        matches = self.find(capability, **query)
        if not matches:
            raise LookupError(f"no ready plugin declares {dict(query, capability=capability)}; declared capabilities: "
                              f"{sorted(self.capabilities())}")
        if len(matches) > 1:
            raise AmbiguousCapability(matches, dict(query, capability=capability))
        return matches[0]

    # -- diagnosis -------------------------------------------------------------------------------------------------
    def doctor(self, name: str | None = None) -> list[PluginRecord]:
        """The records to diagnose, per plugin, in name order (``name`` narrows to one and never raises)."""
        if name is None:
            self.records()
            return [self._records[k] for k in sorted(self._records)]
        return [self.inspect(name)]

    def disable(self, name: str) -> Path:
        """Write ``name`` into the configuration's ``disabled`` list and stop using it in this process.

        The package is not touched: it stays importable, and ``enable`` is one line away. A plugin that is running
        has ``shutdown()`` called and its capabilities leave :meth:`find` immediately; what it registered in the
        library before that (a model in ``pp.list_models()``) stays registered until the process ends, because nothing
        here keeps a list of what a plugin put in someone else's registry.
        """
        if name not in self._entries:
            raise UnknownPlugin(name)
        disabled = sorted(set(self.schema.disabled) | {name})
        self.config["disabled"] = disabled
        self.config.setdefault("schema", CONFIG_SCHEMA)
        path = save_config(self.config_file, self.config)
        self.schema = ConfigSchema(self.config)
        obj = self._objects.pop(name, None)
        if obj is not None:
            self._shutdown_quietly(name, obj)
        record = self._records.get(name) or PluginRecord(name=name, source=self._entries[name].source)
        record.state = "disabled"
        record.capabilities = {}
        record.reason = f"disabled in {path}"
        self._records[name] = record
        self.log(f"plugin {name}: disabled ({path})")
        return path

    def enable(self, name: str) -> Path:
        """Remove ``name`` from the ``disabled`` list. It is not loaded here: the next call that loads does."""
        if name not in self._entries:
            raise UnknownPlugin(name)
        self.config["disabled"] = [n for n in self.schema.disabled if n != name]
        self.config.setdefault("schema", CONFIG_SCHEMA)
        path = save_config(self.config_file, self.config)
        self.schema = ConfigSchema(self.config)
        if name in self._records and self._records[name].state == "disabled":
            self._records.pop(name)
        self.log(f"plugin {name}: enabled ({path})")
        return path


# ----------------------------------------------------------------------------------------------- helpers
def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": str(detail)}


def _watch_registry(into: dict[str, list[str]]) -> Any:
    """Record what a plugin registers while ``initialize`` runs, through the library's registry protocol.

    ``pp.register`` reports every registration as it happens (:func:`pinneapple_core.registry.observe`), so this is
    what the plugin itself added — not the presets and models that a first import of a library module may register
    while it is imported inside ``initialize``.
    """
    from pinneapple_core.registry import observe

    def listener(action: str, kind: str, name: str) -> None:
        names = into.setdefault(kind, [])
        if action == "register":
            if name not in names:
                names.append(name)
        elif name in names:                             # unregistered within the same initialize
            names.remove(name)
            if not names:
                into.pop(kind, None)

    return observe(listener)


def registration_problem(declared: Mapping[str, Sequence[str]],
                         registered: Mapping[str, Sequence[str]]) -> str:
    """The refusal message when a plugin's ``registers`` and its actual registrations disagree, else ``""``.

    Both directions matter, and only one of them looks like a bug at load time: a plugin that registers what it never
    declared adds to the library silently, and a plugin that declares what it never registers is the plugin written
    against last year's API — it loads, does half its steps, and looks successful. Nothing here is a warning: a
    capability the library can find but that is not there is worse than a plugin that refuses to load.

    A plugin that declares nothing at all (contract ``1.0``, written before ``registers`` existed) is compatible and is
    not refused: what it registers is recorded, shown by ``doctor``, and says plainly that it was not verified — there
    is no declaration to hold it to, and deriving one from what it happened to register would verify nothing.
    """
    if not declared:
        return ""
    extra = {k: [n for n in v if n not in (declared.get(k) or ())] for k, v in registered.items()}
    extra = {k: v for k, v in extra.items() if v}
    missing = {k: [n for n in v if n not in (registered.get(k) or ())] for k, v in declared.items()}
    missing = {k: v for k, v in missing.items() if v}
    if not extra and not missing:
        return ""
    parts = []
    if extra:
        parts.append(f"registered but not declared in `registers`: {_pairs(extra)}")
    if missing:
        parts.append(f"declared in `registers` but not registered: {_pairs(missing)}")
    return ("; ".join(parts) + ". The plugin was refused and its registrations were rolled back: a plugin that does "
            "half of what it declares loads and looks successful, which is worse than one that refuses to load. "
            "Declare what `initialize` registers through `pp.register` (kind -> names) and register what you declare, "
            "or drop it from `registers`.")


def _unknown_kinds(declared: Mapping[str, Sequence[str]]) -> str:
    """A ``registers`` key that is not one of the registry's kinds, named as such — a typo there would otherwise read
    as "declared but not registered" about a kind the library does not have."""
    if not declared:
        return ""
    try:
        from pinneapple_core.registry import kinds
    except ImportError:                                 # pragma: no cover - the registry ships with the library
        return ""
    known = kinds()
    unknown = sorted(str(k) for k in declared if str(k) not in known)
    if not unknown:
        return ""
    return (f"`registers` names kind(s) the registry does not have: {unknown}; known kinds: {known}. The plugin was "
            f"refused before anything it registered was kept")


def _pairs(table: Mapping[str, Sequence[str]]) -> str:
    return ", ".join(f"{kind}={list(names)}" for kind, names in sorted(table.items()))


def _registered_summary(declared: Mapping[str, Sequence[str]],
                        registered: Mapping[str, Sequence[str]]) -> str:
    if not declared and not registered:
        return "nothing registered (and nothing declared)"
    if not declared:
        return (f"registered {_pairs(registered)}, declared nothing: recorded, not verified — declare `registers` to "
                f"have this checked")
    return f"matches `registers`: {_pairs(registered) or 'nothing'}"


def _summary(details: Mapping[str, Any]) -> str:
    bits: list[str] = []
    exes = details.get("executables") or {}
    if exes:
        bits.append(", ".join(f"{k}: {v}" for k, v in exes.items()))
    mods = details.get("python") or {}
    if mods:
        bits.append("modules " + ", ".join(f"{k} {v}" for k, v in mods.items()))
    plugins = details.get("plugins") or {}
    if plugins:
        bits.append("plugins " + ", ".join(f"{k} {v}" for k, v in plugins.items()))
    if details.get("pinneapple"):
        bits.append(f"pinneapple {details['pinneapple']}")
    return "; ".join(bits) or "no declared dependency"


def _matches(values: Sequence[str], query: Mapping[str, Any], record: PluginRecord) -> bool:
    """Every axis of the query against the plugin: on the capability's own values, or on another capability."""
    for axis, want in query.items():
        haystack = list(values)
        if axis != "capability":
            haystack += list(record.capabilities.get(axis) or [])
        wanted = [want] if isinstance(want, (str, bool, int, float)) else list(want or [])
        for item in wanted:
            as_text = str(item)
            if as_text not in haystack and as_text.lower() not in [v.lower() for v in haystack]:
                if bool(item) is True and axis in haystack:      # a flag axis: the key itself is the claim
                    continue
                return False
    return True


def _module_present(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _tb() -> str:
    return traceback.format_exc().rstrip()


# Re-exported so a plugin author imports the contract from one place.
Contract = Plugin
