"""What an installed package must implement to extend PINNeAPPle, and how the version of that contract is checked.

This module is deliberately importable on its own: a third-party package depends on ``pinneapple-plugins`` (or imports
these names without importing the rest of the library) and never pulls the heavy core, the same reasoning that put
``pinneapple_stability`` in a package of its own. It uses the standard library only.

The contract
------------
A plugin is an object with four declarations and two calls::

    from pinneapple_plugins import Plugin, PluginContext

    class OpenFOAM(Plugin):
        name = "openfoam"                       # the identity of the plugin
        version = "0.3.1"                       # its own version
        contract_version = "1.0"                # the version of *this* contract it was written against
        requires = {"pinneapple": ">=0.8",      # a specifier over the installed library
                    "plugins": ["cfd-common"],  # other plugins it needs
                    "executables": ["blockMesh"],  # programs that must be on PATH
                    "python": ["numpy"],        # modules that must import
                    "config": ["case_root"]}    # keys the user must have configured
        capabilities = {"solver": ["cfd", "steady_state"], "geometry": ["stl", "step"]}
        permissions = ("read", "run")           # recorded and shown, never enforced (see #1475)

        def initialize(self, context: PluginContext) -> None:
            ...                                 # register what it provides; called once
            context.log("openfoam ready")

        def shutdown(self) -> None:
            ...                                 # release resources; safe twice, safe after a failed initialize

Registering what the plugin provides happens inside :meth:`Plugin.initialize` — ``pp.register(...)`` for a model or a
solver, ``pp.benchmark.register_method(...)`` for a benchmark. There is no second hook: one lifecycle call is what a
rule such as "a plugin that fails to initialize must not take the others down" can be written against.

Rules the loader holds itself to
--------------------------------
* **A plugin that fails to load is a record, not an exception.** ``import pinneapple`` and every other plugin keep
  working; the failure is stored with its stage, message and traceback and surfaced by ``pinneapple plugins doctor``.
* **An incompatible ``contract_version`` is refused by name**, with both versions in the message, before
  ``initialize`` runs.
* **A missing declared dependency is reported, not guessed around.**
* ``shutdown()`` is called for every plugin that was initialized, including when a later plugin fails.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_HISTORY",
    "PERMISSIONS",
    "Plugin",
    "PluginContext",
    "PluginError",
    "IncompatibleContract",
    "InvalidDeclaration",
    "MissingDependency",
    "contract_compatible",
    "parse_version",
    "satisfies",
]

#: The version of the contract this module implements. A plugin declares the version it was written against; bump
#: the minor for an addition that old plugins survive, the major for anything that breaks them.
CONTRACT_VERSION = "1.0"

#: What changed, so a refusal can say more than "incompatible".
CONTRACT_HISTORY: dict[str, str] = {
    "1.0": "first version: name, version, contract_version, requires, capabilities, permissions, initialize, shutdown",
}

#: The permission names a plugin may declare. Recorded and shown by ``plugins doctor``; nothing enforces them
#: (a plugin is Python code running with the privileges of whoever imports it — see #1475).
PERMISSIONS: tuple[str, ...] = ("read", "run", "write", "network")

_VERSION = re.compile(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*$")


class PluginError(Exception):
    """Anything a plugin declaration or its loading refuses."""


class InvalidDeclaration(PluginError):
    """The object a plugin entry point resolved to does not satisfy :class:`Plugin`."""


class IncompatibleContract(PluginError):
    """The plugin was written against a contract version this library does not speak."""


class MissingDependency(PluginError):
    """``requires`` names something the environment does not have."""


def parse_version(text: str) -> tuple[int, int, int]:
    """``"1.2.3"`` -> ``(1, 2, 3)``; ``"1.2"`` -> ``(1, 2, 0)``; ``"1"`` -> ``(1, 0, 0)``.

    The contract version is a plain ``major.minor[.patch]`` number: a plugin's own version may be anything
    (PEP 440 and its epochs are the package's business), but the contract version is compared here.
    """
    m = _VERSION.match(str(text))
    if not m:
        raise InvalidDeclaration(f"contract_version must be a number like '1.0', got {text!r}")
    return int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)


def contract_compatible(declared: str, running: str = CONTRACT_VERSION) -> tuple[bool, str]:
    """Whether a plugin written against ``declared`` can run against ``running``, and why not when it cannot.

    Same major version and a minor no newer than the running one: a plugin written against ``1.0`` runs on ``1.3``
    (the additions are additive), one written against ``1.4`` does not (it may call something absent), and ``2.0``
    is a different contract.
    """
    d = parse_version(declared)
    r = parse_version(running)
    if d[0] != r[0]:
        return False, (f"contract_version {declared} is from contract {d[0]}.x, this installation speaks "
                       f"{running} ({r[0]}.x); the plugin was written against a different contract")
    if d[1] > r[1]:
        return False, (f"contract_version {declared} is newer than the running {running}: the plugin may call "
                       f"something this installation does not have")
    return True, ""


# --------------------------------------------------------------------------------------------- version specifiers
_SPEC = re.compile(r"^\s*(==|!=|>=|<=|>|<|~=)\s*([0-9][0-9A-Za-z.\-+]*)\s*$")


def _release(text: str) -> tuple[int, ...]:
    """The leading numeric release of a version string, ignoring a pre-release/local suffix (``1.2.0rc1``)."""
    out: list[int] = []
    for part in str(text).strip().split("."):
        m = re.match(r"^(\d+)", part)
        if not m:
            break
        out.append(int(m.group(1)))
    return tuple(out) or (0,)


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def satisfies(version: str, specifier: str) -> bool:
    """Whether ``version`` satisfies a comma-separated specifier such as ``">=0.8,<0.9"``.

    A small subset, on purpose: this package is dependency-free and only ever compares the library's own release.
    ``==``/``!=`` compare the release (``1.2`` equals ``1.2.0``), ``~=`` is ``>=x.y`` and ``<x+1``, and a pre-release
    suffix is ignored — which is stated here rather than guessed at.
    """
    if not str(specifier or "").strip():
        return True
    for clause in str(specifier).split(","):
        if not clause.strip():
            continue
        m = _SPEC.match(clause)
        if not m:
            raise InvalidDeclaration(f"cannot read version specifier {clause.strip()!r}: expected e.g. '>=0.8,<0.9'")
        op, want = m.group(1), m.group(2)
        c = _cmp(_release(version), _release(want))
        if op == ">=" and c < 0:
            return False
        if op == "<=" and c > 0:
            return False
        if op == ">" and c <= 0:
            return False
        if op == "<" and c >= 0:
            return False
        if op == "==" and c != 0:
            return False
        if op == "!=" and c == 0:
            return False
        if op == "~=":
            if c < 0 or _release(version)[0] != _release(want)[0]:
                return False
    return True


# ---------------------------------------------------------------------------------------------- the lifecycle
@dataclass(frozen=True)
class PluginContext:
    """The read-only context handed to :meth:`Plugin.initialize`.

    ``paths`` are the directories a plugin may write into, ``config`` is its own section of the user configuration
    (``options.<name>`` in ``~/.pinneapple/plugins.json``), ``logger`` writes to the loader's log, and
    ``remote_allowed`` says whether this installation may hand work to a remote executor (#1479). A plugin reads the
    context; nothing here is a handle on the library's internals.
    """

    paths: Mapping[str, Path] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    logger: Callable[[str], None] = lambda line: None
    remote_allowed: bool = False

    def log(self, line: str) -> None:
        """Write one line to the loader's log (the CLI prints it, a test may collect it)."""
        self.logger(str(line))

    def path(self, name: str, default: str | None = None) -> Path | None:
        """One of the declared paths, or ``default`` when it is not there."""
        got = self.paths.get(name)
        return Path(got) if got is not None else (Path(default) if default else None)

    def as_dict(self) -> dict[str, Any]:
        return {"paths": {k: str(v) for k, v in self.paths.items()}, "config_keys": sorted(self.config),
                "remote_allowed": self.remote_allowed}


class Plugin:
    """The contract. Subclass it, or provide the same attributes and two methods (duck typing is deliberate: a plugin
    written for ``1.0`` should not need this class to be its base class)."""

    #: Identity of the plugin. Must match the entry point name it is installed under, or the loader records why not.
    name: str = ""
    #: The plugin's own version (any string; it is recorded, not compared).
    version: str = "0"
    #: The version of *this* contract the plugin was written against.
    contract_version: str = CONTRACT_VERSION
    #: What has to be there before ``initialize`` is called: ``pinneapple`` (a specifier over the library),
    #: ``plugins`` (other plugin names), ``executables`` (programs on PATH), ``python`` (importable modules) and
    #: ``config`` (keys that must be present in the plugin's configuration section).
    requires: Mapping[str, Any] = {}
    #: What the plugin provides, as a mapping of capability -> what it can do (see ``pp.plugins.find``).
    capabilities: Mapping[str, Sequence[str]] = {}
    #: Declared permissions (a subset of :data:`PERMISSIONS`); recorded, shown, and deliberately not enforced.
    permissions: Sequence[str] = ()

    def initialize(self, context: PluginContext) -> None:
        """Register what the plugin provides. Called once; a raise is recorded and this plugin alone is affected."""

    def shutdown(self) -> None:
        """Release resources. Called once, must be safe to call twice and after a failed ``initialize``."""

    # -- helpers a loader may use ---------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "contract_version": self.contract_version,
                "requires": dict(self.requires or {}), "capabilities": {k: list(v) for k, v in
                                                                       (self.capabilities or {}).items()},
                "permissions": list(self.permissions or ())}


def check_declaration(obj: Any) -> dict[str, Any]:
    """Validate the declarations of ``obj`` (a :class:`Plugin` instance, a subclass or a duck-typed object).

    Returns the declarations as a dict; raises :class:`InvalidDeclaration` naming the first thing that is wrong,
    so ``plugins doctor`` can say what to fix rather than what happened to break at import time.
    """
    if isinstance(obj, type):
        try:
            obj = obj()
        except Exception as exc:                        # noqa: BLE001 - the declaration is the plugin's, report it
            raise InvalidDeclaration(f"the contract class could not be instantiated: {type(exc).__name__}: {exc}") from exc
    name = getattr(obj, "name", "")
    if not isinstance(name, str) or not name.strip():
        raise InvalidDeclaration("a plugin needs a non-empty `name`")
    for attr, kind in (("version", str), ("contract_version", str)):
        value = getattr(obj, attr, None)
        if not isinstance(value, kind) or not str(value).strip():
            raise InvalidDeclaration(f"plugin {name!r} needs a non-empty `{attr}` (a string)")
    requires = getattr(obj, "requires", {}) or {}
    if not isinstance(requires, Mapping):
        raise InvalidDeclaration(f"plugin {name!r}: `requires` must be a mapping, got {type(requires).__name__}")
    for key in requires:
        if key not in ("pinneapple", "plugins", "executables", "python", "config"):
            raise InvalidDeclaration(f"plugin {name!r}: `requires` has an unknown key {key!r}; known: "
                                     f"pinneapple, plugins, executables, python, config")
    caps = getattr(obj, "capabilities", {}) or {}
    if not isinstance(caps, Mapping):
        raise InvalidDeclaration(f"plugin {name!r}: `capabilities` must be a mapping capability -> list, got "
                                 f"{type(caps).__name__}")
    for cap, values in caps.items():
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise InvalidDeclaration(f"plugin {name!r}: capability {cap!r} must be a list of strings, not "
                                     f"{type(values).__name__} (one capability per key, several values in the list)")
    perms = list(getattr(obj, "permissions", ()) or ())
    unknown = [p for p in perms if p not in PERMISSIONS]
    if unknown:
        raise InvalidDeclaration(f"plugin {name!r}: unknown permission(s) {unknown}; known: {list(PERMISSIONS)}")
    for method in ("initialize", "shutdown"):
        if not callable(getattr(obj, method, None)):
            raise InvalidDeclaration(f"plugin {name!r} has no callable `{method}`")
    return {"name": name, "version": str(obj.version), "contract_version": str(obj.contract_version),
            "requires": dict(requires), "capabilities": {str(k): [str(v) for v in values]
                                                        for k, values in caps.items()},
            "permissions": perms}
