"""``pinneapple plugins ...``: list, inspect, doctor, enable, disable (#1474).

    pinneapple plugins list                # every discovered plugin: name, version, source package, contract, state
    pinneapple plugins inspect openfoam    # capabilities, declared dependencies, entry point, where it came from
    pinneapple plugins doctor              # per plugin: deps present, contract compatible, executables found, config
    pinneapple plugins enable openfoam     # write to the user configuration
    pinneapple plugins disable openfoam    # write to the user configuration; does not uninstall

Exit codes, like the rest of the CLI: 0 nothing to report, 1 a plugin is broken (``doctor``/``list``), 2 a usage
error — an unknown plugin name for ``enable``/``disable``, an unreadable configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .discovery import ENTRY_GROUP
from .manager import PluginManager, PluginRecord, UnknownPlugin, config_path

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pinneapple plugins",
        description="The plugins installed from packages: what they provide, whether they work here, and which ones "
                    f"are enabled. A plugin is any installed distribution declaring the {ENTRY_GROUP!r} entry point.")
    ap.add_argument("--config", metavar="FILE", help="the enable/disable configuration (default: "
                                                     "$PINNEAPPLE_PLUGINS_CONFIG, else ~/.pinneapple/plugins.json)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every discovered plugin with its state")
    p_inspect = sub.add_parser("inspect", help="everything known about one plugin")
    p_inspect.add_argument("name")
    p_doctor = sub.add_parser("doctor", help="the per-plugin checks, in the order the loader applies them")
    p_doctor.add_argument("name", nargs="?", help="only this plugin")
    p_doctor.add_argument("--verbose", action="store_true", help="print the traceback of a plugin that failed")
    for verb, text in (("enable", "remove a plugin from the disabled list (it loads next time)"),
                       ("disable", "add a plugin to the disabled list; nothing is uninstalled")):
        p = sub.add_parser(verb, help=text)
        p.add_argument("name")
    return ap


def _state_text(record: PluginRecord) -> str:
    if record.state == "broken" and record.stage:
        return f"broken ({record.stage})"
    return record.state


def _list(manager: PluginManager) -> int:
    records = [manager.records()[k] for k in sorted(manager.records())]
    if not records:
        print(f"no plugin installed: nothing declares the {ENTRY_GROUP!r} entry point "
              f"(pip install a PINNeAPPle plugin package to add one)")
        return 0
    rows = [("NAME", "VERSION", "CONTRACT", "STATE", "SOURCE", "CAPABILITIES")]
    for r in records:
        rows.append((r.name, r.version or "-", r.contract_version or "-", _state_text(r), r.source or "-",
                     ", ".join(sorted(r.capabilities)) or "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    broken = [r for r in records if r.state == "broken"]
    for r in broken:
        print(f"\n{r.name}: {r.reason}")
    return 1 if broken else 0


def _inspect(manager: PluginManager, name: str) -> int:
    r = manager.inspect(name)
    caps = "; ".join(f"{k}={v}" for k, v in sorted(r.capabilities.items())) or "-"
    requires = "; ".join(f"{k}={v}" for k, v in sorted(r.requires.items())) or "-"
    print(f"name            {r.name}")
    print(f"state           {_state_text(r)}")
    print(f"version         {r.version or '-'}")
    print(f"contract        {r.contract_version or '-'} (this installation speaks the same major version)")
    print(f"source          {r.source or '-'}")
    print(f"entry point     {r.entry_point or '-'}")
    print(f"capabilities    {caps}")
    print(f"permissions     {', '.join(r.permissions) or '-'} (recorded, not enforced)")
    print(f"requires        {requires}")
    if r.details.get("executables"):
        print("executables     " + "; ".join(f"{k}: {v}" for k, v in r.details["executables"].items()))
    if r.details.get("python"):
        print("python modules  " + "; ".join(f"{k}: {v}" for k, v in r.details["python"].items()))
    if r.details.get("plugins"):
        print("other plugins   " + "; ".join(f"{k}: {v}" for k, v in r.details["plugins"].items()))
    if r.reason:
        print(f"reason          {r.reason}")
    if r.traceback:
        print("\n" + r.traceback)
    return 0 if r.state in ("ready", "disabled") else 1


def _doctor(manager: PluginManager, name: str | None, verbose: bool) -> int:
    records = manager.doctor(name)
    if not records:
        print(f"no plugin installed: nothing declares the {ENTRY_GROUP!r} entry point")
        return 0
    broken = 0
    for r in records:
        print(f"{r.name} {r.version or ''}".rstrip() + f": {_state_text(r)}")
        if r.state in ("disabled", "not_installed"):
            print(f"  - {r.reason}")
            broken += r.state == "not_installed"
            print()
            continue
        for check in r.checks:
            mark = "ok  " if check["ok"] else "FAIL"
            print(f"  {mark} {check['check']:<13} {check['detail']}")
        if not r.checks:
            print("  - not checked: nothing ran for this plugin")
        if r.state == "broken":
            broken += 1
            if verbose and r.traceback:
                print()
                for line in r.traceback.splitlines():
                    print(f"  {line}")
        print()
    counts = {}
    for r in records:
        counts[r.state] = counts.get(r.state, 0) + 1
    print(", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + f" | configuration: {manager.config_file}")
    return 1 if broken else 0


def _toggle(manager: PluginManager, verb: str, name: str) -> int:
    try:
        path = manager.disable(name) if verb == "disable" else manager.enable(name)
    except UnknownPlugin:
        print(f"error: {name!r} is not an installed plugin; installed: "
              f"{', '.join(manager.names()) or 'none'}", file=sys.stderr)
        return 2
    if verb == "disable":
        print(f"plugin {name}: disabled in {path}. Nothing was uninstalled; `pinneapple plugins enable {name}` "
              f"turns it back on, and a run started before this keeps what it already registered.")
    else:
        print(f"plugin {name}: enabled in {path}. It is loaded the next time a command or a run starts.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        manager = PluginManager(config=args.config)
        manager.records()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "list":
        if args.json:
            print(json.dumps([r.to_dict() for r in manager.records().values()], indent=2, sort_keys=True))
            return 0
        return _list(manager)
    if args.command == "inspect":
        record = manager.inspect(args.name)
        if args.json:
            print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
            return 0 if record.state in ("ready", "disabled") else 1
        return _inspect(manager, args.name)
    if args.command == "doctor":
        if args.json:
            records = manager.doctor(args.name)
            print(json.dumps([r.to_dict() for r in records], indent=2, sort_keys=True))
            return 1 if any(r.state in ("broken", "not_installed") for r in records) else 0
        return _doctor(manager, args.name, args.verbose)
    if args.command in ("enable", "disable"):
        return _toggle(manager, args.command, args.name)
    return 2                                                # pragma: no cover - argparse refuses unknown verbs


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())


# ``config_path`` is re-exported for the tests and the docs snippet above.
_ = config_path
