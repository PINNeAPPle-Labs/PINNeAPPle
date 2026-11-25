#!/usr/bin/env python
"""Dependency floors and licences, audited from the files in the tree (#96).

`dependency_licenses.py` answers "what does each core dependency declare about its licence, in *this*
environment". That is not an audit: it says nothing about the optional extras, it says nothing about the
minimum version this repository promises to work with, and it only sees what happens to be installed. This
script is the reproducible half.

Two records are committed, and this gate keeps the tree honest against them:

`docs/dev/dependency_floors.json`
    the **tested minimum** of every declared dependency -- core and every extra. The floor is the lower
    bound in ``pyproject.toml``; the record is what CI installs and runs the fast suite against
    (``constraints/floors.txt``, regenerated here). Loosening a bound in ``pyproject.toml`` without
    updating the record fails, because the installed range would then reach below what was tested.

`docs/dev/dependency_licenses.json`
    the licence each declared dependency declares, read from its installed metadata when it was recorded,
    plus the dependencies the recording environment did not have and the ones it had installed while their
    metadata named no licence (both named, neither guessed). A dependency that is installed here and
    copyleft -- GPL, AGPL, LGPL, SSPL, EUPL -- fails unless it is a decision written in
    ``docs/licensing.md`` first. An unreadable licence fails the same way when it is a core dependency,
    because a core dependency is installed wherever the package is.

The table in ``docs/licensing.md`` between the ``dependency-licences`` markers is generated from the record,
so the licences are listed in the documentation rather than in a JSON file nobody reads.

    python scripts/licensing/supply_chain.py --check          # the gate (also `make audit`)
    python scripts/licensing/supply_chain.py --record         # after a dependency change, with the reason
    python scripts/licensing/supply_chain.py --json           # the audit as data, for a CI artifact

No network, no environment beyond the one running it: the same tree gives the same answer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_licenses import is_copyleft, licence_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
FLOORS = ROOT / "docs" / "dev" / "dependency_floors.json"
LICENCES = ROOT / "docs" / "dev" / "dependency_licenses.json"
CONSTRAINTS = ROOT / "constraints" / "floors.txt"
POLICY = ROOT / "docs" / "licensing.md"
SCHEMA = "pinneapple.supply_chain/1"
MARK_BEGIN = "<!-- dependency-licences:start -->"
MARK_END = "<!-- dependency-licences:end -->"
COPYLEFT_HEADING = "## Dependencies accepted with a copyleft licence"
UNOBSERVED = "not installed in the recording environment, so its metadata was never read"


# ---------------------------------------------------------------- what the tree declares
def _requirement(text: str):
    from packaging.requirements import Requirement

    return Requirement(text)


def declared(pyproject: Path = PYPROJECT) -> dict[str, dict[str, Any]]:
    """key -> {name, extra, marker, floor, requirement} for the core list and every optional extra.

    The key is unique per (name, marker) pair: ``mujoco`` is declared twice, with a different floor and a
    different marker on each side of Python 3.11, and collapsing the two would lose one of the floors.

    ``extra`` is ``"core"`` or the name of the extra; a dependency that appears in several extras keeps the
    sorted list, so the record says where it is used.
    """
    import tomllib

    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    table: dict[str, dict[str, Any]] = {}

    def add(requirement: str, extra: str) -> None:
        try:
            req = _requirement(requirement)
        except Exception:                                    # unparseable: keep the text, no floor
            req = None
        name = req.name if req is not None else re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0]
        marker = str(req.marker).replace(" ", "") if req is not None and req.marker is not None else None
        key = f"{name}[{marker}]" if marker else name
        entry = table.setdefault(key, {"name": name, "marker": marker, "floor": _floor(req, requirement),
                                       "requirement": requirement.strip(), "extras": []})
        if extra not in entry["extras"]:
            entry["extras"].append(extra)
        entry["extras"].sort()

    for requirement in project.get("dependencies", []):
        add(requirement, "core")
    for extra, requirements in (project.get("optional-dependencies") or {}).items():
        for requirement in requirements:
            add(requirement, extra)
    return dict(sorted(table.items()))


def _floor(req, requirement: str) -> str | None:
    """The lower bound of the requirement: ``>=``, ``>`` or ``==``, as written."""
    if req is None:
        parsed = re.search(r"(>=|==|~=)\s*([0-9][0-9A-Za-z.*]*)", requirement)
        return parsed.group(2) if parsed else None
    specifiers = req.specifier
    for operator in (">=", "==", "~=", ">"):
        for spec in specifiers:
            if spec.operator == operator:
                return spec.version
    return None


def _python_floor(pyproject: Path = PYPROJECT) -> str:
    import tomllib

    requires = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get("requires-python", "")
    found = re.search(r">=\s*([0-9]+\.[0-9]+)", requires)
    return found.group(1) if found else ""


# ---------------------------------------------------------------- the records
def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _version(name: str) -> str:
    """The installed version of ``name``, or ``""`` when this interpreter does not have it."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def record_floors(deps: dict[str, dict[str, Any]], pyproject: Path = PYPROJECT) -> dict[str, Any]:
    """The floors record, regenerated from ``pyproject.toml``; the policy text survives a re-record."""
    previous = load(FLOORS)
    core = {key: entry["floor"] or "" for key, entry in deps.items() if "core" in entry["extras"]}
    extras = {key: entry["floor"] or "" for key, entry in deps.items() if "core" not in entry["extras"]}
    return {
        "schema": SCHEMA,
        "recorded": dt.date.today().isoformat(),
        "python": _python_floor(pyproject),
        "policy": previous.get("policy") or (
            "A floor is a claim, and it is tested where it can be installed: `.github/workflows/floors.yml` installs "
            "`constraints/floors.txt` on the lowest supported Python and runs the fast suite, so `numpy>=X` means "
            "numpy X was run, not that numpy X exists. The extra floors are recorded here and exercised at their "
            "latest release by `.github/workflows/extras.yml`; an extra whose floor cannot be installed on the "
            "tested Python is listed in `unverified` with the reason, never dropped."
        ),
        "core": core,
        "extras": extras,
        "unverified": previous.get("unverified") or {},
    }


def record_licences(deps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The licence of what is installed, and why the rest is not there.

    Three states, kept apart because they say different things: ``packages`` is a licence that was read;
    ``unobserved`` is a dependency this environment never had installed, so there was nothing to read;
    ``unreadable`` is one that *is* installed and whose metadata names no licence at all. The last two are
    not the same problem -- a distribution can ship an Apache-2.0 licence file and still declare nothing a
    machine can read -- so the record names the reason instead of folding them together.
    """
    packages, unobserved, unreadable = {}, [], {}
    for key in deps:
        name = deps[key]["name"]
        licence = licence_of(name)
        if licence.startswith("not installed"):
            if name not in unobserved:
                unobserved.append(name)
            continue
        if licence.startswith("unknown"):
            unreadable[name] = licence
            continue
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:                 # pragma: no cover - race with an uninstall
            version = ""
        packages[name] = {"licence": licence, "version": version}
    return {
        "schema": SCHEMA,
        "recorded": dt.date.today().isoformat(),
        "environment": f"CPython {sys.version.split()[0]} on {sys.platform}",
        "note": "Licences are read from installed metadata: PEP 639 License-Expression, else the trove "
                "classifier, else the short License field. `unobserved` names the dependencies this "
                "environment did not have installed; `unreadable` the ones it had, whose metadata declares "
                "no licence. Neither is guessed.",
        "packages": dict(sorted(packages.items())),
        "unobserved": sorted(unobserved),
        "unreadable": dict(sorted(unreadable.items())),
    }


def constraints_text(deps: dict[str, dict[str, Any]], floors: dict[str, Any]) -> str:
    """``constraints/floors.txt``: every core floor pinned with ``==``, markers kept, for the floor job."""
    lines = ["# The tested minimum of every core dependency (#96), regenerated by",
             "# `python scripts/licensing/supply_chain.py --record`. The floor job installs this file and runs",
             "# the fast suite; a floor that stops installing fails there, not in a user's environment.",
             ""]
    for key, entry in deps.items():
        if "core" not in entry["extras"]:
            continue
        floor = floors.get("core", {}).get(key) or entry["floor"]
        if not floor:
            continue
        marker = f"; {entry['marker']}" if entry["marker"] else ""
        lines.append(f"{entry['name']}=={floor}{marker}")
    return "\n".join(lines) + "\n"


def markdown_table(licences: dict[str, Any]) -> str:
    """The table published in ``docs/licensing.md``: what was observed, then what was not and why."""
    packages = licences.get("packages") or {}
    unobserved = sorted(licences.get("unobserved") or [])
    unreadable = licences.get("unreadable") or {}
    rows = ["| Dependency | Licence it declares |", "|---|---|"]
    for name in sorted(packages):
        rows.append(f"| `{name}` | {packages[name]['licence']} |")
    if unobserved:
        rows.append("")
        rows.append("Not observed yet, because no recording environment has had them installed, so their licence "
                    "was not read rather than assumed: " + ", ".join(f"`{n}`" for n in unobserved) +
                    ". Run `python scripts/licensing/supply_chain.py --record` where they are installed.")
    if unreadable:
        rows.append("")
        rows.append("Installed, but declaring no machine-readable licence, so nothing was read from their metadata: "
                    + ", ".join(f"`{n}` ({unreadable[n]})" for n in sorted(unreadable)) +
                    ". Where a distribution ships a licence file, reading it is a human's job; the record does "
                    "not fill the row in.")
    return "\n".join(rows)


def accepted_copyleft(policy: Path = POLICY) -> list[str]:
    """The dependency names a human accepted, under the decision heading of ``docs/licensing.md``."""
    if not policy.is_file():
        return []
    text = policy.read_text(encoding="utf-8")
    start = text.find(COPYLEFT_HEADING)
    if start < 0:
        return []
    section = text[start + len(COPYLEFT_HEADING):]
    stop = section.find("\n## ")                                  # the next top-level heading
    section = section[:stop] if stop >= 0 else section
    return re.findall(r"^\s*[-*]\s+`?([A-Za-z0-9_.-]+)`?", section, re.M)


# ---------------------------------------------------------------- the audit
def audit(deps: dict[str, dict[str, Any]], floors: dict[str, Any], licences: dict[str, Any],
          policy: Path = POLICY, installed: dict[str, str] | None = None) -> list[str]:
    """Every reason this tree fails the audit. An empty list is a pass; each line names what to do."""
    problems: list[str] = []
    recorded = {**(floors.get("core") or {}), **(floors.get("extras") or {})}
    live = {name: licence_of(name) for name in sorted({e["name"] for e in deps.values()})} \
        if installed is None else installed
    for key, entry in deps.items():
        name = entry["name"]
        floor = entry["floor"]
        if floor is None:
            problems.append(f"{key}: declares no lower bound in pyproject.toml, so there is no tested minimum; "
                            f"give it one (the version this repository is actually exercised against)")
        elif key in recorded and recorded[key] != floor:
            recorded_floor = recorded[key] or "none"
            problems.append(f"{key}: pyproject.toml declares >={floor} but the tested minimum is {recorded_floor}: "
                            f"the install can now reach below what was tested -- re-record with the reason, or "
                            f"restore the bound")
        elif key not in recorded:
            problems.append(f"{key}: declared but not in docs/dev/dependency_floors.json, so its minimum was "
                            f"never recorded as tested -- run --record")

    for key in sorted(set(recorded) - set(deps)):
        problems.append(f"{key}: recorded in docs/dev/dependency_floors.json but no longer declared -- "
                        f"run --record")

    # every declared dependency is in the licence record, observed or explicitly listed as not seen
    packages = licences.get("packages") or {}
    unobserved = set(licences.get("unobserved") or [])
    unreadable = set(licences.get("unreadable") or {})
    core_names = {entry["name"] for entry in deps.values() if "core" in entry["extras"] and not entry["marker"]}
    for name in sorted({entry["name"] for entry in deps.values()}):
        if name in packages or name in unobserved or name in unreadable:
            continue
        problems.append(f"{name}: declared but not in docs/dev/dependency_licenses.json -- run --record")
    for name in sorted(core_names & unobserved):
        problems.append(f"{name}: a core dependency with no recorded licence; it is installed wherever the package "
                        f"is, so there is no reason for it to be unobserved -- run --record")
    for name in sorted(core_names & unreadable):
        problems.append(f"{name}: a core dependency whose metadata declares no machine-readable licence, and it is "
                        f"installed wherever the package is -- name its licence in {POLICY.relative_to(ROOT)} or "
                        f"replace the dependency")

    accepted = {n.lower() for n in accepted_copyleft(policy)}
    for name, licence in live.items():
        if licence.startswith(("not installed", "unknown")):
            if name in packages and licence.startswith("unknown"):
                problems.append(f"{name}: installed, but declares no machine-readable licence "
                                f"({licence}); name it in {POLICY.relative_to(ROOT)} or replace it")
            continue
        recorded = packages.get(name) or {}
        other_release = recorded.get("version") not in (None, "", _version(name))
        if recorded.get("licence") not in (None, licence) and other_release:
            # a different release of the same dependency: it declares its own licence, and the record names the
            # release it was read from (pandas 3.0.5 says "BSD License", 3.1.1 says "BSD-3-Clause"). Comparing the
            # two strings would call a true record wrong because this interpreter holds another wheel.
            continue
        if recorded.get("licence") not in (None, licence):
            problems.append(f"{name}: docs/dev/dependency_licenses.json says "
                            f"{recorded['licence']!r}, the installed distribution says {licence!r} -- "
                            f"run --record")
        if is_copyleft(licence) and name.lower() not in accepted:
            problems.append(f"{name}: copyleft licence {licence!r}; add it under {COPYLEFT_HEADING!r} in "
                            f"{POLICY.relative_to(ROOT)} first -- a copyleft dependency is a decision, not an "
                            f"accident")
    return problems


def coverage(deps: dict[str, dict[str, Any]], licences: dict[str, Any]) -> dict[str, int]:
    """How much of the declaration the licence record actually saw, so a weak audit says so."""
    names = {entry["name"] for entry in deps.values()}
    packages = set(licences.get("packages") or {})
    unobserved = set(licences.get("unobserved") or [])
    unreadable = set(licences.get("unreadable") or {})
    return {"declared": len(names), "observed": len(names & packages),
            "unobserved": len(names & unobserved), "unreadable": len(names & unreadable),
            "unaccounted": len(names - packages - unobserved - unreadable)}


def published_matches(policy: Path = POLICY, table: str | None = None) -> bool:
    """Is the table in ``docs/licensing.md`` the one the record generates?"""
    if not policy.is_file():
        return False
    text = policy.read_text(encoding="utf-8")
    start, end = text.find(MARK_BEGIN), text.find(MARK_END)
    if start < 0 or end < start:
        return False
    return text[start + len(MARK_BEGIN):end].strip() == (table or "").strip()


def write_published(policy: Path, table: str) -> None:
    text = policy.read_text(encoding="utf-8")
    start, end = text.find(MARK_BEGIN), text.find(MARK_END)
    if start < 0 or end < start:
        raise SystemExit(f"{policy} has no {MARK_BEGIN} / {MARK_END} block to fill")
    policy.write_text(text[:start + len(MARK_BEGIN)] + "\n" + table + "\n" + text[end:], encoding="utf-8")


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on a floor, licence or record problem")
    parser.add_argument("--record", action="store_true", help="rewrite the records, the constraints and the docs")
    parser.add_argument("--json", action="store_true", help="print the audit as JSON")
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    args = parser.parse_args(argv)
    deps = declared(args.pyproject)

    if args.record:
        floors = record_floors(deps, args.pyproject)
        licences = record_licences(deps)
        FLOORS.write_text(json.dumps(floors, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        LICENCES.write_text(json.dumps(licences, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        CONSTRAINTS.parent.mkdir(parents=True, exist_ok=True)
        CONSTRAINTS.write_text(constraints_text(deps, floors), encoding="utf-8")
        if POLICY.is_file():
            write_published(POLICY, markdown_table(licences))
        counts = coverage(deps, licences)
        print(f"recorded {len(deps)} declared dependencies: {counts['observed']} with a licence, "
              f"{counts['unobserved']} not installed here"
              + (f", {counts['unreadable']} installed without a readable licence" if counts["unreadable"] else ""))
        return 0

    floors, licences = load(FLOORS), load(LICENCES)
    problems = audit(deps, floors, licences)
    counts = coverage(deps, licences)
    if args.json:
        print(json.dumps({"schema": SCHEMA, "problems": problems, "coverage": counts,
                          "floors": floors, "licences": licences}, indent=1))
        return 1 if problems else 0

    if not published_matches(POLICY, markdown_table(licences)):
        problems.append(f"{POLICY.relative_to(ROOT)}: the dependency table between {MARK_BEGIN} and {MARK_END} is "
                        f"not the one docs/dev/dependency_licenses.json generates -- run --record")
    constraints = CONSTRAINTS.read_text(encoding="utf-8") if CONSTRAINTS.is_file() else ""
    if constraints.strip() != constraints_text(deps, floors).strip():
        problems.append(f"{CONSTRAINTS.relative_to(ROOT)}: not the floors this tree declares -- run --record")

    print(f"{counts['declared']} declared dependencies, {counts['observed']} with a licence on record, "
          f"{counts['unobserved']} not installed here"
          + (f", {counts['unreadable']} installed without a readable licence" if counts["unreadable"] else "")
          + (f", {counts['unaccounted']} unaccounted" if counts["unaccounted"] else ""))
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
