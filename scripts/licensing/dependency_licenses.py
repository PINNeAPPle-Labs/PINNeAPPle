#!/usr/bin/env python
"""List the licence of every declared dependency, and refuse copyleft in the core.

The policy is in ``docs/licensing.md``; this is the part a machine can check.
It reads the declared dependencies from ``pyproject.toml`` (no network), looks
each one up in the environment's installed metadata, and reports what it
declares about itself:

* PEP 639 ``License-Expression`` when the distribution has one (numpy does:
  ``BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0``);
* otherwise the ``License ::`` trove classifiers;
* otherwise ``unknown`` -- printed, never guessed, so the line can be checked
  by hand rather than assumed permissive.

    python scripts/licensing/dependency_licenses.py                 # the table
    python scripts/licensing/dependency_licenses.py --fail-on-copyleft
    python scripts/licensing/dependency_licenses.py --markdown      # for the docs

``--fail-on-copyleft`` exits 1 when a declared dependency is GPL, AGPL or LGPL.
The core is distributed under Apache-2.0; a copyleft dependency is not
forbidden by that licence, but it changes what a user of the installed package
inherits, so it may not be added silently.
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPYLEFT = ("GPL", "AGPL", "LGPL", "SSPL", "EUPL")


def declared_dependencies(path: Path = ROOT / "pyproject.toml") -> list[str]:
    """The distribution names from ``[project].dependencies`` that apply to this interpreter.

    A requirement behind an environment marker that does not hold here is not part of
    this installation -- ``tomli`` is declared for ``python_version < '3.11'`` -- so it is
    left out instead of being reported as "not installed", which would read as a problem.
    """
    import tomllib

    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    names = []
    for requirement in project.get("dependencies", []):
        name = _name_if_applicable(requirement)
        if name and name not in names:
            names.append(name)
    return names


def _name_if_applicable(requirement: str) -> str | None:
    """The distribution name, or ``None`` when the requirement's marker excludes this interpreter."""
    try:
        from packaging.requirements import Requirement

        parsed = Requirement(requirement)
        if parsed.marker is not None and not parsed.marker.evaluate():
            return None
        return parsed.name
    except Exception:  # packaging absent, or a requirement it cannot parse: fall back to the name prefix
        return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0] or None


def licence_of(name: str) -> str:
    """What the installed distribution says about its own licence, or ``unknown``."""
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return "not installed"
    expression = meta.get("License-Expression")
    if expression:
        return " ".join(expression.split())
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "; ".join(c.split("::")[-1].strip() for c in classifiers)
    raw = (meta.get("License") or "").strip()
    first = raw.splitlines()[0].strip() if raw else ""
    # A full licence text is not a useful cell; name the file instead.
    if len(first) > 60 or not first:
        return "unknown (see the distribution's licence file)"
    return first


def is_copyleft(licence: str) -> bool:
    tokens = re.findall(r"[A-Z]+L-?[0-9.]*|GPL|AGPL|LGPL|SSPL|EUPL", licence.upper())
    return any(licence.upper().startswith(prefix) for prefix in COPYLEFT) or any(
        token.startswith(COPYLEFT) for token in tokens
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fail-on-copyleft", action="store_true",
                        help="exit 1 if a declared dependency is GPL, AGPL, LGPL, SSPL or EUPL")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args(argv)

    rows = [(name, licence_of(name)) for name in declared_dependencies()]
    if args.markdown:
        print("| Dependency | Licence it declares |")
        print("|---|---|")
        for name, licence in rows:
            print(f"| `{name}` | {licence} |")
    else:
        width = max(len(name) for name, _ in rows)
        for name, licence in rows:
            print(f"{name:{width}}  {licence}")

    offenders = [(name, licence) for name, licence in rows if is_copyleft(licence)]
    unknown = [name for name, licence in rows if licence.startswith(("unknown", "not installed"))]
    if unknown:
        print(f"\n{len(unknown)} dependency(ies) declare no machine-readable licence: {', '.join(unknown)}",
              file=sys.stderr)
    if args.fail_on_copyleft and offenders:
        print("\ncopyleft dependency declared in the core:", file=sys.stderr)
        for name, licence in offenders:
            print(f"  {name}: {licence}", file=sys.stderr)
        print("See docs/licensing.md: a copyleft dependency is a decision, not an accident.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
