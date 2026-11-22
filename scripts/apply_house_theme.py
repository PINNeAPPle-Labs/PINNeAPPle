"""Put every figure of an example under the house theme (#918), in batches, with a gate that keeps it there.

Why a codemod and not hand edits: 151 example scripts import matplotlib and 23 used the house style. Migrating them
one at a time is a change nobody can review and nothing keeps; a mechanical edit plus a gate that reads the files is
reviewable line by line and fails the day somebody adds a figure with matplotlib defaults.

    python scripts/apply_house_theme.py --check examples/first_example/first_example.py
    python scripts/apply_house_theme.py examples/a.py examples/b/c.py
    python scripts/apply_house_theme.py --baseline        # rewrite the baseline of not-yet-migrated files

The edit is one import and one call, placed so it applies to every figure the module makes, plus the removal of a
``plt.rcParams.update`` that only restates what the theme already sets:

    from pinneapple_tools.visualization.style import use_theme
    use_theme("report")

What the gate accepts: a module that calls ``use_theme(...)`` or the older ``use_cfd_style(...)``. A module that
draws with matplotlib and does neither is either migrated or listed in ``examples/.house_theme_baseline`` — the list only shrinks.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BASELINE = EXAMPLES / ".house_theme_baseline"

#: a figure is drawn here if the module imports pyplot or calls savefig/show
_DRAWS = re.compile(r"^\s*(?:import matplotlib\.pyplot|from matplotlib import pyplot|import matplotlib\b)", re.M)
_SAVES = re.compile(r"\.(?:savefig|show)\(")
_THEMED = re.compile(r"\b(?:use_theme|use_cfd_style)\s*\(")
_MPL_IMPORT = re.compile(r"^(?:from matplotlib[^\n]*|import matplotlib[^\n]*)$", re.M)



def _rel(path: Path) -> str:
    """Path relative to the repository, for a path given as ``examples/...`` or as an absolute one."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:                                   # outside the repository: print it as given
        return str(path)


def draws(path: Path) -> bool:
    """True when the module draws figures: it imports pyplot, or it saves or shows something."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return bool(_DRAWS.search(text)) or ("matplotlib" in text and bool(_SAVES.search(text)))


def themed(path: Path) -> bool:
    return bool(_THEMED.search(path.read_text(encoding="utf-8", errors="replace")))


def files() -> list[Path]:
    return sorted(p for p in EXAMPLES.rglob("*.py") if p.is_file())


def offenders() -> list[Path]:
    return [p for p in files() if draws(p) and not themed(p)]


def _theme_lines(indent: str) -> list[str]:
    """The theme import and call, indented like the block they go into (a script that imports pyplot inside a
    ``try`` block gets the theme call there too, so the two stay together when matplotlib is missing)."""
    if indent:
        # inside a try/function the theme import is its own isort section: blank line before it, none after it
        return ["\n",
                f"{indent}from pinneapple_tools.visualization.style import use_theme\n",
                f"{indent}use_theme(\"report\")   # one theme for every figure of the library (#918)\n"]
    return ["from pinneapple_tools.visualization.style import use_theme\n",
            "\n",
            "use_theme(\"report\")   # one theme for every figure of the library (#918)\n"]


_MPL_LINE = re.compile(r"^(?P<indent>[ \t]*)(?:from matplotlib[^\n]*|import matplotlib[^\n]*)$")
_TOP_IMPORT = re.compile(r"^(?:import |from )\S")

#: what ``use_theme("report")`` already sets, with the value it sets. An example that repeats one of these in
#: ``plt.rcParams.update`` is not choosing a look, only a second place to edit: the migration drops the line.
#: ``tests/test_house_theme_gate.py`` compares this table against the theme's own rc so it cannot drift.
_REDUNDANT = {"font.size": 10.0, "axes.spines.top": False, "axes.spines.right": False}
_RC_UPDATE = re.compile(r"^[ \t]*plt\.rcParams\.update\((?P<body>\{.*\})\)[ \t]*$")


def _repeats_the_theme(line: str) -> bool:
    """True when the line is a ``plt.rcParams.update`` that only restates what the theme sets."""
    m = _RC_UPDATE.match(line)
    if not m:
        return False
    try:
        values = ast.literal_eval(m.group("body"))
    except (SyntaxError, ValueError):
        return False
    return bool(values) and all(key in _REDUNDANT and _REDUNDANT[key] == value
                                for key, value in values.items())


def migrate(path: Path, *, dry_run: bool = False) -> bool:
    """Add the theme import and call to ``path``. True when it changed.

    Two shapes, because the examples have two: a module-level import block (the call goes after the last top-level
    import, so the import stays sorted and runs before any figure) and a ``try``/function import (the call goes
    right after it, at the same indentation, so it only runs where matplotlib did)."""
    text = path.read_text(encoding="utf-8")
    if _THEMED.search(text):
        return False
    lines = text.splitlines(keepends=True)
    mpl = [i for i, line in enumerate(lines) if _MPL_LINE.match(line.rstrip("\n"))]
    if not mpl:
        return False
    top = [i for i in mpl if not _MPL_LINE.match(lines[i].rstrip("\n")).group("indent")]
    if top:
        after = max(i for i, line in enumerate(lines) if _TOP_IMPORT.match(line) and i < 200) if any(
            _TOP_IMPORT.match(line) for line in lines[:200]) else top[-1]
        at, indent = after, ""
    else:
        at = mpl[-1]
        indent = _MPL_LINE.match(lines[at].rstrip("\n")).group("indent")
    new_lines = lines[:at + 1] + _theme_lines(indent) + lines[at + 1:]
    # a script that hand-set what the theme sets keeps a second source of truth: drop those lines
    new_lines = [line for line in new_lines if not _repeats_the_theme(line.rstrip("\n"))]
    if not dry_run:
        path.write_text("".join(new_lines), encoding="utf-8")
    return True


def write_baseline(paths: list[Path] | None = None) -> Path:
    """Record the files that still draw without the theme, so the gate fails only on new ones."""
    lines = ["# Examples that draw with matplotlib defaults and are not under the house theme yet (#918).",
             "# One path per line, relative to the repository root. A line goes when its file is migrated:",
             "#     python scripts/apply_house_theme.py <files> && python scripts/apply_house_theme.py --baseline",
             "# so this list only shrinks."]
    for p in sorted(paths or offenders()):
        lines.append(_rel(p))
    BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BASELINE


def read_baseline() -> set[str]:
    if not BASELINE.is_file():
        return set()
    return {line.strip() for line in BASELINE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def problems(examples: Path | None = None, baseline_path: Path | None = None) -> list[str]:
    """What the gate reports: a new figure outside the theme, or a baseline entry that is already fixed.

    ``examples`` and ``baseline_path`` exist so the gate can be pointed at a folder that is not the repository's
    (that is how its own tests exercise it); the defaults are what CI and the local gate run.
    """
    examples = Path(examples or EXAMPLES).resolve()
    baseline_path = Path(baseline_path or BASELINE)
    root = examples.parent
    sources = sorted(p for p in examples.rglob("*.py") if p.is_file())
    baseline = ({ln.strip() for ln in baseline_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")} if baseline_path.is_file() else set())

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:                                   # pointed outside the repository
            return str(path)

    out = []
    for path in sources:
        name = rel(path)
        if draws(path) and not themed(path) and name not in baseline:
            out.append(f"{name}: draws with matplotlib and neither use_theme nor use_cfd_style is called")
        if themed(path) and name in baseline:
            out.append(f"{name}: already themed but still listed in {baseline_path.name} — drop the line")
    kept = {rel(path) for path in sources}
    out.extend(f"{name}: listed in the baseline but the file is gone" for name in sorted(baseline - kept))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apply_house_theme", description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path, help="example scripts to migrate")
    ap.add_argument("--check", action="store_true", help="report what is missing and change nothing")
    ap.add_argument("--baseline", action="store_true", help="rewrite the baseline from what is not migrated yet")
    a = ap.parse_args(argv)
    if a.baseline:
        print(write_baseline())
        return 0
    if a.check:
        problems_ = problems()
        for line in problems_:
            print(line)
        print(f"{len(problems_)} problem(s)")
        return 1 if problems_ else 0
    if not a.paths:
        ap.error("give the scripts to migrate (or --check / --baseline)")
    changed = [p for p in a.paths if migrate(p)]
    for p in changed:
        print(f"themed {_rel(p)}")
    print(f"{len(changed)} of {len(a.paths)} changed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
