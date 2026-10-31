"""Changelog fragments: one file per pull request in ``changelog.d/``, folded into CHANGELOG.md at release time.

Editing the ``## [Unreleased]`` section of CHANGELOG.md in every pull request made each merge conflict with every
other open pull request. A fragment is a new file, so parallel pull requests never touch the same lines.

Fragment name: ``changelog.d/<slug>.<category>.md`` where ``<slug>`` is anything unique (the branch name, an issue
number) and ``<category>`` is one of the CHANGELOG headings below. The body is the Markdown of the entry, usually
one ``- `` bullet (sub-bullets allowed), written exactly as it would appear under that heading.

    python scripts/changelog.py check               # every fragment is well named and non-empty (CI)
    python scripts/changelog.py render              # print the Unreleased section as it would be released
    python scripts/changelog.py release 0.7.0       # fold Unreleased + fragments under "## [0.7.0] - <today>"
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
FRAGMENTS = ROOT / "changelog.d"
# file suffix -> CHANGELOG heading, in the order the headings appear in a release
CATEGORIES = {"security": "Security", "added": "Added", "changed": "Changed", "fixed": "Fixed",
              "removed": "Removed", "documentation": "Documentation", "known-issues": "Known issues"}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(" + "|".join(map(re.escape, CATEGORIES)) + r")\.md$")
UNRELEASED = re.compile(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[)", re.M | re.S)


def fragments(folder: Path = FRAGMENTS) -> list[Path]:
    return sorted(p for p in folder.glob("*.md") if p.name != "README.md") if folder.is_dir() else []


def check(folder: Path = FRAGMENTS) -> list[str]:
    problems = []
    for p in fragments(folder):
        if not NAME.match(p.name):
            problems.append(f"{p.name}: name must be <slug>.<category>.md with category in {', '.join(CATEGORIES)}")
        elif not p.read_text(encoding="utf-8").strip():
            problems.append(f"{p.name}: empty fragment")
    return problems


def _sections(body: str) -> dict[str, list[str]]:
    """Split an Unreleased body into {heading: [text blocks]} keeping each block verbatim."""
    out: dict[str, list[str]] = {}
    current = None
    for block in re.split(r"^### ", body, flags=re.M):
        if current is None and not block.strip():
            current = ""
            continue
        heading, _, text = block.partition("\n")
        current = heading.strip()
        if text.strip():
            out.setdefault(current, []).append(text.strip("\n"))
    return out


def render(changelog: str | None = None, folder: Path = FRAGMENTS) -> str:
    """The Unreleased entries already in CHANGELOG.md plus every fragment, grouped by heading."""
    changelog = CHANGELOG.read_text(encoding="utf-8") if changelog is None else changelog
    m = UNRELEASED.search(changelog)
    sections = _sections(m.group(1)) if m else {}
    for p in fragments(folder):
        cat = NAME.match(p.name)
        if cat:
            sections.setdefault(CATEGORIES[cat.group(1)], []).append(p.read_text(encoding="utf-8").strip("\n"))
    order = list(CATEGORIES.values()) + [h for h in sections if h not in CATEGORIES.values()]
    parts = [f"### {h}\n\n" + "\n".join(sections[h]) for h in order if sections.get(h)]
    return "\n\n".join(parts) + ("\n" if parts else "")


def release(version: str, date: str | None = None, folder: Path = FRAGMENTS) -> None:
    """Move the Unreleased entries and the fragments under a dated version heading, then delete the fragments."""
    problems = check(folder)
    if problems:
        raise SystemExit("\n".join(problems))
    text = CHANGELOG.read_text(encoding="utf-8")
    m = UNRELEASED.search(text)
    if not m:
        raise SystemExit("CHANGELOG.md has no '## [Unreleased]' section followed by a version heading")
    date = date or dt.date.today().isoformat()
    body = render(text, folder)
    new = f"## [Unreleased]\n\n\n## [{version}] - {date}\n\n{body}\n"
    CHANGELOG.write_text(text[:m.start()] + new + text[m.end():], encoding="utf-8")
    for p in fragments(folder):
        p.unlink()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        errs = check()
        print("\n".join(errs) if errs else f"changelog.d ok ({len(fragments())} fragments)")
        sys.exit(1 if errs else 0)
    elif cmd == "render":
        print(render(), end="")
    elif cmd == "release" and len(sys.argv) in (3, 4):
        release(sys.argv[2].lstrip("v"), sys.argv[3] if len(sys.argv) == 4 else None)
    else:
        sys.exit(__doc__)
