"""Fail unless the git tag, pyproject.toml, CITATION.cff, pinneapple.__version__ and CHANGELOG.md agree on the release version.

Usage: ``python scripts/check_release_version.py v0.6.3`` (the tag name, with or without the leading ``v``).
Used by the release workflow so a tag can never publish files whose version says something else.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def versions(tag: str) -> dict[str, str]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m_py = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    m_cff = re.search(r'^version:\s*"?([^"\s]+)"?', citation, re.M)
    init = (ROOT / "pinneapple" / "__init__.py").read_text(encoding="utf-8")
    m_init = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
    return {"tag": tag.lstrip("v"), "pyproject.toml": m_py.group(1) if m_py else "?",
            "CITATION.cff": m_cff.group(1) if m_cff else "?",
            "pinneapple/__init__.py": m_init.group(1) if m_init else "?"}


def check(tag: str) -> list[str]:
    found = versions(tag)
    problems = []
    if len(set(found.values())) != 1:
        problems.append("version mismatch: " + ", ".join(f"{k}={v}" for k, v in found.items()))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(found['tag'])}\] - \d{{4}}-\d{{2}}-\d{{2}}", changelog, re.M):
        problems.append(f"CHANGELOG.md has no dated heading '## [{found['tag']}] - YYYY-MM-DD' "
                        "(move the Unreleased entries under it)")
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    errors = check(sys.argv[1])
    for e in errors:
        print("ERROR:", e)
    sys.exit(1 if errors else 0)
