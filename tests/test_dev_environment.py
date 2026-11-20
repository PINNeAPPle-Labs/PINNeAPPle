"""The development environment has one command per job, and the container uses the same ones (#50).

The promise in ``CONTRIBUTING.md`` is that a contributor runs the fast suite five minutes after cloning. That is
only true if the commands exist, if the suite they run is short, and if the dev container runs the same setup
rather than a second copy of it that can drift. This checks the first and the third; the second is the length of
the list in ``Makefile``, which is checked here against ``CONTRIBUTING.md`` so the two cannot disagree.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"

TARGETS = {"setup", "test", "e2e", "test-full", "lint", "gates", "docs", "clean"}


def _makefile_text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_the_makefile_has_a_target_per_job():
    text = _makefile_text()
    declared = set(re.findall(r"^([a-z][a-z0-9-]*):", text, flags=re.M))
    missing = TARGETS - declared
    assert not missing, f"Makefile is missing {sorted(missing)}"
    assert ".PHONY:" in text, "every target should be .PHONY: none of them names a file"


def test_the_fast_suite_is_a_short_explicit_list():
    text = _makefile_text()
    match = re.search(r"^FAST =(.*?)(?=^[A-Z.])", text, flags=re.M | re.S)
    assert match, "the Makefile should name the fast suite in a FAST variable"
    files = re.findall(r"tests/[\w/]+\.py", match.group(1))
    assert 5 <= len(files) <= 12, f"fast suite is {len(files)} files; it is meant to run in under a minute"
    for name in files:
        assert (ROOT / name).is_file(), f"{name} is in the fast suite but does not exist"


def test_the_fast_suite_in_the_makefile_is_the_one_contributing_documents():
    """A contributor reading CONTRIBUTING.md must run the same list `make test` runs."""
    makefile = _makefile_text()
    match = re.search(r"^FAST =(.*?)(?=^[A-Z.])", makefile, flags=re.M | re.S)
    in_make = sorted(re.findall(r"tests/[\w/]+\.py", match.group(1)))
    contributing = CONTRIBUTING.read_text(encoding="utf-8")
    section = contributing.split("## Running tests", 1)[1].split("\n## ", 1)[0]
    in_docs = sorted(re.findall(r"tests/[\w/]+\.py", section))
    assert in_make == in_docs, f"Makefile and CONTRIBUTING.md disagree: {in_make} != {in_docs}"


def test_contributing_documents_the_commands():
    contributing = CONTRIBUTING.read_text(encoding="utf-8")
    for command in ("make setup", "make test", "make e2e", "make lint", "make gates"):
        assert command in contributing, f"CONTRIBUTING.md does not mention `{command}`"


def test_the_dev_container_runs_the_same_setup():
    raw = DEVCONTAINER.read_text(encoding="utf-8")
    # devcontainer.json is JSONC: comments are allowed by the spec, so strip them before parsing.
    uncommented = re.sub(r"(?m)^\s*//.*$", "", raw)
    config = json.loads(uncommented)
    assert config["postCreateCommand"].startswith("make setup")
    assert config["remoteUser"] and config["image"]
