"""The example catalog: every example runs in CI or is manual with a measured reason (#41).

``examples/`` has about a hundred folders and, before this, nothing said which of them still work: the import check
proves a name exists and the manifests say what an example is about, not whether anyone ran it. These tests hold the
three things that make the declaration worth reading: every folder is covered, the CI tier is affordable and runnable,
and the reasons are facts -- a broken import or an unavailable package -- rather than prose somebody typed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import example_catalog as ec  # noqa: E402


def _tiers(**overrides):
    tiers = dict(ec.load_tiers())
    tiers.update(overrides)
    return tiers


def test_every_example_is_either_run_or_explained():
    problems = ec.check()
    assert problems == [], "\n".join(problems)


def test_the_ci_tier_is_affordable_and_runnable():
    tiers = ec.load_tiers()
    caps = [int(entry["max_seconds"]) for entry in tiers["ci"]]
    assert caps and sum(caps) <= int(tiers["budget_seconds"])
    for entry in tiers["ci"]:
        folder = ROOT / entry["id"]
        assert (folder / "experiment.json").is_file(), entry["id"]
        for name in entry["command"]:
            assert (folder / name).is_file(), f"{entry['id']} runs {name}, which is not there"


def test_a_folder_that_is_neither_run_nor_explained_is_reported():
    tiers = _tiers()
    victim = next(iter(tiers["manual"]))
    tiers["manual"].pop(victim)
    problems = ec.check(tiers)
    assert any(victim in p and "neither" in p for p in problems), problems


def test_a_reason_that_was_never_measured_is_reported():
    tiers = _tiers()
    victim = next(iter(tiers["manual"]))
    tiers["manual"][victim] = "not in the ci tier and not measured yet: run scripts/example_catalog.py --measure"
    assert any(victim in p and "no measured reason" in p for p in ec.check(tiers))


def test_a_ci_entry_that_cannot_run_is_reported():
    tiers = _tiers()
    tiers["ci"] = list(tiers["ci"]) + [{"id": "examples/first_example", "command": ["nope.py"], "max_seconds": 10}]
    problems = ec.check(tiers)
    assert any("twice" in p for p in problems), problems
    tiers["ci"] = [{"id": "examples/nowhere", "command": ["x.py"], "max_seconds": 10}]
    assert any("not an example folder" in p for p in ec.check(tiers))


def test_the_tier_may_not_exceed_its_budget():
    tiers = _tiers(budget_seconds=1)
    assert any("budget" in p for p in ec.check(tiers))


def test_a_known_broken_example_cannot_enter_the_ci_tier():
    """The import gate and this one share the checker, so a fixed import cannot linger in one of them."""
    broken = ec._broken_imports()
    if not broken:
        return
    first = sorted(broken)[0]
    folder = (ROOT / first).parent
    rel = folder.relative_to(ROOT).as_posix()
    if not (folder / "experiment.json").is_file():
        return
    script = Path(first).name
    tiers = _tiers(ci=[{"id": rel, "command": [script], "max_seconds": 10}], manual={})
    assert any("does not exist yet" in p for p in ec.check(tiers))


def test_the_page_lists_every_example():
    tiers = ec.load_tiers()
    page = ec.docs(tiers)
    for entry in tiers["ci"]:
        assert entry["id"] in page
    for rel in tiers["manual"]:
        assert rel in page
    assert page.count("|") > len(tiers["ci"]) + len(tiers["manual"])
    on_disk = (ROOT / "docs" / "guides" / "example_catalog.md").read_text(encoding="utf-8")
    assert on_disk == page, "regenerate with: python scripts/example_catalog.py --docs"


def test_running_the_tier_is_what_ci_does(tmp_path):
    """`--run` executes the declared commands; a doctored entry fails the tier instead of passing silently."""
    tiers = {"schema": ec.SCHEMA, "budget_seconds": 60,
             "ci": [{"id": "examples/first_example", "command": ["-c", "print('ok')"], "max_seconds": 30}],
             "manual": {}}
    path = tmp_path / "run_tiers.json"
    path.write_text(json.dumps(tiers), encoding="utf-8")
    assert ec.run_ci(tiers) == 0
    tiers["ci"][0]["command"] = ["-c", "raise SystemExit(1)"]
    assert ec.run_ci(tiers) == 1
