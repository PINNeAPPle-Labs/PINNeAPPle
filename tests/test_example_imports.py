"""Examples and templates must only import names that exist.

``scripts/example_imports_baseline.txt`` lists files that were already broken when this check was added
(they predate the refactor into the mega-modules). The check fails for any NEW broken file, and also for a
baseline file that now passes: delete its line, so the debt can only shrink.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_example_imports as chk  # noqa: E402

BASELINE = {ln.strip() for ln in (ROOT / "scripts" / "example_imports_baseline.txt").read_text().splitlines() if ln.strip()}


def _broken_files():
    return {p.split(":")[0] for p in chk.check_all()}


def test_no_new_example_with_broken_imports():
    new = sorted(_broken_files() - BASELINE)
    assert not new, "examples/templates importing names that do not exist (fix them):\n" + "\n".join(new)


def test_baseline_only_lists_files_that_are_still_broken():
    fixed = sorted(BASELINE - _broken_files())
    assert not fixed, "now importable: remove from scripts/example_imports_baseline.txt:\n" + "\n".join(fixed)
