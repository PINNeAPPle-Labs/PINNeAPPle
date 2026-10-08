"""The release check: current version is consistent, and a mismatch is caught."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_release_version as crv  # noqa: E402


def _current():
    return re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)


def test_all_version_sources_agree_with_the_current_version():
    found = crv.versions("v" + _current())
    assert len(set(found.values())) == 1, found


def test_changelog_has_a_dated_heading_for_the_current_version_unless_it_is_unreleased_work():
    problems = crv.check("v" + _current())
    assert not any("version mismatch" in p for p in problems)


def test_a_wrong_tag_is_reported():
    problems = crv.check("v9.9.9")
    assert any("version mismatch" in p for p in problems) and any("CHANGELOG" in p for p in problems)
