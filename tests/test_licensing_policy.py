"""The licensing policy (#48): the machine-checkable half of docs/licensing.md."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "licensing" / "dependency_licenses.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT, timeout=120)


def test_every_declared_dependency_is_listed():
    sys.path.insert(0, str(SCRIPT.parent))
    from dependency_licenses import declared_dependencies

    declared = declared_dependencies()
    assert "numpy" in declared and "torch" in declared
    assert all(";" not in d and ">" not in d for d in declared)  # specifiers and markers stripped


def test_a_requirement_for_another_interpreter_is_not_reported_as_missing(tmp_path):
    """`tomli` is declared for python_version < '3.11'; on 3.11+ it is not part of the install, not broken."""
    sys.path.insert(0, str(SCRIPT.parent))
    from dependency_licenses import declared_dependencies

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'dependencies = ["numpy>=1.24", "tomli>=2.0; python_version < \'3.7\'", "typing-extensions>=4.8"]\n',
        encoding="utf-8",
    )
    assert declared_dependencies(pyproject) == ["numpy", "typing-extensions"]


def test_the_core_has_no_copyleft_dependency():
    """The policy allows a copyleft dependency only as a decision recorded in docs/licensing.md first."""
    result = _run("--fail-on-copyleft")
    assert result.returncode == 0, result.stderr


def test_copyleft_is_detected_in_the_forms_it_actually_appears():
    sys.path.insert(0, str(SCRIPT.parent))
    from dependency_licenses import is_copyleft

    assert is_copyleft("GPL-3.0-only")
    assert is_copyleft("AGPL-3.0")
    assert is_copyleft("LGPL-2.1-or-later")
    assert is_copyleft("GNU General Public License v3 (GPLv3)")
    # what this repository's own dependencies declare, none of it copyleft
    assert not is_copyleft("MIT")
    assert not is_copyleft("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0")
    assert not is_copyleft("MPL-2.0 AND MIT")
    assert not is_copyleft("Apache-2.0 AND Apache-2.0 WITH LLVM-exception")


def test_the_policy_page_and_the_notice_exist_and_are_linked():
    policy = (ROOT / "docs" / "licensing.md").read_text(encoding="utf-8")
    assert "no CLA" in policy and "NOTICE" in policy and "dependency_licenses.py" in policy
    assert (ROOT / "NOTICE").is_file() and (ROOT / "LICENSE").is_file()
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "docs/licensing.md" in contributing
    template = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert "dependency_licenses.py" in template and "NOTICE" in template
