"""The supply-chain audit (#96): floors, licences, and the two records the gate reads.

The interesting cases are the failures. A dependency audit that only passes on a clean tree has not been tested:
each test below breaks the tree in one way and asks for the reason, next to the one test that runs the real gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "licensing"
SCRIPT = SCRIPT_DIR / "supply_chain.py"

sys.path.insert(0, str(SCRIPT_DIR))
import supply_chain  # noqa: E402

PYPROJECT = """
[project]
name = "example"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "tomli>=2.0; python_version < '3.11'",
]
[project.optional-dependencies]
viz = ["plotly>=5.18", "mujoco>=3.17,<4; python_version < '3.11'", "mujoco>=4.0; python_version >= '3.11'"]
"""


@pytest.fixture()
def project(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT, encoding="utf-8")
    return path


def _records(deps, core=None, extras=None, packages=None, unobserved=()):
    floors = {"core": core if core is not None else {k: e["floor"] for k, e in deps.items()
                                                    if "core" in e["extras"]},
              "extras": extras if extras is not None else {k: e["floor"] for k, e in deps.items()
                                                          if "core" not in e["extras"]}}
    licences = {"packages": packages if packages is not None else
                {e["name"]: {"licence": "MIT"} for e in deps.values()},
                "unobserved": list(unobserved)}
    return floors, licences


def _audit(project, floors, licences, installed=None, policy=None):
    return supply_chain.audit(supply_chain.declared(project), floors, licences,
                              policy=policy or Path("/nonexistent/licensing.md"),
                              installed={} if installed is None else installed)


# ---------------------------------------------------------------- what the tree declares
def test_a_dependency_declared_for_two_python_versions_keeps_both_floors(project):
    deps = supply_chain.declared(project)
    mujoco = [k for k in deps if k.startswith("mujoco[")]
    assert len(mujoco) == 2
    assert sorted(deps[k]["floor"] for k in mujoco) == ["3.17", "4.0"]


def test_the_key_says_which_extra_declares_it(project):
    deps = supply_chain.declared(project)
    assert deps["numpy"]["extras"] == ["core"] and deps["plotly"]["extras"] == ["viz"]
    assert deps["tomli[python_version<\"3.11\"]"]["extras"] == ["core"]


def test_a_lower_bound_becomes_the_floor_and_a_bare_name_has_none():
    assert supply_chain._floor(None, "numpy>=1.24") == "1.24"
    assert supply_chain._floor(None, "zarr") is None
    assert supply_chain._floor(None, "onnx>=1.15,<2") == "1.15"


# ---------------------------------------------------------------- the floors record
def test_a_loosened_bound_fails_because_the_install_can_go_below_what_was_tested(project, tmp_path):
    deps = supply_chain.declared(project)
    floors, licences = _records(deps)
    floors["core"]["numpy"] = "1.26"
    problems = _audit(project, floors, licences)
    assert any("numpy: pyproject.toml declares >=1.24 but the tested minimum is 1.26" in p for p in problems)


def test_a_dependency_with_no_lower_bound_fails(project, tmp_path):
    """`xarray` and `zarr` were declared without a floor; the audit asks for one rather than inventing it."""
    text = project.read_text(encoding="utf-8").replace('"numpy>=1.24"', '"numpy>=1.24", "zarr"')
    project.write_text(text, encoding="utf-8")
    floors, licences = _records(supply_chain.declared(project))
    problems = _audit(project, floors, licences)
    assert any(p.startswith("zarr: declares no lower bound") for p in problems)


def test_a_new_dependency_without_a_recorded_floor_fails(project):
    floors, licences = _records(supply_chain.declared(project))
    floors["extras"].pop("plotly")
    assert any("plotly: declared but not in docs/dev/dependency_floors.json" in p
               for p in _audit(project, floors, licences))


def test_a_record_for_a_dependency_that_is_gone_fails(project):
    floors, licences = _records(supply_chain.declared(project))
    floors["extras"]["gone"] = "1.0"
    assert any("gone: recorded in docs/dev/dependency_floors.json but no longer declared" in p
               for p in _audit(project, floors, licences))


def test_the_constraints_file_pins_every_core_floor_and_keeps_the_marker(project):
    deps = supply_chain.declared(project)
    floors, _ = _records(deps)
    text = supply_chain.constraints_text(deps, floors)
    assert "numpy==1.24" in text
    assert 'tomli==2.0; python_version<"3.11"' in text
    assert "plotly" not in text                       # the floor job installs the core, not every extra


# ---------------------------------------------------------------- the licence record
def test_a_copyleft_dependency_fails_until_it_is_a_decision_in_the_policy_page(project, tmp_path):
    floors, licences = _records(supply_chain.declared(project), packages={"numpy": {"licence": "GPL-3.0-only"}})
    policy = tmp_path / "licensing.md"
    policy.write_text("## Dependencies\n\nnothing decided\n", encoding="utf-8")
    problems = _audit(project, floors, licences, installed={"numpy": "GPL-3.0-only"}, policy=policy)
    assert any("numpy: copyleft licence" in p for p in problems)

    policy.write_text(supply_chain.COPYLEFT_HEADING + "\n\n- `numpy` -- decided, and here is why.\n", encoding="utf-8")
    assert not [p for p in _audit(project, floors, licences, installed={"numpy": "GPL-3.0-only"}, policy=policy)
                if "copyleft" in p]


def test_a_licence_that_disagrees_with_the_record_fails(project):
    """A new release can change its licence; the record is what the reader trusts, so it has to be current."""
    floors, licences = _records(supply_chain.declared(project), packages={"numpy": {"licence": "BSD-3-Clause"}})
    problems = _audit(project, floors, licences, installed={"numpy": "Apache-2.0"})
    assert any("numpy: docs/dev/dependency_licenses.json says 'BSD-3-Clause'" in p for p in problems)


def test_a_core_dependency_may_not_be_unobserved(project):
    """tomli is core but carries a marker, so it is absent on 3.11+ and that is not a problem; numpy is not."""
    floors, licences = _records(supply_chain.declared(project), unobserved=("numpy", "tomli"))
    problems = _audit(project, floors, licences)
    assert any(p.startswith("numpy: a core dependency with no recorded licence") for p in problems)
    assert not any(p.startswith("tomli: a core dependency") for p in problems)


def test_an_unreadable_licence_is_a_failure_not_an_assumption(project):
    floors, licences = _records(supply_chain.declared(project), packages={"numpy": {"licence": "unknown"}})
    problems = _audit(project, floors, licences,
                      installed={"numpy": "unknown (see the distribution's licence file)"})
    assert any("numpy: installed, but declares no machine-readable licence" in p for p in problems)


# ---------------------------------------------------------------- the tree as it is
def test_the_repository_passes_its_own_audit():
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True,
                            cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_published_table_is_the_one_the_record_generates():
    licences = json.loads((ROOT / "docs" / "dev" / "dependency_licenses.json").read_text(encoding="utf-8"))
    assert supply_chain.published_matches(supply_chain.POLICY, supply_chain.markdown_table(licences))


def test_every_core_dependency_has_a_floor_and_a_licence():
    deps = supply_chain.declared(supply_chain.PYPROJECT)
    floors = json.loads(supply_chain.FLOORS.read_text(encoding="utf-8"))
    licences = json.loads(supply_chain.LICENCES.read_text(encoding="utf-8"))
    core = [k for k, e in deps.items() if "core" in e["extras"]]
    assert core and all(floors["core"].get(k) for k in core)
    assert all(e["name"] in licences["packages"] for k, e in deps.items()
               if "core" in e["extras"] and not e["marker"])
