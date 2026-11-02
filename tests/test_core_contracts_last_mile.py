"""Core coverage (#1129, toward 1.0): mesh quality and its report, the onboarding commands, the command-line entry
points of the validation arena and of fault injection, the fault random streams and the IPM machine cache.

References: a regular hexahedral box mesh has zero non-orthogonality and skewness (OpenFOAM's checkMesh
definitions) and its cell volumes add up to the box; the damaged STL example has open edges and is not watertight;
the same fault stream is drawn twice from the same seed.
"""
import contextlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_analysis.validation.arena import __main__ as ARENA
from pinneapple_analysis.validation.fault_injection import __main__ as FAULTS
from pinneapple_analysis.validation.fault_injection import matrix as MX
from pinneapple_core.onboarding import cli as CLI
from pinneapple_core.onboarding import newproject as NP
from pinneapple_data.cae import inp as INP
from pinneapple_data.cae import io as IO
from pinneapple_data.cae import quality as Q
from pinneapple_data.cae import report as RP

DOC = importlib.import_module("pinneapple_core.onboarding.doctor")   # the package also has a doctor()
EX = Path(__file__).resolve().parents[1] / "apps" / "mesh_quality" / "examples"


def _quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


# ------------------------------------------------------------------------------------------------ mesh quality
@pytest.fixture(scope="module")
def hexbox():
    return INP.deck_mesh(INP.read_deck((EX / "calculix_cantilever_hex.inp").read_text()))


def test_finite_volume_metrics_of_a_regular_box(hexbox):
    fv = Q.fv_metrics(hexbox)
    assert fv["face"]["non_orthogonality"].max() == pytest.approx(0.0, abs=1e-9)          # checkMesh: 0 deg
    assert fv["face"]["skewness_internal"].max() == pytest.approx(0.0, abs=1e-9)
    lo, hi = np.asarray(hexbox.bbox()).reshape(2, 3) if np.asarray(hexbox.bbox()).size == 6 else \
        (hexbox.points.min(0), hexbox.points.max(0))
    assert fv["cg"]["volumes"].sum() == pytest.approx(float(np.prod(np.asarray(hi) - np.asarray(lo))))  # the box


def test_surface_checks_of_a_damaged_stl():
    s = IO.read_stl((EX / "bracket_surface_damaged.stl").read_bytes())
    sc = Q.surface_checks(s)
    assert sc["open_edges"] > 0 and not sc["watertight"] and sc["enclosed_volume"] is None
    assert sc["faces"] == s.n_cells and sc["degenerate_faces"] == 0


def test_report_and_its_view(hexbox):
    rep = RP.build(hexbox)
    assert rep["summary"]["cells"] == 640 and rep["status"] and isinstance(rep["checks"], list)
    v = RP.view_payload(hexbox, rep)
    assert {"points", "tris", "fields"} <= set(v) and len(v["tris"]) > 0
    assert RP.build(IO.read_stl((EX / "bracket_surface_damaged.stl").read_bytes()))["summary"]["kind"]


# ------------------------------------------------------------------------------------------------ onboarding
def test_doctor_profiles_and_checks():
    p = DOC.profiles()
    assert "start" in p and all(isinstance(v, str) and v for v in p.values())
    c = DOC.Check("torch", False, "not installed", fix="pip install torch")
    assert c.line().startswith("  [MISS] torch") and "fix: pip install torch" in c.line()
    assert DOC.Check("x", None).line().startswith("  [?   ]")
    rc, out = _quiet(CLI.doctor_main, ["start", "--json"])
    assert rc in (0, 1) and json.loads(out)
    assert CLI.doctor_main(["no-such-profile", "-q"]) == 2


def test_recommend_and_card_commands():
    rc, out = _quiet(CLI.recommend_main, ["viscous", "burgers", "shock", "--json"])
    assert rc == 0 and "burgers_1d" in out
    rc, out = _quiet(CLI.card_main, ["burgers_1d", "--json"])
    assert rc == 0 and json.loads(out)["name"] == "burgers_1d"
    rc, out = _quiet(CLI.card_main, ["--list"])
    assert rc == 0 and "burgers_1d" in out


def test_new_project_writes_a_runnable_layout(tmp_path):
    root = NP.new_project("my_case", dir=str(tmp_path), verbose=False)
    files = {str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file()}
    assert any(f.endswith(".py") for f in files) and Path(root).name == "my_case"
    for f in files:
        if f.endswith(".py"):
            compile((Path(root) / f).read_text(), f, "exec")                                  # valid Python
    with pytest.raises(ValueError):
        NP.new_project("Bad Name", dir=str(tmp_path), verbose=False)
    rc, out = _quiet(CLI.new_main, ["other_case", "--dir", str(tmp_path), "--dry-run"])
    assert rc == 0 and not (tmp_path / "other_case").exists()                                 # dry run writes nothing


# ------------------------------------------------------------------------------------------------ CLIs, streams
def test_fault_streams_replay():
    a = MX.fault_rng(0, "bias", 0.05, 7).normal(size=5)
    assert np.array_equal(a, MX.fault_rng(0, "bias", 0.05, 7).normal(size=5))
    assert not np.array_equal(a, MX.fault_rng(0, "bias", 0.1, 7).normal(size=5))


def test_fault_injection_and_arena_command_lines(tmp_path):
    rc, out = _quiet(FAULTS.main, ["--problems", "burgers_1d", "--small", "--n-cases", "3", "--out", str(tmp_path)])
    assert rc == 0                                                          # with --out the tables go to files
    assert json.loads((tmp_path / "check_power_burgers_1d.json").read_text())
    assert (tmp_path / "check_power_matrix.md").read_text().startswith("# Check power")
    rc, out = _quiet(ARENA.main, ["--problems", "burgers_1d", "--small"])
    assert rc == 0 and "arena burgers_1d" in out


def test_ipm_machine_is_built_once_per_gap_count():
    from pinneapple_presets.cases.ipm_traction import machine
    with pytest.raises(ValueError, match="multiple"):
        machine(7)                                                          # the gap mesh must fit poles and slots
    m = machine(480)                                                        # the preset's default
    assert machine(480) is m                                                # built once, then cached
