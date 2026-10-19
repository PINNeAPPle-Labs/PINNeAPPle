"""pinneapple_simulation.external_solvers.campaign_cases: case files of a planned campaign."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinneapple_data.contracts import DatasetSpec, ParameterSpace, StudyResult, load, read_table
from pinneapple_data.doe import plan_campaign
from pinneapple_simulation.external_solvers import campaign_cases as cc

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "examples" / "physics_check" / "pass" / "cases" / "cavity"


def test_airfoil_preset_against_the_existing_runs(tmp_path):
    space = cc.airfoil_space(8)
    existing = cc.airfoil_existing()
    assert len(existing) == 436 and set(existing["split"]) == {"train", "test", "naca_reference"}
    assert cc.airfoil_feasible(existing).all()              # every run that exists has a valid shape
    plan = plan_campaign(space, method="maximin", existing=existing, existing_id_column="run_id",
                         feasible=cc.airfoil_feasible, seed=0)
    assert len(plan.cases) == 8 and cc.airfoil_feasible(plan.cases).all()
    assert plan.redundancy["n_existing"] == 418                      # NACA 4412 / 4415: camber M1 = 0.142 > 0.14
    assert plan.redundancy["n_existing_outside_space"] == 18
    files = cc.write_campaign(plan, tmp_path, outputs=["Cl", "Cd"], openfoam="airfoil", case_limit=1)
    case = Path(files["openfoam"][0])
    for f in ("system/controlDict", "constant/polyMesh/points", "0/U", "Allrun", "coefficients.py", "params.json"):
        assert (case / f).is_file(), f
    params = json.loads((case / "params.json").read_text())
    assert params["case_id"] == "case_001" and "alpha_deg" in params
    assert f"{np.cos(np.radians(params['alpha_deg'])):.10f}" in (case / "0/U").read_text()


def test_contract_files(tmp_path):
    plan = plan_campaign(cc.cantilever_space(6), method="lhs", seed=2)
    cc.write_campaign(plan, tmp_path, outputs=["tip_deflection"], units={"tip_deflection": "m"})
    space = load(tmp_path / "space.json")
    assert isinstance(space, ParameterSpace) and space.name == "cantilever_static"
    assert isinstance(load(tmp_path / "study.json"), StudyResult)
    skel = load(tmp_path / "dataset.skeleton.json")
    assert isinstance(skel, DatasetSpec) and skel.status == "skeleton" and skel.outputs == ["tip_deflection"]
    df, contract = read_table(tmp_path / "cases.csv")
    assert len(df) == 6 and contract == space and (df["length"] >= 8 * df["height"]).all()
    assert "cases.csv" in (tmp_path / "README.txt").read_text()


def test_calculix_cantilever_decks(tmp_path):
    plan = plan_campaign(cc.cantilever_space(3), method="lhs", seed=4)
    paths = cc.write_calculix_cantilever(plan.cases, tmp_path)
    assert len(paths) == 3
    deck = (paths[0] / "job.inp").read_text()
    for kw in ("*NODE", "*ELEMENT, TYPE=C3D8I", "*MATERIAL", "*ELASTIC", "*BOUNDARY", "*STATIC", "*CLOAD"):
        assert kw in deck, kw
    info = json.loads((paths[0] / "params.json").read_text())
    row = plan.cases.iloc[0]
    E = cc.MATERIALS[row["material"]]["E"]
    expect = row["tip_load"] * row["length"] ** 3 / (3 * E * row["width"] * row["height"] ** 3 / 12)
    assert info["beam_theory_tip_deflection_m"] == pytest.approx(expect)
    cload = [ln for ln in deck.split("*CLOAD")[1].split("*NODE FILE")[0].strip().splitlines()]
    assert sum(float(ln.split(",")[2]) for ln in cload) == pytest.approx(-row["tip_load"])  # the whole tip load


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx) not installed")
def test_calculix_deck_runs_and_matches_beam_theory(tmp_path):
    from pinneapple_simulation.external_solvers.calculix.frd import read_frd
    from pinneapple_simulation.external_solvers.calculix.runner import run_ccx
    plan = plan_campaign(cc.cantilever_space(2), method="lhs", seed=0)
    for case in cc.write_calculix_cantilever(plan.cases, tmp_path):
        run = run_ccx(str(case), "job", threads=1)
        assert run.returncode == 0
        disp = [s for s in read_frd(run.frd).steps if "DISP" in s][-1]["DISP"]
        beam = json.loads((case / "params.json").read_text())["beam_theory_tip_deflection_m"]
        assert -disp[:, 2].min() == pytest.approx(beam, rel=0.03)   # slender beam: shear adds < 3 %


def test_calculix_template(tmp_path):
    cases = pd.DataFrame({"case_id": ["c1", "c2"], "E": [210e9, 70e9]})
    paths = cc.write_calculix_from_template("*ELASTIC\n{E:.6g}, 0.3\n", cases, tmp_path)
    assert (paths[1] / "job.inp").read_text() == "*ELASTIC\n7e+10, 0.3\n"
    with pytest.raises(KeyError, match="nu"):
        cc.write_calculix_from_template("{nu}", cases, tmp_path)


@pytest.mark.skipif(not CAVITY.is_dir(), reason="example cavity case not in this checkout")
def test_openfoam_from_base_case_and_campaign_plan(tmp_path):
    from pinneapple_simulation.external_solvers.openfoam.campaign import Campaign, get_entry
    cases = pd.DataFrame({"case_id": ["case_001", "case_002"], "priority": [1, 2], "nu": [0.01, 0.002],
                          "U": [1.0, 2.0]})
    edits = [{"file": "constant/transportProperties", "key": "nu", "value": "[0 2 -1 0 0 0 0] {nu}"},
             {"file": "0/U", "key": "boundaryField/movingWall/value", "value": "uniform ({U} 0 0)"}]
    r = cc.write_openfoam_from_base(CAVITY, cases, tmp_path, edits, derived={"Re": "U * 0.1 / nu"}, solver="icoFoam")
    c2 = r["cases"][1]
    assert get_entry((c2 / "constant/transportProperties").read_text(), "nu").strip() == "[0 2 -1 0 0 0 0] 0.002"
    assert "uniform (2.0 0 0)" in (c2 / "0/U").read_text()
    assert json.loads((c2 / "params.json").read_text())["Re"] == pytest.approx(100.0)
    plan = Campaign.from_json(r["plan"])
    assert [run.id for run in plan.runs] == ["case_001", "case_002"] and plan.solver == "icoFoam"
    assert (tmp_path / "base_case" / "system" / "controlDict").is_file()
    with pytest.raises(ValueError, match="not an OpenFOAM case"):
        cc.write_openfoam_from_base(tmp_path / "cases", cases, tmp_path / "x", edits)


def test_python_runner_runs_and_resumes(tmp_path):
    plan = plan_campaign(cc.cantilever_space(4), method="lhs", seed=1)
    cc.write_case_tables(plan.cases, tmp_path, plan.space, formats=("csv",))
    script = cc.write_python_runner(tmp_path, plan.space.names, outputs=["volume"])
    text = script.read_text().replace(
        '    raise NotImplementedError("call your solver here and return {output name: value}")',
        '    if case["case_id"] == "case_002" and not os.path.exists("again"):\n'
        '        open("again", "w").close()\n        raise RuntimeError("diverged")\n'
        '    return {"volume": float(case["length"]) * float(case["width"]) * float(case["height"])}')
    script.write_text(text)
    for _ in range(2):
        subprocess.run([sys.executable, "run_cases.py"], cwd=tmp_path, check=True, capture_output=True)
    res = pd.read_csv(tmp_path / "results.csv")
    done = res[res["status"] == "done"]
    assert sorted(done["case_id"]) == ["case_001", "case_002", "case_003", "case_004"]
    assert (res["status"] == "failed").sum() == 1                    # the first attempt of case_002 is kept
