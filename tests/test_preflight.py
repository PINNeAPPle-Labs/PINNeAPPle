"""Simulation Preflight: each example is a real case (OpenFOAM v1912 tutorials, a CalculiX cantilever) or a copy broken
in one way; the solver's reaction to every broken copy is in apps/simulation_preflight/examples/solver_logs."""
from __future__ import annotations

import os
import sys

import pytest

from pinneapple_data.cae.upload import expand
from pinneapple_data.preflight import preflight

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "simulation_preflight"))
EX = os.path.join(APP, "examples")

CASES = {
    "openfoam_pitzDaily_ok.zip": ("PASS", set()),
    "openfoam_pitzDaily_missing_patch_bc.zip": ("FAIL", {"missing_patch"}),
    "openfoam_pitzDaily_no_pressure_reference.zip": ("FAIL", {"pref"}),
    "openfoam_pitzDaily_epsilon_zero.zip": ("FAIL", {"turb_init"}),
    "openfoam_pitzDaily_no_relaxation.zip": ("FAIL", {"no_relax"}),
    "openfoam_cavity_large_timestep.zip": ("WARNING", {"courant"}),
    "openfoam_cavity_missing_nu.zip": ("FAIL", {"no_nu"}),
    "calculix_cantilever_ok.inp": ("PASS", set()),
    "calculix_heat_missing_conductivity.inp": ("FAIL", {"missing_conductivity"}),
    "calculix_static_no_supports.inp": ("FAIL", {"no_support"}),
    "calculix_frequency_inconsistent_units.inp": ("FAIL", {"units"}),
    "calculix_undefined_node_set.inp": ("FAIL", {"undefined_set", "no_support"}),
    "calculix_inverted_element.inp": ("FAIL", {"inverted"}),
    "calculix_elements_without_section.inp": ("FAIL", {"no_section"}),
}


def _run(name):
    with open(os.path.join(EX, name), "rb") as f:
        return preflight(expand([(name, f.read())]))


@pytest.mark.parametrize("name", sorted(CASES))
def test_example_verdict_and_rules(name):
    status, rules = CASES[name]
    r = _run(name)
    problems = {f["rule"] for f in r["findings"] if f["status"] in ("warn", "fail")}
    assert r["status"] == status, (name, problems)
    assert problems == rules, name
    assert os.path.exists(os.path.join(EX, "solver_logs", name.rsplit(".", 1)[0] + ".log"))


def test_locations_point_at_the_problem():
    r = _run("openfoam_pitzDaily_missing_patch_bc.zip")
    f = next(x for x in r["findings"] if x["rule"] == "missing_patch")
    assert (f["file"], f["entity"]) == ("0/k", "outlet") and f["evidence"].startswith("simpleFoam stops")
    r = _run("calculix_heat_missing_conductivity.inp")
    f = next(x for x in r["findings"] if x["rule"] == "missing_conductivity")
    assert f["entity"] == "STEEL" and f["line"] == 1681            # the *MATERIAL line


def test_courant_estimate_is_an_upper_bound():
    """OpenFOAM measured Co max 3.41 on this case; the pre-run estimate must not be lower."""
    r = _run("openfoam_cavity_large_timestep.zip")
    co = r["info"]["courant_estimate"]["max"]
    assert 3.41 <= co <= 4.5


def test_api_snippets_and_logs():
    pytest.importorskip("fastapi")
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from simulation_preflight.api import app
    with TestClient(app) as c:
        j = c.get("/api/example/openfoam_pitzDaily_missing_patch_bc.zip").json()
        f = next(x for x in j["findings"] if x["rule"] == "missing_patch")
        assert "boundaryField" in f["snippet"]["lines"][f["snippet"]["at"] - f["snippet"]["start"]]
        assert "Cannot find patchField entry for outlet" in c.get("/api/example/openfoam_pitzDaily_missing_patch_bc.zip/log").text
        with open(os.path.join(EX, "calculix_static_no_supports.inp"), "rb") as fh:
            j = c.post("/api/preflight", files={"files": ("beam.inp", fh.read())}).json()
        assert j["status"] == "FAIL"
        assert c.post("/api/preflight", files={"files": ("readme.txt", b"hi")}).status_code == 422
