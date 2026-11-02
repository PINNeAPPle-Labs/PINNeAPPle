"""Core coverage for the front door's preset catalog, the data contracts and the industrial presets' building blocks
(#1129, toward 1.0). Each test checks what the function promises, not just that it runs."""
import hashlib
import json

import numpy as np
import pytest

import pinneapple_presets.core as PC
import pinneapple_presets.registry as PR
from pinneapple_core.onboarding import catalog as OC
from pinneapple_data.contracts import models as M
from pinneapple_data.contracts import tables as T


# ------------------------------------------------------------------------------------------ the preset catalog
def test_search_ranks_the_matching_preset_first():
    hits = OC.search("burgers equation viscous shock", top=3)
    assert hits[0]["name"] == "burgers_1d" and hits[0]["score"] == 1.0
    assert all(h["score"] <= hits[0]["score"] for h in hits)
    assert OC.search("zzzz nothing matches", top=3, min_score=0.5) == []


def test_methods_runnability_and_cards():
    assert OC.best_method("burgers_1d") == OC.working_method("burgers_1d") == "analytic"
    assert OC.runnable("burgers_1d") and OC.probe("burgers_1d") == (True, "")
    d = OC.describe("burgers_1d")
    assert d["name"] == "burgers_1d" and d["spatial_dim"] == 1 and d["transient"]
    lines = OC.format_card(OC.card("burgers_1d"), width=60).splitlines()
    assert lines[0] == "-" * 60 and "burgers_1d" in lines[1] and lines[2] == "-" * 60          # the bars set the width


def test_geometry_tags_and_the_standalone_program():
    assert OC.needs_geometry_tags("plane_stress_2d") and set(OC.tag_names("plane_stress_2d")) == {"fixed", "load"}
    assert not OC.needs_geometry_tags("burgers_1d") and OC.tag_names("burgers_1d") == ()
    assert "fixed" in OC.default_geometry_note("plane_stress_2d")
    compile(OC.standalone_snippet("burgers_1d"), "snippet", "exec")              # a valid Python program
    assert "OpenFOAM" in OC.solver_requirement("openfoam")


# ------------------------------------------------------------------------------------------ data contracts
def test_contract_models_round_trip_through_json():
    res = M.StudyResult(
        id="s1", study_type="doe", title="t",
        parameter_space=None, checks=[M.CheckRecord(name="mass", status="PASS", value=1e-9, threshold=1e-6)],
        provenance=M.Provenance(app="test", run_ids=["r1"]))
    back = M.StudyResult.model_validate_json(res.model_dump_json())
    assert back == res and back.checks[0].status == "PASS"
    with pytest.raises(ValueError):                                             # pydantic's ValidationError
        M.CheckRecord(name="mass", status="pass")                              # statuses are a closed set
    for obj in (M.Column(name="T", role="output", unit="K"), M.DatasetRef(name="d", version="1"),
                M.FileRef(path="a.csv", rows=3), M.Parameter(name="L", low=0.1, high=1.0, unit="m"),
                M.Splits(train=0.7, val=0.15, test=0.15, seed=0), M.Table(columns={"x": [1, 2]})):
        assert type(obj).model_validate(json.loads(obj.model_dump_json())) == obj


def test_json_schemas_and_their_export(tmp_path):
    schemas = M.json_schemas()
    assert {"StudyResult", "DatasetSpec", "ParameterSpace", "ModelRef"} <= set(schemas)
    paths = M.export_schemas(tmp_path)
    assert {p.name for p in paths} == {f"{k}.schema.json" for k in schemas}
    assert json.loads((tmp_path / "StudyResult.schema.json").read_text())["title"] == "StudyResult"


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(bytes(range(256)) * 5000)
    assert M.sha256_file(f, chunk=4096) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_tables_with_their_sidecar_contract(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [0.0, 0.5, 1.0], "T": [300.0, 310.0, 320.0]})
    contract = M.StudyResult(id="s1", study_type="doe")
    path = T.write_table(df, tmp_path / "cases.csv", contract)
    assert T.table_format(path) == "csv" and T.sidecar_path(path).name == "cases.csv.contract.json"
    back, c = T.read_table(path)
    assert back.equals(df) and c == contract and T.read_sidecar(path) == contract
    out = T.convert(path, tmp_path / "cases.json")
    again, c2 = T.read_table(out)
    assert np.allclose(again["T"], df["T"]) and c2 == contract
    T.write_sidecar(tmp_path / "other.csv", contract)
    assert T.read_sidecar(tmp_path / "other.csv") == contract


# ------------------------------------------------------------------------------------------ industrial presets
def test_lhs_puts_one_point_in_every_stratum():
    n, bounds = 8, [(0.0, 1.0), (10.0, 20.0)]
    X = PC.lhs(n, bounds, seed=3)
    assert X.shape == (n, 2)
    for j, (lo, hi) in enumerate(bounds):
        bins = np.floor((X[:, j] - lo) / (hi - lo) * n).astype(int)
        assert sorted(bins) == list(range(n))                                   # Latin hypercube property
    L = PC.lhs(6, [(1.0, 1000.0)], seed=0, log=(True,))
    assert sorted(np.floor(np.log10(L[:, 0]) / 3 * 6).astype(int)) == list(range(6))   # strata in log space


def test_checks_environment_and_quiet_calls():
    assert PC.check("x", True, 1.0, 2.0).status == "supports"
    assert PC.check("y", False).status != "supports"
    assert PC.check("z", None).status not in ("supports",)                       # not run is never a pass
    env = PC.environment()
    assert "python" in json.dumps(env).lower()
    assert PC.call_quiet(lambda a, b=0: a + b, 2, b=3) == 5


def test_registry_lists_loads_registers_and_indexes(tmp_path):
    names = PR.names()
    assert len(names) >= 13 and names == sorted(names)
    rows = PR.list_presets()
    assert {r["name"] for r in rows} == set(names)
    p = PR.load(names[0])
    assert isinstance(p, PC.Preset) and p.params
    idx = PR.index()
    assert {e["name"] for e in idx["presets"]} == set(names) and all(e["outputs"] for e in idx["presets"])
    out = PR.write_index(str(tmp_path / "index.json"))
    assert json.loads(open(out).read())
