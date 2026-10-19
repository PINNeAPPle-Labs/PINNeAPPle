"""pinneapple_data.contracts: validation, JSON / YAML round trips, JSON Schema, tables with sidecars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pinneapple_data.contracts import (
    CheckRecord,
    Column,
    DatasetSpec,
    ExpressionError,
    FileRef,
    ModelRef,
    Parameter,
    ParameterSpace,
    StudyResult,
    Table,
    Variable,
    convert,
    export_schemas,
    json_schemas,
    load,
    read_table,
    sidecar_path,
    validate,
    write_table,
)


def _space() -> ParameterSpace:
    return ParameterSpace(name="fin", budget=20, parameters=[
        Parameter(name="height", low=0.01, high=0.05, unit="m", group="geometry"),
        Parameter(name="k", low=1.0, high=1000.0, log=True, unit="W/(m K)", group="material"),
        Parameter(name="n_fins", kind="integer", low=2, high=6, group="geometry"),
        Parameter(name="material", kind="categorical", values=["Al6061", "Cu"], group="material"),
        Parameter(name="T_amb", kind="constant", value=300.0, unit="K", group="boundary"),
    ], constraints=["height <= 0.04 or material == 'Al6061'"])


def test_unit_mapping_round_trip_and_types():
    space = _space()
    assert space.dim == 4 and space.names[-1] == "T_amb"
    u = np.random.default_rng(0).random((50, space.dim))
    cases = space.from_unit(u)
    assert list(cases.columns) == space.names
    assert cases["n_fins"].dtype.kind == "i" and set(cases["n_fins"]) <= {2, 3, 4, 5, 6}
    assert set(cases["material"]) <= {"Al6061", "Cu"} and (cases["T_amb"] == 300.0).all()
    assert cases["k"].between(1, 1000).all()
    back = space.to_unit(cases)
    cont = [0, 1]                                     # continuous columns map back exactly
    np.testing.assert_allclose(back[:, cont], u[:, cont], atol=1e-12)
    pd.testing.assert_frame_equal(space.from_unit(back), cases)


def test_log_parameter_is_uniform_in_log_space():
    p = Parameter(name="k", low=1.0, high=100.0, log=True)
    assert p.from_unit(np.array([0.5]))[0] == pytest.approx(10.0)


def test_feasible_applies_bounds_and_constraints():
    space = _space()
    cases = pd.DataFrame({"height": [0.03, 0.045, 0.045, 0.2], "k": [10, 10, 10, 10], "n_fins": [3, 3, 3, 3],
                          "material": ["Cu", "Cu", "Al6061", "Cu"], "T_amb": [300.0] * 4})
    assert space.feasible(cases).tolist() == [True, False, True, False]


@pytest.mark.parametrize("bad", [
    dict(name="x", low=1.0, high=0.0),
    dict(name="x", low=0.0, high=1.0, log=True),
    dict(name="x", kind="integer", low=0.5, high=3),
    dict(name="x", kind="categorical", values=["a", "a"]),
    dict(name="x", kind="constant"),
    dict(name="1x", low=0.0, high=1.0),
])
def test_invalid_parameters_are_refused(bad):
    with pytest.raises(ValidationError):
        Parameter(**bad)


def test_constraints_are_checked_and_sandboxed():
    with pytest.raises(ValidationError, match="unknown names"):
        ParameterSpace(name="s", parameters=[Parameter(name="a", low=0, high=1)], constraints=["a + b < 1"])
    with pytest.raises((ValidationError, ExpressionError)):
        ParameterSpace(name="s", parameters=[Parameter(name="a", low=0, high=1)],
                       constraints=["__import__('os').system('true')"])
    with pytest.raises((ValidationError, ExpressionError)):
        ParameterSpace(name="s", parameters=[Parameter(name="a", low=0, high=1)], constraints=["a.real < 1"])


def test_unknown_fields_are_errors_and_validate_reports_them():
    data = _space().to_dict()
    data["paramters"] = []
    problems = validate(data)
    assert problems and any("paramters" in p for p in problems)
    assert validate(_space().to_dict()) == []
    assert validate({"kind": "Nope"})[0].startswith("not a PINNeAPPle contract")


def test_json_and_yaml_round_trip(tmp_path):
    space = _space()
    for name in ("s.json", "s.yaml"):
        p = space.save(tmp_path / name)
        assert load(p) == space
        assert ParameterSpace.load(p) == space
    with pytest.raises(TypeError):
        DatasetSpec.load(tmp_path / "s.json")
    assert space.content_hash() == load(tmp_path / "s.json").content_hash()


def test_dataset_spec_from_table_and_file_verification(tmp_path):
    df = pd.DataFrame({"case": ["c0", "c1"], "height": [0.01, 0.02], "Rth": [1.5, 1.1]})
    path = write_table(df, tmp_path / "cases.csv")
    spec = DatasetSpec.from_table(path, outputs=["Rth"], ids=["case"], units={"height": "m", "Rth": "K/W"},
                                  root=tmp_path, parameter_space=_space())
    assert spec.inputs == ["height"] and spec.outputs == ["Rth"] and spec.n_rows == 2
    assert spec.files[0].path == "cases.csv" and spec.files[0].format == "csv" and spec.files[0].rows == 2
    assert spec.verify_files(tmp_path) == []
    path.write_text(path.read_text() + "c2,0.03,0.9\n")
    assert "SHA-256 differs" in spec.verify_files(tmp_path)[0]
    with pytest.raises(ValidationError):
        DatasetSpec(name="d", columns=[Column(name="a"), Column(name="a")])


def test_model_ref_metrics_must_be_held_out_and_finite():
    ok = ModelRef(name="m", uri="zoo:airfoil_cst_rans_mlp", metrics={"test_rel_l2": 0.03},
                  inputs=[Variable(name="alpha_deg", low=-4, high=12)])
    assert ok.in_range(alpha_deg=[0, 10]) == {"alpha_deg": True}
    assert ok.in_range(alpha_deg=20) == {"alpha_deg": False}
    with pytest.raises(ValidationError, match="held-out"):
        ModelRef(name="m", uri="x", metrics={"rel_l2": 0.01}, metrics_split="train")
    with pytest.raises(ValidationError, match="finite"):
        ModelRef(name="m", uri="x", metrics={"rel_l2": float("nan")})


def test_study_result_checks_need_records():
    with pytest.raises(ValidationError, match="needs a value or evidence"):
        CheckRecord(name="coverage", status="PASS")
    not_run = StudyResult(id="s", study_type="doe", checks=[CheckRecord(name="c", status="NOT_RUN")])
    assert not not_run.passed                       # nothing ran: not "passed"
    res = StudyResult(id="s", study_type="doe", parameter_space=_space(),
                      table=Table.from_frame(pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})),
                      checks=[CheckRecord(name="min_distance", status="PASS", value=0.12, threshold=0.05)])
    assert res.passed and res.table.columns["a"] == [1.0, None]
    assert load_json_round_trip(res) == res
    with pytest.raises(ValidationError, match="equal length"):
        Table(columns={"a": [1], "b": [1, 2]})


def load_json_round_trip(obj):
    return type(obj).from_json(obj.to_json())


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".h5", ".json"])
def test_tables_carry_their_contract(tmp_path, suffix):
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    if suffix == ".h5":
        pytest.importorskip("h5py")
    space = _space()
    cases = space.from_unit(np.random.default_rng(1).random((5, space.dim)))
    path = write_table(cases, tmp_path / f"cases{suffix}", space)
    assert sidecar_path(path).is_file()
    df, contract = read_table(path)
    assert contract == space
    pd.testing.assert_frame_equal(df.reset_index(drop=True), cases, check_dtype=False)
    if suffix in (".parquet", ".h5"):                # embedded copy survives a lost sidecar
        sidecar_path(path).unlink()
        assert read_table(path)[1] == space


def test_convert_between_formats(tmp_path):
    pytest.importorskip("pyarrow")
    space = _space()
    src = write_table(space.from_unit(np.full((3, space.dim), 0.5)), tmp_path / "a.csv", space)
    dst = convert(src, tmp_path / "a.parquet")
    df, contract = read_table(dst)
    assert contract == space and len(df) == 3


def test_parameter_space_from_parameter_table(tmp_path):
    path = tmp_path / "params.csv"
    path.write_text("name,kind,low,high,values,unit,log,group\n"
                    "height,continuous,0.01,0.05,,m,false,geometry\n"
                    "k,continuous,1,1000,,W/(m K),true,material\n"
                    "material,categorical,,,Al6061|Cu,,,material\n")
    space = ParameterSpace.from_table(path)
    assert space.names == ["height", "k", "material"] and space["k"].log and space["material"].values == ["Al6061", "Cu"]
    again = tmp_path / "again.csv"
    space.to_frame().to_csv(again, index=False)
    assert ParameterSpace.from_table(again, name="params").parameters == space.parameters


def test_committed_json_schemas_are_current(tmp_path):
    committed = Path(__file__).resolve().parents[2] / "pinneapple_data" / "contracts" / "schemas"
    export_schemas(tmp_path)
    for kind in json_schemas():
        fresh = json.loads((tmp_path / f"{kind}.schema.json").read_text())
        assert json.loads((committed / f"{kind}.schema.json").read_text()) == fresh, (
            "regenerate with: python -c \"from pinneapple_data.contracts import export_schemas; "
            "export_schemas('pinneapple_data/contracts/schemas')\"")
    assert json_schemas()["DatasetSpec"]["properties"]["kind"]["const"] == "DatasetSpec"


def test_file_ref_of(tmp_path):
    p = tmp_path / "m.inp"
    p.write_text("*NODE\n")
    ref = FileRef.of(p, root=tmp_path)
    assert ref.format == "calculix" and ref.size == 6 and len(ref.sha256) == 64
