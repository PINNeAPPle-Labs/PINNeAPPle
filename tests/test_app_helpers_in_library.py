"""Helpers that used to live in the apps and are now public library API (docs/dev/LIBRARY_CONSOLIDATION.md):
JSON sanitiser, CAE previews and worst cells, tolerance verdicts, lineage report, pre-flight snippets, form helpers
and the unified before/after series."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pinneapple_core.jsonable import round_sig, to_jsonable

ROOT = Path(__file__).resolve().parents[1]


def test_to_jsonable_numpy_nan_paths_and_rounding():
    obj = {"a": np.float32(0.5), "b": np.arange(3), "c": [float("nan"), np.inf], 1: np.bool_(True),
           "_internal": np.zeros(2), "p": Path("x/y"), "n": None}
    out = to_jsonable(obj)
    assert out == {"a": 0.5, "b": [0, 1, 2], "c": [None, None], "1": True, "_internal": [0.0, 0.0], "p": "x/y",
                   "n": None}
    json.dumps(out)
    assert "_internal" not in to_jsonable(obj, drop_private=True)
    assert to_jsonable([float("nan")], nonfinite="str") == ["nan"]
    assert to_jsonable(object(), unknown="repr").startswith("<object")
    assert to_jsonable({"x": 1.23456789, "s": 1.23456789e-5}, digits=6) == {"x": 1.234568, "s": 1.23457e-05}
    assert round_sig(-0.000123456789) == -0.000123457


def test_field_scalar_von_mises_matches_principal_stresses():
    from pinneapple_data.cae.dataset import field_scalar

    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 6))                                   # xx yy zz xy yz zx
    sc, kind = field_scalar(v)
    assert kind == "von Mises"
    for row, s in zip(v, sc):
        xx, yy, zz, xy, yz, zx = row
        s1, s2, s3 = np.linalg.eigvalsh(np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]))
        assert s == pytest.approx(math.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)), rel=1e-10)
    m, kind = field_scalar(np.array([[3.0, 4.0, 0.0]]))
    assert kind == "magnitude" and m[0] == pytest.approx(5.0)
    assert field_scalar(np.zeros((2, 4)))[0] is None


def test_worst_cells_follow_the_metric_direction():
    from pinneapple_data.cae import mesh_report, read_any
    from pinneapple_data.cae.quality import METRICS
    from pinneapple_data.cae.report import worst_cells
    from pinneapple_data.cae.upload import expand

    name = "openfoam_pitzDaily.zip"
    with open(ROOT / "apps" / "mesh_quality" / "examples" / name, "rb") as f:
        mesh = read_any(expand([(name, f.read())]), fields=False)
    rep = mesh_report(mesh)
    worst = worst_cells(mesh, rep, n=5)
    assert worst
    for k, v in rep["_cell_values"].items():
        rows = [w for w in worst if w["metric"] == k]
        if not rows:
            continue
        finite = v[np.isfinite(v)]
        expect = finite.max() if METRICS[k]["better"] == "low" else finite.min()
        assert rows[0]["value"] == pytest.approx(expect)
        assert len(rows) == 5 and rows[0]["centre"] is not None


def test_tolerance_verdicts():
    from pinneapple_data.cae.compare import tolerance_verdicts

    res = {"fields": [{"summary": {"rel_l2": 0.01}}, {"summary": {"rel_l2": 0.2}}, {"summary": {"rel_l2": None}}]}
    assert tolerance_verdicts(res, 0.05) == "FAIL"
    assert [f["verdict"] for f in res["fields"]] == ["pass", "fail", None]
    assert tolerance_verdicts(res, 0.5) == "PASS"
    assert tolerance_verdicts(res, None) is None and res["fields"][0]["verdict"] is None


def test_lineage_report_flags_and_verified_hash():
    from pinneapple_data.lineage import Lineage

    good = hashlib.sha256(b"rev A").hexdigest()
    g = Lineage.from_json({"artifacts": [
        {"id": "CAD", "kind": "geometry", "hash": good[:16], "timestamp": "2026-01-01T00:00:00Z"},
        {"id": "MESH", "kind": "mesh", "inputs": ["CAD", "GHOST"], "timestamp": "2026-01-02T00:00:00Z"}]})
    g.nodes["CAD"].actual_hash = good
    rep = g.report()
    nodes = {n["id"]: n for n in rep["nodes"]}
    assert nodes["CAD"]["verified"] and not nodes["MESH"]["verified"]
    assert nodes["MESH"]["flag"] in ("fail", "warn") and rep["status"] in ("FAIL", "WARNING")
    assert rep["summary"]["artifacts"] == 2 and rep["summary"]["by_kind"] == {"geometry": 1, "mesh": 1}
    assert rep["summary"]["verified"] == 1
    assert g.flags() == {k: v for k, v in g.flags(g.checks()).items()}


def test_preflight_result_files_and_snippets():
    from pinneapple_data.preflight import attach_snippets, drop_results, is_result_file

    assert is_result_file("case/0.5/U") and is_result_file("case/1e-3/p") and is_result_file("run.frd")
    assert not is_result_file("case/0/U") and not is_result_file("case/system/controlDict")
    fs = {"case/system/controlDict": b"\n".join(b"line %d" % i for i in range(1, 21)), "case/10/U": b"x"}
    assert list(drop_results(fs)) == ["case/system/controlDict"]
    rep = {"info": {"root": "case"}, "findings": [{"file": "system/controlDict", "line": 10}, {"file": None}]}
    attach_snippets(rep, fs)
    snip = rep["findings"][0]["snippet"]
    assert snip["start"] == 7 and snip["at"] == 10 and snip["lines"][0] == "line 7" and len(snip["lines"]) == 8
    assert "snippet" not in rep["findings"][1]


def test_formfill_overrides_describe_and_filename():
    from pinneapple_data.formfill import check_overrides, get_spec, list_specs, output_filename

    spec = get_spec(list_specs()[0]["id"] if isinstance(list_specs()[0], dict) else list_specs()[0])
    d = spec.describe()
    assert d["id"] == spec.id and len(d["fields"]) == len(spec.fields)
    with pytest.raises(ValueError, match="unknown item"):
        check_overrides(spec, {"no_such_item": 1})
    assert check_overrides(spec, {spec.fields[0].key: ""}) == {}

    class _Comp:                                                   # output_filename only needs spec + decisions
        decisions = {}
    _Comp.spec = spec
    assert output_filename(_Comp, prefix="datasheet_", day=date(2026, 1, 2)) == \
        f"datasheet_{spec.id}_filled_20260102.pdf".replace(" ", "_")


def test_before_after_series_on_the_example_sources():
    import pandas as pd

    from pinneapple_data.unified import before_after_series, example_sources, make_recipe, propose_mapping, standardize
    from pinneapple_data.data_health import load_table

    srcs = []
    maps = []
    for name, raw in example_sources():
        df = load_table(raw, name)[0] if isinstance(load_table(raw, name), tuple) else load_table(raw, name)
        srcs.append((name, df, raw))
        maps.append(propose_mapping(df, name))
    res = standardize(srcs, make_recipe(maps))
    charts = before_after_series(res, k=50)
    assert charts
    for name, c in charts.items():
        assert len(c["x"]) <= 50
        for s in c["before"] + c["after"]:
            assert len(s["y"]) == len(c["x"])
            assert all(v is None or math.isfinite(v) for v in s["y"])
    json.dumps(charts)
    assert isinstance(pd.Timestamp(next(iter(charts.values()))["x"][0]), pd.Timestamp)
