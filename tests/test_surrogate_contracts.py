"""pinneapple_analysis.surrogate.contracts: studies, models and arena results in the shared data contracts."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
C = pytest.importorskip("pinneapple_data.contracts")
pytest.importorskip("pinneapple_data.contracts.models")       # the package, not a leftover directory

from pinneapple_analysis import surrogate as S  # noqa: E402
from pinneapple_analysis.surrogate import contracts as SC  # noqa: E402
from pinneapple_arena.regimes import run_regime_arena  # noqa: E402


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (80, 2))
    return S.SurrogateData(X, np.stack([X[:, 0] + X[:, 1] ** 2, np.exp(X[:, 0])], 1), ["a", "b"], ["f", "g"],
                           groups=np.repeat(np.arange(20), 4), meta={"group": "case"})


def test_model_ref_and_study_result_validate():
    st = S.run_study(_data(), methods=["ridge", "gp"], budget={"steps": 5}, seeds=1)
    ref = SC.model_ref(st.model, name="demo", version="v1", uri="models/demo/v1", app="surrogate")
    assert C.validate(ref.to_dict()) == []
    assert {v.name for v in ref.inputs} == {"a", "b"} and ref.inputs[0].low is not None
    assert set(ref.metrics) >= {"test_nrmse", "shift_nrmse"} and ref.metrics_split == "test"
    assert ref.trust_decision == st.trust["decision"] and (ref.sha256 is None or len(ref.sha256) == 64)
    assert ref.in_range(a=0.5, b=0.5) == {"a": True, "b": True}
    sr = SC.study_result(st, id="s1", ref=ref)
    assert C.validate(sr.to_dict()) == [] and sr.study_type == "surrogate" and sr.models[0].name == "demo"
    ran = [c for c in sr.checks if c.status in ("PASS", "FAIL")]
    assert ran and all(c.value is not None or c.evidence for c in ran)
    assert sr.limits and len(sr.table.columns["method"]) == len(S.CANDIDATES)


def test_dataset_spec_round_trip():
    d = _data()
    spec = SC.dataset_spec(d, "demo")
    assert spec.inputs == ["a", "b"] and spec.outputs == ["f", "g"] and spec.metadata["group"] == "case"
    df = pd.DataFrame({"a": d.X[:, 0], "b": d.X[:, 1], "f": d.Y[:, 0], "g": d.Y[:, 1], "case": d.groups})
    back = SC.data_from_spec(df, spec)
    assert back.sha256() == d.sha256() and back.groups is not None
    b = S.examples.burgers_fields(n=12, nx=8, nt=4)
    fs = SC.dataset_spec(b, "burgers")
    cols = {**{"nu": b.X[:, 0]}, **{f"u_{i}": b.Y.reshape(12, -1)[:, i] for i in range(32)}}
    fb = SC.data_from_spec(pd.DataFrame(cols), fs)
    assert fb.kind == "field" and fb.grid_shape == (8, 4) and np.allclose(fb.Y, b.Y)


def test_arena_study_result_validates():
    r = run_regime_arena(_data(), methods=["ridge"], budget={"steps": 5}, seeds=1).to_dict()
    sr = SC.arena_study_result(r, id="a1")
    assert C.validate(sr.to_dict()) == [] and sr.study_type == "benchmark"
    assert sr.metrics["overall_winner"] == "ridge" and any("not run" in x for x in sr.limits)


def test_read_table_with_roles(tmp_path):
    d = _data()
    df = pd.DataFrame({"a": d.X[:, 0], "b": d.X[:, 1], "f": d.Y[:, 0], "g": d.Y[:, 1], "case": d.groups})
    C.write_table(df, tmp_path / "t.parquet", SC.dataset_spec(d, "demo"))
    back, roles = SC.read_table_with_roles(tmp_path / "t.parquet")
    assert len(back) == 80 and roles["f"] == "output" and roles["case"] == "id"
    df.to_csv(tmp_path / "plain.csv", index=False)
    assert SC.read_table_with_roles(tmp_path / "plain.csv")[1] is None
    with pytest.raises(ValueError, match="limit"):
        SC.read_table_with_roles(tmp_path / "plain.csv", max_rows=10)
