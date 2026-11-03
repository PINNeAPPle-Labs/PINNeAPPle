"""The Ahmed-body design step (#1182): least robust drag inside the sampled box, the drag crisis flagged, verified by
a recorded CFD run only when one is close, otherwise stated unverified."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
STUDY = Path(__file__).resolve().parents[1] / "examples" / "end_to_end" / "ahmed_body" / "results"


@pytest.fixture(scope="module")
def recs():
    from pinneapple_autolab.domains import ahmed_study as S
    return S.load_records(str(STUDY / "cfd_runs"))


def _res(tmp_path):
    shutil.copy(STUDY / "benchmark.json", tmp_path / "benchmark.json")
    return str(tmp_path)


def test_design_beats_the_25_degree_baseline_and_says_it_is_unverified(recs, tmp_path):
    from pinneapple_autolab.domains import ahmed_study as S
    r = S.design(recs, _res(tmp_path))
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["design.outside_drag_crisis"] == st["design.no_extrapolation"] == "pass"
    assert r["design"]["slant_deg"] <= S.HOLDOUT_DEG
    assert r["predicted"]["CD"] + r["predicted"]["CD_half_width_90"] < r["baseline"]["CD"]
    assert r["families"] == {"CD": "mlp_ensemble", "CL": "gp"}           # the study's own selection
    v = r["verification"]
    assert v["method"] == "nearest recorded run" and v["distance_unit"] > 0.05
    assert st["verify.cfd"] == "inconclusive"                            # never claimed without a close CFD run
    assert (tmp_path / "design.json").exists()


def test_a_loose_tolerance_lets_the_nearest_run_verify_or_refute(recs, tmp_path):
    from pinneapple_autolab.domains import ahmed_study as S
    r = S.design(recs, _res(tmp_path), verify_tol=0.2)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["verify.cfd"] in ("pass", "fail")
    near = next(c for c in r["checks"] if c["code"] == "verify.cfd")
    assert r["verification"]["run"] in near["detail"]


def test_the_node_is_in_the_ahmed_flow():
    import json

    from pinneapple_flow.registry import REGISTRY
    assert "study.design" in REGISTRY.types()
    d = json.loads((Path(__file__).resolve().parents[1] / "pinneapple_flow" / "flows" /
                    "ahmed_body_end_to_end.json").read_text())
    assert any(n["type"] == "study.design" for n in d["nodes"])
    assert any(e["source"] == "cfd" and e["target"] == "design" for e in d["edges"])
