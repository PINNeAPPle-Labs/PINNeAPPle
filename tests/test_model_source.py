"""Use a trained model instead of training (#1190): zoo search, the applicability gate (range, units, reference
solves, the zoo's stored reference), forward and inverse use, and re-verification by the solver."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.surrogate.model_source import gate, open_model, use_model, zoo_candidates

pytest.importorskip("torch")
KNOWN = {"base_width_mm": 100.0, "base_depth_mm": 100.0, "base_thickness_mm": 6.0, "n_fins": 16.0,
         "fin_thickness_mm": 1.5, "fin_height_mm": 40.0, "k_w_mk": 200.0, "source_width_mm": 50.0,
         "source_depth_mm": 50.0, "t_ambient_c": 25.0, "power_w": 150.0}


@pytest.fixture(scope="module")
def heatsink():
    return open_model("zoo", name="heatsink_forced_mlp")


def _ref(**kw):
    from pinneapple_design.thermal.heatsink.surrogate import forced_reference
    return forced_reference(**kw)


def test_zoo_search_finds_the_models_with_the_output():
    names = [c["name"] for c in zoo_candidates(["r_base_k_w"])]
    assert names[:2] == ["heatsink_forced_mlp", "heatsink_natural_mlp"] or set(names[:2]) == {
        "heatsink_forced_mlp", "heatsink_natural_mlp"}
    assert zoo_candidates(["no_such_output"]) == []


def test_the_gate_checks_range_units_and_the_solver(heatsink):
    q = {k: [v] * 4 for k, v in {**KNOWN, "air_velocity_m_s": 2.5}.items()}
    g = gate(heatsink, q, reference=_ref, n_reference=4)
    st = {c["code"]: c["status"] for c in g["checks"]}
    assert g["ok"] and st["gate.reference"] == "pass"
    out = dict(q, power_w=[900.0] * 4)
    bad = gate(heatsink, out)
    assert not bad["ok"] and "power_w = 900" in next(c["detail"] for c in bad["checks"]
                                                    if c["code"] == "gate.validity_range")
    units = gate(heatsink, q, problem_units={"power_w": "m/s"})
    assert next(c for c in units["checks"] if c["code"] == "gate.units")["status"] == "fail"


def test_the_zoo_cards_metrics_are_recomputed_on_its_reference(heatsink):
    q = {k: [v] for k, v in {**KNOWN, "air_velocity_m_s": 2.5}.items()}
    c = next(c for c in gate(heatsink, q)["checks"] if c["code"] == "gate.reference")
    assert c["status"] == "pass" and "recomputed" in c["detail"]


def test_inverse_through_the_model_is_re_verified_by_the_solver(heatsink):
    obs = _ref(**KNOWN, air_velocity_m_s=2.5)["r_base_k_w"]
    r = use_model(heatsink, mode="inverse", observed={"r_base_k_w": obs}, unknown=["air_velocity_m_s"],
                  known=KNOWN, reference=_ref)
    assert r["used"] and r["estimate"]["air_velocity_m_s"] == pytest.approx(2.5, rel=0.02)
    assert r["reverified"]["r_base_k_w"]["rel_diff"] < 0.02


def test_an_unreachable_observation_is_refused_not_reported(heatsink):
    r = use_model(heatsink, mode="inverse", observed={"r_base_k_w": 0.05}, unknown=["air_velocity_m_s"],
                  known=KNOWN)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert not r["used"] and st["inverse.inside_bounds"] == "fail"


def test_a_torchscript_file_with_its_card(tmp_path):
    import torch
    lin = torch.nn.Linear(2, 1)
    with torch.no_grad():
        lin.weight[:] = torch.tensor([[2.0, -1.0]])
        lin.bias[:] = 0.5
    path = tmp_path / "m.pt"
    torch.jit.script(lin).save(str(path))
    p = open_model("torchscript", path=str(path), inputs=["a", "b"], outputs=["y"],
                   bounds={"a": (0, 1), "b": (0, 1)}, units={"a": "m", "b": "m", "y": "m"})
    r = use_model(p, mode="forward", query={"a": [0.5], "b": [0.25]},
                  reference=lambda a, b: {"y": 2 * a - b + 0.5})
    assert r["used"] and r["predictions"]["y"][0] == pytest.approx(1.25, abs=1e-6)
    with pytest.raises(ValueError, match="declared inputs"):
        open_model("torchscript", path=str(path))


def test_onnx_without_onnxruntime_says_what_to_install(tmp_path):
    try:
        import onnxruntime  # noqa: F401
        pytest.skip("onnxruntime is installed")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="onnxruntime"):
        open_model("onnx", path=str(tmp_path / "x.onnx"), inputs=["a"], outputs=["y"], bounds={"a": (0, 1)})


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "model.use" in REGISTRY.types()
    assert match("usar um modelo pré-treinado do zoo sem treinar")[0][1].id == "model.use"
    assert np.isfinite(KNOWN["power_w"])
