"""Shared contracts for uncertainty and sensitivity studies: ParameterSpace <-> distributions, ModelRef <->
StudyModel, StudyResult from a study."""
import numpy as np
import pytest

from pinneapple_analysis.sensitivity import sensitivity_study
from pinneapple_analysis.study_contracts import (
    model_from_ref,
    model_to_ref,
    sensitivity_result,
    space_from_contract,
    space_to_contract,
    uncertainty_result,
)
from pinneapple_analysis.study_models import StudyModel, builtin
from pinneapple_analysis.uncertainty.study import uncertainty_study

contracts = pytest.importorskip("pinneapple_data.contracts")


def test_parameter_space_round_trip_keeps_the_distributions():
    ps = contracts.ParameterSpace(name="s", parameters=[
        contracts.Parameter(name="a", low=0, high=1, unit="m"),
        contracts.Parameter(name="b", low=-3, high=3),
        contracts.Parameter(name="c", kind="constant", value=2.5)],
        metadata={"distributions": {"b": {"dist": "normal", "mean": 0, "std": 1}}})
    params, fixed = space_from_contract(ps)
    assert params[0] == {"name": "a", "dist": "uniform", "low": 0.0, "high": 1.0, "unit": "m"}
    assert params[1]["dist"] == "normal" and fixed == {"c": 2.5}
    back = space_to_contract(params, fixed=fixed)
    assert space_from_contract(back.to_dict()) == (params, fixed)
    lo, hi = back.parameters[1].low, back.parameters[1].high
    assert lo == pytest.approx(-3.0902, abs=1e-3) and hi == pytest.approx(3.0902, abs=1e-3)   # 0.1-99.9 %


def test_parameter_space_refuses_what_it_cannot_sample():
    cat = contracts.ParameterSpace(name="s", parameters=[contracts.Parameter(name="m", kind="categorical",
                                                                             values=["Al", "Cu"])])
    with pytest.raises(ValueError, match="categorical"):
        space_from_contract(cat)
    log = contracts.ParameterSpace(name="s", parameters=[contracts.Parameter(name="k", low=1, high=100, log=True)])
    with pytest.raises(ValueError, match="log-scaled"):
        space_from_contract(log)


def test_model_ref_round_trip_for_a_zoo_model():
    m = model_from_ref({"kind": "ModelRef", "name": "hs", "uri": "zoo:heatsink_forced_mlp",
                        "outputs": [{"name": "r_base_k_w"}]})
    assert m.kind == "zoo" and m.output_names == ["r_base_k_w"]
    ref = model_to_ref(m)
    assert ref.uri == "zoo:heatsink_forced_mlp" and ref.metrics_split == "test"
    assert all(k.startswith("test/") for k in ref.metrics) and ref.dataset.name == "heatsink_forced_convection"
    assert model_from_ref(ref).input_names == m.input_names
    with pytest.raises(ValueError, match="ONNX"):
        model_from_ref({"kind": "ModelRef", "name": "g", "uri": "file:///tmp/g.onnx", "framework": "onnx"})


def test_study_results_carry_the_checks_that_ran():
    b = builtin("ishigami")
    out = sensitivity_study(b, b.metadata["space"], n_sobol=2048, exact=b.metadata["exact"], effects_on=False)
    sr = sensitivity_result(out, b, "s1")
    assert sr.study_type == "sensitivity" and sr.passed
    assert [c.status for c in sr.checks] == ["PASS"]
    assert sr.metrics["top_parameter[y]"] == "x1"
    contracts.StudyResult.model_validate_json(sr.to_json())

    def truth(X):
        return np.sin(3 * X[:, 0]) + X[:, 1] ** 2

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (800, 2))
    y = truth(X) + rng.normal(0, 0.05, 800)
    model = StudyModel.from_function(truth, ["a", "b"], ["y"], bounds={"a": (0, 1), "b": (0, 1)})
    u = uncertainty_study(model, "y", X[:400], y[:400], X[400:], y[400:], nominal={"b": 0.5},
                          uncertain=[{"name": "a", "dist": "uniform", "low": 0.2, "high": 0.8}], n_propagation=1024)
    ur = uncertainty_result(u, model, "u1")
    names = [c.name for c in ur.checks]
    assert names[0] == "held-out coverage of the prediction interval" and "polynomial chaos leave-one-out Q2" in names
    assert ur.metrics["n_test"] == 400 and ur.parameter_space.parameters[0].name == "a"
    assert ur.models[0].uri == "python:function"
