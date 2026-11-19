"""Native safe export must preserve GP, MLP ensemble and quadratic predictions."""
import json
import numpy as np
import pytest

from pinneapple_autolab.surrogates import GPModel,MLPEnsemble,QuadraticRSM
from pinneapple_analysis.surrogate.safe_native_export import (
    export_native_scalar,predict_native_scalar)


@pytest.mark.parametrize("family,model",[
    ("quadratic",QuadraticRSM()),
    ("mlp_ensemble",MLPEnsemble(n=2,max_iter=60,seed=4)),
    ("gp",GPModel(restarts=0,seed=4)),
])
def test_native_roundtrip_without_pickle(tmp_path,family,model):
    x=np.linspace(0,1,30).reshape(-1,1)
    y=np.sin(3*x[:,0])
    model.fit(x,y)
    path=tmp_path/"model.safe.json"
    export_native_scalar({"y":model},{"y":family},{"y":1.5},
                         ["x"],["y"],{"x":(0.,1.)},str(path))
    payload=json.loads(path.read_text())
    points=np.array([[0.1],[0.4],[0.8]])
    actual=predict_native_scalar(payload,[{"x":float(v)} for v in points[:,0]])["y"]
    mean,std=model.predict(points)
    np.testing.assert_allclose(actual["mean"],mean,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(actual["half_width90"],1.5*std,rtol=1e-5,atol=1e-5)
    with pytest.raises(ValueError):
        predict_native_scalar(payload,[{"x":1.5}])
    assert b"pickle" not in path.read_bytes()

@pytest.mark.parametrize("family,model",[
    ("quadratic", QuadraticRSM()),
    ("mlp_ensemble", MLPEnsemble(n=2, max_iter=60, seed=9)),
    ("gp", GPModel(restarts=0, seed=9)),
])
def test_heldout_native_parity_and_uncertainty(family, model, tmp_path):
    train = np.linspace(0.0, 1.0, 42).reshape(-1, 1)
    model.fit(train, np.cos(2.0 * train[:, 0]))
    path = tmp_path / "heldout.safe.json"
    export_native_scalar({"y": model}, {"y": family}, {"y": 1.645},
                         ["x"], ["y"], {"x": (0.0, 1.0)}, str(path))
    artifact = json.loads(path.read_text())
    heldout = np.linspace(0.025, 0.975, 19).reshape(-1, 1)
    predictions = predict_native_scalar(
        artifact, [{"x": float(x)} for x in heldout[:, 0]])["y"]
    expected_mean, expected_std = model.predict(heldout)
    np.testing.assert_allclose(predictions["mean"], expected_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(predictions["half_width90"], 1.645 * expected_std,
                               rtol=1e-5, atol=1e-5)
    assert np.isfinite(predictions["mean"]).all()
    assert np.isfinite(predictions["half_width90"]).all()
