"""Feature-contract and fail-closed verification tests."""
import numpy as np
import pytest
from features import branch_features, trunk_features, BRANCH_FEATURES, TRUNK_FEATURES
from verification import verify_fem


def test_branch_contract():
    f = branch_features("G", {"width": 50, "length": 75, "thickness": 6},
                        70e9, 0.33, [100,0,0], [0,0,5])
    assert f.shape == (31,)
    assert len(BRANCH_FEATURES) == 31
    assert f[8 + 1] == 1
    assert f[8:20].sum() == 1


def test_trunk_requires_concave_edge():
    points = np.array([[0.,0.,0.],[0.1,0.1,0.1]])
    fixed = np.array([[0.,0.,0.]])
    loaded = np.array([[0.1,0.1,0.1]])
    with pytest.raises(ValueError):
        trunk_features(points, fixed, loaded)
    x = trunk_features(points, fixed, loaded, np.array([[0.05,0.05,0.05]]))
    assert x.shape == (2,8)
    assert len(TRUNK_FEATURES) == 8
    assert np.isfinite(x).all()


def test_verification_never_claims_certification():
    nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    result = verify_fem(nodes, np.array([[0,1,2,3]]), np.zeros((4,3)),
                        np.zeros((1,6)), np.zeros(1), np.zeros((4,3)),
                        np.zeros((4,3)), np.array([0]))
    assert result["passed_basic_checks"]
    assert not result["certified"]
    assert not result["checks"]["mesh_convergence"]
