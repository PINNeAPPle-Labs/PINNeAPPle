"""2D/3D finite-volume conservation contracts (not executed)."""
import numpy as np
import pytest

from pinneapple_physics.validation import check_cartesian_fv_conservation


def test_2d_exact_balanced_flux():
    source = np.ones((2, 3))
    fx = np.broadcast_to(np.arange(3)[:, None], (3, 3))
    fy = np.zeros((2, 4))
    report = check_cartesian_fv_conservation(source, {"x": fx, "y": fy})
    assert report["passed"]
    assert report["dimensions"] == 2
    assert report["max_cell_balance_error"] == 0


def test_3d_exact_balanced_flux():
    source = np.ones((2, 2, 2))
    fx = np.broadcast_to(np.arange(3)[:, None, None], (3, 2, 2))
    fy = np.zeros((2, 3, 2))
    fz = np.zeros((2, 2, 3))
    report = check_cartesian_fv_conservation(
        source, {"x": fx, "y": fy, "z": fz})
    assert report["passed"]


def test_global_cancellation_does_not_hide_local_violation():
    source = np.zeros((2, 2))
    fx = np.broadcast_to(np.array([0., 1., 0.])[:, None], (3, 2))
    fy = np.zeros((2, 3))
    report = check_cartesian_fv_conservation(source, {"x": fx, "y": fy})
    assert report["global_passed"]
    assert not report["local_passed"]


def test_rejects_bad_face_shape():
    with pytest.raises(ValueError):
        check_cartesian_fv_conservation(np.ones((2, 2)),
            {"x": np.zeros((2, 2)), "y": np.zeros((2, 3))})
