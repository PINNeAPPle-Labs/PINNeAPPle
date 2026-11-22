"""Unstructured FV topology checks; tests are not executed."""
import pytest

from pinneapple_physics.validation import check_unstructured_fv_conservation


def test_exact_two_cell_balance():
    # One interior face, left source +2, right source -2.
    result = check_unstructured_fv_conservation([0], [1], [2.], [2., -2.])
    assert result["passed"]
    assert result["boundary_face_count"] == 0


def test_local_errors_can_cancel_globally():
    result = check_unstructured_fv_conservation([0], [1], [2.], [0., 0.])
    assert result["global_passed"]
    assert not result["local_passed"]


def test_boundary_target_mismatch():
    result = check_unstructured_fv_conservation(
        [0], [-1], [2.], [2.], boundary_flux_targets=[3.])
    assert result["local_passed"]
    assert not result["boundary_passed"]


@pytest.mark.parametrize("owner,neighbour", [([0], [0]), ([0], [2]), ([-1], [-1])])
def test_rejects_invalid_incidence(owner, neighbour):
    with pytest.raises(ValueError):
        check_unstructured_fv_conservation(owner, neighbour, [1.], [0., 0.])
