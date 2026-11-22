"""Conservation and boundary checks; not executed."""
import pytest

from pinneapple_physics.validation import check_fv_conservation_1d


def test_exact_conservation_with_sources_and_boundaries():
    result = check_fv_conservation_1d(
        [1.0, 2.0, 4.0], [1.0, 2.0], boundary_flux=(1.0, 4.0))
    assert result["passed"]
    assert result["max_cell_balance_error"] == 0
    assert result["global_balance_error"] == 0


def test_local_error_not_hidden_by_global_cancellation():
    result = check_fv_conservation_1d([0.0, 1.0, 0.0], [0.0, 0.0])
    assert result["global_passed"]
    assert not result["local_passed"]
    assert not result["passed"]


def test_boundary_failure_even_with_conservation():
    result = check_fv_conservation_1d(
        [1.0, 1.0], [0.0], boundary_flux=(2.0, 1.0))
    assert result["local_passed"]
    assert not result["boundary_passed"]
    assert not result["passed"]


@pytest.mark.parametrize("flux,source", [
    ([0.0], [0.0]),
    ([0.0, float("nan")], [0.0]),
    ([0.0, 1.0, 2.0], [1.0]),
])
def test_invalid_inputs_fail_closed(flux, source):
    with pytest.raises(ValueError):
        check_fv_conservation_1d(flux, source)
