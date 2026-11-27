import pytest

from pinneapple_analysis.verification.asme_vv20 import validation
from pinneapple_analysis.verification.solver_admission import assess_solver_result


@pytest.mark.parametrize("simulation,uncertainty,converged,expected", [
    (1.01, 0.02, True, True),
    (1.5, 1.0, True, False),
    (1.01, 0.02, False, False),
    (float("nan"), 0.02, True, False),
    (1.04, 0.01, True, False),
])
def test_admission(simulation, uncertainty, converged, expected):
    comparison = validation(simulation, 1.0, u_numerical=uncertainty)
    result = assess_solver_result(
        comparison, max_absolute_error=0.05,
        max_numerical_uncertainty=0.05, max_validation_uncertainty=0.1,
        convergence_confirmed=converged,
    )
    assert result.approved == expected


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        assess_solver_result(
            validation(1., 1., u_numerical=0.01),
            max_absolute_error=float("nan"),
            max_numerical_uncertainty=0.05, max_validation_uncertainty=0.1,
            convergence_confirmed=True,
        )
