import numpy as np
import pytest
from pinneapple_veriphysics.design_adapter import (
    SurrogateAdapter, freeze_design_selection,
)


def test_freezes_without_reference():
    adapter = SurrogateAdapter(
        predict=lambda x: 1 + x[:, 0],
        error_scale=lambda x: np.zeros(len(x)),
        name="mock-pinn",
    )
    x = np.array([[.2], [.8], [.5]])
    frozen = freeze_design_selection(
        adapter, x, [[.2], [.5], [.8]], [[0, 1]],
        baseline_prediction=1., threshold=.1, top_k=2,
    )
    assert frozen["selected_indices"] == [1, 2]
    assert frozen["reference_calls_selection"] == 0
    assert frozen["selection_frozen"] is True


def test_requires_explicit_uncertainty_input():
    adapter = SurrogateAdapter(predict=lambda x: x[:, 0])
    with pytest.raises(ValueError, match="error_scale"):
        adapter.evaluate(np.array([[.2]]))


def test_invalid_prediction_rejected():
    adapter = SurrogateAdapter(predict=lambda x: np.array([np.nan]),
                               error_scale=lambda x: 0.1)
    with pytest.raises(ValueError, match="surrogate"):
        adapter.evaluate(np.array([[.2]]))
