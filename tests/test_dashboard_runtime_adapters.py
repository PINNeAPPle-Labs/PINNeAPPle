"""Integration contracts for opt-in scientific runtime adapters (not executed)."""
import pytest

from pinneapple_tools.visualization.runtime_adapters import (
    PINNTrainingCallback, ModelServingObserver,
)
from pinneapple_tools.visualization.twin_sync import TwinSynchronizer
from pinneapple_tools.visualization.geometry_variants import affine_variant


def test_training_real_values():
    events = []
    PINNTrainingCallback("run_1", events.append).on_epoch_end(
        10, losses={"pde": 0.02, "bc": 0.01},
        gradients={"total": 0.4})
    assert [e["channel"] for e in events] == ["loss", "gradients"]
    assert events[0]["values"]["pde"] == pytest.approx(.02)


def test_model_serving_wrapper_preserves_output():
    events = []
    observer = ModelServingObserver("model_1", lambda x: {"pressure": x * 2}, events.append)
    assert observer.infer(3, observed={"pressure": 5.5}) == {"pressure": 6}
    assert [e["channel"] for e in events] == ["inference", "state"]
    assert events[1]["values"]["obs_pressure"] == 5.5


def test_twin_sync_refuses_stale_measurement():
    events = []
    sync = TwinSynchronizer("twin_1", events.append, tolerance_seconds=.1)
    sync.observe(100.0, {"temperature": 300})
    assert sync.predict(101.0, {"temperature": 302}) is False
    assert events[-1]["values"]["synchronized"] == 0
    assert sync.predict(100.05, {"temperature": 301}) is True
    assert events[-1]["values"]["synchronized"] == 1


def test_geometry_variant_does_not_reuse_physics():
    mesh = {"vertices": [[0, 0, 0]], "triangles": [[0, 0, 0]],
            "kind": "mesh", "scalars": []}
    result = affine_variant(mesh, scale=(2, 1, 1), translation=(1, 0, 0))
    assert result["vertices"] == [[1, 0, 0]]
    with pytest.raises(ValueError):
        affine_variant({**mesh, "scalars": [3]}, scale=(2, 1, 1))
