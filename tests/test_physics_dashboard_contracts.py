"""Contract checks for declarative Physics Dashboard definitions and telemetry.

Authored for later execution; no CI run is implied.
"""
import pytest

from pinneapple_tools.visualization.dashboard import (
    Dashboard, DataBinding, Widget, normalize_event, template, visualization_catalog,
)
from pinneapple_tools.visualization.telemetry import TelemetryEmitter, openfoam_residuals


def test_catalog_and_domain_presets():
    assert len(visualization_catalog()) >= 40
    for domain in ("rover", "training", "cfd", "drift", "pipeline", "blender"):
        board = template(domain, "case_1")
        assert len(board.widgets) == 4
        assert board.to_dict()["version"] == 1


def test_rejects_unsafe_layout_and_nonfinite_telemetry():
    binding = DataBinding("simulation", "case_1", "residuals")
    with pytest.raises(ValueError):
        Widget("w1", "mesh3d", "Mesh", binding, x=10, w=4)
    widget = Widget("w1", "solver_residuals", "Residuals", binding)
    with pytest.raises(ValueError):
        Dashboard("Invalid", [widget, widget])
    with pytest.raises(ValueError):
        normalize_event({"channel": "loss", "step": 1, "values": {"loss": float("inf")}})


def test_training_emitter_and_openfoam_parser():
    emitted = []
    producer = TelemetryEmitter("training", "train_1", emitted.append)
    producer.training_step(3, {"pde": 0.02, "boundary": 0.01}, lr=1e-3)
    assert [e["sequence"] for e in emitted] == [1, 2]
    assert emitted[0]["values"]["pde"] == 0.02
    text = "smoothSolver: Solving for Ux, Initial residual = 0.002, Final residual = 1e-5, No Iterations 2"
    assert openfoam_residuals(text) == [{"field": "Ux", "residual": 0.002}]
