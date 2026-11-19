"""Contract tests for twin state and model inference observability."""
import pytest

from pinneapple_tools.visualization.inference_monitor import DigitalTwinMonitor, InferenceMonitor


def test_prediction_and_ground_truth_separation():
    events = []
    monitor = InferenceMonitor("model_1", events.append)
    monitor.prediction(4, {"pressure": 1.2}, observed={"pressure": 1.1}, latency_ms=7.5)
    assert events[0]["channel"] == "inference"
    assert events[0]["values"]["latency_ms"] == 7.5
    assert events[1]["channel"] == "state"
    assert events[1]["values"] == {"pred_pressure": 1.2, "obs_pressure": 1.1}


def test_twin_state_and_invalid_data():
    events = []
    twin = DigitalTwinMonitor("twin_1", events.append)
    twin.state(3, {"temperature": 340}, {"temperature": 337})
    assert events[0]["values"]["obs_temperature"] == 337
    with pytest.raises(ValueError):
        twin.state(4, {"temperature": float("nan")})
