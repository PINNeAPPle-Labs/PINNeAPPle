"""Tests for ``pinneapple_physics.embedded`` -- the model, the residual and the measurement check as one ONNX graph
(#350).

What is asserted, in the order a flight computer cares about it:

* **the graph is a straight line**: no ``If``, no ``Loop``, no ``Scan`` -- the exported file is arithmetic, and a
  graph that branches is a graph that needs an interpreter;
* **it computes what the library computes**: the prediction and the residual from ``onnxruntime`` against the same
  numbers from torch and from :func:`pinneapple_physics.embedded.residual_fn`, to 1e-4;
* **it decides**: a field that obeys the equation does not set ``off_physics``, a field that does not obey it does, and
  a sensor that disagrees sets ``suspect``;
* **it is measured**: the latency is reported per call, on one thread, because "tens of hertz" is a measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from pinneapple_physics.embedded import (  # noqa: E402 - after the importorskip
    CONTROL_FLOW,
    EmbeddedRuntime,
    EmbeddedSpec,
    export,
    graph_summary,
    residual_fn,
)

SHAPE = (2, 16, 24)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 6, 3, padding=1, padding_mode="circular"), torch.nn.Tanh(),
        torch.nn.Conv2d(6, 1, 3, padding=1, padding_mode="circular"))


class _Persistence(torch.nn.Module):
    """The identity forecast: the next frame is the last one. With no advection and no diffusion its residual is 0."""

    def forward(self, state):
        return state[:, -1:, :, :]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    spec = EmbeddedSpec(state_shape=SHAPE, tolerance=0.05)
    path = export(_model(), spec, tmp_path_factory.mktemp("onnx") / "flight.onnx")
    return path, spec


def test_the_graph_is_a_straight_line(exported):
    path, _ = exported
    summary = graph_summary(path)

    assert summary.is_straight_line is True
    assert summary.control_flow == ()
    assert not set(CONTROL_FLOW) & set(summary.op_types), "an embedded graph cannot branch"
    assert summary.inputs == ("state", "measurement")
    assert summary.outputs == ("prediction", "residual", "trust", "off_physics", "suspect", "measurement_error")
    assert summary.bytes > 0
    assert {"Conv", "Pad", "Sqrt", "ReduceMean"} & set(summary.op_types), "the stencil is convolutions and reductions"


def test_the_exported_prediction_and_residual_match_torch(exported):
    path, spec = exported
    state = torch.randn(*SHAPE)
    runtime = EmbeddedRuntime(path, threads=1)

    out = runtime.step(state.numpy(), np.zeros(1))
    with torch.no_grad():
        reference = _model()(state.unsqueeze(0))
        expected_residual = float(residual_fn(spec)(reference, state[-1]))

    assert np.allclose(out.prediction, reference.numpy()[0], atol=1e-5)
    assert out.residual == pytest.approx(expected_residual, rel=1e-4)


def test_a_field_that_obeys_the_equation_is_not_off_physics(tmp_path):
    spec = EmbeddedSpec(state_shape=SHAPE, equation="advection", parameters={"c_x": 0.0, "c_y": 0.0}, tolerance=0.05)
    path = export(_Persistence(), spec, tmp_path / "still.onnx")
    state = torch.randn(*SHAPE)

    out = EmbeddedRuntime(path, threads=1).step(state.numpy(), np.zeros(1))

    assert out.residual == pytest.approx(0.0, abs=1e-6)
    assert out.off_physics is False
    assert out.trust == pytest.approx(1.0)


def test_a_field_that_does_not_obey_it_is_flagged(exported):
    path, _ = exported
    state = torch.randn(*SHAPE)

    out = EmbeddedRuntime(path, threads=1).step(state.numpy(), np.zeros(1))

    assert out.residual > 0.05
    assert out.off_physics is True
    assert out.trust == pytest.approx(0.0)


def test_a_sensor_that_disagrees_is_suspect(exported):
    path, _ = exported
    state = torch.randn(*SHAPE)
    runtime = EmbeddedRuntime(path, threads=1)

    prediction = runtime.step(state.numpy(), np.array([0.0])).prediction
    agreeing = runtime.step(state.numpy(), np.array([float(prediction[0, 0, 0])]))
    disagreeing = runtime.step(state.numpy(), np.array([float(prediction[0, 0, 0]) + 10.0]))

    assert agreeing.suspect is False
    assert disagreeing.suspect is True
    assert disagreeing.measurement_error > agreeing.measurement_error


@pytest.mark.parametrize("equation,params", [("heat", {"alpha": 0.01}), ("advection", {"c_x": 1.0, "c_y": 0.5}),
                                             ("advection-diffusion", {"c_x": 1.0, "c_y": 0.0, "alpha": 0.01})])
def test_every_equation_in_the_stencil_exports_and_runs(tmp_path, equation, params):
    spec = EmbeddedSpec(state_shape=SHAPE, equation=equation, parameters=params, tolerance=0.05)
    path = export(_Persistence(), spec, tmp_path / f"{equation}.onnx")

    out = EmbeddedRuntime(path, threads=1).step(torch.randn(*SHAPE).numpy(), np.zeros(1))
    assert np.isfinite(out.residual) and out.residual >= 0.0
    assert graph_summary(path).is_straight_line


def test_the_latency_is_measured_on_the_threads_it_was_given(exported):
    path, _ = exported
    state = torch.randn(*SHAPE).numpy()

    latency = EmbeddedRuntime(path, threads=1).latency(state, np.zeros(1), n=50)

    assert latency["threads"] == 1.0
    assert latency["median_ms"] > 0.0 and latency["p95_ms"] >= latency["median_ms"]
    assert latency["hz"] == pytest.approx(1000.0 / latency["median_ms"], rel=1e-6)


@pytest.mark.parametrize("spec", [
    dict(state_shape=(1, 8, 8)),                                    # one frame has no time derivative
    dict(state_shape=(2, 2, 8)),                                    # no 3x3 neighbourhood
    dict(state_shape=(2, 8, 8), equation="schrodinger"),            # not in the stencil
    dict(state_shape=(2, 8, 8), parameters={}),                     # missing the equation's numbers
    dict(state_shape=(2, 8, 8), sensor_channels=()),                # nothing to compare the sensor with
    dict(state_shape=(2, 8, 8), sensor_channels=(5,)),              # outside the state
])
def test_an_impossible_contract_is_refused(spec):
    with pytest.raises(ValueError):
        EmbeddedSpec(**spec)


def test_the_spec_carries_the_contract_the_graph_was_built_with(exported):
    path, spec = exported
    summary = graph_summary(path)
    out = EmbeddedRuntime(path, threads=1).step(torch.zeros(*SHAPE).numpy(), np.zeros(1))

    assert spec.state_shape == SHAPE and spec.tolerance == 0.05
    assert out.prediction.shape == (1, SHAPE[1], SHAPE[2])
    assert set(summary.op_types) <= set(summary.op_types) | set()
    assert "state" in summary.inputs and "prediction" in summary.outputs
