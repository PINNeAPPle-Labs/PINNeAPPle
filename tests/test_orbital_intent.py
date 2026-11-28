import numpy as np
import pytest
from pinneapple_simulation.orbital_intent import (
    propagate, hohmann, ScientificIntent, generate_dataset
)


def test_circular_orbit():
    t = np.linspace(0, 2*np.pi, 101)
    initial = [1, 0, 0, 0, 1, 0]
    y = propagate(1, initial, t)
    np.testing.assert_allclose(y[:, 0], np.cos(t), atol=1e-8)
    np.testing.assert_allclose(y[:, 1], np.sin(t), atol=1e-8)
    assert ScientificIntent("Circular orbit", ("two bodies",), 1e-8, 1e-8).evaluate(1, y)["passed"]
    assert generate_dataset(1, [initial, initial], t).shape == (2, 101, 6)


def test_transfer():
    m = hohmann(1, 1, 2)
    y = propagate(1, [1, 0, 0, 0, 1+m["dv1"], 0], [0, m["coast_time"]])
    assert np.linalg.norm(y[-1, :3]) == pytest.approx(2, abs=1e-8)
    assert np.linalg.norm(y[-1, 3:])+m["dv2"] == pytest.approx(np.sqrt(0.5), abs=1e-8)


def test_backend_contract():
    initial = [1, 0, 0, 0, 1, 0]
    with pytest.raises(ValueError):
        propagate(1, initial, [0, 1], backend=lambda *args: np.zeros((2, 6)))
    with pytest.raises(ValueError):
        propagate(1, initial, [1, 0])
