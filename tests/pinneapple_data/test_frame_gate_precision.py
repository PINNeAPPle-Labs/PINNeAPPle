"""The dtype-aware frame gate, on the precision the archives actually store coordinates in."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.schema import AxisSpec, Field, PhysicsDataBundle, run_gates


def bundle_with_x(x: np.ndarray, t: np.ndarray | None = None) -> PhysicsDataBundle:
    u = np.zeros((len(t) if t is not None else 4, len(x)), dtype=np.float64)
    return PhysicsDataBundle.from_arrays(
        {"field:u": u, "axis:t": (t if t is not None else np.linspace(0, 1, 4)), "axis:x": x},
        fields=[Field.from_array("u", u, axes=("t", "x"))],
        axes=[AxisSpec("t", "1"), AxisSpec("x", "1")], source="grid", license="CC0-1.0", kind="synthetic")


def test_a_float32_axis_passes_against_its_own_rounding():
    """PDEBench stores x as float32: a 1024-point linspace on [-1, 1] is 1.1e-5 from perfectly uniform, which is
    the storage format, not a geometry error (measured on Sod6.hdf5)."""
    x = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    report = run_gates(bundle_with_x(x), gates=("frame",))
    gate = report.get("frame")
    assert gate.passed is True
    assert gate.value > 1e-5                       # the deviation a fixed 1e-5 tolerance would have failed
    assert gate.tolerance >= gate.value            # ... and the tolerance that was actually applied covers it
    assert "float32" in gate.detail


def test_a_float64_axis_is_held_to_the_tight_tolerance():
    x = np.linspace(0.0, 1.0, 64)
    gate = run_gates(bundle_with_x(x), gates=("frame",)).get("frame")
    assert gate.passed is True
    assert gate.tolerance == pytest.approx(1e-5)
    assert "float32" not in gate.detail


def test_a_genuinely_non_uniform_float32_axis_still_fails():
    x = np.linspace(0.0, 1.0, 100, dtype=np.float32).astype(np.float32)
    x[50:] *= 1.1                                   # a 10 % stretch: far above the float32 noise
    gate = run_gates(bundle_with_x(x), gates=("frame",)).get("frame")
    assert gate.passed is False
    assert gate.value > gate.tolerance


def test_a_shuffled_axis_is_refused_whatever_the_dtype():
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x[[2, 5]] = x[[5, 2]]
    gate = run_gates(bundle_with_x(x), gates=("frame",)).get("frame")
    assert gate.passed is False and "monotonic" in gate.detail
