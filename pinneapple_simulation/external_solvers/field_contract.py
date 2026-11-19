"""Strict, solver-neutral sampled field contract.

This contract does not pretend that cell- and node-centered values are
interchangeable or that different physical units can be compared directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

from pinneapple_analysis.spatial_sampling import (
    SamplingProtocol, require_compatible,
)


@dataclass(frozen=True)
class PhysicalField:
    name: str
    units: str
    components: tuple[str, ...]
    coordinates_m: tuple[tuple[float, ...], ...]
    values: tuple[tuple[float, ...], ...]
    protocol: SamplingProtocol
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.name or not self.units or not self.components:
            raise ValueError("Field name, units and components are required")
        if self.protocol.field != self.name or self.protocol.field_units != self.units:
            raise ValueError("Field/protocol name or units mismatch")
        if not self.coordinates_m or len(self.coordinates_m) != len(self.values):
            raise ValueError("Coordinates and values must have equal nonzero lengths")
        ndim = len(self.protocol.reference.axes)
        if len(set(self.components)) != len(self.components):
            raise ValueError("Component names must be unique")
        for point, values in zip(self.coordinates_m, self.values):
            if len(point) != ndim or len(values) != len(self.components):
                raise ValueError("Coordinate or field component shape mismatch")
            if not all(math.isfinite(v) for v in (*point, *values)):
                raise ValueError("Nonfinite coordinate or field value")


def compare_fields(prediction: PhysicalField, reference: PhysicalField) -> dict:
    """Compare aligned point samples only, with explicit protocol equality.

    A zero error is *not* scientific validation without independently
    validated solver data and discretization/convergence evidence.
    """
    require_compatible(prediction.protocol, reference.protocol)
    if prediction.components != reference.components:
        raise ValueError("Field component order differs")
    if prediction.coordinates_m != reference.coordinates_m:
        raise ValueError("Sampling coordinates differ: explicit interpolation required")
    errors = [a - b for row_a, row_b in zip(prediction.values, reference.values)
              for a, b in zip(row_a, row_b)]
    refs = [v for row in reference.values for v in row]
    mse = sum(e * e for e in errors) / len(errors)
    norm_ref = math.sqrt(sum(v * v for v in refs))
    return {
        "rmse": math.sqrt(mse),
        "rel_l2": math.sqrt(sum(e * e for e in errors)) / max(norm_ref, 1e-12),
        "max_abs_error": max(abs(e) for e in errors),
        "n_samples": len(prediction.values),
        "n_components": len(prediction.components),
        "protocol_compatible": True,
        "scientifically_validated": False,
    }
