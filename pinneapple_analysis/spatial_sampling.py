"""VeriPhysics-compatible spatial sampling and comparison guard.

No implicit coordinate zeroing, phase mixing, time-window coercion or
peak-to-mean comparisons. A compatible result is not proof of physics accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math


class IncompatibleComparison(ValueError):
    """Comparison must be blocked until methodology is aligned."""


@dataclass(frozen=True)
class SpatialReference:
    frame: str
    origin_m: tuple[float, ...]
    axes: tuple[str, ...]
    location: str = "cell_center"  # node, cell_center, face_center
    units: str = "m"

    def __post_init__(self) -> None:
        if self.units != "m":
            raise ValueError("SpatialReference origin_m requires SI metres")
        if len(self.origin_m) != len(self.axes) or not self.axes:
            raise ValueError("Origin and axes must have the same nonzero dimension")
        if self.location not in {"node", "cell_center", "face_center"}:
            raise ValueError("Unknown sampling location")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Axes must be unique")
        if not all(math.isfinite(v) for v in self.origin_m):
            raise ValueError("Origin coordinates must be finite")


@dataclass(frozen=True)
class SamplingProtocol:
    reference: SpatialReference
    field: str
    field_units: str
    statistic: str  # mean, peak, minimum
    window_s: tuple[float, float]
    spatial_interval_m: tuple[float, float] | None = None
    phase: str | None = None
    phase_alpha_min: float | None = None

    def __post_init__(self) -> None:
        if self.statistic not in {"mean", "peak", "minimum"}:
            raise ValueError("Unknown statistic")
        if not (self.window_s[0] <= self.window_s[1] and
                all(math.isfinite(x) for x in self.window_s)):
            raise ValueError("Invalid temporal window")
        if self.spatial_interval_m is not None and (
            self.spatial_interval_m[0] > self.spatial_interval_m[1]
            or not all(math.isfinite(x) for x in self.spatial_interval_m)
        ):
            raise ValueError("Invalid spatial interval")
        if self.phase_alpha_min is not None and not 0 <= self.phase_alpha_min <= 1:
            raise ValueError("Phase alpha threshold must be in [0,1]")


@dataclass(frozen=True)
class CompatibilityDecision:
    allowed: bool
    reasons: tuple[str, ...]
    action: str


def compare_protocols(a: SamplingProtocol, b: SamplingProtocol) -> CompatibilityDecision:
    issues: list[str] = []
    for name in ("field", "field_units", "statistic", "window_s",
                 "spatial_interval_m", "phase", "phase_alpha_min"):
        if getattr(a, name) != getattr(b, name):
            issues.append(f"{name} differs")
    for name in ("frame", "origin_m", "axes", "location", "units"):
        if getattr(a.reference, name) != getattr(b.reference, name):
            issues.append(f"spatial_reference.{name} differs")
    return CompatibilityDecision(not issues, tuple(issues),
        "compare" if not issues else "align explicitly and record the transform, then re-check")


def require_compatible(a: SamplingProtocol, b: SamplingProtocol) -> None:
    decision = compare_protocols(a, b)
    if not decision.allowed:
        raise IncompatibleComparison("; ".join(decision.reasons))


def absolute_distance(sample_position_m: float, inlet_plane_m: float) -> float:
    """Position measured from the inlet plane; never from the first sample."""
    if not all(math.isfinite(v) for v in (sample_position_m, inlet_plane_m)):
        raise ValueError("Positions must be finite")
    return sample_position_m - inlet_plane_m


def liquid_peak(values: Sequence[float], alpha: Sequence[float],
                *, min_alpha: float = 0.9) -> tuple[int, float]:
    """Return index/value of the maximum in the liquid phase only."""
    if len(values) != len(alpha) or not values:
        raise ValueError("Values and phase fractions must have matching nonempty lengths")
    if not 0 <= min_alpha <= 1:
        raise ValueError("Invalid alpha threshold")
    valid = [(i, float(v)) for i, (v, a) in enumerate(zip(values, alpha))
             if math.isfinite(float(v)) and math.isfinite(float(a)) and float(a) >= min_alpha]
    if not valid:
        raise IncompatibleComparison("No valid liquid samples: do not report a peak in air")
    return max(valid, key=lambda item: item[1])
