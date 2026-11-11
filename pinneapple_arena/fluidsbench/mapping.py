"""Getting a model's own output onto the benchmark's scoring entities, and saying exactly how far it reached.

The benchmark scores on its own coordinates, its own weights and its own ground truth, joined by ``support_id``; it never
interpolates and never resamples. Everything a model needs in order to meet that support — a query at the canonical
points, a nearest-neighbour transfer from its own mesh, a linear interpolation from its own grid — therefore happens here,
on the submitter's side, and is *declared*: how many entities were scored, how much of the published weight they carry,
how many could not be produced, and how many were extrapolated. A case that does not reach complete coverage of an
official support cannot be submitted, and this raises instead of writing a number that hides it.

    from pinneapple_arena.fluidsbench import map_to_support

    mapped = map_to_support(case, points=mesh.points, fields={"velocity_x_pred": u, "velocity_y_pred": v,
                                                              "pressure_pred": p}, method="nearest")
    print(mapped.coverage.as_dict())     # scored/total, weight fraction, unmapped, extrapolated

Four methods, each declared in the result package:

``identity``   the predictions are already on the support entities, in its order (a model queried at the canonical
               points, which is the cleanest submission: ``unmapped_fraction = 0``, ``method = identity``);
``by_id``      the predictions carry the support ids and are joined exactly; missing ids count as unmapped, unknown or
               duplicate ids are refused;
``nearest``    nearest-native-point transfer, with the distance of every entity reported and an optional
               ``max_distance`` beyond which an entity is extrapolated;
``linear``     linear interpolation in the support's coordinates (scipy), where an entity outside the native hull is
               extrapolated and NaN unless ``fallback="nearest"`` fills it — which still counts as extrapolated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .support import CaseSupport

__all__ = ["Coverage", "MappedPredictions", "CoverageError", "MappingError", "map_to_support"]

METHODS = frozenset({"identity", "by_id", "nearest", "linear"})


class MappingError(ValueError):
    """The native output cannot be placed on the support at all (unknown id, wrong shape, no points)."""


class CoverageError(MappingError):
    """The mapped predictions do not reach the coverage the contract requires."""

    def __init__(self, message: str, coverage: Coverage):
        super().__init__(message)
        self.coverage = coverage


@dataclass(frozen=True)
class Coverage:
    """What the transfer reached, in the vocabulary ``discretization/cases.jsonl`` and ``metrics/cases.json`` use."""
    method: str
    support_count: int
    scored_count: int
    unmapped_count: int
    extrapolated_count: int
    coverage_fraction: float
    weight_coverage_fraction: float
    max_distance: float | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"support_count": self.support_count, "scored_count": self.scored_count,
                               "unmapped_count": self.unmapped_count,
                               "extrapolated_count": self.extrapolated_count,
                               "coverage_fraction": self.coverage_fraction,
                               "weight_coverage_fraction": self.weight_coverage_fraction}
        if self.max_distance is not None:
            out["max_distance"] = self.max_distance
        return out

    def mapping_entry(self, support_id: str, source_output_id: str) -> dict[str, Any]:
        """The ``inference.mappings`` entry of the discretization record for this support."""
        return {"support_id": support_id, "source_output_id": source_output_id,
                "method": {"kind": "nearest_neighbour" if self.method == "nearest" else
                           "linear_interpolation" if self.method == "linear" else
                           "identity" if self.method == "identity" else "support_id_join"},
                "implementation": "pinneapple_arena.fluidsbench.mapping",
                "extrapolation_policy": "forbidden" if self.extrapolated_count == 0 else "declared",
                "unmapped_fraction": (self.unmapped_count / self.support_count) if self.support_count else 0.0,
                "extrapolated_fraction": (self.extrapolated_count / self.support_count) if self.support_count else 0.0,
                "final_coverage_fraction": self.coverage_fraction}


@dataclass(frozen=True)
class MappedPredictions:
    """Predictions on the support's entities, plus the coverage they were obtained with."""
    case_id: str
    support_id: str
    values: Mapping[str, np.ndarray]
    coverage: Coverage = field(default_factory=lambda: Coverage("identity", 0, 0, 0, 0, 1.0, 1.0))

    def complete(self) -> bool:
        return self.coverage.unmapped_count == 0 and self.coverage.coverage_fraction >= 1.0


def _aliases(case: CaseSupport) -> dict[str, str]:
    """Accept a prediction keyed by its ``prediction_field`` or by its component id."""
    out: dict[str, str] = {}
    for quantity in case.definition.quantities:
        for component in quantity.components:
            out[component.prediction_field] = component.prediction_field
            out.setdefault(component.id, component.prediction_field)
    return out


def _resolve(case: CaseSupport, fields: Mapping[str, Any]) -> dict[str, np.ndarray]:
    aliases = _aliases(case)
    resolved: dict[str, np.ndarray] = {}
    for key, value in fields.items():
        target = aliases.get(key)
        if target is None:
            raise MappingError(f"case {case.case_id!r}: {key!r} is not a prediction field of support "
                               f"{case.definition.id!r}; it takes {sorted(set(aliases.values()))}")
        array = np.asarray(value, dtype=np.float64)
        resolved[target] = array[:, None] if array.ndim == 1 else array
    missing = [f for f in case.definition.prediction_fields if f not in resolved]
    if missing:
        raise MappingError(f"case {case.case_id!r}: missing predictions for {missing}")
    return resolved


def _coverage(method: str, case: CaseSupport, scored_mask: np.ndarray, extrapolated_mask: np.ndarray,
              max_distance: float | None, notes: str = "") -> Coverage:
    total = case.entity_count
    scored = int(np.count_nonzero(scored_mask))
    weight_total = float(np.sum(case.weights))
    weight_scored = float(np.sum(case.weights[scored_mask])) if weight_total > 0 else 0.0
    return Coverage(method=method, support_count=total, scored_count=scored,
                    unmapped_count=total - scored, extrapolated_count=int(np.count_nonzero(extrapolated_mask)),
                    coverage_fraction=(scored / total) if total else 0.0,
                    weight_coverage_fraction=(weight_scored / weight_total) if weight_total > 0 else 0.0,
                    max_distance=max_distance, notes=notes)


def _enforce(case: CaseSupport, coverage: Coverage, *, allow_partial: bool) -> None:
    required = dict(case.definition.required_coverage or {})
    count_fraction = float(required.get("count_fraction", 1.0))
    weight_fraction = float(required.get("weight_fraction", 1.0))
    max_unmapped = int(required.get("unmapped_count", 0))
    problems = []
    if coverage.coverage_fraction + 1e-12 < count_fraction:
        problems.append(f"count coverage {coverage.coverage_fraction:.6f} < {count_fraction}")
    if coverage.weight_coverage_fraction + 1e-12 < weight_fraction:
        problems.append(f"weight coverage {coverage.weight_coverage_fraction:.6f} < {weight_fraction}")
    if coverage.unmapped_count > max_unmapped:
        problems.append(f"{coverage.unmapped_count} unmapped entities > {max_unmapped}")
    if case.definition.extrapolation_policy == "forbidden" and coverage.extrapolated_count:
        problems.append(f"{coverage.extrapolated_count} extrapolated entities while the policy forbids extrapolation")
    if problems and not allow_partial:
        raise CoverageError(f"case {case.case_id!r} support {case.definition.id!r}: " + "; ".join(problems)
                            + " — the contract scores complete official entity coverage", coverage)


def map_to_support(case: CaseSupport, *, points: Any = None, fields: Mapping[str, Any], method: str = "identity",
                   support_ids: Any = None, max_distance: float | None = None, tolerance: float = 1e-9,
                   allow_partial: bool = False, fallback: str | None = None) -> MappedPredictions:
    """Place ``fields`` (at ``points``, or keyed by ``support_ids``) onto the support's entities.

    Returns :class:`MappedPredictions` in the support's ``support_id_ascending`` order. A transfer that does not meet the
    support's ``required_coverage`` raises :class:`CoverageError` unless ``allow_partial`` is set, and an
    ``extrapolation_policy`` of ``forbidden`` is enforced the same way."""
    if method not in METHODS:
        raise MappingError(f"method must be one of {sorted(METHODS)}, not {method!r}")
    resolved = _resolve(case, fields)
    n_target = case.entity_count
    values: dict[str, np.ndarray] = {}
    extrapolated = np.zeros(n_target, dtype=bool)
    measured: float | None = None

    if method == "identity":
        for name, array in resolved.items():
            if array.shape[0] != n_target:
                raise MappingError(f"{name!r} has {array.shape[0]} rows, the support has {n_target} entities; "
                                   f"identity mapping needs one row per support entity")
            values[name] = array
        scored = np.ones(n_target, dtype=bool)

    elif method == "by_id":
        if support_ids is None:
            raise MappingError("by_id mapping needs the support_ids of the predicted rows")
        ids = np.asarray([str(i) for i in support_ids])
        if len(ids) != len(set(ids.tolist())):
            repeated = sorted({i for i in ids.tolist() if list(ids).count(i) > 1})[:3]
            raise MappingError(f"duplicate support_id in the predictions (e.g. {repeated})")
        index = {sid: row for row, sid in enumerate(ids.tolist())}
        unknown = sorted(set(index) - set(case.support_ids.tolist()))
        if unknown:
            raise MappingError(f"{len(unknown)} predicted support_id values are not in the support (e.g. {unknown[:3]})")
        rows = np.array([index.get(sid, -1) for sid in case.support_ids.tolist()])
        scored = rows >= 0
        for name, array in resolved.items():
            if array.shape[0] != len(ids):
                raise MappingError(f"{name!r} has {array.shape[0]} rows for {len(ids)} support ids")
            out = np.full((n_target, array.shape[1]), np.nan)
            out[scored] = array[rows[scored]]
            values[name] = out

    else:                                       # nearest / linear
        if points is None:
            raise MappingError(f"{method} mapping needs the native points to transfer from")
        native = np.asarray(points, dtype=np.float64)
        if native.ndim != 2:
            raise MappingError(f"native points must be (n, dim), got {native.shape}")
        target = np.asarray(case.coordinates, dtype=np.float64)
        if native.shape[1] != target.shape[1]:
            raise MappingError(f"native points are {native.shape[1]}-D, the support is {target.shape[1]}-D")
        if method == "nearest":
            distance, index = _nearest(target, native)
            measured = float(np.max(distance)) if distance.size else None
            if max_distance is not None:
                extrapolated = distance > float(max_distance)
            for name, array in resolved.items():
                if array.shape[0] != native.shape[0]:
                    raise MappingError(f"{name!r} has {array.shape[0]} rows for {native.shape[0]} native points")
                values[name] = array[index]
        else:
            interpolated, extruded = _linear(target, native, resolved)
            extrapolated = extruded
            for name, column in interpolated.items():
                if fallback == "nearest":
                    fill = extrapolated & ~np.all(np.isfinite(column), axis=1)
                    if np.any(fill):
                        _, index = _nearest(target[fill], native)
                        column[fill] = resolved[name][index]
                values[name] = column
        scored = np.ones(n_target, dtype=bool)
        if method == "linear":
            scored = np.all([np.all(np.isfinite(v), axis=1) for v in values.values()], axis=0)
        elif max_distance is not None:
            scored = ~extrapolated

    coverage = _coverage(method, case, scored, extrapolated, measured,
                         notes=f"tolerance {tolerance:g}" if method == "identity" else "")
    _enforce(case, coverage, allow_partial=allow_partial)
    return MappedPredictions(case_id=case.case_id, support_id=case.definition.id, values=values, coverage=coverage)


def _nearest(target: np.ndarray, native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(native)
    distance, index = tree.query(target, k=1)
    return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)


def _linear(target: np.ndarray, native: np.ndarray, resolved: Mapping[str, np.ndarray], *,
            off_tolerance: float | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Linear interpolation onto the support, in the affine space the native points actually span.

    A surface model in a 3-D support — exactly the pressure case — is a lower-dimensional point cloud, so the
    interpolation is done on the plane (or line) the native points span, and a target entity that is not on that space is
    *extrapolated*, never quietly projected onto it. Entries outside the hull come back NaN."""
    dim = native.shape[1]
    centre = native.mean(axis=0)
    centred = native - centre
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    scale = float(singular[0]) if singular.size else 0.0
    rank = int(np.count_nonzero(singular > 1e-12 * scale)) if scale > 0 else 0
    if rank == 0:
        raise MappingError("linear interpolation needs at least two distinct native points")
    basis = right[:rank]
    reduced = centred @ basis.T
    delta = target - centre
    target_reduced = delta @ basis.T
    residual = delta - target_reduced @ basis
    span = float(np.ptp(native, axis=0).max()) if native.size else 1.0
    tolerance = off_tolerance if off_tolerance is not None else 1e-9 * max(span, 1e-12)
    off_manifold = np.sqrt(np.sum(residual ** 2, axis=1)) > tolerance

    if rank == 1:
        order = np.argsort(reduced[:, 0])
        axis = reduced[order, 0]
        out = {}
        for name, array in resolved.items():
            column = np.full((target.shape[0], array.shape[1]), np.nan)
            for component in range(array.shape[1]):
                column[:, component] = np.interp(target_reduced[:, 0], axis, array[order, component])
            out[name] = column
        outside = (target_reduced[:, 0] < axis[0]) | (target_reduced[:, 0] > axis[-1]) | off_manifold
        return out, np.asarray(outside, dtype=bool)

    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay, QhullError

    try:
        Delaunay(reduced)
    except QhullError as error:                       # degenerate native cloud: no volume to interpolate in
        raise MappingError(f"linear interpolation needs a non-degenerate native cloud (rank {rank} in {dim}-D): "
                           f"{error}") from error
    out: dict[str, np.ndarray] = {}
    for name, array in resolved.items():
        column = np.full((target.shape[0], array.shape[1]), np.nan)
        for component in range(array.shape[1]):
            interpolator = LinearNDInterpolator(reduced, array[:, component])
            column[:, component] = interpolator(target_reduced)
        out[name] = column
    outside = np.any(~np.isfinite(np.concatenate([v for v in out.values()], axis=1)), axis=1) | off_manifold
    return out, np.asarray(outside, dtype=bool)


def native_entity_counts(fields: Mapping[str, Any], points: Any = None) -> dict[str, Any]:
    """The ``entity_counts`` summary of a native representation, for the discretization record."""
    rows = {np.asarray(v).shape[0] for v in fields.values()}
    if points is not None:
        rows.add(np.asarray(points).shape[0])
    if len(rows) > 1:
        raise MappingError(f"native fields disagree on their row count: {sorted(rows)}")
    count = rows.pop() if rows else 0
    return {"entity": "points", "count": {"kind": "fixed", "value": int(count)}}


def summarize_mapping(mapped: Sequence[MappedPredictions]) -> dict[str, Any]:
    """The fractions one support's mapping records across every case, as the discretization summary states them."""
    if not mapped:
        raise MappingError("no mapped case to summarise")
    unmapped = max(m.coverage.unmapped_count / m.coverage.support_count for m in mapped)
    extrapolated = max(m.coverage.extrapolated_count / m.coverage.support_count for m in mapped)
    final = min(m.coverage.coverage_fraction for m in mapped)
    return {"method": {"kind": mapped[0].coverage.method}, "unmapped_fraction": unmapped,
            "extrapolated_fraction": extrapolated, "final_coverage_fraction": final,
            "support_id": mapped[0].support_id, "case_count": len(mapped)}
