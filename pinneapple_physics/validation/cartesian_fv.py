"""Conservative Cartesian finite-volume balance checks in two and three dimensions.

Face arrays are oriented in the positive coordinate direction. Source
integrals are per cell (not densities). This checker does not infer the
underlying PDE or substitute for an independent numerical reference.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


def check_cartesian_fv_conservation(
    source_integrals: Any,
    face_fluxes: Mapping[str, Any],
    *,
    boundary_fluxes: Mapping[str, tuple[Any, Any]] | None = None,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
    max_cells: int = 2_000_000,
) -> dict[str, Any]:
    source = np.asarray(source_integrals, dtype=np.float64)
    if (type(max_cells) is not int or max_cells < 1 or max_cells > 2_000_000):
        raise ValueError("max_cells must be an integer in 1..2_000_000")
    if source.ndim not in (2, 3) or source.size < 1 or source.size > max_cells:
        raise ValueError("source must be a bounded 2D or 3D cell array")
    if not np.isfinite(source).all():
        raise ValueError("source contains nonfinite values")
    if not (np.isfinite(absolute_tolerance) and absolute_tolerance >= 0 and
            np.isfinite(relative_tolerance) and relative_tolerance >= 0):
        raise ValueError("invalid tolerances")
    axes = "xyz"[:source.ndim]
    if set(face_fluxes) != set(axes):
        raise ValueError("one positive-direction face flux array per axis required")
    if boundary_fluxes is not None and set(boundary_fluxes) != set(axes):
        raise ValueError("boundary targets must cover all axes")
    divergence = np.zeros_like(source)
    local_scale = np.abs(source).copy()
    net_boundary_flux = 0.0
    boundary_errors: dict[str, list[float]] = {}
    boundary_pass = True
    canonical = {"source": source.tolist(), "fluxes": {}, "boundary": {}}
    for axis, name in enumerate(axes):
        flux = np.asarray(face_fluxes[name], dtype=np.float64)
        shape = list(source.shape)
        shape[axis] += 1
        if list(flux.shape) != shape or not np.isfinite(flux).all():
            raise ValueError(f"{name} face flux has invalid shape or values")
        positive = np.take(flux, range(1, flux.shape[axis]), axis=axis)
        negative = np.take(flux, range(0, flux.shape[axis] - 1), axis=axis)
        divergence += positive - negative
        local_scale += np.abs(positive) + np.abs(negative)
        low = np.take(flux, 0, axis=axis)
        high = np.take(flux, -1, axis=axis)
        net_boundary_flux += float(high.sum(dtype=np.float64) - low.sum(dtype=np.float64))
        canonical["fluxes"][name] = flux.tolist()
        if boundary_fluxes is not None:
            if len(boundary_fluxes[name]) != 2:
                raise ValueError(f"{name} boundary target must be a pair")
            low_target = np.asarray(boundary_fluxes[name][0], dtype=np.float64)
            high_target = np.asarray(boundary_fluxes[name][1], dtype=np.float64)
            if (low_target.shape != low.shape or high_target.shape != high.shape or
                    not np.isfinite(low_target).all() or not np.isfinite(high_target).all()):
                raise ValueError(f"{name} boundary target shape mismatch")
            errors = [float(np.max(np.abs(low - low_target))),
                      float(np.max(np.abs(high - high_target)))]
            limits = [absolute_tolerance + relative_tolerance * max(
                float(np.max(np.abs(low))), float(np.max(np.abs(low_target)))),
                      absolute_tolerance + relative_tolerance * max(
                float(np.max(np.abs(high))), float(np.max(np.abs(high_target))))]
            boundary_errors[name] = errors
            boundary_pass &= all(e <= lim for e, lim in zip(errors, limits))
            canonical["boundary"][name] = [low_target.tolist(), high_target.tolist()]
    residual = divergence - source
    local_limit = absolute_tolerance + relative_tolerance * local_scale
    local_pass = bool(np.all(np.abs(residual) <= local_limit))
    global_error = net_boundary_flux - float(source.sum(dtype=np.float64))
    global_scale = max(1.0, abs(net_boundary_flux), float(np.abs(source).sum()))
    global_limit = absolute_tolerance + relative_tolerance * global_scale
    global_pass = bool(abs(global_error) <= global_limit)
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return {
        "schema": f"pinneapple.physics-check.cartesian-fv-{source.ndim}d/1",
        "dimensions": source.ndim, "shape": list(source.shape),
        "input_sha256": digest,
        "max_cell_balance_error": float(np.max(np.abs(residual))),
        "global_balance_error": float(global_error),
        "global_balance_limit": float(global_limit),
        "local_passed": local_pass, "global_passed": global_pass,
        "boundary_passed": boundary_pass if boundary_fluxes is not None else None,
        "boundary_max_errors": boundary_errors,
        "passed": bool(local_pass and global_pass and boundary_pass),
        "scientific_status": "numeric_balance_checked_not_independently_validated",
    }
