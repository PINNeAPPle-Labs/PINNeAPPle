"""Numerical checks for one-dimensional finite-volume conservation and boundary flux.

The checker accepts *independently supplied* face fluxes and cell source
integrals. It cannot infer physical correctness from a model's own claims.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def check_fv_conservation_1d(
    face_flux: Any,
    cell_source_integral: Any,
    *,
    boundary_flux: tuple[float, float] | None = None,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Check flux balance per cell and globally.

    Sign convention: positive face flux points in the +x direction;
    source integral is positive for net production. For cell i:
    face_flux[i+1] - face_flux[i] = source_integral[i].
    Boundary flux targets are left/right +x-oriented fluxes.
    """
    f = np.asarray(face_flux, dtype=np.float64)
    s = np.asarray(cell_source_integral, dtype=np.float64)
    if f.ndim != 1 or s.ndim != 1 or not 1 <= s.size <= 1_000_000 or f.size != s.size + 1:
        raise ValueError("expected N source integrals and N+1 face fluxes")
    if not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("all fluxes and source integrals must be finite")
    if (not np.isfinite(absolute_tolerance) or absolute_tolerance < 0 or
            not np.isfinite(relative_tolerance) or relative_tolerance < 0):
        raise ValueError("tolerances must be finite and nonnegative")
    boundary = None
    if boundary_flux is not None:
        b = np.asarray(boundary_flux, dtype=np.float64)
        if b.shape != (2,) or not np.isfinite(b).all():
            raise ValueError("boundary_flux must contain two finite values")
        boundary = [float(b[0]), float(b[1])]
    residual = np.diff(f) - s
    scale = np.maximum.reduce([np.abs(f[:-1]), np.abs(f[1:]), np.abs(s)])
    local_limits = absolute_tolerance + relative_tolerance * scale
    local_pass = bool(np.all(np.abs(residual) <= local_limits))
    global_residual = float(f[-1] - f[0] - np.sum(s))
    global_scale = max(float(np.sum(np.abs(s))), float(np.max(np.abs(f))), 1.0)
    global_limit = absolute_tolerance + relative_tolerance * global_scale
    global_pass = abs(global_residual) <= global_limit
    boundary_pass = None
    boundary_error = None
    if boundary is not None:
        boundary_error = [float(f[0] - boundary[0]), float(f[-1] - boundary[1])]
        boundary_scale = np.maximum(np.abs(boundary), np.abs([f[0], f[-1]]))
        boundary_pass = bool(np.all(np.abs(boundary_error) <=
                                    absolute_tolerance + relative_tolerance * boundary_scale))
    payload = {"face_flux": f.tolist(), "cell_source_integral": s.tolist(),
               "boundary_flux": boundary}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "schema": "pinneapple.physics-check.fv-conservation-1d/1",
        "input_sha256": digest,
        "cell_count": int(s.size),
        "max_cell_balance_error": float(np.max(np.abs(residual))),
        "global_balance_error": global_residual,
        "global_limit": float(global_limit),
        "local_passed": local_pass,
        "global_passed": bool(global_pass),
        "boundary_error": boundary_error,
        "boundary_passed": boundary_pass,
        "passed": bool(local_pass and global_pass and boundary_pass is not False),
        "scientific_status": "numeric_balance_checked_not_independently_validated",
    }
