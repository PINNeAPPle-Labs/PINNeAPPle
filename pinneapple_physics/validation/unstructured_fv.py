"""Conservation verification for arbitrary face-cell incidence (unstructured FV).

Face fluxes are *integrated*, signed from owner to neighbour; boundary faces
have neighbour -1 and positive flux points outward from owner. Sources are
integrals per cell. This checks balance, not mesh geometry or PDE validity.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def check_unstructured_fv_conservation(
    owner, neighbour, face_flux, cell_source_integral, *,
    boundary_flux_targets=None,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
    max_cells: int = 2_000_000,
    max_faces: int = 8_000_000,
) -> dict:
    o = np.asarray(owner)
    n = np.asarray(neighbour)
    f = np.asarray(face_flux, dtype=np.float64)
    s = np.asarray(cell_source_integral, dtype=np.float64)
    if (s.ndim != 1 or f.ndim != 1 or o.ndim != 1 or n.ndim != 1
            or s.size < 1 or s.size > max_cells or f.size < 1
            or f.size > max_faces or not (o.size == n.size == f.size)):
        raise ValueError("invalid cell/face array dimensions or limits")
    if (o.dtype.kind not in "iu" or n.dtype.kind not in "iu"
            or np.any(o < 0) or np.any(o >= s.size)
            or np.any(n < -1) or np.any(n >= s.size)):
        raise ValueError("owner/neighbour must be bounded integer cell IDs; -1 is boundary")
    if np.any((n != -1) & (o == n)):
        raise ValueError("interior face cannot connect a cell to itself")
    if not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("flux and source integrals must be finite")
    if (not np.isfinite(absolute_tolerance) or absolute_tolerance < 0
            or not np.isfinite(relative_tolerance) or relative_tolerance < 0):
        raise ValueError("tolerances must be finite and nonnegative")
    balance = np.bincount(o, weights=f, minlength=s.size)
    scale = np.bincount(o, weights=np.abs(f), minlength=s.size)
    interior = n >= 0
    balance -= np.bincount(n[interior], weights=f[interior], minlength=s.size)
    scale += np.bincount(n[interior], weights=np.abs(f[interior]), minlength=s.size)
    residual = balance - s
    local_limit = absolute_tolerance + relative_tolerance * (scale + np.abs(s))
    local_pass = bool(np.all(np.abs(residual) <= local_limit))
    net_boundary = float(np.sum(f[~interior], dtype=np.float64))
    global_error = net_boundary - float(np.sum(s, dtype=np.float64))
    global_limit = absolute_tolerance + relative_tolerance * max(
        1.0, abs(net_boundary), float(np.sum(np.abs(s), dtype=np.float64)))
    global_pass = bool(abs(global_error) <= global_limit)
    boundary_pass = None
    boundary_max_error = None
    targets = None
    if boundary_flux_targets is not None:
        targets = np.asarray(boundary_flux_targets, dtype=np.float64)
        if targets.shape != (int(np.sum(~interior)),) or not np.isfinite(targets).all():
            raise ValueError("boundary_flux_targets must match boundary faces")
        delta = np.abs(f[~interior] - targets)
        limit = absolute_tolerance + relative_tolerance * np.maximum(
            np.abs(f[~interior]), np.abs(targets))
        boundary_pass = bool(np.all(delta <= limit))
        boundary_max_error = float(np.max(delta)) if delta.size else 0.0
    payload = {"owner": o.tolist(), "neighbour": n.tolist(),
               "face_flux": f.tolist(), "cell_source_integral": s.tolist(),
               "boundary_flux_targets": None if targets is None else targets.tolist()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return {
        "schema": "pinneapple.physics-check.unstructured-fv/1",
        "input_sha256": digest, "cell_count": int(s.size),
        "face_count": int(f.size), "boundary_face_count": int(np.sum(~interior)),
        "max_cell_balance_error": float(np.max(np.abs(residual))),
        "global_balance_error": global_error,
        "global_balance_limit": float(global_limit),
        "local_passed": local_pass, "global_passed": global_pass,
        "boundary_passed": boundary_pass,
        "boundary_max_error": boundary_max_error,
        "passed": bool(local_pass and global_pass and boundary_pass is not False),
        "scientific_status": "numeric_balance_checked_not_independently_validated",
    }
