"""3D finite-volume conduction solve of the heat-sink base plate.

Independent of the closed-form spreading formula: it resolves the actual
rectangular source footprint (and an off-centre position) on the actual
rectangular base, so it both produces the temperature map shown to the
user and cross-checks :func:`spreading_resistance` (Lee et al.).

Model (same assumptions as the resistance network):
* bottom face: uniform heat flux Q/A_source over the footprint, adiabatic
  elsewhere; side faces adiabatic;
* top face: fins + exposed base lumped into a uniform effective
  coefficient ``h_eff = 1 / (R_fins * A_base)`` to ambient.

The linear system is assembled on cell centres (in-plane cell count held
roughly constant across aspect ratios) and solved directly (scipy.sparse); the energy balance (heat out through the top = Q) is
reported as a check.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def _overlap_1d(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Length of each cell [edges[i], edges[i+1]] inside [lo, hi]."""
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)


def solve_base_field(
    *, base_width: float, base_depth: float, base_thickness: float, k: float,
    power_w: float, r_fins_k_w: float, source_width: float, source_depth: float,
    source_x: Optional[float] = None, source_y: Optional[float] = None,
    nx: int = 48, ny: int = 48, nz: Optional[int] = None,
) -> Dict[str, object]:
    W, D, tb = base_width, base_depth, base_thickness
    # Keep ~nx*ny in-plane cells whatever the aspect ratio, and a few layers
    # through the thickness: the system stays small enough for a direct solve.
    cells = nx * ny
    nx = int(np.clip(round(math.sqrt(cells * W / D)), 8, 4 * math.sqrt(cells)))
    ny = int(np.clip(round(cells / nx), 8, 4 * math.sqrt(cells)))
    nz = nz or int(np.clip(round(tb / min(W / nx, D / ny) * 2), 3, 6))
    dx, dy, dz = W / nx, D / ny, tb / nz
    cx = W / 2 if source_x is None else source_x
    cy = D / 2 if source_y is None else source_y
    x0, x1 = cx - source_width / 2, cx + source_width / 2
    y0, y1 = cy - source_depth / 2, cy + source_depth / 2
    if x0 < -1e-12 or y0 < -1e-12 or x1 > W + 1e-12 or y1 > D + 1e-12:
        raise ValueError("Heat source extends beyond the base.")

    n = nx * ny * nz
    idx = np.arange(n).reshape(nz, ny, nx)
    gx, gy, gz = k * dy * dz / dx, k * dx * dz / dy, k * dx * dy / dz
    h_eff = 1.0 / (r_fins_k_w * W * D)
    g_top = 1.0 / (dz / (2 * k * dx * dy) + 1.0 / (h_eff * dx * dy))

    rows, cols, vals = [], [], []
    diag = np.zeros(n)

    def link(a, b, g):
        a, b = a.ravel(), b.ravel()
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([np.full(a.size, -g), np.full(b.size, -g)])
        np.add.at(diag, a, g)
        np.add.at(diag, b, g)

    link(idx[:, :, :-1], idx[:, :, 1:], gx)
    link(idx[:, :-1, :], idx[:, 1:, :], gy)
    link(idx[:-1, :, :], idx[1:, :, :], gz)
    top = idx[-1].ravel()
    diag[top] += g_top

    A = sps.csr_matrix((np.concatenate(vals + [diag]),
                        (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                       shape=(n, n))
    fx = _overlap_1d(np.linspace(0, W, nx + 1), x0, x1)
    fy = _overlap_1d(np.linspace(0, D, ny + 1), y0, y1)
    area = np.outer(fy, fx)                              # (ny, nx) source area per bottom cell
    q_flux = power_w / (source_width * source_depth)
    rhs = np.zeros(n)
    rhs[idx[0].ravel()] = (q_flux * area).ravel()

    theta = spla.spsolve(A.tocsc(), rhs)
    theta = theta.reshape(nz, ny, nx)   # T - T_ambient

    # Bottom-surface temperature: half-cell conduction correction under the source.
    bottom = theta[0] + q_flux * (area / (dx * dy)) * dz / (2 * k)
    q_out = float(np.sum(g_top * theta[-1]))
    # Top-face (fin-root side) surface temperature: cell centre minus the
    # half-cell conduction drop carrying that cell's outgoing flux.
    top = theta[-1] - (g_top * theta[-1]) * dz / (2 * k * dx * dy)
    under = area > 0
    return {
        "theta_bottom": bottom,                 # (ny, nx) K above ambient
        "theta_max": float(bottom.max()),
        "theta_source_avg": float(np.sum(bottom * area) / np.sum(area)),
        "theta_top_avg": float(theta[-1].mean()),
        "energy_balance_rel_error": abs(q_out - power_w) / power_w,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "hotspot_xy_m": [float((np.argmax(bottom) % nx + 0.5) * dx),
                         float((np.argmax(bottom) // nx + 0.5) * dy)],
        "under_source_cells": int(under.sum()),
        "theta_top": top,                       # (ny, nx) K above ambient
        "x_centers": (np.arange(nx) + 0.5) * dx,
        "y_centers": (np.arange(ny) + 0.5) * dy,
    }
