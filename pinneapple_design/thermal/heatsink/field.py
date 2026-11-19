"""3D finite-volume conduction solve of the heat-sink base plate.

Independent of the closed-form spreading formula: it resolves the actual
rectangular source footprint (and an off-centre position) on the actual
rectangular base, so it both produces the temperature map shown to the
user and cross-checks :func:`spreading_resistance` (Lee et al.).

Model (same assumptions as the resistance network):
* bottom face: uniform heat flux Q/A_source over each footprint, adiabatic
  elsewhere; side faces adiabatic;
* top face: fins + exposed base lumped into a uniform effective
  coefficient ``h_eff = 1 / (R_fins * A_base)`` to ambient.

Several sources are supported (``sources``): each is its own footprint with
its own power, and because the problem is linear the fields superpose -- the
hot spot of the small chip is what the network cannot say and this solve can.

The linear system is assembled on cell centres (in-plane cell count held
roughly constant across aspect ratios) and solved directly (scipy.sparse); the energy balance (heat out through the top = Q) is
reported as a check.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def _overlap_1d(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Length of each cell [edges[i], edges[i+1]] inside [lo, hi]."""
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)


def solve_base_field(
    *, base_width: float, base_depth: float, base_thickness: float, k: float,
    power_w: float = 0.0, r_fins_k_w: float, source_width: float = 0.0, source_depth: float = 0.0,
    source_x: float | None = None, source_y: float | None = None,
    sources: Sequence[dict[str, float]] | None = None,
    nx: int = 48, ny: int = 48, nz: int | None = None,
) -> dict[str, object]:
    """``sources``: list of ``{"power_w", "width", "depth", "x", "y"}`` footprints (``x``/``y`` the centre; default
    centred). When given it replaces the single-source arguments."""
    W, D, tb = base_width, base_depth, base_thickness
    # Keep ~nx*ny in-plane cells whatever the aspect ratio, and a few layers
    # through the thickness: the system stays small enough for a direct solve.
    cells = nx * ny
    nx = int(np.clip(round(math.sqrt(cells * W / D)), 8, 4 * math.sqrt(cells)))
    ny = int(np.clip(round(cells / nx), 8, 4 * math.sqrt(cells)))
    nz = nz or int(np.clip(round(tb / min(W / nx, D / ny) * 2), 3, 6))
    dx, dy, dz = W / nx, D / ny, tb / nz
    srcs = ([{"power_w": power_w, "width": source_width, "depth": source_depth,
              "x": W / 2 if source_x is None else source_x, "y": D / 2 if source_y is None else source_y}]
            if sources is None else [dict(s) for s in sources])
    for s in srcs:
        s.setdefault("x", W / 2)
        s.setdefault("y", D / 2)
        if s["x"] - s["width"] / 2 < -1e-12 or s["y"] - s["depth"] / 2 < -1e-12 \
                or s["x"] + s["width"] / 2 > W + 1e-12 or s["y"] + s["depth"] / 2 > D + 1e-12:
            raise ValueError("Heat source extends beyond the base.")
    total_power = float(sum(s["power_w"] for s in srcs))

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
    xe, ye = np.linspace(0, W, nx + 1), np.linspace(0, D, ny + 1)
    flux = np.zeros((ny, nx))                             # W/m2 on the bottom face, summed over the sources
    per_source_area = []
    for s in srcs:
        area = np.outer(_overlap_1d(ye, s["y"] - s["depth"] / 2, s["y"] + s["depth"] / 2),
                        _overlap_1d(xe, s["x"] - s["width"] / 2, s["x"] + s["width"] / 2))
        per_source_area.append(area)
        flux += s["power_w"] / (s["width"] * s["depth"]) * (area / (dx * dy))    # cell-averaged W/m2
    rhs = np.zeros(n)
    rhs[idx[0].ravel()] = (flux * dx * dy).ravel()

    theta = spla.spsolve(A.tocsc(), rhs)
    theta = theta.reshape(nz, ny, nx)   # T - T_ambient

    # Bottom-surface temperature: half-cell conduction correction under each source.
    bottom = theta[0] + flux * dz / (2 * k)
    q_out = float(np.sum(g_top * theta[-1]))
    # Top-face (fin-root side) surface temperature: cell centre minus the
    # half-cell conduction drop carrying that cell's outgoing flux.
    top = theta[-1] - (g_top * theta[-1]) * dz / (2 * k * dx * dy)
    covered = flux > 0
    chip = []
    for s, area in zip(srcs, per_source_area, strict=True):
        m = area > 0
        chip.append({"power_w": float(s["power_w"]), "theta_max": float(bottom[m].max()) if m.any() else 0.0,
                     "theta_avg": float(np.sum(bottom * area) / np.sum(area)) if area.sum() > 0 else 0.0})
    return {
        "theta_bottom": bottom,                 # (ny, nx) K above ambient
        "theta_max": float(bottom.max()),
        "theta_source_avg": float(np.sum(bottom * flux) / np.sum(flux)) if covered.any() else 0.0,
        "theta_top_avg": float(theta[-1].mean()),
        "energy_balance_rel_error": abs(q_out - total_power) / max(total_power, 1e-12),
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "hotspot_xy_m": [float((np.argmax(bottom) % nx + 0.5) * dx),
                         float((np.argmax(bottom) // nx + 0.5) * dy)],
        "under_source_cells": int(covered.sum()),
        "sources": chip,                        # per-source hot spot and mean, K above ambient
        "total_power_w": total_power,
        "theta_top": top,                       # (ny, nx) K above ambient
        "x_centers": (np.arange(nx) + 0.5) * dx,
        "y_centers": (np.arange(ny) + 0.5) * dy,
    }
