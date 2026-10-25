"""Steady conduction in one unit cell of a plate-fin array (base slab + one fin), cell-centred finite volumes on a
structured grid, sparse direct solve. An independent numerical reference for the 1-D fin-array formula of
:mod:`pinneapple_physics.closed_form.fin_array_conduction`.

Unit cell (x across the fin pitch, z up, y along the fin depth)::

    base slab  0 <= x <= pitch,              0 <= z <= t_base
    fin        (pitch - t_fin)/2 <= x <= (pitch + t_fin)/2,   t_base <= z <= t_base + H

    k (T_xx + T_zz) = 0
    -k dT/dz = q_in              on z = 0 (this fin's share of the power over its footprint pitch x depth)
    -k dT/dn = h (T - T_amb)     on the fin sides, the fin tip and the exposed top of the base
    dT/dn = 0                    on x = 0 and x = pitch (symmetry with the neighbouring cells)

The 3-D cell of the source project (FEniCS, faces y = 0 and y = depth adiabatic) has data independent of y, so its
solution is independent of y and this 2-D cross-section is the same problem. Robin faces use the series
conductance of the half cell and the film, the heated bottom face temperature is extrapolated from the first cell
(T + q dz / 2k). ``junction_c`` is the mean bottom-face temperature (the quantity the 1-D formula gives as the base
temperature). Second order in the cell size: solving with ``cell``, ``cell / 2`` and ``cell / 4`` and passing the
results to :func:`pinneapple_lab.uq.gci` gives the discretisation band.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["solve_fin_cell"]


def _axis(breaks: list[float], target: float) -> np.ndarray:
    """Cell faces through every break point, each segment split into cells no larger than ``target``."""
    faces = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:], strict=False):
        n = max(2, int(np.ceil((b - a) / target)))
        faces.extend(np.linspace(a, b, n + 1)[1:])
    return np.asarray(faces)


def solve_fin_cell(*, pitch: float, t_fin: float, height: float, t_base: float, depth: float, k: float,
                   h_conv: float, power_per_fin: float, t_amb: float = 25.0, cell: float | None = None,
                   ) -> dict[str, Any]:
    """Temperature field of the unit cell; lengths in m, ``power_per_fin`` in W over the footprint pitch x depth.

    ``cell`` is the target cell size (default: the thinnest feature / 6). Returns ``junction_c`` (mean heated-face
    temperature), ``fin_root_c`` / ``fin_tip_c`` (fin centreline at the root and the tip), ``heat_balance_error``
    (|power in - convected| / power in), the grid size and solve time, and the field (``x``, ``z`` cell centres,
    ``T`` with NaN outside the solid)."""
    if not 0 < t_fin < pitch:
        raise ValueError("need 0 < t_fin < pitch")
    t0 = time.perf_counter()
    target = cell or min(t_fin, pitch - t_fin, t_base) / 6.0
    x0, x1 = (pitch - t_fin) / 2.0, (pitch + t_fin) / 2.0
    xf = _axis([0.0, x0, x1, pitch], target)
    zf = _axis([0.0, t_base, t_base + height], max(target, min(height, t_base) / 40.0))
    xc, zc = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (zf[1:] + zf[:-1])
    dx, dz = np.diff(xf), np.diff(zf)
    nx, nz = xc.size, zc.size
    solid = (zc[None, :] < t_base) | ((xc[:, None] > x0) & (xc[:, None] < x1))
    idx = -np.ones((nx, nz), int)
    idx[solid] = np.arange(solid.sum())
    n = int(solid.sum())
    q_in = power_per_fin / (pitch * depth)                     # W/m2 on the heated face
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    rhs = np.zeros(n)
    conv_g = np.zeros(n)                                       # film conductance per cell (W/K per m of depth)

    def link(a: int, b: int, g: float) -> None:
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([-g, -g])
        diag[a] += g
        diag[b] += g

    for i in range(nx):
        for j in range(nz):
            p = idx[i, j]
            if p < 0:
                continue
            if i + 1 < nx and idx[i + 1, j] >= 0:
                link(p, idx[i + 1, j], k * dz[j] / (0.5 * (dx[i] + dx[i + 1])))
            if j + 1 < nz and idx[i, j + 1] >= 0:
                link(p, idx[i, j + 1], k * dx[i] / (0.5 * (dz[j] + dz[j + 1])))
            robin = []                                         # (face length, half-cell distance)
            if i > 0 and idx[i - 1, j] < 0:
                robin.append((dz[j], dx[i] / 2))
            if i + 1 < nx and idx[i + 1, j] < 0:
                robin.append((dz[j], dx[i] / 2))
            if j + 1 == nz or idx[i, j + 1] < 0:
                robin.append((dx[i], dz[j] / 2))
            for area, half in robin:
                g = 1.0 / (1.0 / (h_conv * area) + half / (k * area))
                diag[p] += g
                rhs[p] += g * t_amb
                conv_g[p] += g
            if j == 0:
                rhs[p] += q_in * dx[i]
    A = sp.csr_matrix((np.concatenate([vals, diag]), (np.concatenate([rows, np.arange(n)]),
                                                      np.concatenate([cols, np.arange(n)]))), shape=(n, n))
    T = spla.spsolve(A.tocsc(), rhs)
    field = np.full((nx, nz), np.nan)
    field[solid] = T
    bottom = field[:, 0] + q_in * dz[0] / (2.0 * k)
    centre = int(np.argmin(np.abs(xc - pitch / 2)))
    fin_rows = np.where(zc > t_base)[0]
    q_total = q_in * pitch
    q_out = float(np.sum(conv_g * (T - t_amb)))
    return {"junction_c": float(np.sum(bottom * dx) / pitch),
            "fin_root_c": float(field[centre, fin_rows[0]]) if fin_rows.size else None,
            "fin_tip_c": float(field[centre, fin_rows[-1]]) if fin_rows.size else None,
            "heat_balance_error": abs(q_total - q_out) / q_total,
            "n_cells": n, "cell_size_m": float(target), "solve_time_s": time.perf_counter() - t0,
            "x": xc, "z": zc, "T": field}
