"""Independent reference solutions for the 2D and 3D cases: cell-centred finite volumes, sparse direct solve.

The PINN never sees these fields; they are only used to check h, the temperature field and the hot spot. Grid
convergence is checked by solving on two grids (``convergence``).

2D plate (thin plate, conduction in-plane, convection from both faces, a heat source under the device):
    k t (T_xx + T_yy) - 2 h (T - T_air) + q''(x, y) = 0,   edges adiabatic
3D block (conduction, device flux on the bottom, convection on the top):
    T_xx + T_yy + T_zz = 0,   -k T_z = q''(x, y) at z = 0,   -k T_z = h (T - T_air) at z = H,   sides adiabatic
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["footprint", "plate_2d", "block_3d"]


def footprint(x, y, xc: float, yc: float, half: float, width: float):
    """Smooth indicator of a square device footprint (1 inside, 0 outside, tanh edges of ``width``)."""
    sig = lambda s: 0.5 * (1 + np.tanh(s / width))  # noqa: E731
    return sig(half - np.abs(x - xc)) * sig(half - np.abs(y - yc))


def _footprint_area(xc, yc, half, width, lx, ly, n=2000) -> float:
    x = (np.arange(n) + 0.5) * lx / n
    y = (np.arange(n) + 0.5) * ly / n
    return float(footprint(x[None, :], y[:, None], xc, yc, half, width).sum() * (lx / n) * (ly / n))


def plate_2d(h: float, *, lx: float, ly: float, t: float, k: float, power: float, t_air: float,
             device: Tuple[float, float, float, float], nx: int = 200, ny: int = 120):
    """Temperature (°C) at the cell centres of an (ny, nx) grid; returns (x, y, T)."""
    xc, yc, half, width = device
    q_flux = power / _footprint_area(xc, yc, half, width, lx, ly)           # W/m2 so that the total is ``power``
    dx, dy = lx / nx, ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    idx = lambda i, j: j * nx + i  # noqa: E731
    gx, gy = k * t * dy / dx, k * t * dx / dy
    rows, cols, vals = [], [], []
    rhs = np.zeros(nx * ny)
    for j in range(ny):
        for i in range(nx):
            p = idx(i, j)
            diag = 2 * h * dx * dy                                             # both faces
            for di, dj, g in ((1, 0, gx), (-1, 0, gx), (0, 1, gy), (0, -1, gy)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    rows.append(p); cols.append(idx(ii, jj)); vals.append(-g)  # noqa: E702
                    diag += g
            rows.append(p); cols.append(p); vals.append(diag)  # noqa: E702
            rhs[p] = 2 * h * dx * dy * t_air + q_flux * footprint(x[i], y[j], xc, yc, half, width) * dx * dy
    a = sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    return x, y, spla.spsolve(a, rhs).reshape(ny, nx)


def block_3d(h: float, *, lx: float, ly: float, lz: float, k: float, power: float, t_air: float,
             device: Tuple[float, float, float, float], nx: int = 40, ny: int = 40, nz: int = 12):
    """Temperature (°C) at the cell centres of an (nz, ny, nx) grid; returns (x, y, z, T)."""
    xc, yc, half, width = device
    q_flux = power / _footprint_area(xc, yc, half, width, lx, ly)
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    n = nx * ny * nz
    idx = lambda i, j, kz: (kz * ny + j) * nx + i  # noqa: E731
    g = {0: k * dy * dz / dx, 1: k * dx * dz / dy, 2: k * dx * dy / dz}
    g_top = (h * dx * dy) / (1 + h * dz / (2 * k))                           # half cell + film, in series
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    for kz in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, kz)
                diag = 0.0
                for d, (di, dj, dk) in ((0, (1, 0, 0)), (0, (-1, 0, 0)), (1, (0, 1, 0)), (1, (0, -1, 0)),
                                        (2, (0, 0, 1)), (2, (0, 0, -1))):
                    ii, jj, kk = i + di, j + dj, kz + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        rows.append(p); cols.append(idx(ii, jj, kk)); vals.append(-g[d])  # noqa: E702
                        diag += g[d]
                if kz == nz - 1:
                    diag += g_top
                    rhs[p] += g_top * t_air
                if kz == 0:
                    rhs[p] += q_flux * footprint(x[i], y[j], xc, yc, half, width) * dx * dy
                rows.append(p); cols.append(p); vals.append(diag)  # noqa: E702
    a = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return x, y, z, spla.spsolve(a, rhs).reshape(nz, ny, nx)


def interp_grid(xs, ys, field, px, py):
    """Bilinear interpolation of a cell-centred 2D field at points (px, py), clamped to the grid."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((ys, xs), field, bounds_error=False, fill_value=None)
    return f(np.column_stack([py, px]))


def convergence(fn: Callable, coarse: dict, fine: dict, **kw) -> float:
    """Largest temperature difference (K) between two grids, compared at the coarse cell centres."""
    from scipy.interpolate import RegularGridInterpolator
    a, b = fn(**kw, **coarse), fn(**kw, **fine)
    axes_a, axes_b = a[:-1][::-1], b[:-1][::-1]                         # (y, x) or (z, y, x)
    f = RegularGridInterpolator(axes_b, b[-1], bounds_error=False, fill_value=None)
    pts = np.stack(np.meshgrid(*axes_a, indexing="ij"), axis=-1).reshape(-1, len(axes_a))
    return float(np.max(np.abs(f(pts).reshape(a[-1].shape) - a[-1])))


