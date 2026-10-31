"""Finite-volume conduction in one fin pitch of a straight-fin heat sink: an independent check of the 1-D fin-array
formula (:mod:`pinneapple_physics.closed_form.fin_array_conduction`).

The unit cell is one fin of thickness ``t`` and height ``H`` centred on a strip of base plate one pitch wide
(``pitch = spacing + t``) and ``t_base`` thick, the same cell a 3-D FEM check of a periodic fin array uses:

* bottom face: uniform heat flux, this fin's share of the power, ``q = P / (N * pitch * depth)``;
* exposed base top, fin sides and fin tip: convection ``-k dT/dn = h (T - T_amb)`` (Robin) with the same ``h`` as
  the formula;
* the two side planes ``x = 0`` and ``x = pitch``: symmetry with the neighbouring cells (zero flux);
* the two end planes along the depth: zero flux (the periodic simplification), with ``q`` and ``h`` uniform along
  the depth.

With nothing varying along the depth, the 3-D problem is exactly this 2-D cross-section problem (per unit depth),
which is what is solved here: cell-centred finite volumes on a grid that puts cell faces on the fin walls and the
fin root, two-point fluxes between cell centres (one material), half-cell resistances in series with ``1/h`` on convective
faces, and a direct sparse solve. Unlike the formula it resolves two-dimensional conduction in the fin (no 1-D fin
assumption, no corrected-length tip) and the spreading from the base into the fin root.

The result reports the energy balance (convected heat / applied heat) and, with ``refine=True``, the change of the
junction temperature rise when the grid is halved (a grid-convergence check).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


@dataclass
class FinCellResult:
    """One fin pitch solved by finite volumes.

    Attributes
    ----------
    t_junction_c : float [degC]
        Mean temperature of the heated bottom face.
    t_fin_root_c : float [degC]
    t_fin_tip_c : float [degC]
    energy_balance_rel_error : float [-]
        ``|convected - applied| / applied``.
    n_cells : int [-]
    grid_change_rel : float | None [-]
        Relative change of the rise on a grid twice as fine.
    """
    t_junction_c: float              # mean temperature of the heated bottom face
    t_fin_root_c: float              # mean temperature of the fin at its root (first fin layer)
    t_fin_tip_c: float               # temperature of the fin tip face
    energy_balance_rel_error: float  # |convected - applied| / applied
    n_cells: int
    grid_change_rel: float | None    # |rise(h) - rise(h/2)| / rise(h/2) when refine=True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _edges(length: float, target: float, max_cells: int) -> np.ndarray:
    n = int(np.clip(round(length / target), 2, max_cells))
    return np.linspace(0.0, length, n + 1)


def _solve(pitch, depth, base_t, fin_t, fin_h, k, h_conv, power_per_fin, t_amb, cells_across_fin, max_cells):
    target = min(fin_t / cells_across_fin, base_t / 6.0)
    side = (pitch - fin_t) / 2.0
    xe = np.concatenate([_edges(side, target, max_cells)[:-1], side + _edges(fin_t, target, max_cells)[:-1],
                         side + fin_t + _edges(side, target, max_cells)])
    ze = np.concatenate([_edges(base_t, target, max_cells)[:-1], base_t + _edges(fin_h, target, 4 * max_cells)])
    dx, dz = np.diff(xe), np.diff(ze)
    xc, zc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ze[1:] + ze[:-1])
    nx, nz = len(dx), len(dz)
    in_fin_x = (xc > side) & (xc < side + fin_t)
    in_base_z = zc < base_t
    solid = in_base_z[:, None] | in_fin_x[None, :]          # (nz, nx)
    idx = -np.ones((nz, nx), np.int64)
    idx[solid] = np.arange(int(solid.sum()))
    n = int(solid.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    rhs = np.zeros(n)
    g_conv_total = []                                       # (cell, conductance) on convective faces

    def robin(cell, d_normal, face_len):
        g = 1.0 / (0.5 * d_normal / (k * face_len) + 1.0 / (h_conv * face_len))
        diag[cell] += g                                     # the unknown is theta = T - T_amb, so no rhs term
        g_conv_total.append((cell, g))

    for j in range(nz):
        for i in range(nx):
            c = idx[j, i]
            if c < 0:
                continue
            # x neighbours (the planes x = 0 and x = pitch are symmetry planes: no term)
            if i + 1 < nx:
                if idx[j, i + 1] >= 0:
                    g = k * dz[j] / (0.5 * (dx[i] + dx[i + 1]))
                    a = idx[j, i + 1]
                    rows += [c, a]
                    cols += [a, c]
                    vals += [-g, -g]
                    diag[c] += g
                    diag[a] += g
                else:
                    robin(c, dx[i], dz[j])
            if i > 0 and idx[j, i - 1] < 0:
                robin(c, dx[i], dz[j])
            # z neighbours
            if j + 1 < nz:
                if idx[j + 1, i] >= 0:
                    g = k * dx[i] / (0.5 * (dz[j] + dz[j + 1]))
                    a = idx[j + 1, i]
                    rows += [c, a]
                    cols += [a, c]
                    vals += [-g, -g]
                    diag[c] += g
                    diag[a] += g
                else:
                    robin(c, dz[j], dx[i])                  # exposed base top
            else:
                robin(c, dz[j], dx[i])                      # fin tip
            if j == 0:
                rhs[c] += power_per_fin / (pitch * depth) * dx[i]   # W per metre of depth

    A = sps.csr_matrix((np.concatenate([vals, diag]), (np.concatenate([rows, np.arange(n)]),
                                                       np.concatenate([cols, np.arange(n)]))), shape=(n, n))
    theta = spla.spsolve(A.tocsc(), rhs)
    q_flux = power_per_fin / (pitch * depth)
    bottom = theta[idx[0]] + q_flux * dz[0] / (2.0 * k)    # surface value under a uniform flux
    t_junction = t_amb + float(np.sum(bottom * dx) / pitch)
    root_layer = int(np.searchsorted(zc, base_t))
    fin_cols = np.where(in_fin_x)[0]
    t_root = t_amb + float(np.sum(theta[idx[root_layer, fin_cols]] * dx[fin_cols]) / fin_t)
    tip_cells = idx[-1, fin_cols]
    g_tip = 1.0 / (0.5 * dz[-1] / (k * dx[fin_cols]) + 1.0 / (h_conv * dx[fin_cols]))
    tip_surface = theta[tip_cells] - g_tip * theta[tip_cells] * 0.5 * dz[-1] / (k * dx[fin_cols])
    t_tip = t_amb + float(np.sum(tip_surface * dx[fin_cols]) / fin_t)
    q_out = sum(g * theta[c] for c, g in g_conv_total)
    q_in = q_flux * pitch
    return t_junction, t_root, t_tip, abs(q_out - q_in) / q_in, n


def solve_fin_cell(*, n_fins: int, fin_thickness_m: float, fin_height_m: float, fin_spacing_m: float,
                   depth_m: float, base_thickness_m: float, k_w_mk: float, h_w_m2k: float, power_w: float,
                   t_ambient_c: float = 25.0, cells_across_fin: int = 6, max_cells: int = 160,
                   refine: bool = True) -> FinCellResult:
    """Steady conduction in one fin pitch (module docstring). ``power_w`` is the whole array's power."""
    if min(fin_thickness_m, fin_height_m, fin_spacing_m, depth_m, base_thickness_m, k_w_mk, h_w_m2k) <= 0:
        raise ValueError("every length, k and h must be positive")
    pitch = fin_spacing_m + fin_thickness_m
    args = (pitch, depth_m, base_thickness_m, fin_thickness_m, fin_height_m, k_w_mk, h_w_m2k,
            power_w / max(int(n_fins), 1), t_ambient_c)
    tj, tr, tt, eb, n = _solve(*args, cells_across_fin, max_cells)
    change = None
    if refine:
        tj2 = _solve(*args, 2 * cells_across_fin, 2 * max_cells)[0]
        change = abs((tj - t_ambient_c) - (tj2 - t_ambient_c)) / max(abs(tj2 - t_ambient_c), 1e-12)
    return FinCellResult(t_junction_c=tj, t_fin_root_c=tr, t_fin_tip_c=tt, energy_balance_rel_error=eb, n_cells=n,
                         grid_change_rel=change)
