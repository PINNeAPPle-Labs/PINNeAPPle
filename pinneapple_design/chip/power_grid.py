"""On-chip power grid: IR drop on a resistive mesh fed by C4 bumps, and electromigration of every strap segment
against Black's equation at the segment's temperature.

The grid is the top-metal mesh of straps at ``pitch`` in x and y (one node per crossing); a segment between two
nodes has ``R = rho(T) pitch / (width thickness)``. Vdd bumps at ``bump_pitch`` connect nodes to an ideal supply
through ``r_bump``. The load current of a node is the power of the area it serves divided by Vdd.

Electromigration (Black, IEEE Trans Electron Devices 16:338, 1969): ``MTTF = A J^-n exp(Ea / k T)``. The constant is
fixed by a reference condition (``j_ref`` at ``t_ref_c`` lasts ``mttf_ref_years``), a design-rule style statement
of the technology's EM limit; the defaults (1 MA/cm^2 at 105 C for 10 years, n = 2, Ea = 0.9 eV for copper) are
typical published values, not a foundry's rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

__all__ = ["GridSpec", "solve_grid", "mesh_drop", "uniform_grid_closed_form", "black_mttf_years",
           "POISSON_SQUARE_MAX"]

K_B_EV = 8.617333262e-5
#: max of u for -lap u = 1 on the unit square with u = 0 on the boundary (series solution)
POISSON_SQUARE_MAX = 0.0736713532


@dataclass(frozen=True)
class GridSpec:
    vdd: float = 0.8                   # V
    pitch_m: float = 100e-6            # strap pitch (node spacing)
    width_m: float = 4e-6              # strap width
    thickness_m: float = 2e-6          # top metal thickness
    rho_ohm_m: float = 2.2e-8          # copper at 20 C (thin-film value; bulk is 1.7e-8)
    alpha_per_k: float = 0.0039        # temperature coefficient of resistivity
    bump_pitch_m: float = 400e-6       # Vdd bumps
    r_bump_ohm: float = 0.02
    j_ref_a_m2: float = 1e10           # 1 MA/cm^2
    t_ref_c: float = 105.0
    mttf_ref_years: float = 10.0
    black_n: float = 2.0
    ea_ev: float = 0.9

    def metal_fraction(self, width_m: float | None = None) -> float:
        """Area fraction of the top metal used by the x and y straps."""
        f = (self.width_m if width_m is None else width_m) / self.pitch_m
        return 2 * f - f * f


def black_mttf_years(j_a_m2, t_c, spec: GridSpec) -> np.ndarray:
    """MTTF of a segment at current density ``j`` and temperature ``t_c`` from the reference condition."""
    j = np.maximum(np.abs(np.asarray(j_a_m2, float)), 1e-30)
    T, Tr = np.asarray(t_c, float) + 273.15, spec.t_ref_c + 273.15
    return spec.mttf_ref_years * (spec.j_ref_a_m2 / j) ** spec.black_n * np.exp(spec.ea_ev / K_B_EV * (1 / T - 1 / Tr))


def _laplacian(nx: int, ny: int, gx: np.ndarray, gy: np.ndarray) -> sps.csr_matrix:
    """Weighted graph Laplacian of an (nx, ny) node mesh; ``gx`` (nx-1, ny), ``gy`` (nx, ny-1) segment conductances."""
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * ny)
    for a, b, g in ((idx[:-1], idx[1:], gx), (idx[:, :-1], idx[:, 1:], gy)):
        a, b, g = a.ravel(), b.ravel(), g.ravel()
        rows += [a, b]
        cols += [b, a]
        vals += [-g, -g]
        np.add.at(diag, a, g)
        np.add.at(diag, b, g)
    rows.append(np.arange(nx * ny))
    cols.append(np.arange(nx * ny))
    vals.append(diag)
    return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nx * ny, nx * ny))


def mesh_drop(load_a: np.ndarray, gx: np.ndarray, gy: np.ndarray, *, pads: np.ndarray | None = None,
              g_pad: float = np.inf, fixed: np.ndarray | None = None) -> np.ndarray:
    """IR drop (Vdd - V) of every node for node load currents ``load_a``: ``L d + G_pad d = I``, with ``fixed``
    nodes held at zero drop (ideal supply) and ``pads`` connected to it through ``g_pad``."""
    nx, ny = load_a.shape
    L = _laplacian(nx, ny, gx, gy).tolil()
    b = load_a.ravel().astype(float).copy()
    if pads is not None and np.isfinite(g_pad):
        p = np.flatnonzero(pads.ravel())
        L[p, p] = L[p, p].toarray().ravel() + g_pad
    fx = np.zeros(nx * ny, bool)
    if fixed is not None:
        fx |= fixed.ravel()
    if pads is not None and not np.isfinite(g_pad):
        fx |= pads.ravel()
    L = L.tocsr()
    free = np.flatnonzero(~fx)
    d = np.zeros(nx * ny)
    d[free] = spla.spsolve(L[free][:, free].tocsc(), b[free])
    return d.reshape(nx, ny)


def uniform_grid_closed_form(n: int, g: float, i_node: float) -> np.ndarray:
    """Exact drop of an (n x n) interior mesh with every node sinking ``i_node`` and the boundary ring at zero drop:
    the discrete sine expansion of the five-point Laplacian (eigenvalues ``4 - 2 cos(p pi/(n+1)) - 2 cos(q pi/(n+1))``).
    """
    from scipy.fft import dstn, idstn
    k = np.arange(1, n + 1) * np.pi / (n + 1)
    lam = 4 - 2 * np.cos(k)[:, None] - 2 * np.cos(k)[None, :]
    f = np.full((n, n), i_node / g)
    return idstn(dstn(f, type=1) / lam, type=1)


def solve_grid(power_map_w: np.ndarray, die_w_m: float, die_h_m: float, spec: GridSpec | None = None, *,
               width_m: float | None = None, temperature_c: np.ndarray | None = None) -> dict[str, Any]:
    """IR drop and electromigration for a power map (W per cell of any (mx, my) grid over the die), with the strap
    resistivity and Black's equation at ``temperature_c`` (same grid as the power map, or a scalar)."""
    spec = spec or GridSpec()
    w = spec.width_m if width_m is None else float(width_m)
    nx, ny = max(2, int(round(die_w_m / spec.pitch_m))), max(2, int(round(die_h_m / spec.pitch_m)))
    P = np.asarray(power_map_w, float)

    def to_nodes(a, reduce):
        """Resample a (mx, my) cell map onto the (nx, ny) nodes (nearest cell)."""
        a = np.asarray(a, float)
        if a.ndim == 0:
            return np.full((nx, ny), float(a))
        ix = np.minimum((np.arange(nx) + 0.5) / nx * a.shape[0], a.shape[0] - 1).astype(int)
        iy = np.minimum((np.arange(ny) + 0.5) / ny * a.shape[1], a.shape[1] - 1).astype(int)
        return a[np.ix_(ix, iy)] * (reduce(a.shape) if reduce else 1.0)
    load = to_nodes(P, lambda s: (s[0] * s[1]) / (nx * ny)) / spec.vdd       # same total current
    Tn = to_nodes(25.0 if temperature_c is None else temperature_c, None)
    rho_x = spec.rho_ohm_m * (1 + spec.alpha_per_k * (0.5 * (Tn[:-1] + Tn[1:]) - 20.0))
    rho_y = spec.rho_ohm_m * (1 + spec.alpha_per_k * (0.5 * (Tn[:, :-1] + Tn[:, 1:]) - 20.0))
    gx = w * spec.thickness_m / (rho_x * spec.pitch_m)
    gy = w * spec.thickness_m / (rho_y * spec.pitch_m)
    step = max(1, int(round(spec.bump_pitch_m / spec.pitch_m)))
    pads = np.zeros((nx, ny), bool)
    pads[step // 2::step, step // 2::step] = True
    d = mesh_drop(load, gx, gy, pads=pads, g_pad=1.0 / spec.r_bump_ohm)
    ix_ = gx * (d[:-1] - d[1:])
    iy_ = gy * (d[:, :-1] - d[:, 1:])
    area = w * spec.thickness_m
    jx, jy = np.abs(ix_) / area, np.abs(iy_) / area
    tx, ty = 0.5 * (Tn[:-1] + Tn[1:]), 0.5 * (Tn[:, :-1] + Tn[:, 1:])
    mx, my = black_mttf_years(jx, tx, spec), black_mttf_years(jy, ty, spec)
    worst = float(min(mx.min(), my.min()))
    i_tot = float(load.sum())
    bump = float(pads.sum())
    # balance: the bumps deliver exactly the load current
    i_bumps = float((d[pads] / spec.r_bump_ohm).sum())
    return {"drop": d, "max_drop_v": float(d.max()), "max_drop_fraction": float(d.max() / spec.vdd),
            "mean_drop_v": float(d.mean()), "j_max_a_m2": float(max(jx.max(), jy.max())),
            "mttf_min_years": worst, "current_a": i_tot, "bump_current_a": i_bumps, "n_bumps": int(bump),
            "max_bump_current_a": float((d[pads] / spec.r_bump_ohm).max()), "nodes": [nx, ny], "width_m": w,
            "metal_fraction": spec.metal_fraction(w), "current_balance": abs(i_bumps - i_tot) / max(i_tot, 1e-300)}
