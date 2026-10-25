"""Cell-centred finite volumes for steady and transient conduction on a voxel box, and the reference solver.

``div(k grad T) + q = 0`` on a box of ``(nx, ny, nz)`` cells (uniform ``dx`` laterally, ``dz`` per z row), with

* the harmonic-mean face conductance between neighbours, plus a contact resistance ``R`` (m^2 K / W) on chosen
  horizontal faces: ``G = A / (dz_1 / (2 k_1) + R + dz_2 / (2 k_2))``;
* Robin cooling on the top and bottom faces to a zero ambient: ``G = A / (dz / (2 k) + 1 / h)``;
* adiabatic lateral faces, or *ports*: one node per lateral boundary face, joined to its cell by the half-cell
  conductance. Two boxes sharing a port node are, after eliminating the node, exactly the standard discretisation of
  the joined box; that is what makes block condensation exact (:mod:`.substructure`).

Unknown order: cells in C order ``(a, b, k)``, then ports ``x-, x+`` (each ``(b, k)``), ``y-, y+`` (each ``(a, k)``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _kxyz(k):
    if isinstance(k, (tuple, list)):
        return tuple(np.asarray(a, float) for a in k)
    k = np.asarray(k, float)
    return k, k, k


def _harm(d1, k1, d2, k2, extra=0.0):
    return 1.0 / (d1 / (2 * k1) + extra + d2 / (2 * k2))


@dataclass
class System:
    """An assembled conduction operator: ``K`` (W/K), the cell count, the port blocks, the Robin conductances of the
    top and bottom cell rows (``(nx, ny)``) and the vertical face conductances ``Gz`` (``(nx, ny, nz - 1)``)."""
    K: sp.csr_matrix
    shape: tuple[int, int, int]
    ports: dict[str, slice] = field(default_factory=dict)
    g_top: np.ndarray | None = None
    g_bottom: np.ndarray | None = None
    Gz: np.ndarray | None = None
    C: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz


def assemble(k, dz, R, dx: float, h_top: float, h_bottom: float, *, ports: bool = False, rho_c=None,
             dy: float | None = None) -> System:
    """Assemble the conduction operator of a voxel box (see the module docstring). ``k``: ``(nx, ny, nz)`` array or
    a ``(kx, ky, kz)`` tuple; ``R``: contact resistance of each horizontal face, ``(nz - 1,)`` or
    ``(nx, ny, nz - 1)``; ``rho_c``: volumetric heat capacity, returned as the diagonal ``C`` (J/K) per cell."""
    kx, ky, kz = _kxyz(k)
    dy = dx if dy is None else dy
    nx, ny, nz = kx.shape
    dz = np.asarray(dz, float)
    R = np.broadcast_to(np.asarray(R, float), (nx, ny, nz - 1))
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * ny * nz + (2 * (ny + nx) * nz if ports else 0))

    def couple(i, j, g):
        i, j, g = i.ravel(), j.ravel(), g.ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((-g, -g))
        np.add.at(diag, i, g)
        np.add.at(diag, j, g)

    dzb = dz[None, None, :]
    couple(idx[:-1], idx[1:], dy * dzb * _harm(dx, kx[:-1], dx, kx[1:]))
    couple(idx[:, :-1], idx[:, 1:], dx * dzb * _harm(dy, ky[:, :-1], dy, ky[:, 1:]))
    Gz = dx * dy * _harm(dz[:-1][None, None], kz[..., :-1], dz[1:][None, None], kz[..., 1:], R)
    couple(idx[..., :-1], idx[..., 1:], Gz)
    g_top = dx * dy / (dz[-1] / (2 * kz[..., -1]) + 1.0 / h_top) if h_top > 0 else np.zeros((nx, ny))
    g_bot = dx * dy / (dz[0] / (2 * kz[..., 0]) + 1.0 / h_bottom) if h_bottom > 0 else np.zeros((nx, ny))
    np.add.at(diag, idx[..., -1].ravel(), g_top.ravel())
    np.add.at(diag, idx[..., 0].ravel(), g_bot.ravel())
    port_sl = {}
    if ports:
        off = nx * ny * nz
        for name, cells, g in (
                ("x-", idx[0], dy * dzb[0] * 2 * kx[0] / dx), ("x+", idx[-1], dy * dzb[0] * 2 * kx[-1] / dx),
                ("y-", idx[:, 0], dx * dzb[0] * 2 * ky[:, 0] / dy), ("y+", idx[:, -1], dx * dzb[0] * 2 * ky[:, -1] / dy)):
            m = cells.size
            couple(cells, off + np.arange(m).reshape(cells.shape), g)
            port_sl[name] = slice(off, off + m)
            off += m
    nn = diag.size
    rows.append(np.arange(nn))
    cols.append(np.arange(nn))
    vals.append(diag)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nn, nn))
    C = None
    if rho_c is not None:
        C = (np.asarray(rho_c, float) * dx * dy * dzb).ravel()
    return System(K, (nx, ny, nz), port_sl, g_top, g_bot, Gz, C)


def assemble_package(pkg, *, transient: bool = False) -> System:
    """The whole package (:class:`.package.Package`) as one box, adiabatic on the sides: the reference model."""
    st = pkg.stack
    p = pkg.global_properties()
    return assemble((p["k_lat"], p["k_lat"], p["k"]), p["dz"], p["R"], st.dx, st.h_top, st.h_bottom,
                    rho_c=p["rho_c"] if transient else None)


def source_vector(pkg, power: np.ndarray | None = None) -> np.ndarray:
    """Heat source of every cell (W): the power map on the active cell row."""
    nx, ny = pkg.shape[0] * pkg.stack.n, pkg.shape[1] * pkg.stack.n
    f = np.zeros((nx, ny, pkg.stack.nz))
    f[:, :, pkg.stack.active_k] = pkg.power if power is None else power
    return f.ravel()


def factor(K: sp.spmatrix):
    """Sparse LU of a symmetric matrix (minimum degree on ``K + K^T``, symmetric pivoting)."""
    return spla.splu(sp.csc_matrix(K), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})


@dataclass
class SolveInfo:
    method: str
    seconds: float
    iterations: int = 0
    residual: float = 0.0


def solve(K: sp.spmatrix, f: np.ndarray, *, method: str = "auto", rtol: float = 1e-11,
          maxiter: int = 4000) -> tuple[np.ndarray, SolveInfo]:
    """Solve ``K T = f``. ``auto``: sparse direct (SuperLU, minimum-degree ordering on ``K + K^T``) up to 6e4
    unknowns, else conjugate gradients with an
    algebraic multigrid preconditioner (``pyamg``, if installed; Jacobi otherwise) to relative residual ``rtol``."""
    t0 = time.perf_counter()
    n = K.shape[0]
    if method == "auto":
        method = "direct" if n <= 60_000 else "amg"
    if method == "direct":
        T = factor(K).solve(f)
        res = np.linalg.norm(K @ T - f) / max(np.linalg.norm(f), 1e-300)
        return T, SolveInfo("direct", time.perf_counter() - t0, 0, float(res))
    M = None
    if method == "amg":
        try:
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(K.tocsr(), symmetry="symmetric", max_coarse=2000)
            M = ml.aspreconditioner(cycle="V")
        except ImportError:
            method = "cg-jacobi"
    if M is None:
        d = K.diagonal()
        M = spla.LinearOperator(K.shape, matvec=lambda x: x / d)
    it = [0]

    def cb(_):
        it[0] += 1
    T, code = spla.cg(K, f, rtol=rtol, maxiter=maxiter, M=M, callback=cb)
    res = np.linalg.norm(K @ T - f) / max(np.linalg.norm(f), 1e-300)
    if code != 0 and res > 1e3 * rtol:
        raise RuntimeError(f"CG did not converge: residual {res:.2e} after {it[0]} iterations")
    return T, SolveInfo(method, time.perf_counter() - t0, it[0], float(res))


def solve_package(pkg, power: np.ndarray | None = None, **kw) -> tuple[np.ndarray, SolveInfo]:
    """Reference steady temperature rise of every cell, shape ``(nx, ny, nz)``."""
    sysm = assemble_package(pkg)
    T, info = solve(sysm.K, source_vector(pkg, power), **kw)
    return T.reshape(sysm.shape), info
