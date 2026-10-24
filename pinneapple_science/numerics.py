"""Small numerical building blocks shared by the experiments: graded tensor grids, a finite-volume solver for
``div(eps grad phi) = -rho`` in 1-D, 2-D and 3-D with fixed-potential nodes and mirror (symmetry) faces, the
discrete field energy, and a scalar Helmholtz eigen-solver for waveguide cross-sections.

The finite-volume scheme is the standard two-point flux on a tensor grid of nodes: the flux through the face
between nodes ``i`` and ``i+1`` is ``eps_face (phi_{i+1} - phi_i) / (x_{i+1} - x_i)`` times the face area (the
product of the dual widths along the other axes); ``eps_face`` is the harmonic mean of the two node values. It is
second-order accurate on smooth graded grids and conservative (Patankar, *Numerical Heat Transfer and Fluid
Flow*, 1980, ch. 4). The discrete energy ``1/2 sum eps_face (dphi)^2 A / dx`` equals ``1/2 Q V`` of the
discrete system, so capacitance from the energy converges like the solution. ``mean="arithmetic"`` with
``eps = r`` on an ``(r, z)`` grid gives the axisymmetric operator ``(1/r) d/dr (r dphi/dr) + d2phi/dz2`` (each
radial face weighted by its radius).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["graded_axis", "TensorGrid", "solve_potential", "field_energy", "helmholtz_eigs", "spectral_peaks"]


def graded_axis(core: float, h: float, length: float, growth: float = 1.12, start: float = 0.0) -> np.ndarray:
    """Nodes from ``start``: spacing exactly ``h`` over at least ``core`` (rounded up to whole cells), then
    growing geometrically by ``growth`` until ``start + length``."""
    n = max(int(math.ceil(core / h - 1e-9)), 1)
    x = list(start + np.arange(n + 1) * h)
    dx = h
    while x[-1] < start + length - 1e-12:
        dx *= growth
        x.append(min(x[-1] + dx, start + length))
    return np.asarray(x)


@dataclass
class TensorGrid:
    """Node coordinates per axis (1, 2 or 3 axes)."""

    axes: tuple[np.ndarray, ...]

    def __post_init__(self):
        self.axes = tuple(np.asarray(a, float) for a in self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.axes)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def mesh(self) -> list[np.ndarray]:
        return np.meshgrid(*self.axes, indexing="ij")

    def dual_widths(self, axis: int) -> np.ndarray:
        """Width of each node's control volume along ``axis`` (half cells at the ends)."""
        x = self.axes[axis]
        d = np.diff(x)
        w = np.zeros_like(x)
        w[:-1] += d / 2
        w[1:] += d / 2
        return w


def _face_eps(eps: np.ndarray, axis: int, mean: str = "harmonic") -> np.ndarray:
    a = np.take(eps, np.arange(eps.shape[axis] - 1), axis=axis)
    b = np.take(eps, np.arange(1, eps.shape[axis]), axis=axis)
    if mean == "arithmetic":
        return 0.5 * (a + b)
    return 2 * a * b / np.where(a + b == 0, 1.0, a + b)


def _bcast(v: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = v.size
    return v.reshape(shape)


def _operator(grid: TensorGrid, eps: np.ndarray, mean: str = "harmonic"):
    """Sparse matrix of ``-div(eps grad)`` integrated over the control volumes (natural, zero-flux faces)."""
    shape = grid.shape
    N = int(np.prod(shape))
    idx = np.arange(N).reshape(shape)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for ax in range(grid.ndim):
        h = np.diff(grid.axes[ax])
        area = np.ones([1] * grid.ndim)
        for o in range(grid.ndim):
            if o != ax:
                area = area * _bcast(grid.dual_widths(o), o, grid.ndim)
        ef = _face_eps(eps, ax, mean)
        coef = ef * area / _bcast(h, ax, grid.ndim)
        i0 = np.take(idx, np.arange(shape[ax] - 1), axis=ax).ravel()
        i1 = np.take(idx, np.arange(1, shape[ax]), axis=ax).ravel()
        c = np.broadcast_to(coef, ef.shape).ravel()
        rows += [i0, i1]
        cols += [i1, i0]
        vals += [-c, -c]
        np.add.at(diag, i0, c)
        np.add.at(diag, i1, c)
    rows.append(np.arange(N))
    cols.append(np.arange(N))
    vals.append(diag)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))


def _volumes(grid: TensorGrid) -> np.ndarray:
    v = np.ones([1] * grid.ndim)
    for ax in range(grid.ndim):
        v = v * _bcast(grid.dual_widths(ax), ax, grid.ndim)
    return np.broadcast_to(v, grid.shape)


def solve_potential(grid: TensorGrid, fixed: np.ndarray, values, *, eps=None, rho=None, tol: float = 1e-10,
                    direct: bool | None = None, mean: str = "harmonic") -> np.ndarray:
    """Solve ``-div(eps grad phi) = rho`` with ``phi = values`` on the ``fixed`` nodes (boolean array; put the
    outer box there for a grounded enclosure). Faces without fixed nodes are zero-flux, which is the mirror
    condition of a symmetry plane. ``eps`` (relative or absolute, per node) defaults to 1, ``rho`` to 0.
    ``direct`` (default: 1-D and 2-D) picks a sparse LU solve, else conjugate gradients with a Jacobi
    preconditioner (3-D, where LU fill-in is slower than CG)."""
    shape = grid.shape
    eps = np.ones(shape) if eps is None else np.broadcast_to(np.asarray(eps, float), shape)
    fixed = np.asarray(fixed, bool)
    vals = np.broadcast_to(np.asarray(values, float), shape)
    A = _operator(grid, eps, mean)
    b = np.zeros(A.shape[0]) if rho is None else (np.broadcast_to(rho, shape) * _volumes(grid)).ravel()
    phi = np.where(fixed, vals, 0.0).ravel()
    free = ~fixed.ravel()
    Aff = A[free][:, free]
    rhs = b[free] - A[free][:, ~free] @ phi[~free]
    if direct is None:
        direct = grid.ndim < 3 or Aff.shape[0] < 5_000
    if direct:
        x = spla.spsolve(Aff.tocsc(), rhs)
    else:
        M = sp.diags(1.0 / Aff.diagonal())
        x, info = spla.cg(Aff, rhs, rtol=tol, maxiter=20000, M=M)
        if info != 0:
            raise RuntimeError(f"conjugate gradients did not converge (info={info})")
    phi[free] = x
    return phi.reshape(shape)


def field_energy(grid: TensorGrid, phi: np.ndarray, eps=None, mean: str = "harmonic") -> float:
    """Discrete ``1/2 sum eps (grad phi)^2 dV`` (per unit length in 2-D), with the face fluxes of the scheme."""
    eps = np.ones(grid.shape) if eps is None else np.broadcast_to(np.asarray(eps, float), grid.shape)
    W = 0.0
    for ax in range(grid.ndim):
        h = _bcast(np.diff(grid.axes[ax]), ax, grid.ndim)
        area = np.ones([1] * grid.ndim)
        for o in range(grid.ndim):
            if o != ax:
                area = area * _bcast(grid.dual_widths(o), o, grid.ndim)
        d = np.diff(phi, axis=ax)
        W += 0.5 * float(np.sum(_face_eps(eps, ax, mean) * d * d * area / h))
    return W


def helmholtz_eigs(nx: int, ny: int, a: float, b: float, kind: str, n: int = 6,
                   inside: np.ndarray | None = None, modes: bool = False):
    """The ``n`` smallest cut-off wavenumbers ``k_c`` of a waveguide cross-section from the scalar Helmholtz
    eigenproblem ``-lap psi = k_c^2 psi`` on an ``nx`` x ``ny`` cell-centred grid of the box ``a`` x ``b``:
    ``kind="TM"`` (``psi = E_z``, Dirichlet on the wall) or ``"TE"`` (``psi = H_z``, Neumann). ``inside``
    (boolean ``(nx, ny)``) restricts the guide to a sub-region of the box (staircase wall). ``modes=True`` also
    returns the mode shapes ``(n, nx, ny)`` (zero outside). Pozar, *Microwave Engineering*, 4th ed., sec. 3.3."""
    dx, dy = a / nx, b / ny
    inside = np.ones((nx, ny), bool) if inside is None else np.asarray(inside, bool)
    idx = -np.ones((nx, ny), int)
    idx[inside] = np.arange(int(inside.sum()))
    N = int(inside.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for (di, dj, h) in ((1, 0, dx), (-1, 0, dx), (0, 1, dy), (0, -1, dy)):
        for i, j in zip(*np.nonzero(inside), strict=True):
            k = idx[i, j]
            ii, jj = i + di, j + dj
            nb = 0 <= ii < nx and 0 <= jj < ny and inside[ii, jj]
            if nb:
                rows.append(k)
                cols.append(idx[ii, jj])
                vals.append(-1 / h ** 2)
                diag[k] += 1 / h ** 2
            elif kind == "TM":          # wall half a cell away: ghost = -psi
                diag[k] += 2 / h ** 2
            elif kind != "TE":
                raise ValueError("kind is 'TM' or 'TE'")
    A = sp.csr_matrix((vals + list(diag), (rows + list(range(N)), cols + list(range(N)))), shape=(N, N))
    k = n + (1 if kind == "TE" else 0)
    w, v = spla.eigsh(A, k=k, sigma=-1e-6 if kind == "TE" else 0.0, which="LM")
    order = np.argsort(np.abs(w))
    w, v = np.abs(w[order]), v[:, order]
    if kind == "TE":
        w, v = w[1:], v[:, 1:]       # the constant mode (k_c = 0) is not a waveguide mode
    kc = np.sqrt(w[:n])
    if not modes:
        return kc
    shapes = np.zeros((n, nx, ny))
    for q in range(n):
        shapes[q][inside] = v[:, q]
    return kc, shapes


def spectral_peaks(signal, dt: float, f_max: float, *, n_peaks: int = 8, min_rel: float = 1e-3,
                   pad: int = 8) -> np.ndarray:
    """Frequencies of the strongest local maxima of the spectrum of ``signal`` below ``f_max`` (Hann window,
    ``pad``-fold zero padding, parabolic interpolation of the log spectrum), ascending. Peaks below ``min_rel``
    times the largest are dropped."""
    v = np.asarray(signal, float)
    nfft = pad * len(v)
    fr = np.fft.rfftfreq(nfft, dt)
    sp_ = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)), nfft))
    sel = (fr > 0) & (fr < f_max)
    k = np.nonzero(sel[1:-1] & (sp_[1:-1] > sp_[:-2]) & (sp_[1:-1] >= sp_[2:]))[0] + 1
    k = k[sp_[k] >= min_rel * sp_[k].max()] if k.size else k
    k = k[np.argsort(sp_[k])[::-1][:n_peaks]]
    out = []
    for i in k:
        y0, y1, y2 = np.log(sp_[i - 1:i + 2])
        out.append((i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * (fr[1] - fr[0]))
    return np.sort(np.asarray(out))


def refine_ladder(base: int, levels: int, ratio: int = 2) -> Sequence[int]:
    return [base * ratio ** i for i in range(levels)]
