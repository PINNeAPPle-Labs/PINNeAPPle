"""Grids for the underground-storage simulator: a structured cartesian grid and a 1-D radial one.

Both grids hand the simulator the same three things, and nothing else:

* ``volumes`` -- the bulk volume of every cell [m3];
* ``phi0`` -- the porosity at the reference pressure [-];
* ``connections`` -- ``(a, b, T)`` for every pair of neighbouring cells, with the transmissibility
  ``T = A k_h / d`` [m3] (the harmonic mean of the two permeabilities, over the centre-to-centre distance,
  times the face area). A radial grid uses the exact steady radial transmissibility
  ``T = 2 pi k h / ln(r2/r1)``, which is what makes a logarithmically spaced radial grid the right tool for a
  line-source verification;
* ``well_index(cell, r_w, skin)`` -- Peaceman's well index [m3] for a well completed in one cell.

Boundaries are closed (no flow) unless a pressure is set on a face with :meth:`Grid.set_boundary_pressure`,
which adds ``T (p_boundary - p_cell)`` to the mass balance of the cells named -- the constant-pressure outer
boundary an aquifer is usually lumped into.

Heterogeneity comes from :func:`gaussian_field`, a log-normal permeability with a correlation length (a
FFT-filtered white field), the cheap stand-in for an SPE10-class realisation: same statistics, seconds instead
of a download.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Grid", "CartesianGrid", "RadialGrid", "gaussian_field"]


def _field(value, shape: tuple[int, int, int]) -> np.ndarray:
    """Broadcast a scalar, a 1-D, a 2-D or a 3-D array of cell values to the grid's three-dimensional shape."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 3:
        raise ValueError(f"a cell field cannot have {arr.ndim} dimensions")
    padded = arr.reshape(arr.shape + (1,) * (3 - arr.ndim)) if arr.ndim else arr
    try:
        return np.broadcast_to(padded, shape)
    except ValueError as exc:
        raise ValueError(f"a field of shape {arr.shape} does not fit a grid of shape {shape}") from exc


def gaussian_field(shape, spacing, corr_length: float, sigma: float = 1.0, seed: int = 0):
    """Log-normal field [dimensionless multiplier] with a gaussian covariance of length ``corr_length``.

    White noise filtered by a gaussian kernel in Fourier space, then exponentiated: ``exp(sigma g - sigma^2/2)``,
    whose mean is 1 whatever ``sigma`` is. ``spacing`` is the cell size along each axis, so ``corr_length`` is
    in metres. One cell of the field is one cell of the grid it is laid on.
    """
    shape = tuple(int(s) for s in np.atleast_1d(shape))
    spacing = tuple(float(s) for s in np.atleast_1d(spacing))
    if len(spacing) == 1:
        spacing = spacing * len(shape)
    rng = np.random.default_rng(seed)
    white = rng.normal(size=shape)
    freq = np.meshgrid(*[np.fft.fftfreq(n, d=d) for n, d in zip(shape, spacing, strict=True)], indexing="ij")
    k2 = sum(f * f for f in freq)
    filt = np.exp(-k2 * (np.pi * corr_length) ** 2)
    g = np.real(np.fft.ifftn(np.fft.fftn(white) * filt))
    g = g / max(np.std(g), 1e-30)
    return np.exp(sigma * g - 0.5 * sigma * sigma)


class Grid:
    """What the simulator is allowed to assume about a grid."""

    n_cells: int
    volumes: np.ndarray
    phi0: np.ndarray
    conn_a: np.ndarray
    conn_b: np.ndarray
    conn_T: np.ndarray
    boundary_cells: np.ndarray
    boundary_T: np.ndarray
    boundary_p: np.ndarray

    def __init__(self) -> None:
        self.boundary_cells = np.zeros(0, dtype=int)
        self.boundary_T = np.zeros(0)
        self.boundary_p = np.zeros(0)

    # ------------------------------------------------------------------ boundaries
    def set_boundary_pressure(self, cells, transmissibility: float, pressure: float) -> None:
        """Hold the cells named against a fixed pressure through ``transmissibility`` [m3]."""
        cells = np.atleast_1d(np.asarray(cells, dtype=int))
        self.boundary_cells = cells
        self.boundary_T = np.full(cells.shape, float(transmissibility))
        self.boundary_p = np.full(cells.shape, float(pressure))

    def pore_volume(self) -> float:
        """Total pore volume [m3] at ``phi0``."""
        return float(np.sum(self.volumes * self.phi0))

    def well_index(self, cell: int, r_w: float, skin: float = 0.0) -> float:
        raise NotImplementedError


class CartesianGrid(Grid):
    """A structured cell-centred grid in one, two or three dimensions.

    ``shape`` and ``spacing`` are given with one, two or three entries; a one-dimensional grid is
    ``(n,), (dx,)`` and its section is ``dy = dz = 1 m``, since only the product of the two matters for a
    1-D problem. ``permeability`` is a scalar, a field of ``shape`` (isotropic) or a tuple of three fields
    ``(kx, ky, kz)``; ``porosity`` a scalar or a field of ``shape``.
    """

    def __init__(self, shape, spacing, *, permeability=1e-14, porosity=0.2):
        super().__init__()
        shape = tuple(int(s) for s in np.atleast_1d(shape))
        spacing = tuple(float(s) for s in np.atleast_1d(spacing))
        if len(spacing) == 1:
            spacing = spacing * len(shape)
        if len(shape) != len(spacing):
            raise ValueError(f"shape {shape} and spacing {spacing} must have the same length")
        if len(shape) > 3:
            raise ValueError("at most three dimensions")
        self.shape = shape + (1,) * (3 - len(shape))
        self.spacing = spacing + (1.0,) * (3 - len(spacing))
        self.dx = [np.full(self.shape[0], self.spacing[0]),
                   np.full(self.shape[1], self.spacing[1]),
                   np.full(self.shape[2], self.spacing[2])]
        self.dims = len(shape)
        self.n_cells = int(np.prod(self.shape))

        perm = permeability if isinstance(permeability, (tuple, list)) else (permeability,) * 3
        if len(perm) != 3:
            raise ValueError("permeability must be a scalar, a field, or a (kx, ky, kz) tuple of three")
        # kx, ky, kz in that order, whatever the number of dimensions the grid has
        self.k = [np.ascontiguousarray(_field(p, self.shape)) for p in perm]
        self.phi0 = np.ascontiguousarray(_field(porosity, self.shape)).ravel()

        outer = np.outer
        self.volumes = np.ascontiguousarray(
            outer(outer(self.dx[0], self.dx[1]), self.dx[2])).ravel()
        self._build_connections()

    def _build_connections(self) -> None:
        idx = np.arange(self.n_cells, dtype=np.int64).reshape(self.shape)
        a_list, b_list, t_list = [], [], []
        areas = {
            0: np.broadcast_to((self.dx[1][:, None] * self.dx[2][None, :])[None, :, :], self.shape),
            1: np.broadcast_to((self.dx[0][:, None] * self.dx[2][None, :])[:, None, :], self.shape),
            2: np.broadcast_to((self.dx[0][:, None] * self.dx[1][None, :])[:, :, None], self.shape),
        }
        for axis in range(3):
            n = self.shape[axis]
            if n < 2:
                continue
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_a[axis] = slice(0, n - 1)
            sl_b[axis] = slice(1, n)
            sl_a, sl_b = tuple(sl_a), tuple(sl_b)
            k1 = self.k[axis][sl_a]
            k2 = self.k[axis][sl_b]
            k_h = 2.0 * k1 * k2 / (k1 + k2)
            dist = 0.5 * (self.dx[axis][:-1] + self.dx[axis][1:])
            shape = list(self.shape)
            shape[axis] -= 1
            dist = np.broadcast_to(dist.reshape([-1 if i == axis else 1 for i in range(3)]), shape)
            area = areas[axis][sl_a]
            a_list.append(idx[sl_a].ravel())
            b_list.append(idx[sl_b].ravel())
            t_list.append((area * k_h / dist).ravel())
        self.conn_a = np.concatenate(a_list)
        self.conn_b = np.concatenate(b_list)
        self.conn_T = np.concatenate(t_list)

    def well_index(self, cell: int, r_w: float, skin: float = 0.0) -> float:
        """Peaceman's well index [m3].

        One-dimensional (``shape = (n,)``): ``r_eq = 0.2 dx``, the radius at which the cartesian and the radial
        solutions agree (Peaceman 1978). Two and three dimensions::

            r_eq = 0.28 [ sqrt(ky/kx) dx^2 + sqrt(kx/ky) dy^2 ]^0.5 / [ (ky/kx)^0.25 + (kx/ky)^0.25 ]
            WI   = 2 pi sqrt(kx ky) h / (ln(r_eq/r_w) + s)

        with ``h`` the completed thickness (the cell's own ``dz``), Peaceman (1983), SPEJ 23:531.
        """
        i, j, k = np.unravel_index(int(cell), self.shape)
        dx, dy, dz = self.dx[0][i], self.dx[1][j], self.dx[2][k]
        kx, ky = self.k[0][i, j, k], self.k[1][i, j, k]
        if self.dims == 1:
            r_eq = 0.2 * dx
        else:
            ratio = ky / kx
            r_eq = 0.28 * np.sqrt(np.sqrt(ratio) * dx * dx + np.sqrt(1.0 / ratio) * dy * dy) / (
                ratio ** 0.25 + ratio ** -0.25)
        if r_eq <= r_w:
            raise ValueError(f"well radius {r_w:g} m does not fit inside the equivalent radius {r_eq:g} m")
        return float(2.0 * np.pi * np.sqrt(kx * ky) * dz / (np.log(r_eq / r_w) + skin))


class RadialGrid(Grid):
    """A one-dimensional radial grid: ``n_r`` annular cells from the well radius to ``r_out``.

    Faces are geometric from ``r_w`` to ``r_out`` when ``geometric`` (the usual choice for a well test: the
    cells are small where the gradient is) and uniform otherwise. The transmissibility between two annuli is the
    exact steady radial one, ``2 pi k h / ln(r2/r1)``, so the only spatial discretisation left is the storage
    term and the well model.
    """

    def __init__(self, n_r: int = 40, *, r_w: float = 0.1, r_out: float = 1.0e4, height: float = 10.0,
                 permeability=1e-13, porosity=0.2, geometric: bool = True):
        super().__init__()
        if r_out <= r_w:
            raise ValueError("r_out must be larger than r_w")
        self.n_r = int(n_r)
        self.r_w = float(r_w)
        self.r_out = float(r_out)
        self.height = float(height)
        self.shape = (self.n_r, 1, 1)
        self.dims = 1
        self.n_cells = self.n_r
        if geometric:
            self.faces = np.geomspace(r_w, r_out, self.n_r + 1)
        else:
            self.faces = np.linspace(r_w, r_out, self.n_r + 1)
        r1, r2 = self.faces[:-1], self.faces[1:]
        self.centre = np.sqrt(0.5 * (r1 * r1 + r2 * r2))
        self.volumes = np.pi * (r2 * r2 - r1 * r1) * self.height
        k = np.broadcast_to(np.asarray(permeability, dtype=float), (self.n_r,)).copy()
        self.k = [k.reshape(-1, 1, 1)] * 3
        self.phi0 = np.broadcast_to(np.asarray(porosity, dtype=float), (self.n_r,)).astype(float).copy()
        self.conn_a = np.arange(self.n_r - 1)
        self.conn_b = np.arange(1, self.n_r)
        k1, k2 = k[:-1], k[1:]
        k_h = 2.0 * k1 * k2 / (k1 + k2)
        # ln(r_c2/r_c1) for cell-centred unknowns: the two half-cell resistances ln(r_face/r_c1) +
        # ln(r_c2/r_face) add up to it exactly
        self.conn_T = 2.0 * np.pi * k_h * self.height / np.log(self.centre[1:] / self.centre[:-1])

    def well_index(self, cell: int, r_w: float, skin: float = 0.0) -> float:
        """``2 pi k h / (ln(r_eq/r_w) + s)`` with ``r_eq`` the volume-equivalent radius of the first cell."""
        if int(cell) != 0:
            raise ValueError("the radial grid holds one well, in the first cell")
        k = float(self.k[0][0, 0, 0])
        return float(2.0 * np.pi * k * self.height / (np.log(self.centre[0] / r_w) + skin))
