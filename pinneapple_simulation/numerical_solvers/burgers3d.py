"""Viscous Burgers in three dimensions, ``u_t + u.grad(u) = nu lap(u)``, on a periodic box, by a pseudo-spectral
(2/3-dealiased) space discretisation with the ETDRK4 scheme of :mod:`.etdrk4` in time.

The field is the VECTOR ``u = (u_x, u_y, u_z)`` of shape ``(3, nx, ny, nz)``, one value per cell centre of
``[0, Lx] x [0, Ly] x [0, Lz]`` (``x_i = i Lx/nx``, endpoint excluded, so the grid is genuinely periodic). That is
one convention, and the only one this module accepts: an initial condition returning a scalar field is refused by
name, because a scalar ``u`` would be a different equation. The one-dimensional reduction is the special case whose
two last components are zero and whose first varies only along ``x``.

Why this scheme
---------------
The linear part of the vector equation is diagonal in Fourier space, ``L = -nu (kx^2 + ky^2 + kz^2)``, identical for
the three components, and the remaining term ``N(u) = -u.grad(u)`` is evaluated pseudo-spectrally (product in
physical space, derivatives as ``i k`` multipliers) and multiplied by a 2/3 dealiasing mask, one cutoff per axis.
ETDRK4 integrates ``L`` exactly and ``N`` to fourth order (Cox & Matthews 2002; the phi-functions by contour
integral, Kassam & Trefethen 2005), so:

- there is no time-step limit from diffusion at all -- the usual ``dt < dx^2/(6 nu)`` of an explicit
  advection-diffusion scheme is gone, which is what makes a fine grid affordable at small ``nu``;
- the space error is spectral: a smooth field is resolved to machine precision on a grid that resolves its
  spectrum, so the analytic checks below can be asserted at 1e-8 rather than at a discretisation error;
- the cost is 4 non-linear evaluations per step, each five three-dimensional transforms of the three components.

An explicit finite-difference scheme (as in the 1-D :mod:`.fdm` and 2-D :mod:`.fvm` Burgers solvers) was the
alternative: simpler, but first order for the advection term and limited by ``dt <= 1/(2 nu/dx^2 + max|u|/dx)``,
which on a 3-D grid is thousands of steps for the same accuracy. The spectral route is the one that makes the 3-D
viscous decay check exact, so it is the one taken.

The stability guard, and what it is not
---------------------------------------
With the linear part exact there is no CFL number for diffusion. The one bound left is the explicit non-linearity,
and it is the advective Courant number on the finest grid scale,

    cfl = dt * max|u| * (1/dx + 1/dy + 1/dz),      dx = Lx/nx, dy = Ly/ny, dz = Lz/nz

(the multidimensional spectral advection bound: the dealiasing keeps only modes below ``(2/3)(n/2)`` per axis, and
the fastest wave the non-linearity can produce travels at ``max|u|``). This module uses ``cfl`` in three ways:
the default ``dt`` is ``0.4 / (max|u| (1/dx + 1/dy + 1/dz))``; a run is allowed but warned about above
``cfl_warn = 0.5``; and a step above ``cfl_max = 1.0`` is refused with a :class:`ValueError` instead of being run
into a NaN. ``max|u|`` is taken from the initial field and never has to be looked up again, because each component
of this equation obeys a maximum principle: at a point where ``u_i`` is maximal ``grad u_i = 0``, so
``u.grad(u_i) = 0`` there and ``d u_i/dt = nu lap(u_i) <= 0`` -- ``max|u_i|`` cannot grow.

What is analytic here
---------------------
**A single mode that does not advect itself decays at the viscous rate.** Take ``u = (sin y, 0, 0)`` on
``[0, 2 pi]^3``. The only non-zero component does not vary along its own direction (``du_x/dx = 0``) and the other
two are zero, so every term of ``u.grad(u)`` vanishes identically and the problem reduces to the heat equation,
component by component. With ``|k|^2 = 1`` the exact solution is ``u_x(y, t) = exp(-nu t) sin y``, ``u_y = u_z = 0``,
and ``tests/test_burgers3d_solver.py`` asserts the solver reproduces it to 1e-12; measured: 1.1e-15 on a
``8 x 32 x 8`` grid with ``nu = 0.1`` over ``t = 0.5``, i.e. machine precision -- the scheme is exact for that
problem up to rounding. (The tempting ``u = (sin z, sin x, sin y)`` does NOT have this property:
``u.grad(u_x) = u_y du_x/dy = sin x cos z`` is not zero, so there is no closed form and it is not used. What makes
``(sin y, 0, 0)`` special is that the one non-zero component is constant along its own axis.)

**A field that varies only along x is 1-D Burgers.** With ``u = (f(x), 0, 0)``, ``u.grad(u) = (f f', 0, 0)`` and
``nu lap(u) = (nu f'', 0, 0)``, so ``f_t + f f_x = nu f_xx`` with the other two components exactly zero -- the same
problem :mod:`.fdm` solves in 1-D. The test compares against the Cole-Hopf exact solution of that 1-D problem on
the periodic interval, ``u = -2 nu d/dx log phi`` with ``phi_t = nu phi_xx``, computed spectrally from
``phi(x, 0) = exp(-(1/2 nu) integral_0^x f)`` (which needs ``f`` to have zero mean, as ``sin x`` does). Measured at
``nu = 0.1`` up to ``t = 0.5`` on 96 points along ``x``: a maximum error of 2.5e-8 (1.5e-8 relative L2) against that
exact answer, and 4.0e-3 relative against the library's own 1-D ``fdm`` Burgers solver -- whose own error against
Cole-Hopf is 3.1e-3, so the difference between the two solvers is the 1-D solver's first-order upwind error, not
this one's.

References: Burgers (1948), Adv. Appl. Mech. 1:171-199; Cox & Matthews (2002), J. Comput. Phys. 176:430-455;
Kassam & Trefethen (2005), SIAM J. Sci. Comput. 26(4):1214-1233; Orszag (1971) for the 2/3 dealiasing rule.

Quick start
-----------
>>> from pinneapple_simulation.numerical_solvers.burgers3d import Burgers3D, BurgersConfig3D
>>> cfg = BurgersConfig3D(nu=0.01, shape=(32, 32, 32), ic="taylor_green", t_end=0.5)
>>> out = Burgers3D(cfg).solve()
>>> out.u.shape          # (frames, 3, 32, 32, 32)
>>> out.meta["energy"]   # kinetic energy at every saved frame, decreasing
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .etdrk4 import ETDRK4, SimulationBlowUp, check_finite, fourier_wavenumbers
from .fdm3d import SolverOutput3D
from .registry import SolverRegistry

__all__ = [
    "BurgersConfig3D",
    "Burgers3D",
    "Burgers3DSpectralSolver",
    "dealias_mask_3d",
    "burgers_cole_hopf_1d",
]

# The state is (component, x, y, z): the three spatial axes are the trailing ones, so a (nx, ny, nz) wavenumber
# array broadcasts against it without a reshape.
_SPATIAL = (1, 2, 3)


def dealias_mask_3d(shape: tuple[int, int, int], fraction: float = 2.0 / 3.0) -> np.ndarray:
    """The 2/3 dealiasing mask of an ``(nx, ny, nz)`` periodic grid: ``True`` where a mode is kept.

    The rule is the one the library already uses for reaction-diffusion: along each axis, only modes whose index is
    below two thirds of the Nyquist index are kept, and a mode survives only if it does so on all three axes. It
    removes the modes that a product of two resolved modes would fold onto (the quadratic aliasing of the
    ``u.grad(u)`` term) without touching anything the grid resolves.
    """
    keep = [np.abs(np.fft.fftfreq(int(n)) * n) < int(fraction * (int(n) // 2)) for n in shape]
    meshed = np.meshgrid(*keep, indexing="ij")
    mask = meshed[0]
    for other in meshed[1:]:
        mask = mask & other
    return mask


def burgers_cole_hopf_1d(u0: np.ndarray, length: float, nu: float, t: float) -> np.ndarray:
    """Exact solution of the 1-D viscous Burgers equation at time ``t`` by Cole-Hopf, for the periodic interval.

    ``u_t + u u_x = nu u_xx`` with ``u(x, 0) = u0`` on ``[0, length]``. The Hopf-Cole substitution ``u = -2 nu
    phi_x/phi`` turns it into the heat equation ``phi_t = nu phi_xx`` with ``phi = exp(-F/(2 nu))``, ``F' = u0``;
    ``F`` is periodic only when ``u0`` has zero mean, which is checked. ``phi`` is then advanced exactly in Fourier
    space, so the reference has no time-stepping error and only the spectral truncation of ``phi``. Used by the
    tests as the independent reference for the one-dimensional reduction of the 3-D solver; not part of any solve.
    """
    u0 = np.asarray(u0, dtype=np.float64).ravel()
    n = u0.size
    if abs(float(np.mean(u0))) > 1e-12 * max(1.0, float(np.max(np.abs(u0)))):
        raise ValueError("cole-hopf on a periodic interval needs a zero-mean initial condition")
    k = fourier_wavenumbers(n, length)
    # F(x) = integral_0^x u0: the antiderivative in Fourier space (exact and periodic; a cumulative sum of the
    # trapezoid rule would carry a second-order drift that shows up as a 1e-4 error in the reference)
    u0_hat = np.fft.fft(u0)
    F_hat = np.zeros_like(u0_hat)
    nonzero = k != 0
    F_hat[nonzero] = u0_hat[nonzero] / (1j * k[nonzero])
    F = np.fft.ifft(F_hat).real
    phi = np.exp(-F / (2.0 * nu))
    phihat = np.fft.fft(phi) * np.exp(-nu * k ** 2 * t)
    phi_t = np.fft.ifft(phihat).real
    phi_x = np.fft.ifft(1j * k * phihat).real
    return -2.0 * nu * phi_x / phi_t


@dataclass
class BurgersConfig3D:
    """Configuration for the 3-D viscous Burgers solver.

    Parameters
    ----------
    nu : float
        Kinematic viscosity, strictly positive. ``nu = 0`` is the inviscid equation, which shocks and is not
        documented here.
    shape : tuple of int
        Grid points ``(nx, ny, nz)``, at least 8 per axis (the 2/3 rule leaves one resolved mode below that).
    length : float or tuple of float
        Box side ``(Lx, Ly, Lz)``; a scalar for a cube. The box is ``[0, Lx] x [0, Ly] x [0, Lz]``, periodic.
    dt : float, optional
        Time step. ``None`` (the default) uses the advective Courant estimate
        ``0.4 / (max|u| (1/dx + 1/dy + 1/dz))``; a value above ``cfl_max`` is refused.
    nt : int
        Number of time steps when ``t_end`` is not given (default 100).
    t_end : float, optional
        Final time; when given it wins over ``nt`` and ``dt`` is shortened to land on it exactly.
    ic : str, callable or array
        ``"taylor_green"`` (the classical ``A (sin x cos y cos z, -cos x sin y cos z, 0)`` with one period per box
        side), ``"gaussian"`` (``A exp(-|r - centre|^2/(2 sigma^2))`` carried by the z component), a callable
        ``u0(X, Y, Z)`` or an array, in both of those cases the vector field itself, of shape
        ``(3, nx, ny, nz)``.
    amplitude : float
        Amplitude ``A`` of the two built-in initial fields (default 1.0).
    sigma : float, optional
        Width of the ``"gaussian"`` initial field (default ``min(Lx, Ly, Lz)/6``).
    save_every : int
        Save a frame every this many steps (the first and the last are always saved).
    cfl_warn, cfl_max : float
        The advective Courant number above which the run is warned about (0.5) and above which it is refused (1.0).
    """

    nu: float = 0.01
    shape: tuple[int, int, int] = (32, 32, 32)
    length: float | tuple[float, float, float] = 2 * np.pi
    dt: float | None = None
    nt: int = 100
    t_end: float | None = None
    ic: str | Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = "taylor_green"
    amplitude: float = 1.0
    sigma: float | None = None
    save_every: int = 1
    cfl_warn: float = 0.5
    cfl_max: float = 1.0


class Burgers3D:
    """``u_t + u.grad(u) = nu lap(u)`` for the vector field ``u = (u_x, u_y, u_z)`` on a periodic box.

    Pseudo-spectral in space (``i k`` derivatives, 2/3 dealiasing per axis) with the ETDRK4 time scheme of
    :mod:`.etdrk4`: the linear part ``-nu |k|^2`` is integrated exactly, so there is no diffusive time-step limit,
    and the only guard is the advective Courant number (see the module docstring).

    Attributes
    ----------
    shape, sides : tuple
        Grid points and box sides per axis.
    x : tuple of ndarray
        Cell centres per axis, ``i L/n`` (the periodic grid excludes the right endpoint).
    mask : ndarray
        The 2/3 dealiasing mask, ``(nx, ny, nz)``.
    k2 : ndarray
        ``kx^2 + ky^2 + kz^2`` of the grid.
    """

    def __init__(self, cfg: BurgersConfig3D | None = None):
        self.cfg = cfg or BurgersConfig3D()
        cfg = self.cfg
        self.nu = float(cfg.nu)
        if not self.nu > 0.0:
            raise ValueError(
                f"Burgers3D solves the VISCOUS Burgers equation and needs nu > 0; nu = {self.nu:g} is the inviscid "
                f"equation, which forms shocks and needs an entropy-satisfying scheme this module does not have.")
        shape = cfg.shape if isinstance(cfg.shape, (tuple, list)) else (cfg.shape,)
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3:
            raise ValueError(f"Burgers3D is three-dimensional; shape must be (nx, ny, nz), not {self.shape}")
        if min(self.shape) < 8:
            raise ValueError(
                f"Burgers3D keeps only modes below two thirds of Nyquist per axis, which below 8 points leaves a "
                f"single resolved mode; shape {self.shape} is too small.")
        sides = (cfg.length,) * 3 if np.isscalar(cfg.length) else tuple(cfg.length)
        if len(sides) != 3:
            raise ValueError("length must be a scalar or one side per axis (Lx, Ly, Lz)")
        self.sides = tuple(float(v) for v in sides)
        if min(self.sides) <= 0.0:
            raise ValueError(f"the box sides must be positive, not {self.sides}")

        self.x = tuple(L * np.arange(n) / n for n, L in zip(self.shape, self.sides, strict=True))
        self.dx = tuple(L / n for n, L in zip(self.shape, self.sides, strict=True))
        self.grid = tuple(np.meshgrid(*self.x, indexing="ij"))
        self.k = tuple(fourier_wavenumbers(n, L) for n, L in zip(self.shape, self.sides, strict=True))
        kx, ky, kz = np.meshgrid(*self.k, indexing="ij")
        self.k2 = kx ** 2 + ky ** 2 + kz ** 2
        self.K = (kx, ky, kz)
        self.mask = dealias_mask_3d(self.shape)
        self.keep = [np.abs(np.fft.fftfreq(n) * n) < int((2.0 / 3.0) * (n // 2)) for n in self.shape]
        self.k_max = tuple(float(np.max(np.abs(k[m]))) for k, m in zip(self.k, self.keep, strict=True))
        self.volume = self.dx[0] * self.dx[1] * self.dx[2]

    # ------------------------------------------------------------------
    # Initial fields
    # ------------------------------------------------------------------

    def initial_field(self) -> np.ndarray:
        """The starting vector field, of shape ``(3, nx, ny, nz)``."""
        ic = self.cfg.ic
        expected = (3,) + self.shape
        if isinstance(ic, np.ndarray):
            u = np.asarray(ic, dtype=np.float64)
        elif callable(ic):
            u = np.asarray(ic(*self.grid), dtype=np.float64)
        else:
            u = None
        if u is not None:
            if u.shape != expected:
                raise ValueError(
                    f"the initial field has shape {u.shape}; this solver's field is the vector u = (u_x, u_y, u_z) "
                    f"and must be {expected}. A scalar field is a different equation -- wrap it as "
                    f"np.stack([f(X, Y, Z), 0*f, 0*f]) if it is meant to be the u_x component.")
            if not np.all(np.isfinite(u)):
                raise ValueError("the initial field has non-finite values")
            return u
        if ic == "taylor_green":
            X, Y, Z = self.grid
            kx, ky, kz = (2.0 * np.pi / L for L in self.sides)
            a = float(self.cfg.amplitude)
            return a * np.stack([
                np.sin(kx * X) * np.cos(ky * Y) * np.cos(kz * Z),
                -np.cos(kx * X) * np.sin(ky * Y) * np.cos(kz * Z),
                np.zeros(self.shape),
            ])
        if ic == "gaussian":
            X, Y, Z = self.grid
            sigma = float(self.cfg.sigma) if self.cfg.sigma else min(self.sides) / 6.0
            centre = [L / 2.0 for L in self.sides]
            r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2
            blob = float(self.cfg.amplitude) * np.exp(-r2 / (2.0 * sigma ** 2))
            return np.stack([np.zeros(self.shape), np.zeros(self.shape), blob])
        raise ValueError(
            f"ic must be 'taylor_green', 'gaussian' or a callable u0(X, Y, Z) returning (3, nx, ny, nz), not {ic!r}")

    # ------------------------------------------------------------------
    # The scheme's two pieces
    # ------------------------------------------------------------------

    def nonlinear(self, uh: np.ndarray, t: float) -> np.ndarray:
        """``N(u) = -u.grad(u)`` in Fourier space: products in physical space, derivatives as ``i k``, dealiased.

        ``uh`` is the Fourier transform of the vector field along its three spatial axes, shape
        ``(3, nx, ny, nz)``; the return value has the same shape.
        """
        u = np.fft.ifftn(uh, axes=_SPATIAL).real
        conv = np.zeros_like(u)
        for axis in range(3):
            # d/dx_axis of EVERY component: entry i is du_i/dx_axis
            du = np.fft.ifftn(1j * self.K[axis] * uh, axes=_SPATIAL).real
            # u_axis * du_i/dx_axis, summed over the axis -> component i of u.grad(u_i)
            conv += u[axis] * du
        return -np.fft.fftn(conv, axes=_SPATIAL) * self.mask

    def courant_rate(self, umax: float) -> float:
        """``max|u| (1/dx + 1/dy + 1/dz)``: the inverse of the largest stable step at Courant number 1."""
        return float(umax) * (1.0 / self.dx[0] + 1.0 / self.dx[1] + 1.0 / self.dx[2])

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def solve(self) -> SolverOutput3D:
        """Advance the configured initial field and return every saved frame.

        The output's ``u`` is ``(frames, 3, nx, ny, nz)``; ``meta["energy"]`` is the kinetic energy
        ``0.5 integral |u|^2 dV`` at each saved frame and ``meta["cfl"]`` the advective Courant number of the step.
        """
        cfg = self.cfg
        u = self.initial_field()
        umax = float(np.max(np.abs(u)))
        rate = self.courant_rate(umax)
        dt = cfg.dt
        if dt is None:
            # a zero field never evolves (and would divide by zero): fall back to a unit reference velocity
            dt = 0.4 / (rate if rate > 0.0 else self.courant_rate(1.0))
        dt = float(dt)
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, not {dt:g}")
        if cfg.t_end is not None:
            nt = max(1, int(np.ceil(float(cfg.t_end) / dt)))
            dt = float(cfg.t_end) / nt
        else:
            nt = max(1, int(cfg.nt))

        cfl = self.courant_rate(umax) * dt
        if cfl > float(cfg.cfl_max):
            raise ValueError(
                f"Burgers3D: the advective Courant number is {cfl:.2f} > cfl_max={cfg.cfl_max:g} "
                f"(dt={dt:.3g}, max|u|={umax:.3g}, 1/dx+1/dy+1/dz={rate / max(umax, 1e-300):.3g}). ETDRK4 has no "
                f"diffusion limit but the explicit non-linearity does: dt at or below {0.4 / rate if rate else 0.0:.3g} "
                f"is the recommended step.")
        elif cfl > float(cfg.cfl_warn):
            warnings.warn(
                f"Burgers3D: the advective Courant number is {cfl:.2f} > {cfg.cfl_warn:g}; the non-linear term is "
                f"approaching its stability limit. Reduce dt to about {0.4 / rate if rate else 0.0:.3g}.",
                stacklevel=2,
            )

        scheme = ETDRK4(-self.nu * self.k2, self.nonlinear, dt)
        uh = np.fft.fftn(u, axes=_SPATIAL)
        save_every = max(1, int(cfg.save_every))
        frames, times, energies, maxima = [], [], [], []
        t = 0.0
        for step in range(nt + 1):
            if step % save_every == 0 or step == nt:
                field = np.fft.ifftn(uh, axes=_SPATIAL).real
                frames.append(field)
                times.append(t)
                energies.append(0.5 * float(np.sum(field ** 2)) * self.volume)
                maxima.append(float(np.max(np.abs(field))))
            if step == nt:
                break
            uh = scheme.step(uh, t)
            t = (step + 1) * dt
            try:
                check_finite(uh, limit=1e30, step=step + 1, t=t)
            except SimulationBlowUp as exc:
                raise SimulationBlowUp(
                    f"{exc} -- the run diverged; with cfl={cfl:.2f} the advective step is the suspect: reduce dt "
                    f"below {0.4 / rate if rate else 0.0:.3g} or resolve the field on a finer grid.",
                    step=exc.step, t=exc.t) from None

        return SolverOutput3D(
            u=np.stack(frames),
            coords={"x": self.x[0], "y": self.x[1], "z": self.x[2], "t": np.asarray(times)},
            meta={
                "solver": "Burgers3D",
                "scheme": "ETDRK4 (Cox-Matthews, Kassam-Trefethen contour) with 2/3-dealiased pseudo-spectral advection",
                "nu": self.nu,
                "dt": dt,
                "nt": nt,
                "t_end": times[-1],
                "cfl": cfl,
                "max_abs_u": maxima[-1],
                "energy": np.asarray(energies),
                "shape": self.shape,
                "sides": self.sides,
                "dealiased": True,
                "dealias_k_max": self.k_max,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# The registry face: a ProblemSpec in, the 3-D Burgers solve out
# ──────────────────────────────────────────────────────────────────────────────


@SolverRegistry.register(
    name="burgers3d_spectral",
    family="pde",
    description="3-D viscous Burgers u_t + u.grad(u) = nu lap(u) on a periodic box: pseudo-spectral space with "
                "2/3 dealiasing, ETDRK4 in time (stiff diffusion exact, no diffusive CFL)",
    tags=["etdrk4", "burgers", "3d", "periodic", "spectral", "turbulence"],
)
class Burgers3DSpectralSolver(SolverBase):
    """``solve_from_spec`` for :class:`Burgers3D`: the viscous Burgers equation for the vector field
    ``u = (u_x, u_y, u_z)`` on a periodic box, solved pseudo-spectrally with ETDRK4.

    The spec's parameters (``spec.pde.params``):

    - ``nu`` kinematic viscosity [m^2/s], required and strictly positive (``viscosity`` is read as a fallback);
    - ``n`` grid points per axis (default 32), with ``nx``/``ny``/``nz`` overriding a single axis;
    - ``dt`` time step (default: ``0.4 / (max|u| (1/dx + 1/dy + 1/dz))``, the advective Courant estimate);
    - ``nt`` steps (default 100), or ``t_end`` final time, which wins over ``nt``;

    and the initial field, in this precedence: ``ic`` (``"taylor_green"``, ``"gaussian"`` or a callable
    ``u0(X, Y, Z)`` returning ``(3, nx, ny, nz)``), else the spec's own ``initial`` condition sampled on the grid
    (its ``value_fn`` must return three components -- a scalar start is refused, since the field here is the
    vector ``u``), else ``"taylor_green"``. ``amplitude``/``sigma`` shape the two built-in starts, and
    ``save_every`` (default: ``nt``, i.e. the first and the last frame only) how many frames are kept.

    The box is the spec's ``domain_bounds`` (its extents; ``2 pi`` per axis when the spec gives none). The returned
    grid is labelled ``0 .. L`` per axis, which on a periodic box is a relabelling of the spec's own origin; an
    ``initial`` condition is nevertheless sampled at the spec's own coordinates, not at the relabelled ones.

    What it refuses, by name:

    - a kind that is not Burgers, and ``inviscid_burgers`` / ``nu = 0``: the inviscid equation forms shocks and is
      a different problem, not a special case of this one;
    - ``dim != 3``: the 1-D and 2-D Burgers solvers are ``fdm`` and ``fvm``;
    - any condition that is not an ``initial`` one: the box is periodic and has no wall, so a Dirichlet, Neumann,
      Robin or interface condition would have to be dropped (solving a different problem) or faked. The condition
      kinds are named in the error;
    - an initial condition whose ``value_fn`` does not return three components.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if not any(k in kind for k in ("burgers",)):
            raise ValueError(
                f"burgers3d_spectral solves the viscous Burgers equation in 3-D and the spec's kind is '{kind}'. "
                f"Its documented scope is Burgers3D: u_t + u.grad(u) = nu lap(u) for the vector field "
                f"u = (u_x, u_y, u_z) on a periodic box.")
        if "inviscid" in kind:
            raise ValueError(
                f"burgers3d_spectral solves the VISCOUS Burgers equation, and the spec's kind '{kind}' is the "
                f"inviscid one (u_t + u.grad(u) = 0). Without nu there is no diffusion to balance the steepening, "
                f"the solution breaks into shocks and needs an entropy-satisfying scheme; a pseudo-spectral solver "
                f"would ring around them instead of resolving them.")
        dim = int(getattr(spec, "dim", 3))
        if dim != 3:
            raise ValueError(
                f"burgers3d_spectral is three-dimensional and the spec is {dim}-D. The 1-D Burgers solver is 'fdm' "
                f"and the 2-D one is 'fvm'; this module documents 3-D only.")

        conditions = tuple(getattr(spec, "conditions", ()) or ())
        walls = [c for c in conditions if not str(getattr(c, "kind", "")).startswith("initial")]
        if walls:
            named = sorted({f"{getattr(c, 'kind', '?')} '{getattr(c, 'name', '')}'" for c in walls})
            raise ValueError(
                f"burgers3d_spectral solves on a periodic box, which has no wall to hold at a value and no "
                f"boundary at all: the spec carries {named}. Dropping "
                f"{'that condition' if len(walls) == 1 else 'those conditions'} would solve a different problem, and "
                f"imposing one on a periodic box would need a forcing or a buffer this solver does not have.")

        params = dict(getattr(spec.pde, "params", {}) or {})
        nu = params.get("nu", params.get("viscosity"))
        if nu is None:
            raise ValueError("burgers3d_spectral needs the viscosity in spec.pde.params['nu'] (or 'viscosity')")
        nu = float(nu)
        if not nu > 0.0:
            raise ValueError(
                f"burgers3d_spectral needs nu > 0 and the spec gives nu = {nu:g}, which is the inviscid Burgers "
                f"equation -- shocks, a different solution, and not what this solver discretises.")

        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        sides = tuple(
            abs(float(bounds[axis][1]) - float(bounds[axis][0])) if axis in bounds
            else float(params.get("length", 2 * np.pi))
            for axis in ("x", "y", "z"))
        n = int(params.get("n", 32))
        shape = (int(params.get("nx", n)), int(params.get("ny", n)), int(params.get("nz", n)))
        ic = self._initial_condition(spec, params, shape, bounds)

        dt = params.get("dt")
        cfg = BurgersConfig3D(
            nu=nu, shape=shape, length=sides, dt=None if dt is None else float(dt),
            nt=int(params.get("nt", 100)),
            t_end=None if params.get("t_end") is None else float(params["t_end"]),
            ic=ic, amplitude=float(params.get("amplitude", 1.0)),
            sigma=None if params.get("sigma") is None else float(params["sigma"]),
            save_every=int(params.get("save_every", max(1, int(params.get("nt", 100))))),
        )
        out = Burgers3D(cfg).solve()
        final = np.asarray(out.u[-1], dtype=np.float64)
        energy = float(out.meta["energy"][-1])
        max_abs_u = float(out.meta["max_abs_u"])
        return SolverOutput(
            result=torch.from_numpy(final.astype(np.float32)),
            losses={"max_abs_u": torch.tensor(max_abs_u)},
            extras={
                "method": "burgers3d_spectral",
                "scheme": out.meta["scheme"],
                "nu": nu,
                "shape": tuple(shape),
                "sides": sides,
                "dt": float(out.meta["dt"]),
                "nt": int(out.meta["nt"]),
                "t_end": float(out.meta["t_end"]),
                "cfl": float(out.meta["cfl"]),
                "dealiased": True,
                "dealias_k_max": out.meta["dealias_k_max"],
                "energy": energy,
                "ic": ("spec_initial_condition" if isinstance(ic, np.ndarray)
                       else ("callable" if callable(ic) else str(ic))),
                "coords": out.coords,
                "history_shape": tuple(np.shape(out.u)),
            },
        )

    @staticmethod
    def _initial_condition(spec, params, shape, bounds):
        """The start this solve uses: ``params["ic"]`` when given, else the spec's own initial condition sampled on
        the solver's grid, else the Taylor-Green vortex.

        The spec's initial condition is evaluated at the spec's OWN coordinates (``domain_bounds``, not the
        solver's relabelled ``[0, L]``), so a field written for ``[-pi, pi]`` is read where the spec says it is.
        """
        if params.get("ic") is not None:
            return params["ic"]
        initial = [c for c in (getattr(spec, "conditions", ()) or ())
                   if str(getattr(c, "kind", "")).startswith("initial")]
        if not initial:
            return "taylor_green"
        cond = initial[0]
        coords = tuple(getattr(spec, "coords", ()) or ())
        axes = []
        for axis, n in zip(("x", "y", "z"), shape, strict=True):
            lo, hi = bounds.get(axis, (0.0, 2 * np.pi))
            axes.append(float(lo) + (float(hi) - float(lo)) * np.arange(n) / n)
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        if "t" in coords:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        values = np.asarray(cond.values(pts.astype(np.float32), {}), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            given = f"{values.shape[1]} value(s) per point" if values.ndim == 2 else "no value per point"
            raise ValueError(
                f"the initial condition '{getattr(cond, 'name', '')}' returns {given}, and this solver's field is the "
                f"vector u = (u_x, u_y, u_z) of shape (3, {shape[0]}, {shape[1]}, {shape[2]}): give a three-component "
                f"value_fn, or the field itself in spec.pde.params['ic'] as a callable u0(X, Y, Z).")
        field = np.moveaxis(values.reshape(shape + (3,)), -1, 0)
        if not np.all(np.isfinite(field)):
            raise ValueError(f"the initial condition '{getattr(cond, 'name', '')}' is not finite on the grid")
        return field
