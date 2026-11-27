"""Stable Fluids: unconditionally stable 2-D incompressible flow on a periodic grid.

Each step, in this order:

1. vorticity confinement (optional) -- Fedkiw, Stam & Jensen (2001),
   "Visual simulation of smoke", SIGGRAPH: force
   ``f = eps * dx * (N x omega)``, ``N = grad|omega| / |grad|omega||``;
2. semi-Lagrangian advection of velocity and passive scalars (dye) --
   Stam (1999), "Stable fluids", SIGGRAPH: backtrace ``x - dt u(x)`` and
   bilinearly interpolate (periodic wrap);
3. viscous diffusion, integrated exactly in Fourier space (``exp(-nu k^2 dt)``);
4. pressure projection onto discretely divergence-free fields.

Where this departs from the WebGL reference that was studied (Pavel
Dobryakov's *WebGL-Fluid-Simulation*, MIT -- the step order and kernels were
located with REA, see ``docs/dev/rea_harvest/ledger.json``):

- The WebGL code projects with a fixed number of Jacobi sweeps on the
  5-point Laplacian while measuring divergence and subtracting the pressure
  gradient with *central* differences. That pair of operators is not
  consistent (``div(grad)`` with central differences is the wide, stride-2
  Laplacian), so the projected field is never exactly divergence-free even
  with infinite sweeps. Here the projection solves with the symbol of the
  very operators used (``-(sin(k dx)/dx)^2``) via FFT, so the central-
  difference divergence after projection is zero to round-off. The
  unresolvable Nyquist / checkerboard modes are removed.
- Lengths and the confinement force are in physical units (``dx``), not
  texel units.

Dimensions: arrays are indexed ``[iy, ix]``; velocity is ``(2, ny, nx)`` with
component 0 = ``u_x`` and 1 = ``u_y``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["StableFluids2D", "StableFluidsState", "stable_time_step"]


def stable_time_step(max_velocity: float, dx: float, *, courant: float = 0.5,
                     viscosity: float = 0.0, ndim: int = 2, implicit_diffusion: bool = True) -> float:
    """Advective CFL step ``courant * dx / max_velocity``, capped by the
    explicit-diffusion limit ``dx^2 / (2 ndim nu)`` unless diffusion is
    implicit (as in :class:`StableFluids2D`).

    Same contract as jax-cfd's ``equations.stable_time_step`` (Apache-2.0),
    except the diffusion bound uses the FTCS von Neumann limit ``2*ndim``
    (jax-cfd uses ``2**ndim``, identical in 2-D, stricter in 3-D) and returns
    the smaller step instead of raising.
    """
    dt = np.inf if max_velocity <= 0 else courant * dx / max_velocity
    if viscosity > 0 and not implicit_diffusion:
        dt = min(dt, dx * dx / (2 * ndim * viscosity))
    return float(dt)


@dataclass
class StableFluidsState:
    velocity: np.ndarray                       # (2, ny, nx)
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    t: float = 0.0


class StableFluids2D:
    """Periodic box ``[0, Lx) x [0, Ly)`` with ``nx x ny`` cell-centred samples.

    Parameters
    ----------
    nx, ny : grid size (even numbers recommended).
    length : ``(Lx, Ly)``; must give square cells (``Lx/nx == Ly/ny``).
    viscosity : kinematic viscosity ``nu`` (exact spectral diffusion).
    confinement : vorticity-confinement strength ``eps`` (0 disables it).
    scalar_dissipation : decay rate ``d`` of passive scalars, applied as
        ``s / (1 + d dt)`` (the WebGL rule).
    """

    def __init__(self, nx: int, ny: int, *, length: tuple[float, float] = (2 * np.pi, 2 * np.pi),
                 viscosity: float = 0.0, confinement: float = 0.0, scalar_dissipation: float = 0.0):
        self.nx, self.ny = int(nx), int(ny)
        self.Lx, self.Ly = map(float, length)
        self.dx = self.Lx / self.nx
        if not np.isclose(self.dx, self.Ly / self.ny):
            raise ValueError("cells must be square: Lx/nx == Ly/ny")
        self.nu = float(viscosity)
        self.eps = float(confinement)
        self.scalar_dissipation = float(scalar_dissipation)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dx)
        self.KX, self.KY = np.meshgrid(kx, ky)          # (ny, nx)
        self.K2 = self.KX ** 2 + self.KY ** 2
        # Symbol of the central first difference: D = i sin(k dx)/dx.
        self.DX = 1j * np.sin(self.KX * self.dx) / self.dx
        self.DY = 1j * np.sin(self.KY * self.dx) / self.dx
        self._lap_c = (self.DX * self.DX + self.DY * self.DY).real  # <= 0
        self._unresolved = np.isclose(self._lap_c, 0.0) & (self.K2 > 0)
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dx
        self.X, self.Y = np.meshgrid(xs, ys)

    # -- discrete operators (central differences, periodic) -------------
    def _ddx(self, f: np.ndarray) -> np.ndarray:
        return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * self.dx)

    def _ddy(self, f: np.ndarray) -> np.ndarray:
        return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2 * self.dx)

    def divergence(self, vel: np.ndarray) -> np.ndarray:
        return self._ddx(vel[0]) + self._ddy(vel[1])

    def vorticity(self, vel: np.ndarray) -> np.ndarray:
        return self._ddx(vel[1]) - self._ddy(vel[0])

    # -- the four sub-steps ---------------------------------------------
    def project(self, vel: np.ndarray) -> np.ndarray:
        """Remove the gradient part so that :meth:`divergence` is ~0."""
        u_h = np.fft.fft2(vel[0])
        v_h = np.fft.fft2(vel[1])
        div_h = self.DX * u_h + self.DY * v_h
        lap = np.where(self._lap_c == 0, 1.0, self._lap_c)
        p_h = np.where(self._lap_c == 0, 0.0, div_h / lap)
        u_h = u_h - self.DX * p_h
        v_h = v_h - self.DY * p_h
        # Modes invisible to the central stencil (k dx = pi) can be neither
        # measured nor corrected; drop them.
        u_h[self._unresolved] = 0.0
        v_h[self._unresolved] = 0.0
        return np.stack([np.fft.ifft2(u_h).real, np.fft.ifft2(v_h).real])

    def _sample(self, f: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Periodic bilinear interpolation of cell-centred ``f`` at (x, y)."""
        gx = x / self.dx - 0.5
        gy = y / self.dx - 0.5
        i0 = np.floor(gx).astype(int)
        j0 = np.floor(gy).astype(int)
        fx, fy = gx - i0, gy - j0
        i0 %= self.nx
        j0 %= self.ny
        i1 = (i0 + 1) % self.nx
        j1 = (j0 + 1) % self.ny
        return ((1 - fy) * ((1 - fx) * f[..., j0, i0] + fx * f[..., j0, i1])
                + fy * ((1 - fx) * f[..., j1, i0] + fx * f[..., j1, i1]))

    def advect(self, field_: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        """Semi-Lagrangian: value at x is the old value at ``x - dt u(x)``."""
        return self._sample(field_, self.X - dt * vel[0], self.Y - dt * vel[1])

    def diffuse(self, vel: np.ndarray, dt: float) -> np.ndarray:
        if self.nu == 0:
            return vel
        decay = np.exp(-self.nu * self.K2 * dt)
        return np.fft.ifft2(np.fft.fft2(vel, axes=(-2, -1)) * decay, axes=(-2, -1)).real

    def confine(self, vel: np.ndarray, dt: float) -> np.ndarray:
        if self.eps == 0:
            return vel
        w = self.vorticity(vel)
        aw = np.abs(w)
        nx_, ny_ = self._ddx(aw), self._ddy(aw)
        norm = np.sqrt(nx_ ** 2 + ny_ ** 2) + 1e-12
        nx_, ny_ = nx_ / norm, ny_ / norm
        # (N x omega z_hat) = (N_y * w, -N_x * w)
        fx = self.eps * self.dx * ny_ * w
        fy = -self.eps * self.dx * nx_ * w
        return vel + dt * np.stack([fx, fy])

    # -- driver ----------------------------------------------------------
    def initial_state(self, velocity: np.ndarray, scalars: dict[str, np.ndarray] | None = None) -> StableFluidsState:
        vel = np.asarray(velocity, dtype=float)
        if vel.shape != (2, self.ny, self.nx):
            raise ValueError(f"velocity must have shape (2, {self.ny}, {self.nx})")
        return StableFluidsState(self.project(vel), dict(scalars or {}))

    def step(self, state: StableFluidsState, dt: float,
             force: np.ndarray | None = None) -> StableFluidsState:
        vel = state.velocity
        if force is not None:
            vel = vel + dt * force
        vel = self.confine(vel, dt)
        new_scalars = {k: self.advect(s, vel, dt) / (1 + self.scalar_dissipation * dt)
                       for k, s in state.scalars.items()}
        vel = self.advect(vel, vel, dt)
        vel = self.diffuse(vel, dt)
        vel = self.project(vel)
        return StableFluidsState(vel, new_scalars, state.t + dt)

    def run(self, state: StableFluidsState, t_end: float, *, courant: float = 0.5,
            dt: float | None = None) -> StableFluidsState:
        """Advance to ``t_end``; ``dt`` defaults to :func:`stable_time_step`."""
        while state.t < t_end - 1e-12:
            umax = float(np.sqrt((state.velocity ** 2).sum(axis=0)).max())
            h = dt or stable_time_step(umax, self.dx, courant=courant)
            h = min(h, t_end - state.t)
            state = self.step(state, h)
        return state
