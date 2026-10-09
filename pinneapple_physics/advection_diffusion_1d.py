"""Periodic 1-D advection-diffusion u_t + c u_x = nu u_xx on [0, 2 pi): exact solution, classical finite-difference
schemes and the PDE residual of any predicted space-time field.

A small, exactly solvable test bed for comparing and ensembling physics models: each Fourier mode decays and travels
exactly, u_k(t) = u_k(0) exp((-i c k - nu k^2) t), so any surrogate, numerical method or PINN can be scored without a
reference solver. The schemes have known, regime-dependent errors: first-order upwind adds numerical diffusion
(LeVeque 2002, §8.6), Lax-Wendroff is second order but dispersive, and a coarse grid loses accuracy when nu is small.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AdvectionDiffusion1D"]


@dataclass
class AdvectionDiffusion1D:
    nx: int = 64
    nt: int = 21
    t_end: float = 1.0

    def __post_init__(self):
        self.x = 2 * np.pi * np.arange(self.nx) / self.nx
        self.k = np.fft.fftfreq(self.nx, 1 / self.nx)
        self.t = np.linspace(0, self.t_end, self.nt)

    def random_case(self, rng: np.random.Generator, nu: float, c: float = 1.0, n_modes: int = 4) -> dict[str, object]:
        u0 = 1.0 + sum(rng.normal() / m * np.sin(m * self.x + rng.uniform(0, 2 * np.pi)) for m in range(1, n_modes + 1))
        return {"u0": u0, "nu": float(nu), "c": float(c)}

    def exact(self, case) -> np.ndarray:
        """Space-time field (nt, nx)."""
        U0 = np.fft.fft(case["u0"])
        phase = np.exp((-1j * case["c"] * self.k[None] - case["nu"] * self.k[None] ** 2) * self.t[:, None])
        return np.real(np.fft.ifft(U0[None] * phase, axis=1))

    def finite_difference(self, case, scheme: str = "lax_wendroff", coarsen: int = 1, cfl: float = 0.4) -> np.ndarray:
        """Explicit scheme ("upwind" or "lax_wendroff" for advection, central diffusion) on every ``coarsen``-th grid
        point, spectrally interpolated back to the full grid."""
        x = self.x[::coarsen]
        u = np.asarray(case["u0"], dtype=float)[::coarsen].copy()
        dx, c, nu = x[1] - x[0], case["c"], case["nu"]
        dt = cfl * min(dx / max(abs(c), 1e-12), dx * dx / (2 * nu + 1e-12))
        out, t = [u.copy()], 0.0
        for tn in self.t[1:]:
            while t < tn - 1e-12:
                h = min(dt, tn - t)
                up, um = np.roll(u, -1), np.roll(u, 1)
                if scheme == "upwind":
                    adv = -c * h / dx * ((u - um) if c >= 0 else (up - u))
                elif scheme == "lax_wendroff":
                    adv = -c * h / (2 * dx) * (up - um) + 0.5 * (c * h / dx) ** 2 * (up - 2 * u + um)
                else:
                    raise ValueError("scheme must be 'upwind' or 'lax_wendroff'")
                u = u + adv + nu * h / dx ** 2 * (up - 2 * u + um)
                t += h
            out.append(u.copy())
        out = np.array(out)
        if coarsen == 1:
            return out
        U = np.fft.rfft(out, axis=1)
        full = np.zeros((out.shape[0], self.nx // 2 + 1), dtype=complex)
        full[:, : U.shape[1]] = U * coarsen
        return np.fft.irfft(full, n=self.nx, axis=1)

    def residual(self, field: np.ndarray, case) -> float:
        """Relative RMS of u_t + c u_x - nu u_xx (spectral in x, centred differences in t) over interior times."""
        F = np.fft.fft(field, axis=1)
        ux = np.real(np.fft.ifft(1j * self.k * F, axis=1))
        uxx = np.real(np.fft.ifft(-self.k ** 2 * F, axis=1))
        ut = np.gradient(field, self.t, axis=0)
        r = ut + case["c"] * ux - case["nu"] * uxx
        return float(np.sqrt(np.mean(r[1:-1] ** 2)) / (np.sqrt(np.mean(ut[1:-1] ** 2)) + 1e-12))

    def total(self, field_row: np.ndarray) -> float:
        """Integral over the period (conserved by the PDE)."""
        return float(np.sum(field_row) * 2 * np.pi / self.nx)
