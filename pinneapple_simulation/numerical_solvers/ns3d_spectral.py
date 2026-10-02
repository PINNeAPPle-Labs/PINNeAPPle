"""Pseudo-spectral solver for the 3-D incompressible Navier-Stokes equations on the periodic
box [0, 2*pi)^3 (rotational form, 2/3-rule dealiasing, classical RK4, exact integrating
factor for viscosity).

    du/dt = P[u x omega] - nu (-lap) u,     div u = 0

Diagnostics per step: kinetic energy E, enstrophy Z, dissipation eps = 2 nu Z, max |omega|,
and the Beale-Kato-Majda integral int_0^t max|omega| ds (a finite-time blow-up of a smooth
solution requires this integral to diverge -- BKM 1984).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["SpectralNS3D", "taylor_green_ic"]


def taylor_green_ic(N: int):
    x = np.arange(N) * 2 * np.pi / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return np.stack([u, v, w])


@dataclass
class SpectralNS3D:
    N: int
    nu: float
    dt: float

    def __post_init__(self):
        k = np.fft.fftfreq(self.N, 1.0 / self.N)
        kr = np.fft.rfftfreq(self.N, 1.0 / self.N)
        self.K = np.stack(np.meshgrid(k, k, kr, indexing="ij"))
        self.K2 = (self.K ** 2).sum(0)
        self.K2i = np.where(self.K2 == 0, 1.0, 1.0 / np.maximum(self.K2, 1e-30))
        kmax = self.N // 3
        self.dealias = ((np.abs(self.K[0]) < kmax * 1.0) & (np.abs(self.K[1]) < kmax) & (np.abs(self.K[2]) < kmax)).astype(float)

    # ---------------------------------------------------------------- transforms
    def fwd(self, f):
        return np.fft.rfftn(f, axes=(-3, -2, -1))

    def inv(self, fh):
        return np.fft.irfftn(fh, s=(self.N,) * 3, axes=(-3, -2, -1))

    def curl_hat(self, uh):
        K = self.K
        return 1j * np.stack([K[1] * uh[2] - K[2] * uh[1], K[2] * uh[0] - K[0] * uh[2], K[0] * uh[1] - K[1] * uh[0]])

    def project(self, fh):
        kdotf = (self.K * fh).sum(0)
        return fh - self.K * kdotf * self.K2i

    def rhs(self, uh):
        u = self.inv(uh)
        w = self.inv(self.curl_hat(uh))
        nl = np.stack([u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]])
        return self.project(self.fwd(nl) * self.dealias)

    # ---------------------------------------------------------------- integration
    def run(self, u0: np.ndarray, t_end: float, diag_every: int = 5,
            callback: Optional[Callable[[float, Dict], None]] = None) -> Dict[str, List[float]]:
        uh = self.project(self.fwd(u0))
        dt, nu = self.dt, self.nu
        E_half = np.exp(-nu * self.K2 * dt / 2)
        E_full = np.exp(-nu * self.K2 * dt)
        n = int(round(t_end / dt))
        hist = {"t": [], "E": [], "Z": [], "eps": [], "wmax": [], "bkm": []}
        bkm, t = 0.0, 0.0
        last_w = None
        for s in range(n + 1):
            if s % diag_every == 0 or s == n:
                u = self.inv(uh)
                w = self.inv(self.curl_hat(uh))
                E = 0.5 * float((u ** 2).sum(0).mean())
                Z = 0.5 * float((w ** 2).sum(0).mean())
                wmax = float(np.sqrt((w ** 2).sum(0)).max())
                if last_w is not None:
                    bkm += 0.5 * (wmax + last_w[1]) * (t - last_w[0])
                last_w = (t, wmax)
                for k, v in zip(("t", "E", "Z", "eps", "wmax", "bkm"), (t, E, Z, 2 * nu * Z, wmax, bkm)):
                    hist[k].append(v)
                if not np.isfinite(E):
                    raise FloatingPointError(f"non-finite energy at t={t}")
                if callback:
                    callback(t, {k: v[-1] for k, v in hist.items()})
            if s == n:
                break
            # RK4 with integrating factor for the viscous term
            k1 = self.rhs(uh)
            k2 = self.rhs(E_half * (uh + 0.5 * dt * k1))
            k3 = self.rhs(E_half * uh + 0.5 * dt * k2)
            k4 = self.rhs(E_full * uh + dt * E_half * k3)
            uh = E_full * uh + dt / 6 * (E_full * k1 + 2 * E_half * (k2 + k3) + k4)
            t += dt
        return hist
