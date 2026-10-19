"""Kuramoto-Sivashinsky in one dimension, ``u_t + (u^2 / 2)_x + u_xx + u_xxxx = 0`` on a periodic interval, with the
ETDRK4 scheme of :mod:`.etdrk4` (Kassam & Trefethen 2005 set-up, 2/3 dealiasing).

The same discretisation is cross-validated against Exponax to round-off in
``tests/test_rea_harvest_numerics.py::test_etdrk4_matches_exponax_to_round_off``. For ``L = 32 pi`` the dynamics are
chaotic: two trajectories from nearby states separate exponentially, while the long-run statistics (histogram of
``u``, time-averaged spectrum) are those of the attractor. The spatial mean is conserved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .etdrk4 import ETDRK4, fourier_wavenumbers

__all__ = ["KuramotoSivashinsky1D"]


@dataclass
class KuramotoSivashinsky1D:
    n: int = 128
    length: float = 32 * np.pi
    dt: float = 0.25

    def __post_init__(self):
        self.k = fourier_wavenumbers(self.n, self.length)
        self.mask = np.abs(np.fft.fftfreq(self.n) * self.n) < int((2 / 3) * (self.n // 2))
        k, mask = self.k, self.mask

        def nonlin(u_hat, t):
            u = np.fft.ifft(u_hat).real
            return -0.5j * k * np.fft.fft(u * u) * mask
        self.scheme = ETDRK4(k ** 2 - k ** 4, nonlin, dt=self.dt)

    @property
    def x(self) -> np.ndarray:
        return self.length * np.arange(self.n) / self.n

    def initial_state(self, seed: int = 0, amplitude: float = 1.0) -> np.ndarray:
        """A smooth random state (a few low modes)."""
        rng = np.random.default_rng(seed)
        x = self.x
        return amplitude * sum(rng.standard_normal() * np.cos(m * 2 * np.pi * x / self.length + rng.uniform(0, 6.3))
                               for m in range(1, 6))

    def step(self, u: np.ndarray, steps: int = 1) -> np.ndarray:
        """Advance ``u`` (``(n,)`` or ``(batch, n)``) by ``steps`` time steps."""
        uh = np.fft.fft(np.asarray(u, float), axis=-1)
        for _ in range(steps):
            uh = self.scheme.step(uh, 0.0)
        return np.fft.ifft(uh, axis=-1).real

    def trajectory(self, u0: np.ndarray, n_frames: int, every: int = 1) -> np.ndarray:
        """``(n_frames + 1, ...)`` states, ``every`` steps apart, starting with ``u0``."""
        out = [np.asarray(u0, float)]
        for _ in range(n_frames):
            out.append(self.step(out[-1], every))
        return np.stack(out)
