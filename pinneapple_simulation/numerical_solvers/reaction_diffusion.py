"""One-dimensional reaction-diffusion on a periodic interval, ``u_t = D u_xx + R(u)``, with the ETDRK4 scheme of
:mod:`.etdrk4` (stiff diffusion exact, reaction explicit, 2/3 dealiasing).

Reactions: ``"allen_cahn"`` ``R = u - u^3`` (phase separation into domains u = +-1 joined by interfaces of width
``sqrt(2 D)`` that coarsen slowly) and ``"fisher"`` ``R = u (1 - u)`` (Fisher-KPP: an autocatalytic front invading
the unstable state at speed ``2 sqrt(D)``). Checked against the travelling-front speed and the exact interface
profile ``tanh(x / sqrt(2 D))`` in ``tests/test_validation_gallery.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .etdrk4 import ETDRK4, fourier_wavenumbers

__all__ = ["ReactionDiffusion1D"]

_REACTIONS = {"allen_cahn": lambda u: u - u ** 3, "fisher": lambda u: u * (1.0 - u)}


@dataclass
class ReactionDiffusion1D:
    reaction: str = "allen_cahn"
    diffusion: float = 1e-3
    n: int = 128
    length: float = 2.0
    dt: float = 0.05

    def __post_init__(self):
        if self.reaction not in _REACTIONS:
            raise ValueError(f"reaction must be one of {sorted(_REACTIONS)}")
        self.k = fourier_wavenumbers(self.n, self.length)
        mask = np.abs(np.fft.fftfreq(self.n) * self.n) < int((2 / 3) * (self.n // 2))
        R = _REACTIONS[self.reaction]

        def nonlin(u_hat, t):
            return np.fft.fft(R(np.fft.ifft(u_hat, axis=-1).real), axis=-1) * mask
        self.scheme = ETDRK4(-self.diffusion * self.k ** 2, nonlin, dt=self.dt)

    @property
    def x(self) -> np.ndarray:
        return self.length * np.arange(self.n) / self.n

    def initial_state(self, seed: int = 0, amplitude: float = 0.5, modes: int = 6) -> np.ndarray:
        """A smooth random state (a few low modes); for Fisher-KPP shifted into [0, 1]."""
        rng = np.random.default_rng(seed)
        u = amplitude * sum(rng.standard_normal() * np.cos(2 * np.pi * m * self.x / self.length
                                                           + rng.uniform(0, 2 * np.pi)) for m in range(1, modes + 1))
        return np.clip(0.5 + u, 0.0, 1.0) if self.reaction == "fisher" else u

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
