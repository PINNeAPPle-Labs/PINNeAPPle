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
import torch

from .base import SolverBase, SolverOutput
from .etdrk4 import ETDRK4, fourier_wavenumbers
from .registry import SolverRegistry

__all__ = ["ReactionDiffusion1D", "ReactionDiffusion1DSolver"]

_REACTIONS = {"allen_cahn": lambda u: u - u ** 3, "fisher": lambda u: u * (1.0 - u)}


@dataclass
class ReactionDiffusion1D:
    """1D reaction-diffusion system (Allen-Cahn by default): grid, initial state, step and trajectory with ETDRK4."""
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


@SolverRegistry.register(
    name="reaction_diffusion1d",
    family="pde",
    description="1-D reaction-diffusion u_t = D u_xx + R(u) on a periodic interval (Allen-Cahn u - u^3 or "
                "Fisher-KPP u (1 - u)) with the ETDRK4 scheme",
    tags=["etdrk4", "reaction_diffusion", "1d", "periodic", "spectral"],
)
class ReactionDiffusion1DSolver(SolverBase):
    """``solve_from_spec`` for :class:`ReactionDiffusion1D`.

    The spec's parameters (``spec.pde.params``): ``reaction`` (``"allen_cahn"`` or ``"fisher"``, default
    ``"allen_cahn"``), ``diffusion`` (``D``, default 1e-3), ``n`` points (default 128), ``length`` (default
    2.0), ``dt`` (default 0.05) and ``nt`` steps (default 200). The state starts from
    :meth:`ReactionDiffusion1D.initial_state` with ``seed``/``amplitude``/``modes`` from the parameters.

    The interval is periodic, so a spec carrying a boundary condition is refused by name: there is no wall here
    to hold at a value, and silently dropping the condition would solve a different problem.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if "reaction" not in kind:
            raise ValueError(f"reaction_diffusion1d solves reaction-diffusion, and the spec's kind is '{kind}'.")
        if int(getattr(spec, "dim", 1)) != 1:
            raise ValueError(f"reaction_diffusion1d documents a 1-D periodic interval; the spec is "
                             f"{getattr(spec, 'dim', '?')}-D. Nothing in the library documents reaction-diffusion "
                             f"in 2-D or 3-D.")
        conditions = tuple(getattr(spec, "conditions", ()) or ())
        if conditions:
            raise ValueError(
                "reaction_diffusion1d solves on a periodic interval and has no boundary to impose: the spec "
                f"carries {sorted({str(getattr(c, 'kind', '?')) for c in conditions})}.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        system = ReactionDiffusion1D(
            reaction=str(params.get("reaction", "allen_cahn")),
            diffusion=float(params.get("diffusion", 1e-3)),
            n=int(params.get("n", 128)),
            length=float(params.get("length", 2.0)),
            dt=float(params.get("dt", 0.05)),
        )
        states = int(params.get("nt", 200))
        u0 = system.initial_state(seed=int(params.get("seed", 0)),
                                  amplitude=float(params.get("amplitude", 0.5)),
                                  modes=int(params.get("modes", 6)))
        u = system.step(u0, states)
        return SolverOutput(
            result=torch.from_numpy(np.asarray(u, dtype=np.float32)),
            losses={"mean": torch.tensor(float(np.mean(u)))},
            extras={"method": "reaction_diffusion1d", "reaction": system.reaction,
                    "diffusion": system.diffusion, "n": system.n, "length": system.length,
                    "dt": system.dt, "nt": states, "min": float(np.min(u)), "max": float(np.max(u)),
                    "x": system.x},
        )
