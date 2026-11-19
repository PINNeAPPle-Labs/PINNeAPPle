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

__all__ = ["ReactionDiffusion1D", "ReactionDiffusion1DSolver", "ReactionDiffusion",
           "ReactionDiffusionNDSolver"]

_REACTIONS = {"allen_cahn": lambda u: u - u ** 3, "fisher": lambda u: u * (1.0 - u),
              "none": lambda u: np.zeros_like(u)}


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


@dataclass
class ReactionDiffusion:
    """Reaction-diffusion ``u_t = D lap(u) + R(u)`` on a periodic box, in one, two or three dimensions, with the
    same ETDRK4 scheme and 2/3 dealiasing as the 1-D class above.

    ``shape`` is the number of points per axis, ``length`` the box side per axis (a scalar for a cube). The three
    reactions are ``"allen_cahn"`` (``u - u^3``), ``"fisher"`` (``u (1 - u)``) and ``"none"`` -- pure diffusion,
    which is what a quantitative check of the scheme can be written against: a single Fourier mode decays as
    ``exp(-D |k|^2 t)`` and nothing else happens.
    """

    reaction: str = "allen_cahn"
    diffusion: float = 1e-3
    shape: tuple[int, ...] = (128,)
    length: float | tuple[float, ...] = 2.0
    dt: float = 0.05

    def __post_init__(self):
        if self.reaction not in _REACTIONS:
            raise ValueError(f"reaction must be one of {sorted(_REACTIONS)}")
        self.shape = tuple(int(n) for n in (self.shape if isinstance(self.shape, (tuple, list)) else (self.shape,)))
        if not 1 <= len(self.shape) <= 3:
            raise ValueError(f"shape must be 1, 2 or 3 dimensional, not {self.shape}")
        sides = (self.length,) * len(self.shape) if np.isscalar(self.length) else tuple(self.length)
        self.sides = tuple(float(v) for v in sides)
        if len(self.sides) != len(self.shape):
            raise ValueError("length must be a scalar or one side per axis")
        axes_k = [fourier_wavenumbers(n, L) for n, L in zip(self.shape, self.sides, strict=True)]
        mesh = np.meshgrid(*axes_k, indexing="ij")
        self.k2 = sum(k ** 2 for k in mesh)
        # 2/3 dealiasing, per axis: the modes kept are the ones below two thirds of the Nyquist wavenumber
        keep = [np.abs(np.fft.fftfreq(n) * n) < int((2 / 3) * (n // 2)) for n in self.shape]
        self.mask = np.meshgrid(*keep, indexing="ij")[0]
        for other in np.meshgrid(*keep, indexing="ij")[1:]:
            self.mask = self.mask & other
        R = _REACTIONS[self.reaction]
        axes = tuple(range(len(self.shape)))

        def nonlin(u_hat, t):
            return np.fft.fftn(R(np.fft.ifftn(u_hat, axes=axes).real), axes=axes) * self.mask
        self.scheme = ETDRK4(-self.diffusion * self.k2, nonlin, dt=self.dt)

    def initial_state(self, seed: int = 0, amplitude: float = 0.5, modes: int = 6) -> np.ndarray:
        """A smooth random state built from the mask's low modes; for Fisher-KPP shifted into [0, 1]."""
        rng = np.random.default_rng(seed)
        spectrum = (rng.standard_normal(self.shape) + 1j * rng.standard_normal(self.shape)) * self.mask
        u = np.real(np.fft.ifftn(spectrum))
        peak = float(np.max(np.abs(u))) or 1.0
        u = amplitude * u / peak
        return np.clip(0.5 + u, 0.0, 1.0) if self.reaction == "fisher" else u

    def step(self, u: np.ndarray, steps: int = 1) -> np.ndarray:
        """Advance ``u`` (of ``shape``, or with a leading batch axis) by ``steps`` time steps."""
        axes = tuple(range(-len(self.shape), 0))
        uh = np.fft.fftn(np.asarray(u, float), axes=axes)
        for _ in range(steps):
            uh = self.scheme.step(uh, 0.0)
        return np.fft.ifftn(uh, axes=axes).real

    def trajectory(self, u0: np.ndarray, n_frames: int, every: int = 1) -> np.ndarray:
        """``(n_frames + 1, *shape)`` states, ``every`` steps apart, starting with ``u0``."""
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
        if "reaction" not in kind and "fisher" not in kind:
            raise ValueError(f"reaction_diffusion1d solves reaction-diffusion (or a Fisher-KPP front), and the "
                             f"spec's kind is '{kind}'.")
        if int(getattr(spec, "dim", 1)) != 1:
            raise ValueError(f"reaction_diffusion1d documents a 1-D periodic interval; the spec is "
                             f"{getattr(spec, 'dim', '?')}-D. The 2-D and 3-D case is reaction_diffusion_nd.")
        conditions = tuple(getattr(spec, "conditions", ()) or ())
        if conditions:
            raise ValueError(
                "reaction_diffusion1d solves on a periodic interval and has no boundary to impose: the spec "
                f"carries {sorted({str(getattr(c, 'kind', '?')) for c in conditions})}.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        system = ReactionDiffusion1D(
            reaction=str(params.get("reaction") or ("fisher" if "fisher" in kind else "allen_cahn")),
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


@SolverRegistry.register(
    name="reaction_diffusion_nd",
    family="pde",
    description="Reaction-diffusion in 1, 2 or 3 dimensions on a periodic box (Allen-Cahn, Fisher-KPP or pure "
                "diffusion) with the ETDRK4 scheme",
    tags=["etdrk4", "reaction_diffusion", "2d", "3d", "periodic", "spectral"],
)
class ReactionDiffusionNDSolver(SolverBase):
    """``solve_from_spec`` for :class:`ReactionDiffusion`, the 2-D and 3-D face of reaction-diffusion.

    The spec's parameters (``spec.pde.params``): ``reaction`` (``"allen_cahn"``, ``"fisher"`` or ``"none"`` for
    pure diffusion), ``diffusion`` (``D``, default 1e-3), ``n`` points per axis (default 64), ``dt`` (default
    0.02) and ``nt`` steps (default 200), plus ``seed``/``amplitude``/``modes`` for the initial state. The box is
    the spec's ``domain_bounds`` (its extents; default 1.0 per axis).

    The box is periodic, so a spec carrying a boundary condition is refused by name: there is no wall here to hold
    at a value, and dropping the condition would solve a different problem.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if "reaction" not in kind and "fisher" not in kind:
            raise ValueError(f"reaction_diffusion_nd solves reaction-diffusion (or a Fisher-KPP front), and the "
                             f"spec's kind is '{kind}'.")
        dim = int(getattr(spec, "dim", 2))
        if dim not in (1, 2, 3):
            raise ValueError(f"reaction_diffusion_nd is 1-, 2- or 3-D; the spec is {dim}-D.")
        conditions = tuple(getattr(spec, "conditions", ()) or ())
        if conditions:
            raise ValueError(
                "reaction_diffusion_nd solves on a periodic box and has no boundary to impose: the spec carries "
                f"{sorted({str(getattr(c, 'kind', '?')) for c in conditions})}.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        axes = ("x", "y", "z")[:dim]
        sides = tuple(abs(bounds[a][1] - bounds[a][0]) if a in bounds else float(params.get("length", 1.0))
                      for a in axes)
        n = int(params.get("n", 64))
        system = ReactionDiffusion(
            reaction=str(params.get("reaction") or ("fisher" if "fisher" in kind else "allen_cahn")),
            diffusion=float(params.get("diffusion", 1e-3)),
            shape=(n,) * dim,
            length=sides,
            dt=float(params.get("dt", 0.02)),
        )
        steps = int(params.get("nt", 200))
        u0 = system.initial_state(seed=int(params.get("seed", 0)),
                                  amplitude=float(params.get("amplitude", 0.4)),
                                  modes=int(params.get("modes", 6)))
        u = system.step(u0, steps)
        return SolverOutput(
            result=torch.from_numpy(np.asarray(u, dtype=np.float32)),
            losses={"mean": torch.tensor(float(np.mean(u)))},
            extras={"method": "reaction_diffusion_nd", "dim": dim, "reaction": system.reaction,
                    "diffusion": system.diffusion, "shape": system.shape, "sides": system.sides,
                    "dt": system.dt, "nt": steps, "min": float(np.min(u)), "max": float(np.max(u))},
        )
