"""Test problems of published physics-ML papers, as the papers (or, where the paper is silent, their official code)
define them: domain, equation, initial / boundary data, training and test grids, and a reference solution.

Used by ``examples/paper_reproductions``; every method in a comparison gets the same points and reference from
here, and is scored by :mod:`pinneapple_analysis.reproduction`.

PINNsFormer cases (Zhao, Ding & Prakash, ICLR 2024). Settings from the official code, AdityaLab/pinnsformer at
commit 00d790f50c3f77599fbe263b9e8035f9860a1027: ``util.get_data`` (grids), ``util.make_time_sequence``
(pseudo-sequences) and the demo notebooks ``demo/convection/convection_pinnsformer.ipynb``,
``demo/1d_reaction/1d_reaction_pinnsformer.ipynb`` and ``demo/1d_wave/1d_wave_pinnsformer.ipynb`` (equations,
coefficients, losses, analytic solutions of the reaction and wave cases). No numbers in this module are results.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GridCase", "pinnsformer_case", "PINNSFORMER_CASES", "tensor_grid", "time_sequence",
           "convection_exact", "reaction_exact", "wave_exact"]


def tensor_grid(x_range, t_range, nx: int, nt: int) -> dict[str, np.ndarray]:
    """The official ``get_data`` grid: ``meshgrid(x, t)`` ('xy' indexing, rows are times), flattened row-major.

    Returns ``res`` (nt*nx, 2) with columns (x, t), and the four edges with the official names:
    ``left`` = the t = t0 row (initial condition), ``upper`` = the x = x1 column, ``lower`` = the x = x0 column,
    ``right`` = the t = t1 row (unused by the losses).
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    X, T = np.meshgrid(x, t)
    data = np.stack([X, T], axis=-1)                    # (nt, nx, 2)
    return {"res": data.reshape(-1, 2), "left": data[0], "right": data[-1], "upper": data[:, -1],
            "lower": data[:, 0], "shape": np.array([nt, nx])}


def time_sequence(points: np.ndarray, k: int = 5, dt: float = 1e-4) -> np.ndarray:
    """PINNsFormer's pseudo-sequence: each (x, t) repeated k times with t + i*dt, i = 0..k-1 -> (N, k, 2)."""
    seq = np.repeat(np.asarray(points, dtype=np.float64)[:, None, :], k, axis=1)
    seq[:, :, -1] += dt * np.arange(k)[None, :]
    return seq


def convection_exact(x, t, beta: float = 50.0):
    """u_t + beta u_x = 0, u(x, 0) = sin x, periodic on [0, 2 pi]: u = sin(x - beta t)."""
    return np.sin(np.asarray(x) - beta * np.asarray(t))


def _reaction_h(x):
    return np.exp(-(np.asarray(x) - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))


def reaction_exact(x, t, rho: float = 5.0):
    """u_t = rho u (1 - u), u(x, 0) = h(x) = exp(-(x - pi)^2 / (2 (pi/4)^2)): logistic solution
    u = h e^{rho t} / (h e^{rho t} + 1 - h)."""
    h = _reaction_h(x)
    e = np.exp(rho * np.asarray(t))
    return h * e / (h * e + 1 - h)


def wave_exact(x, t, c2: float = 4.0):
    """u_tt = c2 u_xx on [0, 1], u(0) = u(1) = 0, u(x, 0) = sin(pi x) + 0.5 sin(3 pi x), u_t(x, 0) = 0:
    u = sin(pi x) cos(c pi t) + 0.5 sin(3 pi x) cos(3 c pi t), c = sqrt(c2)."""
    c = np.sqrt(c2)
    x, t = np.asarray(x), np.asarray(t)
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(3 * c * np.pi * t)


@dataclass(frozen=True)
class GridCase:
    """One test problem on a tensor (x, t) grid with an analytic reference."""
    name: str
    equation: str
    x_range: tuple[float, float]
    t_range: tuple[float, float]
    exact: Callable[[np.ndarray, np.ndarray], np.ndarray]
    n_train: int = 51                      # points per axis of the training grid
    n_test: int = 101                      # points per axis of the test grid
    bc: str = "periodic"                   # "periodic" (u(x0) = u(x1)) or "dirichlet0"
    lbfgs_steps: int = 500                 # official notebook: outer L-BFGS steps
    coef: dict = field(default_factory=dict)
    source: str = ""

    def train(self) -> dict[str, np.ndarray]:
        return tensor_grid(self.x_range, self.t_range, self.n_train, self.n_train)

    def test(self) -> tuple[np.ndarray, np.ndarray]:
        """Test points (n_test^2, 2) and the reference on them, in the official ordering."""
        g = tensor_grid(self.x_range, self.t_range, self.n_test, self.n_test)
        return g["res"], self.exact(g["res"][:, 0], g["res"][:, 1])

    def ic(self, x):
        return self.exact(np.asarray(x), np.zeros_like(np.asarray(x, dtype=float)))


_SRC = "AdityaLab/pinnsformer@00d790f5"
PINNSFORMER_CASES: dict[str, GridCase] = {
    "convection": GridCase("convection", "u_t + 50 u_x = 0", (0.0, 2 * np.pi), (0.0, 1.0), convection_exact,
                           bc="periodic", lbfgs_steps=500, coef={"beta": 50.0},
                           source=f"{_SRC} demo/convection/convection_pinnsformer.ipynb cells 2-4"),
    "reaction": GridCase("reaction", "u_t - 5 u (1 - u) = 0", (0.0, 2 * np.pi), (0.0, 1.0), reaction_exact,
                         bc="periodic", lbfgs_steps=500, coef={"rho": 5.0},
                         source=f"{_SRC} demo/1d_reaction/1d_reaction_pinnsformer.ipynb cells 2-6"),
    "wave": GridCase("wave", "u_tt - 4 u_xx = 0", (0.0, 1.0), (0.0, 1.0), wave_exact, bc="dirichlet0",
                     lbfgs_steps=1000, coef={"c2": 4.0},
                     source=f"{_SRC} demo/1d_wave/1d_wave_pinnsformer.ipynb cells 2-6"),
}


def pinnsformer_case(name: str) -> GridCase:
    return PINNSFORMER_CASES[name]
