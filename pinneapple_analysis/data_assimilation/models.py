"""Small chaotic models used to test data-assimilation methods, written in PyTorch so they are differentiable.

* Lorenz-96 (Lorenz 1996, "Predictability: a problem partly solved", ECMWF Seminar):
  dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F, cyclic; F = 8 gives chaos with a doubling time of ~0.4 time
  units for 40 variables, the standard data-assimilation test bed.
* Lorenz-63 (Lorenz 1963, J. Atmos. Sci. 20:130).

Each model step is one RK4 step of size ``dt`` and accepts batched states (..., n).
"""
from __future__ import annotations

import torch

__all__ = ["lorenz96_tendency", "lorenz96_step", "lorenz63_step", "rk4"]


def rk4(f, x: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz96_tendency(x: torch.Tensor, F: float = 8.0) -> torch.Tensor:
    return (torch.roll(x, -1, -1) - torch.roll(x, 2, -1)) * torch.roll(x, 1, -1) - x + F


def lorenz96_step(x: torch.Tensor, dt: float = 0.05, F: float = 8.0) -> torch.Tensor:
    return rk4(lambda y: lorenz96_tendency(y, F), x, dt)


def lorenz63_step(x: torch.Tensor, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0,
                  beta: float = 8.0 / 3.0) -> torch.Tensor:
    def f(y):
        a, b, c = y[..., 0], y[..., 1], y[..., 2]
        return torch.stack([sigma * (b - a), a * (rho - c) - b, a * b - beta * c], dim=-1)
    return rk4(f, x, dt)
