"""Exogenous signal providers for ``PINNFactory``: known inputs such as sensor data or forcing.

An equation like ``Derivative(T(t), t) - a*S(t) + b*(T(t) - Tamb(t))`` uses ``S`` and ``Tamb``
as *known* functions of time (weather data), not as network outputs. The factory needs their
values at whatever points it evaluates the residual (collocation, conditions, data). A provider
is either a tensor with one row per point of that batch, or a callable ``f(*inputs) -> Tensor``
that can be evaluated anywhere -- ``TabulatedSignal`` is the usual choice for sampled data.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

ArrayLike = Union[Sequence[float], np.ndarray, torch.Tensor]


class TabulatedSignal:
    """Piecewise-linear interpolation of a sampled signal, evaluated in torch.

    Parameters
    ----------
    x : sample locations (strictly increasing) along ``input_index`` of the model inputs.
    values : sample values, same length as ``x``.
    input_index : which model input the signal depends on (0 = first independent variable).
    extrapolate : ``"clamp"`` (hold the end values, default) or ``"linear"``.
    """

    def __init__(self, x: ArrayLike, values: ArrayLike, *, input_index: int = 0, extrapolate: str = "clamp"):
        xs = torch.as_tensor(np.asarray(x, dtype=np.float64)).reshape(-1)
        ys = torch.as_tensor(np.asarray(values, dtype=np.float64)).reshape(-1)
        if xs.numel() != ys.numel() or xs.numel() < 2:
            raise ValueError("x and values must have the same length (>= 2)")
        if not bool(torch.all(xs[1:] > xs[:-1])):
            raise ValueError("x must be strictly increasing")
        if extrapolate not in ("clamp", "linear"):
            raise ValueError("extrapolate must be 'clamp' or 'linear'")
        self.x, self.y = xs, ys
        self.input_index = input_index
        self.extrapolate = extrapolate

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        q = inputs[self.input_index]
        shape = q.shape
        qf = q.reshape(-1)
        x = self.x.to(device=qf.device, dtype=qf.dtype)
        y = self.y.to(device=qf.device, dtype=qf.dtype)
        if self.extrapolate == "clamp":
            qf = qf.clamp(float(x[0]), float(x[-1]))
        i = torch.searchsorted(x, qf.detach().contiguous()).clamp(1, x.numel() - 1)
        x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
        out = y0 + (y1 - y0) * (qf - x0) / (x1 - x0)
        return out.reshape(shape if len(shape) == 2 else (-1, 1))
