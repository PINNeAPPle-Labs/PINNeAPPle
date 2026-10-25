"""
Input adapters between the arena's pointwise data and PINNeAPPle architectures.

Every arena problem is pointwise field regression: a batch of coordinates
X of shape (N, in_dim) maps to field values of shape (N, out_dim), and the
physics loss differentiates u(x_i) with respect to x_i by autograd. Many
PINNeAPPle architectures, however, take grids or sequences — channels-first
(B, C, L) for FNO / conv1d, (B, C, H, W) for conv2d, (B, H, W, C) for AFNO,
(B, T, D) for transformers, RNNs, TCNs and reservoirs. Handed a flat (N, d)
tensor they either crash or silently treat the N points as one sample.

The adapters below treat every point as its own sample of spatial / temporal
extent 1. That keeps each prediction a function of its own point only (no
mixing across points, independent of point order), which is exactly what the
autograd physics residual requires.

Three groups:
- PointwiseAdapter  — gradient-trained models; reshape in, flatten out.
- ClosedFormAdapter — models whose readout is solved in closed form via
  fit(X, y) (ELM, hybrid RBF, ESN readouts); trained by the pipeline's
  closed-form path instead of gradient descent.
- NOT_APPLICABLE    — unsupervised trajectory / snapshot models (Koopman/EDMD,
  POD, DMD, HAVOK, ARIMA). Their fit() takes a time series or snapshot matrix
  and no target, so a coordinates→field regression cannot train them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover — the pipeline falls back to numpy
    HAS_TORCH = False


class ArchitectureNotApplicable(ValueError):
    """The architecture cannot be trained on pointwise coordinate→field data."""


NOT_APPLICABLE: dict[str, str] = {
    "koopman": "Koopman/EDMD fits a linear operator on trajectories x(t)→x(t+1) with no target field",
    "pod":     "POD extracts modes from a snapshot matrix; it has no coordinate→field mapping to train",
    "dmd":     "DMD fits linear dynamics between sequential snapshots, not a coordinate→field regression",
    "havok":   "HAVOK embeds a single chaotic time series in a Hankel matrix; there is no target field",
    "arima":   "ARIMA forecasts a univariate time series from its own past; there is no coordinate input",
}


def _as_tensor(out: Any) -> torch.Tensor:
    if hasattr(out, "y") and out.y is not None:
        return out.y
    if isinstance(out, tuple):
        return out[0]
    return out


# layout -> (reshape (N, d) into the model's input, extra positional args)
_LAYOUTS: dict[str, Callable[[torch.Tensor], tuple]] = {
    "channels_first_1d": lambda x: (x[:, :, None],),
    "channels_first_2d": lambda x: (x[:, :, None, None],),
    "channels_first_3d": lambda x: (x[:, :, None, None, None],),
    "channels_last_2d":  lambda x: (x[:, None, None, :],),
    "sequence":          lambda x: (x[:, None, :],),
    # continuous-time sequence models take (x, t) with t of shape (T,)
    "sequence_with_time": lambda x: (x[:, None, :], torch.zeros(1, dtype=x.dtype, device=x.device)),
}

POINTWISE_LAYOUTS: dict[str, str] = {
    "fno": "channels_first_1d",
    "conv1d": "channels_first_1d",
    "conv2d": "channels_first_2d",
    "conv3d": "channels_first_3d",
    "afno": "channels_last_2d",
    "transformer": "sequence",
    "pinnsformer": "sequence",
    "bayesian_rnn": "sequence",
    "deep_state_space_model": "sequence",
    "tcn": "sequence",
    "ode_rnn": "sequence_with_time",
    "neural_cde": "sequence_with_time",
}

# arch -> (layout, extra fit kwargs)
CLOSED_FORM: dict[str, tuple] = {
    "elm": (None, {}),
    "hybrid_rbf": (None, {}),
    "esn": ("sequence", {"one_step_ahead": False}),
    "esn_rc": ("sequence", {}),
}


if HAS_TORCH:

    class PointwiseAdapter(nn.Module):
        """Run a grid/sequence model on N independent points of extent 1."""

        def __init__(self, inner: nn.Module, layout: str, in_dim: int, out_dim: int):
            super().__init__()
            self.inner = inner
            self.layout = layout
            self.out_dim = out_dim
            # Some models emit a fixed width (e.g. ODE-RNN's head returns input_dim);
            # probe once and add a linear head when it differs from out_dim.
            with torch.no_grad():
                width = self._raw(torch.zeros(2, in_dim)).shape[-1]
            self.head: nn.Module | None = nn.Linear(width, out_dim) if width != out_dim else None

        def _raw(self, x: torch.Tensor) -> torch.Tensor:
            out = _as_tensor(self.inner(*_LAYOUTS[self.layout](x)))
            return out.reshape(x.shape[0], -1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self._raw(x)
            return self.head(y) if self.head is not None else y

    class ClosedFormAdapter(nn.Module):
        """Wrap a model trained by fit(X, y) instead of gradient descent."""

        closed_form = True

        def __init__(self, inner: nn.Module, layout: str | None, fit_kwargs: dict, out_dim: int):
            super().__init__()
            self.inner = inner
            self.layout = layout
            self.fit_kwargs = fit_kwargs
            self.out_dim = out_dim

        def _shape(self, x: torch.Tensor) -> torch.Tensor:
            return _LAYOUTS[self.layout](x)[0] if self.layout else x

        def fit(self, X: np.ndarray, y: np.ndarray) -> ClosedFormAdapter:
            # Solve in float64: hybrid RBF leaves its bias column unregularised by
            # scaling it ×1e6, so XᵀX mixes ~1e12 entries with an l2 of 1e-6 —
            # singular in float32. One linear solve, so the cost is negligible.
            x_t = torch.as_tensor(X, dtype=torch.float64)
            y_t = torch.as_tensor(y, dtype=torch.float64).reshape(len(X), -1)
            y_in = y_t[:, None, :] if self.layout == "sequence" else y_t
            self.inner.double()
            try:
                with torch.no_grad():
                    self.inner.fit(self._shape(x_t), y_in, **self.fit_kwargs)
            finally:
                self.inner.float()
            return self

        def _apply(self, fn, recurse=True):
            # Closed-form readouts keep solved weights in plain tensor attributes
            # that .to(device) would not move, so the wrapped model stays on the
            # CPU; forward() moves inputs there and outputs back.
            return self

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = _as_tensor(self.inner(self._shape(x.detach().to("cpu", torch.float32))))
            return out.reshape(x.shape[0], -1)[:, : self.out_dim].to(x.device)


def adapt_model(arch_id: str, model: Any, in_dim: int, out_dim: int) -> Any:
    """Return `model` wrapped so it accepts (N, in_dim) and returns (N, out_dim)."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return model
    if arch_id in POINTWISE_LAYOUTS:
        return PointwiseAdapter(model, POINTWISE_LAYOUTS[arch_id], in_dim, out_dim)
    if arch_id in CLOSED_FORM:
        layout, fit_kwargs = CLOSED_FORM[arch_id]
        return ClosedFormAdapter(model, layout, fit_kwargs, out_dim)
    return model


def check_applicable(arch_id: str) -> None:
    reason = NOT_APPLICABLE.get(arch_id)
    if reason:
        raise ArchitectureNotApplicable(
            f"Not applicable to pointwise field problems: {reason}."
        )
