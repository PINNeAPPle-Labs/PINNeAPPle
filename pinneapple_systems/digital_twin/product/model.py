"""What a model must accept to be run as a twin, and an adapter so existing surrogates fit.

The contract is one signature::

    model(X, theta) -> {field_name: array of shape (n,)}

where ``X`` is ``(n, n_coords)`` in the column order the spec declares, and ``theta`` is the
latent state the twin is estimating — the fouling resistance, the wall thickness, the bias.

Passing ``theta`` down to the model is what makes a twin more than a plot: the model has to
be able to *reproduce the observed world* for some value of the latent state, and the filter
finds that value. A surrogate that ignores ``theta`` still works as a twin, but it can only
be corrected on its outputs, not its inputs — which is why :class:`BiasModel` exists.

Limits
------
Nothing here validates that the model is physically right. If the model cannot express how
the latent state changes the outputs, the filter will converge to a plausible wrong answer
and report a tight band around it. Check the backtest against held-out sensors before
believing the band.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["TwinModelError", "as_array_dict", "BiasModel", "ensure_twin_model"]

Fields = Mapping[str, np.ndarray]


class TwinModelError(ValueError):
    """The model does not honour the twin contract."""


def as_array_dict(out: Any, field_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Normalise a model output to ``{field: (n,) float32}``.

    Accepts a dict, a 2-D array, or a 1-D array for a single-field twin — the three shapes
    surrogates in this library already return.
    """
    names = list(field_names)
    if isinstance(out, Mapping):
        missing = [f for f in names if f not in out]
        if missing:
            raise TwinModelError(f"model output is missing field(s) {missing}; it returned {sorted(out)}")
        arrays = {f: np.asarray(out[f], dtype=np.float32).ravel() for f in names}
    else:
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1:
            if len(names) != 1:
                raise TwinModelError(
                    f"model returned a 1-D array but the twin predicts {len(names)} fields {names}"
                )
            arrays = {names[0]: arr}
        elif arr.ndim == 2:
            if arr.shape[1] < len(names):
                raise TwinModelError(
                    f"model returned {arr.shape[1]} column(s) for {len(names)} field(s) {names}"
                )
            arrays = {f: arr[:, i] for i, f in enumerate(names)}
        else:
            raise TwinModelError(f"model output has {arr.ndim} dimensions; expected 1 or 2")

    lengths = {len(v) for v in arrays.values()}
    if len(lengths) > 1:
        raise TwinModelError(f"model returned fields of different lengths: {lengths}")
    return arrays


class BiasModel:
    """Wrap a plain ``model(X)`` so it satisfies the twin contract with an output bias.

    The latent state is one additive offset per field, in field units. This is the cheapest
    honest correction: it fixes a model that is *right in shape, wrong in level* (a sensor
    drift, a fouled surface, an off-calibration). It cannot fix a model whose error depends
    on the operating point — for that, give the model its own ``theta``.

    Parameters
    ----------
    model :
        Callable ``model(X) -> dict | array`` with the signature surrogates already use.
    field_names :
        The fields the twin predicts, in order.
    """

    def __init__(self, model: Callable[[np.ndarray], Any], field_names: Sequence[str]) -> None:
        self.model = model
        self.field_names = list(field_names)

    def __call__(self, X: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float32).ravel()
        if theta.size != len(self.field_names):
            raise TwinModelError(
                f"bias model expects {len(self.field_names)} latent offsets, got {theta.size}"
            )
        base = as_array_dict(self.model(X), self.field_names)
        return {f: base[f] + theta[i] for i, f in enumerate(self.field_names)}


def _positional_arity(model: Callable[..., Any]) -> int:
    """How many positional arguments ``model`` takes, best effort.

    Uses :func:`inspect.signature`, which handles functions, ``functools.partial``, bound
    methods and ``nn.Module.__call__``. When the signature cannot be read (some C callables),
    assume two — the documented twin contract — rather than silently dropping ``theta``.
    """
    try:
        params = inspect.signature(model).parameters.values()
    except (TypeError, ValueError):
        return 2
    n = 0
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind is p.VAR_POSITIONAL:
            return max(n, 2)
    return n


def ensure_twin_model(
    model: Callable[..., Any],
    field_names: Sequence[str],
    n_latent: int,
) -> Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    """Return a callable that honours the contract, wrapping ``model`` when it does not.

    * already takes ``theta`` (arity 2) → used as is, output normalised;
    * takes only ``X`` (arity 1) and ``n_latent == len(field_names)`` → wrapped in
      :class:`BiasModel`, the natural reading of "one latent offset per field";
    * takes only ``X`` and ``n_latent == 0`` → wrapped with an empty latent state;
    * anything else → :class:`TwinModelError`, because guessing would hide a modelling bug.
    """
    names = list(field_names)
    takes_theta = _positional_arity(model) >= 2

    if takes_theta:
        def call(X: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
            return as_array_dict(model(X, theta), names)
        return call

    if n_latent == 0:
        def call_zero(X: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
            return as_array_dict(model(X), names)
        return call_zero

    if n_latent != len(names):
        raise TwinModelError(
            f"model takes only X, so the latent state must be one offset per field "
            f"({len(names)}), but the spec declares {n_latent}. Give the model a theta argument."
        )
    return BiasModel(model, names)
