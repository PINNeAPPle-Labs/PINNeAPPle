"""Downsampling of long series for plotting and JSON payloads: one implementation for the library and the apps.

A chart of a 10^6-sample signal needs a few thousand points. :func:`downsample` picks them; the default ``"minmax"``
keeps, in each of ``k / 2`` equal buckets, the smallest and the largest sample, so spikes and dips survive (an evenly
strided pick can step over them). The other methods reproduce the strides the tools used before this module:

- ``"even"``: ``min(k, n)`` indices evenly spaced from the first to the last sample (``linspace``);
- ``"stride"``: every ``ceil(n / k)``-th sample from the first (keeps alignment with other strided arrays);
- ``"minmax"``: the extrema of each bucket, in time order, plus the first and the last sample.

:func:`bucket_edges` gives the ``k`` equal buckets themselves, for reductions such as a per-bucket median::

    from pinneapple_core.series import downsample
    xs, ys = downsample(t, signal, 2000)              # min/max preserving
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["METHODS", "downsample", "downsample_indices", "bucket_edges"]

METHODS = ("minmax", "even", "stride")


def bucket_edges(n: int, k: int) -> np.ndarray:
    """``min(k, n) + 1`` integer edges splitting ``range(n)`` into equal consecutive buckets (``edges[i]:edges[i+1]``)."""
    n, k = int(n), int(k)
    return np.linspace(0, n, max(min(k, n), 0) + 1).astype(int)


def downsample_indices(n: int, k: int, method: str = "minmax", y=None) -> np.ndarray:
    """The indices (sorted, unique) of at most ``k`` samples of a series of length ``n``.

    ``method``: ``"minmax"`` (needs ``y``; non-finite values are ignored when looking for the extrema), ``"even"`` or
    ``"stride"`` (see the module docstring). All indices are returned when ``n <= k``.
    """
    n, k = int(n), int(k)
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}, not {method!r}")
    if k < 1:
        raise ValueError("k must be at least 1")
    if n <= 0:
        return np.zeros(0, int)
    if method == "stride":
        return np.arange(0, n, max(1, math.ceil(n / k)))
    if n <= k:
        return np.arange(n)
    if method == "even":
        return np.linspace(0, n - 1, k).astype(int)
    if y is None:
        raise ValueError("method 'minmax' needs the values y")
    v = np.asarray(y, dtype=float).ravel()
    if len(v) != n:
        raise ValueError(f"y has {len(v)} values, expected {n}")
    if k < 4:                                                    # no room for a bucket pair: fall back to even
        return np.linspace(0, n - 1, k).astype(int)
    edges = bucket_edges(n, (k - 2) // 2)
    keep = [0, n - 1]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        seg = v[a:b]
        fin = np.isfinite(seg)
        if not fin.any():
            keep.append(a)
            continue
        idx = np.flatnonzero(fin)
        s = seg[fin]
        keep += [a + int(idx[np.argmin(s)]), a + int(idx[np.argmax(s)])]
    return np.unique(np.asarray(keep, int))


def downsample(x, y, k: int = 2000, method: str = "minmax") -> tuple[np.ndarray, np.ndarray]:
    """``(x[idx], y[idx])`` with ``idx = downsample_indices(len(y), k, method, y)``. ``x`` may be ``None`` (then the
    sample indices are returned as x)."""
    v = np.asarray(y)
    idx = downsample_indices(len(v), k, method, v if method == "minmax" else None)
    xs = idx.copy() if x is None else np.asarray(x)[idx]
    return xs, v[idx]
