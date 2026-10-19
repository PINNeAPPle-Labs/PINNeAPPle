from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class Split:
    """A single backtesting fold expressed as indices of the windowed dataset."""
    fold: int
    train_idx: List[int]
    val_idx: List[int]
    test_idx: Optional[List[int]] = None


@dataclass
class ExpandingWindowSplitter:
    """Walk-forward validation with an expanding training window.

    Indices refer to items of the *windowed dataset* (not raw timestamps).
    """
    initial_train_size: int
    val_size: int
    step_size: int = 1
    gap: int = 0
    max_folds: Optional[int] = None

    def split(self, n_samples: int) -> Iterator[Split]:
        fold = 0
        train_end = int(self.initial_train_size)

        while True:
            val_start = train_end + int(self.gap)
            val_end = val_start + int(self.val_size)
            if val_end > n_samples:
                return

            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))
            yield Split(fold=fold, train_idx=train_idx, val_idx=val_idx)

            fold += 1
            if self.max_folds is not None and fold >= int(self.max_folds):
                return
            train_end += int(self.step_size)


@dataclass
class RollingWindowSplitter:
    """Walk-forward validation with a fixed-size rolling training window."""
    train_size: int
    val_size: int
    step_size: int = 1
    gap: int = 0
    max_folds: Optional[int] = None

    def split(self, n_samples: int) -> Iterator[Split]:
        fold = 0
        train_start = 0
        train_end = train_start + int(self.train_size)

        while True:
            val_start = train_end + int(self.gap)
            val_end = val_start + int(self.val_size)
            if val_end > n_samples:
                return

            train_idx = list(range(train_start, train_end))
            val_idx = list(range(val_start, val_end))
            yield Split(fold=fold, train_idx=train_idx, val_idx=val_idx)

            fold += 1
            if self.max_folds is not None and fold >= int(self.max_folds):
                return
            train_start += int(self.step_size)
            train_end += int(self.step_size)

# ---------------------------------------------------------------------------
# Purged walk-forward and k-fold splits over forecast origins
# ---------------------------------------------------------------------------
#
# A sample of a direct multi-horizon forecaster is a forecast origin t: its inputs use the observations up to t
# and its labels the observations t+1 .. t+H. Two samples overlap in information when their label windows
# overlap, so a training sample whose labels reach into the validation block leaks the validation targets into
# training ("purge" removes those), and in k-fold splits a training sample right after the validation block shares
# serial correlation with it ("embargo" removes those). López de Prado (2018), Advances in Financial Machine
# Learning, ch. 7.

import numpy as _np


def _purge_mask(times, val_lo, val_hi, purge, embargo):
    """True for samples allowed in training given a validation block of times [val_lo, val_hi]."""
    t = _np.asarray(times)
    before = t < val_lo - purge
    after = t > val_hi + embargo
    return before | after


@dataclass
class PurgedWalkForwardSplitter:
    """Walk-forward (rolling-origin) validation over forecast origins with purge and embargo.

    ``times`` passed to :meth:`split` are the sorted origin times of the samples. The last ``n_folds`` blocks of
    ``val_size`` samples are the validation blocks (oldest first). Training uses only samples whose origin is more
    than ``purge`` time steps before the block (set ``purge`` to the largest horizon so no training label falls in
    the block). ``window`` limits training to the most recent samples (a rolling instead of an expanding window).
    """
    n_folds: int = 3
    val_size: int = 500
    purge: int = 0
    embargo: int = 0
    window: Optional[int] = None
    min_train: int = 50

    def split(self, times) -> Iterator[Split]:
        t = _np.asarray(times)
        n = len(t)
        for fold in range(self.n_folds):
            lo = n - (self.n_folds - fold) * self.val_size
            hi = lo + self.val_size
            if lo <= 0:
                continue
            val = list(range(lo, hi))
            ok = _np.nonzero(t < t[lo] - self.purge)[0]
            if self.window is not None:
                ok = ok[-int(self.window):]
            if len(ok) < self.min_train:
                continue
            yield Split(fold=fold, train_idx=ok.tolist(), val_idx=val)


@dataclass
class PurgedKFoldSplitter:
    """K contiguous folds; training on both sides of the validation fold with purge before and embargo after."""
    n_folds: int = 5
    purge: int = 0
    embargo: int = 0

    def split(self, times) -> Iterator[Split]:
        t = _np.asarray(times)
        n = len(t)
        edges = _np.linspace(0, n, self.n_folds + 1).astype(int)
        for fold in range(self.n_folds):
            lo, hi = edges[fold], edges[fold + 1]
            mask = _purge_mask(t, t[lo], t[hi - 1], self.purge, self.embargo)
            yield Split(fold=fold, train_idx=_np.nonzero(mask)[0].tolist(), val_idx=list(range(lo, hi)))


@dataclass
class ShuffledKFoldSplitter:
    """Ordinary shuffled k-fold, which ignores time: kept to measure how optimistic it is on serial data."""
    n_folds: int = 5
    seed: int = 0

    def split(self, times) -> Iterator[Split]:
        n = len(times)
        perm = _np.random.default_rng(self.seed).permutation(n)
        for fold, val in enumerate(_np.array_split(perm, self.n_folds)):
            mask = _np.ones(n, bool)
            mask[val] = False
            yield Split(fold=fold, train_idx=_np.nonzero(mask)[0].tolist(), val_idx=sorted(val.tolist()))
