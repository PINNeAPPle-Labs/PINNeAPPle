"""Train / validation / held-out test / distribution-shift splits for surrogate studies.

The distribution-shift split is carved out first and never seen in training, so its error is a measured
extrapolation error, not an estimate:

``"group"``     whole groups held out (a geometry, a case, a machine never seen in training);
``"edge"``      the samples farthest from the centre of the input space (normalised distance), i.e. the boundary of
                the design space, while training covers the interior;
``{"column": name, "side": "high" | "low"}``  the top or bottom of one input (extrapolation along that parameter);
``"auto"``      ``data.meta["shift"]`` when the dataset recommends one, else ``"group"`` when the data has groups,
                else ``"edge"``; ``None`` for no shift split.

The rest is split at random into train, validation (model selection and early stopping only) and the held-out test.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import SurrogateData

__all__ = ["Splits", "make_splits"]


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    shift: np.ndarray
    shift_kind: str | None = None
    shift_description: str = ""
    seed: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"n_train": int(len(self.train)), "n_val": int(len(self.val)), "n_test": int(len(self.test)),
                "n_shift": int(len(self.shift)), "shift_kind": self.shift_kind,
                "shift_description": self.shift_description, "seed": self.seed, **self.info}

    def few_data(self, fraction: float, seed: int = 0) -> Splits:
        """The same test and shift sets, a random ``fraction`` of the training samples (at least 4)."""
        rng = np.random.default_rng(seed + 7919)
        k = max(4, int(round(fraction * len(self.train))))
        tr = np.sort(rng.choice(self.train, size=min(k, len(self.train)), replace=False))
        return Splits(tr, self.val, self.test, self.shift, self.shift_kind, self.shift_description, self.seed,
                      {**self.info, "few_data_fraction": fraction})


def _shift(data: SurrogateData, shift, frac: float, rng) -> tuple[np.ndarray, str | None, str]:
    n = data.n
    if shift is None or frac <= 0:
        return np.zeros(0, int), None, ""
    if shift == "auto" and data.meta.get("shift"):
        shift = data.meta["shift"]                          # the dataset's own recommendation
    if shift == "auto":
        shift = "group" if data.groups is not None and len(np.unique(data.groups)) >= 5 else "edge"
    if shift == "group":
        if data.groups is None:
            raise ValueError("shift='group' needs data.groups")
        g = np.unique(data.groups)
        k = max(1, int(round(frac * len(g))))
        held = rng.choice(g, size=k, replace=False)
        idx = np.flatnonzero(np.isin(data.groups, held))
        return idx, "group", f"{k} of {len(g)} groups never seen in training ({len(idx)} samples)"
    k = max(1, int(round(frac * n)))
    if shift == "edge":
        Z = (data.X - data.X.mean(0)) / (data.X.std(0) + 1e-12)
        d = np.sqrt((Z ** 2).mean(1))
        idx = np.argsort(d)[-k:]
        return np.sort(idx), "edge", (f"the {k} samples farthest from the centre of the input space "
                                      f"(normalised distance > {d[idx].min():.2f}); training covers the interior")
    if isinstance(shift, Mapping) and "column" in shift:
        col = shift["column"]
        if col not in data.input_names:
            raise ValueError(f"shift column {col!r} is not an input")
        v = data.X[:, data.input_names.index(col)]
        order = np.argsort(v, kind="stable")
        side = shift.get("side", "high")
        idx = order[-k:] if side == "high" else order[:k]
        thr = v[idx].min() if side == "high" else v[idx].max()
        rel = ">=" if side == "high" else "<="
        return np.sort(idx), f"column:{col}:{side}", f"{col} {rel} {thr:.4g} ({k} samples): extrapolation in {col}"
    raise ValueError(f"unknown shift split {shift!r}")


def make_splits(data: SurrogateData, *, test_frac: float = 0.2, val_frac: float = 0.15, shift: Any = "auto",
                shift_frac: float = 0.15, seed: int = 0) -> Splits:
    """Disjoint index sets; the shift set is removed before the random train/val/test split."""
    if not 0 < test_frac < 1 or not 0 <= val_frac < 1 or test_frac + val_frac >= 1:
        raise ValueError("need 0 < test_frac, 0 <= val_frac and test_frac + val_frac < 1")
    rng = np.random.default_rng(seed)
    sh, kind, desc = _shift(data, shift, shift_frac, rng)
    rest = np.setdiff1d(np.arange(data.n), sh)
    if len(rest) < 10:
        raise ValueError(f"only {len(rest)} samples left after the shift split; need at least 10")
    rest = rng.permutation(rest)
    n_te = max(2, int(round(test_frac * len(rest))))
    n_va = max(1, int(round(val_frac * len(rest)))) if val_frac > 0 else 0
    te, va, tr = rest[:n_te], rest[n_te:n_te + n_va], rest[n_te + n_va:]
    return Splits(np.sort(tr), np.sort(va), np.sort(te), sh, kind, desc, seed)
